#include "AvxMath.h"
#include <array>

namespace AvxMath
{
	static inline __m128i getLowInt( __m256d vec )
	{
		return _mm256_castsi256_si128( _mm256_castpd_si256( vec ) );
	}

	static inline __m128i getHighInt( __m256d vec )
	{
		__m256i iv = _mm256_castpd_si256( vec );
#if _AM_AVX2_INTRINSICS_
		return _mm256_extracti128_si256( iv, 1 );
#else
		return _mm256_extractf128_si256( iv, 1 );
#endif
	}

	// The hashing algorithms are from there: https://github.com/Cyan4973/xxHash
	// Reworked to vectorize, and extract source data directly from the vectors instead of RAM

	// ==== 64 bit hashes ====

	// Multiply unsigned 64-bit integers "a" and "b", return 128 bits
	static inline uint64_t mulx( uint64_t a, uint64_t b, uint64_t& high )
	{
#if _AM_AVX2_INTRINSICS_
		// That instruction is from BMI2 set.
		// According to Wikipedia https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set#Supporting_CPUs was implemented by Intel and AMD at the same time as AVX2
		return _mulx_u64( a, b, &high );
#else
#ifdef _MSC_VER
		return _umul128( a, b, &high );
#else
		__uint128_t const product = (__uint128_t)a * (__uint128_t)b;
		high = (uint64_t)( product >> 64 );
		return (uint64_t)product;
#endif
#endif
	}

	static inline uint64_t hashStep( uint64_t a, uint64_t b )
	{
		uint64_t low, high;
		low = mulx( a, b, high );
		return low ^ high;
	}

	// Copy-pasted from there: https://github.com/Cyan4973/xxHash/blob/v0.8.1/xxhash.h#L3166-L3186
	alignas( 32 ) static const std::array<uint8_t, 32> g_secret = {
	0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
	0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
	};

	constexpr uint64_t prime64 = 0x9E3779B185EBCA87ull;

	static inline uint64_t xorShift( uint64_t x, int shift )
	{
		return x ^ ( x >> shift );
	}

	static inline uint64_t avalanche( uint64_t hash )
	{
		hash = xorShift( hash, 37 );
		hash *= 0x165667919E3779F9ull;
		hash = xorShift( hash, 32 );
		return hash;
	}

	template<bool fourLanes>
	static inline uint64_t vectorHash64Impl( __m256d vec )
	{
		uint64_t acc = prime64 * ( fourLanes ? 32 : 24 );
		const uint64_t* const secret = (const uint64_t*)g_secret.data();

		__m128i tmp = getLowInt( vec );
		acc += hashStep( (uint64_t)_mm_cvtsi128_si64( tmp ), secret[ 0 ] );
		acc += hashStep( (uint64_t)_mm_extract_epi64( tmp, 1 ), secret[ 1 ] );

		tmp = getHighInt( vec );
		acc += hashStep( (uint64_t)_mm_cvtsi128_si64( tmp ), secret[ 2 ] );
		if constexpr( fourLanes )
			acc += hashStep( (uint64_t)_mm_extract_epi64( tmp, 1 ), secret[ 3 ] );

		return avalanche( acc );
	}

	uint64_t vectorHash64( __m128d vec )
	{
		uint64_t acc = prime64 * 16;
		const uint64_t* const secret = (const uint64_t*)g_secret.data();
		__m128i tmp = _mm_castpd_si128( vec );

		acc += hashStep( (uint64_t)_mm_cvtsi128_si64( tmp ), secret[ 0 ] );
		acc += hashStep( (uint64_t)_mm_extract_epi64( tmp, 1 ), secret[ 1 ] );
		return avalanche( acc );
	}

	uint64_t vectorHash64( __m256d vec )
	{
		return vectorHash64Impl<true>( vec );
	}

	uint64_t vector3Hash64( __m256d vec )
	{
		return vectorHash64Impl<false>( vec );
	}

	// ==== 32 bit hashes ====

	constexpr uint32_t prime32_1 = 0x9E3779B1u;
	constexpr uint32_t prime32_2 = 0x85EBCA77U;
	constexpr uint32_t prime32_3 = 0xC2B2AE3DU;

	// We want these magic vectors placed at adjacent memory addresses, that's why the structure
	alignas( 64 ) static const struct
	{
		const __m128i initialState = _mm_setr_epi32( prime32_1 + prime32_2, prime32_2, 0, -(int)prime32_1 );
		const __m128i prime2 = _mm_set1_epi32( prime32_2 );
		const __m128i prime1 = _mm_set1_epi32( prime32_1 );
#if _AM_AVX2_INTRINSICS_
		const __m128i leftShift = _mm_setr_epi32( 1, 7, 12, 18 );
		const __m128i rightShift = _mm_setr_epi32( 32 - 1, 32 - 7, 32 - 12, 32 - 18 );
#endif
	}
	g_32bit;

	static inline __m128i updateState( __m128i acc, __m128i data )
	{
		acc = _mm_add_epi32( acc, _mm_mullo_epi32( data, g_32bit.prime2 ) );

		__m128i a = _mm_slli_epi32( acc, 13 );
		__m128i b = _mm_srli_epi32( acc, 32 - 13 );
		acc = _mm_or_si128( a, b );

		acc = _mm_mullo_epi32( acc, g_32bit.prime1 );
		return acc;
	}

	static inline uint32_t avalanche( uint32_t hash )
	{
		hash ^= hash >> 15;
		hash *= prime32_2;
		hash ^= hash >> 13;
		hash *= prime32_3;
		hash ^= hash >> 16;
		return hash;
	}

	static inline uint32_t finalize( __m128i acc, uint32_t len )
	{
		uint32_t hash;

#if _AM_AVX2_INTRINSICS_
		__m128i a = _mm_sllv_epi32( acc, g_32bit.leftShift );
		__m128i b = _mm_srlv_epi32( acc, g_32bit.rightShift );
		acc = _mm_or_si128( a, b );

		acc = _mm_add_epi32( acc, _mm_srli_si128( acc, 8 ) );
		hash = (uint32_t)_mm_extract_epi32( acc, 1 ) + (uint32_t)_mm_cvtsi128_si32( acc );
#else
		hash = _rotl( (uint32_t)_mm_extract_epi32( acc, 3 ), 18 );
		hash += _rotl( (uint32_t)_mm_extract_epi32( acc, 2 ), 12 );
		hash += _rotl( (uint32_t)_mm_extract_epi32( acc, 1 ), 7 );
		hash += _rotl( (uint32_t)_mm_cvtsi128_si32( acc ), 1 );
#endif

		hash += len;
		return avalanche( hash );
	}

	template<bool fourLanes>
	static inline uint32_t vectorHash32Impl( __m256d vec )
	{
		__m128i acc = g_32bit.initialState;
		acc = updateState( acc, getLowInt( vec ) );

		if constexpr( fourLanes )
		{
			acc = updateState( acc, getHighInt( vec ) );
			return finalize( acc, 32 );
		}
		else
		{
			__m128i a2 = updateState( acc, getHighInt( vec ) );
			acc = _mm_blend_epi16( a2, acc, 0b11110000 );
			return finalize( acc, 24 );
		}
	}

	uint32_t vectorHash32( __m128d vec )
	{
		__m128i acc = g_32bit.initialState;
		acc = updateState( acc, _mm_castpd_si128( vec ) );
		return finalize( acc, 16 );
	}

	uint32_t vectorHash32( __m256d vec )
	{
		return vectorHash32Impl<true>( vec );
	}

	uint32_t vector3Hash32( __m256d vec )
	{
		return vectorHash32Impl<false>( vec );
	}
}