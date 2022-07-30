// Miscellaneous routines to compare vectors for total order
#pragma once

namespace AvxMath
{
	// Compare 4D vectors for exact equality
	inline bool vectorEqual( __m256d a, __m256d b )
	{
		// https://stackoverflow.com/a/64191351/126995
		__m256d ne = _mm256_cmp_pd( a, b, _CMP_NEQ_UQ );
		return (bool)_mm256_testz_pd( ne, ne );
	}

	// Compare 3D vectors for exact equality
	inline bool vector3Equal( __m256d a, __m256d b )
	{
		__m256d zero = _mm256_setzero_pd();
		__m256d ne = _mm256_cmp_pd( a, b, _CMP_NEQ_UQ );
		ne = _mm256_blend_pd( ne, zero, 0b1000 );
		return (bool)_mm256_testz_pd( ne, ne );
	}

	// Compare 2D vectors for exact equality
	inline bool vectorEqual( __m128d a, __m128d b )
	{
		__m128d ne = _mm_cmp_pd( a, b, _CMP_NEQ_UQ );
		return (bool)_mm_testz_pd( ne, ne );
	}

	// Compare 4D vectors for a < b; W is the most significant lane
	inline bool vectorLess( __m256d a, __m256d b )
	{
		__m256d lt = _mm256_cmp_pd( a, b, _CMP_LT_OQ );
		__m256d gt = _mm256_cmp_pd( a, b, _CMP_GT_OQ );
		uint32_t m1 = (uint32_t)_mm256_movemask_pd( lt );
		uint32_t m2 = (uint32_t)_mm256_movemask_pd( gt );
		return m1 > m2;
	}

	// Compare 3D vectors for a < b; Z is the most significant lane
	inline bool vector3Less( __m256d a, __m256d b )
	{
		__m256d zero = _mm256_setzero_pd();
		a = _mm256_blend_pd( a, zero, 0b1000 );
		b = _mm256_blend_pd( b, zero, 0b1000 );
		return vectorLess( a, b );
	}

	// Compare 2D vectors for a < b; Y is the most significant lane
	inline bool vectorLess( __m128d a, __m128d b )
	{
		__m128d lt = _mm_cmp_pd( a, b, _CMP_LT_OQ );
		__m128d gt = _mm_cmp_pd( a, b, _CMP_GT_OQ );
		uint32_t m1 = (uint32_t)_mm_movemask_pd( lt );
		uint32_t m2 = (uint32_t)_mm_movemask_pd( gt );
		return m1 > m2;
	}

	// Compare 4D vectors for bitwise equality, i.e. -0.0 != 0.0
	inline bool vectorBitwiseEqual( __m256d a, __m256d b )
	{
		__m256i xx = _mm256_castpd_si256( _mm256_xor_pd( a, b ) );
		// Surprisingly, that instruction is from AVX1 set
		return (bool)_mm256_testz_si256( xx, xx );
	}

	// Compare 3D vectors for bitwise equality
	inline bool vector3BitwiseEqual( __m256d a, __m256d b )
	{
		__m256d zero = _mm256_setzero_pd();
		a = _mm256_blend_pd( a, zero, 0b1000 );
		b = _mm256_blend_pd( b, zero, 0b1000 );
		return vectorBitwiseEqual( a, b );
	}

	// Compare 2D vectors for bitwise equality
	inline bool vectorBitwiseEqual( __m128d a, __m128d b )
	{
		__m128i xx = _mm_castpd_si128( _mm_xor_pd( a, b ) );
		return (bool)_mm_testz_si128( xx, xx );
	}

	// TODO: some hash function for these vectors, to use as keys in unordered containers in combination with bitwise equality implemented above
	// Maybe rework Meow optimizing for latency with small things https://github.com/cmuratori/meow_hash
	// Or maybe that helps somehow https://github.com/dragontamer/AESRand/blob/master/AESRand/AESRand/AESRand.cpp
	// One thing to consider, std::unordered_map needs uint64_t hashes while CAtlMap wants uint32_t, ideally we'd need both versions.
}