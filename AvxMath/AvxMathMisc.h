#pragma once

namespace AvxMath
{
	// Extract [ X, Y ] slice of the vector; the function is free in runtime, compiles into no instructions
	inline __m128d low2( __m256d vec )
	{
		return _mm256_castpd256_pd128( vec );
	}

	// Extract [ Z, W ] slice of the vector
	inline __m128d high2( __m256d vec )
	{
		return _mm256_extractf128_pd( vec, 1 );
	}

	inline __m256d dup2( __m128d vec )
	{
		return _mm256_insertf128_pd( _mm256_castpd128_pd256( vec ), vec, 1 );
	}

	// Permute XYZW => ZWXY
	inline __m256d flipHighLow( __m256d vec )
	{
		return _mm256_permute2f128_pd( vec, vec, 1 );
	}

	// -vec
	inline __m256d vectorNegate( __m256d vec )
	{
		return _mm256_sub_pd( _mm256_setzero_pd(), vec );
	}

	// abs( vec )
	inline __m256d vectorAbs( __m256d vec )
	{
		return _mm256_max_pd( vec, vectorNegate( vec ) );
	}

	inline __m256d vectorMultiplyAdd( __m256d a, __m256d b, __m256d c )
	{
#if _AM_FMA3_INTRINSICS_
		return _mm256_fmadd_pd( a, b, c );
#else
		return _mm256_add_pd( _mm256_mul_pd( a, b ), c );
#endif
	}

	// Tests whether the components of a 4D vector are within set bounds, i.e. -bounds <= vec <= bounds
	inline bool vector4InBounds( __m256d vec, __m256d bounds )
	{
		// Compute absolute value
		vec = vectorAbs( vec );
		// Compare for vec > bounds but we want NAN to result in TRUE that's why the weird predicate,
		// https://stackoverflow.com/a/64191351/126995
		__m256d cmp = _mm256_cmp_pd( vec, bounds, _CMP_NLE_UQ );
		// Return true (meaning in bounds) when the cmp is [ false, false, false, false ]
		return (bool)_mm256_testz_pd( cmp, cmp );
	}

	inline double vectorGetX( __m256d vec )
	{
		return _mm256_cvtsd_f64( vec );
	}
	inline double vectorGetY( __m256d vec )
	{
		return _mm_cvtsd_f64( _mm_permute_pd( low2( vec ), _MM_SHUFFLE2( 1, 1 ) ) );
	}
	inline double vectorGetZ( __m256d vec )
	{
		return _mm_cvtsd_f64( high2( vec ) );
	}
	inline double vectorGetW( __m256d vec )
	{
		__m128d high = high2( vec );
		high = _mm_permute_pd( high, _MM_SHUFFLE2( 1, 1 ) );
		return _mm_cvtsd_f64( high );
	}

	inline __m256d vectorSplatX( __m256d vec )
	{
		__m128d low = low2( vec );
#if _AM_AVX2_INTRINSICS_
		return _mm256_broadcastsd_pd( low );
#else
		low = _mm_permute_pd( low, _MM_SHUFFLE2( 0, 0 ) );
		return _mm256_setr_m128d( low, low );
#endif
	}

	inline __m256d vectorSplatY( __m256d vec )
	{
		__m128d low = low2( vec );
		low = _mm_permute_pd( low, _MM_SHUFFLE2( 1, 1 ) );
#if _AM_AVX2_INTRINSICS_
		return _mm256_broadcastsd_pd( low );
#else
		return _mm256_setr_m128d( low, low );
#endif
	}

	inline __m256d vectorSplatZ( __m256d vec )
	{
#if _AM_AVX2_INTRINSICS_
		return _mm256_permute4x64_pd( vec, _MM_SHUFFLE( 2, 2, 2, 2 ) );
#else
		__m128d high = high2( vec );
		high = _mm_permute_pd( high, _MM_SHUFFLE2( 0, 0 ) );
		return _mm256_setr_m128d( high, high );
#endif
	}

	inline __m256d vectorSplatW( __m256d vec )
	{
#if _AM_AVX2_INTRINSICS_
		return _mm256_permute4x64_pd( vec, _MM_SHUFFLE( 3, 3, 3, 3 ) );
#else
		__m128d high = high2( vec );
		high = _mm_permute_pd( high, _MM_SHUFFLE2( 1, 1 ) );
		return _mm256_setr_m128d( high, high );
#endif
	}

	__declspec( selectany ) extern double infinity = std::numeric_limits<double>::infinity();
	__declspec( selectany ) extern double quietNaN = std::numeric_limits<double>::quiet_NaN();
	__declspec( selectany ) extern double one = 1;
	__declspec( selectany ) extern __m256d g_flipXyz = _mm256_setr_pd( -0.0, -0.0, -0.0, 0 );

#ifndef _MSC_VER
	inline __m128i _mm_loadu_si32( const void* p )
	{
		return _mm_cvtsi32_si128( *(const int*)( p ) );
	}
#endif

	namespace details
	{
		inline void splatVector( Matrix4x4& mat, __m256d vec )
		{
			mat.r0 = vectorSplatX( vec );
			mat.r1 = vectorSplatY( vec );
			mat.r2 = vectorSplatZ( vec );
			mat.r3 = vectorSplatW( vec );
		}
	}

	void vectorSinCos( __m256d& sin, __m256d& cos, __m256d angles );
}