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
	inline __m128d vectorNegate( __m128d vec )
	{
		return _mm_sub_pd( _mm_setzero_pd(), vec );
	}

	// Selectively negate some lanes in the vector
	template<int mask>
	inline __m256d vectorNegateLanes( __m256d vec )
	{
		static_assert( mask >= 0 && mask <= 0b1111, "The lanes mask should be in [ 0b0000 - 0b1111 ] range" );
		if constexpr( 0 == mask )
			return vec;
		__m256d neg = vectorNegate( vec );
		if constexpr( 0b1111 == mask )
			return neg;
		return _mm256_blend_pd( vec, neg, mask );
	}

	// abs( vec )
	inline __m256d vectorAbs( __m256d vec )
	{
		return _mm256_max_pd( vec, vectorNegate( vec ) );
	}
	inline __m128d vectorAbs( __m128d vec )
	{
		return _mm_max_pd( vec, vectorNegate( vec ) );
	}

	inline __m256d vectorMultiplyAdd( __m256d a, __m256d b, __m256d c )
	{
#if _AM_FMA3_INTRINSICS_
		return _mm256_fmadd_pd( a, b, c );
#else
		return _mm256_add_pd( _mm256_mul_pd( a, b ), c );
#endif
	}
	inline __m128d vectorMultiplyAdd( __m128d a, __m128d b, __m128d c )
	{
#if _AM_FMA3_INTRINSICS_
		return _mm_fmadd_pd( a, b, c );
#else
		return _mm_add_pd( _mm_mul_pd( a, b ), c );
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

	// Load a scalar, broadcast to 4 lanes; the broadcasting is free.
	inline __m256d broadcast( const double& v )
	{
		return _mm256_broadcast_sd( &v );
	}

	// Load a scalar, broadcast to 2 lanes; the broadcasting is free.
	inline __m128d broadcast2( const double& v )
	{
		return _mm_load1_pd( &v );
	}

	inline double vectorGetX( __m128d vec )
	{
		return _mm_cvtsd_f64( vec );
	}
	inline double vectorGetY( __m128d vec )
	{
		return _mm_cvtsd_f64( _mm_permute_pd( vec, 0b11 ) );
	}

	inline double vectorGetX( __m256d vec )
	{
		return _mm256_cvtsd_f64( vec );
	}
	inline double vectorGetY( __m256d vec )
	{
		return _mm_cvtsd_f64( _mm_permute_pd( low2( vec ), 0b11 ) );
	}
	inline double vectorGetZ( __m256d vec )
	{
		return _mm_cvtsd_f64( high2( vec ) );
	}
	inline double vectorGetW( __m256d vec )
	{
		__m128d high = high2( vec );
		high = _mm_permute_pd( high, 0b11 );
		return _mm_cvtsd_f64( high );
	}

	inline __m256d vectorSplatX( __m256d vec )
	{
		__m128d low = low2( vec );
#if _AM_AVX2_INTRINSICS_
		return _mm256_broadcastsd_pd( low );
#else
		low = _mm_permute_pd( low, 0b00 );
		return _mm256_setr_m128d( low, low );
#endif
	}

	inline __m256d vectorSplatY( __m256d vec )
	{
		__m128d low = low2( vec );
		low = _mm_permute_pd( low, 0b11 );
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
		high = _mm_permute_pd( high, 0b00 );
		return _mm256_setr_m128d( high, high );
#endif
	}

	inline __m256d vectorSplatW( __m256d vec )
	{
#if _AM_AVX2_INTRINSICS_
		return _mm256_permute4x64_pd( vec, _MM_SHUFFLE( 3, 3, 3, 3 ) );
#else
		__m128d high = high2( vec );
		high = _mm_permute_pd( high, 0b11 );
		return _mm256_setr_m128d( high, high );
#endif
	}

	extern const struct sMiscConstants
	{
		const double negativeZero = -0.0f;
		const double one = 1;
		const double negativeOne = -1;
		const double oneHalf = 0.5;
		const double infinity = std::numeric_limits<double>::infinity();
		const double quietNaN = std::numeric_limits<double>::quiet_NaN();
	}
	g_misc;

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

	// The sine/cosine functions are using 11-degree minimax approximation for sine, 10-degree for cosine.
	// Absolute errors for sine/cosine, compared to standard library of VC++, are within 1.8E-08, i.e. these are reasonably accurate approximations, despite way faster.

	// Compute both sine and cosine of 4 angles in radian
	void _AM_CALL_ vectorSinCos( __m256d& sin, __m256d& cos, __m256d angles );

	// Compute both sine and cosine of the angle, return a vector with [ cos, sin ] values
	__m128d scalarSinCos( double a );

	// Sine of the angle
	double scalarSin( double a );
	// Cosine of the angle
	double scalarCos( double a );

	// Tangent and cotangent are using Padé approximation of the degrees 7/6 for numerator/denominator
	__m256d _AM_CALL_ vectorTan( __m256d a );
	__m256d _AM_CALL_ vectorCot( __m256d a );
	double scalarTan( double a );
	double scalarCot( double a );

	// A low-precision approximation of hyperbolic tangent
	__m256d _AM_CALL_ vectorTanH( __m256d vec );

	// Round number to nearest integer without function calls
	inline double round( double a )
	{
		__m128d v = _mm_set_sd( a );
		v = _mm_round_sd( v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
		return _mm_cvtsd_f64( v );
	}

	// Round number to nearest integer, return int64 value
	inline int64_t lround( double a )
	{
		__m128d v = _mm_set_sd( a );
		v = _mm_round_sd( v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
		return _mm_cvtsd_si64( v );
	}

	// Round number to nearest integer, return int32 value
	inline int iround( double a )
	{
		__m128d v = _mm_set_sd( a );
		v = _mm_round_sd( v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
		return _mm_cvtsd_si32( v );
	}

	constexpr double g_pi = 3.141592653589793238;
}