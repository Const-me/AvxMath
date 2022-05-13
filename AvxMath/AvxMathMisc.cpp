#include "AvxMath.h"
#include <array>

namespace AvxMath
{
	const struct sMiscConstants g_misc;

	constexpr double g_pi = 3.141592653589793238;

	alignas( 32 ) static const struct
	{
		const double pi = g_pi;
		const double halfPi = g_pi / 2;
		const double twoPi = g_pi * 2.0;
		const double inv2pi = 1.0 / ( 2.0 * g_pi );
	}
	g_piConstants;

	static inline __m256d broadcast( const double& v )
	{
		return _mm256_broadcast_sd( &v );
	}

	inline __m256d vectorModAngles( __m256d a )
	{
		const __m256d inv2pi = broadcast( g_piConstants.inv2pi );
		__m256d v = _mm256_mul_pd( a, inv2pi );
		v = _mm256_round_pd( v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );

		const __m256d twoPi = broadcast( g_piConstants.twoPi );
		v = _mm256_mul_pd( v, twoPi );

		return _mm256_sub_pd( a, v );
	}

	static const std::array<double, 5> g_sinCoefficients
	{
		// g_XMSinCoefficients0
		-0.16666667, +0.0083333310, -0.00019840874, +2.7525562e-06,
		// g_XMSinCoefficients1
		-2.3889859e-08
	};
	static const std::array<double, 5> g_cosCoefficients
	{
		// g_XMCosCoefficients0
		-0.5, +0.041666638, -0.0013888378, +2.4760495e-05,
		// g_XMCosCoefficients1
		-2.6051615e-07
	};

	void _AM_CALL_ vectorSinCos( __m256d& sin, __m256d& cos, __m256d x )
	{
		// Force the value within the bounds of pi
		x = vectorModAngles( x );

		const __m256d one = broadcast( g_misc.one );
		const __m256d neg0 = broadcast( g_misc.negativeZero );

		// Map in [-pi/2,pi/2] with sin(y) = sin(x), cos(y) = sign*cos(x).
		__m256d sign = _mm256_and_pd( x, neg0 );
		__m256d c = _mm256_or_pd( broadcast( g_piConstants.pi ), sign );  // pi when x >= 0, -pi when x < 0
		__m256d absx = _mm256_andnot_pd( sign, x );  // |x|
		__m256d rflx = _mm256_sub_pd( c, x );
		__m256d comp = _mm256_cmp_pd( absx, broadcast( g_piConstants.halfPi ), _CMP_LE_OQ );
		x = _mm256_blendv_pd( rflx, x, comp );
		sign = _mm256_andnot_pd( comp, neg0 );

		__m256d x2 = _mm256_mul_pd( x, x );

		// Compute polynomial approximation of sine
		__m256d vConstants = broadcast( g_sinCoefficients[ 4 ] );
		__m256d Result = _mm256_mul_pd( vConstants, x2 );

		vConstants = broadcast( g_sinCoefficients[ 3 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( g_sinCoefficients[ 2 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( g_sinCoefficients[ 1 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( g_sinCoefficients[ 0 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );
		Result = _mm256_add_pd( Result, one );
		Result = _mm256_mul_pd( Result, x );
		sin = Result;

		// Compute polynomial approximation of cosine
		vConstants = broadcast( g_cosCoefficients[ 4 ] );
		Result = _mm256_mul_pd( vConstants, x2 );

		vConstants = broadcast( g_cosCoefficients[ 3 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( g_cosCoefficients[ 2 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( g_cosCoefficients[ 1 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( g_cosCoefficients[ 0 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );
		Result = _mm256_add_pd( Result, one );
		Result = _mm256_or_pd( Result, sign );
		cos = Result;
	}

	__m256d _AM_CALL_ quaternionRollPitchYaw( __m256d angles )
	{
		__m256d HalfAngles = _mm256_mul_pd( angles, broadcast( g_misc.oneHalf ) );

		__m256d SinAngles, CosAngles;
		vectorSinCos( SinAngles, CosAngles, HalfAngles );

		// P0 = XMVectorPermute<XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1X, XM_PERMUTE_1X>(SinAngles, CosAngles);
		__m256d P0 = _mm256_blend_pd( vectorSplatX( CosAngles ), SinAngles, 0b0001 );

		// Y0 = XMVectorPermute<XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_1Y, XM_PERMUTE_1Y>( SinAngles, CosAngles );
		__m256d Y0 = _mm256_blend_pd( vectorSplatY( CosAngles ), SinAngles, 0b0010 );

		// R0 = XMVectorPermute<XM_PERMUTE_1Z, XM_PERMUTE_1Z, XM_PERMUTE_0Z, XM_PERMUTE_1Z>( SinAngles, CosAngles );
		__m256d R0 = _mm256_blend_pd( vectorSplatZ( CosAngles ), SinAngles, 0b0100 );

		// P1 = XMVectorPermute<XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1X, XM_PERMUTE_1X>( CosAngles, SinAngles );
		__m256d P1 = _mm256_blend_pd( vectorSplatX( SinAngles ), CosAngles, 0b0001 );

		// Y1 = XMVectorPermute<XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_1Y, XM_PERMUTE_1Y>( CosAngles, SinAngles );
		__m256d Y1 = _mm256_blend_pd( vectorSplatY( SinAngles ), CosAngles, 0b0010 );

		// R1 = XMVectorPermute<XM_PERMUTE_1Z, XM_PERMUTE_1Z, XM_PERMUTE_0Z, XM_PERMUTE_1Z>( CosAngles, SinAngles );
		__m256d R1 = _mm256_blend_pd( vectorSplatZ( SinAngles ), CosAngles, 0b0100 );

		// static const XMVECTORF32  Sign = { { { 1.0f, -1.0f, -1.0f, 1.0f } } };
		// XMVECTOR Q1 = _mm256_mul_pd( P1, Sign.v );
		__m256d p1Neg = vectorNegate( P1 );
		__m256d Q1 = _mm256_blend_pd( P1, p1Neg, 0b0110 );

		__m256d Q0 = _mm256_mul_pd( P0, Y0 );
		Q1 = _mm256_mul_pd( Q1, Y1 );
		Q0 = _mm256_mul_pd( Q0, R0 );
		__m256d Q = vectorMultiplyAdd( Q1, R1, Q0 );
		return Q;
	}
}