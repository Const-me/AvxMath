#include "AvxMath.h"

namespace AvxMath
{
	__m256d _AM_CALL_ quaternionMultiply( __m256d a, __m256d b )
	{
		const __m256d af = flipHighLow( a );	// a.ZWXY

		// b.X * a.WZYX
		__m256d bb = vectorSplatX( b );
		__m256d tmp = _mm256_permute_pd( af, 0b0101 );	// a.WZYX
		__m256d res = _mm256_mul_pd( bb, tmp );

		// b.W * a.XYZW [ +-+- ] b.X * a.WZYX
		bb = vectorSplatW( b );
#if _AM_FMA3_INTRINSICS_
		res = _mm256_fmsubadd_pd( bb, a, res );
#else
		bb = _mm256_mul_pd( bb, a );
		tmp = _mm256_addsub_pd( _mm256_setzero_pd(), res );
		res = _mm256_sub_pd( bb, tmp );
#endif

		// [ ++-- ] b.Y * a.ZWXY
		bb = vectorSplatY( b );
		bb = vectorNegateLanes<0b1100>( bb );
		res = vectorMultiplyAdd( bb, af, res );

		// [ -++- ] b.Z * a.YXWZ
		bb = vectorSplatZ( b );
		bb = vectorNegateLanes<0b1001>( bb );
		tmp = _mm256_permute_pd( a, 0b0101 );
		res = vectorMultiplyAdd( bb, tmp, res );

		return res;
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
		__m256d Q1 = vectorNegateLanes<0b0110>( P1 );

		__m256d Q0 = _mm256_mul_pd( P0, Y0 );
		Q1 = _mm256_mul_pd( Q1, Y1 );
		Q0 = _mm256_mul_pd( Q0, R0 );
		return vectorMultiplyAdd( Q1, R1, Q0 );
	}
}