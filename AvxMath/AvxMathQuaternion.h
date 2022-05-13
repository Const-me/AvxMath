#pragma once

namespace AvxMath
{
	inline __m256d quaternionNormalize( __m256d q )
	{
		return vector4Normalize( q );
	}

	inline __m256d quaternionConjugate( __m256d q )
	{
		__m256d neg = vectorNegate( q );
		return _mm256_blend_pd( neg, q, 0b1000 );
	}

	// Product of two quaternions
	inline __m256d quaternionMultiply( __m256d a, __m256d b )
	{
		// b.W * a.XYZW
		__m256d res = _mm256_mul_pd( vectorSplatW( b ), a );

		// [ +-+- ] b.X * a.WZYX
		__m256d bb = vectorSplatX( b );
		const __m256d af = flipHighLow( a );	// a.ZWXY
		__m256d tmp = _mm256_permute_pd( af, _MM_SHUFFLE2( 0, 1 ) );	// a.WZYX
#if _AM_FMA3_INTRINSICS_
		res = _mm256_fmsubadd_pd( bb, tmp, res );
#else
		tmp = _mm256_mul_pd( bb, tmp );
		tmp = _mm256_addsub_pd( _mm256_setzero_pd(), tmp );
		res = _mm256_sub_pd( res, tmp );
#endif

		const __m256d invertHigh = _mm256_setr_pd( 0, 0, -0.0, -0.0 );

		// [ ++-- ] b.Y * a.ZWXY
		bb = vectorSplatY( b );
		tmp = _mm256_xor_pd( af, invertHigh );
#if _AM_FMA3_INTRINSICS_
		res = _mm256_fmadd_pd( bb, af, res );
#else
		tmp = _mm256_mul_pd( bb, af );
		res = _mm256_add_pd( res, tmp );
#endif

		// [ -++- ] b.Z * a.YXWZ
		bb = vectorSplatZ( b );
		tmp = _mm256_permute_pd( a, _MM_SHUFFLE2( 0, 1 ) );
		tmp = _mm256_xor_pd( tmp, invertHigh );
#if _AM_FMA3_INTRINSICS_
		res = _mm256_fmaddsub_pd( bb, tmp, res );
#else
		tmp = _mm256_mul_pd( bb, tmp );
		res = _mm256_addsub_pd( res, tmp );
#endif

		return res;
	}

	__m256d _AM_CALL_ quaternionRollPitchYaw( __m256d angles );
}