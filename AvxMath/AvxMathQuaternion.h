#pragma once

namespace AvxMath
{
	inline __m256d quaternionNormalize( __m256d q )
	{
		return vector4Normalize( q );
	}

	inline __m256d quaternionConjugate( __m256d q )
	{
		return vectorNegateLanes<0b0111>( q );
	}

	// Product of two quaternions
	__m256d _AM_CALL_ quaternionMultiply( __m256d a, __m256d b );

	// Create rotation quaternion based on a vector containing the Euler angles (pitch, yaw, and roll).
	__m256d _AM_CALL_ quaternionRollPitchYaw( __m256d angles );

	// Create rotation quaternion from normalized axis of rotation; angles are measured clockwise when looking along the rotation axis toward the origin.
	inline __m256d quaternionRotationNormal( __m256d normalAxis, double angle )
	{
		// Create a vector with [ sin, sin, sin, cos ]
		__m128d cosSin = scalarSinCos( angle * g_misc.oneHalf );
#if _AM_AVX2_INTRINSICS_
		__m256d q = _mm256_permute4x64_pd( _mm256_castpd128_pd256( cosSin ), _MM_SHUFFLE( 0, 1, 1, 1 ) );
#else
		__m128d ss = _mm_permute_pd( cosSin, _MM_SHUFFLE2( 1, 1 ) );
		__m128d sc = _mm_permute_pd( cosSin, _MM_SHUFFLE2( 0, 1 ) );
		__m256d q = _mm256_setr_m128d( ss, sc );
#endif
		// Set W lane of the normal to 1.0
		normalAxis = vector3Homogeneous( normalAxis );

		// Return their product
		return _mm256_mul_pd( normalAxis, q );
	}

	// Create rotation quaternion from axis of rotation; angles are measured clockwise when looking along the rotation axis toward the origin.
	inline __m256d quaternionRotationAxis( __m256d normalAxis, double angle )
	{
		return quaternionRotationNormal( vector3Normalize( normalAxis ), angle );
	}
}