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
	__m256d _AM_CALL_ quaternionMultiply( __m256d a, __m256d b );

	// 
	__m256d _AM_CALL_ quaternionRollPitchYaw( __m256d angles );
}