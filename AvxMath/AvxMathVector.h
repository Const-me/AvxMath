// Routines operating on 2D, 3D and 4D vectors with FP64 lanes
#pragma once

namespace AvxMath
{
	// Compute dot product of 4D vectors, broadcast to both lanes of SSE vector
	inline __m128d vector4Dot2( __m256d a, __m256d b )
	{
		__m256d p = _mm256_mul_pd( a, b );
		__m128d low = low2( p );
		__m128d high = high2( p );

		low = _mm_add_pd( low, high );	// [ z + x, y + w ]
		low = _mm_add_pd( low, _mm_permute_pd( low, 0b01 ) ); // z + x + y + w in both lanes
		return low;
	}

	// Compute dot product of 4D vectors, broadcast to all 4 lanes of AVX vector
	inline __m256d vector4Dot( __m256d a, __m256d b )
	{
		__m128d dp = vector4Dot2( a, b );
#if _AM_AVX2_INTRINSICS_
		return _mm256_broadcastsd_pd( dp );
#else
		return dup2( dp );
#endif
	}

	// ==== 2D vectors ====

	inline __m128d vector2Dot( __m128d a, __m128d b )
	{
		return _mm_dp_pd( a, b, 0b00110011 );
	}
	inline __m128d vector2LengthSq( __m128d a )
	{
		return vector2Dot( a, a );
	}
	inline __m128d vector2Length( __m128d a )
	{
		return _mm_sqrt_pd( vector2LengthSq( a ) );
	}
	// Compute 2D cross product
	inline __m128d vector2Cross( __m128d a, __m128d b )
	{
		// According to XMVector2Cross, the formula is a.x * b.y - a.y * b.x
		b = _mm_permute_pd( b, _MM_SHUFFLE2( 0, 1 ) );	// [ b.y, b.x ]
		__m128d res = _mm_mul_pd( a, b );	// [ a.x * b.y, a.y * b.x ]
		res = _mm_sub_sd( res, _mm_permute_pd( res, _MM_SHUFFLE2( 1, 1 ) ) );
		return _mm_permute_pd( res, _MM_SHUFFLE2( 0, 0 ) );
	}

	// Normalize a 2D vector. For a vector of length 0, returns zero vector. For a vector with infinite length, returns a vector of QNaN
	inline __m128d vector2Normalize( __m128d v )
	{
		__m128d lsq = vector2LengthSq( v );
		// Assuming zero-length or infinite vectors are rare and the following branches are both well-predicted
		double s = _mm_cvtsd_f64( lsq );
		if( s != g_misc.infinity )
		{
			if( s > 0 )
				return _mm_div_pd( v, _mm_sqrt_pd( lsq ) );
			return _mm_setzero_pd();
		}
		return _mm_loaddup_pd( &g_misc.quietNaN );
	}

	// ==== 3D vectors ====

	// Dot product of 3D vectors, broadcast to both lanes of SSE vector
	inline __m128d vector3Dot2( __m256d a, __m256d b )
	{
		__m256d p = _mm256_mul_pd( a, b );
		__m128d low = low2( p );
		__m128d high = high2( p );

		low = _mm_add_sd( low, high );	// [ z + x, y ]
		low = _mm_add_pd( low, _mm_permute_pd( low, _MM_SHUFFLE2( 0, 1 ) ) ); // z + x + y in both lanes
		return low;
	}

	// Compute dot product of 3D vectors, broadcast to all 4 lanes of AVX vector
	inline __m256d vector3Dot( __m256d a, __m256d b )
	{
		__m128d dp = vector3Dot2( a, b );
#if _AM_AVX2_INTRINSICS_
		return _mm256_broadcastsd_pd( dp );
#else
		return dup2( dp );
#endif
	}

	// Normalize a 3D vector. For a vector of length 0, returns zero vector. For a vector with infinite length, returns a vector of QNaN
	inline __m256d vector3Normalize( __m256d vec )
	{
		__m128d lsq = vector3Dot2( vec, vec );

		// Assuming zero-length or infinite vectors are rare and the following branches are both well-predicted
		double s = _mm_cvtsd_f64( lsq );
		if( s != g_misc.infinity )
		{
			if( s > 0 )
			{
#if _AM_AVX2_INTRINSICS_
				__m128d len1 = _mm_sqrt_sd( lsq, lsq );
				__m256d len4 = _mm256_broadcastsd_pd( len1 );
#else
				__m128d len2 = _mm_sqrt_pd( lsq );
				__m256d len4 = dup2( len2 );
#endif
				return _mm256_div_pd( vec, len4 );
			}
			return _mm256_setzero_pd();
		}
		return _mm256_broadcast_sd( &g_misc.quietNaN );
	}

	// Compute cross product between two 3D vectors
	// The unused W lane is set to 0 unless there's INF or NAN in W lanes of the inputs
	inline __m256d vector3Cross( __m256d a, __m256d b )
	{
		// The formula is a.yzx * b.zxy - a.zxy * b.yzx
		// https://en.wikipedia.org/wiki/Cross_product#Coordinate_notation

#if _AM_AVX2_INTRINSICS_
		const __m256d a1 = _mm256_permute4x64_pd( a, _MM_SHUFFLE( 3, 0, 2, 1 ) );	// a.yzxw
		const __m256d b2 = _mm256_permute4x64_pd( b, _MM_SHUFFLE( 3, 1, 0, 2 ) );	// b.zxyw
		const __m256d a2 = _mm256_permute4x64_pd( a, _MM_SHUFFLE( 3, 1, 0, 2 ) );	// a.zxyw
		const __m256d b1 = _mm256_permute4x64_pd( b, _MM_SHUFFLE( 3, 0, 2, 1 ) );	// b.yzxw
#else
		const __m256d af = flipHighLow( a );	// a.zwxy
		const __m256d bf = flipHighLow( b );	// b.zwxy

		__m256d a1 = _mm256_shuffle_pd( a, af, 0b0101 ); // a.yzwx
		__m256d b1 = _mm256_shuffle_pd( b, bf, 0b0101 ); // b.yzwx
		const __m256d b2 = _mm256_shuffle_pd( bf, b, 0b1100 ); // b.zxyw
		const __m256d a2 = _mm256_shuffle_pd( af, a, 0b1100 ); // a.zxyw
		a1 = _mm256_permute_pd( a1, 0b0110 );	// a.yzxw
		b1 = _mm256_permute_pd( b1, 0b0110 );	// b.yzxw
#endif
		return _mm256_sub_pd( _mm256_mul_pd( a1, b2 ), _mm256_mul_pd( a2, b1 ) );
	}

	// Convert homogeneous vector to Cartesian, i.e. divide by W
	inline __m256d vector4Carthesian( __m256d vec )
	{
		return _mm256_div_pd( vec, vectorSplatW( vec ) );
	}

	// Convert Cartesian vector to homogeneous, i.e. reset W to 1.0
	inline __m256d vector3Homogeneous( __m256d vec )
	{
		__m256d bc = broadcast( g_misc.one );
		return _mm256_blend_pd( vec, bc, 0b1000 );
	}

	// Convert 2D Cartesian vector to 4D homogeneous, i.e. set ZW lanes to [ 0, 1 ]
	inline __m256d vector2Homogeneous( __m128d vec )
	{
		__m128d high = _mm_setzero_pd();
		high = _mm_loadh_pd( high, &g_misc.one );
		return _mm256_setr_m128d( vec, high );
	}

	// Normalize a 4D vector. For a vector of length 0, returns zero vector. For a vector with infinite length, returns a vector of QNaN
	inline __m256d vector4Normalize( __m256d vec )
	{
		__m256d p = _mm256_mul_pd( vec, vec );

		__m128d lsq = _mm_add_pd( low2( p ), high2( p ) );
		lsq = _mm_add_pd( lsq, _mm_permute_pd( lsq, _MM_SHUFFLE2( 0, 1 ) ) );

		double s = _mm_cvtsd_f64( lsq );
		if( s != g_misc.infinity )
		{
			if( s > 0 )
			{
#if _AM_AVX2_INTRINSICS_
				__m128d len1 = _mm_sqrt_sd( lsq, lsq );
				__m256d len4 = _mm256_broadcastsd_pd( len1 );
#else
				__m128d len2 = _mm_sqrt_pd( lsq );
				__m256d len4 = dup2( len2 );
#endif
				return _mm256_div_pd( vec, len4 );
			}
			return _mm256_setzero_pd();
		}
		return broadcast( g_misc.quietNaN );
	}

	// Clamp 4 values into [ 0 .. 1 ] interval
	inline __m256d saturate( __m256d vec )
	{
		const __m256d one = broadcast( g_misc.one );
		vec = _mm256_max_pd( vec, _mm256_setzero_pd() );
		return _mm256_min_pd( vec, one );
	}

	// Clamp 2 values into [ 0 .. 1 ] interval
	inline __m128d saturate( __m128d vec )
	{
		const __m128d one = broadcast2( g_misc.one );
		vec = _mm_max_pd( vec, _mm_setzero_pd() );
		return _mm_min_pd( vec, one );
	}
}