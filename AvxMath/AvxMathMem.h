// Load and store routines
#pragma once

namespace AvxMath
{
	// Load 3D vector, set W to 0.0f
	inline __m256d loadDouble3( const double* rsi )
	{
		__m128d low = _mm_loadu_pd( rsi );
		__m128d high = _mm_load_sd( rsi + 2 );
		return _mm256_setr_m128d( low, high );
	}

	// Load 4D vector
	inline __m256d loadDouble4( const double* rsi )
	{
		return _mm256_loadu_pd( rsi );
	}

	// Store 3D vector
	inline void storeDouble3( double* rdi, __m256d vec )
	{
		_mm_storeu_pd( rdi, low2( vec ) );
		_mm_store_sd( rdi + 2, high2( vec ) );
	}

	// Store 4D vector
	inline void storeDouble4( double* rdi, __m256d vec )
	{
		return _mm256_storeu_pd( rdi, vec );
	}

	// Load 4x4 matrix
	inline Matrix4x4 loadMatrix( const double* rsi )
	{
		Matrix4x4 res;
		res.r0 = _mm256_loadu_pd( rsi );
		res.r1 = _mm256_loadu_pd( rsi + 4 );
		res.r2 = _mm256_loadu_pd( rsi + 8 );
		res.r3 = _mm256_loadu_pd( rsi + 12 );
		return res;
	}

	// Store 4x4 matrix
	inline void storeMatrix( double* rdi, Matrix4x4 mat )
	{
		_mm256_storeu_pd( rdi, mat.r0 );
		_mm256_storeu_pd( rdi + 4, mat.r1 );
		_mm256_storeu_pd( rdi + 8, mat.r2 );
		_mm256_storeu_pd( rdi + 12, mat.r3 );
	}

	// Load 4x4 matrix, transposing on the fly to the column-major layout
	// Slightly faster than transposing with vector shuffles after loading
	inline Matrix4x4 loadMatrixTransposed( const double* rsi, size_t stride = 4 )
	{
		assert( stride >= 4 );

		// Load top half of the matrix into low half of 4 registers
		__m256d t0 = _mm256_castpd128_pd256( _mm_loadu_pd( rsi ) );     // 00, 01
		__m256d t1 = _mm256_castpd128_pd256( _mm_loadu_pd( rsi + 2 ) ); // 02, 03
		rsi += stride;
		__m256d t2 = _mm256_castpd128_pd256( _mm_loadu_pd( rsi ) );     // 10, 11
		__m256d t3 = _mm256_castpd128_pd256( _mm_loadu_pd( rsi + 2 ) ); // 12, 13
		rsi += stride;
		// Load bottom half of the matrix into high half of these registers
		t0 = _mm256_insertf128_pd( t0, _mm_loadu_pd( rsi ), 1 );    // 00, 01, 20, 21
		t1 = _mm256_insertf128_pd( t1, _mm_loadu_pd( rsi + 2 ), 1 );// 02, 03, 22, 23
		rsi += stride;
		t2 = _mm256_insertf128_pd( t2, _mm_loadu_pd( rsi ), 1 );    // 10, 11, 30, 31
		t3 = _mm256_insertf128_pd( t3, _mm_loadu_pd( rsi + 2 ), 1 );// 12, 13, 32, 33

		Matrix4x4 mat;
		// Transpose 2x2 blocks in registers.
		// Due to the tricky way we loaded stuff, that's enough to transpose the complete 4x4 matrix.
		mat.r0 = _mm256_unpacklo_pd( t0, t2 ); // 00, 10, 20, 30
		mat.r1 = _mm256_unpackhi_pd( t0, t2 ); // 01, 11, 21, 31
		mat.r2 = _mm256_unpacklo_pd( t1, t3 ); // 02, 12, 22, 32
		mat.r3 = _mm256_unpackhi_pd( t1, t3 ); // 03, 13, 23, 33
		return mat;
	}
}