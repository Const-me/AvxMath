#pragma once

namespace AvxMath
{
	// Transforms 4D vector by the matrix
	inline __m256d vector4Transform( __m256d vec, const Matrix4x4& mat )
	{
		// Compute the 16 products
		__m256d a = _mm256_mul_pd( vec, mat.r0 );
		__m256d b = _mm256_mul_pd( vec, mat.r1 );
		__m256d c = _mm256_mul_pd( vec, mat.r2 );
		__m256d d = _mm256_mul_pd( vec, mat.r3 );

		// Add even/odd lanes duplicating the values
		constexpr int flip = 0b0101;
		a = _mm256_add_pd( a, _mm256_permute_pd( a, flip ) );
		b = _mm256_add_pd( b, _mm256_permute_pd( b, flip ) );
		c = _mm256_add_pd( c, _mm256_permute_pd( c, flip ) );
		d = _mm256_add_pd( d, _mm256_permute_pd( d, flip ) );

		// Merge vectors pairwise, blend is slightly faster than shuffle or unpack
		__m256d ab = _mm256_blend_pd( a, b, 0b1010 );
		__m256d cd = _mm256_blend_pd( c, d, 0b1010 );

		// Compute [ ab.x + ab.z, ab.y + ab.w, cd.x + cd.z, cd.y + cd.w ]
		__m256d t2 = _mm256_permute2f128_pd( ab, cd, 0x31 );	// ab.z, ab.w, cd.z, cd.w
		__m256d t1 = _mm256_insertf128_pd( ab, low2( cd ), 1 );	// ab.x, ab.y, cd.x, cd.y
		__m256d res = _mm256_add_pd( t1, t2 );
		return res;
	}

	// Transform 3D vector by the matrix. This function ignores W component of the input vector, uses 1.0 instead.
	inline __m256d vector3Transform( __m256d vec, const Matrix4x4& mat )
	{
		vec = vector3Homogeneous( vec );
		return vector4Transform( vec, mat );
	}

	// Transforms 4D vector by the column-major matrix; substantially faster than vector4Transform
	inline __m256d vector4TransformTransposed( __m256d vec, const Matrix4x4& mat )
	{
		// Compute the 16 products
		__m256d a = _mm256_mul_pd( vec, mat.r0 );
		__m256d b = _mm256_mul_pd( vec, mat.r1 );
		__m256d c = _mm256_mul_pd( vec, mat.r2 );
		__m256d d = _mm256_mul_pd( vec, mat.r3 );

		// For the column-major matrix we just need to sum all 4 of these vectors
		return _mm256_add_pd( _mm256_add_pd( a, b ), _mm256_add_pd( c, d ) );
	}

	// Transpose 4x4 matrix in place
	inline void matrixTranspose( Matrix4x4& mat )
	{
		__m256d a = _mm256_unpacklo_pd( mat.r0, mat.r1 );
		__m256d b = _mm256_unpackhi_pd( mat.r0, mat.r1 );
		__m256d c = _mm256_unpacklo_pd( mat.r2, mat.r3 );
		__m256d d = _mm256_unpackhi_pd( mat.r2, mat.r3 );

		mat.r2 = _mm256_permute2f128_pd( a, c, 0x31 );
		mat.r3 = _mm256_permute2f128_pd( b, d, 0x31 );
		mat.r0 = _mm256_insertf128_pd( a, low2( c ), 1 );
		mat.r1 = _mm256_insertf128_pd( b, low2( d ), 1 );
	}

	// Compute product of two matrices
	inline Matrix4x4 matrixMultiply( const Matrix4x4& a, const Matrix4x4& b )
	{
		Matrix4x4 result;

		result.r0 = _mm256_mul_pd( vectorSplatX( a.r0 ), b.r0 );
		result.r1 = _mm256_mul_pd( vectorSplatY( a.r0 ), b.r1 );
		result.r2 = _mm256_mul_pd( vectorSplatZ( a.r0 ), b.r2 );
		result.r3 = _mm256_mul_pd( vectorSplatW( a.r0 ), b.r3 );

		result.r0 = _mm256_add_pd( result.r0, _mm256_mul_pd( vectorSplatX( a.r1 ), b.r0 ) );
		result.r1 = _mm256_add_pd( result.r1, _mm256_mul_pd( vectorSplatY( a.r1 ), b.r1 ) );
		result.r2 = _mm256_add_pd( result.r2, _mm256_mul_pd( vectorSplatZ( a.r1 ), b.r2 ) );
		result.r3 = _mm256_add_pd( result.r3, _mm256_mul_pd( vectorSplatW( a.r1 ), b.r3 ) );

		result.r0 = _mm256_add_pd( result.r0, _mm256_mul_pd( vectorSplatX( a.r2 ), b.r0 ) );
		result.r1 = _mm256_add_pd( result.r1, _mm256_mul_pd( vectorSplatY( a.r2 ), b.r1 ) );
		result.r2 = _mm256_add_pd( result.r2, _mm256_mul_pd( vectorSplatZ( a.r2 ), b.r2 ) );
		result.r3 = _mm256_add_pd( result.r3, _mm256_mul_pd( vectorSplatW( a.r2 ), b.r3 ) );

		result.r0 = _mm256_add_pd( result.r0, _mm256_mul_pd( vectorSplatX( a.r3 ), b.r0 ) );
		result.r1 = _mm256_add_pd( result.r1, _mm256_mul_pd( vectorSplatY( a.r3 ), b.r1 ) );
		result.r2 = _mm256_add_pd( result.r2, _mm256_mul_pd( vectorSplatZ( a.r3 ), b.r2 ) );
		result.r3 = _mm256_add_pd( result.r3, _mm256_mul_pd( vectorSplatW( a.r3 ), b.r3 ) );

		return result;
	}

	// Create an identity matrix
	inline Matrix4x4 matrixIdentity()
	{
		const __m256d zero = _mm256_setzero_pd();
		const __m256d one = broadcast( g_misc.one );
		Matrix4x4 m;
		m.r0 = _mm256_blend_pd( zero, one, 0b0001 );
		m.r1 = _mm256_blend_pd( zero, one, 0b0010 );
		m.r2 = _mm256_blend_pd( zero, one, 0b0100 );
		m.r3 = _mm256_blend_pd( zero, one, 0b1000 );
		return m;
	}
}