#include "testDx.h"
#include "testsMisc.h"
#include <DirectXMath.h>

struct Vec
{
	__m128 sse;
	__m256d avx;

	operator __m128( ) const { return sse; }
	operator __m256d( ) const { return avx; }

	Vec( float x, float y, float z, float w )
	{
		sse = _mm_setr_ps( x, y, z, w );
		avx = _mm256_cvtps_pd( sse );
	}
	Vec() = default;

	void assertEqual() const
	{
		::assertEqual( _mm256_cvtps_pd( sse ), avx );
	}
};

bool testDx()
{
	const Vec a{ 1, 2, 3, 4 };
	const Vec b{ 5, 6, 7, 8 };

	Vec test;
	using namespace DirectX;
	using namespace AvxMath;

	test.sse = XMVector4Dot( a, b );
	test.avx = vector4Dot( a, b );
	test.assertEqual();

	test.sse = XMVector3Dot( a, b );
	test.avx = vector3Dot( a, b );
	test.assertEqual();

	test.sse = XMVector2Dot( a, b );
	test.avx = dup2( vector2Dot( low2( a ), low2( b ) ) );
	test.assertEqual();

	test.sse = XMVector3Cross( a, b );
	test.avx = vector3Cross( a, b );
	test.assertEqual();

	Vec test2;
	XMVectorSinCos( &test.sse, &test2.sse, a );
	vectorSinCos( test.avx, test2.avx, a );
	test.assertEqual();
	test2.assertEqual();

	test.sse = XMQuaternionRotationRollPitchYawFromVector( a );
	test.avx = quaternionRollPitchYaw( a );
	test.assertEqual();

	test2.sse = XMQuaternionMultiply( test.sse, test.sse );
	test2.avx = quaternionMultiply( test.avx, test.avx );
	test2.assertEqual();

	test.sse = XMQuaternionRotationAxis( a, 13 );
	test.avx = quaternionRotationAxis( a, 13 );
	test.assertEqual();

	return true;
}