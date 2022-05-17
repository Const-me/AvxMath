#include "testDx.h"
#include "AvxMath/AvxMath.h"
#include <DirectXMath.h>
#include <cmath>

static void assertEqual( __m256d a, __m256d b )
{
	__m256d diff = _mm256_sub_pd( a, b );
	using namespace AvxMath;
	diff = vectorAbs( diff );
	constexpr double tolerance = 1E-6;
	if( vector4InBounds( diff, _mm256_set1_pd( tolerance ) ) )
		return;
	__debugbreak();
}

static void assertEqual( __m128d a, __m128d b )
{
	assertEqual( _mm256_setr_m128d( a, a ), _mm256_setr_m128d( b, b ) );
}

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

inline __m256d stdSin( __m256d v )
{
	using namespace AvxMath;
	return _mm256_setr_pd(
		std::sin( vectorGetX( v ) ),
		std::sin( vectorGetY( v ) ),
		std::sin( vectorGetZ( v ) ),
		std::sin( vectorGetW( v ) )
	);
}

inline __m256d stdCos( __m256d v )
{
	using namespace AvxMath;
	return _mm256_setr_pd(
		std::cos( vectorGetX( v ) ),
		std::cos( vectorGetY( v ) ),
		std::cos( vectorGetZ( v ) ),
		std::cos( vectorGetW( v ) )
	);
}

inline __m128d stdSinCos( double a )
{
	return _mm_setr_pd( std::cos( a ), std::sin( a ) );
}

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

	assertEqual( test.avx, stdSin( a ) );
	assertEqual( test2.avx, stdCos( a ) );

	for( int i = -4; i <= 4; i++ )
	{
		__m128d my = scalarSinCos( i );
		__m128d std = stdSinCos( i );
		__m128d my2 = _mm_setr_pd( scalarCos( i ), scalarSin( i ) );
		assertEqual( std, my );
		assertEqual( std, my2 );
	}

	test.sse = XMQuaternionRotationRollPitchYawFromVector( a );
	test.avx = quaternionRollPitchYaw( a );
	test.assertEqual();

	test2.sse = XMQuaternionMultiply( test.sse, test.sse );
	test2.avx = quaternionMultiply( test.avx, test.avx );
	test2.assertEqual();

	return true;
}