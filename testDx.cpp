#include "testDx.h"
#include "AvxMath/AvxMath.h"
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

	void assertEqual()
	{
		__m256d ref = _mm256_cvtps_pd( sse );
		__m256d diff = _mm256_sub_pd( ref, avx );
		using namespace AvxMath;
		diff = vectorAbs( diff );

		constexpr double tolerance = 1E-6;
		if( vector4InBounds( diff, _mm256_set1_pd( tolerance ) ) )
			return;

		__debugbreak();
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

	return true;
}