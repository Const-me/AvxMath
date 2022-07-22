#include "testStdlib.h"
#include <cmath>
#include <stdio.h>

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

inline __m256d stdTan( __m256d v )
{
	using namespace AvxMath;
	return _mm256_setr_pd(
		std::tan( vectorGetX( v ) ),
		std::tan( vectorGetY( v ) ),
		std::tan( vectorGetZ( v ) ),
		std::tan( vectorGetW( v ) )
	);
}

inline __m256d testScalarTan( __m256d v )
{
	using namespace AvxMath;
	return _mm256_setr_pd(
		scalarTan( vectorGetX( v ) ),
		scalarTan( vectorGetY( v ) ),
		scalarTan( vectorGetZ( v ) ),
		scalarTan( vectorGetW( v ) )
	);
}

inline __m128d stdSinCos( double a )
{
	return _mm_setr_pd( std::cos( a ), std::sin( a ) );
}

static void computeSinCosError()
{
	using namespace AvxMath;

	__m128d errors = _mm_setzero_pd();

	constexpr double mul = g_pi * 2.0 / 0x100000;
	for( int i = 0; i <= 0x100000; i++ )
	{
		__m128d my = scalarSinCos( i );
		__m128d std = stdSinCos( i );

		// Just in case, verify the sin/cos are in [ -1 .. +1 ] range - we don't want 1.0 + 1E-12 despite the 1E-12 is very small error
		__m128d a = vectorAbs( my );
		if( vectorGetX( a ) > 1 || vectorGetY( a ) > 1 )
		{
#ifdef _MSC_VER
			__debugbreak();
#else
			assert( false );
#endif
		}

		// Accumulate maximum absolute error
		__m128d diff = _mm_sub_pd( my, std );
		diff = vectorAbs( diff );
		errors = _mm_max_pd( errors, diff );
	}

	printf( "Maximum errors for sin/cos: %g / %g\n", vectorGetY( errors ), vectorGetX( errors ) );
}

bool testStdlib()
{
	using namespace AvxMath;

	const __m256d a = _mm256_setr_pd( 1, 2, 3, 4 );
	const __m256d b = _mm256_setr_pd( 5, 6, 7, 8 );

	__m256d x, y;
	vectorSinCos( x, y, a );

	assertEqual( x, stdSin( a ) );
	assertEqual( y, stdCos( a ) );

	x = vectorTan( a );
	y = stdTan( a );
	assertEqual( x, y );
	x = testScalarTan( a );
	assertEqual( x, y );

	for( int i = -4; i <= 4; i++ )
	{
		__m128d my = scalarSinCos( i );
		__m128d std = stdSinCos( i );
		__m128d my2 = _mm_setr_pd( scalarCos( i ), scalarSin( i ) );
		assertEqual( std, my );
		assertEqual( std, my2 );
	}

	computeSinCosError();
	return true;
}