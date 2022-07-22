#pragma once
#include "AvxMath/AvxMath.h"
#include <assert.h>

static void assertEqual( __m256d a, __m256d b )
{
	__m256d diff = _mm256_sub_pd( a, b );
	using namespace AvxMath;
	diff = vectorAbs( diff );
	constexpr double tolerance = 1E-6;
	if( vector4InBounds( diff, _mm256_set1_pd( tolerance ) ) )
		return;
#ifdef _MSC_VER
	__debugbreak();
#else
	assert( false );
#endif
}

static void assertEqual( __m128d a, __m128d b )
{
	using namespace AvxMath;
	assertEqual( dup2( a ), dup2( b ) );
}