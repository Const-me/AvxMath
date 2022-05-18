#include "AvxMath.h"
#include <array>

namespace AvxMath
{
	const struct sMiscConstants g_misc;



	static const struct TanhConstants
	{
		const double _600 = 600;
		const double _270 = 270;
		const double _70 = 70;
		const double _11 = 11;
		const double last = 1.0 / 24.0;
	}
	g_tanh;

	__m256d _AM_CALL_ vectorTanH( __m256d x )
	{
		// https://math.stackexchange.com/a/107666/467444

		const __m256d _600 = broadcast( g_tanh._600 );
		const __m256d x2 = _mm256_mul_pd( x, x );		// x^2

		__m256d b = broadcast( g_tanh._270 );
		__m256d den = vectorMultiplyAdd( x2, b, _600 );	// 600 + 270 * x^2

		const __m256d x4 = _mm256_mul_pd( x2, x2 );	// x^4
		__m256d num = _mm256_add_pd( x4, _600 );	// x^4 + 600

		const __m256d x6 = _mm256_mul_pd( x4, x2 );	// x^6
		b = broadcast( g_tanh._11 );
		den = vectorMultiplyAdd( x4, b, den );	// 600 + 270 * x^2 + 11 * x^4

		b = broadcast( g_tanh._70 );
		num = vectorMultiplyAdd( x2, b, num );	// x^4 + 70 * x^2 + 600

		b = broadcast( g_tanh.last );
		den = vectorMultiplyAdd( x6, b, den );	// 600 + 270 * x^2 + 11 * x^4 + (1/24)*x^6
		num = _mm256_mul_pd( num, x );	// x * ( x^4 + 70 * x^2 + 600 )

		return _mm256_div_pd( num, den );
	}
}