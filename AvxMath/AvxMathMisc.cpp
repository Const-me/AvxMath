#include "AvxMath.h"
#include <array>

namespace AvxMath
{
	const struct sMiscConstants g_misc;

	alignas( 32 ) static const struct
	{
		const double pi = g_pi;
		const double halfPi = g_pi / 2;
		const double twoPi = g_pi * 2.0;
		const double inv2pi = 1.0 / ( 2.0 * g_pi );
	}
	g_piConstants;

	inline __m256d vectorModAngles( __m256d a )
	{
		const __m256d inv2pi = broadcast( g_piConstants.inv2pi );
		__m256d v = _mm256_mul_pd( a, inv2pi );
		v = _mm256_round_pd( v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );

		const __m256d twoPi = broadcast( g_piConstants.twoPi );
		v = _mm256_mul_pd( v, twoPi );

		return _mm256_sub_pd( a, v );
	}

	inline double scalarModAngles( double a )
	{
		double v = a * g_piConstants.inv2pi;
		v = round( v );
		v *= g_piConstants.twoPi;
		return a - v;
	}

	// Merged and interleaved magic numbers from g_XMCosCoefficients0, g_XMSinCoefficients0, g_XMCosCoefficients1, and g_XMSinCoefficients1 vectors
	// The X lanes correspond to cosine, Y lanes to sine.
	// This way scalarSinCos function can use memory operands for optimal performance.
	// The rest of the functions which use these numbers are loading scalars, RAM layout doesn't matter much.
	static const std::array<__m128d, 5> g_cosSinCoefficients
	{
		_mm_setr_pd( -0.5,           -0.16666667 ),
		_mm_setr_pd( +0.041666638,   +0.0083333310 ),
		_mm_setr_pd( -0.0013888378,  -0.00019840874 ),
		_mm_setr_pd( +2.4760495e-05, +2.7525562e-06 ),
		_mm_setr_pd( -2.6051615e-07, -2.3889859e-08 ),
	};

	void _AM_CALL_ vectorSinCos( __m256d& sin, __m256d& cos, __m256d x )
	{
		// Force the value within the bounds of pi
		x = vectorModAngles( x );

		const __m256d one = broadcast( g_misc.one );
		const __m256d neg0 = broadcast( g_misc.negativeZero );

		// Map in [-pi/2,pi/2] with sin(y) = sin(x), cos(y) = sign*cos(x).
		__m256d sign = _mm256_and_pd( x, neg0 );
		__m256d c = _mm256_or_pd( broadcast( g_piConstants.pi ), sign );  // pi when x >= 0, -pi when x < 0
		__m256d absx = _mm256_andnot_pd( sign, x );  // |x|
		__m256d rflx = _mm256_sub_pd( c, x );
		__m256d comp = _mm256_cmp_pd( absx, broadcast( g_piConstants.halfPi ), _CMP_LE_OQ );
		x = _mm256_blendv_pd( rflx, x, comp );
		sign = _mm256_andnot_pd( comp, neg0 );

		const __m256d x2 = _mm256_mul_pd( x, x );

		const double* const coeffs = (const double*)g_cosSinCoefficients.data();
		// Compute polynomial approximation of sine
		__m256d vConstants = broadcast( coeffs[ 9 ] );
		__m256d Result = _mm256_mul_pd( vConstants, x2 );

		vConstants = broadcast( coeffs[ 7 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( coeffs[ 5 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( coeffs[ 3 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( coeffs[ 1 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );
		Result = _mm256_add_pd( Result, one );
		Result = _mm256_mul_pd( Result, x );
		sin = Result;

		// Compute polynomial approximation of cosine
		vConstants = broadcast( coeffs[ 8 ] );
		Result = _mm256_mul_pd( vConstants, x2 );

		vConstants = broadcast( coeffs[ 6 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( coeffs[ 4 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( coeffs[ 2 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );

		vConstants = broadcast( coeffs[ 0 ] );
		Result = _mm256_add_pd( Result, vConstants );
		Result = _mm256_mul_pd( Result, x2 );
		Result = _mm256_add_pd( Result, one );
		Result = _mm256_or_pd( Result, sign );
		cos = Result;
	}

	__m128d scalarSinCos( double a )
	{
		a = scalarModAngles( a );
		__m128d x = _mm_set1_pd( a );

		const __m128d one = broadcast2( g_misc.one );
		const __m128d neg0 = broadcast2( g_misc.negativeZero );

		// Map in [-pi/2,pi/2] with sin(y) = sin(x), cos(y) = sign*cos(x).
		__m128d sign = _mm_and_pd( x, neg0 );
		__m128d c = _mm_or_pd( broadcast2( g_piConstants.pi ), sign );  // pi when x >= 0, -pi when x < 0
		__m128d absx = _mm_andnot_pd( sign, x );  // |x|
		__m128d rflx = _mm_sub_pd( c, x );
		__m128d comp = _mm_cmp_pd( absx, broadcast2( g_piConstants.halfPi ), _CMP_LE_OQ );
		x = _mm_blendv_pd( rflx, x, comp );
		sign = _mm_andnot_pd( comp, neg0 );

		const __m128d x2 = _mm_mul_pd( x, x );
		const __m128d* const coeffs = g_cosSinCoefficients.data();

		// Compute both polynomials using 2 lanes of the SSE vector
		__m128d vec = _mm_mul_pd( x2, coeffs[ 4 ] );

		vec = _mm_add_pd( vec, coeffs[ 3 ] );
		vec = _mm_mul_pd( vec, x2 );

		vec = _mm_add_pd( vec, coeffs[ 2 ] );
		vec = _mm_mul_pd( vec, x2 );

		vec = _mm_add_pd( vec, coeffs[ 1 ] );
		vec = _mm_mul_pd( vec, x2 );

		vec = _mm_add_pd( vec, coeffs[ 0 ] );
		vec = _mm_mul_pd( vec, x2 );
		vec = _mm_add_pd( vec, one );

		// For sine, multiply by X; for cosine, multiply by the sign
		__m128d mul = _mm_or_pd( one, sign );
		mul = _mm_blend_pd( mul, x, 0b10 );
		return _mm_mul_pd( vec, mul );
	}

	double scalarSin( double a )
	{
		a = scalarModAngles( a );

		__m128d x = _mm_set_sd( a );

		const __m128d one = _mm_load_sd( &g_misc.one );
		const __m128d neg0 = _mm_load_sd( &g_misc.negativeZero );

		__m128d sign = _mm_and_pd( x, neg0 );
		__m128d c = _mm_or_pd( _mm_load_sd( &g_piConstants.pi ), sign );  // pi when x >= 0, -pi when x < 0
		__m128d absx = _mm_andnot_pd( sign, x );  // |x|
		__m128d rflx = _mm_sub_sd( c, x );
		__m128d comp = _mm_cmp_sd( absx, _mm_load_sd( &g_piConstants.halfPi ), _CMP_LE_OQ );
		x = _mm_blendv_pd( rflx, x, comp );

		double x2 = _mm_cvtsd_f64( x );
		x2 *= x2;

		const double* const coeffs = (const double*)g_cosSinCoefficients.data();

		// Compute polynomial approximation of sine
		double res = x2 * coeffs[ 9 ];

		res += coeffs[ 7 ];
		res *= x2;

		res += coeffs[ 5 ];
		res *= x2;

		res += coeffs[ 3 ];
		res *= x2;

		res += coeffs[ 1 ];
		res *= x2;
		res += _mm_cvtsd_f64( one );

		return res * _mm_cvtsd_f64( x );
	}

	double scalarCos( double a )
	{
		a = scalarModAngles( a );

		__m128d x = _mm_set_sd( a );

		const __m128d one = _mm_load_sd( &g_misc.one );
		const __m128d neg0 = _mm_load_sd( &g_misc.negativeZero );

		__m128d sign = _mm_and_pd( x, neg0 );
		__m128d c = _mm_or_pd( _mm_load_sd( &g_piConstants.pi ), sign );  // pi when x >= 0, -pi when x < 0
		__m128d absx = _mm_andnot_pd( sign, x );  // |x|
		__m128d rflx = _mm_sub_sd( c, x );
		__m128d comp = _mm_cmp_sd( absx, _mm_load_sd( &g_piConstants.halfPi ), _CMP_LE_OQ );
		x = _mm_blendv_pd( rflx, x, comp );
		sign = _mm_andnot_pd( comp, neg0 );

		double x2 = _mm_cvtsd_f64( x );
		x2 *= x2;

		const double* const coeffs = (const double*)g_cosSinCoefficients.data();

		// Compute polynomial approximation of cosine
		double res = x2 * coeffs[ 8 ];

		res += coeffs[ 6 ];
		res *= x2;

		res += coeffs[ 4 ];
		res *= x2;

		res += coeffs[ 2 ];
		res *= x2;

		res += coeffs[ 0 ];
		res *= x2;
		res += _mm_cvtsd_f64( one );

		__m128d tmp = _mm_set_sd( res );
		tmp = _mm_or_pd( tmp, sign );
		return _mm_cvtsd_f64( tmp );
	}

	static const struct
	{
		const double invPi = 1.0 / g_pi;
		const double mul0 = -424539.12324285670928;  // -135135 * Pi
		const double div0 = -135135;                 // -135135
		const double mul2 = 537183.74348619438457;   // 62370 * Pi^2
		const double div2 = 615567.22649594329707;   // 62370 * Pi^2
		const double mul4 = -115675.44084883638934;  // -378 * Pi^5
		const double div4 = -306838.63675710767731;  // -3150 * Pi^4
		const double mul6 = 3020.2932277767920678;   // Pi^7
		const double div6 = 26918.897420108524239;   // 28 * Pi^6
	}
	g_TanConstants;

	__m256d _AM_CALL_ vectorTan( __m256d a )
	{
		// Wrap into [ -pi/2 .. +pi/2 ] interval.
		// Don't multiply back, we include that multiplier into these Padé magic numbers.
		a = _mm256_mul_pd( a, broadcast( g_TanConstants.invPi ) );
		__m256d tmp = _mm256_round_pd( a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
		a = _mm256_sub_pd( a, tmp );

#if 1
		__m256d mul = broadcast( g_TanConstants.mul0 );
		__m256d div = broadcast( g_TanConstants.div0 );

		const __m256d x2 = _mm256_mul_pd( a, a );
		mul = vectorMultiplyAdd( broadcast( g_TanConstants.mul2 ), x2, mul );	// 1 + mul2*x^2
		div = vectorMultiplyAdd( broadcast( g_TanConstants.div2 ), x2, div );	// 1 + div2*x^2

		const __m256d x4 = _mm256_mul_pd( x2, x2 );
		mul = vectorMultiplyAdd( broadcast( g_TanConstants.mul4 ), x4, mul );	// 1 + a1*x^2 + a2*x^4
		div = vectorMultiplyAdd( broadcast( g_TanConstants.div4 ), x4, div );	// 1 + b1*x^2 + b2*x^4

		const __m256d x6 = _mm256_mul_pd( x4, x2 );
		mul = vectorMultiplyAdd( broadcast( g_TanConstants.mul6 ), x6, mul );	// 1 + a1*x^2 + a2*x^4
		div = vectorMultiplyAdd( broadcast( g_TanConstants.div6 ), x6, div );	// 1 + b1*x^2 + b2*x^4

#else
		// Because the source value is in [ -0.5 .. 0.5 ] as opposed to (-pi/2 .. +pi/2), we're sure x^6 << x^4 << x^2.
		// For optimal numerical precision, starting to compute these polynomials with smaller numbers.
		// Practically speaking, the difference is very small.

		const __m256d x2 = _mm256_mul_pd( a, a );
		const __m256d x4 = _mm256_mul_pd( x2, x2 );
		const __m256d x6 = _mm256_mul_pd( x4, x2 );

		__m256d mul = broadcast( g_TanConstants.mul6 );
		__m256d div = broadcast( g_TanConstants.div6 );
		mul = _mm256_mul_pd( mul, x6 );
		div = _mm256_mul_pd( div, x6 );

		mul = vectorMultiplyAdd( broadcast( g_TanConstants.mul4 ), x4, mul );	// 1 + a1*x^2 + a2*x^4
		div = vectorMultiplyAdd( broadcast( g_TanConstants.div4 ), x4, div );	// 1 + b1*x^2 + b2*x^4

		mul = vectorMultiplyAdd( broadcast( g_TanConstants.mul2 ), x2, mul );	// 1 + mul2*x^2
		div = vectorMultiplyAdd( broadcast( g_TanConstants.div2 ), x2, div );	// 1 + div2*x^2

		mul = _mm256_add_pd( mul, broadcast( g_TanConstants.mul0 ) );
		div = _mm256_add_pd( div, broadcast( g_TanConstants.div0 ) );
#endif

		mul = _mm256_mul_pd( mul, a );
		return _mm256_div_pd( mul, div );
	}

	double scalarTan( double a )
	{
		// Wrap into [ -pi/2 .. +pi/2 ] interval.
		a *= g_TanConstants.invPi;
		a -= round( a );

		// Use 16-byte loads to compute polynomials for both numerator and denominator in two lanes of the vector
		const __m128d av = _mm_set1_pd( a );
		__m128d acc = _mm_loadu_pd( &g_TanConstants.mul0 );
		const __m128d x2 = _mm_mul_pd( av, av );
		acc = vectorMultiplyAdd( x2, _mm_loadu_pd( &g_TanConstants.mul2 ), acc );

		const __m128d x4 = _mm_mul_pd( x2, x2 );
		acc = vectorMultiplyAdd( x4, _mm_loadu_pd( &g_TanConstants.mul4 ), acc );

		const __m128d x6 = _mm_mul_pd( x4, x2 );
		acc = vectorMultiplyAdd( x6, _mm_loadu_pd( &g_TanConstants.mul6 ), acc );

		// Extract lanes from the vector, compute final result
		double mul = vectorGetX( acc ) * vectorGetX( av );
		double div = vectorGetY( acc );
		return mul / div;
	}

	__m256d _AM_CALL_ vectorCot( __m256d a )
	{
		// cot( a ) = tan( Pi/2 - a )
		// https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Reflections
		a = _mm256_sub_pd( broadcast( g_piConstants.halfPi ), a );
		return vectorTan( a );
	}

	double scalarCot( double a )
	{
		return scalarTan( g_piConstants.halfPi - a );
	}

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