#include "AvxMath.h"
#include <array>

namespace AvxMath
{
	const struct sMiscConstants g_misc;

	constexpr double g_pi = 3.141592653589793238;

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

		// For sine, multiply by X; for cos, multiply by the sign
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