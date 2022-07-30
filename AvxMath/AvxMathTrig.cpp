#include "AvxMath.h"
#include <array>

namespace AvxMath
{
	static const double g_mulRadians = g_pi / 180.0;
	static const double g_mulDegrees = 180.0 / g_pi;

	__m256d radians( __m256d deg )
	{
		return _mm256_mul_pd( deg, broadcast( g_mulRadians ) );
	}
	__m128d radians( __m128d deg )
	{
		return _mm_mul_pd( deg, broadcast2( g_mulRadians ) );
	}
	double radians( double deg )
	{
		return deg * g_mulRadians;
	}

	__m256d degrees( __m256d rad )
	{
		return _mm256_mul_pd( rad, broadcast( g_mulDegrees ) );
	}
	__m128d degrees( __m128d rad )
	{
		return _mm_mul_pd( rad, broadcast2( g_mulDegrees ) );
	}
	double degrees( double rad )
	{
		return rad * g_mulDegrees;
	}

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

	// Interleaved magic numbers for cosine and sine polynomial approximations.
	// See GTE_C_COS_DEG10_C[1-5] and GTE_C_SIN_DEG11_C[1-5] macros in that header https://www.geometrictools.com/GTE/Mathematics/Math.h
	// The X lanes of these vectors correspond to cosine, Y lanes to sine.
	// This way scalarSinCos function can use memory operands for optimal performance.
	// The rest of the functions which use these numbers are loading scalars, RAM layout doesn't matter much.
	static const std::array<__m128d, 5> g_cosSinCoefficients
	{
		_mm_setr_pd( -4.9999999508695869e-01,  -1.6666666601721269e-01 ),
		_mm_setr_pd( +4.1666638865338612e-02,  +8.3333303183525942e-03 ),
		_mm_setr_pd( -1.3888377661039897e-03,  -1.9840782426250314e-04 ),
		_mm_setr_pd( +2.4760495088926859e-05,  +2.7521557770526783e-06 ),
		_mm_setr_pd( -2.6051615464872668e-07,  -2.3828544692960918e-08 ),
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

		// Compute both polynomial approximations, using 16-byte broadcast loads for the magic numbers
		const __m256d y1 = _mm256_unpacklo_pd( x2, x2 );	// x2.xxzz
		const __m256d y2 = _mm256_unpackhi_pd( x2, x2 );	// x2.yyww

		__m256d r1 = _mm256_broadcast_pd( &g_cosSinCoefficients[ 4 ] );
		__m256d r2 = r1;

		__m256d tmp = _mm256_broadcast_pd( &g_cosSinCoefficients[ 3 ] );
		r1 = vectorMultiplyAdd( r1, y1, tmp );
		r2 = vectorMultiplyAdd( r2, y2, tmp );

		tmp = _mm256_broadcast_pd( &g_cosSinCoefficients[ 2 ] );
		r1 = vectorMultiplyAdd( r1, y1, tmp );
		r2 = vectorMultiplyAdd( r2, y2, tmp );

		tmp = _mm256_broadcast_pd( &g_cosSinCoefficients[ 1 ] );
		r1 = vectorMultiplyAdd( r1, y1, tmp );
		r2 = vectorMultiplyAdd( r2, y2, tmp );

		tmp = _mm256_broadcast_pd( &g_cosSinCoefficients[ 0 ] );
		r1 = vectorMultiplyAdd( r1, y1, tmp );
		r2 = vectorMultiplyAdd( r2, y2, tmp );

		r1 = vectorMultiplyAdd( r1, y1, one );
		r2 = vectorMultiplyAdd( r2, y2, one );

		__m256d rs = _mm256_unpackhi_pd( r1, r2 );
		__m256d rc = _mm256_unpacklo_pd( r1, r2 );

		sin = _mm256_mul_pd( rs, x );
		cos = _mm256_or_pd( rc, sign );
	}

	__m256d vectorSin( __m256d x )
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

		const __m256d x2 = _mm256_mul_pd( x, x );

		const double* const coeffs = (const double*)g_cosSinCoefficients.data();
		// Compute polynomial approximation of sine
		__m256d vec = vectorMultiplyAdd( x2, broadcast( coeffs[ 9 ] ), broadcast( coeffs[ 7 ] ) );
		vec = vectorMultiplyAdd( vec, x2, broadcast( coeffs[ 5 ] ) );
		vec = vectorMultiplyAdd( vec, x2, broadcast( coeffs[ 3 ] ) );
		vec = vectorMultiplyAdd( vec, x2, broadcast( coeffs[ 1 ] ) );
		vec = vectorMultiplyAdd( vec, x2, one );
		vec = _mm256_mul_pd( vec, x );
		return vec;
	}

	__m256d vectorCos( __m256d x )
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

		// Compute polynomial approximation of cosine
		__m256d vec = vectorMultiplyAdd( x2, broadcast( coeffs[ 8 ] ), broadcast( coeffs[ 6 ] ) );
		vec = vectorMultiplyAdd( vec, x2, broadcast( coeffs[ 4 ] ) );
		vec = vectorMultiplyAdd( vec, x2, broadcast( coeffs[ 2 ] ) );
		vec = vectorMultiplyAdd( vec, x2, broadcast( coeffs[ 0 ] ) );
		vec = vectorMultiplyAdd( vec, x2, one );
		vec = _mm256_or_pd( vec, sign );
		return vec;
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
		__m128d vec = vectorMultiplyAdd( x2, coeffs[ 4 ], coeffs[ 3 ] );
		vec = vectorMultiplyAdd( vec, x2, coeffs[ 2 ] );
		vec = vectorMultiplyAdd( vec, x2, coeffs[ 1 ] );
		vec = vectorMultiplyAdd( vec, x2, coeffs[ 0 ] );
		vec = vectorMultiplyAdd( vec, x2, one );

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

	alignas( 16 ) static const struct
	{
		const double mul0 = -424539.12324285670928;  // -135135 * Pi
		const double div0 = -135135;                 // -135135
		const double mul2 = 537183.74348619438457;   // 17325 * Pi^3
		const double div2 = 615567.22649594329707;   // 62370 * Pi^2
		const double mul4 = -115675.44084883638934;  // -378 * Pi^5
		const double div4 = -306838.63675710767731;  // -3150 * Pi^4
		const double mul6 = 3020.2932277767920678;   // Pi^7
		const double div6 = 26918.897420108524239;   // 28 * Pi^6
		const double invPi = 1.0 / g_pi;
	}
	g_TanConstants;

	__m256d _AM_CALL_ vectorTan( __m256d a )
	{
		// Wrap into [ -pi/2 .. +pi/2 ] interval.
		// Don't multiply back, we include that multiplier into these Padé magic numbers.
		a = _mm256_mul_pd( a, broadcast( g_TanConstants.invPi ) );
		__m256d tmp = _mm256_round_pd( a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
		a = _mm256_sub_pd( a, tmp );

		const __m256d x2 = _mm256_mul_pd( a, a );

		__m256d mul = vectorMultiplyAdd( x2, broadcast( g_TanConstants.mul6 ), broadcast( g_TanConstants.mul4 ) );
		__m256d div = vectorMultiplyAdd( x2, broadcast( g_TanConstants.div6 ), broadcast( g_TanConstants.div4 ) );

		mul = vectorMultiplyAdd( mul, x2, broadcast( g_TanConstants.mul2 ) );
		div = vectorMultiplyAdd( div, x2, broadcast( g_TanConstants.div2 ) );

		mul = vectorMultiplyAdd( mul, x2, broadcast( g_TanConstants.mul0 ) );
		div = vectorMultiplyAdd( div, x2, broadcast( g_TanConstants.div0 ) );

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
		const __m128d x2 = _mm_mul_pd( av, av );

		__m128d acc = vectorMultiplyAdd( x2, _mm_loadu_pd( &g_TanConstants.mul6 ), _mm_loadu_pd( &g_TanConstants.mul4 ) );
		acc = vectorMultiplyAdd( acc, x2, _mm_loadu_pd( &g_TanConstants.mul2 ) );
		acc = vectorMultiplyAdd( acc, x2, _mm_loadu_pd( &g_TanConstants.mul0 ) );

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
}