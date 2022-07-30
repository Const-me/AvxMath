// Trigonometric functions
#pragma once

namespace AvxMath
{
	// Scale 4 angles from degrees to radians
	__m256d radians( __m256d deg );
	// Scale 2 angles from degrees to radians
	__m128d radians( __m128d deg );
	// Scale the angle from degrees to radians
	double radians( double deg );

	// Scale 4 angles from radians to degrees
	__m256d degrees( __m256d rad );
	// Scale 2 angles from radians to degrees
	__m128d degrees( __m128d rad );
	// Scale the angle from radians to degrees
	double degrees( double rad );

	// The sine/cosine are both using minimax polynomial approximations: 11-degree for sine, 10-degree for cosine.
	// Absolute errors compared to the standard library of VC++ for sine / cosine are within 5.7E-11 / 3.1E-10 when using FMA3, i.e. pretty accurate despite way faster.

	// Compute both sine and cosine of 4 angles in radian
	void _AM_CALL_ vectorSinCos( __m256d& sin, __m256d& cos, __m256d angles );
	// Compute sine of 4 angles
	__m256d vectorSin( __m256d angles );
	// Compute cosine of 4 angles
	__m256d vectorCos( __m256d angles );

	// Compute both sine and cosine of the angle, make a 2D vector with [ cos, sin ] values
	__m128d scalarSinCos( double a );

	// Compute sine of the angle
	double scalarSin( double a );
	// Compute cosine of the angle
	double scalarCos( double a );

	// Tangent and cotangent are using Padé approximation of degrees 7/6 for numerator/denominator

	// Compute tangents of 4 angles in radians
	__m256d _AM_CALL_ vectorTan( __m256d a );
	// Compute cotangents of 4 angles in radians
	__m256d _AM_CALL_ vectorCot( __m256d a );

	// Compute tangent of the angle
	double scalarTan( double a );
	// Compute cotangent of the angle
	double scalarCot( double a );
}