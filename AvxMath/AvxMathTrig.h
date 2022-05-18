﻿#pragma once

namespace AvxMath
{
	// The sine/cosine are both using minimax polynomial approximations: 11-degree for sine, 10-degree for cosine.
	// Absolute errors for sine/cosine, compared to standard library of VC++, are within 1.8E-08, i.e. reasonably accurate despite way faster.

	// Compute both sine and cosine of 4 angles in radian
	void _AM_CALL_ vectorSinCos( __m256d& sin, __m256d& cos, __m256d angles );

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