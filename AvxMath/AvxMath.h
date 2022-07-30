// The main header of the AvxMath library, includes the rest of the headers.
#pragma once

#if !defined(_AM_AVX2_INTRINSICS_) && defined(__AVX2__)
#define _AM_AVX2_INTRINSICS_ 1
#endif

#if !defined(_AM_FMA3_INTRINSICS_) && defined(_AM_AVX2_INTRINSICS_)
#define _AM_FMA3_INTRINSICS_ 1
#endif

#ifdef _MSC_VER
#define _AM_CALL_  __vectorcall
#else
#define _AM_CALL_
#endif

#include <immintrin.h>
#include <stdint.h>
#include <assert.h>
#include <limits>

namespace AvxMath
{
	struct Matrix4x4
	{
		__m256d r0, r1, r2, r3;
	};
}

#include "AvxMathMisc.h"
#include "AvxMathMem.h"
#include "AvxMathTrig.h"
#include "AvxMathVector.h"
#include "AvxMathMatrix.h"
#include "AvxMathQuaternion.h"