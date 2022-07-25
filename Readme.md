# AvxMath

The code in this repository implements small yet hopefully useful subset of [DirectXMath](https://github.com/Microsoft/DirectXMath),
ported from FP32 to FP64 precision.

The code requires [AVX1](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions),
and can optionally use AVX2 and FMA3 instructions when enabled.

The current version was tested with C++/17 language version.<br/>
It doesn’t use advanced language features though, should be trivial to downgrade if needed.

I have tested with Visual Studio 2022 on Windows 10, and GCC 7.4 on Microsoft Linux.<br/>
Should hopefully work on other OSes and compilers as well.

## Motivation

For real-time graphics applications, FP32 precision implemented in that Microsoft’s library is almost always good enough.

However, some other applications, like CAM/CAE, actually need the FP64 precision.<br/>
And sometimes FP64 can even be faster than FP32 because iterative algorithms like conjugate gradient need fewer iterations to converge.

The source code in AvxMath folder implements some low-level math routines for 3D vectors in FP64 precision, stored in `__m256d` hardware registers.

## Usage

Copy-paste the content of AvxMath folder into your project, add the `*.cpp` files from that folder to your build system.

Include the `AvxMath.h` header, and use the functions from `AvxMath` namespace.