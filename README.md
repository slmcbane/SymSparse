# SymSparse
This repository provides a single file, `SymSparse.hpp`, implementing a class
representing a symmetric, sparse matrix. This data structure is optimized for
finite element assembly. It always stores only the upper triangle of a matrix,
and otherwise implements only the ability to multiply it by a vector or to
eliminate a degree of freedom from the matrix (and the corresponding right
hand side). That is, given a system `A*x = b`, and then fixing `x_i = c`,
modify `A` and `b` to represent the new system.

The matrix class implements the matrix-vector product with the intent of
supporting direct use in a matrix-free solver, but I imagine it primarily being
an intermediary data structure to perform assembly before conversion to the
format expected by a sophisticated solver library.

## Dependency
This code depends on a utility data structure to hold the representation of
each row; the file `SmallVector.hpp` is available
[here](https://github.com/slmcbane/SmallVector) and is licensed under the same
copyright terms as this repository. Most recently, `SymSparse` was tested against `SmallVector` v1.0.0.

## Author
Sean McBane (<sean.mcbane@protonmail.com>)

## Version
This is v1.0.0.

## Copyright
Copyright 2020 The University of Texas at Austin.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
