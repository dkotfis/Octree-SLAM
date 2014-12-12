/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <nih/linalg/vector.h>
#include <cmath>

namespace nih {

/*! \addtogroup linalg Linear Algebra
 *  \{
 */

/*! \addtogroup matrices Matrices
 *  \ingroup linalg
 *  \{
 */

///
/// A dense N x M matrix class over a templated type T.
///
template <typename T, int N, int M> struct NIH_API_CS Matrix
{
public:
	typedef T value_type;
	typedef T Field_type;

    typedef Vector<T,M> row_vector;
    typedef Vector<T,N> column_vector;

public:
NIH_HOST NIH_DEVICE inline                      Matrix     ();
NIH_HOST NIH_DEVICE inline        explicit      Matrix     (const T s);
NIH_HOST NIH_DEVICE inline                      Matrix     (const Matrix<T,N,M>&);
NIH_HOST NIH_DEVICE inline                      Matrix     (const Vector<T,M> *v);
NIH_HOST NIH_DEVICE inline                      Matrix     (const T *v);
NIH_HOST NIH_DEVICE inline                      Matrix     (const T **v);
//inline                      Matrix     (const T v[N][M]);

NIH_HOST NIH_DEVICE inline        Matrix<T,N,M>&    operator  = (const Matrix<T,N,M>&);
NIH_HOST NIH_DEVICE inline        Matrix<T,N,M>&    operator += (const Matrix<T,N,M>&);
NIH_HOST NIH_DEVICE inline        Matrix<T,N,M>&    operator -= (const Matrix<T,N,M>&);
NIH_HOST NIH_DEVICE inline        Matrix<T,N,M>&    operator *= (T);
NIH_HOST NIH_DEVICE inline        Matrix<T,N,M>&    operator /= (T);

NIH_HOST NIH_DEVICE inline        const Vector<T,M>& operator [] (int) const;
NIH_HOST NIH_DEVICE inline        Vector<T,M>&       operator [] (int);
NIH_HOST NIH_DEVICE inline        const Vector<T,M>& get (int) const;
NIH_HOST NIH_DEVICE inline        void               set (int, const Vector<T,M>&);

NIH_HOST NIH_DEVICE inline        T           operator () (int i, int j) const;
NIH_HOST NIH_DEVICE inline        T&          operator () (int i, int j);

NIH_HOST NIH_DEVICE inline        T           det() const;

friend NIH_API_CS NIH_HOST NIH_DEVICE int            operator == <T,N,M> (const Matrix<T,N,M>&,  const Matrix<T,N,M>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE int            operator != <T,N,M> (const Matrix<T,N,M>&,  const Matrix<T,N,M>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M>  operator  - <T,N,M> (const Matrix<T,N,M>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M>  operator  + <T,N,M> (const Matrix<T,N,M>&,  const Matrix<T,N,M>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M>  operator  - <T,N,M> (const Matrix<T,N,M>&,  const Matrix<T,N,M>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M>  operator  * <T,N,M> (const Matrix<T,N,M>&,  T);
friend NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M>  operator  * <T,N,M> (T,                     const Matrix<T,N,M>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Vector<T,M>    operator  * <T,N,M> (const Vector<T,N>&,    const Matrix<T,N,M>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Vector<T,N>    operator  * <T,N>   (const Vector<T,N>&,    const Matrix<T,N,N>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Vector<T,N>    operator  * <T,N,M> (const Matrix<T,N,M>&,  const Vector<T,M>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Vector<T,N>    operator  * <T,N>   (const Matrix<T,N,N>&,  const Vector<T,N>&);
friend NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M>  operator  / <T,N,M> (const Matrix<T,N,M>&,  T);

public:
	Vector<T,M> r[N];
};

typedef Matrix<float,3,3>  Matrix3x3f;
typedef Matrix<double,3,3> Matrix3x3d;
typedef Matrix<float,4,4>  Matrix4x4f;
typedef Matrix<double,4,4> Matrix4x4d;

template <typename T, int N, int M, int Q> NIH_API_CS Matrix<T,N,Q>& multiply   (const Matrix<T,N,M>&,  const Matrix<T,M,Q>&,  Matrix<T,N,Q>&);
template <typename T, int N, int M, int Q> NIH_API_CS Matrix<T,N,Q>  operator * (const Matrix<T,N,M>&,  const Matrix<T,M,Q>&);
template <typename T, int N, int M> NIH_API_CS Vector<T,M>&     multiply    (const Vector<T,N>&,    const Matrix<T,N,M>&,  Vector<T,M>&);
template <typename T, int N, int M> NIH_API_CS Vector<T,N>&     multiply    (const Matrix<T,N,M>&,  const Vector<T,M>&,    Vector<T,N>&);
template <typename T, int N, int M> NIH_API_CS Matrix<T,M,N>    transpose   (const Matrix<T,N,M>&);
template <typename T, int N, int M> NIH_API_CS Matrix<T,M,N>&   transpose   (const Matrix<T,N,M>&,  Matrix<T,M,N>&);
template <typename T, int N, int M> NIH_API_CS bool             invert      (const Matrix<T,N,M>&,  Matrix<T,M,N>&); // gives inv(A^t * A)*A^t
template <typename T, int N, int M> NIH_API_CS T                det         (const Matrix<T,N,M>&);

/// build a 3d translation matrix
template <typename T>
NIH_HOST NIH_DEVICE Matrix<T,4,4> translate(const Vector<T,3>& vec);

/// build a 3d perspective matrix
template <typename T>
Matrix<T,4,4> perspective(T fovy, T aspect, T zNear, T zFar);

/// build a 3d look at matrix
template <typename T>
Matrix<T,4,4> look_at(const Vector<T,3>& eye, const Vector<T,3>& center, const Vector<T,3>& up);

/// build a 3d rotation around the X axis
template <typename T>
NIH_HOST NIH_DEVICE Matrix<T,4,4> rotation_around_X(const T q);

/// build a 3d rotation around the Y axis
template <typename T>
NIH_HOST NIH_DEVICE Matrix<T,4,4> rotation_around_Y(const T q);

/// build a 3d rotation around the Z axis
template <typename T>
NIH_HOST NIH_DEVICE Matrix<T,4,4> rotation_around_Z(const T q);

/// transform a 3d point with a perspective transform
NIH_HOST NIH_DEVICE inline Vector3f ptrans(const Matrix4x4f& m, const Vector3f& v);

/// transform a 3d vector with a perspective transform
NIH_HOST NIH_DEVICE inline Vector3f vtrans(const Matrix4x4f& m, const Vector3f& v);

/*! \}
 */

} // namespace nih

#include <nih/linalg/matrix_inline.h>
