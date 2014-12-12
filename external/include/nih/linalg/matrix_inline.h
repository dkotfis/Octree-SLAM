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

//
// I M P L E M E N T A T I O N
//

#ifdef _V
#undef _V
#endif
#define _V(v, i)   (((Vector<T,M>&)v).x[(i)])

#ifdef _M
#undef _M
#endif
#define _M(m, i, j)   (((Matrix<T,N,M>&)m).r[(i)].x[(j)])

#ifdef _CM
#undef _CM
#endif
#define _CM(m, i, j)   (((const Matrix<T,N,M>&)m).r[(i)].x[(j)])


//
// Matrix inline methods
//

template <typename T, int N, int M> Matrix<T,N,M>::Matrix() { }

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const T s)
{
	for (int i = 0; i < N; i++)
        r[i] = Vector<T,M>(s);
}

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const Matrix<T,N,M>& m)
{
	for (int i = 0; i < N; i++)
		r[i] = m.r[i];
}

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const Vector<T,M> *v)
{
	for (int i = 0; i < N; i++)
		r[i] = v[i];
}

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const T *v)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[i][j] = v[i*M + j];
}

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const T **v)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[i][j] = v[i][j];
}
/*
template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const T v[N][M])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[i][j] = v[i][j];
}
*/
template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator  = (const Matrix<T,N,M>& m)
{
	for (int i = 0; i < N; i++)
		r[i] = m.r[i];
	return *this;
}

template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator += (const Matrix<T,N,M>& m)
{
	for (int i = 0; i < N; i++)
		r[i] += m.r[i];
	return *this;
}

template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator -= (const Matrix<T,N,M>& m)
{
	for (int i = 0; i < N; i++)
		r[i] -= m.r[i];
	return *this;
}

template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator *= (T k)
{
	for (int i = 0; i < N; i++)
		r[i] *= k;
	return *this;
}

template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator /= (T k)
{
	for (int i = 0; i < N; i++)
		r[i] /= k;
	return *this;
}

template <typename T, int N, int M> const Vector<T,M>& Matrix<T,N,M>::operator [] (int i) const
{
	return r[i];
}
template <typename T, int N, int M> Vector<T,M>& Matrix<T,N,M>::operator [] (int i)
{
	return r[i];
}

template <typename T, int N, int M> const Vector<T,M>& Matrix<T,N,M>::get(int i) const
{
	return r[i];
}
template <typename T, int N, int M> void Matrix<T,N,M>::set(int i, const Vector<T,M>& v)
{
	r[i] = v;
}

template <typename T, int N, int M> T Matrix<T,N,M>::operator () (int i, int j) const
{
	return r[i][j];
}
template <typename T, int N, int M> T& Matrix<T,N,M>::operator () (int i, int j)
{
	return r[i][j];
}

template <typename T, int N, int M> T Matrix<T,N,M>::det() const
{
	return 0.0;
}

template <typename T, int N, int M, int Q> NIH_HOST NIH_DEVICE Matrix<T,N,Q>& multiply(const Matrix<T,N,M>& a, const Matrix<T,M,Q>& b, Matrix<T,N,Q>& r)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < Q; j++)
		{
			r[i][j] = 0.0;
			for (int k = 0; k < M; k++)
				r[i][j] += a[i][k]*b[k][j];
		}
	}
	return r;
}

// OPTIMAL
template <typename T, int N, int M> NIH_HOST NIH_DEVICE Vector<T,M>& multiply(const Vector<T,N>& v, const Matrix<T,N,M>& m, Vector<T,M>& r)
{
	for (int i = 0; i < M; i++)
	{
		r[i] = 0.0;
		for (int j = 0; j < N; j++)
			r[i] += v[j]*m(j,i);
	}
	return r;
}
// OPTIMAL
template <typename T, int N, int M> NIH_HOST NIH_DEVICE Vector<T,N>& multiply(const Matrix<T,N,M>& m, const Vector<T,M>& v, Vector<T,N>& r)
{
	for (int i = 0; i < N; i++)
	{
		r[i] = 0.0;
		for (int j = 0; j < M; j++)
			r[i] += m(i,j)*v[j];
	}
	return r;
}

template <typename T, int N, int M> NIH_HOST NIH_DEVICE Matrix<T,M,N> transpose(const Matrix<T,N,M>& m)
{
	Matrix<T,M,N> r;

	return transpose(m, r);
}

template <typename T, int N, int M> NIH_HOST NIH_DEVICE Matrix<T,M,N>& transpose(const Matrix<T,N,M>& m, Matrix<T,M,N>& r)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[j][i] = m[i][j];

	return r;
}

template <typename T, int N, int M> bool invert(const Matrix<T,N,M>& a, Matrix<T,M,N>& r)
{
    T d = a.det();
	if (__TZero__(d)) return false;

	return true;
}

template <typename T>
bool invert(const Matrix<T,4,4>& a, Matrix<T,4,4>& r)
{
	T t1, t2, t3, t4, t5, t6;

	// We calculate the adjoint matrix of a, then we divide it by the determinant of a
	// We indicate with Aij the cofactor of a[i][j]:   Aij = (-1)^(i+j)*Det(Tij), where
	// Tij is the minor complementary of a[i][j]

	// First block ([0,0] - [3,1])

	t1 = a(2,2)*a(3,3) - a(2,3)*a(3,2);
	t2 = a(2,1)*a(3,3) - a(2,3)*a(3,1);
	t3 = a(2,1)*a(3,2) - a(2,2)*a(3,1);
	t4 = a(2,0)*a(3,3) - a(2,3)*a(3,0);
	t5 = a(2,0)*a(3,2) - a(2,2)*a(3,0);
	t6 = a(2,0)*a(3,1) - a(2,1)*a(3,0);

	r[0][0] =   a(1,1) * t1 - a(1,2) * t2 + a(1,3) * t3;      // A00
	r[0][1] = -(a(0,1) * t1 - a(0,2) * t2 + a(0,3) * t3);     // A10
	r[1][0] = -(a(1,0) * t1 - a(1,2) * t4 + a(1,3) * t5);     // A01
	r[1][1] =   a(0,0) * t1 - a(0,2) * t4 + a(0,3) * t5;      // A11
	r[2][0] =   a(1,0) * t2 - a(1,1) * t4 + a(1,3) * t6;      // A02
	r[2][1] = -(a(0,0) * t2 - a(0,1) * t4 + a(0,3) * t6);     // A21
	r[3][0] = -(a(1,0) * t3 - a(1,1) * t5 + a(1,2) * t6);     // A03
	r[3][1] =   a(0,0) * t3 - a(0,1) * t5 + a(0,2) * t6;      // A13

	// Second block ([0,2] - [3,2])

	t1 = a(1,2)*a(3,3) - a(1,3)*a(3,2);
	t2 = a(1,1)*a(3,3) - a(1,3)*a(3,1);
	t3 = a(1,1)*a(3,2) - a(1,2)*a(3,1);
	t4 = a(1,0)*a(3,3) - a(1,3)*a(3,0);
	t5 = a(1,0)*a(3,2) - a(1,2)*a(3,0);
	t6 = a(1,0)*a(3,1) - a(1,1)*a(3,0);

	r[0][2] =   a(0,1) * t1 - a(0,2) * t2 + a(0,3) * t3;      // A20
	r[1][2] = -(a(0,0) * t1 - a(0,2) * t4 + a(0,3) * t5);     // A21
	r[2][2] =   a(0,0) * t2 - a(0,1) * t4 + a(0,3) * t6;      // A22
	r[3][2] = -(a(0,0) * t3 - a(0,1) * t5 + a(0,2) * t6);     // A23

	// Third block ([0,3] - [3,3])

	t1 = a(1,2)*a(2,3) - a(1,3)*a(2,2);
	t2 = a(1,1)*a(2,3) - a(1,3)*a(2,1);
	t3 = a(1,1)*a(2,2) - a(1,2)*a(2,1);
	t4 = a(1,0)*a(2,3) - a(1,3)*a(2,0);
	t5 = a(1,0)*a(2,2) - a(1,2)*a(2,0);
	t6 = a(1,0)*a(2,1) - a(1,1)*a(2,0);

	r[0][3] = -(a(0,1) * t1 - a(0,2) * t2 + a(0,3) * t3);     // A30
	r[1][3] =   a(0,0) * t1 - a(0,2) * t4 + a(0,3) * t5;      // A31
	r[2][3] = -(a(0,0) * t2 - a(0,1) * t4 + a(0,3) * t6);     // A32
	r[3][3] =   a(0,0) * t3 - a(0,1) * t5 + a(0,2) * t6;      // A33

	// We save some time calculating Det(a) this way (now r is adjoint of a)
	// Det(a) = a00 * A00 + a01 * A01 + a02 * A02 + a03 * A03
	T d = a(0,0)*r[0][0] + a(0,1)*r[1][0] + a(0,2)*r[2][0] + a(0,3)*r[3][0];

	if (d == T(0.0)) return false; // Singular matrix => no inverse

	d = T(1.0)/d;

	r *= d;
	return true;
}

template <typename T, int N, int M> T det(const Matrix<T,N,M>& m)
{
	return m.det();
}

//
// Matrix<T,N,M> template <typename T, int N, int M> functions (not members)
//

template <typename T, int N, int M> NIH_API_CS NIH_HOST NIH_DEVICE int operator == (const Matrix<T,N,M>& a, const Matrix<T,N,M>& b)
{
	for (int i = 0; i < N; i++)
	{
		if (a[i] != b[i])
			return 0;
	}
	return 1;
}

template <typename T, int N, int M> NIH_API_CS NIH_HOST NIH_DEVICE int operator != (const Matrix<T,N,M>& a, const Matrix<T,N,M>& b)
{
	return !(a == b);
}

template <typename T, int N, int M> NIH_API_CS Matrix<T,N,M> NIH_HOST NIH_DEVICE operator  - (const Matrix<T,N,M>& a)
{
	return (Matrix<T,N,M>(a) *= -1.0);
}

template <typename T, int N, int M> NIH_API_CS Matrix<T,N,M> NIH_HOST NIH_DEVICE operator  + (const Matrix<T,N,M>& a, const Matrix<T,N,M>& b)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			_M(r,i,j) = _CM(a,i,j) + _CM(b,i,j);

	return r;
}

template <typename T, int N, int M> NIH_API_CS Matrix<T,N,M> NIH_HOST NIH_DEVICE operator  - (const Matrix<T,N,M>& a, const Matrix<T,N,M>& b)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			_M(r,i,j) = _CM(a,i,j) - _CM(b,i,j);

	return r;
}

template <typename T, int N, int M, int Q> NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,Q> operator * (const Matrix<T,N,M>& a, const Matrix<T,M,Q>& b)
{
	Matrix<T,N,Q> r;

	return multiply(a, b, r);
}

template <typename T, int N, int M> NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M> operator  * (const Matrix<T,N,M>& a, T k)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			_M(r,i,j) = _CM(a,i,j) * k;

	return r;
}

template <typename T, int N, int M> NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M> operator  * (T k, const Matrix<T,N,M>& a)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			_M(r,i,j) = _CM(a,i,j) * k;

	return r;
}

template <typename T, int N, int M> NIH_API_CS NIH_HOST NIH_DEVICE Vector<T,M> operator * (const Vector<T,N>& v, const Matrix<T,N,M>& m)
{
	Vector<T,M> r;

	return multiply(v, m, r);
}
template <typename T, int N, int M> NIH_API_CS NIH_HOST NIH_DEVICE Vector<T,N> operator * (const Matrix<T,N,M>& m, const Vector<T,M>& v)
{
	Vector<T,N> r;

    return multiply(m, v, r);
}

template <typename T, int N, int M> NIH_API_CS NIH_HOST NIH_DEVICE Matrix<T,N,M> operator  / (const Matrix<T,N,M>& a, T k)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			_M(r,i,j) = _CM(a,i,j) / k;

	return r;
}

template <typename T>
NIH_HOST NIH_DEVICE Matrix<T,4,4> translate(const Vector<T,3>& vec)
{
    Matrix<T,4,4> m( T(0) );

    m(0,0) = m(1,1) = m(2,2) = m(3,3) = 1.0f;

    m(0,3) = vec.x;
    m(1,3) = vec.y;
    m(2,3) = vec.z;

    return m;
}

template <typename T>
Matrix<T,4,4> perspective(T fovy, T aspect, T zNear, T zFar)
{
    Matrix<T,4,4> m( T(0) );

    T f = T(1) / std::tan(fovy / T(2));

    m(0,0) = f / aspect;
    m(1,1) = f;
    m(2,2) = (zFar + zNear) / (zNear - zFar);
    m(2,3) = T(2) * zFar * zNear / (zNear - zFar);
    m(3,2) = T(-1);
    m(3,3) = T(0);

    return m;
}
template <typename T>
Matrix<T,4,4> look_at(const Vector<T,3>& eye, const Vector<T,3>& center, const Vector<T,3>& up)
{
    Vector<T,3> f = normalize(center - eye);
    Vector<T,3> s = normalize(cross(f, up));
    Vector<T,3> u = cross(s, f);

    Matrix<T,4,4> m(T(0));
    m(0,0) = s.x;
    m(0,1) = s.y;
    m(0,2) = s.z;
    m(1,0) = u.x;
    m(1,1) = u.y;
    m(1,2) = u.z;
    m(2,0) = -f.x;
    m(2,1) = -f.y;
    m(2,2) = -f.z;

    return m * translate(-eye);
}

template <typename T>
NIH_HOST NIH_DEVICE Matrix<T,4,4> rotation_around_X(const T q)
{
    Matrix<T,4,4> m;

	const float sin_q = sin(q);
	m[1][1] = m[2][2] = cos(q);
	m[1][2] = -sin_q;
	m[2][1] =  sin_q;
	m[0][0] = m[3][3] = T(1.0f);
	m[0][1] =
	m[0][2] =
	m[0][3] =
	m[1][0] =
	m[1][3] =
	m[2][0] =
	m[2][3] =
	m[3][0] =
	m[3][1] =
	m[3][2] = T(0.0f);
	return m;
}
template <typename T>
NIH_HOST NIH_DEVICE Matrix<T,4,4> rotation_around_Y(const T q)
{
    Matrix<T,4,4> m;

    const float sin_q = sin(q);
	m[0][0] = m[2][2] = cos(q);
	m[2][0] = -sin_q;
	m[0][2] =  sin_q;
	m[1][1] = m[3][3] = T(1.0f);
	m[0][1] =
	m[0][3] =
	m[1][0] =
	m[1][2] =
	m[1][3] =
	m[2][1] =
	m[2][3] =
	m[3][0] =
	m[3][1] =
	m[3][2] = T(0.0f);
	return m;
}
template <typename T>
NIH_HOST NIH_DEVICE Matrix<T,4,4> rotation_around_Z(const T q)
{
    Matrix<T,4,4> m;

    const float sin_q = sin(q);
	m[0][0] = m[1][1] = cos(q);
	m[1][0] =  sin_q;
	m[0][1] = -sin_q;
	m[2][2] = m[3][3] = T(1.0f);
	m[0][2] =
	m[0][3] =
	m[1][2] =
	m[1][3] =
	m[2][0] =
	m[2][1] =
	m[2][3] =
	m[3][0] =
	m[3][1] =
	m[3][2] = T(0.0f);
	return m;
}
NIH_HOST NIH_DEVICE inline Vector3f ptrans(const Matrix4x4f& m, const Vector3f& v)
{
	const Vector4f r = m * Vector4f(v,1.0f);
	return Vector3f( r[0], r[1], r[2] );
}
NIH_HOST NIH_DEVICE inline Vector3f vtrans(const Matrix4x4f& m, const Vector3f& v)
{
	const Vector4f r = m * Vector4f(v,0.0f);
	return Vector3f( r[0], r[1], r[2] );
}

#undef _V
#undef _M
#undef _CM

} // namespace nih
