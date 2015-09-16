#pragma once


/** **************************************************************************************
* ldp_basic_mat
* @author Dongping Li
* Column Majored Matrix Class
* This class is suitable for small matrices, E. G., size 2, 3, 4, 5, 6...
* | 1 5 9  |
* | 2 6 10 |
* | 3 7 11 |
* | 4 8 12 |
* The above is an example of ldp_basic_mat<T, 4, 3>.
* For matrices of size 2, 3 and 4, ldp_basic_mat2/3/4, child classes of ldp_basic_mat, can be used.
* This class is designed firstly for convinience and then for efficiency.
* ****************************************************************************************/

#include "ldpdef.h"
#include "ldp_basic_vec.h"

namespace ldp
{

/** **************************************************************************************
* pre-declares
* ***************************************************************************************/

//for eigenvalues and eigenvectors computation
template<typename _Tp> bool
JacobiImpl_( _Tp* A, size_t astep, _Tp* W, _Tp* V, size_t vstep, int n, int* buf );

/** **************************************************************************************
* classes
* ***************************************************************************************/
template<class T, size_t N, size_t M>
class ldp_basic_mat
{
public:
	const static size_t NUM_ELEMENTS = N*M;
protected:
	T _data[NUM_ELEMENTS];
public:
	/**
	* Constructors
	* */
	ldp_basic_mat(){memset(_data, 0, NUM_ELEMENTS*sizeof(T));}
	ldp_basic_mat(const T* data){copy(data);}
	ldp_basic_mat(const T& v)
	{
		for(size_t i=0; i<NUM_ELEMENTS; i++)
				(*this)[i] = v;
	}
	template<class E> ldp_basic_mat(const ldp_basic_mat<E,N,M>& rhs)
	{
		for(size_t i=0; i<NUM_ELEMENTS; i++)
				(*this)[i] = rhs[i];
	}
	template<class E1, class E2, size_t K> ldp_basic_mat(const ldp_basic_mat<E1,K,M>& rhs1, const ldp_basic_mat<E2,N-K,M>& rhs2)
	{
		for(size_t i=0; i<K; i++)
			for(size_t j=0; j<M; j++)
				(*this)(i, j) = rhs1(i,j);
		for(size_t i=K; i<N; i++)
			for(size_t j=0; j<M; j++)
				(*this)(i, j) = rhs2(i-K, j);
	}
	template<class E1, class E2, size_t K> ldp_basic_mat(const ldp_basic_mat<E1,N,K>& rhs1, const ldp_basic_mat<E2,N,M-K>& rhs2)
	{
		for(size_t i=0; i<N; i++)
		{
			for(size_t j=0; j<K; j++)
				(*this)(i, j) = rhs1(i,j);
			for(size_t j=K; j<M; j++)
				(*this)(i, j) = rhs2(i, j-K);
		}
	}

	/**
	* Data Access Methods
	* */
	Size2 size()const {return Size2(N,M);}
	size_t nRow()const{return N;}
	size_t nCol()const{return M;}
	const T* ptr()const {return _data;}
	T* ptr() {return _data;}
	const T& operator() (size_t i, size_t j)const {return _data[j*N+i];}
	T& operator() (size_t i, size_t j) {return _data[j*N+i];}
	const T& operator[] (size_t i)const {return _data[i];}
	T& operator[] (size_t i) {return _data[i];}

	/**
	* Utilities
	* */
	ldp_basic_mat<T, M, N> trans()const
	{
		ldp_basic_mat<T, M, N> out;
		for(size_t i=0; i<N; i++)
			for(size_t j=0; j<M; j++)
				out(i, j) = (*this)(j, i);
		return out;
	}
	ldp_basic_mat<T, N, M>& zeros()
	{
		memset(_data, 0, NUM_ELEMENTS*sizeof(T));
		return *this;
	}
	ldp_basic_mat<T, N, M>& ones()
	{
		for(size_t i=0; i<NUM_ELEMENTS; i++)
			(*this)[i] = 1;
		return *this;
	}
	ldp_basic_mat<T, N, M>& copy(const T* data)
	{
		memcpy(_data, data, NUM_ELEMENTS*sizeof(T));
		return *this;
	}

	// note: this is the F-norm
	T norm()const
	{
		T sum = 0;
		for (int y = 0; y < N; y++)
		for (int x = 0; x < N; x++)
			sum += (*this)(y, x) * (*this)(y, x);
		return sqrt(sum);
	}

	/**
	* Operators: +,- with mat; +,-,*,/ with scalar
	* */
#define LDP_BASIC_MAT_ARITHMATIC(OP)																					\
	template<typename E>																								\
	typename ldp_basic_mat<typename type_promote<T,E>::type, N, M> operator OP (const ldp_basic_mat<E,N,M>& rhs)const	\
	{																													\
		typename ldp_basic_mat<typename type_promote<T,E>::type, N, M> out;												\
		for(size_t i=0; i<NUM_ELEMENTS; i++)																			\
			out[i] = (*this)[i] OP rhs[i];																				\
		return out;																										\
	}																							
#define LDP_BASIC_MAT_ARITHMATIC_SCALAR(OP)														\
	ldp_basic_mat<T,N,M> operator OP (const T& rhs)const										\
	{																							\
		ldp_basic_mat<T,N,M> out;																\
		for(size_t i=0; i<NUM_ELEMENTS; i++)													\
			out[i] = (*this)[i] OP rhs;															\
		return out;																				\
	}																							\
	//friend ldp_basic_mat<T,N,M> operator OP (const T& lhs, const ldp_basic_mat<T,N,M>& rhs);
	/**
	* Operators: +=, -=, = with mat; +=, -=, *=, /=, = with scalar
	* */
#define LDP_BASIC_MAT_ARITHMATIC2(OP)															\
	template<class E>																			\
	ldp_basic_mat<T, N, M>& operator OP (const ldp_basic_mat<E,N,M>& rhs)						\
	{																							\
		for(size_t i=0; i<NUM_ELEMENTS; i++)													\
			(*this)[i] OP static_cast<T>(rhs[i]);												\
		return *this;																			\
	}																							
#define LDP_BASIC_MAT_ARITHMATIC2_SCALAR(OP)													\
	ldp_basic_mat<T, N, M>& operator OP (const T& rhs)											\
	{																							\
		for(size_t i=0; i<NUM_ELEMENTS; i++)													\
			(*this)[i] OP rhs;																	\
		return *this;																			\
	}	

	LDP_BASIC_MAT_ARITHMATIC(+)
	LDP_BASIC_MAT_ARITHMATIC(-)
	LDP_BASIC_MAT_ARITHMATIC_SCALAR(+)
	LDP_BASIC_MAT_ARITHMATIC_SCALAR(-)
	LDP_BASIC_MAT_ARITHMATIC_SCALAR(*)
	LDP_BASIC_MAT_ARITHMATIC_SCALAR(/)
	LDP_BASIC_MAT_ARITHMATIC2(+=)
	LDP_BASIC_MAT_ARITHMATIC2(-=)
	LDP_BASIC_MAT_ARITHMATIC2_SCALAR(+=)
	LDP_BASIC_MAT_ARITHMATIC2_SCALAR(-=)
	LDP_BASIC_MAT_ARITHMATIC2_SCALAR(*=)
	LDP_BASIC_MAT_ARITHMATIC2_SCALAR(/=)
	LDP_BASIC_MAT_ARITHMATIC2_SCALAR(=)

	/**
	* Arithmatic Operators: +,-,*,/ scale with mat
	* */
	#define LDP_BASIC_MAT_ARITHMATIC_FRIEND(OP)																		\
		friend ldp_basic_mat<T,N,M> operator OP (const T& lhs, const ldp_basic_mat<T,N,M>& rhs)						\
		{																											\
			ldp::ldp_basic_mat<T, N, M> out;																		\
			for(size_t i=0; i<NUM_ELEMENTS; i++)																	\
				out[i] = lhs OP rhs[i];																				\
			return out;																								\
		}

		LDP_BASIC_MAT_ARITHMATIC_FRIEND(+)
		LDP_BASIC_MAT_ARITHMATIC_FRIEND(-)
		LDP_BASIC_MAT_ARITHMATIC_FRIEND(*)
		LDP_BASIC_MAT_ARITHMATIC_FRIEND(/)
	

	#undef LDP_BASIC_MAT_ARITHMATIC
	#undef LDP_BASIC_MAT_ARITHMATIC_SCALAR
	#undef LDP_BASIC_MAT_ARITHMATIC2
	#undef LDP_BASIC_MAT_ARITHMATIC2_SCALAR
	#undef LDP_BASIC_MAT_ARITHMATIC_FRIEND
	

	/**
	* Point-Wise mult and divide
	* */
	template<typename E>																								
	typename ldp_basic_mat<typename type_promote<T,E>::type, N, M> pmul (const ldp_basic_mat<E,N,M>& rhs)const	
	{																													
		typename ldp_basic_mat<typename type_promote<T,E>::type, N, M> out;												
		for(size_t i=0; i<NUM_ELEMENTS; i++)																			
			out[i] = (*this)[i] * rhs[i];																				
		return out;																										
	}	
	template<typename E>																								
	typename ldp_basic_mat<typename type_promote<T,E>::type, N, M> pdiv (const ldp_basic_mat<E,N,M>& rhs)const	
	{																													
		typename ldp_basic_mat<typename type_promote<T,E>::type, N, M> out;												
		for(size_t i=0; i<NUM_ELEMENTS; i++)																			
			out[i] = (*this)[i] / rhs[i];																				
		return out;																										
	}	

	/**
	* Operator: * with mat and vec
	* */
	template<typename E, size_t K>																								
	typename ldp_basic_mat<typename type_promote<T,E>::type, N, K> operator * (const ldp_basic_mat<E,M,K>& rhs)const	
	{																													
		typename ldp_basic_mat<typename type_promote<T,E>::type, N, K> out;												
		for(size_t i=0; i<N; i++)
		{
			for(size_t k=0; k<K; k++)
			{
				typename type_promote<T,E>::type s = 0;
				for(size_t j=0; j<M; j++)
					s += (*this)(i,j) * rhs(j,k);
				out(i,k) = s;
			}
		}
		return out;																										
	}
	template<typename E>																								
	typename ldp_basic_vec<typename type_promote<T,E>::type, N> operator * (const ldp_basic_vec<E,M>& rhs)const	
	{																													
		typename ldp_basic_vec<typename type_promote<T,E>::type, N> out;												
		for(size_t i=0; i<N; i++)
		{
			typename type_promote<T,E>::type s = 0;
			for(size_t j=0; j<M; j++)
				s += (*this)(i,j) * rhs[j];
			out[i] = s;
		}
		return out;																										
	}
};//ldp_basic_mat

/**
* I/O
* */
template<typename T, size_t N, size_t M>
inline std::ostream& operator<<(std::ostream& out, const ldp_basic_mat<T,N,M>& v) 
{
	for(size_t i=0; i<N; i++)
	{
		for(size_t j=0; j<M; j++)
			out << std::left << std::setw(10) << v(i, j) << " ";
		out << std::endl;
	}
    return out;
}

template<typename T, size_t N, size_t M>
inline std::istream& operator>>(std::istream& in, const ldp_basic_mat<T,N,M>& v) 
{
	for(size_t i=0; i<N; i++)
		for(size_t j=0; j<M; j++)
			in >> v(i, j);
    return in;
}

/** *************************************************************************************
* Square Matrix
* **************************************************************************************/
template<class T, size_t N>
class ldp_basic_mat_sqr : public ldp_basic_mat<T, N, N>
{
public:
	/**
	* Constructors
	* */
	ldp_basic_mat_sqr():ldp_basic_mat<T,N,N>(){}
	ldp_basic_mat_sqr(const T* data):ldp_basic_mat<T,N,N>(data){}
	ldp_basic_mat_sqr(const T& v):ldp_basic_mat<T,N,N>(v){}
	template<class E> ldp_basic_mat_sqr(const ldp_basic_mat<E,N,N>& rhs):ldp_basic_mat<T,N,N>(rhs){}
	template<class E1, class E2, size_t K> ldp_basic_mat_sqr(const ldp_basic_mat<E1,K,N>& rhs1, const ldp_basic_mat<E2,N-K,N>& rhs2)
	:ldp_basic_mat<T,N,N>(rhs1, rhs2){}
	template<class E1, class E2, size_t K> ldp_basic_mat_sqr(const ldp_basic_mat<E1,N,K>& rhs1, const ldp_basic_mat<E2,N,N-K>& rhs2)
	:ldp_basic_mat<T,N,N>(rhs1, rhs2){}

	/**
	* Utilities
	* */
	ldp_basic_mat_sqr<T, N>& eye()
	{
		zeros();
		for(size_t i=0; i<N; i++)
			(*this)(i,i) = 1;
		return *this;
	}

	T trace()const
	{
		T s = T(0);
		for(size_t i=0; i<N; i++)
			s += (*this)(i,i);
		return s;
	}

	ldp_basic_vec<T, N> diag()const
	{
		ldp_basic_vec<T, N> x;
		for(size_t i=0; i<N; i++)
			x[i] = (*this)(i,i);
		return x;
	}

	ldp_basic_mat_sqr<T, N>& fromDiag(const ldp_basic_vec<T,N>& x)
	{
		zeros();
		for(size_t i=0; i<N; i++)
			(*this)(i,i) = x[i];
		return *this;
	}

	/**
	* LU Decomposition for small matrices with Row permutation
	* A may be (*this)
	* Return:	0 if pivot detected
	*			1 if matrix is positive definite
	*			-1 if matrix is negative definite
	* */
	int lu(ldp_basic_mat<T,N,N>& A, ldp_basic_vec<int,N>& permute)const
	{
		int i, j, k, p = 1;  
		A = (*this);
		for( i = 0; i < N; i++)	permute[i] = i;
		for( i = 0; i < N; i++ )
		{   
			k = i;
			for( j = i+1; j < N; j++ )
				if( std::abs(A(j,i)) > std::abs(A(k,i)) )
					k = j;
			std::swap(permute[i], permute[k]);
			if( std::abs(A(k,i)) < std::numeric_limits<T>::epsilon() )
				return 0;
			if( k != i )
			{
				for( j = 0; j < N; j++ )
					std::swap(A(i,j), A(k,j));
				p = -p;
			}
        
			T d = -1/A(i,i);
        
			for( j = i+1; j < N; j++ )
			{
				T alpha = A(j,i)*d;
				for( k = i+1; k < N; k++ )
					A(j,k) += alpha*A(i,k);
				A(j, i) = -alpha;
			}
		}
		return p;
	}
	/**
	* LU Decomposition for small matrices with Row permutation
	* L * U = (*this) and L & U must not be (*this)
	* Return:	0 if pivot detected
	*			1 if matrix is positive definite
	*			-1 if matrix is negative definite
	* */
	int lu(ldp_basic_mat<T,N,N>& L, ldp_basic_mat<T,N,N>& U) const
	{
		ldp_basic_mat_sqr<T,N> A;
		ldp_basic_vec<int, N> p;
		int r = lu(A, p);
		L = T(0);
		U = T(0);
		for(int x=0; x<N;x++)
		{
			for(int y=0; y<N; y++)
			{
				if(y==x) L(p[y],x) = 1, U(y,x) = A(y,x);
				else if(y>x) L(p[y],x) = A(y,x);
				else U(y, x) = A(y,x);
			}
		}
		return r;
	}
	int lu(ldp_basic_mat<T,N,N>& L, ldp_basic_mat<T,N,N>& U, ldp_basic_vec<int, N>& permute) const
	{
		ldp_basic_mat_sqr<T,N> A;
		int r = lu(A, permute);
		L = T(0);
		U = T(0);
		for(int x=0; x<N;x++)
		{
			for(int y=0; y<N; y++)
			{
				if(y==x) L(y,x) = 1, U(y,x) = A(y,x);
				else if(y>x) L(y,x) = A(y,x);
				else U(y, x) = A(y,x);
			}
		}
		return r;
	}

	/**
	* back substitution given lu factorized matrix
	* (*this) = p[L U], L*U*x = p(b)
	* */
	void luBackSubst(const ldp_basic_vec<int,N>& p, ldp_basic_vec<T,N> b, ldp_basic_vec<T,N>& x)const
	{
		ldp_basic_vec<T, N> ux;
		//backup substitution for L: L*ux = P(b)
		for(size_t i=0; i<N; ++i)
		{
			ux[i] = b[p[i]];
			for(size_t j= i+1; j<N; ++j)
			{
				b[p[j]] -= b[p[i]] * (*this)(j,i);
			}//end for j
		}//end for i

		//backup subsitution for U: U*x = ux
		for(int i=int(N-1); i>=0; --i)
		{
			const T invd = 1/(*this)(i,i);
			x[i] = ux[i] * invd;
			if(i == 0) break;
			for(int j=i-1; j>=0; --j)
			{
				ux[j] -= ux[i] * (*this)(j,i) * invd;
			}//end for j
		}//end for i
	}

	/**
	* A*x = b, solve x = A\b
	* */
	ldp_basic_vec<T,N> solve(const ldp_basic_vec<T,N>& b)const
	{
		ldp_basic_mat_sqr<T, N> A = (*this);
		ldp_basic_vec<T, N> x;
		ldp_basic_vec<int, N> p;
		int r = lu(A, p);

		if(r == 0)
		{
			x = std::numeric_limits<T>::infinity();
			return x;
		}

		A.luBackSubst(p, b, x);

		return x;
	}
	

	/**
	* Matrix Inverse
	* */
	void inv(ldp_basic_mat<T,N,N>& result)const
	{
		ldp_basic_vec<T,N> x,b;
		ldp_basic_vec<int,N> p;
		ldp_basic_mat_sqr<T,N> A;
		int r = lu(A, p);
		if(r == 0)
		{
			result = std::numeric_limits<T>::infinity();
			return;
		}
		for(size_t i=0; i<N; i++)
		{
			b = 0;
			b[i] = 1;
			A.luBackSubst(p, b, x);
			for(size_t j=0; j<N; j++)
				result(j,i) = x[j];
		}
	}
	ldp_basic_mat<T, N, N> inv()const
	{
		ldp_basic_mat<T,N,N> r;
		inv(r);
		return r;
	}

	/**
	* Matrix Determinant
	* */
	T det()const
	{
		ldp_basic_mat_sqr<T, N> A = (*this);
		ldp_basic_vec<int, N> p;
		int r = A.lu(A, p);
		if(r==0) return 0;
		T s = 1;
		for(size_t i=0; i<N; i++)
			s *= A(i,i);
		return r*s;
	}

	/**
	* EigenValues and EigenVectors of symmetric matrix
	* */
	void eig(ldp_basic_vec<T,N>& eigVals, ldp_basic_mat<T,N,N>& eigVecs)const
	{
		int buf[N*N + N*5 + 32];
		ldp_basic_mat_sqr<T,N> A = *this;
		JacobiImpl_(A.ptr(), N, eigVals.ptr(), eigVecs.ptr(), N, N, buf);
	}
	void eig(ldp_basic_vec<T,N>& eigVals)const
	{
		int buf[N*N + N*5 + 32];
		ldp_basic_mat_sqr<T,N> A = *this;
		JacobiImpl_(A.ptr(), N, eigVals.ptr(), (T*)0, N, N, buf);
	}
	ldp_basic_vec<T,N> eig()const
	{
		ldp_basic_vec<T,N> x;
		eig(x);
		return x;
	}

	/**
	* Operators: *=
	* */
	template<typename E>																								
	ldp_basic_mat_sqr<T,N>& operator *= (const ldp_basic_mat<E,N,N>& rhs)	
	{																													
		return (*this) = ((*this)*rhs);																									
	}

	//Newton iteration method for polor decomposition.
	//argmin{||self-R'*R||} subject to R'*R=I. and self = R*S.
	inline void polorDecomposition(ldp_basic_mat_sqr<T,N>& R, ldp_basic_mat_sqr<T,N>& S, 
		const T& tol = T(1e-5))
	{
		R = *this;
		ldp_basic_mat_sqr<T, N> R1, Y;
		while ((R1 - R).norm()> tol)
		{
			//R1 = R;
			//Y = R.inverse();
			//double a = sqrt(R.norm(1) * R.norm(-1));
			//double b = sqrt(Y.norm(1) * Y.norm(-1));
			//double gama = sqrt(b/a);
			//R = (gama*R + 1.0/gama * Y.transpose())*0.5;
			R1 = R;
			R = 0.5*(R + R.inv().trans());
		}
		S = R.inv() * (*this);
	}
};

/** *************************************************************************************
* Some Specifications: ldp_basic_mat2/3/4
* Optimization: handly unrolling
* **************************************************************************************/
template<class T>
class ldp_basic_mat2 : public ldp_basic_mat_sqr<T, 2>
{
public:
	/**
	* Constructors
	* */
	ldp_basic_mat2():ldp_basic_mat_sqr<T,2>(){}
	ldp_basic_mat2(const T* data):ldp_basic_mat_sqr<T,2>(data){}
	ldp_basic_mat2(const T& v):ldp_basic_mat_sqr<T,2,2>(v){}
	template<class E> ldp_basic_mat2(const ldp_basic_mat<E,2,2>& rhs):ldp_basic_mat_sqr<T,2>(rhs){}
	template<class E1, class E2, size_t K> ldp_basic_mat2(const ldp_basic_mat<E1,K,2>& rhs1, const ldp_basic_mat<E2,2-K,2>& rhs2)
	:ldp_basic_mat_sqr<T,2>(rhs1, rhs2){}
	template<class E1, class E2, size_t K> ldp_basic_mat2(const ldp_basic_mat<E1,2,K>& rhs1, const ldp_basic_mat<E2,2,2-K>& rhs2)
	:ldp_basic_mat_sqr<T,2>(rhs1, rhs2){}

	/**
	* Operator: * with mat and vec
	* */
	template<typename E, size_t K>																								
	typename ldp_basic_mat<typename type_promote<T,E>::type, 2, K> operator * (const ldp_basic_mat<E,2,K>& rhs)const	
	{																													
		typename ldp_basic_mat<typename type_promote<T,E>::type, 2, K> out;	
		for(size_t i=0; i<K; i++)
		{
			out(i, 0) = _data[0]*rhs(i,0) + _data[2]*rhs(i,1);
			out(i, 1) = _data[1]*rhs(i,0) + _data[3]*rhs(i,1);
		}
		return out;
	}
	template<typename E>																								
	typename ldp_basic_mat2<typename type_promote<T,E>::type> operator * (const ldp_basic_mat<E,2,2>& rhs)const	
	{																													
		typename ldp_basic_mat2<typename type_promote<T,E>::type> out;												
		out._data[0] = _data[0] * rhs[0] + _data[2] * rhs[1];										
		out._data[1] = _data[1] * rhs[0] + _data[3] * rhs[1];										
		out._data[2] = _data[0] * rhs[2] + _data[2] * rhs[3];										
		out._data[3] = _data[1] * rhs[2] + _data[3] * rhs[3];
		return out;																										
	}

	template<typename E>																								
	typename ldp_basic_vec2<typename type_promote<T,E>::type> operator * (const ldp_basic_vec<E,2>& rhs)const	
	{																													
		typename ldp_basic_vec2<typename type_promote<T,E>::type> out;												
		out[0] = _data[0] * rhs[0] + _data[2] * rhs[1];												
		out[1] = _data[1] * rhs[0] + _data[3] * rhs[1];	
		return out;																										
	}	
	ldp_basic_mat2<T> operator * (const T& rhs)const										
	{																							
		return ldp_basic_mat<T,2,2>::operator*(rhs);																			
	}	
	/**
	* Operators: *=
	* */
	template<typename E>																								
	ldp_basic_mat2<T>& operator *= (const ldp_basic_mat<E,2,2>& rhs)	
	{																													
		return (*this) = ((*this)*rhs);																									
	}
	ldp_basic_mat2<T>& operator *= (const T& x)
	{
		for(size_t i=0; i<NUM_ELEMENTS; i++)
			(*this)[i] *= x;
		return *this;
	}
	/**
	* Matrix Determinant
	* */
	T det()const
	{
		return _data[0]*_data[3] - _data[1]*_data[2];
	}
	/**
	* Matrix Inverse
	* */
	void inv(ldp_basic_mat<T,2,2>& result)const
	{
		T dt = 1/det();
		result(0,0) = (*this)(1,1) * dt;
		result(1,0) = -(*this)(1,0) * dt;
		result(0,1) = -(*this)(0,1) * dt;
		result(1,1) = (*this)(0,0) * dt;
	}
	ldp_basic_mat2<T> inv()const
	{
		ldp_basic_mat2<T> r;
		inv(r);
		return r;
	}
	/**
	* A*x = b, solve x = A\b
	* */
	ldp_basic_vec2<T> solve(const ldp_basic_vec<T,2>& b)const
	{
		return inv()*b;
	}
	/**
	* EigenValues and EigenVectors of symmetric matrix
	* */
	void eig(ldp_basic_vec<T,2>& eigVals, ldp_basic_mat<T,2,2>& eigVecs)const
	{
		T b=-trace();
		T c=det();
		T delta = sqrt(b*b-4*c);
		eigVals[0]=(-b+delta)*T(0.5);
		eigVals[1]=(-b-delta)*T(0.5);
		if(eigVals[0] < eigVals[1])
			std::swap(eigVals[0], eigVals[1]);
		ldp_basic_vec2<T> v1(_data[0]-eigVals[1],_data[1]);
		ldp_basic_vec2<T> v2(_data[0]-eigVals[0],_data[1]);
		T len1 = v1.sqrLength(), len2 = v2.sqrLength();
		if(len1==0 && len2==0)
		{
			eigVecs[0]=eigVecs[3]=1;
			eigVecs[1]=eigVecs[2]=0;
		}
		else
		{
			v1 /= sqrt(len1);
			v2 /= sqrt(len2);
			eigVecs[0]=v1[0];
			eigVecs[1]=v1[1];
			eigVecs[2]=v2[0];
			eigVecs[3]=v2[1];
		}
	}
	void eig(ldp_basic_vec<T,2>& eigVals)const
	{
		T b=-trace();
		T c=det();
		T delta = sqrt(b*b-4*c);
		eigVals[0]=(-b+delta)*T(0.5);
		eigVals[1]=(-b-delta)*T(0.5);
		if(eigVals[0] < eigVals[1])
			std::swap(eigVals[0], eigVals[1]);
	}
	ldp_basic_vec2<T> eig()const
	{
		ldp_basic_vec2<T> x;
		eig(x);
		return x;
	}
};


template<class T>
class ldp_basic_mat3 : public ldp_basic_mat_sqr<T, 3>
{
public:
	/**
	* Constructors
	* */
	ldp_basic_mat3():ldp_basic_mat_sqr<T,3>(){}
	ldp_basic_mat3(const T* data):ldp_basic_mat_sqr<T,3>(data){}
	ldp_basic_mat3(const T& v):ldp_basic_mat_sqr<T,3>(v){}
	template<class E> ldp_basic_mat3(const ldp_basic_mat<E,3,3>& rhs):ldp_basic_mat_sqr<T,3>(rhs){}
	template<class E1, class E2, size_t K> ldp_basic_mat3(const ldp_basic_mat<E1,K,3>& rhs1, const ldp_basic_mat<E2,3-K,3>& rhs2)
	:ldp_basic_mat_sqr<T,3>(rhs1, rhs2){}
	template<class E1, class E2, size_t K> ldp_basic_mat3(const ldp_basic_mat<E1,3,K>& rhs1, const ldp_basic_mat<E2,3,3-K>& rhs2)
	:ldp_basic_mat_sqr<T,3>(rhs1, rhs2){}

	/**
	* Operator: * with mat and vec
	* */
	template<typename E, size_t K>																								
	typename ldp_basic_mat<typename type_promote<T,E>::type, 3, K> operator * (const ldp_basic_mat<E,3,K>& rhs)const	
	{																													
		typename ldp_basic_mat<typename type_promote<T,E>::type, 3, K> out;	
		for(size_t i=0; i<K; i++)
		{
			out(0, i) = _data[0]*rhs(0,i) + _data[3]*rhs(1,i) + _data[6]*rhs(2,i);
			out(1, i) = _data[1]*rhs(0,i) + _data[4]*rhs(1,i) + _data[7]*rhs(2,i);
			out(2, i) = _data[2]*rhs(0,i) + _data[5]*rhs(1,i) + _data[8]*rhs(2,i);
		}		 
		return out;
	}
	template<typename E>																								
	typename ldp_basic_mat3<typename type_promote<T,E>::type> operator * (const ldp_basic_mat<E,3,3>& rhs)const	
	{																													
		typename ldp_basic_mat3<typename type_promote<T,E>::type> out;												
		out[0] = _data[0] * rhs[0] + _data[3] * rhs[1] + _data[6] * rhs[2];				
		out[1] = _data[1] * rhs[0] + _data[4] * rhs[1] + _data[7] * rhs[2];				
		out[2] = _data[2] * rhs[0] + _data[5] * rhs[1] + _data[8] * rhs[2];					
		out[3] = _data[0] * rhs[3] + _data[3] * rhs[4] + _data[6] * rhs[5];				
		out[4] = _data[1] * rhs[3] + _data[4] * rhs[4] + _data[7] * rhs[5];				
		out[5] = _data[2] * rhs[3] + _data[5] * rhs[4] + _data[8] * rhs[5];					
		out[6] = _data[0] * rhs[6] + _data[3] * rhs[7] + _data[6] * rhs[8];				
		out[7] = _data[1] * rhs[6] + _data[4] * rhs[7] + _data[7] * rhs[8];				
		out[8] = _data[2] * rhs[6] + _data[5] * rhs[7] + _data[8] * rhs[8];	
		return out;																										
	}

	template<typename E>																								
	typename ldp_basic_vec3<typename type_promote<T,E>::type> operator * (const ldp_basic_vec<E,3>& rhs)const	
	{																													
		typename ldp_basic_vec3<typename type_promote<T,E>::type> out;												
		out[0] = _data[0] * rhs[0] + _data[3] * rhs[1] + _data[6] * rhs[2];									
		out[1] = _data[1] * rhs[0] + _data[4] * rhs[1] + _data[7] * rhs[2];								
		out[2] = _data[2] * rhs[0] + _data[5] * rhs[1] + _data[8] * rhs[2];
		return out;																										
	}
	ldp_basic_mat3<T> operator * (const T& rhs)const										
	{																							
		return ldp_basic_mat<T,3,3>::operator*(rhs);																			
	}	
	/**
	* Operators: *=
	* */
	template<typename E>																								
	ldp_basic_mat3<T>& operator *= (const ldp_basic_mat<E,3,3>& rhs)	
	{																													
		return (*this) = ((*this)*rhs);																									
	}
	ldp_basic_mat3<T>& operator *= (const T& x)
	{
		for(size_t i=0; i<NUM_ELEMENTS; i++)
			(*this)[i] *= x;
		return *this;
	}
	/**
	* Matrix Determinant
	* */
	T det()const
	{
		return 		_data[0]*(_data[4]*_data[8] - _data[7]*_data[5]) 
				-	_data[3]*(_data[1]*_data[8] - _data[7]*_data[2])
				+	_data[6]*(_data[1]*_data[5] - _data[4]*_data[2]);
	}
	/**
	* Matrix Inverse
	* */
	void inv(ldp_basic_mat<T,3,3>& result)const
	{
		T dt = 1/det();
		result(0,0) = dt * ( (*this)(1,1) * (*this)(2,2) - (*this)(1,2) * (*this)(2,1) );
		result(1,0) = dt * ( (*this)(1,2) * (*this)(2,0) - (*this)(1,0) * (*this)(2,2) );
		result(2,0) = dt * ( (*this)(1,0) * (*this)(2,1) - (*this)(1,1) * (*this)(2,0) );
		
		result(0,1) = dt * ( (*this)(0,2) * (*this)(2,1) - (*this)(0,1) * (*this)(2,2) );
		result(1,1) = dt * ( (*this)(0,0) * (*this)(2,2) - (*this)(0,2) * (*this)(2,0) );
		result(2,1) = dt * ( (*this)(0,1) * (*this)(2,0) - (*this)(0,0) * (*this)(2,1) );
		
		result(0,2) = dt * ( (*this)(0,1) * (*this)(1,2) - (*this)(0,2) * (*this)(1,1) );
		result(1,2) = dt * ( (*this)(0,2) * (*this)(1,0) - (*this)(0,0) * (*this)(1,2) );
		result(2,2) = dt * ( (*this)(0,0) * (*this)(1,1) - (*this)(0,1) * (*this)(1,0) );
	}
	ldp_basic_mat3<T> inv()const
	{
		ldp_basic_mat3<T> r;
		inv(r);
		return r;
	}
	/**
	* A*x = b, solve x = A\b
	* */
	ldp_basic_vec3<T> solve(const ldp_basic_vec<T,3>& b)const
	{
		return inv()*b;
	}
	/**
	* EigenValues and EigenVectors of symmetric matrix
	* */
	void eig(ldp_basic_vec<T,3>& eigVals, ldp_basic_mat<T,3,3>& eigVecs)const
	{
		ldp_basic_mat_sqr<T,3>::eig();
	}
	void eig(ldp_basic_vec<T,3>& eigVals)const
	{
		const ldp_basic_mat3<T>& A = (*this);
		T p = sqr(A(0,1)) + sqr(A(0,2)) + sqr(A(1,2));
		if(p == T(0))
		{
			eigVals[0] = A(0,0);
			eigVals[1] = A(1,1);
			eigVals[2] = A(2,2);
		}
		else
		{
			T q = A.trace() * (T(1)/T(3));
			p = sqr(A(0,0)-q) + sqr(A(1,1)-q) + sqr(A(2,2)-q) + 2*p;
			p = sqrt( p/T(6) );
			ldp_basic_mat3<T> B = A;
			for(size_t i=0; i<3; i++)
				B(i,i) -= q;
			B *= T(1)/p;
			T r = B.det() / T(2);
			T phi = acos(r) / T(3);
			if(r <= T(-1))
				phi = T(LDP_CONST_PI)/T(3);
			else if (r >= T(1))
				phi = T(0);
			eigVals[0] = q + T(2) * p * cos(phi);
			eigVals[2] = q + T(2) * p * cos(phi + T(LDP_CONST_PI) * T(2)/T(3));
			eigVals[1] = T(3) * q - eigVals[0] - eigVals[2]; 
		}
	}
	ldp_basic_vec3<T> eig()const
	{
		ldp_basic_vec3<T> x;
		eig(x);
		return x;
	}
};


template<class T>
class ldp_basic_mat4 : public ldp_basic_mat_sqr<T, 4>
{
public:
	/**
	* Constructors
	* */
	ldp_basic_mat4():ldp_basic_mat_sqr<T,4>(){}
	ldp_basic_mat4(const T* data):ldp_basic_mat_sqr<T,4>(data){}
	ldp_basic_mat4(const T& v):ldp_basic_mat_sqr<T,4>(v){}
	template<class E> ldp_basic_mat4(const ldp_basic_mat<E,4,4>& rhs):ldp_basic_mat_sqr<T,4>(rhs){}
	template<class E1, class E2, size_t K> ldp_basic_mat4(const ldp_basic_mat<E1,K,4>& rhs1, const ldp_basic_mat<E2,4-K,4>& rhs2)
	:ldp_basic_mat_sqr<T,4>(rhs1, rhs2){}
	template<class E1, class E2, size_t K> ldp_basic_mat4(const ldp_basic_mat<E1,4,K>& rhs1, const ldp_basic_mat<E2,4,4-K>& rhs2)
	:ldp_basic_mat_sqr<T,4>(rhs1, rhs2){}

	/**
	* Operator: * with mat and vec
	* */
	template<typename E, size_t K>																								
	typename ldp_basic_mat<typename type_promote<T,E>::type, 4, K> operator * (const ldp_basic_mat<E,4,K>& rhs)const	
	{																													
		typename ldp_basic_mat<typename type_promote<T,E>::type, 4, K> out;	
		for(size_t i=0; i<K; i++)
		{
			out(0, i) = _data[0]*rhs(0,i) + _data[4]*rhs(1,i) + _data[8 ]*rhs(2,i) + _data[12]*rhs(3,i);
			out(1, i) = _data[1]*rhs(0,i) + _data[5]*rhs(1,i) + _data[9 ]*rhs(2,i) + _data[13]*rhs(3,i);
			out(2, i) = _data[2]*rhs(0,i) + _data[6]*rhs(1,i) + _data[10]*rhs(2,i) + _data[14]*rhs(3,i);
			out(3, i) = _data[3]*rhs(0,i) + _data[7]*rhs(1,i) + _data[11]*rhs(2,i) + _data[15]*rhs(3,i);
		}
		return out;
	}

	template<typename E>																								
	typename ldp_basic_vec4<typename type_promote<T,E>::type> operator * (const ldp_basic_vec<E,4>& rhs)const	
	{																													
		typename ldp_basic_vec4<typename type_promote<T,E>::type> out;									
		out[0] = _data[0] * rhs[0] + _data[4] * rhs[1] + _data[8 ] * rhs[2] + _data[12] * rhs[3];
		out[1] = _data[1] * rhs[0] + _data[5] * rhs[1] + _data[9 ] * rhs[2] + _data[13] * rhs[3];
		out[2] = _data[2] * rhs[0] + _data[6] * rhs[1] + _data[10] * rhs[2] + _data[14] * rhs[3];
		out[3] = _data[3] * rhs[0] + _data[7] * rhs[1] + _data[11] * rhs[2] + _data[15] * rhs[3];	
		return out;																										
	}
	ldp_basic_mat4<T> operator * (const T& rhs)const										
	{																							
		return ldp_basic_mat<T,4,4>::operator*(rhs);																			
	}
	/**
	* Operators: *=
	* */
	template<typename E>																								
	ldp_basic_mat4<T>& operator *= (const ldp_basic_mat<E,4,4>& rhs)	
	{																													
		return (*this) = ((*this)*rhs);																									
	}
	ldp_basic_mat4<T>& operator *= (const T& x)
	{
		for(size_t i=0; i<NUM_ELEMENTS; i++)
			(*this)[i] *= x;
		return *this;
	}

	ldp_basic_mat3<T> getRotationPart()const
	{
		ldp_basic_mat3<T> R;
		for (int y = 0; y < 3; y++)
		for (int x = 0; x < 3; x++)
			R(y, x) = (*this)(y, x);
		return R;
	}

	void setRotationPart(const ldp_basic_mat3<T>& R)
	{
		for (int y = 0; y < 3; y++)
		for (int x = 0; x < 3; x++)
			(*this)(y, x) = R(y, x);
	}

	ldp_basic_vec3<T> getTranslationPart()const
	{
		ldp_basic_vec3<T> t;
		for (int y = 0; y < 3; y++)
			t[y] = (*this)(y, 3);
		return t;
	}

	void setTranslationPart(const ldp_basic_vec3<T>& t)
	{
		for (int y = 0; y < 3; y++)
			(*this)(y, 3) = t[y];
	}
};

/** ************************************************************************************
* typedefs, for convinence
* *************************************************************************************/
typedef ldp_basic_mat2<char> Mat2c;
typedef ldp_basic_mat3<char> Mat3c;
typedef ldp_basic_mat4<char> Mat4c;
typedef ldp_basic_mat_sqr<char, 5> Mat5c;
typedef ldp_basic_mat_sqr<char, 6> Mat6c;
typedef ldp_basic_mat2<unsigned char> Mat2b;
typedef ldp_basic_mat3<unsigned char> Mat3b;
typedef ldp_basic_mat4<unsigned char> Mat4b;
typedef ldp_basic_mat_sqr<unsigned char, 5> Mat5b;
typedef ldp_basic_mat_sqr<unsigned char, 6> Mat6b;
typedef ldp_basic_mat2<short> Mat2s;
typedef ldp_basic_mat3<short> Mat3s;
typedef ldp_basic_mat4<short> Mat4s;
typedef ldp_basic_mat_sqr<short, 5> Mat5s;
typedef ldp_basic_mat_sqr<short, 6> Mat6s;
typedef ldp_basic_mat2<unsigned short> Mat2us;
typedef ldp_basic_mat3<unsigned short> Mat3us;
typedef ldp_basic_mat4<unsigned short> Mat4us;
typedef ldp_basic_mat_sqr<unsigned short, 5> Mat5us;
typedef ldp_basic_mat_sqr<unsigned short, 6> Mat6us;
typedef ldp_basic_mat2<int> Mat2i;
typedef ldp_basic_mat3<int> Mat3i;
typedef ldp_basic_mat4<int> Mat4i;
typedef ldp_basic_mat_sqr<int, 5> Mat5i;
typedef ldp_basic_mat_sqr<int, 6> Mat6i;
typedef ldp_basic_mat2<unsigned int> Mat2u;
typedef ldp_basic_mat3<unsigned int> Mat3u;
typedef ldp_basic_mat4<unsigned int> Mat4u;
typedef ldp_basic_mat_sqr<unsigned int, 5> Mat5u;
typedef ldp_basic_mat_sqr<unsigned int, 6> Mat6u;
typedef ldp_basic_mat2<float> Mat2f;
typedef ldp_basic_mat3<float> Mat3f;
typedef ldp_basic_mat4<float> Mat4f;
typedef ldp_basic_mat_sqr<float, 5> Mat5f;
typedef ldp_basic_mat_sqr<float, 6> Mat6f;
typedef ldp_basic_mat2<double> Mat2d;
typedef ldp_basic_mat3<double> Mat3d;
typedef ldp_basic_mat4<double> Mat4d;
typedef ldp_basic_mat_sqr<double, 5> Mat5d;
typedef ldp_basic_mat_sqr<double, 6> Mat6d;

/** ***************************************************************************************************
* Some details
* *****************************************************************************************************/
template<typename _Tp> bool
JacobiImpl_( _Tp* A, size_t astep, _Tp* W, _Tp* V, size_t vstep, int n, int* buf )
{
    const _Tp eps = std::numeric_limits<_Tp>::epsilon();
    int i, j, k, m;
    
    if( V )
    {
        for( i = 0; i < n; i++ )
        {
            for( j = 0; j < n; j++ )
                V[i*vstep + j] = (_Tp)0;
            V[i*vstep + i] = (_Tp)1;
        }
    }
    
    int iters, maxIters = n*n*30;
    
    int* indR = buf;
    int* indC = indR + n;
    _Tp mv = (_Tp)0;
    
    for( k = 0; k < n; k++ ) 
    {
        W[k] = A[(astep + 1)*k];
        if( k < n - 1 )
        {
            for( m = k+1, mv = std::abs(A[astep*k + m]), i = k+2; i < n; i++ )
            {
                _Tp val = std::abs(A[astep*k+i]);
                if( mv < val )
                    mv = val, m = i;
            }
            indR[k] = m;
        }
        if( k > 0 )
        {
            for( m = 0, mv = std::abs(A[k]), i = 1; i < k; i++ )
            {
                _Tp val = std::abs(A[astep*i+k]);
                if( mv < val )
                    mv = val, m = i;
            }
            indC[k] = m;
        }
    }
    
    if( n > 1 ) for( iters = 0; iters < maxIters; iters++ )
    {
        // find index (k,l) of pivot p
        for( k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n-1; i++ )
        {
            _Tp val = std::abs(A[astep*i + indR[i]]);
            if( mv < val )
                mv = val, k = i;
        }
        int l = indR[k];
        for( i = 1; i < n; i++ )
        {
            _Tp val = std::abs(A[astep*indC[i] + i]);
            if( mv < val )
                mv = val, k = indC[i], l = i;
        }
        
        _Tp p = A[astep*k + l];
        if( std::abs(p) <= eps )
            break;
        _Tp y = (_Tp)((W[l] - W[k])*0.5);
        _Tp t = std::abs(y) + hypot(p, y);
        _Tp s = hypot(p, t);
        _Tp c = t/s;
        s = p/s; t = (p/t)*p;
        if( y < 0 )
            s = -s, t = -t;
        A[astep*k + l] = 0;
        
        W[k] -= t;
        W[l] += t;
        
        _Tp a0, b0;
        
#undef ldp_eig_rotate
#define ldp_eig_rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c
        
        // rotate rows and columns k and l
        for( i = 0; i < k; i++ )
            ldp_eig_rotate(A[astep*i+k], A[astep*i+l]);
        for( i = k+1; i < l; i++ )
            ldp_eig_rotate(A[astep*k+i], A[astep*i+l]);
        for( i = l+1; i < n; i++ )
            ldp_eig_rotate(A[astep*k+i], A[astep*l+i]);
        
        // rotate eigenvectors
        if( V )
            for( i = 0; i < n; i++ )
                ldp_eig_rotate(V[vstep*k+i], V[vstep*l+i]);
        
#undef ldp_eig_rotate
        
        for( j = 0; j < 2; j++ )
        {
            int idx = j == 0 ? k : l;
            if( idx < n - 1 )
            {
                for( m = idx+1, mv = std::abs(A[astep*idx + m]), i = idx+2; i < n; i++ )
                {
                    _Tp val = std::abs(A[astep*idx+i]);
                    if( mv < val )
                        mv = val, m = i;
                }
                indR[idx] = m;
            }
            if( idx > 0 )
            {
                for( m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++ )
                {
                    _Tp val = std::abs(A[astep*i+idx]);
                    if( mv < val )
                        mv = val, m = i;
                }
                indC[idx] = m;
            }
        }
    }
    
    // sort eigenvalues & eigenvectors
    for( k = 0; k < n-1; k++ )
    {
        m = k;
        for( i = k+1; i < n; i++ )
        {
            if( W[m] < W[i] )
                m = i;
        }
        if( k != m )
        {
            std::swap(W[m], W[k]);
            if( V )
                for( i = 0; i < n; i++ )
                    std::swap(V[vstep*m + i], V[vstep*k + i]);
        }
    }
    
    return true;
}

}//namespace ldp