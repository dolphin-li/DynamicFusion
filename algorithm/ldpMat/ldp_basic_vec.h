#pragma once

/** **************************************************************************************
* ldp_basic_vec
* @author Dongping Li
* This class is suitable for small vectors, E. G., size 2, 3, 4, 5, 6...
* For vector of size 2, 3 and 4, ldp_basic_vec2/3/4, child classes of ldp_basic_vec, can be used.
* This class is designed firstly for convinience and then for efficiency.
* ****************************************************************************************/


#include "ldpdef.h"

namespace ldp
{

template<typename T, size_t N>
class ldp_basic_vec
{
public:
	T _data[N];

public:
	/**
	* Constructors
	* */
	ldp_basic_vec() {memset(ptr(), 0, N*sizeof(T));}
	ldp_basic_vec(const T*data) {memcpy(ptr(), data, N*sizeof(T));}
	ldp_basic_vec(const T& v)
	{
		for(size_t i=0; i<N; i++)
			(*this)[i] = static_cast<T>(v);
	}
	template<class E> ldp_basic_vec(const ldp_basic_vec<E,N>& rhs)
	{
		for(size_t i=0; i<N; i++)
			(*this)[i] = static_cast<T>(rhs[i]);
	}
	template<class E1, class E2, size_t M> ldp_basic_vec(const ldp_basic_vec<E1,M>& A, const ldp_basic_vec<E2,N-M>& B)
	{
		for(size_t i=0; i<M; i++)
			(*this)[i] = static_cast<T>(A[i]);
		for(size_t i=M; i<N; i++)
			(*this)[i] = static_cast<T>(B[i-M]);
	}

	/**
	* Data Access Methods
	* */
	size_t size()const {return N;}
	const T* ptr()const {return _data;}
	T* ptr() {return _data;}
	const T& operator() (size_t i)const {return ptr()[i];}
	T& operator() (size_t i) {return ptr()[i];}
	const T& operator[] (size_t i)const {return ptr()[i];}
	T& operator[] (size_t i) {return ptr()[i];}

	/**
	* Useful Functions
	* dot, dotSelf
	* sqrDist
	* */
	template<typename E>
	typename type_promote<T,E>::type dot (const ldp_basic_vec<E,N>& rhs)const
	{
		typename type_promote<T,E>::type sum = 0;
		for(size_t i=0; i<N; i++)
			sum += (*this)[i] * rhs[i];
		return sum;
	}
	
	T sqrLength ()const
	{
		T sum = 0;
		for(size_t i=0; i<N; i++)
			sum += (*this)[i] * (*this)[i];
		return sum;
	}

	T length()const
	{
		return sqrt(sqrLength());
	}

	ldp_basic_vec<T, N> normalize()const
	{
		return (*this) / length();
	}

	ldp_basic_vec<T, N>& normalizeLocal()
	{
		(*this) /= length();
		return *this;
	}

	template<typename E>
	typename type_promote<T,E>::type sqrDist (const ldp_basic_vec<E,N>& rhs)const
	{
		typename type_promote<T,E>::type sum = 0;
		for(size_t i=0; i<N; i++)
		{
			typename type_promote<T,E>::type v = (*this)[i] - rhs[i];
			sum += v * v;
		}
		return sum;
	}

	/**
	* Operators: +,-,*,/
	* */
#define LDP_BASIC_VEC_ARITHMATIC(OP)															\
	template<typename E>																			\
	typename ldp_basic_vec<typename type_promote<T,E>::type, N> operator OP (const ldp_basic_vec<E,N>& rhs)const	\
	{																							\
		typename ldp_basic_vec<typename type_promote<T,E>::type, N> out;												\
		for(size_t i=0; i<N; i++)																\
			out[i] = (*this)[i] OP rhs[i];														\
		return out;																				\
	}																							\
	ldp_basic_vec<T,N> operator OP (const T& rhs)const					\
	{																							\
		ldp_basic_vec<T,N> out;												\
		for(size_t i=0; i<N; i++)																\
			out[i] = (*this)[i] OP rhs;															\
		return out;																				\
	}																							\
	//friend ldp_basic_vec<T,N> operator OP (const T& lhs, const ldp_basic_vec<T,N>& rhs);
	/**
	* Operators: +=, -=, *=, /=, =
	* */
#define LDP_BASIC_VEC_ARITHMATIC2(OP)															\
	template<class E>																			\
	ldp_basic_vec<T, N>& operator OP (const ldp_basic_vec<E,N>& rhs)							\
	{																							\
		for(size_t i=0; i<N; i++)																\
			(*this)[i] OP static_cast<T>(rhs[i]);													\
		return *this;																			\
	}																							\
	ldp_basic_vec<T, N>& operator OP (const T& rhs)												\
	{																							\
		for(size_t i=0; i<N; i++)																\
			(*this)[i] OP rhs;													\
		return *this;																			\
	}																							\

	LDP_BASIC_VEC_ARITHMATIC(+)
	LDP_BASIC_VEC_ARITHMATIC(-)
	LDP_BASIC_VEC_ARITHMATIC(*)
	LDP_BASIC_VEC_ARITHMATIC(/)
	LDP_BASIC_VEC_ARITHMATIC2(+=)
	LDP_BASIC_VEC_ARITHMATIC2(-=)
	LDP_BASIC_VEC_ARITHMATIC2(*=)
	LDP_BASIC_VEC_ARITHMATIC2(/=)
	LDP_BASIC_VEC_ARITHMATIC2(=)

	/**
	* Logic Operators: >,<,>=,<=,==,!=
	* */
	bool operator > (const ldp_basic_vec& rhs)const
	{
		for(size_t i=0; i<N; i++)
		{
			if((*this)[i] < rhs[i])
				return false;
			else if((*this)[i] > rhs[i])
				return true;
		}
		return false;
	}
	bool operator < (const ldp_basic_vec& rhs)const
	{
		for(size_t i=0; i<N; i++)
		{
			if((*this)[i] > rhs[i])
				return false;
			else if((*this)[i] < rhs[i])
				return true;
		}
		return false;
	}
	bool operator >= (const ldp_basic_vec& rhs)const
	{
		return !((*this)<rhs);
	}
	bool operator <= (const ldp_basic_vec& rhs)const
	{
		return !((*this)>rhs);
	}
	bool operator == (const ldp_basic_vec& rhs)const
	{
		for(size_t i=0; i<N; i++)
		{
			if((*this)[i] != rhs[i])
				return false;
		}
		return true;
	}
	bool operator != (const ldp_basic_vec& rhs)const
	{
		return !((*this)==rhs);
	}

};//template<T,N> ldp_basic_vec

/**
* Arithmatic Operators: +,-,*,/
* */
#define LDP_BASIC_VEC_ARITHMATIC_FRIEND(OP)																		\
	template<typename T, size_t N>																		\
	inline ldp_basic_vec<T,N>operator OP (const T& lhs, const ldp_basic_vec<T,N>& rhs)		\
	{																											\
		ldp_basic_vec<T, N> out;																\
		for(size_t i=0; i<N; i++)																				\
			out[i] = lhs OP rhs[i];																				\
		return out;																								\
	}

	LDP_BASIC_VEC_ARITHMATIC_FRIEND(+)
	LDP_BASIC_VEC_ARITHMATIC_FRIEND(-)
	LDP_BASIC_VEC_ARITHMATIC_FRIEND(*)
	LDP_BASIC_VEC_ARITHMATIC_FRIEND(/)
#undef LDP_BASIC_VEC_ARITHMATIC
#undef LDP_BASIC_VEC_ARITHMATIC2
#undef LDP_BASIC_VEC_ARITHMATIC_FRIEND
/**
* I/O
* */
template<typename T, size_t N>
inline std::ostream& operator<<(std::ostream& out, const ldp_basic_vec<T,N>& v) 
{
	for(size_t i=0; i<N-1; i++)
		out << std::left << std::setw(10) << v[i] << " ";
	if(N >= 1) out << v[N-1];
    return out;
}

template<typename T, size_t N>
inline std::istream& operator>>(std::istream& in, const ldp_basic_vec<T,N>& v) 
{
	for(size_t i=0; i<N; i++)
		in >> v[i];
    return in;
}

/** *************************************************************************************
* Some Specifications: ldp_basic_vec2/3/4
* **************************************************************************************/
template<class T>
class ldp_basic_vec2: public ldp_basic_vec<T,2>
{
public:
	ldp_basic_vec2(){(*this)[0]=0,(*this)[1]=0;}
	ldp_basic_vec2(const T& v){(*this)[0]=v, (*this)[1]=v;}
	ldp_basic_vec2(const T& xx, const T& yy){(*this)[0]=xx, (*this)[1]=yy;}
	template<class E> ldp_basic_vec2(const ldp_basic_vec<E,2>& rhs){(*this)[0]=T(rhs[0]), (*this)[1]=T(rhs[1]);}

	const T& x()const{return (*this)[0];}
	T& x(){return (*this)[0];}
	const T& y()const{return (*this)[1];}
	T& y(){return (*this)[1];}

	template<class E> typename type_promote<T,E>::type cross(const ldp_basic_vec<E,2>& rhs)const
	{
		return (*this)[0]*rhs[1] - (*this)[1]*rhs[0];
	}
};

template<class T>
class ldp_basic_vec3: public ldp_basic_vec<T,3>
{
public:
	ldp_basic_vec3(){(*this)[0]=0,(*this)[1]=0,(*this)[2]=0;}
	ldp_basic_vec3(const T& v){(*this)[0]=v, (*this)[1]=v,(*this)[2]=v;}
	ldp_basic_vec3(const T& xx, const T& yy, const T& zz){(*this)[0]=xx, (*this)[1]=yy,(*this)[2]=zz;}
	template<class E> ldp_basic_vec3(const ldp_basic_vec<E,3>& rhs){(*this)[0]=T(rhs[0]), (*this)[1]=T(rhs[1]),(*this)[2]=T(rhs[2]);}

	const T& x()const{return (*this)[0];}
	T& x(){return (*this)[0];}
	const T& y()const{return (*this)[1];}
	T& y(){return (*this)[1];}
	const T& z()const{return (*this)[2];}
	T& z(){return (*this)[2];}

	template<class E> typename ldp_basic_vec3<typename type_promote<T,E>::type> cross(const ldp_basic_vec<E,3>& rhs)const
	{
		return ldp_basic_vec3<typename type_promote<T,E>::type>( (*this)[1]*rhs[2] - (*this)[2]*rhs[1],
							(*this)[2]*rhs[0] - (*this)[0]*rhs[2], (*this)[0]*rhs[1] - (*this)[1]*rhs[0] );
	}
};

template<class T>
class ldp_basic_vec4: public ldp_basic_vec<T,4>
{
public:
	ldp_basic_vec4(){(*this)[0]=0,(*this)[1]=0,(*this)[2]=0,(*this)[3]=0;}
	ldp_basic_vec4(const T& v){(*this)[0]=v, (*this)[1]=v,(*this)[2]=v,(*this)[3]=v;}
	ldp_basic_vec4(const T& xx, const T& yy, const T& zz, const T& ww){(*this)[0]=xx, (*this)[1]=yy,(*this)[2]=zz,(*this)[3]=ww;}
	template<class E> ldp_basic_vec4(const ldp_basic_vec<E,4>& rhs)
	{(*this)[0]=T(rhs[0]), (*this)[1]=T(rhs[1]),(*this)[2]=T(rhs[2]),(*this)[3]=T(rhs[3]);}

	const T& x()const{return (*this)[0];}
	T& x(){return (*this)[0];}
	const T& y()const{return (*this)[1];}
	T& y(){return (*this)[1];}
	const T& z()const{return (*this)[2];}
	T& z(){return (*this)[2];}
	const T& w()const{return (*this)[3];}
	T& w(){return (*this)[3];}
};


/** ************************************************************************************
* typedefs, for convinence
* *************************************************************************************/
typedef ldp_basic_vec2<char> Char2;
typedef ldp_basic_vec3<char> Char3;
typedef ldp_basic_vec4<char> Char4;
typedef ldp_basic_vec<char, 5> Char5;
typedef ldp_basic_vec<char, 6> Char6;
typedef ldp_basic_vec2<unsigned char> UChar2;
typedef ldp_basic_vec3<unsigned char> UChar3;
typedef ldp_basic_vec4<unsigned char> UChar4;
typedef ldp_basic_vec<unsigned char, 5> UChar5;
typedef ldp_basic_vec<unsigned char, 6> UChar6;
typedef ldp_basic_vec2<short> Short2;
typedef ldp_basic_vec3<short> Short3;
typedef ldp_basic_vec4<short> Short4;
typedef ldp_basic_vec<short, 5> Short5;
typedef ldp_basic_vec<short, 6> Short6;
typedef ldp_basic_vec2<unsigned short> UShort2;
typedef ldp_basic_vec3<unsigned short> UShort3;
typedef ldp_basic_vec4<unsigned short> UShort4;
typedef ldp_basic_vec<unsigned short, 5> UShort5;
typedef ldp_basic_vec<unsigned short, 6> UShort6;
typedef ldp_basic_vec2<int> Int2;
typedef ldp_basic_vec3<int> Int3;
typedef ldp_basic_vec4<int> Int4;
typedef ldp_basic_vec<int, 5> Int5;
typedef ldp_basic_vec<int, 6> Int6;
typedef ldp_basic_vec2<unsigned int> UInt2;
typedef ldp_basic_vec3<unsigned int> UInt3;
typedef ldp_basic_vec4<unsigned int> UInt4;
typedef ldp_basic_vec<unsigned int, 5> UInt5;
typedef ldp_basic_vec<unsigned int, 6> UInt6;
typedef ldp_basic_vec2<float> Float2;
typedef ldp_basic_vec3<float> Float3;
typedef ldp_basic_vec4<float> Float4;
typedef ldp_basic_vec<float, 5> Float5;
typedef ldp_basic_vec<float, 6> Float6;
typedef ldp_basic_vec2<double> Double2;
typedef ldp_basic_vec3<double> Double3;
typedef ldp_basic_vec4<double> Double4;
typedef ldp_basic_vec<double, 5> Double5;
typedef ldp_basic_vec<double, 6> Double6;

typedef ldp_basic_vec2<size_t> Size2;
typedef ldp_basic_vec3<size_t> Size3;
typedef ldp_basic_vec4<size_t> Size4;
typedef ldp_basic_vec<size_t, 5> Size5;
typedef ldp_basic_vec<size_t, 6> Size6;

/** ************************************************************************************
* Finally, undef all we used...
* *************************************************************************************/


}//namespace ldp