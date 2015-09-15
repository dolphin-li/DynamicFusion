#ifndef __QUATERNION_H__
#define __QUATERNION_H__

#include "ldp_basic_mat.h"

namespace ldp
{

	template<class DataType>
	class Quaternion
	{
	public:
		typedef ldp::ldp_basic_vec3<DataType> Vec;
		DataType w;
		Vec v;
	public:
		Quaternion()
		{
			w = (DataType)0;
			v[0] = (DataType)0;
			v[1] = (DataType)0;
			v[2] = (DataType)0;
		}
		
		template<class E>
		Quaternion(const Quaternion<E>& q)
		{
			w = static_cast<DataType>(q.w);
			v = q.v;
		}
		Quaternion(DataType x, DataType y, DataType z, DataType w)
		{
			this->v[0] = x;
			this->v[1] = y;
			this->v[2] = z;
			this->w = w;
		}
		Quaternion(Vec v, DataType w)
		{
			this->v = v;
			this->w = w;
		}

		// order:
		// [0]: w
		// [1]: x
		// [2]: y
		// [3]: z
		DataType& operator[](int i)
		{
			return ((DataType*)(this))[i];
		}

		// order:
		// [0]: w
		// [1]: x
		// [2]: y
		// [3]: z
		const DataType& operator[](int i)const
		{
			return ((DataType*)(this))[i];
		}

		Quaternion& setIdentity()
		{
			v = Vec(DataType(0));
			w = DataType(1);
			return *this;
		}

		bool isIdentity()const
		{
			return v == Vec(DataType(0)) && w == DataType(1);
		}

		Quaternion& setZero()
		{
			v = Vec(DataType(0));
			w = DataType(0);
			return *this;
		}

		DataType dot(Quaternion other)const
		{
			return v.dot(other.v) + w*other.w;
		}

		DataType norm()const
		{
			return sqrt(w * w + v.sqrLength());
		}

		Quaternion& normalizeLocal()
		{
			(*this) /= norm();
			return *this;
		}

		Quaternion normalize()const
		{
			Quaternion q = *this;
			return q.normalizeLocal();
		}

		Quaternion& inverseLocal()
		{
			DataType norm2 = dot(*this);
			if (norm2 > DataType(0)) {
				DataType invNorm2 = DataType(1) / norm2;
				v *= -invNorm2;
				w *= invNorm2;
			}
			return *this;
		}

		Quaternion inverse()const
		{
			Quaternion q = *this;
			return q.inverseLocal();
		}

		Quaternion conjugate()const
		{
			return Quaternion(-v, w);
		}

		Vec applyVec(Vec p)const
		{
			// nVidia SDK implementation
			Vec vp, vvp;
			vp = v.cross(p);
			vvp = v.cross(vp);
			vp *= DataType(2.0) * w;
			vvp *= DataType(2.0);

			return p + vp + vvp;
		}

		/**
		* Operators
		* */
		Quaternion operator*(const DataType& scalar)const
		{
			return Quaternion(scalar*v, scalar * w);
		}
		Quaternion& operator*=(const DataType& scalar)
		{
			v *= scalar;
			w *= scalar;
			return *this;
		}
		Quaternion operator/(const DataType& scalar)const
		{
			return Quaternion(v/scalar, w/scalar);
		}
		Quaternion& operator/=(const DataType& scalar)
		{
			v /= scalar;
			w /= scalar;
			return *this;
		}

		Quaternion operator*(const Quaternion &other)const
		{
			Quaternion r;
			r.w = (other.w * w) - other.v.dot(v);
			r.v[0] = (other.w * v[0]) + (other.v[0] * w) - (other.v[1] * v[2]) + (other.v[2] * v[1]);
			r.v[1] = (other.w * v[1]) + (other.v[1] * w) - (other.v[2] * v[0]) + (other.v[0] * v[2]);
			r.v[2] = (other.w * v[2]) + (other.v[2] * w) - (other.v[0] * v[1]) + (other.v[1] * v[0]);
			return r;
		}
		Quaternion& operator*=(const Quaternion &other)
		{
			(*this) = (*this) * other;
			return *this;
		}
		Quaternion operator+(const Quaternion& q)const
		{
			return Quaternion(v + q.v, w + q.w);
		}
		Quaternion& operator+=(const Quaternion& q)
		{
			v += q.v;
			w += q.w;
			return *this;
		}
		Quaternion operator-(const Quaternion& q)const
		{
			return Quaternion(v-q.v, w-q.w);
		}
		Quaternion& operator-=(const Quaternion& q)
		{
			v -= q.v;
			w -= q.w;
			return *this;
		}
		Quaternion operator-()const
		{
			return Quaternion(Vec(DataType(0)) - v,  - w);
		}

		Quaternion& fromAngles(Vec eularAngle)
		{
			DataType angle;

			angle = eularAngle[0] * DataType(0.5);
			const DataType sr = sin(angle);
			const DataType cr = cos(angle);

			angle = eularAngle[1] * DataType(0.5);
			const DataType sp = sin(angle);
			const DataType cp = cos(angle);

			angle = eularAngle[2] * DataType(0.5);
			const DataType sy = sin(angle);
			const DataType cy = cos(angle);

			const DataType cpcy = cp * cy;
			const DataType spcy = sp * cy;
			const DataType cpsy = cp * sy;
			const DataType spsy = sp * sy;

			v[0] = (DataType)(sr * cpcy - cr * spsy);
			v[1] = (DataType)(cr * spcy + sr * cpsy);
			v[2] = (DataType)(cr * cpsy - sr * spcy);
			w = (DataType)(cr * cpcy + sr * spsy);

			return normalizeLocal();
		}

		Vec toAngles()const
		{
			const DataType sqw = w*w;
			const DataType sqx = v[0]*v[0];
			const DataType sqy = v[1]*v[1];
			const DataType sqz = v[2]*v[2];
			const DataType test = DataType(2.0) * (v[1]*w - v[0]*v[2]);

			Vec euler;
			if (abs(test - DataType(1.0)) < DataType(0.000001))
			{
				// heading = rotation about z-axis
				euler[2] = (DataType)(-2.0*atan2(X, W));
				// bank = rotation about x-axis
				euler[0] = 0;
				// attitude = rotation about y-axis
				euler[1] = (DataType)(ldp::PI_D / 2.0);
			}
			else if (abs(test - DataType(-1.0)) < DataType(0.000001))
			{
				// heading = rotation about z-axis
				euler[2] = (DataType)(2.0*atan2(v[0], w));
				// bank = rotation about x-axis
				euler[0] = 0;
				// attitude = rotation about y-axis
				euler[1] = (DataType)(ldp::PI_D / -2.0);
			}
			else
			{
				// heading = rotation about z-axis
				euler[2] = (DataType)atan2(2.0 * (v[0]*v[1] + v[2]*w), (sqx - sqy - sqz + sqw));
				// bank = rotation about x-axis
				euler[0] = (DataType)atan2(2.0 * (v[1]*v[2] + v[0]*w), (-sqx - sqy + sqz + sqw));
				// attitude = rotation about y-axis
				euler[1] = (DataType)asin(std::max((DataType)-1.0, std::min((DataType)1.0, test)));
			}
		}

		Quaternion& fromAngleAxis(const DataType& angle, const Vec& axis)
		{
			const DataType fHalfAngle = DataType(0.5)*angle;
			const DataType fSin = sinf(fHalfAngle);
			w = cosf(fHalfAngle);
			v = fSin * axis.normalize();
			return *this;
		}

		void toAngleAxis(Vec& axis, DataType& angle)const
		{
			const DataType scale = v.length();

			if ((scale == DataType(0)) || w > DataType(1) || w < DataType(-1))
			{
				angle = 0.0f;
				axis[0] = 0.0f;
				axis[1] = 1.0f;
				axis[2] = 0.0f;
			}
			else
			{
				angle = DataType(2.0) * acosf(w);
				axis = v / scale;
			}
		}

		Quaternion& fromRotationVecs(const Vec& from, const Vec& to)
		{
			// Based on Stan Melax's article in Game Programming Gems
			// Copy, since cannot modify local
			Vec v0 = from;
			Vec v1 = to;
			v0.normalize();
			v1.normalize();

			const DataType d = v0.dot(v1);
			if (d >= DataType(1.0)) // If dot == 1, vectors are the same
			{
				return setIdentity();
			}
			else if (d <= DataType(-1.0)) // exactly opposite
			{
				Vec axis(1.0, 0., 0.);
				axis = axis.cross(v0);
				if (axis.length() == 0)
				{
					axis = Vec(0.f, 1.f, 0.f);
					axis = axis.cross(v0);
				}
				// same as fromAngleAxis(core::PI, axis).normalize();
				return Quaternion(axis, 0).normalize();
			}

			const DataType s = sqrt((DataType(1) + d) * DataType(2)); // optimize inv_sqrt
			const DataType invs = 1.f / s;
			const Vec c = v0.cross(v1)*invs;
			v = c;
			w = s * DataType(0.5);
			return normalizeLocal();
		}

		Quaternion& fromRotationMatrix(const ldp::ldp_basic_mat3<DataType>& M)
		{
			const DataType diag = M.trace() + DataType(1);

			if (diag > DataType(0))
			{
				const DataType scale = sqrtf(diag) * DataType(2); // get scale from diagonal

				// TODO: speed this up
				v[0] = (M(2,1) - M(1,2)) / scale;
				v[1] = (M(0,2) - M(2,0)) / scale;
				v[2] = (M(1,0) - M(0,1)) / scale;
				w = DataType(0.25) * scale;
			}
			else
			{
				if (M(0,0)>M(1,1) && M(0,0)>M(2,2))
				{
					// 1st element of diag is greatest value
					// find scale according to 1st element, and double it
					const DataType scale = sqrt(DataType(1) + M(0,0) - M(1,1) - M(2,2)) * DataType(2);

					// TODO: speed this up
					v[0] = DataType(0.25) * scale;
					v[1] = (M(0,1) + M(1,0)) / scale;
					v[2] = (M(2,0) + M(0,2)) / scale;
					w = (M(2,1) - M(1,2)) / scale;
				}
				else if (M(1,1)>M(2,2))
				{
					// 2nd element of diag is greatest value
					// find scale according to 2nd element, and double it
					const DataType scale = sqrt(DataType(1) + M(1,1) - M(0,0) - M(2,2)) * DataType(2);

					// TODO: speed this up
					v[0] = (M(0,1) + M(1,0)) / scale;
					v[1] = DataType(0.25) * scale;
					v[2] = (M(1,2) + M(2,1)) / scale;
					w = (M(0,2) - M(2,0)) / scale;
				}
				else
				{
					// 3rd element of diag is greatest value
					// find scale according to 3rd element, and double it
					const DataType scale = sqrtf(DataType(1) + M(2,2) - M(0,0) - M(1,1)) * DataType(2);

					// TODO: speed this up
					v[0] = (M(0,2) + M(2,0)) / scale;
					v[1] = (M(2,1) + M(1,2)) / scale;
					v[2] = DataType(0.25) * scale;
					w = (M(1,0) - M(0,1)) / scale;
				}
			}

			return normalizeLocal();
		}

		Quaternion& fromRotationMatrix(const ldp::ldp_basic_mat4<DataType>& M)
		{
			return fromRotationMatrix(M.getRotationPart());
		}

		ldp::ldp_basic_mat4<DataType> toRotationMatrix()const
		{
			ldp::ldp_basic_mat4<DataType> dest;
			dest.eye();

			const static DataType one = DataType(1);
			const static DataType two = DataType(2);

			dest(0, 0) = v[0]*v[0] + w*w - v[1] * v[1] - v[2] * v[2];
			dest(1, 0) = two*v[0] * v[1] + two*v[2] * w;
			dest(2, 0) = two*v[0] * v[2] - two*v[1] * w;

			dest(0, 1) = two*v[0] * v[1] - two*v[2] * w;
			dest(1, 1) = v[1]*v[1] + w*w - v[0] * v[0] - v[2] * v[2];
			dest(2, 1) = two*v[2] * v[1] + two*v[0] * w;

			dest(0,2) = two*v[0]*v[2] + two*v[1]*w;
			dest(1,2) = two*v[2]*v[1] - two*v[0]*w;
			dest(2, 2) = v[2]*v[2] + w*w - v[0] * v[0] - v[1] * v[1];

			return dest;
		}

		ldp::ldp_basic_mat3<DataType> toRotationMatrix3()const
		{
			return toRotationMatrix().getRotationPart();
		}
	};

	template<class DataType>
	Quaternion<DataType> operator*(const DataType& scalar, const Quaternion<DataType>& q)
	{
		return q*scalar;
	}

	typedef Quaternion<float> QuaternionF;
	typedef Quaternion<double> QuaternionD;
/** ***********************************************************************************************
* dual quaternion
* *************************************************************************************************/

template<class DataType>
class DualQuaternion
{
public:
	Quaternion<DataType> dq[2];
public:
	DualQuaternion()
	{
		setZero();
	}

	template<class E1, class E2>
	DualQuaternion(const Quaternion<E1>& q1, const Quaternion<E2>& q2)
	{
		dq[0] = q1;
		dq[1] = q2;
	}

	template<class E>
	DualQuaternion(const DualQuaternion<E>& r)
	{
		dq[0] = r.dq[0];
		dq[1] = r.dq[1];
	}

	DualQuaternion& setZero()
	{
		dq[0].setZero();
		dq[1].setZero();
		return *this;
	}

	DualQuaternion& setIdentity()
	{
		dq[0].setIdentity();
		dq[1].setZero();
		return *this;
	}

	DualQuaternion& normalizeLocal()
	{
		DataType len = dq[0].norm();
		dq[0] /= len;
		dq[1] /= len;
		dq[1] -= dq[0].dot(dq[1]) / (len*len*len) * dq[0];
		return *this;
	}

	DualQuaternion normalize()const
	{
		DualQuaternion q = (*this);
		return q.normalizeLocal();
	}

	DualQuaternion inverse()const
	{
		DualQuaternion result;
		result.dq[0] = dq[0].inverse();
		result.dq[1] = -result.dq[0] * dq[1] * result.dq[0];
		return result;
	}

	DualQuaternion& inverseLocal()
	{
		*this = inverse();
		return *this;
	}

	DualQuaternion conjugate()const
	{
		DualQuaternion result;
		result.dq[0] = dq[0].conjugate();
		result.dq[1] = dq[1].conjugate();
		return result;
	}

	DualQuaternion operator * (const DualQuaternion& other)const
	{
		DualQuaternion r;
		r.dq[0] = dq[0] * other.dq[0];
		r.dq[1] = dq[0] * other.dq[1] + dq[1] * other.dq[0];
		return r;
	}

	DualQuaternion& operator *= (const DualQuaternion& other)
	{
		*this = (*this) * other;
		return *this;
	}

	DualQuaternion operator * (DataType v)const
	{
		DualQuaternion r;
		r.dq[0] = dq[0] * v;
		r.dq[1] = dq[1] * v;
		return r;
	}

	DualQuaternion& operator *= (DataType v)
	{
		dq[0] *= v;
		dq[1] *= v;
		return *this;
	}

	DualQuaternion operator / (DataType v)const
	{
		DualQuaternion r;
		r.dq[0] = dq[0] / v;
		r.dq[1] = dq[1] / v;
		return r;
	}

	DualQuaternion& operator /= (DataType v)
	{
		dq[0] /= v;
		dq[1] /= v;
		return *this;
	}

	DualQuaternion operator + (const DualQuaternion& rhs)const
	{
		DualQuaternion r;
		r.dq[0] = dq[0] + rhs.dq[0];
		r.dq[1] = dq[1] + rhs.dq[1];
		return r;
	}

	DualQuaternion& operator += (const DualQuaternion& rhs)
	{
		dq[0] += rhs.dq[0];
		dq[1] += rhs.dq[1];
		return *this;
	}

	DualQuaternion operator - (const DualQuaternion& rhs)const
	{
		DualQuaternion r;
		r.dq[0] = dq[0] - rhs.dq[0];
		r.dq[1] = dq[1] - rhs.dq[1];
		return r;
	}

	DualQuaternion operator - ()const
	{
		DualQuaternion r;
		r.dq[0] = -dq[0];
		r.dq[1] = -dq[1];
		return r;
	}

	DualQuaternion& operator -= (const DualQuaternion& rhs)
	{
		dq[0] -= rhs.dq[0];
		dq[1] -= rhs.dq[1];
		return *this;
	}

	// input: unit quaternion 'q0', translation vector 't' 
	// output: unit dual quaternion 'dq'
	inline void setFromQuatTrans(Quaternion<DataType> q0, ldp::ldp_basic_vec3<DataType> t)
	{
		// non-dual part (just copy q0):
		dq[0] = q0;

		// dual part: 0.5 * <t,q0>
		dq[1][0] = -0.5*(t[0] * q0[1] + t[1] * q0[2] + t[2] * q0[3]);
		dq[1][1] = 0.5*(t[0] * q0[0] + t[1] * q0[3] - t[2] * q0[2]);
		dq[1][2] = 0.5*(-t[0] * q0[3] + t[1] * q0[0] + t[2] * q0[1]);
		dq[1][3] = 0.5*(t[0] * q0[2] - t[1] * q0[1] + t[2] * q0[0]);
	}

	// input: dual quat. 'dq' with non-zero non-dual part
	// output: unit quaternion 'q0', translation vector 't'
	inline void getQuatTrans(Quaternion<DataType>& q0, ldp::ldp_basic_vec3<DataType>& t)const
	{
		DataType len2 = dq[0].dot(dq[0]);
		DataType len = sqrt(len2);

		if (len > std::numeric_limits<DataType>::epsilon())
			for (int i = 0; i<4; i++) q0[i] = dq[0][i] / len;
		else
		{
			q0[0] = 1;
			q0[1] = 0;
			q0[2] = 0;
			q0[3] = 0;
		}

		// dual part: 2*q_e*q_0^(-1)
		t[0] = 2.0*(-dq[1][0] * dq[0][1] + dq[1][1] * dq[0][0] - dq[1][2] * dq[0][3] + dq[1][3] * dq[0][2]) / len2;
		t[1] = 2.0*(-dq[1][0] * dq[0][2] + dq[1][1] * dq[0][3] + dq[1][2] * dq[0][0] - dq[1][3] * dq[0][1]) / len2;
		t[2] = 2.0*(-dq[1][0] * dq[0][3] - dq[1][1] * dq[0][2] + dq[1][2] * dq[0][1] + dq[1][3] * dq[0][0]) / len2;
	}

	inline void getQuatTransNoNormalize(Quaternion<DataType>& q0, ldp::ldp_basic_vec3<DataType>& t)const
	{
		for (int i = 0; i<4; i++) q0[i] = dq[0][i];

		// dual part: 2*q_e*q_0^(-1)
		t[0] = 2.0*(-dq[1][0] * dq[0][1] + dq[1][1] * dq[0][0] - dq[1][2] * dq[0][3] + dq[1][3] * dq[0][2]);
		t[1] = 2.0*(-dq[1][0] * dq[0][2] + dq[1][1] * dq[0][3] + dq[1][2] * dq[0][0] - dq[1][3] * dq[0][1]);
		t[2] = 2.0*(-dq[1][0] * dq[0][3] - dq[1][1] * dq[0][2] + dq[1][2] * dq[0][1] + dq[1][3] * dq[0][0]);
	}

	// M must be a rotation matrix plused with a translation
	// or the result is not guaranteed
	inline void setFromTransform(const ldp::ldp_basic_mat4<DataType>& M)
	{
		Quaternion<DataType> q;
		q.fromRotationMatrix(M.getRotationPart());
		ldp::ldp_basic_vec3<DataType> t(M(0, 3), M(1, 3), M(2, 3));
		setFromQuatTrans(q, t);
	}

	inline void getTransform(ldp::ldp_basic_mat4<DataType>& M)const
	{
		Quaternion<DataType> q;
		ldp::ldp_basic_vec3<DataType> t;
		getQuatTrans(q, t);
		M = q.toRotationMatrix();
		M(0, 3) = t[0];
		M(1, 3) = t[1];
		M(2, 3) = t[2];
		M(3, 3) = 1;
	}

	inline void getTransformNoNormalize(ldp::ldp_basic_mat4<DataType>& M)const
	{
		Quaternion<DataType> q;
		ldp::ldp_basic_vec3<DataType> t;
		getQuatTransNoNormalize(q, t);
		M = q.toRotationMatrix();
		M(0, 3) = t[0];
		M(1, 3) = t[1];
		M(2, 3) = t[2];
		M(3, 3) = 1;
	}
};

template<class DataType>
DualQuaternion<DataType> operator * (DataType v, const DualQuaternion<DataType>& rhs)
{
	return rhs * v;
}

typedef DualQuaternion<float> DualQuaternionF;
typedef DualQuaternion<double> DualQuaternionD;


/**
* Assert R is a rotation matrix, then this method calcualtes the log
* */
template<class T>
inline ldp_basic_mat3<T> transform_matrix_log(const ldp_basic_mat3<T>& R)
{
	Quaternion<T> q;
	q.fromRotationMatrix(R);
	T theta;
	ldp_basic_vec3<T> a;
	q.toAngleAxis(a, theta);

	// quaternion returns theta \in [0, 2*pi], 
	// here we favors [-pi, pi]
	if (theta > T(ldp::PI_D))
		theta -= T(ldp::PI_D * 2);

	ldp_basic_mat3<T> M;
	M(0, 0) = 0;			M(0, 1) = -theta*a[2];	M(0, 2) = theta*a[1];
	M(1, 0) = theta*a[2];	M(1, 1) = 0;			M(1, 2) = -theta*a[0];
	M(2, 0) = -theta*a[1];	M(2, 1) = theta*a[0];	M(2, 2) = 0;
	return M;
}
/**
* Assert A is a log rotation matrix, then this method calcualtes the exp
* */
template<class T>
inline ldp_basic_mat3<T> transform_matrix_exp(const ldp_basic_mat3<T>& A)
{
	ldp_basic_mat3<T> R;
	ldp::ldp_basic_vec3<T> a(-A(1, 2), A(0, 2), -A(0, 1));
	T theta = a.length();
	if (theta < std::numeric_limits<T>::epsilon())
	{
		R.eye();
	}
	else
	{
		a /= theta;
		R = Quaternion<T>().fromAngleAxis(theta, a).toRotationMatrix3();
	}

	return R;
}

/**
* Assert R is a rotation + translation matrix, then this method calcualtes the log
* */
template<class T>
inline ldp_basic_mat4<T> transform_matrix_log(const ldp_basic_mat4<T>& R)
{
	// firstly, calculate the rotation part
	Quaternion<T> q;
	q.fromRotationMatrix(R);
	T theta;
	ldp_basic_vec3<T> a;
	q.toAngleAxis(a, theta);

	// quaternion returns theta \in [0, 2*pi], 
	// here we favors [-pi, pi]
	if (theta > T(ldp::PI_D))
		theta -= T(ldp::PI_D * 2);

	ldp_basic_mat4<T> M;
	M(0, 0) = 0;			M(0, 1) = -theta*a[2];	M(0, 2) = theta*a[1];
	M(1, 0) = theta*a[2];	M(1, 1) = 0;			M(1, 2) = -theta*a[0];
	M(2, 0) = -theta*a[1];	M(2, 1) = theta*a[0];	M(2, 2) = 0;
	M(3, 0) = 0;			M(3, 1) = 0;			M(3, 2) = 0;
	ldp_basic_mat3<T> W;
	W(0, 0) = 0;		W(0, 1) = -a[2];	W(0, 2) = a[1];
	W(1, 0) = a[2];		W(1, 1) = 0;		W(1, 2) = -a[0];
	W(2, 0) = -a[1];	W(2, 1) = a[0];		W(2, 2) = 0;

	// secondly, calculate the translation part
	ldp_basic_vec3<T> t(R(0, 3), R(1, 3), R(2, 3));

	if (theta < std::numeric_limits<T>::epsilon())
	{
		M(0, 3) = t[0];
		M(1, 3) = t[1];
		M(2, 3) = t[2];
		M(3, 3) = 0;
	}
	else
	{
		ldp_basic_mat3<T> A;
		for (int iy = 0; iy < 3; iy++)
		for (int ix = 0; ix < 3; ix++)
			A(iy, ix) = T(iy == ix) - R(iy, ix);
		A *= W;
		for (int iy = 0; iy < 3; iy++)
		for (int ix = 0; ix < 3; ix++)
			A(iy, ix) += a[iy] * a[ix] * theta;
		ldp_basic_vec3<T> v = A.inv() * t;
		M(0, 3) = v[0] * theta;
		M(1, 3) = v[1] * theta;
		M(2, 3) = v[2] * theta;
		M(3, 3) = 0;
	}
	return M;
}
/**
* Assert A is a log rotation matrix, then this method calcualtes the exp
* */
template<class T>
inline ldp_basic_mat4<T> transform_matrix_exp(const ldp_basic_mat4<T>& A)
{
	ldp_basic_mat4<T> R;
	ldp_basic_vec3<T> a(-A(1, 2), A(0, 2), -A(0, 1));
	ldp_basic_vec3<T> v(A(0, 3), A(1, 3), A(2, 3));
	T theta = a.length();
	if (theta < std::numeric_limits<T>::epsilon())
	{
		R.eye();
		R(0, 3) = A(0, 3);
		R(1, 3) = A(1, 3);
		R(2, 3) = A(2, 3);
	}
	else
	{
		a /= theta;
		R = Quaternion<T>().fromAngleAxis(theta, a).toRotationMatrix();

		ldp_basic_mat3<T> M;
		for (int iy = 0; iy < 3; iy++)
		for (int ix = 0; ix < 3; ix++)
			M(iy, ix) = T(iy == ix) - R(iy, ix);
		ldp_basic_vec3<T> t = M * a.cross(v) + a.dot(v)*theta*a;
		R(0, 3) = t[0] / theta;
		R(1, 3) = t[1] / theta;
		R(2, 3) = t[2] / theta;
	}

	return R;
}

/**
* I/O
* */
template<typename T>
inline std::ostream& operator<<(std::ostream& out, const Quaternion<T>& v)
{
	for (size_t i = 0; i<3; i++)
		out << std::left << std::setw(10) << v[i] << " ";
	out << std::left << std::setw(10) << v[3];
	return out;
}

template<typename T>
inline std::istream& operator>>(std::istream& in, const Quaternion<T>& v)
{
	for (size_t i = 0; i<4; i++)
		in >> v[i];
	return in;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const DualQuaternion<T>& v)
{
	out << std::left << std::setw(10) << v.dq[0] << ", " << v.dq[1];
	return out;
}

template<typename T>
inline std::istream& operator>>(std::istream& in, const DualQuaternion<T>& v)
{
	in >> v.dq[0] >> v.dq[1];
	return in;
}

}//namespace ldp

#endif/*define quaternion*/