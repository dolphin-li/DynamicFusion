#ifndef TOOL_BOX_DUAL_QUAT_CU_HPP__
#define TOOL_BOX_DUAL_QUAT_CU_HPP__

#include "dualquaternion/quat_cu.hpp"

// =============================================================================
namespace Tbx {
// =============================================================================

/** @class Dual_quat_cu
    @brief Representation of a dual quaternion to express rotation and translation

    A dual quaternion (DQ) is based on the algebra of dual numbers. Dual numbers
    are somewhat close to complex numbers (a + ib) as they are writen :
    nd + eps * d where nd is the non-dual part and d the dual part and
    (eps)^2=0.

    Dual quaternion are represented as follow : q0 + eps * qe where q0
    is the non-dual part (a quaternion) and qe the dual part (another quaternion)

    With dual quaternion we can express a rotation and a translation. This
    enables us to substitute rigid transformations matrices to dual quaternions
    and transform a point with the method 'transform()'

    As a dual quaternions is the sum of two quaternions we have to store eight
    coefficients corresponding to the two quaternions.

    To move a point with a rigid transformation (i.e. solely composed
    of a translation and a rotation) you just need to construct the DQ with a
    quaternion wich express the rotation and a translation vector. You can
    now translate and rotate the point at the same time with 'transform()'.

    Linear blending of dual quaternions (DLB) is possible (dq0*w0 + dq1*w1 ...)
    where w0, w1 ... wn are scalar weights whose sum equal one. The weights
    defines the influence of each transformations expressed by the dual
    quaternions dq0, dq1 ... dqn.
    N.B : this is often used to compute mesh deformation for animation systems.

    You can compute DLB with the overloaded operators (+) and (*) and use
    the method transform() of the resulting dual quaternion to deform a point
    according to the DLB.

    @note Article: "Geometric skinning with approximate dual quaternion blending"
 */
class __align__(16) Dual_quat_cu{
    public:
		__device__ __host__ static float epsilon(){ return 1e-6; }
    // -------------------------------------------------------------------------
    /// @name Constructors
    // -------------------------------------------------------------------------

    /// Default constructor generates a dual quaternion with no translation
    /// and no rotation either
	__device__ __host__ Dual_quat_cu()
    {
        Dual_quat_cu res = dual_quat_from(Quat_cu(), Vec3(0.f, 0.f, 0.f));
        *this = res;
    }


    /// Fill directly the dual quaternion with two quaternion for the non-dual
    /// and dual part
	__device__ __host__ Dual_quat_cu(const Quat_cu& q0, const Quat_cu& qe)
    {
        _quat_0 = q0;
        _quat_e = qe;
    }

    /// Construct a dual quaternion with a quaternion 'q' which express the
    /// rotation and a translation vector
	__device__ __host__ Dual_quat_cu(const Quat_cu& q, const Vec3& t)
    {
        Dual_quat_cu res = dual_quat_from(q, t);
        *this = res;
    }

    /// Construct from rigid transformation 't'
	__device__ __host__  Dual_quat_cu(const Transfo& t)
    {
        Quat_cu q(t);
        Vec3 translation(t.m[3], t.m[7], t.m[11]);
        Dual_quat_cu res = dual_quat_from(q, translation);
        *this = res;
    }


    // -------------------------------------------------------------------------
    /// @name Methods
    // -------------------------------------------------------------------------

	__device__ __host__ __forceinline__ float norm()const
	{
		return _quat_0.norm();
	}

	__device__ __host__ __forceinline__ void normalize()
    {
        float norm = _quat_0.norm();
        _quat_0 = _quat_0 / norm;
        _quat_e = _quat_e / norm;
    }

	__device__ __host__ Dual_quat_cu conjugate()
	{
		return Dual_quat_cu(_quat_0.conjugate(), _quat_e.conjugate());
	}

    /// Transformation of point p with the dual quaternion
	/// NOTE: perform normalize() before this if nedded
	__device__ __host__ Point3 transform(const Point3& p) const
    {
#if 0
        // As the dual quaternions may be the results from a
        // linear blending we have to normalize it :
        float norm = _quat_0.norm();
        Quat_cu qblend_0 = _quat_0 / norm;
        Quat_cu qblend_e = _quat_e / norm;

        // Translation from the normalized dual quaternion equals :
        // 2.f * qblend_e * conjugate(qblend_0)
        Vec3 v0 = qblend_0.get_vec_part();
        Vec3 ve = qblend_e.get_vec_part();
        Vec3 trans = (ve*qblend_0.w() - v0*qblend_e.w() + v0.cross(ve)) * 2.f;

        // Rotate
        return qblend_0.rotate(p) + trans;
#else
		// Translation from the normalized dual quaternion equals :
		// 2.f * qblend_e * conjugate(qblend_0)
		Vec3 v0 = _quat_0.get_vec_part();
		Vec3 ve = _quat_e.get_vec_part();
		Vec3 trans = (ve*_quat_0.w() - v0*_quat_e.w() + v0.cross(ve)) * 2.f;

		// Rotate
		return _quat_0.rotate(p) + trans;
#endif
    }

    /// Rotate a vector with the dual quaternion
	__device__ __host__ Vec3 rotate(const Vec3& v) const
    {
#if 0
        Quat_cu tmp = _quat_0;
        tmp.normalize();
        return tmp.rotate(v);
#else
		return _quat_0.rotate(v);
#endif
    }

	__device__ __host__ static Dual_quat_cu dual_quat_from(const Quat_cu& q, const Vec3& t)
    {
        float w = -0.5f*( t.x * q.i() + t.y * q.j() + t.z * q.k());
        float i =  0.5f*( t.x * q.w() + t.y * q.k() - t.z * q.j());
        float j =  0.5f*(-t.x * q.k() + t.y * q.w() + t.z * q.i());
        float k =  0.5f*( t.x * q.j() - t.y * q.i() + t.z * q.w());

        return Dual_quat_cu(q, Quat_cu(w, i, j, k));
    }

    /// Convert the dual quaternion to a homogenous matrix
    /// N.B: Dual quaternion is normalized before conversion
	__device__ __host__ Transfo to_transformation()const
    {
        Vec3 t;
        float norm = _quat_0.norm();
		float norm2 = norm*norm;

        // Rotation matrix from non-dual quaternion part
        Mat3 m = (_quat_0 / norm).to_matrix3();

        // translation vector from dual quaternion part:
        t.x = 2.f*(-_quat_e.w()*_quat_0.i() + _quat_e.i()*_quat_0.w() - _quat_e.j()*_quat_0.k() + _quat_e.k()*_quat_0.j()) / norm2;
        t.y = 2.f*(-_quat_e.w()*_quat_0.j() + _quat_e.i()*_quat_0.k() + _quat_e.j()*_quat_0.w() - _quat_e.k()*_quat_0.i()) / norm2;
        t.z = 2.f*(-_quat_e.w()*_quat_0.k() - _quat_e.i()*_quat_0.j() + _quat_e.j()*_quat_0.i() + _quat_e.k()*_quat_0.w()) / norm2;

        return Transfo(m, t);
    }

	__device__ __host__ Transfo to_transformation_after_normalize()const
	{
		Vec3 t;
		// Rotation matrix from non-dual quaternion part
		Mat3 m = _quat_0.to_matrix3();

		// translation vector from dual quaternion part:
		t.x = 2.f*(-_quat_e.w()*_quat_0.i() + _quat_e.i()*_quat_0.w() - _quat_e.j()*_quat_0.k() + _quat_e.k()*_quat_0.j());
		t.y = 2.f*(-_quat_e.w()*_quat_0.j() + _quat_e.i()*_quat_0.k() + _quat_e.j()*_quat_0.w() - _quat_e.k()*_quat_0.i());
		t.z = 2.f*(-_quat_e.w()*_quat_0.k() - _quat_e.i()*_quat_0.j() + _quat_e.j()*_quat_0.i() + _quat_e.k()*_quat_0.w());

		return Transfo(m, t);
	}

    // -------------------------------------------------------------------------
    /// @name Operators
    // -------------------------------------------------------------------------
	__device__ __host__ float& operator[](int i)
	{
#if 0
		if (i < 4)
			return _quat_0[i];
		else
			return _quat_e[i-4];
#else
		return ((float*)this)[i];
#endif
	}
	__device__ __host__ const float& operator[](int i)const
	{
#if 0
		if (i < 4)
			return _quat_0[i];
		else
			return _quat_e[i - 4];
#else
		return ((float*)this)[i];
#endif
	}
	__device__ __host__ Dual_quat_cu operator+(const Dual_quat_cu& dq) const
    {
        return Dual_quat_cu(_quat_0 + dq._quat_0, _quat_e + dq._quat_e);
    }
	__device__ __host__ Dual_quat_cu& operator+=(const Dual_quat_cu& dq)
	{
		_quat_0 += dq._quat_0;
		_quat_e += dq._quat_e;
		return *this;
	}

	__device__ __host__ Dual_quat_cu operator-(const Dual_quat_cu& dq) const
	{
		return Dual_quat_cu(_quat_0 - dq._quat_0, _quat_e - dq._quat_e);
	}
	__device__ __host__ Dual_quat_cu& operator-=(const Dual_quat_cu& dq)
	{
		_quat_0 -= dq._quat_0;
		_quat_e -= dq._quat_e;
		return *this;
	}

	__device__ __host__ Dual_quat_cu operator*(float scalar) const
    {
        return Dual_quat_cu(_quat_0 * scalar, _quat_e * scalar);
    }

	__device__ __host__ Dual_quat_cu& operator*=(float scalar)
	{
		_quat_0 *= scalar;
		_quat_e *= scalar;
		return *this;
	}

	__device__ __host__ Dual_quat_cu operator * (const Dual_quat_cu& other)const
	{
		Dual_quat_cu r;
		r._quat_0 = _quat_0 * other._quat_0;
		r._quat_e = _quat_0 * other._quat_e + _quat_e * other._quat_0;
		return r;
	}

    /// Return a dual quaternion with no translation and no rotation
	__device__ __host__ static Dual_quat_cu identity()
    {
        return Dual_quat_cu(Quat_cu(1.f, 0.f, 0.f, 0.f),
                            Vec3(0.f, 0.f, 0.f) );
    }

	// assume self-normalized.
	__device__ __host__ void to_twist(Vec3& r, Vec3& t)const
	{
		float sign = (_quat_0.w() > 0.f) - (_quat_0.w() < 0.f);
		float norm = acos(_quat_0.w()*sign);
		if (norm > epsilon())
		{
			float inv_sinNorm_norm = norm / sin(norm) * sign;
			r.x = _quat_0.i() * inv_sinNorm_norm;
			r.y = _quat_0.j() * inv_sinNorm_norm;
			r.z = _quat_0.k() * inv_sinNorm_norm;
		}
		else
			r.x = r.y = r.z = 0;

		t.x = 2.f*(-_quat_e.w()*_quat_0.i() + _quat_e.i()*_quat_0.w() - _quat_e.j()*_quat_0.k() + _quat_e.k()*_quat_0.j());
		t.y = 2.f*(-_quat_e.w()*_quat_0.j() + _quat_e.i()*_quat_0.k() + _quat_e.j()*_quat_0.w() - _quat_e.k()*_quat_0.i());
		t.z = 2.f*(-_quat_e.w()*_quat_0.k() - _quat_e.i()*_quat_0.j() + _quat_e.j()*_quat_0.i() + _quat_e.k()*_quat_0.w());
	}

	__device__ __host__ void from_twist(Vec3 r, Vec3 t)
	{
		float norm = r.norm();
		if (norm > epsilon())
		{
			float cosNorm = cos(norm);
			float sign = (cosNorm > 0.f) - (cosNorm < 0.f);
			cosNorm *= sign;
			float sinNorm_norm = sign * sin(norm) / norm;
			_quat_0 = Quat_cu(cosNorm, r.x*sinNorm_norm, r.y*sinNorm_norm, r.z*sinNorm_norm);
		}
		else
			_quat_0 = Quat_cu();
		*this = dual_quat_from(_quat_0, t);
	}

    // -------------------------------------------------------------------------
    /// @name Getters
    // -------------------------------------------------------------------------

	__device__ __host__ Quat_cu get_dual_part() const { return _quat_e; }

	__device__ __host__ Quat_cu get_non_dual_part() const { return _quat_0; }

	__device__ __host__ Quat_cu translation() const { return _quat_e; }

	__device__ __host__ Quat_cu rotation() const { return _quat_0; }

	__device__ __host__ void set_rotation(const Quat_cu& q){ _quat_0 = q; }

    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------

private:
    /// Non-dual part of the dual quaternion. It also represent the rotation.
    /// @warning If you want to compute the rotation with this don't forget
    /// to normalize the quaternion as it might be the result of a
    /// dual quaternion linear blending
    /// (when overloaded operators like '+' or '*' are used)
    Quat_cu _quat_0;

    /// Dual part of the dual quaternion which represent the translation.
    /// translation can be extracted by computing
    /// 2.f * _quat_e * conjugate(_quat_0)
    /// @warning don't forget to normalize quat_0 and quat_e :
    /// quat_0 = quat_0 / || quat_0 || and quat_e = quat_e / || quat_0 ||
    Quat_cu _quat_e;
};

}// END Tbx NAMESPACE ==========================================================

#endif // TOOL_BOX_DUAL_QUAT_CU_HPP__
