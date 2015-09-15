/*
This file is part of Mitsuba, a physically based rendering system.

Copyright (c) 2007-2012 by Wenzel Jakob and others.

Mitsuba is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License Version 3
as published by the Free Software Foundation.

Mitsuba is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "common.h"

namespace ldp
{

	namespace kdtree
	{
		inline bool solveQuadratic(real a, real b, real c, real &x0, real &x1) {
			/* Linear case */
			if (a == 0) {
				if (b != 0) {
					x0 = x1 = -c / b;
					return true;
				}
				return false;
			}

			real discrim = b*b - 4.0f*a*c;

			/* Leave if there is no solution */
			if (discrim < 0)
				return false;

			real temp, sqrtDiscrim = std::sqrt(discrim);

			/* Numerically stable version of (-b (+/-) sqrtDiscrim) / (2 * a)
			*
			* Based on the observation that one solution is always
			* accurate while the other is not. Finds the solution of
			* greater magnitude which does not suffer from loss of
			* precision and then uses the identity x1 * x2 = c / a
			*/
			if (b < 0)
				temp = -0.5f * (b - sqrtDiscrim);
			else
				temp = -0.5f * (b + sqrtDiscrim);

			x0 = temp / a;
			x1 = c / temp;

			/* Return the results so that x0 < x1 */
			if (x0 > x1)
				std::swap(x0, x1);

			return true;
		}


		/** \brief Simple n-dimensional ray data structure with
		* minimum / maximum extent information.
		*
		* The somewhat peculiar ordering of the attributes is due
		* to alignment purposes and should not be changed.
		*
		* \ingroup libcore
		* \ingroup libpython
		*/
		template <typename _PointType, typename _VectorType> struct TRay {
			typedef _PointType                  PointType;
			typedef _VectorType                 VectorType;
			typedef typename real  Scalar;

			/* The somewhat peculiar ordering of the attributes is for
			alignment purposes in the 3D case and should not be changed. */

			PointType o;     ///< Ray origin
			Scalar mint;     ///< Minimum range for intersection tests
			VectorType d;    ///< Ray direction
			Scalar maxt;     ///< Maximum range for intersection tests
			VectorType dRcp; ///< Componentwise reciprocals of the ray direction
			real time;  ///< Time value associated with this ray

			/// Construct a new ray
			inline TRay() : mint(EPSILON),
				maxt(std::numeric_limits<Scalar>::infinity()), time(0) {
			}

			/// Copy constructor (1)
			inline TRay(const TRay &ray)
				: o(ray.o), mint(ray.mint), d(ray.d), maxt(ray.maxt),
				dRcp(ray.dRcp), time(ray.time) {
			}

			/// Copy constructor (2)
			inline TRay(const TRay &ray, Scalar mint, Scalar maxt)
				: o(ray.o), mint(mint), d(ray.d), maxt(maxt),
				dRcp(ray.dRcp), time(ray.time) { }

			/// Construct a new ray, while not specifying a direction yet
			inline TRay(const PointType &o, Scalar time) : o(o), mint(Epsilon),
				maxt(std::numeric_limits<Scalar>::infinity()), time(time) { }

			/// Construct a new ray
			inline TRay(const PointType &o, const VectorType &d, Scalar time)
				: o(o), mint(EPSILON), d(d),
				maxt(std::numeric_limits<Scalar>::infinity()), time(time) {

				for (int i = 0; i < 3; ++i)
					dRcp[i] = (Scalar)1 / d[i];

			}

			/// Construct a new ray
			inline TRay(const PointType &o, const VectorType &d)
				: TRay(o, d, Scalar(0))
				{
			}

			/// Construct a new ray
			inline TRay(const PointType &o, const VectorType &d, Scalar mint, Scalar maxt,
				Scalar time) : o(o), mint(mint), d(d), maxt(maxt), time(time) {

				for (int i = 0; i < 3; ++i)
					dRcp[i] = (Scalar)1 / d[i];

			}

			/// Set the origin
			inline void setOrigin(const PointType &pos) { o = pos; }

			/// Set the time
			inline void setTime(Scalar tval) { time = tval; }

			/// Set the direction and update the reciprocal
			inline void setDirection(const VectorType &dir) {
				d = dir;
				for (int i = 0; i < 3; ++i)
					dRcp[i] = (Scalar)1 / dir[i];
			}

			/**
			* \brief Return the position of a point along the ray
			*
			* \remark In the Python bindings, this operator is
			* exposed as a function named \c eval -- i.e.
			* position lookups should be written as \c ray.eval(t)
			*/
			inline PointType operator() (Scalar t) const { return o + t * d; }

			/// Return a string representation of this ray
			inline std::string toString() const {
				return "";
			}
		};


		typedef TRay<Point, Vec3>   Ray;
		/** \brief %Ray differential -- enhances the basic ray class with
		information about the rays of adjacent pixels on the view plane
		\ingroup libcore
		*/
		struct RayDifferential : public Ray {
			Point rxOrigin, ryOrigin;
			Vec3 rxDirection, ryDirection;
			bool hasDifferentials;

			inline RayDifferential()
				: hasDifferentials(false) {
			}

			inline RayDifferential(const Point &p, const Vec3 &d, real time)
				: Ray(p, d, time), hasDifferentials(false) {
			}

			inline explicit RayDifferential(const Ray &ray)
				: Ray(ray), hasDifferentials(false) {
			}

			inline RayDifferential(const RayDifferential &ray)
				: Ray(ray), rxOrigin(ray.rxOrigin), ryOrigin(ray.ryOrigin),
				rxDirection(ray.rxDirection), ryDirection(ray.ryDirection),
				hasDifferentials(ray.hasDifferentials) {
			}

			void scaleDifferential(real amount) {
				rxOrigin = o + (rxOrigin - o) * amount;
				ryOrigin = o + (ryOrigin - o) * amount;
				rxDirection = d + (rxDirection - d) * amount;
				ryDirection = d + (ryDirection - d) * amount;
			}

			inline void operator=(const RayDifferential &ray) {
				o = ray.o;
				mint = ray.mint;
				d = ray.d;
				maxt = ray.maxt;
#ifdef MTS_DEBUG_FP
				bool state = disableFPExceptions();
#endif
				dRcp = ray.dRcp;
#ifdef MTS_DEBUG_FP
				restoreFPExceptions(state);
#endif
				hasDifferentials = ray.hasDifferentials;
				rxOrigin = ray.rxOrigin;
				ryOrigin = ray.ryOrigin;
				rxDirection = ray.rxDirection;
				ryDirection = ray.ryDirection;
			}

			inline void operator=(const Ray &ray) {
				o = ray.o;
				mint = ray.mint;
				d = ray.d;
				maxt = ray.maxt;
#ifdef MTS_DEBUG_FP
				bool state = disableFPExceptions();
#endif
				dRcp = ray.dRcp;
#ifdef MTS_DEBUG_FP
				restoreFPExceptions(state);
#endif
				hasDifferentials = false;
			}

			/// Return a string representation of this ray
			inline std::string toString() const {
				return "";
			}
		};
	}// namespace mitsuba

}//namespace ldp
