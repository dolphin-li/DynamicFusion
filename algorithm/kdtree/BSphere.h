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

#include "Ray.h"
#include <algorithm>

namespace ldp
{
	namespace kdtree
	{
#undef min
#undef max

		/** \brief Bounding sphere data structure in three dimensions
		*
		* \ingroup libcore
		* \ingroup libpython
		*/
		struct BSphere {
			Point center;
			real radius;

			/// Construct a bounding sphere at the origin having radius zero
			inline BSphere() : center(0.0f), radius(0.0f) { }

			/// Create a bounding sphere from a given center point and radius
			inline BSphere(const Point &center, real radius)
				: center(center), radius(radius) {
			}

			/// Copy constructor
			inline BSphere(const BSphere &boundingSphere)
				: center(boundingSphere.center), radius(boundingSphere.radius) {
			}

			/// Return whether this bounding sphere has a radius of zero or less.
			inline bool isEmpty() const {
				return radius <= 0.0f;
			}

			/// Expand the bounding sphere radius to contain another point.
			inline void expandBy(const Point p) {
				radius = std::max(radius, (p - center).length());
			}

			/// Check whether the specified point is inside or on the sphere
			inline bool contains(const Point p) const {
				return (p - center).length() <= radius;
			}

			/// Equality test
			inline bool operator==(const BSphere &boundingSphere) const {
				return center == boundingSphere.center && radius == boundingSphere.radius;
			}

			/// Inequality test
			inline bool operator!=(const BSphere &boundingSphere) const {
				return center != boundingSphere.center || radius != boundingSphere.radius;
			}

			/**
			* \brief Calculate the intersection points with the given ray
			* \return \c true if the ray intersects the bounding sphere
			*
			* \remark In the Python bindings, this function returns the
			* \c nearT and \c farT values as a tuple (or \c None, when no
			* intersection was found)
			*/
			inline bool rayIntersect(const Ray &ray, real &nearHit, real &farHit) const {
				Vec3 o = ray.o - center;
				real A = ray.d.sqrLength();
				real B = 2 * o.dot(ray.d);
				real C = o.sqrLength() - radius*radius;

				return solveQuadratic(A, B, C, nearHit, farHit);
			}

			/// Return a string representation of the bounding sphere
			inline std::string toString() const {
				return "";
			}
		};
	}// namespace mitsuba
}// namespace ldp
