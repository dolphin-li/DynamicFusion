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

#include "AABB.h"

namespace ldp
{
	namespace kdtree
	{


		Point AABB::getCorner(uint8_t corner) const {
			return Point(corner & 1 ? max[0] : min[0],
				corner & 2 ? max[1] : min[1],
				corner & 4 ? max[2] : min[2]);
		}


		bool AABB::overlaps(const BSphere &sphere) const {
			real distance = 0;
			for (int i = 0; i < 3; ++i) {
				if (sphere.center[i] < min[i]) {
					real d = sphere.center[i] - min[i];
					distance += d*d;
				}
				else if (sphere.center[i] > max[i]) {
					real d = sphere.center[i] - max[i];
					distance += d*d;
				}
			}
			return distance < sphere.radius*sphere.radius;
		}

		BSphere AABB::getBSphere() const {
			Point3 center = getCenter();
			return BSphere(center, (center - max).length());
		}

	}//namespace mitsuba

}// namespace ldp
