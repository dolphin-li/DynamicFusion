#pragma once

#include "ldpMat\ldp_basic_mat.h"
#include "ldpMat\Quaternion.h"

#include "camera\Camera.h"
#include <algorithm>

#undef min
#undef max
namespace ldp
{
	namespace kdtree
	{
		typedef float real;
		typedef ldp_basic_vec2<real> Vec2;
		typedef ldp_basic_vec3<real> Vec3;
		typedef ldp_basic_vec4<real> Vec4;
		typedef float Float;
		typedef Vec3 vector3;
		typedef Vec4  Color;
		typedef Vec3  Point;
		typedef Point Point3;
		typedef unsigned int Pixel;
		typedef unsigned char uint8_t;
		typedef unsigned int uint32_t;
		typedef unsigned short uint16_t;
		typedef unsigned long long uint64_t;
		typedef long long int64_t;

#define FINLINE __forceinline

#define SAMPLES			128
#define TRACEDEPTH		4
#define MAXTREEDEPTH	20
#define IMPORTANCE
#define TILESIZE		64
#define MAX_THREADS		8

		// Intersection method return values
#define HIT		 1		// Ray hit primitive
#define MISS	 0		// Ray missed primitive
#define INPRIM	-1		// Ray started inside primitive

#define MAXLIGHTS	10

#ifndef LDP_DEFINE_VECTOR_DOUBLE_PRECISION
#define EPSILON			0.0001f
#else
#define EPSILON			0.0000001f
#endif

#define DOT(A,B)		(A.dot(B))
#define NORMALIZE(A)	{A.normalizeLocal();}
#define LENGTH(A)		(A.length())
#define SQRLENGTH(A)	(A.sqrLength())
#define SQRDISTANCE(A,B) (A.sqrDist(B))

#define PI				3.141592653589793238462f

		static std::string fullNameToPath(const char* name)
		{
			// get file path
			std::string scenePath = name;
			int pos1 = scenePath.find_last_of("\\");
			if (!(pos1 >= 0 && pos1 < (int)scenePath.size()))
				pos1 = 0;
			int pos2 = scenePath.find_last_of("/");
			if (!(pos2 >= 0 && pos2 < (int)scenePath.size()))
				pos2 = 0;
			int pos = std::max(pos1, pos2);
			if (pos) pos++;
			scenePath = scenePath.substr(0, pos);
			return scenePath;
		}
	}
}