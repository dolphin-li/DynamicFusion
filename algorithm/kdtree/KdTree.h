#pragma once

#include <algorithm>
#include "common.h"
#include "AABB.h"
#include <vector>

namespace ldp
{
	namespace kdtree
	{

		//Weights for the SAH evaluation
#define KT 1.0F
#define KI 10.0F

		namespace kdtree_build_internal
		{

			//It is basically the same structure which was used in 
			//BVH, but the index of each traingle is stored to lead 
			//the construction of the tree.
			//A stack simulates the recursion process.
			struct BuildStateStruct
			{
				//This vector contains an index with the triangles.
				std::vector<uint32_t> triangles;
				AABB volume;
				uint32_t nodeIndex;
				int depth;
			};

			struct TravElem
			{
				uint32_t nodeIndex;
				real t_near;
				real t_far;
			};

			//This is some kind of functor which allow sorting the nodes of the tree.
			struct SortElem
			{
				real value;
				//The type of node.
				int type;

				SortElem()
				{
					value = FLT_MAX;
					type = -1;
				}

				SortElem(real val, int typ)
				{
					value = val;
					type = typ;
				}
			};

			//This is some kind of functor which allow sorting the nodes of the tree.
			struct SortElem1
			{
				real value;
				uint32_t id_type_dim;

				SortElem1()
				{
					value = FLT_MAX;
					id_type_dim = 0xffffffff;

				}

				SortElem1(real val, int typ, int dim, int triId)
				{
					value = val;
					setTriId(triId);
					setType(typ);
					setDim(dim);
				}

				FINLINE int getDim()const{ return int(id_type_dim & 3); }
				FINLINE int getType()const{ return int((id_type_dim >> 2) & 3); }
				FINLINE int getTriId()const{ return int(id_type_dim >> 4); }
				FINLINE void setDim(int d){ id_type_dim = (id_type_dim & 0xfffffffc) + d; }
				FINLINE void setType(int t){ id_type_dim = (id_type_dim & 0xfffffff3) + (t << 2); }
				FINLINE void setTriId(int t){ id_type_dim = getDim() + getType() + (t << 4); }
			};


			struct BuildStateStruct1
			{
				//This vector contains an index with the triangles.
				std::vector<SortElem1*> pEvents[3];
				std::vector<uint32_t> triangleIds;
				AABB volume;
				uint32_t nodeIndex;
				int depth;
			};
		}

		class Primitive;
		//Extended KD-Tree, which is capable of selecting the "best" splitting plane, 
		//using the SAH (Surface Area Heuristic).
		class SAHKDTree
		{
		private:

			///Info. returned by the SA heuristic.
			struct SAHReturn
			{
				//Axis of the split.
				int dimension;
				//Split position.
				real value;
				//Cost.
				real cost;
				//Decision variable. It decides whether the triangles are placed to 
				//the left or right branch.
				bool left;
			};

			///A node in the kd-tree. Similar to the definition used in BVH.
			struct Node
			{
				const static uint32_t NODE_TYPE_MASK = 0x80000000;
				const static uint32_t NODE_DIM_NUM_BITS = 2;
				const static uint32_t NODE_DIM_MASK = 0x00000003;
				const static uint32_t NODE_DIM_MASK_INV = 0xffffffff - NODE_DIM_MASK;
				const static uint32_t INDEX_MASK = NODE_TYPE_MASK - 1 - NODE_DIM_MASK;
				const static uint32_t NODE_DIM_INVALID = 3;

				real splitVal;
				uint32_t dataIndex_dim;

				FINLINE int getDimension()const{ return int(dataIndex_dim & NODE_DIM_MASK); }
				FINLINE void setDimension(int d)
				{
					dataIndex_dim = ((dataIndex_dim & NODE_DIM_MASK_INV) | d);
				}

				FINLINE bool isLeaf() const
				{
					return (dataIndex_dim & NODE_TYPE_MASK) != 0;
				}

				FINLINE uint32_t getLeftChildOrLeaf() const
				{
					return (dataIndex_dim & INDEX_MASK) >> NODE_DIM_NUM_BITS;
				}

				FINLINE void setDataIndexAndLeafMask(uint32_t idx, bool isLeaf)
				{
					int dim = getDimension();
					idx = (idx << NODE_DIM_NUM_BITS);
					assert(idx < NODE_TYPE_MASK);
					if (isLeaf) idx |= NODE_TYPE_MASK;
					dataIndex_dim = (idx | dim);
				}

				Node(int dim, real value)
				{
					setDimension(dim);
					splitVal = value;
				}

				Node()
				{
					setDimension(NODE_DIM_INVALID);
					splitVal = FLT_MAX;
				}

			};

			///Scene bounding box (or the objects loaded so far).
			AABB m_sceneBox;

			/// Same data structures as in BVH.
			std::vector<Node> m_nodes;
			std::vector<Primitive*> m_leafData;

			/// SAH cost function. It returns the cost of the surface area.
			FINLINE real C(real Pl, real Pr, int Nl, int Nr)
			{
				real b = real((Nl == 0 || Nr == 0));
				return (1.f - b*0.2f) * (KT + KI * (Pl * Nl + Pr * Nr));
			}

			/// Main method to evaluate the costs.
			/// It returns the value of the heuristic for the given plane., 
			/// if the tringles lying exactly in the plane are put on the left side, 
			/// otherwise the are on the right side.
			FINLINE std::pair<real, bool>
				SAH(int planeDim, real planeValue, AABB volume, int Nl, int Nr, int Np)
			{
					real invTotalArea = real(1) / volume.getSurfaceArea();
					//The  current volume is splitted into two parts: right and left side.
					//The left side is assigned to the current volume, while the right side 
					//to a new volume.
					AABB right = volume.split(planeDim, planeValue);
					real Pl = volume.getSurfaceArea() * invTotalArea;
					real Pr = right.getSurfaceArea() * invTotalArea;
					//The cost function enables us to decide whether to put the triangle.
					real Cl = C(Pl, Pr, Nl + Np, Nr);
					real Cr = C(Pl, Pr, Nl, Np + Nr);
					std::pair<real, bool> ret;
					ret.second = (Cl < Cr);
					real mul = real(ret.second);
					ret.first = Cl * mul + Cr * (1 - mul);
					return ret;
					//if (Cl < Cr)
					//{
					//	ret.first = Cl;
					//	ret.second = true;
					//	return ret;
					//}
					//else
					//{
					//	ret.first = Cr;
					//	ret.second = false;
					//	return ret;
					//}
				}

			/// This method allow finding the "best" splitting plane.
			SAHReturn findPlane(std::vector<uint32_t>* triangles, AABB volume,
				std::vector<AABB> &triBoxes);

			void fastBuild(std::vector<Primitive> &_objects);
			SAHReturn fastFindPlane(std::vector<kdtree_build_internal::SortElem1*> pEvents[3],
				std::vector<uint32_t>& triangleIds, AABB volume,
				std::vector<AABB> &triBoxes);

		public:

			/// Builds the hierarchy over a set of bounded primitives
			void build(std::vector<Primitive> &_objects);

			/// Intersects a ray with the SAH KdTree.
			int intersect
				(const Ray &_ray, real& dist, Primitive*& _prim, int thread_id) const;

			bool collude(const Ray &_ray, real& _dist, Primitive* self, int thread_id)const;

			/// It returns the the bounding box for the root of the SAH Kd Tree.
			AABB getSceneBBox() const
			{
				return m_sceneBox;
			};
		};
	}
}