#include "kdtree.h"
#include <vector>
#include <stack>
#include "Primitive.h"
#include <map>
namespace ldp
{
	namespace kdtree
	{

		const real _EPS = 0.000001f;
		//Hard code: Maximal depth used to build the tree.
		const int MAX_DEPTH = 10;

		using namespace kdtree_build_internal;

		//It checkss whether the node e1 is less than, equal to or greater than e2.
		//If both are equal, then the decision is made based on the type of node.
		inline bool compE(SortElem e1, SortElem e2)
		{
			return (e1.value < e2.value) || ((e1.value == e2.value) && (e1.type < e2.type));
		}
		inline bool compE1(SortElem1 e1, SortElem1 e2)
		{
			return (e1.value < e2.value) || ((e1.value == e2.value) && (e1.getType() < e2.getType()));
		}

		//An iterative split by SAH build for KD-trees
		void SAHKDTree::build(std::vector<Primitive> &_objects)
		{
			return fastBuild(_objects);
			//memory
			m_nodes.clear();
			m_leafData.clear();
			m_nodes.reserve(_objects.size() * 2);
			m_leafData.reserve(_objects.size());

			//Bounding boxes for all objects to store, no order given.
			std::vector<AABB> objectBBoxes(_objects.size());
			BuildStateStruct curState;
			m_sceneBox = AABB(real(0));

			//initializing bounding boxes and bentroid boxes
			for (uint32_t i = 0; i < objectBBoxes.size(); i++)
			{
				objectBBoxes[i] = _objects[i].getAABB();

				//The bounding box of the original scene is computed
				m_sceneBox.expandBy(objectBBoxes[i]);

				curState.triangles.push_back(i);
			}

			//initializing state for stack simulated recursion
			curState.volume = m_sceneBox;
			curState.nodeIndex = 0;
			curState.depth = 0;
			//initialize the tree
			m_nodes.resize(1);

			//creating recursion stack
			std::stack<BuildStateStruct> buildStack;

			//loop till stack is empty
			for (;;)
			{

				//Number of object in the current segment.
				uint32_t objCnt = curState.triangles.size();

				//Diagonal of the volume.
				Vec3 boxDiag = curState.volume.getExtents();

				//Splitting plane.
				SAHKDTree::SAHReturn plane =
					findPlane(&curState.triangles, curState.volume, objectBBoxes);

				int splitDim = plane.dimension;
				real splitVal = plane.value;

				//Boundary conditions. We don't keep splitting if the MAX_DEPTH is 
				//reached or if the plane is null. Therefore, a new leaf is created.
				if (plane.cost > KI * curState.triangles.size() ||
					curState.depth > MAX_DEPTH)
				{
					//The new node is initialized, data index determines if the leaf 
					//is a node by some bit operations
					//m_nodes[curState.nodeIndex].dataIndex =
					//	m_leafData.size() | Node::NODE_TYPE_MASK;
					m_nodes[curState.nodeIndex].setDataIndexAndLeafMask(
						m_leafData.size(), true);

					for (int i = 0; i < (int)curState.triangles.size(); i++)
					{
						m_leafData.push_back(&_objects[curState.triangles[i]]);
					}
					curState.triangles.clear();

					//There is no more information after reaching the leaf.
					m_leafData.push_back(NULL);

					if (buildStack.empty())
						break;

					//The stack is updated.
					curState = buildStack.top();
					buildStack.pop();

					continue;
				}

				//A new state for the right child is created.
				BuildStateStruct rightState;
				//The volumen is splitted in two part. The right is assigned to this 
				//variable.
				rightState.volume = curState.volume.split(splitDim, splitVal);
				std::vector<uint32_t> leftTris;

				++curState.depth;
				rightState.depth = curState.depth;

				for (int i = 0; i < (int)curState.triangles.size(); i++)
				{
					uint32_t triIndex = curState.triangles[i];
					AABB triBox = objectBBoxes[triIndex];
					if (triBox.min[splitDim] == splitVal &&
						triBox.min[splitDim] == triBox.max[splitDim])
					{
						//triangle is planar and lies in the spliting plane
						//assign it to the side, the SAH determined
						if (plane.left)
						{
							leftTris.push_back(triIndex);
						}
						else
						{
							rightState.triangles.push_back(triIndex);
						}
						continue;
					}
					else
					{
						//It assigned the triangle to the correct side, based on the 
						//splitting value.
						if (triBox.max[splitDim] > splitVal)
						{
							rightState.triangles.push_back(triIndex);
						}
						if (triBox.min[splitDim] < splitVal)
						{
							leftTris.push_back(triIndex);
						}
					}
				}

				curState.triangles.swap(leftTris);

				//Data index is the the number of nodes added before
				m_nodes[curState.nodeIndex].setDataIndexAndLeafMask(m_nodes.size(), false);
				m_nodes[curState.nodeIndex].setDimension(splitDim);
				m_nodes[curState.nodeIndex].splitVal = splitVal;

				//The left child is stored at the end of m_nodes.
				curState.nodeIndex = m_nodes.size();
				//The right child is stored on position after the left child.
				rightState.nodeIndex = curState.nodeIndex + 1;

				//The right child is pushed on the stack, we go on with the
				//left child in the next loop iteration.
				buildStack.push(rightState);

				//m_nodes is extended so that it can hold the left and right child.
				m_nodes.resize(rightState.nodeIndex + 1);
			}
		}


		struct IntRet
		{
			real distance;
			int  transObjCounter;
		};

		//Intersection method for a kd-tree.
		int SAHKDTree::intersect
			(const Ray &_ray, real &_dist, Primitive*& _prim, int thread_id) const
		{
			int returnVal = MISS;

			std::pair<real, real> t;
			m_sceneBox.rayIntersect(_ray, t.first, t.second);
			t.first = std::max(real(0), t.first);
			t.second = std::min(t.second, _dist);
			if (t.first > t.second)
				return false;

			std::stack<TravElem> traverseStack;

			//Root of the tree. Starting point for the search.
			TravElem curNode;
			curNode.nodeIndex = 0;
			curNode.t_near = t.first;
			curNode.t_far = t.second;

			//Inverse direction of ray is precomputed.
			Point inverse;
			inverse[0] = 1.f / _ray.d[0];
			inverse[1] = 1.f / _ray.d[1];
			inverse[2] = 1.f / _ray.d[2];

			bool leftIsNear[3];
			leftIsNear[0] = _ray.d[0] > 0;
			leftIsNear[1] = _ray.d[1] > 0;
			leftIsNear[2] = _ray.d[2] > 0;

			const real _EPS = 0.000001f;
			//This map allow us to count the number of transparent triangles which 
			//are in front of the intersected object.
			std::map<real, int> transpMap;

			for (;;)
			{
				const SAHKDTree::Node& node = m_nodes[curNode.nodeIndex];
				//If the node is a leaf, we shoot the ray for all the triangles in 
				//the node.
				if (node.isLeaf())
				{
					bool changed = false;
					uint32_t idx = node.getLeftChildOrLeaf();
					//If transparent objects are ignored, we must keep track of the 
					//transparent ones which might lie in front of the opaque we can 
					//hit here.
					while (m_leafData[idx] != NULL)
					{
						int rt = MISS;
						bool enable = true;
						if (m_leafData[idx]->GetLight() != NULL)
						if (m_leafData[idx]->GetLight()->AsPrimitive() == false)
							enable = false;
						if (enable)
						if ((rt = m_leafData[idx]->Intersect(_ray, _dist, thread_id)))
						{
							returnVal = rt;
							_prim = m_leafData[idx];
							changed = true;
						}
						idx++;
					}

					if (changed && _dist < curNode.t_far)
						break;

					//If the stack is empty, we stop the loop.
					if (traverseStack.empty())
						break;

					curNode = traverseStack.top();

					traverseStack.pop();
				}
				else
				{
					real t_split =
						(node.splitVal - _ray.o[node.getDimension()]) * inverse[node.getDimension()];

					//The split is closer to the far child. The near child is ignored.
					if (curNode.t_near - t_split > _EPS)
					{
						curNode.nodeIndex = node.getLeftChildOrLeaf()
							+ leftIsNear[node.getDimension()];
						continue;
					}
					//The split is closer to the near child. The far child is ignored.
					else if (t_split - curNode.t_far > _EPS)
					{
						curNode.nodeIndex = node.getLeftChildOrLeaf()
							+ 1 - leftIsNear[node.getDimension()];
						continue;
					}

					//Otherwise, the right child is put on the stack.
					TravElem rightElem;

					rightElem.nodeIndex = node.getLeftChildOrLeaf()
						+ leftIsNear[node.getDimension()];
					rightElem.t_near = t_split;
					rightElem.t_far = curNode.t_far;
					traverseStack.push(rightElem);

					curNode.nodeIndex = node.getLeftChildOrLeaf()
						+ 1 - leftIsNear[node.getDimension()];
					curNode.t_far = t_split;
				}
			}

			return returnVal;
		}

		bool SAHKDTree::collude(const Ray &_ray, real& _dist, Primitive* self, int thread_id)const
		{
			std::pair<real, real> t;
			m_sceneBox.rayIntersect(_ray, t.first, t.second);
			t.first = std::max(real(0), t.first);
			t.second = std::min(t.second, _dist);
			if (t.first > t.second)
				return false;

			std::stack<TravElem> traverseStack;

			//Root of the tree. Starting point for the search.
			TravElem curNode;
			curNode.nodeIndex = 0;
			curNode.t_near = t.first;
			curNode.t_far = t.second;

			//Inverse direction of ray is precomputed.
			Point inverse;
			inverse[0] = 1.f / _ray.d[0];
			inverse[1] = 1.f / _ray.d[1];
			inverse[2] = 1.f / _ray.d[2];

			bool leftIsNear[3];
			leftIsNear[0] = _ray.d[0] > 0;
			leftIsNear[1] = _ray.d[1] > 0;
			leftIsNear[2] = _ray.d[2] > 0;

			const real _EPS = 0.000001f;

			for (;;)
			{
				const SAHKDTree::Node& node = m_nodes[curNode.nodeIndex];
				//If the node is a leaf, we shoot the ray for all the triangles in 
				//the node.
				if (node.isLeaf())
				{
					uint32_t idx = node.getLeftChildOrLeaf();
					//If transparent objects are ignored, we must keep track of the 
					//transparent ones which might lie in front of the opaque we can 
					//hit here.
					while (m_leafData[idx] != NULL)
					{
						if (m_leafData[idx] != self)
						if (m_leafData[idx]->Intersect(_ray, _dist, thread_id) != MISS)
							return true;
						idx++;
					}

					//If the stack is empty, we stop the loop.
					if (traverseStack.empty())
						break;

					curNode = traverseStack.top();

					traverseStack.pop();
				}
				else
				{
					real t_split =
						(node.splitVal - _ray.o[node.getDimension()]) * inverse[node.getDimension()];

					//The split is closer to the far child. The near child is ignored.
					if (curNode.t_near - t_split > _EPS)
					{
						curNode.nodeIndex = node.getLeftChildOrLeaf()
							+ leftIsNear[node.getDimension()];
						continue;
					}
					//The split is closer to the near child. The far child is ignored.
					else if (t_split - curNode.t_far > _EPS)
					{
						curNode.nodeIndex = node.getLeftChildOrLeaf()
							+ 1 - leftIsNear[node.getDimension()];
						continue;
					}

					//Otherwise, the right child is put on the stack.
					TravElem rightElem;

					rightElem.nodeIndex = node.getLeftChildOrLeaf()
						+ leftIsNear[node.getDimension()];
					rightElem.t_near = t_split;
					rightElem.t_far = curNode.t_far;
					traverseStack.push(rightElem);

					curNode.nodeIndex = node.getLeftChildOrLeaf()
						+ 1 - leftIsNear[node.getDimension()];
					curNode.t_far = t_split;
				}
			}

			return false;
		}

		SAHKDTree::SAHReturn SAHKDTree::findPlane
			(std::vector<uint32_t>* triangles, AABB volume, std::vector<AABB> &triBoxes)
		{
			//Splitting plane. By default the splitting plane must tell us that no 
			//further split is possible.
			SAHKDTree::SAHReturn plane;
			plane.dimension = Node::NODE_DIM_INVALID;
			plane.value = FLT_MAX;
			plane.cost = FLT_MAX;
			//Number of triangles to the left, to the right and in the middle.
			int Nl, Np, Nr;

			//The area is too small. Therefore, we have to stop.
			if (volume.getVolume() < _EPS)
				return plane;

			//Sorting list.
			std::vector<SortElem> eventList(1);

			//The three branches are checked for all the triangles.
			for (int k = 0; k < 3; k++)
			{
				//New event list for each branch.
				eventList.clear();

				for (int i = 0; i < (int)triangles->size(); i++)
				{
					AABB triBox = triBoxes[(*triangles)[i]];
					real triMin = std::max(triBox.min[k], volume.min[k]);
					real triMax = std::min(triBox.max[k], volume.max[k]);
					//The cut is in the middle (planar case).
					if (triMin == triMax)
					{
						eventList.push_back(SortElem(triMin, 1));
					}
					//We look for new events to the left and right.
					else
					{
						//Starting event.
						eventList.push_back(SortElem(triMin, 2));
						//End event.
						eventList.push_back(SortElem(triMax, 0));
					}
				}

				//Sorting the list, based on the costs.
				sort(eventList.begin(), eventList.end(), compE);

				Nl = 0;
				Np = 0;
				Nr = triangles->size();

				//Counters for the iterations.
				int PPlus, PMinus, PPlanar;
				//This variable stores the cost of the current plane.
				real PXi;

				//Based on the list, we find the best splitting plane.
				int i = 0;
				while (i < (int)eventList.size())
					//for(int i = 0; i < eventList.size(); )
				{
					PPlus = 0; PMinus = 0; PPlanar = 0;
					//Current cost value.
					PXi = eventList[i].value;

					//We count all the triangles which are to the left/right and in 
					//the middle.
					while (i < (int)eventList.size() && eventList[i].value == PXi
						&& eventList[i].type == 0)
					{
						++PMinus; ++i;
					}
					while (i < (int)eventList.size() && eventList[i].value == PXi
						&& eventList[i].type == 1)
					{
						++PPlanar; ++i;
					}
					while (i < (int)eventList.size() && eventList[i].value == PXi
						&& eventList[i].type == 2)
					{
						++PPlus; ++i;
					}

					//Found new plane, evaluate SAH for old plane.
					Np = PPlanar;
					Nr -= PPlanar;
					Nr -= PMinus;

					std::pair<real, bool> helpCost;
					//If the splitting is far enough from the volume boundaries, 
					//we don't evaluate anything and the cost must be infty so that the
					//plane cannot be updated afterwards.
					if (PXi <= volume.min[k] + _EPS || PXi + _EPS >= volume.max[k])
					{
						helpCost.first = FLT_MAX;
					}
					//Otherwise, the cost of the current plane is evaluated.
					else
					{
						helpCost = SAH(k, PXi, volume, Nl, Nr, Np);
					}

					//Updating the counts.
					Nl += PPlus;
					Nl += PPlanar;
					Np = 0;

					///If the current cost is better than the cost of the plane, the 
					//plane is updated.
					if (helpCost.first < plane.cost)
					{
						plane.cost = helpCost.first;
						plane.dimension = k;
						plane.value = PXi;
						plane.left = helpCost.second;
					}

				}
			}

			return plane;
		}



		void SAHKDTree::fastBuild(std::vector<Primitive> &_objects)
		{
			//memory
			m_nodes.clear();
			m_leafData.clear();
			m_nodes.reserve(_objects.size() * 2);
			m_leafData.reserve(_objects.size());

			//Bounding boxes for all objects to store, no order given.
			std::vector<AABB> objectBBoxes(_objects.size());
			std::vector<SortElem1> eventListSorted[3];
			std::vector<uint8_t> triLR(_objects.size(), -1); //left:1, right:2
			std::vector<SortElem1*> leftE(_objects.size() * 2, 0);
			std::vector<uint32_t> leftTri(_objects.size(), -1);
			BuildStateStruct1 curState;
			m_sceneBox = AABB(real(0));

			//initializing bounding boxes and bentroid boxes
			for (uint32_t i = 0; i < objectBBoxes.size(); i++)
			{
				objectBBoxes[i] = _objects[i].getAABB();
				//The bounding box of the original scene is computed
				m_sceneBox.expandBy(objectBBoxes[i]);
				curState.triangleIds.push_back(i);
			}


			// build the sorted event list
#pragma omp parallel for
			for (int k = 0; k < 3; k++)
			{
				//New event list for each branch.
				eventListSorted[k].reserve(_objects.size());

				for (int i = 0; i < (int)_objects.size(); i++)
				{
					AABB triBox = objectBBoxes[i];
					real triMin = triBox.min[k];// std::max(triBox.min[k], volume.min[k]);
					real triMax = triBox.max[k];// std::min(triBox.max[k], volume.max[k]);
					//The cut is in the middle (planar case).
					if (triMin == triMax)
						eventListSorted[k].push_back(SortElem1(triMin, 1, 0, i));
					//We look for new events to the left and right.
					else
					{
						//Starting event.
						eventListSorted[k].push_back(SortElem1(triMin, 2, 0, i));
						//End event.
						eventListSorted[k].push_back(SortElem1(triMax, 0, 0, i));
					}
				}

				//Sorting the list, based on the costs.
				sort(eventListSorted[k].begin(), eventListSorted[k].end(), compE1);

				//get mapping from triangle id to event list
				curState.pEvents[k].reserve(eventListSorted[k].size());
				for (uint32_t i = 0; i < eventListSorted[k].size(); i++)
				{
					SortElem1* pE = &eventListSorted[k][i];
					curState.pEvents[k].push_back(pE);
				}
			}


			//initializing state for stack simulated recursion
			curState.volume = m_sceneBox;
			curState.nodeIndex = 0;
			curState.depth = 0;
			//initialize the tree
			m_nodes.resize(1);

			//creating recursion stack
			std::stack<BuildStateStruct1> buildStack;

			double time1 = 0, time2 = 0, time3 = 0;

			//loop till stack is empty
			for (;;)
			{
				//Diagonal of the volume.
				Vec3 boxDiag = curState.volume.getExtents();

				//Splitting plane.
				SAHKDTree::SAHReturn plane =
					fastFindPlane(curState.pEvents, curState.triangleIds, curState.volume, objectBBoxes);

				int splitDim = plane.dimension;
				real splitVal = plane.value;

				//Boundary conditions. We don't keep splitting if the MAX_DEPTH is 
				//reached or if the plane is null. Therefore, a new leaf is created.
				if (plane.cost > KI * real(curState.triangleIds.size()) ||
					curState.depth > MAX_DEPTH)
				{
					//The new node is initialized, data index determines if the leaf 
					//is a node by some bit operations
					//m_nodes[curState.nodeIndex].dataIndex =
					//	m_leafData.size() | Node::NODE_TYPE_MASK;
					m_nodes[curState.nodeIndex].setDataIndexAndLeafMask(
						m_leafData.size(), true);

					for (int i = 0; i < (int)curState.triangleIds.size(); i++)
						m_leafData.push_back(&_objects[curState.triangleIds[i]]);
					curState.triangleIds.clear();
					for (int k = 0; k < 3; k++)
						curState.pEvents[k].clear();

					//There is no more information after reaching the leaf.
					m_leafData.push_back(NULL);

					if (buildStack.empty())
						break;

					//The stack is updated.
					curState = buildStack.top();
					buildStack.pop();

					continue;
				}

				//A new state for the right child is created.
				buildStack.push(BuildStateStruct1());
				BuildStateStruct1& rightState = buildStack.top();
				rightState.volume = curState.volume.split(splitDim, splitVal);
				++curState.depth;
				rightState.depth = curState.depth;

				// left-right split
				int leftNum = 0, rightNum = 0;
				for (size_t i = 0; i < curState.triangleIds.size(); i++)
				{
					uint32_t triIndex = curState.triangleIds[i];
					AABB triBox = objectBBoxes[triIndex];
					bool m1 = triBox.min[splitDim] == splitVal &&
						triBox.min[splitDim] == triBox.max[splitDim];
					bool bLeft = (m1 && plane.left) || (!m1 &&  triBox.min[splitDim] < splitVal);
					bool bRight = (m1 && !plane.left) || (!m1 && triBox.max[splitDim] > splitVal);
					leftNum += bLeft;
					rightNum += bRight;
					triLR[triIndex] = bLeft + (bRight << 1);
				}

				int iLtri = 0, iRtri = 0;
				rightState.triangleIds.resize(rightNum);
				for (size_t i = 0; i < curState.triangleIds.size(); i++)
				{
					uint32_t triIndex = curState.triangleIds[i];
					uint8_t bLR = triLR[triIndex];
					if (bLR & 1) leftTri[iLtri++] = triIndex;
					if (bLR & 2) rightState.triangleIds[iRtri++] = triIndex;
				}
				curState.triangleIds.assign(leftTri.begin(), leftTri.begin() + iLtri);

				for (int k = 0; k < 3; k++)
				{
					int iL = 0, iR = 0;
					std::vector<SortElem1*>& curE = curState.pEvents[k];
					std::vector<SortElem1*>& rtE = rightState.pEvents[k];
					rightState.pEvents[k].resize(rightNum * 2);
					for (size_t i = 0; i < curE.size(); i++)
					{
						SortElem1* pE = curE[i];
						uint32_t triIndex = pE->getTriId();
						uint8_t bLR = triLR[triIndex];
						if (bLR & 1) leftE[iL++] = pE;
						if (bLR & 2) rtE[iR++] = pE;
					}//end for i

					curE.assign(leftE.begin(), leftE.begin() + iL);
					rtE.resize(iR);
				}//k


				//Data index is the the number of nodes added before
				m_nodes[curState.nodeIndex].setDataIndexAndLeafMask(m_nodes.size(), false);
				m_nodes[curState.nodeIndex].setDimension(splitDim);
				m_nodes[curState.nodeIndex].splitVal = splitVal;

				//The left child is stored at the end of m_nodes.
				curState.nodeIndex = m_nodes.size();
				//The right child is stored on position after the left child.
				rightState.nodeIndex = curState.nodeIndex + 1;

				//m_nodes is extended so that it can hold the left and right child.
				m_nodes.resize(rightState.nodeIndex + 1);
			}

			return;
		}

		SAHKDTree::SAHReturn SAHKDTree::fastFindPlane(std::vector<SortElem1*>* pEvents,
			std::vector<uint32_t>& triangleIds, AABB volume,
			std::vector<AABB> &triBoxes)
		{
			//Splitting plane. By default the splitting plane must tell us that no 
			//further split is possible.
			SAHKDTree::SAHReturn plane;
			plane.dimension = Node::NODE_DIM_INVALID;
			plane.value = FLT_MAX;
			plane.cost = FLT_MAX;

			for (int k = 0; k < 3; k++)
			{
				std::vector<SortElem1*>& eventList = pEvents[k];
				int N = (int)pEvents[k].size();
				int Nr = (int)triangleIds.size(), Nl = 0, Np = 0;
				int i = 0;
				while (i < N)
				{
					const SortElem1& e = (*eventList[i]);

					int PPlus = 0, PMinus = 0, PPlanar = 0;

					//We count all the triangles which are to the left/right and in 
					//the middle.
					while (i < N && eventList[i]->value == e.value
						&& eventList[i]->getType() == 0)
					{
						++PMinus; ++i;
					}
					while (i < N && eventList[i]->value == e.value
						&& eventList[i]->getType() == 1)
					{
						++PPlanar; ++i;
					}
					while (i < N && eventList[i]->value == e.value
						&& (*eventList[i]).getType() == 2)
					{
						++PPlus; ++i;
					}

					//Found new plane, evaluate SAH for old plane.
					Np = PPlanar;
					Nr -= PPlanar;
					Nr -= PMinus;

					std::pair<real, bool> helpCost;
					//If the splitting is far enough from the volume boundaries, 
					//we don't evaluate anything and the cost must be infty so that the
					//plane cannot be updated afterwards.
					if (e.value <= volume.min[k] + _EPS || e.value + _EPS >= volume.max[k])
						helpCost.first = FLT_MAX;
					//Otherwise, the cost of the current plane is evaluated.
					else
						helpCost = SAH(k, e.value, volume, Nl, Nr, Np);

					///If the current cost is better than the cost of the plane, the 
					//plane is updated.
					if (helpCost.first < plane.cost)
					{
						plane.cost = helpCost.first;
						plane.dimension = k;
						plane.value = e.value;
						plane.left = helpCost.second;
					}

					//Updating the counts.
					Nl += PPlus;
					Nl += PPlanar;
					Np = 0;
				}//i
			}//k
			return plane;
		}
	}
}//ldp