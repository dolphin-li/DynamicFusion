#include "GpuKdTree.h"
#include "cudpp\thrust_wrapper.h"
#include "helper_math.h"
namespace dfusion
{
	texture<int2, cudaTextureType1D, cudaReadModeElementType> g_splits;
	texture<int, cudaTextureType1D, cudaReadModeElementType> g_child1;
	texture<int, cudaTextureType1D, cudaReadModeElementType> g_parent;
	texture<float4, cudaTextureType1D, cudaReadModeElementType> g_aabbLow;
	texture<float4, cudaTextureType1D, cudaReadModeElementType> g_aabbHigh;
	texture<float4, cudaTextureType1D, cudaReadModeElementType> g_elements;

	typedef GpuKdTree::SplitInfo SplitInfo;
	//! used to update the left/right pointers and aabb infos after the node splits
	struct SetLeftAndRightAndAABB
	{
		int maxPoints;
		int nElements;

		SplitInfo* nodes;
		int* counts;
		int* labels;
		float4* aabbMin;
		float4* aabbMax;
		const float* x, *y, *z;
		const int* ix, *iy, *iz;

		__host__ __device__ void operator()(int i)
		{
			int index = labels[i];
			int right = 0;
			int left = counts[i];
			nodes[index].left = left;
			if (i < nElements - 1) {
				right = counts[i + 1];
			}
			else { // index==nNodes
				right = maxPoints;
			}
			nodes[index].right = right;
			aabbMin[index].x = x[ix[left]];
			aabbMin[index].y = y[iy[left]];
			aabbMin[index].z = z[iz[left]];
			aabbMax[index].x = x[ix[right - 1]];
			aabbMax[index].y = y[iy[right - 1]];
			aabbMax[index].z = z[iz[right - 1]];
		}
	};

	//! computes the scatter target address for the split operation, see Sengupta,Harris,Zhang,Owen: Scan Primitives for GPU Computing
	//! in my use case, this is about 2x as fast as thrust::partition
	struct set_addr3
	{
		const int* val_, *f_;

		int npoints_;
		__device__ int operator()(int id)
		{
			int nf = f_[npoints_ - 1] + (val_[npoints_ - 1]);
			int f = f_[id];
			int t = id - f + nf;
			return val_[id] ? f : t;
		}
	};

	//! just for convenience: access a float4 by an index in [0,1,2]
	//! (casting it to a float* and accessing it by the index is way slower...)
	__host__ __device__ static float get_value_by_index(const float4& f, int i)
	{
		switch (i) {
		case 0:
			return f.x;
		case 1:
			return f.y;
		default:
			return f.z;
		}
	}

	//! - decide whether a node has to be split
	//! if yes:
	//! - allocate child nodes
	//! - set split axis as axis of maximum aabb length
	struct SplitNodes
	{
		typedef GpuKdTree::SplitInfo SplitInfo;
		int maxPointsPerNode;
		int* node_count;
		int* nodes_allocated;
		int* out_of_space;
		int* child1_;
		int* parent_;
		float4* aabbMin_;
		float4* aabbMax_;
		SplitInfo* splits_;

		// float4: aabbMin, aabbMax
		__device__ void operator()(int my_index) 
		{
			int child1 = child1_[my_index];
			SplitInfo& s = splits_[my_index];
			float4 aabbMin = aabbMin_[my_index];
			float4 aabbMax = aabbMax_[my_index];

			bool split_node = false;
			// first, each thread block counts the number of nodes that it needs to allocate...
			__shared__ int block_nodes_to_allocate;
			if (threadIdx.x == 0) block_nodes_to_allocate = 0;
			__syncthreads();

			// don't split if all points are equal
			// (could lead to an infinite loop, and doesn't make any sense anyway)
			bool all_points_in_node_are_equal = aabbMin.x == aabbMax.x 
				&& aabbMin.y == aabbMax.y && aabbMin.z == aabbMax.z;

			int offset_to_global = 0;

			// maybe this could be replaced with a reduction...
			if ((child1 == -1) && (s.right - s.left > maxPointsPerNode) 
				&& !all_points_in_node_are_equal) { // leaf node
				split_node = true;
				offset_to_global = atomicAdd(&block_nodes_to_allocate, 2);
			}

			__syncthreads();
			__shared__ int block_left;
			__shared__ bool enough_space;
			// ... then the first thread tries to allocate this many nodes...
			if (threadIdx.x == 0) {
				block_left = atomicAdd(node_count, block_nodes_to_allocate);
				enough_space = block_left + block_nodes_to_allocate < *nodes_allocated;
				// if it doesn't succeed, no nodes will be created by this block
				if (!enough_space) {
					atomicAdd(node_count, -block_nodes_to_allocate);
					*out_of_space = 1;
				}
			}

			__syncthreads();
			// this thread needs to split it's node && there was enough space for all the nodes
			// in this block.
			//(The whole "allocate-per-block-thing" is much faster than letting each element allocate
			// its space on its own, because shared memory atomics are A LOT faster than
			// global mem atomics!)
			if (split_node && enough_space) {
				int left = block_left + offset_to_global;

				splits_[left].left = s.left;
				splits_[left].right = s.right;
				splits_[left + 1].left = 0;
				splits_[left + 1].right = 0;

				// split axis/position: middle of longest aabb extent
				float4 aabbDim = aabbMax - aabbMin;
				int maxDim = 0;
				float maxDimLength = aabbDim.x;
				float4 splitVal = (aabbMax + aabbMin);
				splitVal *= 0.5f;
				for (int i = 1; i <= 2; i++) {
					float val = get_value_by_index(aabbDim, i);
					if (val > maxDimLength) {
						maxDim = i;
						maxDimLength = val;
					}
				}
				s.split_dim = maxDim;
				s.split_val = get_value_by_index(splitVal, maxDim);

				child1_[my_index] = left;
				splits_[my_index] = s;

				parent_[left] = my_index;
				parent_[left + 1] = my_index;
				child1_[left] = -1;
				child1_[left + 1] = -1;
			}
		}
	};

	//! mark a point as belonging to the left or right child of its current parent
	//! called after parents are split
	struct MovePointsToChildNodes
	{
		typedef GpuKdTree::SplitInfo SplitInfo;
		MovePointsToChildNodes(int* child1, SplitInfo* splits, 
			float* x, float* y, float* z, int* ox, int* oy, 
			int* oz, int* lrx, int* lry, int* lrz)
		: child1_(child1), splits_(splits), x_(x), y_(y), z_(z), ox_(ox), 
		oy_(oy), oz_(oz), lrx_(lrx), lry_(lry), lrz_(lrz){}

		//  int dim;
		//  float threshold;
		int* child1_;
		SplitInfo* splits_;

		// coordinate values
		float* x_, *y_, *z_;
		// owner indices -> which node does the point belong to?
		int* ox_, *oy_, *oz_;
		// temp info: will be set to 1 of a point is moved to the right child node, 0 otherwise
		// (used later in the scan op to separate the points of the children into continuous ranges)
		int* lrx_, *lry_, *lrz_;

		__device__ void operator()(int index, int point_ind1, int point_ind2, int point_ind3)
		{
			int owner = ox_[index]; 
			int leftChild = child1_[owner];
			int split_dim;
			float dim_val1, dim_val2, dim_val3;
			SplitInfo split;
			lrx_[index] = 0;
			lry_[index] = 0;
			lrz_[index] = 0;
			// this element already belongs to a leaf node -> everything alright, no need to change anything
			if (leftChild == -1) 
				return;

			// otherwise: load split data, and assign this index to the new owner
			split = splits_[owner];
			split_dim = split.split_dim;
			switch (split_dim) {
			case 0:
				dim_val1 = x_[point_ind1];
				dim_val2 = x_[point_ind2];
				dim_val3 = x_[point_ind3];
				break;
			case 1:
				dim_val1 = y_[point_ind1];
				dim_val2 = y_[point_ind2];
				dim_val3 = y_[point_ind3];
				break;
			default:
				dim_val1 = z_[point_ind1];
				dim_val2 = z_[point_ind2];
				dim_val3 = z_[point_ind3];
				break;

			}

			int r1 = leftChild + (dim_val1 > split.split_val);
			ox_[index] = r1;
			int r2 = leftChild + (dim_val2 > split.split_val);
			oy_[index] = r2;
			oz_[index] = leftChild + (dim_val3 > split.split_val);

			lrx_[index] = (dim_val1 > split.split_val);
			lry_[index] = (dim_val2 > split.split_val);
			lrz_[index] = (dim_val3 > split.split_val);
		}
	};
	__global__ void splitNode_kernel(SplitNodes s, int n)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid < n)
			s(tid);
	}

	__global__ void movePointsToChildNodes_kernel(MovePointsToChildNodes s, 
		int* index_x, int* index_y, int* index_z, int n)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid < n)
		{
			s(tid, index_x[tid], index_y[tid], index_z[tid]);
		}
	}

	__global__ void for_each_SetLeftAndRightAndAABB_kernel(SetLeftAndRightAndAABB s, int n)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid < n)
			s(tid);
	}

	__global__ void collect_aabb_kernel(float4* aabb_min, float4* aabb_max,
		const float* x, const int* ix,
		const float* y, const int* iy,
		const float* z, const int* iz, int n)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid == 0)
		{
			aabb_min[0] = make_float4(x[ix[0]], y[iy[0]], z[iz[0]], 0);
			aabb_max[0] = make_float4(x[ix[n-1]], y[iy[n-1]], z[iz[n-1]], 0);
		}
	}

	__global__ void init_splitInfo_kernel(GpuKdTree::SplitInfo* splits, int n, int nPoints)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid < n)
		{
			GpuKdTree::SplitInfo s;
			s.left = 0;
			s.right = 0;
			if (tid == 0)
				s.right = nPoints;
			splits[tid] = s;
		}
	}

	__global__ void copy_points_kernel(const void* points_in, int points_in_stride,
		float4* points_out, int n)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid < n)
		{
			points_out[tid] = *(const float4*)((const char*)points_in + points_in_stride*tid);
		}
	}

	__global__ void set_addr3_kernel(set_addr3 sa, int* out, int n)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid < n)
		{
			out[tid] = sa(tid);
		}
	}

	template<class T>
	__global__ void resize_vec_kernel(const T* oldVec, T* newVec, int oldSize, int newSize, T val)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid < newSize)
		{
			if (tid < oldSize)
				newVec[tid] = oldVec[tid];
			else
				newVec[tid] = val;
		}
	}

	GpuKdTree::GpuKdTree(const void* points, int points_stride, int n, int max_leaf_size)
	{
		max_leaf_size_ = max_leaf_size;
		delete_node_info_ = 0;

		points_.create(n);
		{
			dim3 block(256);
			dim3 grid(divUp(points_.size(), block.x));
			copy_points_kernel << <grid, block >> >(points, points_stride,
				points_.ptr(), n);
		}
		int prealloc = divUp(points_.size() * 16, max_leaf_size_);
		allocation_info_host_.resize(3);
		allocation_info_.create(3);
		allocation_info_host_[GpuKdTree::NodeCount] = 1;
		allocation_info_host_[GpuKdTree::NodesAllocated] = prealloc;
		allocation_info_host_[GpuKdTree::OutOfSpace] = 0;
		cudaSafeCall(cudaMemcpy(allocation_info_.ptr(), allocation_info_host_.data(),
			allocation_info_host_.size()*sizeof(int), cudaMemcpyHostToDevice));

		child1_.create(prealloc);
		thrust_wrapper::assign(child1_.ptr(), -1, child1_.size());
		parent_.create(prealloc);
		thrust_wrapper::assign(parent_.ptr(), -1, parent_.size());
		splits_.create(prealloc);
		{
			dim3 block(256);
			dim3 grid(divUp(splits_.size(),block.x));
			init_splitInfo_kernel << <grid, block >> >(splits_.ptr(),
				splits_.size(), points_.size());
		}

		aabb_min_.create(prealloc);
		thrust_wrapper::assign(aabb_min_.ptr(), make_float4(0.f, 0.f, 0.f, 0.f), aabb_min_.size());
		aabb_max_.create(prealloc);
		thrust_wrapper::assign(aabb_max_.ptr(), make_float4(0.f, 0.f, 0.f, 0.f), aabb_max_.size());

		index_x_.create(points_.size());
		thrust_wrapper::assign(index_x_.ptr(), 0, index_x_.size());
		index_y_.create(points_.size());
		thrust_wrapper::assign(index_y_.ptr(), 0, index_y_.size());
		index_z_.create(points_.size());
		thrust_wrapper::assign(index_z_.ptr(), 0, index_z_.size());

		owners_x_.create(points_.size());
		thrust_wrapper::assign(owners_x_.ptr(), 0, owners_x_.size());
		owners_y_.create(points_.size());
		thrust_wrapper::assign(owners_y_.ptr(), 0, owners_y_.size());
		owners_z_.create(points_.size());
		thrust_wrapper::assign(owners_z_.ptr(), 0, owners_z_.size());

		leftright_x_.create(points_.size());
		thrust_wrapper::assign(leftright_x_.ptr(), 0, leftright_x_.size());
		leftright_y_.create(points_.size());
		thrust_wrapper::assign(leftright_y_.ptr(), 0, leftright_y_.size());
		leftright_z_.create(points_.size());
		thrust_wrapper::assign(leftright_z_.ptr(), 0, leftright_z_.size());

		tmp_index_.create(points_.size());
		thrust_wrapper::assign(tmp_index_.ptr(), 0, tmp_index_.size());
		tmp_owners_.create(points_.size());
		thrust_wrapper::assign(tmp_owners_.ptr(), 0, tmp_owners_.size());
		tmp_misc_.create(points_.size());
		thrust_wrapper::assign(tmp_misc_.ptr(), 0, tmp_misc_.size());

		points_x_.create(points_.size());
		thrust_wrapper::assign(points_x_.ptr(), 0, points_x_.size());
		points_y_.create(points_.size());
		thrust_wrapper::assign(points_y_.ptr(), 0, points_y_.size());
		points_z_.create(points_.size());
		thrust_wrapper::assign(points_z_.ptr(), 0, points_z_.size());
		delete_node_info_ = false;
	}

	void GpuKdTree::buildTree()
	{
		thrust_wrapper::seperate_channels(points_.ptr(), points_x_.ptr(), 
			points_y_.ptr(), points_z_.ptr(), points_.size());

		thrust_wrapper::make_counting_array(index_x_.ptr(), points_.size(), 0);
		thrust_wrapper::copy(index_y_.ptr(), index_x_.ptr(), index_x_.size());
		thrust_wrapper::copy(index_z_.ptr(), index_x_.ptr(), index_x_.size());

		DeviceArray<float> tmpv(points_.size());

		// create sorted index list -> can be used to compute AABBs in O(1)
		thrust_wrapper::copy(tmpv.ptr(), points_x_.ptr(), points_x_.size());
		thrust_wrapper::sort_by_key(tmpv.ptr(), index_x_.ptr(), index_x_.size());
		thrust_wrapper::copy(tmpv.ptr(), points_y_.ptr(), points_y_.size());
		thrust_wrapper::sort_by_key(tmpv.ptr(), index_y_.ptr(), index_y_.size());
		thrust_wrapper::copy(tmpv.ptr(), points_z_.ptr(), points_z_.size());
		thrust_wrapper::sort_by_key(tmpv.ptr(), index_z_.ptr(), index_z_.size());

		// bounding box info
		{
			dim3 block(1);
			dim3 grid(1);
			collect_aabb_kernel << <grid, block >> >(aabb_min_.ptr(), aabb_max_.ptr(),
				points_x_.ptr(), index_x_.ptr(), points_y_.ptr(), index_y_.ptr(),
				points_z_.ptr(), index_z_.ptr(), points_.size());
		}
		
		int last_node_count = 0;
		for (int i = 0;; i++) 
		{
			SplitNodes sn;
			sn.maxPointsPerNode = max_leaf_size_;
			sn.node_count = allocation_info_.ptr() + NodeCount;
			sn.nodes_allocated = allocation_info_.ptr() + NodesAllocated;
			sn.out_of_space = allocation_info_.ptr() + OutOfSpace;
			sn.child1_ = child1_.ptr();
			sn.parent_ = parent_.ptr();
			sn.splits_ = splits_.ptr();
			sn.aabbMin_ = aabb_min_.ptr();
			sn.aabbMax_ = aabb_max_.ptr();
			if (last_node_count)
			{
				dim3 block(256);
				dim3 grid(divUp(last_node_count, block.x));
				splitNode_kernel << <grid, block >> >(sn, last_node_count);
			}

			// copy allocation info to host
			cudaSafeCall(cudaMemcpy(allocation_info_host_.data(), allocation_info_.ptr(),
				allocation_info_host_.size()*sizeof(int), cudaMemcpyDeviceToHost));

			if (last_node_count == allocation_info_host_[NodeCount]) // no more nodes were split -> done
				break;
			
			last_node_count = allocation_info_host_[NodeCount];

			// a node was un-splittable due to a lack of space
			if (allocation_info_host_[OutOfSpace] == 1) 
			{
				resize_node_vectors(allocation_info_host_[NodesAllocated] * 2);
				allocation_info_host_[OutOfSpace] = 0;
				allocation_info_host_[NodesAllocated] *= 2;
				cudaSafeCall(cudaMemcpy(allocation_info_.ptr(), allocation_info_host_.data(),
					allocation_info_host_.size()*sizeof(int), cudaMemcpyHostToDevice));
			}

			// foreach point: point was in node that was split?move it to child (leaf) node : do nothing
			MovePointsToChildNodes sno(child1_.ptr(), splits_.ptr(), points_x_.ptr(),
				points_y_.ptr(), points_z_.ptr(), owners_x_.ptr(), owners_y_.ptr(),
				owners_z_.ptr(), leftright_x_.ptr(), leftright_y_.ptr(), leftright_z_.ptr()
				);
			{
				dim3 block(256);
				dim3 grid(divUp(points_.size(), block.x));
				movePointsToChildNodes_kernel << <grid, block >> >(sno, 
					index_x_.ptr(), index_y_.ptr(), index_z_.ptr(),
					points_.size());
			}

			// move points around so that each leaf node's points are continuous
			separate_left_and_right_children(index_x_, owners_x_, tmp_index_, tmp_owners_, leftright_x_);
			tmp_index_.copyTo(index_x_);
			tmp_owners_.copyTo(owners_x_);
			separate_left_and_right_children(index_y_, owners_y_, tmp_index_, tmp_owners_, leftright_y_, false);
			tmp_index_.copyTo(index_y_);
			separate_left_and_right_children(index_z_, owners_z_, tmp_index_, tmp_owners_, leftright_z_, false);
			tmp_index_.copyTo(index_z_);

			// calculate new AABB etc
			update_leftright_and_aabb(points_x_, points_y_, points_z_, index_x_, 
				index_y_, index_z_, owners_x_, splits_, aabb_min_, aabb_max_);
		} 

		DeviceArray<float4> points_backup;
		points_.copyTo(points_backup);
		thrust_wrapper::gather(points_backup.ptr(), index_x_.ptr(), points_.ptr(), points_.size());

		// bind src to texture
		size_t offset;
		cudaChannelFormatDesc desc_int2 = cudaCreateChannelDesc<int2>();
		cudaBindTexture(&offset, &g_splits, splits_.ptr(), &desc_int2,
			splits_.size()*sizeof(int2));
		assert(offset == 0);
		cudaChannelFormatDesc desc_int = cudaCreateChannelDesc<int>();
		cudaBindTexture(&offset, &g_child1, child1_.ptr(), &desc_int,
			child1_.size()*sizeof(int));
		assert(offset == 0);
		cudaBindTexture(&offset, &g_parent, parent_.ptr(), &desc_int,
			parent_.size()*sizeof(int));
		assert(offset == 0);
		cudaChannelFormatDesc desc_f4 = cudaCreateChannelDesc<float4>();
		cudaBindTexture(&offset, &g_aabbLow, aabb_min_.ptr(), &desc_f4,
			aabb_min_.size()*sizeof(float4));
		assert(offset == 0);
		cudaBindTexture(&offset, &g_aabbHigh, aabb_max_.ptr(), &desc_f4,
			aabb_max_.size()*sizeof(float4));
		assert(offset == 0);
		cudaBindTexture(&offset, &g_elements, points_.ptr(), &desc_f4,
			points_.size()*sizeof(float4));
		assert(offset == 0);
	}

	namespace KdTreeCudaPrivate
	{	
		//! thrust transform functor
		//! transforms indices in the internal data set back to the original indices
		struct map_indices
		{
			const int* v_;

			map_indices(const int* v) : v_(v) {
			}

			__host__ __device__ int operator() (const int&i) const
			{
				if (i >= 0) return v_[i];
				else return i;
			}
		};
		//! implementation of L2 distance for the CUDA kernels
		struct CudaL2Distance
		{
			static float __host__ __device__ __forceinline__ axisDist(float a, float b)
			{
				return (a - b)*(a - b);
			}

			static float __host__ __device__ __forceinline__ dist(float4 a, float4 b)
			{
				return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z);
			}
		};

		//! result set for the 1nn search. Doesn't do any global memory accesses on its own,
		template< typename DistanceType >
		struct SingleResultSet
		{
			int bestIndex;
			DistanceType bestDist;

			__device__ __host__ SingleResultSet() : 
				bestIndex(-1), bestDist(INFINITY){ }

			__device__ inline float worstDist()
			{
				return bestDist;
			}

			__device__ inline void insert(int index, DistanceType dist)
			{
				if (dist <= bestDist) {
					bestIndex = index;
					bestDist = dist;
				}
			}

			DistanceType* resultDist;
			int* resultIndex;

			__device__ inline void setResultLocation(DistanceType* dists, int* index, int thread)
			{
				resultDist = dists + thread;
				resultIndex = index + thread;
				resultDist[0] = INFINITY;
				resultIndex[0] = -1;
			}

			__device__ inline void finish()
			{
				resultDist[0] = bestDist;
				resultIndex[0] = bestIndex;
			}
		};

		template< typename DistanceType >
		struct GreaterThan
		{
			__device__
			bool operator()(DistanceType a, DistanceType b)
			{
				return a>b;
			}
		};


		// using this and the template uses 2 or 3 registers more than the direct implementation in the kNearestKernel, but
		// there is no speed difference.
		// Setting useHeap as a template parameter leads to a whole lot of things being
		// optimized away by nvcc.
		// Register counts are the same as when removing not-needed variables in explicit specializations
		// and the "if( useHeap )" branches are eliminated at compile time.
		// The downside of this: a bit more complex kernel launch code.
		template< typename DistanceType>
		struct KnnResultSet
		{
			int foundNeighbors;
			DistanceType largestHeapDist;
			int maxDistIndex;
			const int k;
			const bool sorted;

			__device__ __host__ KnnResultSet(int knn, bool sortResults) : 
				foundNeighbors(0), largestHeapDist(INFINITY), k(knn), sorted(sortResults){ }

			__device__ inline DistanceType worstDist()
			{
				return largestHeapDist;
			}

			__device__ inline void insert(int index, DistanceType dist)
			{
				if (foundNeighbors < k) {
					resultDist[foundNeighbors] = dist;
					resultIndex[foundNeighbors] = index;
					if (foundNeighbors == k - 1)
						findLargestDistIndex();
					foundNeighbors++;
				}
				else if (dist < largestHeapDist) {
					resultDist[maxDistIndex] = dist;
					resultIndex[maxDistIndex] = index;
					findLargestDistIndex();
				}
			}

			__device__ void findLargestDistIndex()
			{
				largestHeapDist = resultDist[0];
				maxDistIndex = 0;
				for (int i = 1; i<k; i++)
				if (resultDist[i] > largestHeapDist) {
					maxDistIndex = i;
					largestHeapDist = resultDist[i];
				}
			}

			float* resultDist;
			int* resultIndex;

			__device__ inline void setResultLocation(DistanceType* dists, int* index, int thread)
			{
				resultDist = dists + thread*k;
				resultIndex = index + thread*k;
				for (int i = 0; i < k; i++) {
					resultDist[i] = INFINITY;
					resultIndex[i] = -1;
				}
			}

			__host__ __device__ inline void finish()
			{
				if (sorted) {
					//if (!useHeap) flann::cuda::heap::make_heap(resultDist, resultIndex, k, 
					//	GreaterThan<DistanceType>());
					//for (int i = k - 1; i>0; i--) {
					//	swap(resultDist[0], resultDist[i]);
					//	swap(resultIndex[0], resultIndex[i]);
					//	flann::cuda::heap::sift_down(resultDist, resultIndex, 0, i, GreaterThan<DistanceType>());
					//}
				}
			}
		};

		template< typename GPUResultSet>
		__device__ void searchNeighbors(const float4& q,
			GPUResultSet& result)
		{
			bool backtrack = false;
			int lastNode = -1;
			int current = 0;

			GpuKdTree::SplitInfo split;
			while (true) {
				if (current == -1) break;
				split = tex1Dfetch(g_splits, current);

				float diff1 = (split.split_dim == 0)*(q.x - split.split_val)
					+ (split.split_dim == 1)*(q.y - split.split_val)
					+ (split.split_dim == 2)*(q.z - split.split_val);

				// children are next to each other: leftChild+1 == rightChild
				int leftChild = tex1Dfetch(g_child1, current);
				int bestChild = leftChild + (diff1>=0);
				int otherChild = leftChild + (diff1<0);

				if (!backtrack) {
					/* If this is a leaf node, then do check and return. */
					if (leftChild == -1) {
						for (int i = split.left; i < split.right; ++i) {
							float dist = CudaL2Distance::dist(tex1Dfetch(g_elements, i), q);
							result.insert(i, dist);
						}

						backtrack = true;
						lastNode = current;
						current = tex1Dfetch(g_parent, current); 
					}
					else { // go to closer child node
						lastNode = current;
						current = bestChild;
					}
				}
				else { 
					// continue moving back up the tree or visit far node?
					// minimum possible distance between query point and a point inside the AABB
					float4 aabbMin = tex1Dfetch(g_aabbLow, otherChild);
					float4 aabbMax = tex1Dfetch(g_aabbHigh, otherChild);
					float mindistsq = (q.x < aabbMin.x) * CudaL2Distance::axisDist(q.x, aabbMin.x)
						+ (q.x > aabbMax.x) * CudaL2Distance::axisDist(q.x, aabbMax.x)
						+ (q.y < aabbMin.y) * CudaL2Distance::axisDist(q.y, aabbMin.y)
						+ (q.y > aabbMax.y) * CudaL2Distance::axisDist(q.y, aabbMax.y)
						+ (q.z < aabbMin.z) * CudaL2Distance::axisDist(q.z, aabbMin.z)
						+ (q.z > aabbMax.z) * CudaL2Distance::axisDist(q.z, aabbMax.z);

					//  the far node was NOT the last node (== not visited yet) 
					//  AND there could be a closer point in it
					if ((lastNode == bestChild) && (mindistsq <= result.worstDist())) 
					{
						lastNode = current;
						current = otherChild;
						backtrack = false;
					}
					else {
						lastNode = current;
						current = tex1Dfetch(g_parent, current); 
					}
				}
			}
		}

		template< typename GPUResultSet>
		__global__ void nearestKernel(const float4* query,
			int* resultIndex, float* resultDist,
			int querysize, GPUResultSet result)
		{
			typedef float DistanceType;
			typedef float ElementType;
			//                  typedef DistanceType float;
			int tid = blockDim.x*blockIdx.x + threadIdx.x;

			if (tid >= querysize) return;

			float4 q = query[tid];

			result.setResultLocation(resultDist, resultIndex, tid);
			searchNeighbors(q, result);
			result.finish();
		}

		__global__ void mapIndicesKerenel(map_indices mp, int* indices_in, int* indices_out, int n)
		{
			size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
			if (tid < n)
				indices_out[tid] = mp(indices_in[tid]);
		}
	}

	void GpuKdTree::knnSearchGpu(const float4* queries, int* indices, float* dists, size_t knn, size_t n) const
	{
		int threadsPerBlock = 128;
		int blocksPerGrid = divUp(n, threadsPerBlock);
		bool sorted = false;

		if (knn == 1) {
			KdTreeCudaPrivate::nearestKernel << <blocksPerGrid, threadsPerBlock >> > (
				queries,
				indices,
				dists,
				n, 
				KdTreeCudaPrivate::SingleResultSet<float>());
		}
		else {
			KdTreeCudaPrivate::nearestKernel << <blocksPerGrid, threadsPerBlock >> > (
				queries,
				indices,
				dists,
				n,
				KdTreeCudaPrivate::KnnResultSet<float>(knn, sorted)
				);
		}

		DeviceArray<int> tmp(n);
		thrust_wrapper::copy(tmp.ptr(), indices, n);
		KdTreeCudaPrivate::mapIndicesKerenel << <blocksPerGrid, threadsPerBlock >> >
			(KdTreeCudaPrivate::map_indices(index_x_), tmp.ptr(), indices, n);
	}

	void GpuKdTree::update_leftright_and_aabb(
		const DeviceArray<float>& x,
		const DeviceArray<float>& y,
		const DeviceArray<float>& z,
		const DeviceArray<int>& ix,
		const DeviceArray<int>& iy,
		const DeviceArray<int>& iz,
		const DeviceArray<int>& owners,
		DeviceArray<SplitInfo>& splits,
		DeviceArray<float4>& aabbMin,
		DeviceArray<float4>& aabbMax)
	{
		DeviceArray<int>& labelsUnique = tmp_owners_;
		DeviceArray<int>& countsUnique = tmp_index_;
		// assume: points of each node are continuous in the array

		// find which nodes are here, and where each node's points begin and end
		int unique_labels = thrust_wrapper::unique_counting_by_key_copy(
			owners.ptr(), 0, labelsUnique.ptr(), countsUnique.ptr(), owners.size());

		// update the info
		SetLeftAndRightAndAABB s;
		s.maxPoints = x.size();
		s.nElements = unique_labels;
		s.nodes = splits.ptr();
		s.counts = countsUnique.ptr();
		s.labels = labelsUnique.ptr();
		s.x = x.ptr();
		s.y = y.ptr();
		s.z = z.ptr();
		s.ix = ix.ptr();
		s.iy = iy.ptr();
		s.iz = iz.ptr();
		s.aabbMin = aabbMin.ptr();
		s.aabbMax = aabbMax.ptr();

		dim3 block(256);
		dim3 grid(divUp(unique_labels, block.x));
		for_each_SetLeftAndRightAndAABB_kernel << <grid, block >> >(s, unique_labels);
		cudaSafeCall(cudaGetLastError());
	}

	//! Separates the left and right children of each node into continuous parts of the array.
	//! More specifically, it seperates children with even and odd node indices because nodes are always
	//! allocated in pairs -> child1==child2+1 -> child1 even and child2 odd, or vice-versa.
	//! Since the split operation is stable, this results in continuous partitions
	//! for all the single nodes.
	//! (basically the split primitive according to sengupta et al)
	//! about twice as fast as thrust::partition
	void GpuKdTree::separate_left_and_right_children(
		DeviceArray<int>& key_in, 
		DeviceArray<int>& val_in, 
		DeviceArray<int>& key_out, 
		DeviceArray<int>& val_out, 
		DeviceArray<int>& left_right_marks, 
		bool scatter_val_out)
	{
		DeviceArray<int>* f_tmp = &val_out;
		DeviceArray<int>* addr_tmp = &tmp_misc_;

		thrust_wrapper::exclusive_scan(left_right_marks.ptr(), f_tmp->ptr(), left_right_marks.size());

		set_addr3 sa;
		sa.val_ = left_right_marks.ptr();
		sa.f_ = f_tmp->ptr();
		sa.npoints_ = key_in.size();
		{
			dim3 block(256);
			dim3 grid(divUp(val_in.size(), block.x));
			set_addr3_kernel << <grid, block >> >(sa, addr_tmp->ptr(), val_in.size());
			cudaSafeCall(cudaGetLastError());
		}
		thrust_wrapper::scatter(key_in.ptr(), addr_tmp->ptr(), key_out.ptr(), key_in.size());
		if (scatter_val_out) 
			thrust_wrapper::scatter(val_in.ptr(), addr_tmp->ptr(), val_out.ptr(), val_in.size());
	}

	template<class T>
	static void resize_vec(DeviceArray<T>& oldVec, int new_size, T val)
	{
		DeviceArray<T> newVec;
		newVec.create(new_size);

		dim3 block(256);
		dim3 grid(divUp(new_size, block.x));
		resize_vec_kernel<<<grid, block>>>(oldVec.ptr(), newVec.ptr(), oldVec.size(), newVec.size(), val);
	}

	//! allocates additional space in all the node-related vectors.
	//! new_size elements will be added to all vectors.
	void GpuKdTree::resize_node_vectors(size_t new_size)
	{
		resize_vec(child1_, new_size, -1);
		resize_vec(parent_, new_size, -1);
		SplitInfo s;
		s.left = 0;
		s.right = 0;
		resize_vec(splits_, new_size, s);
		float4 f = make_float4(0,0,0,0);
		resize_vec(aabb_min_, new_size, f);
		resize_vec(aabb_max_, new_size, f);
	}
}