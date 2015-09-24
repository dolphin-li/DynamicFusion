#pragma once

#include "definations.h"

namespace dfusion
{
	class GpuKdTree
	{
	public:
		enum AllocationInfo
		{
			NodeCount = 0,
			NodesAllocated = 1,
			OutOfSpace = 2
		};
		//! normal node: contains the split dimension and value
		//! leaf node: left == index of first points, right==index of last point +1
		struct SplitInfo
		{
			union {
				struct
				{
					// begin of child nodes
					int left;
					// end of child nodes
					int right;
				};
				struct
				{
					int split_dim;
					float split_val;
				};
			};
			__device__ __host__ SplitInfo(int2 tex)
			{
				left = tex.x;
				right = tex.y;
			}
			__device__ __host__ SplitInfo()
			{
				left = right = 0;
			}
		};

	public:
		GpuKdTree(const void* points, int points_stride, int n, int max_leaf_size);
		void buildTree();

		static void test();

		/**
		* \brief Perform k-nearest neighbor search
		* \param[in] queries The query points for which to find the nearest neighbors, size n
		* \param[out] indices The indices of the nearest neighbors found, size knn*n
		* \param[out] dists Distances to the nearest neighbors found, size knn*n
		* \param[in] knn Number of nearest neighbors to return
		*/
		void knnSearchGpu(const float4* queries, int* indices, float* dists, size_t knn, size_t n) const;

	protected:
		void update_leftright_and_aabb(
			const DeviceArray<float>& x,
			const DeviceArray<float>& y,
			const DeviceArray<float>& z,
			const DeviceArray<int>& ix,
			const DeviceArray<int>& iy,
			const DeviceArray<int>& iz,
			const DeviceArray<int>& owners,
			DeviceArray<SplitInfo>& splits,
			DeviceArray<float4>& aabbMin,
			DeviceArray<float4>& aabbMax);
		void separate_left_and_right_children(
			DeviceArray<int>& key_in,
			DeviceArray<int>& val_in,
			DeviceArray<int>& key_out,
			DeviceArray<int>& val_out,
			DeviceArray<int>& left_right_marks,
			bool scatter_val_out = true);
		void resize_node_vectors(size_t new_size);
	private:
		DeviceArray<float4> points_;

		// tree data, those are stored per-node

		//! left child of each node. (right child==left child + 1, due to the alloc mechanism)
		//! child1_[node]==-1 if node is a leaf node
		DeviceArray<int> child1_;
		//! parent node of each node
		DeviceArray<int> parent_;
		//! split info (dim/value or left/right pointers)
		DeviceArray<SplitInfo> splits_;
		//! min aabb value of each node
		DeviceArray<float4> aabb_min_;
		//! max aabb value of each node
		DeviceArray<float4> aabb_max_;

		// those were put into a single vector of 3 elements so that only one mem transfer will be needed for all three of them
		//  thrust::device_vector<int> out_of_space_;
		//  thrust::device_vector<int> node_count_;
		//  thrust::device_vector<int> nodes_allocated_;
		DeviceArray<int> allocation_info_;
		std::vector<int> allocation_info_host_;

		int max_leaf_size_;

		// coordinate values of the points
		DeviceArray<float> points_x_, points_y_, points_z_;
		// indices
		DeviceArray<int> index_x_, index_y_, index_z_;
		// owner node
		DeviceArray<int> owners_x_, owners_y_, owners_z_;
		// contains info about whether a point was partitioned to the left or right child after a split
		DeviceArray<int> leftright_x_, leftright_y_, leftright_z_;
		DeviceArray<int> tmp_index_, tmp_owners_, tmp_misc_;
		bool delete_node_info_;
	};

}