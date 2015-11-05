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
		GpuKdTree();
		void buildTree(const float4* points, int n, int stride_in_float4, int max_leaf_size=32);

		/**
		* \brief Perform k-nearest neighbor search
		* \param[in] queries The query points for which to find the nearest neighbors, size n
		* \param[out] indices The indices of the nearest neighbors found, size knn*n
		* \param[out] dists Distances to the nearest neighbors found, size knnStride*n
		* \param[in] knn Number of nearest neighbors to return
		* \param[in] knnStride stride of input indices
		* \param[in] excludeSelf if the query is the same with the tree points, we may want
		*			 to avoid self-search, set this param to true then
		*/
		void knnSearchGpu(const float4* queries, int query_stride_in_float4, 
			int* indices, float* dists, size_t knn, size_t n, size_t knnStride, 
			bool excludeSelf = false) const;
		void knnSearchGpu(const float4* queries, int query_stride_in_float4,
			ushort* indices, float* dists, size_t knn, size_t n, size_t knnStride,
			bool excludeSelf = false) const;

		/**
		* \brief Perform k-nearest neighbor search on a grid: (x,y,z) = origion + (ix,iy,iz)*voxelSize
		* \param[out] volumeSurf the volume of type WarpField::KnnIdx (generally uchar4, ushort4)
		* \param[in] begin the x0y0z0 of the input grid
		* \param[in] end the x1y1z1 of the input grid: valid range: begin:end
		* \param[in] origion the origion point of the input grid
		* \param[in] voxelSize the voxel size of the input grid
		* \param[in] knn Number of nearest neighbors to return
		*/
		void knnSearchGpu(cudaSurfaceObject_t volumeSurf, int3 begin, int3 end, 
			float3 origion, float voxelSize, size_t knn) const;

		// for debug
		static void test();
	protected:
		void update_leftright_and_aabb(
			const float* x,
			const float* y,
			const float* z,
			const int* ix,
			const int* iy,
			const int* iz,
			const int* owners,
			SplitInfo* splits,
			float4* aabbMin,
			float4* aabbMax);
		void separate_left_and_right_children(
			int* key_in,
			int* val_in,
			int* key_out,
			int* val_out,
			int* left_right_marks,
			bool scatter_val_out = true);

		void allocateMemPool(int nInputPoints, int maxLeafSize);
		void resize_node_vectors(size_t new_size);
		void bindTextures()const;

		int input_points_offset_byte()const{ return (const char*)input_points_ptr_ - (const char*)mempool_.ptr(); }
		int points_offset_byte()const{ return (const char*)points_ptr_ - (const char*)mempool_.ptr(); }
		int aabb_min_offset_byte()const{ return (const char*)aabb_min_ptr_ - (const char*)mempool_.ptr(); }
		int aabb_max_offset_byte()const{ return (const char*)aabb_max_ptr_ - (const char*)mempool_.ptr(); }
		int points_x_offset_byte()const{ return (const char*)points_x_ptr_ - (const char*)mempool_.ptr(); }
		int points_y_offset_byte()const{ return (const char*)points_y_ptr_ - (const char*)mempool_.ptr(); }
		int points_z_offset_byte()const{ return (const char*)points_z_ptr_ - (const char*)mempool_.ptr(); }
		int tpt_x_offset_byte()const{ return (const char*)tmp_pt_x_ptr_ - (const char*)mempool_.ptr(); }
		int tpt_y_offset_byte()const{ return (const char*)tmp_pt_y_ptr_ - (const char*)mempool_.ptr(); }
		int tpt_z_offset_byte()const{ return (const char*)tmp_pt_z_ptr_ - (const char*)mempool_.ptr(); }
		int splits_offset_byte()const{ return (const char*)splits_ptr_ - (const char*)mempool_.ptr(); }
		int child1_offset_byte()const{ return (const char*)child1_ptr_ - (const char*)mempool_.ptr(); }
		int parent_offset_byte()const{ return (const char*)parent_ptr_ - (const char*)mempool_.ptr(); }
		int index_x_offset_byte()const{ return (const char*)index_x_ptr_ - (const char*)mempool_.ptr(); }
		int index_y_offset_byte()const{ return (const char*)index_y_ptr_ - (const char*)mempool_.ptr(); }
		int index_z_offset_byte()const{ return (const char*)index_z_ptr_ - (const char*)mempool_.ptr(); }
		int owner_x_offset_byte()const{ return (const char*)owner_x_ptr_ - (const char*)mempool_.ptr(); }
		int owner_y_offset_byte()const{ return (const char*)owner_y_ptr_ - (const char*)mempool_.ptr(); }
		int owner_z_offset_byte()const{ return (const char*)owner_z_ptr_ - (const char*)mempool_.ptr(); }
		int leftright_x_offset_byte()const{ return (const char*)leftright_x_ptr_ - (const char*)mempool_.ptr(); }
		int leftright_y_offset_byte()const{ return (const char*)leftright_y_ptr_ - (const char*)mempool_.ptr(); }
		int leftright_z_offset_byte()const{ return (const char*)leftright_z_ptr_ - (const char*)mempool_.ptr(); }
		int tmp_index_offset_byte()const{ return (const char*)tmp_index_ptr_ - (const char*)mempool_.ptr(); }
		int tmp_owners_offset_byte()const{ return (const char*)tmp_owners_ptr_ - (const char*)mempool_.ptr(); }
		int tmp_misc_offset_byte()const{ return (const char*)tmp_misc_ptr_ - (const char*)mempool_.ptr(); }
		int allocation_info_offset_byte()const{ return (const char*)allocation_info_ptr_ - (const char*)mempool_.ptr(); }
	private:
		int nInputPoints_;
		int nAllocatedPoints_; // bigger than nInputPoints_, to prevent allocation each time
		int max_leaf_size_;

		// divUp(nAllocatedPoints_ * 16, max_leaf_size_);
		int prealloc_;
		
		// memory pool
		DeviceArray<int> mempool_;

		float4* input_points_ptr_;
		float4* points_ptr_;
		float4* aabb_min_ptr_;
		float4* aabb_max_ptr_;
		float* points_x_ptr_;
		float* points_y_ptr_;
		float* points_z_ptr_;
		float* tmp_pt_x_ptr_;
		float* tmp_pt_y_ptr_;
		float* tmp_pt_z_ptr_;
		SplitInfo* splits_ptr_;
		int* child1_ptr_;
		int* parent_ptr_;
		int* index_x_ptr_;
		int* index_y_ptr_;
		int* index_z_ptr_;
		int* owner_x_ptr_;
		int* owner_y_ptr_;
		int* owner_z_ptr_;
		int* leftright_x_ptr_;
		int* leftright_y_ptr_;
		int* leftright_z_ptr_;
		int* tmp_index_ptr_;
		int* tmp_owners_ptr_;
		int* tmp_misc_ptr_;
		// tmp_misc_ptr_ + nAllocatedPoints_, num = 3
		// those were put into a single vector of 3 elements so that only one mem transfer will be needed for all three of them
		//  thrust::device_vector<int> out_of_space_;
		//  thrust::device_vector<int> node_count_;
		//  thrust::device_vector<int> nodes_allocated_;
		int* allocation_info_ptr_;
		std::vector<int> allocation_info_host_;
	};

}