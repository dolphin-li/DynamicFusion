#pragma once

#include "definations.h"

#include "voxelhash\voxel_hashing_internal.h"
#include <thrust/device_vector.h>
#include "voxelhash\chunk.h"

namespace dfusion
{
	class SparseVolume
	{
		friend class DynamicFusionProcessor;
	public:
		SparseVolume();
		~SparseVolume();

		void init(int3 resolution, float voxel_size, float3 origion);

		/**
		set the size of each voxel
		*/
		void setVoxelSize(float voxel_size);
		float getVoxelSize()const{ return voxel_size_; }
		float getBlockSize()const{ return getVoxelSize()*BLOCK_DIM; }

		/** \brief Sets Tsdf truncation distance. Must be greater than 2 * volume_voxel_size
		* \param[in] distance TSDF truncation distance
		*/
		void setTsdfTruncDist(float distance);
		float getTsdfTruncDist()const{ return tranc_dist_; }

	protected:
		void allocate(int3 resolution, float voxel_size, float3 origion);
	private:
		float voxel_size_;
		float3 origion_;
		int3 resolution_;
		float tranc_dist_;

		int hash_entry_size_;
		int voxel_block_size_;
		int voxel_block_number_;
		int selected_hash_entry_number_;
		int selected_voxel_block_number_;
		float active_region_offset_;
		float active_region_radius_;
		DeviceArray<HashEntry> hash_entry_;
		DeviceArray<HashEntry> selected_hash_entry_;
		DeviceArray<VoxelBlock> voxel_block_;
		DeviceArray<VoxelBlock> voxel_block_buffer_;
		DeviceArray<unsigned int> hash_bucket_atomic_lock_;
		DeviceArray<int> available_voxel_block_;
		DeviceArray<int> hash_parameters_;
		thrust::device_vector<int> hash_entry_scan_;
		DeviceArray<unsigned char> delete_hash_entry_;
		DeviceArray<unsigned char> chunk_on_CPU_;
		DeviceArray<float3> selected_voxel_block_vertex_list_;
		std::vector<Chunk*> chunk_;
		std::vector<HashEntry> selected_hash_entry_h_;
		std::vector<VoxelBlock> voxel_block_buffer_h_;
	};
}
