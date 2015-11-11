#include "SparseVolume.h"
#include <algorithm>
namespace dfusion
{
	SparseVolume::SparseVolume()
	{
		voxel_size_ = 0.f;
		origion_ = make_float3(0.f, 0.f, 0.f);
		resolution_ = make_int3(0, 0, 0);
		tranc_dist_ = 0.f;

		hash_entry_size_ = 0;
		voxel_block_size_ = 0;
		voxel_block_number_ = 0;
		selected_hash_entry_number_ = 0;
		selected_voxel_block_number_ = 0;
		active_region_offset_ = 0.f;
		active_region_radius_ = 0.f;
	}

	SparseVolume::~SparseVolume()
	{

	}

	void SparseVolume::init(int3 resolution, float voxel_size, float3 origion)
	{
		allocate(resolution, voxel_size, origion);
	}

	void SparseVolume::allocate(int3 resolution, float voxel_size, float3 origion)
	{
		tranc_dist_ = 0.f;
		resolution_ = resolution;
		origion_ = origion;
		setVoxelSize(voxel_size);
		setTsdfTruncDist(voxel_size_ * 5);

		hash_entry_size_ = 0x200000;
		voxel_block_size_ = 0x40000;
		assert(hash_entry_size_ % BUCKET_SIZE == 0);
		int	hash_bucket_size = hash_entry_size_ / BUCKET_SIZE;
		int parameter_table_size = 100;
		int selected_hash_entry_size = std::min(hash_entry_size_, voxel_block_size_);

		hash_entry_.create(hash_entry_size_);
		voxel_block_.create(voxel_block_size_);

		hash_bucket_atomic_lock_.create(hash_bucket_size);
		available_voxel_block_.create(voxel_block_size_);
		hash_parameters_.create(parameter_table_size);
		voxel_block_number_ = 0;

		selected_hash_entry_.create(selected_hash_entry_size);
		initMemoryi(hash_entry_scan_, hash_entry_size_);
		delete_hash_entry_.create(selected_hash_entry_size);
		selected_hash_entry_number_ = 0;

		selected_voxel_block_number_ = 0;
		selected_voxel_block_vertex_list_.create(selected_hash_entry_size * 8);

		active_region_offset_ = 2.0f;
		active_region_radius_ = 4.0f;
		voxel_block_buffer_.create(BUFFER_SIZE);
		chunk_on_CPU_.create(CHUNK_DIM_X * CHUNK_DIM_Y * CHUNK_DIM_Z);
		chunk_.resize(CHUNK_DIM_X * CHUNK_DIM_Y * CHUNK_DIM_Z);
		selected_hash_entry_h_.resize(selected_hash_entry_size);
		voxel_block_buffer_h_.resize(BUFFER_SIZE);

		//	init parameters by set data on CPU and upload to GPU
		HashEntry init_hash_entry;
		init_hash_entry.position[0] = 0;
		init_hash_entry.position[1] = 0;
		init_hash_entry.position[2] = 0;
		init_hash_entry.offset = 0;
		init_hash_entry.pointer = -1;
		std::vector<HashEntry> hash_entry_h;
		hash_entry_h.resize(hash_entry_size_);
		for (int i = 0; i<hash_entry_h.size(); i++)
			hash_entry_h[i] = init_hash_entry;
		hash_entry_.upload(hash_entry_h.data(), hash_entry_size_);

		std::vector<int> available_voxel_h;
		available_voxel_h.resize(voxel_block_size_);
		for (int i = 0; i<available_voxel_h.size(); i++)
			available_voxel_h[i] = i;
		available_voxel_block_.upload(available_voxel_h.data(), voxel_block_size_);

		std::vector<int> hash_parameters_h;
		hash_parameters_h.resize(parameter_table_size);
		hash_parameters_h[0] = 0;	//	a GPU copy of voxel_block_number
		hash_parameters_h[1] = 0;	//	a debug value
		hash_parameters_.upload(&hash_parameters_h[0], parameter_table_size);

		memset(chunk_.data(), NULL, sizeof(Chunk*)*chunk_.size());
		std::vector<unsigned char>		chunk_on_CPU_h;
		chunk_on_CPU_h.resize(CHUNK_DIM_X * CHUNK_DIM_Y * CHUNK_DIM_Z);
		memset(chunk_on_CPU_h.data(), 0, sizeof(unsigned char)*chunk_on_CPU_h.size());
		chunk_on_CPU_.upload(chunk_on_CPU_h.data(), chunk_on_CPU_.size());
	}

	void SparseVolume::setVoxelSize(float voxel_size)
	{
		voxel_size_ = voxel_size;
	}

	void SparseVolume::setTsdfTruncDist(float distance)
	{
		tranc_dist_ = distance;
	}

}