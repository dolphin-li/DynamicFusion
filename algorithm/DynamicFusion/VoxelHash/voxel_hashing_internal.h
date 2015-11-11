/***********************************************************/
/**	\file
	\brief		voxel hashing related data
	\details	
	\author		Yizhong Zhang
	\date		12/7/2013
*/
/***********************************************************/
#ifndef	__VOXEL_HASHING_H__
#define __VOXEL_HASHING_H__

#include "cuda_runtime_api.h"
#include <thrust/device_vector.h>
#include "definations.h"

#define BUCKET_SIZE		2						///	two entries each bucket
#define BLOCK_DIM		8						///	each block has 8*8*8 voxels

#define	BUFFER_SIZE		65536					///	size of the buffer for GPU-host streaming
#define CHUNK_SIZE		0.64f					///	the size of each chunk
#define CHUNK_DIM_X		256						///	dimension of chunk
#define CHUNK_DIM_Y		256
#define CHUNK_DIM_Z		256
#define CHUNK_MIN_X		(- CHUNK_DIM_X*CHUNK_SIZE*0.5f)		///	min coordinate of the chunks
#define CHUNK_MIN_Y		(- CHUNK_DIM_X*CHUNK_SIZE*0.5f)
#define CHUNK_MIN_Z		(- CHUNK_DIM_X*CHUNK_SIZE*0.5f)

//	pack
#if (defined(_WIN32) || defined(__WIN32__))
#	pragma pack(push, 1)
#else
#	pragma pack(1)
#endif

/**
	one hash entry
*/
struct HashEntry{
	short			position[3];
	short			offset;
	int				pointer;
};


/**
	data stored in one voxel
*/
struct Voxel{
	float			sdf;
	unsigned char	colorRGB[3];
	unsigned char	weight;
};


/**
	a voxel block
*/
struct VoxelBlock{
	Voxel	voxel[BLOCK_DIM * BLOCK_DIM * BLOCK_DIM];
};


//	unpack
#if (defined(_WIN32) || defined(__WIN32__))
#	pragma pack(pop)
#else
#	pragma pack()
#endif



/**
	functions to operate thrust device vector
*/
extern "C" void initMemoryc(thrust::device_vector<char>& vec, int size);
extern "C" void initMemoryf(thrust::device_vector<float>& vec, int size);
extern "C" void initMemoryi(thrust::device_vector<int>& vec, int size);
extern "C" void initMemoryull(thrust::device_vector<unsigned long long>& vec, int size);

void scan_inclusive(thrust::device_vector<int>& vec_d);
void scan_exclusive(thrust::device_vector<int>& vec_d);

void scan_inclusive(thrust::device_vector<int>& vec_d, int data_num);
void scan_exclusive(thrust::device_vector<int>& vec_d, int data_num);

extern "C" void scan_inclusiveull(thrust::device_vector<unsigned long long>& vec_d);
extern "C" void scan_exclusiveull(thrust::device_vector<unsigned long long>& vec_d);


/**
	alloc voxel blocks near the depth samples
*/
void allocVoxelBlock(
	const PtrStepSz<float>&		depth, 
	const dfusion::Intr&		intr, 
	const dfusion::Mat33&		Rc2w,
	const float3&				tc2w,
	float						block_size, 
	float						trunc_dist,
	DeviceArray<HashEntry>&		hash_entry, 
	int							bucket_size,
	DeviceArray<unsigned int>&	hash_bucket_atomic_lock, 
	DeviceArray<VoxelBlock>&	voxel_block,
	DeviceArray<int>&			available_voxel_block, 
	DeviceArray<int>&			hash_parameters,
	int&						voxel_block_number,
	int3						chunk_dim,
	float3						chunk_min_xyz,
	float						chunk_size,
	DeviceArray<unsigned char>&	chunk_on_CPU );

/**
	select all visible hash entry into visible hash entry
*/
void selectVisibleHashEntry(
	thrust::device_vector<int>& hash_entry_scan,
	int&						visible_hash_entry_number,
	const dfusion::Intr&		intr, 
	int							cols, 
	int							rows,
	float						z_near, 
	float						z_far,
	const dfusion::Mat33&		Rw2c,
	const float3&				tw2c,
	float						block_size, 
	DeviceArray<HashEntry>&		hash_entry, 
	DeviceArray<HashEntry>&		visible_hash_entry,
	float3*						selected_voxel_block_vertex_list_d );


/**
	update voxel block (fuse depth into volume)
*/
void updateVoxelBlock(
	const PtrStepSz<float>&			depth, 
	const dfusion::Intr&			intr,
	const dfusion::Mat33&			Rw2c,
	const float3&					tw2c,
	float							block_size, 
	float							voxel_size,
	float							trunc_dist,
	DeviceArray<HashEntry>&			hash_entry, 
	int								bucket_size,
	DeviceArray<unsigned int>&		hash_bucket_atomic_lock, 
	DeviceArray<VoxelBlock>&		voxel_block,
	DeviceArray<int>&				available_voxel_block, 
	DeviceArray<int>&				hash_parameters,
	const DeviceArray<HashEntry>&	visible_hash_entry,
	int								visible_hash_entry_number,
	DeviceArray<unsigned char>		delete_hash_entry,
	float							abs_tsdf_thre );

/**
	extract surface by ray casting

	we extract id map, weight map, block and voxel map at the same time
*/
void extractSurface(
	const dfusion::Intr&			intr,
	int								cols,
	int								rows,
	const dfusion::Mat33&			Rc2w,
	const float3&					tc2w,
	float							block_size, 
	float							voxel_size,
	float							trunc_dist,
	const DeviceArray<VoxelBlock>&	voxel_block,
	const DeviceArray2D<float>&		min_depth,
	const DeviceArray2D<float>&		max_depth,
	dfusion::MapArr&				v_map,
	dfusion::MapArr&				n_map,
	dfusion::DepthMap&				d_map,
	DeviceArray2D<int>&				id_map,
	DeviceArray2D<unsigned char>&	weight_map,
	DeviceArray2D<int>&				block_id_map,
	DeviceArray2D<int>&				voxel_id_map,
	DeviceArray<HashEntry>&			hash_entry,
	int								bucket_size );

/**
	write id map to voxel
*/
void writeIdMapToVoxel(	
	DeviceArray<VoxelBlock>&		voxel_block,
	const DeviceArray2D<int>&		id_map,
	const DeviceArray2D<int>&		block_id_map,
	const DeviceArray2D<int>&		voxel_id_map );


/**
	select all out of active region hash entry into selected hash entry	
*/
void selectOutActiveRegionHashEntry(
	thrust::device_vector<int>& hash_entry_scan,
	int&						out_active_region_hash_entry_number,
	float3						active_region_center,
	float						active_region_radius,
	float3						chunk_min_xyz,
	int3						chunk_dim,
	float						block_size, 
	DeviceArray<HashEntry>&		hash_entry, 
	DeviceArray<HashEntry>&		selected_hash_entry );

/**
	select all visible hash entry into visible hash entry
*/
void selectInsideAABBHashEntry(
	thrust::device_vector<int>& hash_entry_scan,
	int&						selected_hash_entry_number,
	const float3&				aabb_min,
	const float3&				aabb_max,
	float						block_size, 
	DeviceArray<HashEntry>&		hash_entry, 
	DeviceArray<HashEntry>&		selected_hash_entry );


/**
	stream selected entry and voxel data from GPU to buffer
*/
void streamGPU2Buffer(
	DeviceArray<HashEntry>&			hash_entry, 
	int								bucket_size,
	DeviceArray<unsigned int>&		hash_bucket_atomic_lock, 
	DeviceArray<VoxelBlock>&		voxel_block,
	DeviceArray<VoxelBlock>&		voxel_block_buffer,
	DeviceArray<int>&				available_voxel_block, 
	DeviceArray<int>&				hash_parameters,
	const DeviceArray<HashEntry>&	selected_hash_entry,
	int								selected_hash_entry_number,
	int3							chunk_dim,
	float3							chunk_min_xyz,
	float							chunk_size,
	float							block_size, 
	DeviceArray<unsigned char>&		chunk_on_CPU );

/**
	stream selected entry and voxel data from buffer to GPU
*/
void streamBuffer2GPU(
	DeviceArray<HashEntry>&			hash_entry, 
	int								bucket_size,
	DeviceArray<unsigned int>&		hash_bucket_atomic_lock, 
	DeviceArray<VoxelBlock>&		voxel_block,
	DeviceArray<VoxelBlock>&		voxel_block_buffer,
	DeviceArray<int>&				available_voxel_block, 
	DeviceArray<int>&				hash_parameters,
	const DeviceArray<HashEntry>&	selected_hash_entry,
	int								selected_hash_entry_number,
	DeviceArray<unsigned char>&		chunk_on_CPU,
	int								chunk_idx );


#endif
