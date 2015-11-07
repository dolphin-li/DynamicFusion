/***********************************************************/
/**	\file
	\brief		voxel block streaming
	\details	
	\author		Yizhong Zhang
	\date		12/12/2013
*/
/***********************************************************/
#include "device_utils.h"
#include "voxel_hashing_device.h"
#include "voxel_hashing_internal.h"
#include "voxel_block_hash_table.cuh"

//	==================================================================

struct GPU2BufferStreamer : public VoxelBlockHashTable{
	PtrSz<VoxelBlock>			voxel_block;				//	the array of voxel block

	mutable PtrSz<HashEntry>	selected_hash_entry;	
	mutable PtrSz<VoxelBlock>	voxel_block_buffer;			//	buffer to copy voxel block data

	__device__ __forceinline__ void operator() () const{
		int entry_idx = blockIdx.x;
		int voxel_idx = threadIdx.x;

		int pointer = selected_hash_entry[entry_idx].pointer;
		voxel_block_buffer[entry_idx].voxel[voxel_idx] = voxel_block[pointer].voxel[voxel_idx];

		if( voxel_idx == 0 ){//	only the first thread is used to operate hash entry, others are used to copy data
			int X = selected_hash_entry[entry_idx].position[0];
			int Y = selected_hash_entry[entry_idx].position[1];
			int Z = selected_hash_entry[entry_idx].position[2];

			//	label this chunk exist on CPU
			int chunk_idx = GetChunkIdx(X, Y, Z);
			chunk_on_CPU[chunk_idx] = 1;

			//	delete from hash table
			int ret = DeleteHashEntry(X, Y, Z);	//	delete doesn't effect voxel data and selected_hash_entry value, so no need to sycn

			if( ret != 1 ){//	delete hash entry failed, this should never happen, just for debug purpose
				printf("error: GPU2BufferStreamer, delete hash entry failed");
			}
		}
	}

};


__global__ void streamGPU2BufferKernel (const GPU2BufferStreamer streamer) {
	streamer ();
}

//	==================================================================

struct Buffer2GPUStreamer : public VoxelBlockHashTable{
	mutable PtrSz<VoxelBlock>	voxel_block;				//	the array of voxel block

	int							chunk_idx;					//	the index of chunk that streaming
	mutable PtrSz<HashEntry>	selected_hash_entry;	
	PtrSz<VoxelBlock>			voxel_block_buffer;			//	buffer to copy voxel block data

	__device__ __forceinline__ void operator() () const{
		int entry_idx = blockIdx.x;
		int voxel_idx = threadIdx.x;

		//	Insert Hash Entry no stagger
		if( voxel_idx == 0 ){
			//if( chunk_idx < 0 || chunk_idx >= CHUNK_DIM_X*CHUNK_DIM_Y*CHUNK_DIM_Z )
			//	printf("error chunk idx: %d\n", chunk_idx);
			chunk_on_CPU[chunk_idx] = 0;

			int entry_id;
			int X = selected_hash_entry[entry_idx].position[0];
			int Y = selected_hash_entry[entry_idx].position[1];
			int Z = selected_hash_entry[entry_idx].position[2];
			int ret = InsertHashEntry(entry_id, X, Y, Z);
			if( ret == 1 ){
				selected_hash_entry[entry_idx].pointer = hash_entry[entry_id].pointer;
			}
			else{
				selected_hash_entry[entry_idx].pointer = - 1;
				printf("error: Buffer2GPUStreamer : public VoxelBlockHashTable, insert failed, ret val: %d", ret);
			}
		}

		__syncthreads();

		int pointer = selected_hash_entry[entry_idx].pointer;
		if( pointer >= 0 ){
			voxel_block[pointer].voxel[voxel_idx] = voxel_block_buffer[entry_idx].voxel[voxel_idx];
		}
	}

};


__global__ void streamBuffer2GPUKernel (const Buffer2GPUStreamer streamer) {
	streamer ();
}

//	==================================================================

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
	DeviceArray<unsigned char>&		chunk_on_CPU )
{
	//	setup block bucket atomic lock
	int threadPerBlock = 256;
	int blocksPerGrid = divUp(hash_bucket_atomic_lock.size(), threadPerBlock);

	initHashBucketAtomicLock<<<blocksPerGrid, threadPerBlock>>>(hash_bucket_atomic_lock);


	//	alloc block
	GPU2BufferStreamer streamer;

	streamer.hash_table_size			= hash_entry.size() / bucket_size;
	streamer.bucket_size				= bucket_size;
	streamer.hash_entry					= hash_entry;
	streamer.hash_bucket_atomic_lock	= hash_bucket_atomic_lock;
	streamer.available_voxel_block		= available_voxel_block;
	streamer.hash_parameters			= hash_parameters;

	streamer.voxel_block				= voxel_block;
	streamer.selected_hash_entry		= selected_hash_entry;
	streamer.voxel_block_buffer			= voxel_block_buffer;

	streamer.chunk_dim					= chunk_dim;
	streamer.chunk_min_xyz				= chunk_min_xyz;
	streamer.chunk_size					= chunk_size;
	streamer.block_size					= block_size;
	streamer.chunk_on_CPU				= chunk_on_CPU;

	threadPerBlock	= BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;
	blocksPerGrid	= selected_hash_entry_number;

	streamGPU2BufferKernel<<<blocksPerGrid, threadPerBlock>>>(streamer);
}


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
	int								chunk_idx )
{
	//	setup block bucket atomic lock
	int threadPerBlock = 256;
	int blocksPerGrid = divUp(hash_bucket_atomic_lock.size(), threadPerBlock);

	initHashBucketAtomicLock<<<blocksPerGrid, threadPerBlock>>>(hash_bucket_atomic_lock);


	//	alloc block
	Buffer2GPUStreamer streamer;

	streamer.hash_table_size			= hash_entry.size() / bucket_size;
	streamer.bucket_size				= bucket_size;
	streamer.hash_entry					= hash_entry;
	streamer.hash_bucket_atomic_lock	= hash_bucket_atomic_lock;
	streamer.available_voxel_block		= available_voxel_block;
	streamer.hash_parameters			= hash_parameters;
	streamer.chunk_on_CPU				= chunk_on_CPU;

	streamer.voxel_block				= voxel_block;
	streamer.chunk_idx					= chunk_idx;
	streamer.selected_hash_entry		= selected_hash_entry;
	streamer.voxel_block_buffer			= voxel_block_buffer;

	threadPerBlock	= BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;
	blocksPerGrid	= selected_hash_entry_number;

	streamBuffer2GPUKernel<<<blocksPerGrid, threadPerBlock>>>(streamer);
}




