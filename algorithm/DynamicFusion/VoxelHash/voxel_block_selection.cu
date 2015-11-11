/***********************************************************/
/**	\file
	\brief		visible voxel block selection
	\details	
	\author		Yizhong Zhang
	\date		12/8/2013
*/
/***********************************************************/
#include "device_utils.h"
#include "voxel_hashing_device.h"
#include "voxel_hashing_internal.h"
#include <helper_math.h>

//	==================================================================

struct VisibilityMarker{
	dfusion::Intr	intr;			//	intrinsic of camera
	int		cols;			//	camera width
	int		rows;			//	camera height
	float	z_near;			//	the near plane of visible
	float	z_far;			//	the far plane of visible

	dfusion::Mat33	Rw2c;			//	world to camera
	float3	tw2c;

	float	block_size;		//	size of each voxel

	PtrSz<HashEntry> hash_entry;			//	the hash table, organized in sequence of each entry
	int* visible_entry_mark_d_ptr;			//	mark 1 if this block is visible, 0 if not visible

	__device__ __forceinline__ float3 GetCoord(int X, int Y, int Z) const{
		return make_float3(
			X * block_size,
			Y * block_size,
			Z * block_size );
	}

	__device__ __forceinline__ bool IsVisible(int X, int Y, int Z) const{
		//	calculate the coordinate of this point in camera coordinate
		float3 p = Rw2c * GetCoord(X, Y, Z) + tw2c;

		//	z not inside visible range
		if( p.z < z_near || p.z > z_far )
			return false;

		//	calculate projection onto the screen, if out of camera range, not visible
		float3 ukr = intr.xyz2uvd(X, Y, Z);
		if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows )
			return false;

		return true;
	}

	__device__ __forceinline__ void operator () () const{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if( idx >= hash_entry.size )
			return;

		//	this entry is not allocated yet
		if( hash_entry[idx].pointer < 0 ){
			visible_entry_mark_d_ptr[idx] = 0;
			return;
		}

		int X = hash_entry[idx].position[0];
		int Y = hash_entry[idx].position[1];
		int Z = hash_entry[idx].position[2];

		//	if one of the 8 corners of the cube is projected onto the screen, this voxel block is visible
		if( IsVisible(X+0, Y+0, Z+0) ||	
			IsVisible(X+1, Y+0, Z+0) ||
			IsVisible(X+0, Y+1, Z+0) ||
			IsVisible(X+1, Y+1, Z+0) ||
			IsVisible(X+0, Y+0, Z+1) ||
			IsVisible(X+1, Y+0, Z+1) ||
			IsVisible(X+0, Y+1, Z+1) ||
			IsVisible(X+1, Y+1, Z+1) )
			visible_entry_mark_d_ptr[idx] = 1;
		else
			visible_entry_mark_d_ptr[idx] = 0;
	}

};

__global__ void visibilityMarkerKernel (const VisibilityMarker marker) {
	marker ();
}

//	==================================================================

struct OutActiveRegionMarker{
	float3	active_region_center;
	float	active_region_radius;

	float3	chunk_min_xyz;			//	the min coordinate of the chunk table
	int3	chunk_dim;				//	the dimension of chunk table
	float	block_size;				//	size of each voxel

	PtrSz<HashEntry> hash_entry;			//	the hash table, organized in sequence of each entry
	int* selected_entry_mark_d_ptr;			//	mark 1 if this block is out of active region, 0 if not visible

	__device__ __forceinline__ float3 GetCoord(int X, int Y, int Z) const{
		return make_float3(
			X * block_size,
			Y * block_size,
			Z * block_size );
	}

	__device__ __forceinline__ bool IsOutOfActiveRegion(int X, int Y, int Z) const{
		float3 p = GetCoord(X, Y, Z);
		float3 r = p - active_region_center;

		return length(r) > active_region_radius;
	}

	__device__ __forceinline__ void operator () () const{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if( idx >= hash_entry.size )
			return;

		//	this entry is not allocated yet
		if( hash_entry[idx].pointer < 0 ){
			selected_entry_mark_d_ptr[idx] = 0;
			return;
		}

		int X = hash_entry[idx].position[0];
		int Y = hash_entry[idx].position[1];
		int Z = hash_entry[idx].position[2];

		if( IsOutOfActiveRegion(X, Y, Z) )
			selected_entry_mark_d_ptr[idx] = 1;
		else
			selected_entry_mark_d_ptr[idx] = 0;
	}

};

__global__ void outActiveRegionMarkerKernel (const OutActiveRegionMarker marker) {
	marker ();
}

//	==================================================================

struct InsideAABBMarker{
	float3	aabb_min;
	float3	aabb_max;

	float	block_size;				//	size of each voxel

	PtrSz<HashEntry> hash_entry;			//	the hash table, organized in sequence of each entry
	int* selected_entry_mark_d_ptr;			//	mark 1 if this block is out of active region, 0 if not visible

	__device__ __forceinline__ float3 GetCoord(int X, int Y, int Z) const{
		return make_float3(
			X * block_size,
			Y * block_size,
			Z * block_size );
	}

	__device__ __forceinline__ bool IsInsideAABB(int X, int Y, int Z) const{
		float3 p0 = GetCoord(X, Y, Z);
		float3 p1 = GetCoord(X+1, Y+1, Z+1);

		if( p1.x < aabb_min.x || p0.x > aabb_max.x ||
			p1.y < aabb_min.y || p0.y > aabb_max.y ||
			p1.z < aabb_min.z || p0.z > aabb_max.z )
			return false;

		return true;
	}

	__device__ __forceinline__ void operator () () const{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if( idx >= hash_entry.size )
			return;

		//	this entry is not allocated yet
		if( hash_entry[idx].pointer < 0 ){
			selected_entry_mark_d_ptr[idx] = 0;
			return;
		}

		int X = hash_entry[idx].position[0];
		int Y = hash_entry[idx].position[1];
		int Z = hash_entry[idx].position[2];

		if( IsInsideAABB(X+0, Y+0, Z+0) ||	
			IsInsideAABB(X+1, Y+1, Z+1) )
			selected_entry_mark_d_ptr[idx] = 1;
		else
			selected_entry_mark_d_ptr[idx] = 0;
	}

};

__global__ void insideAABBMarkerKernel (const InsideAABBMarker marker) {
	marker ();
}

//	==================================================================

__global__ void createSelectedEntry(
	PtrSz<HashEntry>	hash_entry,
	int*				hash_entry_scan_d_ptr,
	PtrSz<HashEntry>	selected_hash_entry)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx >= hash_entry.size - 1 )	//	we skip the last entry 
		return;

	int this_scan_value = hash_entry_scan_d_ptr[idx];
	int next_scan_value = hash_entry_scan_d_ptr[idx+1];
	if( this_scan_value < next_scan_value ){
		selected_hash_entry[this_scan_value] = hash_entry[idx];
	}
}


//	==================================================================

__global__ void createVoxelBlockVertexList(
	PtrSz<HashEntry>	hash_entry,
	float				block_size,
	float3*				selected_voxel_block_vertex_list_d,
	int					selected_hash_entry_number)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx >= selected_hash_entry_number )	
		return;

	int X = hash_entry[idx].position[0];
	int Y = hash_entry[idx].position[1];
	int Z = hash_entry[idx].position[2];
	selected_voxel_block_vertex_list_d[idx*8  ].x = block_size * X;
	selected_voxel_block_vertex_list_d[idx*8  ].y = block_size * Y;
	selected_voxel_block_vertex_list_d[idx*8  ].z = block_size * Z;
	selected_voxel_block_vertex_list_d[idx*8+1].x = block_size * (X+1);
	selected_voxel_block_vertex_list_d[idx*8+1].y = block_size * Y;
	selected_voxel_block_vertex_list_d[idx*8+1].z = block_size * Z;
	selected_voxel_block_vertex_list_d[idx*8+2].x = block_size * X;
	selected_voxel_block_vertex_list_d[idx*8+2].y = block_size * (Y+1);
	selected_voxel_block_vertex_list_d[idx*8+2].z = block_size * Z;
	selected_voxel_block_vertex_list_d[idx*8+3].x = block_size * (X+1);
	selected_voxel_block_vertex_list_d[idx*8+3].y = block_size * (Y+1);
	selected_voxel_block_vertex_list_d[idx*8+3].z = block_size * Z;

	selected_voxel_block_vertex_list_d[idx*8+4].x = block_size * X;
	selected_voxel_block_vertex_list_d[idx*8+4].y = block_size * Y;
	selected_voxel_block_vertex_list_d[idx*8+4].z = block_size * (Z+1);
	selected_voxel_block_vertex_list_d[idx*8+5].x = block_size * (X+1);
	selected_voxel_block_vertex_list_d[idx*8+5].y = block_size * Y;
	selected_voxel_block_vertex_list_d[idx*8+5].z = block_size * (Z+1);
	selected_voxel_block_vertex_list_d[idx*8+6].x = block_size * X;
	selected_voxel_block_vertex_list_d[idx*8+6].y = block_size * (Y+1);
	selected_voxel_block_vertex_list_d[idx*8+6].z = block_size * (Z+1);
	selected_voxel_block_vertex_list_d[idx*8+7].x = block_size * (X+1);
	selected_voxel_block_vertex_list_d[idx*8+7].y = block_size * (Y+1);
	selected_voxel_block_vertex_list_d[idx*8+7].z = block_size * (Z+1);
}

//	==================================================================

void selectVisibleHashEntry(
	thrust::device_vector<int>& hash_entry_scan,
	int&						visible_hash_entry_number,
	const dfusion::Intr&					intr, 
	int							cols, 
	int							rows,
	float						z_near, 
	float						z_far,
	const dfusion::Mat33&				Rw2c,
	const float3&				tw2c,
	float						block_size, 
	DeviceArray<HashEntry>&		hash_entry, 
	DeviceArray<HashEntry>&		visible_hash_entry,
	float3*						selected_voxel_block_vertex_list_d )
{
	int* hash_entry_scan_d_ptr = thrust::raw_pointer_cast( hash_entry_scan.data() );

	//	mark visible voxel blocks
	VisibilityMarker marker;
	marker.intr		= intr;
	marker.cols		= cols;
	marker.rows		= rows;
	marker.z_near	= z_near;
	marker.z_far	= z_far;

	marker.Rw2c		= Rw2c;
	marker.tw2c		= tw2c;
	
	marker.block_size = block_size;

	marker.hash_entry = hash_entry;
	marker.visible_entry_mark_d_ptr = hash_entry_scan_d_ptr;

	int threadPerBlock = 256;
	int blocksPerGrid = divUp(hash_entry.size(), threadPerBlock);
	visibilityMarkerKernel << <blocksPerGrid, threadPerBlock >> >(marker);
	cudaSafeCall(cudaGetLastError(), "selectVisibleHashEntry::visibilityMarkerKernel");

	//	scan
	scan_exclusive( hash_entry_scan );
	visible_hash_entry_number = hash_entry_scan.back();

	//	create visible hash entry table
	createSelectedEntry<<<blocksPerGrid, threadPerBlock>>>(
		hash_entry, hash_entry_scan_d_ptr, visible_hash_entry);
	cudaSafeCall(cudaGetLastError(), "selectVisibleHashEntry::createSelectedEntry");

	//	calculate vertex list
	if (selected_voxel_block_vertex_list_d && visible_hash_entry_number)
	{
		blocksPerGrid = divUp(visible_hash_entry_number, threadPerBlock);
		createVoxelBlockVertexList << <blocksPerGrid, threadPerBlock >> >(
			visible_hash_entry, block_size, selected_voxel_block_vertex_list_d, visible_hash_entry_number);
		cudaSafeCall(cudaGetLastError(), "selectVisibleHashEntry::createVoxelBlockVertexList");
	}
}


void selectOutActiveRegionHashEntry(
	thrust::device_vector<int>& hash_entry_scan,
	int&						out_active_region_hash_entry_number,
	float3						active_region_center,
	float						active_region_radius,
	float3						chunk_min_xyz,
	int3						chunk_dim,
	float						block_size, 
	DeviceArray<HashEntry>&		hash_entry, 
	DeviceArray<HashEntry>&		selected_hash_entry )
{
	int* hash_entry_scan_d_ptr = thrust::raw_pointer_cast( hash_entry_scan.data() );

	//	mark visible voxel blocks
	OutActiveRegionMarker marker;
	marker.active_region_center	= active_region_center;
	marker.active_region_radius	= active_region_radius;
	marker.chunk_min_xyz		= chunk_min_xyz;
	marker.chunk_dim			= chunk_dim;
	marker.block_size			= block_size;

	marker.hash_entry = hash_entry;
	marker.selected_entry_mark_d_ptr = hash_entry_scan_d_ptr;

	int threadPerBlock = 256;
	int blocksPerGrid = divUp(hash_entry.size(), threadPerBlock);
	outActiveRegionMarkerKernel<<<blocksPerGrid, threadPerBlock>>>( marker );

	//	scan
	scan_exclusive( hash_entry_scan );
	out_active_region_hash_entry_number= hash_entry_scan.back();

	//	create out of active region hash entry table
	createSelectedEntry<<<blocksPerGrid, threadPerBlock>>>(
		hash_entry, hash_entry_scan_d_ptr, selected_hash_entry);
}


void selectInsideAABBHashEntry(
	thrust::device_vector<int>& hash_entry_scan,
	int&						selected_hash_entry_number,
	const float3&				aabb_min,
	const float3&				aabb_max,
	float						block_size, 
	DeviceArray<HashEntry>&		hash_entry, 
	DeviceArray<HashEntry>&		selected_hash_entry )
{
	int* hash_entry_scan_d_ptr = thrust::raw_pointer_cast( hash_entry_scan.data() );

	//	mark visible voxel blocks
	InsideAABBMarker marker;
	marker.aabb_min				= aabb_min;
	marker.aabb_max				= aabb_max;
	marker.block_size			= block_size;

	marker.hash_entry					= hash_entry;
	marker.selected_entry_mark_d_ptr	= hash_entry_scan_d_ptr;

	int threadPerBlock = 256;
	int blocksPerGrid = divUp(hash_entry.size(), threadPerBlock);
	insideAABBMarkerKernel<<<blocksPerGrid, threadPerBlock>>>( marker );

	//	scan
	scan_exclusive( hash_entry_scan );
	selected_hash_entry_number = hash_entry_scan.back();

	//	create out of active region hash entry table
	createSelectedEntry<<<blocksPerGrid, threadPerBlock>>>(
		hash_entry, hash_entry_scan_d_ptr, selected_hash_entry);
}

