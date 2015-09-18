#pragma once

#include "definations.h"

namespace dfusion
{
	class GpuMesh;
	class TsdfVolume;
	class MarchingCubes
	{
	public:
		struct Tile
		{
			int3 begin;
			int3 end;
			float3 origion;
			float voxelSize;
			int level;	// w.r.t. a voxel, volume.voxelSize<<level
			int nverts;
			int vert_start_id;
			int num_voxels;
			int num_activeVoxels;
		};
	public:
		MarchingCubes();
		~MarchingCubes();

		// tile_size: the algorithm is tiled, each tile is processed and then another
		// step: w.r.t. voxel, level=1 means calculated on a coarser levelx2
		void init(const TsdfVolume* volume, int tile_size, int level);

		void run(GpuMesh& mesh, float isoValue = 0.f);
	protected:
		void generate_tiles();
		void process_a_tile(Tile& tile, GpuMesh& result);

		void classifyVoxel(Tile& tile);
		void generateTriangles(const Tile& tile, GpuMesh& result);
	private:
		const TsdfVolume* m_volume;
		int m_tile_size;
		int m_level;
		float m_isoValue;
		std::vector<Tile> m_tiles;

		// buffered for each tile
		DeviceArray<unsigned int> m_voxelVerts;
		DeviceArray<unsigned int> m_voxelVertsScan;
		DeviceArray<unsigned int> m_voxelOccupied;
		DeviceArray<unsigned int> m_voxelOccupiedScan;
		DeviceArray<unsigned int> m_compVoxelArray;
		cudaTextureObject_t m_volTex;
	};
}
