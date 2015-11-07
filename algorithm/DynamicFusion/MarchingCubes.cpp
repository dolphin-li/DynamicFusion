#include "MarchingCubes.h"
#include "TsdfVolume.h"
#include <algorithm>
#include "GpuMesh.h"
#include "ldp_basic_mat.h"

namespace dfusion
{
#undef min
#undef max
	MarchingCubes::MarchingCubes()
	{
		m_volume = nullptr;
	}

	MarchingCubes::~MarchingCubes()
	{
	}

	void MarchingCubes::init(const TsdfVolume* volume, Param param)
	{
		m_volume = volume;
		m_param = param;
		generate_tiles();

		// allocate memory
		if (m_tiles.size())
		{
			m_voxelVerts.create(m_tiles[0].max_num_activeVoxels);
			m_voxelVertsScan.create(m_tiles[0].max_num_activeVoxels);
			m_compVoxelArray.create(m_tiles[0].max_num_activeVoxels);
		}
	}

	void MarchingCubes::run(GpuMesh& mesh, float isoValue)
	{
		if (m_volume == nullptr)
			throw std::exception("Before MarchingCubes::run(), call init() first!");
		m_param.marching_cube_isoValue = isoValue;

		m_volTex = m_volume->getTexture();

		// count for each tile the number of vertices
		// if only one tile, we directly write the results
		// else, we write to some temp buffers and then gather
		for (size_t tid = 0; tid < m_tiles.size(); tid++)
		{
			if (m_tiles.size() == 1)
				process_a_tile(m_tiles[tid], mesh);
			else
				process_a_tile(m_tiles[tid], m_tiledMeshes[tid]);
		}// end for tid
		
		if (m_tiles.size() > 1)
		{
			// scan to get the start index of each tile
			for (size_t tid = 1; tid < m_tiles.size(); tid++)
				m_tiles[tid].vert_start_id = m_tiles[tid - 1].vert_start_id + m_tiles[tid - 1].nverts;

			// allocate memory
			mesh.create(m_tiles.back().vert_start_id + m_tiles.back().nverts);

			// gather all
			mesh.lockVertsNormals();
			for (size_t tid = 0; tid < m_tiles.size(); tid++)
			{
				m_tiledMeshes[tid].lockVertsNormals();
				cudaMemcpy(mesh.verts() + m_tiles[tid].vert_start_id, 
					m_tiledMeshes[tid].verts(), m_tiledMeshes[tid].num()*sizeof(GpuMesh::PointType),
					cudaMemcpyDeviceToDevice);
				cudaMemcpy(mesh.normals() + m_tiles[tid].vert_start_id,
					m_tiledMeshes[tid].normals(), m_tiledMeshes[tid].num()*sizeof(GpuMesh::PointType),
					cudaMemcpyDeviceToDevice);
				m_tiledMeshes[tid].unlockVertsNormals();
			}
			mesh.unlockVertsNormals();
		}// end if m_tiles.size() > 1
	}

	void MarchingCubes::generate_tiles()
	{
		m_tiles.clear();
		const float vsz = m_volume->getVoxelSize();
		const int3 res = m_volume->getResolution();
		const float3 ori = m_volume->getOrigion();
		for (int z = 0; z < res.z; z += m_param.marching_cube_tile_size)
		{
			for (int y = 0; y < res.y; y += m_param.marching_cube_tile_size)
			{
				for (int x = 0; x < res.x; x += m_param.marching_cube_tile_size)
				{
					Tile tile;
					tile.begin = make_int3(x, y, z);
					tile.end = make_int3(std::min(res.x, x + m_param.marching_cube_tile_size),
						std::min(res.y, y + m_param.marching_cube_tile_size),
						std::min(res.z, z + m_param.marching_cube_tile_size));
					tile.origion = ori;
					tile.voxelSize = vsz;
					tile.level = m_param.marching_cube_level;

					tile.vert_start_id = 0;
					tile.nverts = 0;
					tile.num_voxels = ((tile.end.x - tile.begin.x) >> m_param.marching_cube_level)*
						((tile.end.y - tile.begin.y) >> m_param.marching_cube_level) * 
						((tile.end.z - tile.begin.z) >> m_param.marching_cube_level);
					tile.num_activeVoxels = 0;
					tile.max_num_activeVoxels = ceil(tile.num_voxels * m_param.marching_cube_max_activeVoxel_ratio);

					m_tiles.push_back(tile);
				}
			}// y
		}// z
		m_tiledMeshes.resize(m_tiles.size());
	}

	void MarchingCubes::process_a_tile(Tile& tile, GpuMesh& result)
	{
		classifyVoxel(tile);
		generateTriangles(tile, result);
	}
}