#pragma once

#include "definations.h"
#include "DynamicFusionParam.h"
namespace dfusion
{
	class GpuMesh;
	class TsdfVolume;
	class GpuKdTree;

	__device__ __host__ __forceinline__ static Tbx::Quat_cu pack_quat(float4 a)
	{
		return Tbx::Quat_cu(a.x, a.y, a.z, a.w);
	}
	__device__ __host__ __forceinline__ static float4 unpack_quat(Tbx::Quat_cu a)
	{
		return make_float4(a.w(), a.i(), a.j(), a.k());
	}
	__device__ __host__ __forceinline__ static Tbx::Dual_quat_cu pack_dual_quat(float4 a, float4 b)
	{
		return Tbx::Dual_quat_cu(pack_quat(a), pack_quat(b));
	}
	__device__ __host__ __forceinline__ static void unpack_dual_quat(Tbx::Dual_quat_cu dq, float4& a, float4& b)
	{
		a = unpack_quat(dq.get_non_dual_part());
		b = unpack_quat(dq.get_dual_part());
	}

	class WarpField
	{
	public:
		typedef ushort IdxType;
		typedef ushort4 KnnIdx;
		enum{
			GraphLevelNum = 4,
			MaxNodeNum = 4096,
			KnnK = 4, // num of KnnIdx
			KnnInvalidId = MaxNodeNum,
		};
	public:
		WarpField();
		~WarpField();

		void init(TsdfVolume* volume, Param param);

		// update graph nodes based on a given mesh
		void updateWarpNodes(GpuMesh& src);

		// warp src to dst
		void warp(GpuMesh& src, GpuMesh& dst);

		Tbx::Transfo get_rigidTransform()const{ return m_rigidTransform; }
		void set_rigidTransform(Tbx::Transfo T){ m_rigidTransform = T; }

		int getNumLevels()const{ return GraphLevelNum; }
		int getNumNodesInLevel(int level)const{ return m_numNodes[level]; }

		// memory: [q0-q1-vw][q0-q1-vw]....
		float4* getNodesDqVwPtr(int level){ return m_nodesQuatTransVw.ptr() + 3*MaxNodeNum*level; }
		const float4* getNodesDqVwPtr(int level)const{ return m_nodesQuatTransVw.ptr() + 3*MaxNodeNum*level; }

		// each level is connected to k-nn of its coarser level
		KnnIdx* getNodesEdgesPtr(int level){ return m_nodesGraph.ptr() + MaxNodeNum*level; }
		const KnnIdx* getNodesEdgesPtr(int level)const{ return m_nodesGraph.ptr() + MaxNodeNum*level; }

		cudaSurfaceObject_t bindKnnFieldSurface();
		void unBindKnnFieldSurface(cudaSurfaceObject_t t);
		cudaTextureObject_t bindKnnFieldTexture();
		void unBindKnnFieldTexture(cudaTextureObject_t t);
		cudaTextureObject_t bindNodesDqVwTexture();
		void unBindNodesDqVwTexture(cudaTextureObject_t t);
	protected:
		void initKnnField();
		void insertNewNodes(GpuMesh& src);
		void updateAnnField();
		void updateGraph(int level);
	private:
		Param m_param;
		TsdfVolume* m_volume;
		Tbx::Transfo m_rigidTransform;

		int m_lastNumNodes[GraphLevelNum];
		int m_numNodes[GraphLevelNum];

		// store quaternion-translation parts:
		DeviceArray<float4> m_nodesQuatTransVw;//q0-q1-vw-q0-q1-vw...
		DeviceArray<KnnIdx> m_nodesGraph;//knn from level to level+1
		
		// process the input GpuMesh
		int3 m_nodesGridSize;
		int m_current_point_buffer_size;
		DeviceArray<float4> m_meshPointsSorted;
		DeviceArray<int> m_meshPointsKey;
		DeviceArray<int> m_meshPointsFlags;
		DeviceArray<float> m_tmpBuffer;

		GpuKdTree* m_nodeTree[GraphLevelNum];

		// type: KnnIdx
		cudaArray_t m_knnField;
	};

}