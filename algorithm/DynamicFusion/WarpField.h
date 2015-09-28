#pragma once

#include "definations.h"
#include "DynamicFusionParam.h"
namespace dfusion
{
	class GpuMesh;
	class TsdfVolume;
	class GpuKdTree;

	class WarpField
	{
	public:
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
		Tbx::Dual_quat_cu* getNodesDqPtr(int level){ return m_nodesQuatTrans.ptr() + MaxNodeNum*level; }
		const Tbx::Dual_quat_cu* getNodesDqPtr(int level)const{ return m_nodesQuatTrans.ptr() + MaxNodeNum*level; }
		float4* getNodesVwPtr(int level){ return m_nodesVW.ptr() + MaxNodeNum*level; }
		const float4* getNodesVwPtr(int level)const{ return m_nodesVW.ptr() + MaxNodeNum*level; }

		cudaSurfaceObject_t bindKnnFieldSurface();
		void unBindKnnFieldSurface(cudaSurfaceObject_t t);
		cudaTextureObject_t bindKnnFieldTexture();
		void unBindKnnFieldTexture(cudaTextureObject_t t);
	protected:
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
		DeviceArray<Tbx::Dual_quat_cu> m_nodesQuatTrans;
		DeviceArray<float4> m_nodesVW;
		
		// process the input GpuMesh
		int3 m_nodesGridSize;
		int m_current_point_buffer_size;
		DeviceArray<float4> m_meshPointsSorted;
		DeviceArray<int> m_meshPointsKey;
		DeviceArray<int> m_meshPointsFlags;

		GpuKdTree* m_nodeTree;

		// type: KnnIdx
		cudaArray_t m_knnField;
	};

}