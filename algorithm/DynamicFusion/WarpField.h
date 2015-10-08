#pragma once
#undef min
#undef max
#include "definations.h"
#include "DynamicFusionParam.h"
#include <helper_math.h>
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
	__device__ __forceinline__ float norm2(float3 v)
	{
		return dot(v, v);
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

#if defined(__CUDACC__)
		__device__ __forceinline__ static IdxType get_by_arrayid(KnnIdx knn, int i)
		{
			return ((WarpField::IdxType*)(&knn))[i];
		}
		__device__ __forceinline__ static Tbx::Dual_quat_cu calc_dual_quat_blend_on_p(
			cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex, 
			float3 p, float3 origion, float invVoxelSize)
		{
			Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));

			float3 p1 = (make_float3(p.x, p.y, p.z) - origion)*invVoxelSize;
			int x = int(p1.x);
			int y = int(p1.y);
			int z = int(p1.z);
			KnnIdx knnIdx = make_ushort4(0, 0, 0, 0);
			tex3D(&knnIdx, knnTex, x, y, z);

			if (get_by_arrayid(knnIdx, 0) >= MaxNodeNum)
			{
				dq_blend = Tbx::Dual_quat_cu::identity();
			}
			else
			{
				Tbx::Dual_quat_cu dq0;
				for (int k = 0; k < KnnK; k++)
				{
					if (get_by_arrayid(knnIdx, k) < MaxNodeNum)
					{
						IdxType nn3 = get_by_arrayid(knnIdx, k) * 3;
						float4 q0, q1, vw;
						tex1Dfetch(&q0, nodesDqVwTex, nn3 + 0);
						tex1Dfetch(&q1, nodesDqVwTex, nn3 + 1);
						tex1Dfetch(&vw, nodesDqVwTex, nn3 + 2);
						// note: we store 1.f/radius in vw.w
						float w = __expf(-norm2(make_float3(vw.x - p.x, vw.y - p.y,
							vw.z - p.z)) * 2 * (vw.w*vw.w));
						Tbx::Dual_quat_cu dq = pack_dual_quat(q0, q1);
						if(k == 0)
							dq0 = dq;
						else
						{
							if (dq0.get_non_dual_part().dot(dq.get_non_dual_part()) < 0)
								w = -w;
						}
						dq_blend = dq_blend + dq*w;
					}
				}
				dq_blend.normalize();
			}
			return dq_blend;
		}
		__device__ __forceinline__ static Tbx::Dual_quat_cu calc_dual_quat_blend_on_voxel(
			cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex,
			int x, int y, int z, float3 origion, float voxelSize)
		{
			Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));

			float3 p = make_float3(x*voxelSize, y*voxelSize, z*voxelSize) + origion;
			KnnIdx knnIdx = make_ushort4(0, 0, 0, 0);
			tex3D(&knnIdx, knnTex, x, y, z);

			if (get_by_arrayid(knnIdx, 0) >= MaxNodeNum)
			{
				dq_blend = Tbx::Dual_quat_cu::identity();
			}
			else
			{
				for (int k = 0; k < KnnK; k++)
				{
					if (get_by_arrayid(knnIdx, k) < MaxNodeNum)
					{
						IdxType nn3 = get_by_arrayid(knnIdx, k) * 3;
						float4 q0, q1, vw;
						tex1Dfetch(&q0, nodesDqVwTex, nn3 + 0);
						tex1Dfetch(&q1, nodesDqVwTex, nn3 + 1);
						tex1Dfetch(&vw, nodesDqVwTex, nn3 + 2);
						// note: we store 1.f/radius in vw.w
						float w = __expf(-norm2(make_float3(vw.x - p.x, vw.y - p.y,
							vw.z - p.z)) * 2 * (vw.w*vw.w));
						Tbx::Dual_quat_cu dq = pack_dual_quat(q0, q1);
						dq_blend = dq_blend + dq*w;
					}
				}
				dq_blend.normalize();
			}
			return dq_blend;
		}
#endif
	public:
		WarpField();
		~WarpField();

		void init(TsdfVolume* volume, Param param);

		// update graph nodes based on a given mesh
		void updateWarpNodes(GpuMesh& src);

		// warp src to dst
		void warp(GpuMesh& src, GpuMesh& dst);
		void warp(const MapArr& srcVmap, const MapArr& srcNmap,
			MapArr& dstVmap, MapArr& dstNmap);

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

		cudaSurfaceObject_t bindKnnFieldSurface()const;
		void unBindKnnFieldSurface(cudaSurfaceObject_t t)const;
		cudaTextureObject_t bindKnnFieldTexture()const;
		void unBindKnnFieldTexture(cudaTextureObject_t t)const;
		cudaTextureObject_t bindNodesDqVwTexture()const;
		void unBindNodesDqVwTexture(cudaTextureObject_t t)const;

		const TsdfVolume* getVolume()const;

		// given a vertex map in volume coordinate, 
		// we extract the knn for vmap-2-level0-node,
		// NOTE: INVALID index is marked as the number of all nodes : getNumAllNodes()
		void extract_knn_for_vmap(const MapArr& vmap, DeviceArray2D<KnnIdx>& vmapKnn)const;

		// extract the knn among nodes
		// twist is the 6-tuple (\alpha,\beta,\gamma, t_x, t_y, t_z) from nodes dual-quaternions.
		// NOTE: INVALID index is marked as the number of all nodes: getNumAllNodes()
		void extract_nodes_info(DeviceArray<KnnIdx>& nodesKnn, DeviceArray<float>& twist,
			DeviceArray<float4>& vw)const;

		void update_nodes_via_twist(const DeviceArray<float>& twist);

		int getNumAllNodes()const{
			int n = 0;
			for (int k = 0; k < GraphLevelNum; k++)
				n += getNumNodesInLevel(k);
			return n;
		}

		void setActiveVisualizeNodeId(int id);
		int getActiveVisualizeNodeId()const;
	protected:
		void initKnnField();
		void insertNewNodes(GpuMesh& src);
		void updateAnnField();
		void updateGraph(int level);
	private:
		Param m_param;
		TsdfVolume* m_volume;
		Tbx::Transfo m_rigidTransform;
		int m_activeVisualizeNodeId;

		int m_lastNumNodes[GraphLevelNum];
		int m_numNodes[GraphLevelNum];

		// store quaternion-translation-vertex(x,y,z,1/w) parts:
		// note we store 1/w to avoid additonal divisions
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