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

	class WarpField
	{
	public:
		enum{
			GraphLevelNum = 4,
			MaxNodeNum = 4096,
			KnnInvalidId = MaxNodeNum,
		};

#if defined(__CUDACC__)
		__device__ __forceinline__ static int sign(float a)
		{
			return (a > 0) - (a < 0);
		}
		__device__ __forceinline__ static Tbx::Dual_quat_cu calc_dual_quat_blend_on_p(
			cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex, 
			float3 p, float3 origion, float invVoxelSize)
		{
			Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));

			float3 p1 = (make_float3(p.x, p.y, p.z) - origion)*invVoxelSize;
			int x = __float2int_rn(p1.x);
			int y = __float2int_rn(p1.y);
			int z = __float2int_rn(p1.z);
			KnnIdx knnIdx = read_knn_tex(knnTex, x, y, z);

			if (knn_k(knnIdx, 0) >= MaxNodeNum)
			{
				dq_blend = Tbx::Dual_quat_cu::identity();
			}
			else
			{
				// the first quat
				float4 q0, q1, vw;
				int nn3;
				//Tbx::Dual_quat_cu dq_avg;
				nn3 = knn_k(knnIdx, 0) * 3;
				tex1Dfetch(&q0, nodesDqVwTex, nn3 + 0);
				tex1Dfetch(&q1, nodesDqVwTex, nn3 + 1);
				tex1Dfetch(&vw, nodesDqVwTex, nn3 + 2);
				float dist2_0 = norm2(make_float3(vw.x - p.x, vw.y - p.y, vw.z - p.z));

				dq_blend = pack_dual_quat(q0, q1);

				// the other quats
				for (int k = 1; k < KnnK; k++)
				{
					if(knn_k(knnIdx, k) < MaxNodeNum)
					{
						nn3 = knn_k(knnIdx, k) * 3;
						tex1Dfetch(&q0, nodesDqVwTex, nn3 + 0);
						tex1Dfetch(&q1, nodesDqVwTex, nn3 + 1);
						tex1Dfetch(&vw, nodesDqVwTex, nn3 + 2);
						Tbx::Dual_quat_cu dq = pack_dual_quat(q0, q1);

						// note: we store 1.f/radius in vw.w
						float dist2 = norm2(make_float3(vw.x - p.x, vw.y - p.y, vw.z - p.z));
						float w = __expf(-(dist2 - dist2_0) * 0.5f * vw.w * vw.w)
							*sign(dq_blend[0] * dq[0] + dq_blend[1] * dq[1] + 
							dq_blend[2] * dq[2] + dq_blend[3] * dq[3]);
						dq_blend += dq*w;
					}
				}
				dq_blend *= 1.f / dq_blend.norm();
				return dq_blend;
			}
			return dq_blend;
		}
#endif
	public:
		WarpField();
		~WarpField();

		void init(TsdfVolume* volume, Param param);
		void clear();

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

		cudaSurfaceObject_t getKnnFieldSurface()const;
		cudaTextureObject_t getKnnFieldTexture()const;
		cudaTextureObject_t getNodesDqVwTexture()const;

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

		// we will assume the memory are prepared and not call malloc inside
		void extract_nodes_info_no_allocation(DeviceArray<KnnIdx>& nodesKnn, DeviceArray<float>& twist,
			DeviceArray<float4>& vw)const;

		// we will assume that the given buffer is at least the matched size (but may be larger)
		void update_nodes_via_twist(const DeviceArray<float>& twist);

		int getNumAllNodes()const{
			int n = 0;
			for (int k = 0; k < GraphLevelNum; k++)
				n += getNumNodesInLevel(k);
			return n;
		}

		void setActiveVisualizeNodeId(int id);
		int getActiveVisualizeNodeId()const;

		// slow: copy from cuda array to a single host pos
		KnnIdx getKnnAt(float3 volumePos)const;
		KnnIdx getKnnAt(int3 gridXYZ)const;

		void save(const char* filename)const;
		void load(const char* filename);
	protected:
		void initKnnField();
		void insertNewNodes(GpuMesh& src);
		void updateAnnField();
		void updateGraph(int level);
		void updateGraph_singleLevel();
		void remove_small_graph_components();

		void bindKnnFieldSurface();
		void unBindKnnFieldSurface();
		void bindKnnFieldTexture();
		void unBindKnnFieldTexture();
		void bindNodesDqVwTexture();
		void unBindNodesDqVwTexture();
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

		cudaSurfaceObject_t m_knnFieldSurface;
		cudaTextureObject_t m_knnFieldTexture;
		cudaTextureObject_t m_nodesDqVeTexture;
	};

}