#pragma once

#include "definations.h"
#include "DynamicFusionParam.h"
namespace dfusion
{
	class GpuMesh;
	class TsdfVolume;

	struct WarpNode
	{
		Tbx::Dual_quat_cu dq; // dg_se3 in the paper
		float4 v_w; // dg_v, dg_w in the paper
		float4 dummy; // free for now, may be used later
		__device__ __host__ void set(float4 a, float4 b, float4 c)
		{
			dq = Tbx::Dual_quat_cu(Tbx::Quat_cu(a.x, a.y, a.z, a.w),
				Tbx::Quat_cu(b.x, b.y, b.z, b.w));
			v_w = c;
		}
		__device__ __host__ void get(float4& a, float4& b, float4& c)
		{
			Tbx::Quat_cu q0 = dq.get_non_dual_part();
			Tbx::Quat_cu q1 = dq.get_dual_part();
			a = make_float4(q0.w(), q0.i(), q0.j(), q0.k());
			b = make_float4(q1.w(), q1.i(), q1.j(), q1.k());
			c = v_w;
		}
	};

	class WarpField
	{
	public:
		typedef uchar4 KnnIdx;
		enum{
			GraphLevelNum = 4,
			MaxNodeNum = 256,
			KnnK = 4, // num of KnnIdx
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
	protected:

	private:
		Param m_param;
		TsdfVolume* m_volume;
		Tbx::Transfo m_rigidTransform;

		int m_numNodes[GraphLevelNum];

		// store quaternion-translation parts:
		DeviceArray<WarpNode> m_nodesQuatTrans;
		
		// process the input GpuMesh
		int3 m_nodesGridSize;
		DeviceArray<float4> m_meshPointsSorted;
		DeviceArray<int> m_meshPointsKey;

		// type: KnnIdx
		cudaArray_t m_knnField;
	};

}