#include "WarpField.h"
#include "TsdfVolume.h"
#include "GpuMesh.h"
namespace dfusion
{
	WarpField::WarpField()
	{
		m_volume = nullptr;
		m_knnField = nullptr;
	}

	WarpField::~WarpField()
	{
	}

	void WarpField::init(TsdfVolume* vol, Param param)
	{
		m_param = param;
		m_volume = vol;
		m_rigidTransform = Tbx::Transfo::identity();
		m_nodesQuatTrans.create(MaxNodeNum*GraphLevelNum*sizeof(WarpNode));
		for (int lv = 0; lv < GraphLevelNum; lv++)
			m_numNodes[lv] = 0;
		int3 res = m_volume->getResolution();
		float vsz = m_volume->getVoxelSize();

		// knn volume
		// malloc 3D texture
		if (m_knnField)
			cudaSafeCall(cudaFreeArray(m_knnField), "cudaFreeArray");
		cudaExtent ext = make_cudaExtent(res.x, res.y, res.z);
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<KnnIdx>();
		cudaSafeCall(cudaMalloc3DArray(&m_knnField, &desc, ext), "cudaMalloc3D");

		// nodes grid
		m_nodesGridSize = make_int3(divUp(res.x*vsz, m_param.warp_radius_search_epsilon),
			divUp(res.y*vsz, m_param.warp_radius_search_epsilon),
			divUp(res.z*vsz, m_param.warp_radius_search_epsilon));
	}
}