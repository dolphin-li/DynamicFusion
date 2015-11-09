#include "WarpField.h"
#include "TsdfVolume.h"
#include "GpuMesh.h"
#include "GpuKdTree.h"
#include "ldp_basic_mat.h"
namespace dfusion
{
	WarpField::WarpField()
	{
		m_volume = nullptr;
		m_knnField = nullptr;
		for (int k = 0; k < GraphLevelNum; k++)
			m_nodeTree[k] = nullptr;
		m_current_point_buffer_size = 0;
		memset(m_numNodes, 0, sizeof(m_numNodes));
		memset(m_lastNumNodes, 0, sizeof(m_lastNumNodes));
		m_activeVisualizeNodeId = -1;
		m_knnFieldSurface = 0;
		m_knnFieldTexture = 0;
		m_nodesDqVeTexture = 0;
	}

	WarpField::~WarpField()
	{
		clear();
	}

	void WarpField::init(TsdfVolume* vol, Param param)
	{
		m_param = param;
		m_volume = vol;
		m_rigidTransform = Tbx::Transfo::identity();
		m_nodesQuatTransVw.create(MaxNodeNum*GraphLevelNum * 3);
		m_nodesGraph.create(MaxNodeNum*GraphLevelNum);
		cudaMemset(m_nodesQuatTransVw.ptr(), 0, m_nodesQuatTransVw.size()*m_nodesQuatTransVw.elem_size);
		cudaMemset(m_nodesGraph.ptr(), 0, m_nodesGraph.size()*m_nodesGraph.elem_size);
		unBindNodesDqVwTexture();
		bindNodesDqVwTexture();
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
		cudaSafeCall(cudaMalloc3DArray(&m_knnField, &desc, ext), "WarpField::init, cudaMalloc3D");
		unBindKnnFieldSurface();
		unBindKnnFieldTexture();
		bindKnnFieldSurface();
		bindKnnFieldTexture();
		initKnnField();

		for (int k = 0; k < GraphLevelNum; k++)
		{
			if (m_nodeTree[k])
				delete m_nodeTree[k];
			m_nodeTree[k] = new GpuKdTree();
		}

		// nodes grid
		m_nodesGridSize = make_int3(std::ceil(res.x*vsz/m_param.warp_radius_search_epsilon),
			std::ceil(res.y*vsz/m_param.warp_radius_search_epsilon),
			std::ceil(res.z*vsz/m_param.warp_radius_search_epsilon));
	}

	void WarpField::clear()
	{
		unBindNodesDqVwTexture();
		unBindKnnFieldSurface();
		unBindKnnFieldTexture();
		for (int k = 0; k < GraphLevelNum; k++)
		{
			if (m_nodeTree[k])
			{
				delete m_nodeTree[k];
				m_nodeTree[k] = nullptr;
			}
		}
		if (m_knnField)
		{
			cudaSafeCall(cudaFreeArray(m_knnField), "cudaFreeArray");
			m_knnField = nullptr;
		}
		m_nodesQuatTransVw.release();
		m_nodesGraph.release();

		m_meshPointsSorted.release();
		m_meshPointsKey.release();
		m_meshPointsFlags.release();
		m_tmpBuffer.release();

		m_current_point_buffer_size = 0;
		memset(m_numNodes, 0, sizeof(m_numNodes));
		memset(m_lastNumNodes, 0, sizeof(m_lastNumNodes));
		m_activeVisualizeNodeId = -1;
	}

	cudaSurfaceObject_t WarpField::getKnnFieldSurface()const
	{
		return m_knnFieldSurface;
	}

	cudaTextureObject_t WarpField::getKnnFieldTexture()const
	{
		return m_knnFieldTexture;
	}

	cudaTextureObject_t WarpField::getNodesDqVwTexture()const
	{
		return m_nodesDqVeTexture;
	}

	void WarpField::updateWarpNodes(GpuMesh& src)
	{
		memcpy(m_lastNumNodes, m_numNodes, sizeof(m_numNodes));

		insertNewNodes(src);

		if (m_param.graph_single_level)
		{
			updateGraph_singleLevel();
		}
		else
		{
			for (int lv = 1; lv < GraphLevelNum; lv++)
				updateGraph(lv);
		}

		remove_small_graph_components();

		if (m_lastNumNodes[0] != m_numNodes[0])
			updateAnnField();
	}

	void WarpField::bindKnnFieldSurface()
	{
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = m_knnField;
		cudaSafeCall(cudaCreateSurfaceObject(&m_knnFieldSurface, &surfRes),
			"WarpField::bindKnnFieldSurface");
	}

	void WarpField::unBindKnnFieldSurface()
	{
		cudaSafeCall(cudaDestroySurfaceObject(m_knnFieldSurface),
			"WarpField::unBindKnnFieldSurface");
	}

	void WarpField::bindKnnFieldTexture()
	{
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = m_knnField;
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = 0;
		texDescr.filterMode = cudaFilterModePoint;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		cudaSafeCall(cudaCreateTextureObject(&m_knnFieldTexture, &texRes, &texDescr, NULL),
			"WarpField::bindKnnFieldTexture");
	}

	void WarpField::unBindKnnFieldTexture()
	{
		cudaSafeCall(cudaDestroyTextureObject(m_knnFieldTexture),
			"WarpField::unBindKnnFieldTexture");
	}

	void WarpField::bindNodesDqVwTexture()
	{
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeLinear;
		texRes.res.linear.devPtr = (void*)m_nodesQuatTransVw.ptr();
		texRes.res.linear.desc = cudaCreateChannelDesc<float4>();
		texRes.res.linear.sizeInBytes = m_nodesQuatTransVw.size()*sizeof(float4);
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = 0;
		texDescr.filterMode = cudaFilterModePoint;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		cudaSafeCall(cudaCreateTextureObject(&m_nodesDqVeTexture, &texRes, &texDescr, NULL),
			"WarpField::bindNodesDqVwTexture");
	}

	void WarpField::unBindNodesDqVwTexture()
	{
		cudaSafeCall(cudaDestroyTextureObject(m_nodesDqVeTexture),
			"WarpField::unBindNodesDqVwTexture");
	}

	const TsdfVolume* WarpField::getVolume()const
	{
		return m_volume;
	}

	void WarpField::setActiveVisualizeNodeId(int id)
	{
		m_activeVisualizeNodeId = min(getNumNodesInLevel(0)-1, id);
	}

	int WarpField::getActiveVisualizeNodeId()const
	{
		return m_activeVisualizeNodeId;
	}

	void WarpField::save(const char* filename)const
	{
		FILE* pFile = fopen(filename, "wb");
		if (!pFile)
			throw std::exception(("save failed" + std::string(filename)).c_str());

		fwrite(&m_rigidTransform, sizeof(m_rigidTransform), 1, pFile);
		fwrite(m_numNodes, sizeof(m_numNodes), 1, pFile);

		std::vector<float4> tmp;
		m_nodesQuatTransVw.download(tmp);
		int ntmp = tmp.size();
		fwrite(&ntmp, sizeof(int), 1, pFile);
		fwrite(tmp.data(), sizeof(float4), tmp.size(), pFile);

		std::vector<KnnIdx> tmpIdx;
		m_nodesGraph.download(tmpIdx);
		int ntmpidx = tmpIdx.size();
		fwrite(&ntmpidx, sizeof(int), 1, pFile);
		fwrite(tmpIdx.data(), sizeof(KnnIdx), tmpIdx.size(), pFile);

		fclose(pFile);
	}

	void WarpField::load(const char* filename)
	{
		if (m_volume == nullptr)
			throw std::exception("Error: not initialzied before loading WarpField!");

		init(m_volume, m_param);

		FILE* pFile = fopen(filename, "rb");
		if (!pFile)
			throw std::exception(("load failed" + std::string(filename)).c_str());
	
		memset(m_lastNumNodes, 0, sizeof(m_lastNumNodes));
		fread(&m_rigidTransform, sizeof(m_rigidTransform), 1, pFile);
		fread(m_numNodes, sizeof(m_numNodes), 1, pFile);
		
		std::vector<float4> tmp(m_nodesQuatTransVw.size(), make_float4(0.f,0.f,0.f,0.f));
		int ntmp = 0;
		fread(&ntmp, sizeof(int), 1, pFile);
		if (ntmp != tmp.size())
			throw std::exception("size not matched in WarpField::load nodesQuatTransVw");
		fread(tmp.data(), sizeof(float4), tmp.size(), pFile);
		m_nodesQuatTransVw.upload(tmp);		
		unBindNodesDqVwTexture();
		bindNodesDqVwTexture();

		std::vector<KnnIdx> tmpIdx(m_nodesGraph.size(), make_knn(0));
		int ntmpidx = 0;
		fread(&ntmpidx, sizeof(int), 1, pFile);
		if (ntmpidx != tmpIdx.size())
			throw std::exception("size not matched in WarpField::load nodesGraph");
		fread(tmpIdx.data(), sizeof(KnnIdx), tmpIdx.size(), pFile);
		m_nodesGraph.upload(tmpIdx);

		fclose(pFile);

		updateAnnField();

		printf("warp field loaded: %d %d %d %d nodes\n", m_numNodes[0],
			m_numNodes[1], m_numNodes[2], m_numNodes[3]);
	}

}