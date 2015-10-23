#include "TsdfVolume.h"
#include "VolumeData.h"

namespace dfusion
{
	inline int3 get_int3(ldp::Int3 r)
	{
		return make_int3(r[0], r[1], r[2]);
	}

	inline float3 get_float3(ldp::Float3 r)
	{
		return make_float3(r[0], r[1], r[2]);
	}

	TsdfVolume::TsdfVolume()
	{
		tranc_dist_ = 0.f;
		setVoxelSize(0);
		tranc_dist_ = 0.f;
		resolution_ = make_int3(0, 0, 0);
		setVoxelSize(0);
		setTsdfTruncDist(voxel_size_ * 5);
		volume_ = nullptr;
	}

	void TsdfVolume::init(int3 resolution, float voxel_size, float3 origion)
	{
		allocate(resolution, voxel_size, origion);

		reset();
	}

	void TsdfVolume::initFromHost(const mpu::VolumeData* vhost)
	{
		release_assert(vhost);

		allocate(get_int3(vhost->getResolution()), vhost->getVoxelSize(), get_float3(vhost->getBound().min));

		copyFromHost(vhost->data());
	}

	void TsdfVolume::setVoxelSize(float voxel_size)
	{
		voxel_size_ = voxel_size;
	}

	void TsdfVolume::setTsdfTruncDist(float distance)
	{
		tranc_dist_ = distance;
	}
	
	cudaArray_t TsdfVolume::data() const
	{
		return volume_;
	}

	cudaArray_t TsdfVolume::data()
	{
		return volume_;
	}

	cudaTextureObject_t TsdfVolume::getTexture()const
	{
		return tex_;
	}

	cudaSurfaceObject_t TsdfVolume::getSurface()const
	{
		return surf_;
	}

	void TsdfVolume::bindTexture()
	{
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = data();
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = 0;
#ifdef USE_FLOAT_TSDF_VOLUME
		texDescr.filterMode = cudaFilterModeLinear;
#else
		texDescr.filterMode = cudaFilterModePoint;
#endif
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		cudaSafeCall(cudaCreateTextureObject(&tex_, &texRes, &texDescr, NULL),
			"TsdfVolume::bindTexture 1");
	}

	void TsdfVolume::unbindTexture()
	{
		cudaSafeCall(cudaDestroyTextureObject(tex_),
			"TsdfVolume::unbindTexture");
	}

	void TsdfVolume::bindSurface()
	{
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = data();
		cudaSafeCall(cudaCreateSurfaceObject(&surf_, &surfRes),
			"TsdfVolume::bindSurface");
	}

	void TsdfVolume::unbindSurface()
	{
		cudaSafeCall(cudaDestroySurfaceObject(surf_),
			"TsdfVolume::unBindSurface");
	}

	const int3& TsdfVolume::getResolution() const
	{
		return resolution_;
	}

	const float3& TsdfVolume::getOrigion() const
	{
		return origion_;
	}

	float TsdfVolume::getTsdfTruncDist() const
	{
		return tranc_dist_;
	}

	void TsdfVolume::downloadRawVolume(std::vector<TsdfData>& tsdf) const
	{
		tsdf.resize(resolution_.x*resolution_.y*resolution_.z);

		copyToHostRaw(tsdf.data());
	}

	void TsdfVolume::download(mpu::VolumeData* vhost)const
	{
		release_assert(vhost);

		vhost->resize(ldp::UShort3(resolution_.x, resolution_.y, resolution_.z), voxel_size_);

		kdtree::AABB box;
		box.min = ldp::Float3(origion_.x, origion_.y, origion_.z);
		box.max = box.min + voxel_size_ * ldp::Float3(vhost->getResolution());

		vhost->setBound(box);

		copyToHost(vhost->data());
	}

	void TsdfVolume::allocate(int3 resolution, float voxel_size, float3 origion)
	{
		tranc_dist_ = 0.f;
		resolution_ = resolution;
		origion_ = origion;
		setVoxelSize(voxel_size);
		setTsdfTruncDist(voxel_size_ * 5);

		// malloc 3D texture
		if (volume_)
			cudaSafeCall(cudaFreeArray(volume_), "cudaFreeArray");
		cudaExtent ext;
		ext.width = resolution.x;
		ext.height = resolution.y;
		ext.depth = resolution.z;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<TsdfData>();
		cudaSafeCall(cudaMalloc3DArray(&volume_, &desc, ext), "TsdfVolume::allocate, cudaMalloc3D");
		unbindTexture();
		unbindSurface();
		bindTexture();
		bindSurface();
	}

	void TsdfVolume::save(const char* filename)const
	{
		std::vector<TsdfData> tsdf;
		downloadRawVolume(tsdf);

		FILE* pFile = fopen(filename, "wb");
		if (!pFile)
			throw std::exception(("save failed: " + std::string(filename)).c_str());

		fwrite(&voxel_size_, sizeof(voxel_size_), 1, pFile);
		fwrite(&origion_, sizeof(origion_), 1, pFile);
		fwrite(&resolution_, sizeof(resolution_), 1, pFile);
		fwrite(&tranc_dist_, sizeof(tranc_dist_), 1, pFile);
		fwrite(tsdf.data(), sizeof(TsdfData), tsdf.size(), pFile);

		fclose(pFile);
	}

	void TsdfVolume::load(const char* filename)
	{

		FILE* pFile = fopen(filename, "rb");
		if (!pFile)
			throw std::exception(("load failed: " + std::string(filename)).c_str());

		fread(&voxel_size_, sizeof(voxel_size_), 1, pFile);
		fread(&origion_, sizeof(origion_), 1, pFile);
		fread(&resolution_, sizeof(resolution_), 1, pFile);
		fread(&tranc_dist_, sizeof(tranc_dist_), 1, pFile);

		std::vector<TsdfData> tsdf(resolution_.x * resolution_.y * resolution_.z);
		fread(tsdf.data(), sizeof(TsdfData), tsdf.size(), pFile);

		fclose(pFile);

		allocate(resolution_, voxel_size_, origion_);

		uploadRawVolume(tsdf);
	}
}