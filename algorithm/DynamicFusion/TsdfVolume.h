#pragma once

#include "kinect_util.h"
#include <vector>

namespace mpu
{
	class VolumeData;
}
namespace dfusion
{
	/** \brief TsdfVolume class
	* \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
	*/
	class TsdfVolume
	{
	public:
		/** \brief Constructor
		* \param[in] resolution volume resolution
		*/
		TsdfVolume();

		void init(int3 resolution, float voxel_size, float3 origion);
		void initFromHost(const mpu::VolumeData* vhost);
		/**
			set the size of each voxel
			*/
		void setVoxelSize(float voxel_size);
		float getVoxelSize()const{ return voxel_size_; }

		/** \brief Sets Tsdf truncation distance. Must be greater than 2 * volume_voxel_size
		* \param[in] distance TSDF truncation distance
		*/
		void setTsdfTruncDist(float distance);

		/** \brief Returns tsdf volume container that point to data in GPU memroy */
		cudaArray_t data() const;
		cudaArray_t data();

		cudaTextureObject_t getTexture()const;
		cudaSurfaceObject_t getSurface()const;

		/** \brief Returns volume resolution */
		const int3& getResolution() const;

		const float3& getOrigion() const;

		/** \brief Returns tsdf truncation distance in meters */
		float getTsdfTruncDist() const;

		/** \brief Resets tsdf volume data to uninitialized state */
		void reset();

		/** \brief Downloads tsdf volume from GPU memory.
		* \param[out] tsdf Array with tsdf values. if volume resolution is 512x512x512, so for voxel (x,y,z) tsdf value can be retrieved as volume[512*512*z + 512*y + x];
		* \param[in] voxels with weight <= this threshold be be viewed as "null"
		*/
		void downloadRawVolume(std::vector<TsdfData>& tsdf)const;
		void download(mpu::VolumeData* vhost)const;
		void uploadRawVolume(std::vector<TsdfData>& tsdf);

		void save(const char* filename)const;
		void load(const char* filename);
	protected:
		void copyFromHost(const float* data);
		void copyToHost(float* data)const;
		void copyToHostRaw(TsdfData* data)const;
		void allocate(int3 resolution, float voxel_size, float3 origion);

		void bindTexture();
		void unbindTexture();
		void bindSurface();
		void unbindSurface();
	private:
		/** \brief tsdf volume size in meters */
		float voxel_size_;

		float3 origion_;

		/** \brief tsdf volume resolution */
		int3 resolution_;

		/** \brief tsdf volume data container */
		cudaArray_t volume_;

		/** \brief tsdf truncation distance */
		float tranc_dist_;

		cudaTextureObject_t tex_;
		cudaSurfaceObject_t surf_;
	};
}