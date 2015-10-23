#include "TsdfVolume.h"
#include "device_utils.h"
namespace dfusion
{
	__global__ void initializeVolume(int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
				write_tsdf_surface(surf, pack_tsdf(0, 0), x, y, z);
		}
	}

	void TsdfVolume::reset()
	{
		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(resolution_.x, block.x);
		grid.y = divUp(resolution_.y, block.y);

		// surface object
		cudaSurfaceObject_t surf = bindSurface();

		initializeVolume << <grid, block >> >(resolution_, surf);
		cudaSafeCall(cudaGetLastError());

		unbindSurface(surf);
	}

	__global__ void copyFromHostKernel(const float* data, int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
			{
				int pos = (z*res.y + y)*res.x + x;
				write_tsdf_surface(surf, pack_tsdf(data[pos], 1.f), x, y, z);
			}
		}
	}

	void TsdfVolume::copyFromHost(const float* data)
	{
		DeviceArray<float> tmp;
		tmp.upload(data, resolution_.x*resolution_.y*resolution_.z);

		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(resolution_.x, block.x);
		grid.y = divUp(resolution_.y, block.y);

		// surface object
		cudaSurfaceObject_t surf = bindSurface();

		copyFromHostKernel << <grid, block >> >(tmp.ptr(), resolution_, surf);
		cudaSafeCall(cudaGetLastError(), "TsdfVolume::copyFromHost");
		cudaSafeCall(cudaThreadSynchronize(), "TsdfVolume::copyFromHost");
		unbindSurface(surf);
	}

	__global__ void copyToHostKernel(float* data, int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
			{
				int pos = (z*res.y + y)*res.x + x;
				TsdfData val = read_tsdf_surface(surf, x, y, z);
				if (unpack_tsdf(val).y == 0.f)
					data[pos] = numeric_limits<float>::quiet_NaN();
				else
					data[pos] = unpack_tsdf(val).x;
			}
		}
	}

	void TsdfVolume::copyToHost(float* data)const
	{
		DeviceArray<float> tmp;
		tmp.create(resolution_.x*resolution_.y*resolution_.z);

		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(resolution_.x, block.x);
		grid.y = divUp(resolution_.y, block.y);

		// surface object
		cudaSurfaceObject_t surf = bindSurface();

		copyToHostKernel << <grid, block >> >(tmp.ptr(), resolution_, surf);
		cudaSafeCall(cudaGetLastError(), "TsdfVolume::copyToHost");
		cudaSafeCall(cudaThreadSynchronize(), "TsdfVolume::copyToHost");

		unbindSurface(surf);

		tmp.download(data);
	}

	__global__ void copyToHostRawKernel(TsdfData* data, int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
			{
				int pos = (z*res.y + y)*res.x + x;
				data[pos] = read_tsdf_surface(surf, x, y, z);
			}
		}
	}

	void TsdfVolume::copyToHostRaw(TsdfData* data)const
	{
		DeviceArray<TsdfData> tmp;
		tmp.create(resolution_.x*resolution_.y*resolution_.z);

		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(resolution_.x, block.x);
		grid.y = divUp(resolution_.y, block.y);

		// surface object
		cudaSurfaceObject_t surf = bindSurface();

		copyToHostRawKernel << <grid, block >> >(tmp.ptr(), resolution_, surf);
		cudaSafeCall(cudaGetLastError(), "TsdfVolume::copyToHostRaw");
		cudaSafeCall(cudaThreadSynchronize(), "TsdfVolume::copyToHostRaw");

		unbindSurface(surf);

		tmp.download(data);
	}

	__global__ void uploadRawVolumeKernel(const TsdfData* data, int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
			{
				int pos = (z*res.y + y)*res.x + x;
				write_tsdf_surface(surf, data[pos], x, y, z);
			}
		}
	}

	void TsdfVolume::uploadRawVolume(std::vector<TsdfData>& tsdf)
	{
		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(resolution_.x, block.x);
		grid.y = divUp(resolution_.y, block.y);

		cudaSurfaceObject_t surf = bindSurface();

		DeviceArray<TsdfData> tmp;
		tmp.upload(tsdf);

		uploadRawVolumeKernel << <grid, block >> >(tmp.ptr(), resolution_, surf);
		cudaSafeCall(cudaGetLastError(), "TsdfVolume::uploadRawVolume");
		cudaSafeCall(cudaThreadSynchronize(), "TsdfVolume::uploadRawVolume");

		unbindSurface(surf);
	}
}