#include "TsdfVolume.h"
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

#if 0
		// bind the surface
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<TsdfData>();
		cudaSafeCall(cudaBindSurfaceToArray(&g_tsdf_volume_surf, volume_, &desc));
		initializeVolume << <grid, block >> >(resolution_);
		cudaSafeCall(cudaGetLastError());
#else
		// surface object
		cudaSurfaceObject_t surf;
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = volume_;
		cudaSafeCall(cudaCreateSurfaceObject(&surf, &surfRes));

		initializeVolume << <grid, block >> >(resolution_, surf);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDestroySurfaceObject(surf));
#endif

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

#if 0
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<TsdfData>();
		cudaSafeCall(cudaBindSurfaceToArray(&g_tsdf_volume_surf, volume_));
		copyFromHostKernel << <grid, block >> >(tmp.ptr(), resolution_);
		cudaSafeCall(cudaGetLastError());
#else
		// surface object
		cudaSurfaceObject_t surf;
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = volume_;
		cudaSafeCall(cudaCreateSurfaceObject(&surf, &surfRes));

		copyFromHostKernel << <grid, block >> >(tmp.ptr(), resolution_, surf);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDestroySurfaceObject(surf));
#endif
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

#if 0
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<TsdfData>();
		cudaSafeCall(cudaBindSurfaceToArray(&g_tsdf_volume_surf, volume_, &desc));
		copyToHostKernel << <grid, block >> >(tmp.ptr(), resolution_);
		cudaSafeCall(cudaGetLastError());
#else
		// surface object
		cudaSurfaceObject_t surf;
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = volume_;
		cudaSafeCall(cudaCreateSurfaceObject(&surf, &surfRes));

		copyToHostKernel << <grid, block >> >(tmp.ptr(), resolution_, surf);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDestroySurfaceObject(surf));
#endif

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

#if 0
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<TsdfData>();
		cudaSafeCall(cudaBindSurfaceToArray(&g_tsdf_volume_surf, volume_, &desc));
		copyToHostRawKernel << <grid, block >> >(tmp.ptr(), resolution_);
		cudaSafeCall(cudaGetLastError());
#else
		// surface object
		cudaSurfaceObject_t surf;
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = volume_;
		cudaSafeCall(cudaCreateSurfaceObject(&surf, &surfRes));

		copyToHostRawKernel << <grid, block >> >(tmp.ptr(), resolution_, surf);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDestroySurfaceObject(surf));
#endif

		tmp.download(data);
	}
}