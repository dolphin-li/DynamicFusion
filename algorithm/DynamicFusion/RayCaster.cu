#include "RayCaster.h"
#include "TsdfVolume.h"
#include "device_utils.h"
namespace dfusion
{
#undef min
#undef max
	const static float RAY_CASTING_TIME_STEP = 0.4f;

	__device__ __forceinline__ float getMinTime(const float3& volume_max, const float3& origin, const float3& dir)
	{
		float txmin = ((dir.x > 0 ? 0.f : volume_max.x) - origin.x) / dir.x;
		float tymin = ((dir.y > 0 ? 0.f : volume_max.y) - origin.y) / dir.y;
		float tzmin = ((dir.z > 0 ? 0.f : volume_max.z) - origin.z) / dir.z;

		return fmax(fmax(txmin, tymin), tzmin);
	}

	__device__ __forceinline__ float getMaxTime(const float3& volume_max, const float3& origin, const float3& dir)
	{
		float txmax = ((dir.x > 0 ? volume_max.x : 0.f) - origin.x) / dir.x;
		float tymax = ((dir.y > 0 ? volume_max.y : 0.f) - origin.y) / dir.y;
		float tzmax = ((dir.z > 0 ? volume_max.z : 0.f) - origin.z) / dir.z;

		return fmin(fmin(txmax, tymax), tzmax);
	}

	struct RayCasterG
	{
		enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };

		cudaTextureObject_t volumeData_tex;

		Mat33	Rc2v;		//	camera to volume
		float3	tc2v;

		Mat33	Rv2c;		//	volume to camera
		float3	tv2c;

		int3	volume_resolution;		//	resolution of the volume
		float	voxel_size;
		float	voxel_size_inv;
		float3	volume_size;

		float	time_step;
		int		cols, rows;

		Intr intr;

		mutable PtrStep<float> nmap;
		mutable PtrStep<float> vmap;

		__device__ __forceinline__ float3 get_point_in_camera_coord(int x, int y, float d) const 
		{
			return intr.uvd2xyz(x, y, d);
		}

		__device__ __forceinline__ int sgn(float val) const 
		{
			return (0.0f < val) - (val < 0.0f);
		}

		__device__ __forceinline__ float3 get_ray_next(int x, int y) const
		{
			return intr.uvd2xyz(x, y, 1);
		}

		__device__ __forceinline__ bool checkInds(const int3& g) const
		{
			return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < volume_resolution.x && 
				g.y < volume_resolution.y && g.z < volume_resolution.z);
		}

		__device__ __forceinline__ float readTsdf(float x, float y, float z) const
		{
			// since empty voxels = 0, adding 1e-5 can prevent nemeric errors bewteen +-0
			return unpack_tsdf(read_tsdf_texture(volumeData_tex,x,y,z)).x + 1e-5f;
		}

		__device__ __forceinline__ int3 getVoxel(float3 point) const
		{
			int vx = __float2int_rd(point.x * voxel_size_inv);        // round to negative infinity
			int vy = __float2int_rd(point.y * voxel_size_inv);
			int vz = __float2int_rd(point.z * voxel_size_inv);
			return make_int3(vx, vy, vz);
		}

		__device__ __forceinline__ float interpolateTrilineary(const float3& origin, const float3& dir, float time) const
		{
			return interpolateTrilineary(origin + dir * time);
		}

		__device__ __forceinline__ float interpolateTrilineary(const float3& point) const
		{
			// since empty voxels = 0, adding 1e-5 can prevent nemeric errors bewteen +-0
			return read_tsdf_texture_value_trilinear(volumeData_tex, 
				point.x  * voxel_size_inv, point.y  * voxel_size_inv,
				point.z  * voxel_size_inv) + 1e-5f;
		}

		__device__ __forceinline__ void operator () () const
		{
			int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

			if (x >= cols || y >= rows)
				return;

			vmap.ptr(y)[x] = numeric_limits<float>::quiet_NaN();
			nmap.ptr(y)[x] = numeric_limits<float>::quiet_NaN();

			float3 screen_point = get_point_in_camera_coord(x, y, 1.0f);
			float3 ray_start = tc2v;
			float3 ray_next = Rc2v * get_ray_next(x, y) + tc2v;

			float3 ray_dir = normalized(ray_next - ray_start);

			//ensure that it isn't a degenerate case
			ray_dir.x = (ray_dir.x == 0.f) ? 1e-15 : ray_dir.x;
			ray_dir.y = (ray_dir.y == 0.f) ? 1e-15 : ray_dir.y;
			ray_dir.z = (ray_dir.z == 0.f) ? 1e-15 : ray_dir.z;

			ray_dir = normalized(ray_dir);

			// computer time when entry and exit volume
			float time_start_volume = getMinTime(volume_size, ray_start, ray_dir);
			float time_exit_volume = getMaxTime(volume_size, ray_start, ray_dir);

			const float min_dist = 0.f;         //in meters
			time_start_volume = fmax(time_start_volume, min_dist);
			if (time_start_volume >= time_exit_volume)
				return;

			float time_curr = time_start_volume;
			int3 g = getVoxel(ray_start + ray_dir * time_curr);
			g.x = max(0, min(g.x, volume_resolution.x - 1));
			g.y = max(0, min(g.y, volume_resolution.y - 1));
			g.z = max(0, min(g.z, volume_resolution.z - 1));

			// ldp fix here: use interpolating instead of a single voxel
			float tsdf = interpolateTrilineary(ray_start, ray_dir, time_curr);
			//float tsdf = readTsdf (g.x, g.y, g.z);

			//infinite loop guard
			const float max_time = 3 * (volume_size.x + volume_size.y + volume_size.z);

			for (; time_curr < max_time; time_curr += time_step)
			{
				float tsdf_prev = tsdf;
				g = getVoxel(ray_start + ray_dir * (time_curr + time_step));
				if (!checkInds(g))
					break;

				// ldp fix here: use interpolating instead of a single voxel
				tsdf = interpolateTrilineary(ray_start, ray_dir, time_curr + time_step);
				//tsdf = readTsdf (g.x, g.y, g.z);

				if (tsdf_prev >= 0.f && tsdf <= 0.f)           //zero crossing
				{
					float Ftdt = interpolateTrilineary(ray_start, ray_dir, time_curr + time_step);
					float Ft = interpolateTrilineary(ray_start, ray_dir, time_curr);

					float coef = (Ftdt == Ft) ? 0.f : Ft / (Ftdt - Ft);
					float Ts = time_curr - time_step * coef;
					float3 vetex_found = ray_start + ray_dir * Ts;			//	volume coordinate
					float3 vertex_found_w = Rv2c * vetex_found + tv2c;		//	world coordinate

					vmap.ptr(y)[x] = vertex_found_w.x;
					vmap.ptr(y + rows)[x] = vertex_found_w.y;
					vmap.ptr(y + 2 * rows)[x] = vertex_found_w.z;

					int3 g = getVoxel(ray_start + ray_dir * time_curr);
					if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < volume_resolution.x - 2 && g.y < volume_resolution.y - 2 && g.z < volume_resolution.z - 2)
					{
						float3 t;
						float3 n;
						float half_voxel_size = voxel_size * 0.5f;

						t = vetex_found;
						t.x += half_voxel_size;
						float Fx1 = interpolateTrilineary(t);
						t = vetex_found;
						t.x -= half_voxel_size;
						float Fx2 = interpolateTrilineary(t);
						n.x = (Fx1 - Fx2);

						t = vetex_found;
						t.y += half_voxel_size;
						float Fy1 = interpolateTrilineary(t);
						t = vetex_found;
						t.y -= half_voxel_size;
						float Fy2 = interpolateTrilineary(t);
						n.y = (Fy1 - Fy2);

						t = vetex_found;
						t.z += half_voxel_size;
						float Fz1 = interpolateTrilineary(t);
						t = vetex_found;
						t.z -= half_voxel_size;
						float Fz2 = interpolateTrilineary(t);
						n.z = (Fz1 - Fz2);

						n = normalized(Rv2c * n);

						nmap.ptr(y)[x] = n.x;
						nmap.ptr(y + rows)[x] = n.y;
						nmap.ptr(y + 2 * rows)[x] = n.z;
					}
					break;
				}

			}          /* for(;;)  */
		}
	};


	__global__ void rayCastKernel(const RayCasterG rc)
	{
		rc();
	}

	void RayCaster::raycast()
	{
		// init rc structure
		RayCasterG rc;
		rc.Rc2v = m_Rc2v;
		rc.tc2v = m_tc2v;
		rc.Rv2c = m_Rv2c;
		rc.tv2c = m_tv2c;
		rc.volume_resolution.x = m_volume->getResolution().x;
		rc.volume_resolution.y = m_volume->getResolution().y;
		rc.volume_resolution.z = m_volume->getResolution().z;
		rc.voxel_size = m_volume->getVoxelSize();
		rc.voxel_size_inv = 1.f / rc.voxel_size;
		rc.volume_size.x = rc.volume_resolution.x * rc.voxel_size;
		rc.volume_size.y = rc.volume_resolution.y * rc.voxel_size;
		rc.volume_size.z = rc.volume_resolution.z * rc.voxel_size;
		rc.time_step = m_volume->getTsdfTruncDist() * RAY_CASTING_TIME_STEP;
		rc.cols = m_vmap.cols();
		rc.rows = m_vmap.rows();
		rc.intr = m_intr;
		rc.vmap = m_vmap;
		rc.nmap = m_nmap;
		rc.volumeData_tex = m_volume->getTexture();

		// lunch the kernel
		dim3 block(RayCasterG::CTA_SIZE_X, RayCasterG::CTA_SIZE_Y);
		dim3 grid(divUp(rc.cols, block.x), divUp(rc.rows, block.y));

		rayCastKernel << <grid, block >> >(rc);
		cudaSafeCall(cudaGetLastError(), "raycast");
	}
}



