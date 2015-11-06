#pragma once
#include <device_functions.h>
#include <driver_functions.h>
#include <vector_functions.h>
#include <cuda_fp16.h>
#include <channel_descriptor.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include <surface_types.h>
#include <surface_functions.h>
#include "device_array.h"
#include "cuda_utils.h"
#include "dualquaternion\dual_quat_cu.hpp"
namespace dfusion
{
	// kinect 2 is not suitable for fusion, give up it.
//#define ENABLE_KINECT_20
	//// use float2 or short2 or half2 as TsdfData
	//// NOTE:
	////	short2.x will be converted to float[-1,1] and then calculate, 
	////		thus any values larger than 1 is not accepted
	////	half2 and float2 are not limited to this.
//#define USE_FLOAT_TSDF_VOLUME 
//#define USE_SHORT_TSDF_VOLUME
#define USE_HALF_TSDF_VOLUME

	/** **********************************************************************
	* types
	* ***********************************************************************/
	struct __align__(4) PixelRGBA{
		unsigned char r, g, b, a;
	};

	struct LightSource
	{
		float3 pos;
		float3 diffuse;
		float3 amb;
		float3 spec;
	};

	/** \brief 3x3 Matrix for device code
	* it is row majored: each data[i] is a row
	*/
	struct Mat33
	{
		float3 data[3];
	};

	/** \brief Camera intrinsics structure
	*/
	class Intr
	{
	public:
		float fx, fy, cx, cy, fx_inv, fy_inv;
	public:
		Intr() {}
		Intr(float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) 
		{
			fx_inv = 1.f / fx;
			fy_inv = 1.f / fy;
		}

		Intr operator()(int level_index) const
		{
			int div = 1 << level_index;
			return (Intr(fx / div, fy / div, cx / div, cy / div));
		}

		__device__ __host__ __forceinline__ float3 uvd2xyz(float u, float v, float d)const
		{
			float x = d * (u - cx) * fx_inv;
			float y = -d * (v - cy) * fy_inv;
			float z = -d;
			return make_float3(x, y, z);
		}

		__device__ __host__ __forceinline__ float3 uvd2xyz(float3 uvd)const
		{
			return uvd2xyz(uvd.x, uvd.y, uvd.z);
		}

		__device__ __host__ __forceinline__ float3 xyz2uvd(float x, float y, float z)const
		{
			float d = -z;
			float u = x * fx / d + cx;
			float v = -y * fy / d + cy;
			return make_float3(u, v, d);
		}

		__device__ __host__ __forceinline__ float3 xyz2uvd(float3 xyz)const
		{
			return xyz2uvd(xyz.x, xyz.y, xyz.z);
		}
	};

	typedef unsigned short ushort;
	typedef float depthtype;
	typedef DeviceArray2D<float4> MapArr;
	typedef DeviceArray2D<depthtype> DepthMap;
	typedef DeviceArray2D<PixelRGBA> ColorMap;
#ifdef USE_FLOAT_TSDF_VOLUME
	typedef float2 TsdfData; // value(low)-weight(high) stored in a voxel
	__device__ __host__ __forceinline__ TsdfData pack_tsdf(float v, float w)
	{
		return make_float2(v, w);
	}
	__device__ __host__ __forceinline__ float2 unpack_tsdf(TsdfData td)
	{
		return make_float2(float(td.x), float(td.y));
	}
#endif
#ifdef USE_SHORT_TSDF_VOLUME
	typedef short2 TsdfData; // value(low)-weight(high) stored in a voxel
#define TSDF_DIVISOR 0x7fff
#define TSDF_INV_DIVISOR 3.051850947599719e-05f
	// NOTE: vmust in [-1,1]
	__device__ __host__ __forceinline__ TsdfData pack_tsdf(float v, float w)
	{
		return make_short2(v*TSDF_DIVISOR, w);
	}
	__device__ __host__ __forceinline__ float2 unpack_tsdf(TsdfData td)
	{
		return make_float2(float(td.x) * TSDF_INV_DIVISOR, float(td.y));
	}
#endif
#ifdef USE_HALF_TSDF_VOLUME
	typedef int TsdfData; // value(low)-weight(high) stored in a voxel
#define TSDF_DIVISOR 1.f
#if defined(__CUDACC__)
	__device__ __forceinline__ TsdfData pack_tsdf(float v, float w)
	{
		half2 val = __floats2half2_rn(v, w);
		return *((int*)&val);
	}
	__device__ __forceinline__ float2 unpack_tsdf(TsdfData td)
	{
		return __half22float2(*((half2*)&td));
	}
#endif
#endif


#if defined(__CUDACC__)
	__device__ __forceinline__ int sgn(float val) 
	{
		return (0.0f < val) - (val < 0.0f);
	}
	__device__ __forceinline__ TsdfData read_tsdf_texture(cudaTextureObject_t t, float x, float y, float z)
	{
#ifdef USE_FLOAT_TSDF_VOLUME
		TsdfData val = tex3D<TsdfData>(t, x, y, z);
#endif
#ifdef USE_SHORT_TSDF_VOLUME
		TsdfData val = tex3D<TsdfData>(t, x, y, z);
#endif
#ifdef USE_HALF_TSDF_VOLUME
		TsdfData val = tex3D<TsdfData>(t, x, y, z);
#endif
		return val;
	}

	__device__ __forceinline__ float read_tsdf_texture_value_trilinear(cudaTextureObject_t t, float x, float y, float z)
	{
#ifdef USE_FLOAT_TSDF_VOLUME
		return unpack_tsdf(read_tsdf_texture(t,x,y,z)).x;
#else
		int x0 = __float2int_rd(x);
		int y0 = __float2int_rd(y);
		int z0 = __float2int_rd(z);
		x0 += -(sgn(x0 - x) + 1) >> 1;		//x0 = (x < x0) ? (x - 1) : x;
		y0 += -(sgn(y0 - y) + 1) >> 1;		//y0 = (y < y0) ? (y - 1) : y;
		z0 += -(sgn(z0 - z) + 1) >> 1;		//z0 = (z < z0) ? (z - 1) : z;
		float a0 = x - x0;
		float b0 = y - y0;
		float c0 = z - z0;
		float a1 = 1.0f - a0;
		float b1 = 1.0f - b0;
		float c1 = 1.0f - c0;

		return(
				(
				unpack_tsdf(read_tsdf_texture(t, x0 + 0, y0 + 0, z0 + 0)).x * c1 +
				unpack_tsdf(read_tsdf_texture(t, x0 + 0, y0 + 0, z0 + 1)).x * c0
				) * b1 + (
				unpack_tsdf(read_tsdf_texture(t, x0 + 0, y0 + 1, z0 + 0)).x * c1 +
				unpack_tsdf(read_tsdf_texture(t, x0 + 0, y0 + 1, z0 + 1)).x * c0
				) * b0
			) * a1
			+ (
				(
				unpack_tsdf(read_tsdf_texture(t, x0 + 1, y0 + 0, z0 + 0)).x * c1 +
				unpack_tsdf(read_tsdf_texture(t, x0 + 1, y0 + 0, z0 + 1)).x * c0
				) * b1 + (
				unpack_tsdf(read_tsdf_texture(t, x0 + 1, y0 + 1, z0 + 0)).x * c1 +
				unpack_tsdf(read_tsdf_texture(t, x0 + 1, y0 + 1, z0 + 1)).x * c0
				) * b0
			) * a0;
#endif
	}

	__device__ __forceinline__ void write_tsdf_surface(cudaSurfaceObject_t t, TsdfData val, int x, int y, int z)
	{
#ifdef USE_HALF_TSDF_VOLUME
		surf3Dwrite(val, t, x*sizeof(TsdfData), y, z);
#else
		surf3Dwrite(val, t, x*sizeof(TsdfData), y, z);
#endif
	}

	__device__ __forceinline__ TsdfData read_tsdf_surface(cudaSurfaceObject_t t, int x, int y, int z)
	{
		TsdfData val;
#ifdef USE_HALF_TSDF_VOLUME
		surf3Dread(&val, t, x*sizeof(TsdfData), y, z);
#else
		surf3Dread(&val, t, x*sizeof(TsdfData), y, z);
#endif
		return val;
	}
#endif

#ifdef ENABLE_KINECT_20
	enum{
		KINECT_WIDTH = 512,
		KINECT_HEIGHT = 424
	};

#define KINECT_DEPTH_H_FOV 70.6
#define KINECT_DEPTH_V_FOV 60.0
#define KINECT_IMAGE_H_FOV 70.6
#define KINECT_IMAGE_V_FOV 60.0
#define KINECT_NEAREST_METER 0.5
#else
	enum{
		KINECT_WIDTH = 640,
		KINECT_HEIGHT = 480
	};
#define KINECT_DEPTH_H_FOV 58.5
#define KINECT_DEPTH_V_FOV 45.6
#define KINECT_IMAGE_H_FOV 62.0
#define KINECT_IMAGE_V_FOV 48.6
#define KINECT_NEAREST_METER 0.3
#endif

	__device__ __host__ __forceinline__  Tbx::Vec3 convert(float3 a)
	{
		return Tbx::Vec3(a.x, a.y, a.z);
	}

	__device__ __host__ __forceinline__  float3 convert(Tbx::Vec3 a)
	{
		return make_float3(a.x, a.y, a.z);
	}

	__device__ __host__ __forceinline__  Tbx::Mat3 convert(Mat33 a)
	{
		return Tbx::Mat3(a.data[0].x, a.data[0].y, a.data[0].z,
						a.data[1].x, a.data[1].y, a.data[1].z,
						a.data[2].x, a.data[2].y, a.data[2].z);
	}

	__device__ __host__ __forceinline__  Mat33 convert(Tbx::Mat3 a)
	{
		Mat33 b;
		b.data[0] = make_float3(a.a, a.b, a.c);
		b.data[1] = make_float3(a.d, a.e, a.f);
		b.data[2] = make_float3(a.g, a.h, a.i);
		return b;
	}

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
		return v.x*v.x + v.y*v.y + v.z*v.z;
	}

	// it seems cudaResGLRegister may conflict for 
	// the same id in different context (especially QT&wGL context)
	// thus we use this function to explicitly make each id different 
	// once called, the id will be marked as used
	bool is_cuda_pbo_vbo_id_used_push_new(unsigned int id);

	/** ***************************************************************
	* Knn related
	* ****************************************************************/
	typedef ushort KnnIdxType;
	typedef uint4 KnnIdx;
	enum{
		KnnK = sizeof(KnnIdx) / sizeof(KnnIdxType)
	};

	__device__ __host__ __forceinline__ KnnIdxType& knn_k(KnnIdx& knn, int i)
	{
		return ((KnnIdxType*)(&knn))[i];
	}
	__device__ __host__ __forceinline__ const KnnIdxType& knn_k(const KnnIdx& knn, int i)
	{
		return ((KnnIdxType*)(&knn))[i];
	}
	__device__ __host__ __forceinline__ KnnIdx make_knn(const ushort* data)
	{
		KnnIdx knn;
		for (int k = 0; k < KnnK; k++)
			knn_k(knn, k) = data[k];
		return knn;
	}
	__device__ __host__ __forceinline__ KnnIdx make_knn(int c)
	{
		KnnIdx knn;
		for (int k = 0; k < KnnK; k++)
			knn_k(knn, k) = c;
		return knn;
	}
	__device__ __host__ __forceinline__ KnnIdx make_knn(const int* data)
	{
		KnnIdx knn;
		for (int k = 0; k < KnnK; k++)
			knn_k(knn, k) = data[k];
		return knn;
	}
#if defined(__CUDACC__)
	__device__ __forceinline__ KnnIdx read_knn_tex(cudaTextureObject_t knnTex, int x, int y, int z)
	{
		KnnIdx knn;
		tex3D(&knn, knnTex, x, y, z);
		return knn;
	}
	__device__ __forceinline__ KnnIdx read_knn_surf(cudaSurfaceObject_t knnTex, int x, int y, int z)
	{
		KnnIdx knn;
		surf3Dread(&knn, knnTex, x*sizeof(KnnIdx), y, z);
		return knn;
	}
	__device__ __forceinline__ void write_knn(KnnIdx knn, cudaSurfaceObject_t knnTex, int x, int y, int z)
	{
		surf3Dwrite(knn, knnTex, x*sizeof(KnnIdx), y, z);
	}
#endif
}