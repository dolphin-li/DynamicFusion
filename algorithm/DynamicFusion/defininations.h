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
namespace dfusion
{
	//// use float2 or short2 or half2 as TsdfData
	//// NOTE:
	////	short2 will be converted to float[-1,1] and then calculate, 
	////		thus any weightings/values larger than 1 is not accepted
	////	half2 and float2 are not limited to this.
#define USE_FLOAT_TSDF_VOLUME 
//#define USE_SHORT_TSDF_VOLUME
//#define USE_HALF_TSDF_VOLUME
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
	struct Intr
	{
		float fx, fy, cx, cy;
		Intr() {}
		Intr(float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

		Intr operator()(int level_index) const
		{
			int div = 1 << level_index;
			return (Intr(fx / div, fy / div, cx / div, cy / div));
		}
	};

	typedef unsigned short ushort;
	typedef ushort depthtype;
	typedef DeviceArray2D<float> MapArr;
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
	// NOTE: v, w must in [-1,1]
	__device__ __host__ __forceinline__ TsdfData pack_tsdf(float v, float w)
	{
		return make_short2(v*TSDF_DIVISOR, w*TSDF_DIVISOR);
	}
	__device__ __host__ __forceinline__ float2 unpack_tsdf(TsdfData td)
	{
		return make_float2(float(td.x) * TSDF_INV_DIVISOR, float(td.y) * TSDF_INV_DIVISOR);
	}
#endif
#ifdef USE_HALF_TSDF_VOLUME
	typedef int TsdfData; // value(low)-weight(high) stored in a voxel
#define TSDF_DIVISOR 1.f
#if defined(__CUDACC__)
	// NOTE: v, w must in [-1,1]
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

	enum{
		KINECT_WIDTH = 640,
		KINECT_HEIGHT = 480
	};

	// each pbo id seems only be used once when combined with cuda, 
	// even if they are in different context.
	// if not used, then the id is marked as used.
	bool is_pbo_id_used_push_new(unsigned int id);
}