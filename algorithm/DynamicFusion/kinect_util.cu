#include "kinect_util.h"
#include "device_utils.h"
#include <set>
namespace dfusion
{
#pragma region --copy_kernel

	__global__ void copy_colormap_kernel(PtrStepSz<PixelRGBA> src,
		PtrStepSz<uchar4> dst)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= dst.cols || v >= dst.rows)
			return;

		PixelRGBA p = src(v, u);
		dst(v, u) = make_uchar4(p.r, p.g, p.b, p.a);
	}

	void copyColorMapToPbo(PtrStepSz<PixelRGBA> src, PtrStepSz<uchar4> dst)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(dst.cols, block.x);
		grid.y = divUp(dst.rows, block.y);

		copy_colormap_kernel << <grid, block >> >(src, dst);
		cudaSafeCall(cudaGetLastError(), "copyColorMapToPbo");
	}
#pragma endregion

#pragma region --calculate jet
	/////////////////////////////////////////////////////////////////////////////////
	// jet calculation, for visualization
	__constant__ float g_jet_w[9][6] =
	{
		{ 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.5f },
		{ 0.0f, 0.0f, 0.5f, -0.5f, 0.0f, 1.0f },
		{ 0.0f, 0.0f, 0.5f, -0.5f, 0.0f, 1.0f },
		{ 0.5f, -1.5f, 0.0f, 1.0f, -0.5f, 2.5f },
		{ 0.5f, -1.5f, 0.0f, 1.0f, -0.5f, 2.5f },
		{ 0.0f, 1.0f, -0.5f, 3.5f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, -0.5f, 3.5f, 0.0f, 0.0f },
		{ -0.5f, 4.5f, 0.0f, 0.0f, 0.0f, 0.0f },
		{ -0.5f, 4.5f, 0.0f, 0.0f, 0.0f, 0.0f }
	};

	__global__ void calcTemperatureJetKernel(PtrStepSz<depthtype> depth_d, PtrStepSz<uchar4> jetRgb_d, float shift, float div)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= depth_d.cols || v >= depth_d.rows)
			return;
		float val = float(depth_d(v, u)-shift)/div;
		val = 8.f * min(1.f, max(0.f, val));
		const float* const c = g_jet_w[int(val)];
		jetRgb_d(v, u) = make_uchar4(255 * (c[0] * val + c[1]), 255 * (c[2] * val + c[3]), 255 * (c[4] * val + c[5]), 255);
	}

	void calc_temperature_jet(PtrStepSz<depthtype> depth_d, PtrStepSz<uchar4> jetRgb_d, float shift, float div)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(jetRgb_d.cols, block.x);
		grid.y = divUp(jetRgb_d.rows, block.y);

		calcTemperatureJetKernel << <grid, block >> >(depth_d, jetRgb_d, shift, div);
		cudaSafeCall(cudaGetLastError());
	}

#pragma endregion

#pragma region --generate image
	struct ImageGenerator
	{
		enum
		{
			CTA_SIZE_X = 32, CTA_SIZE_Y = 8
		};

		PtrStep<float> vmap;
		PtrStep<float> nmap;

		LightSource light;

		mutable PtrStepSz<PixelRGBA> dst;

		__device__ __forceinline__ void convertToColorHSV(float rgb[3], float value, float min_value, float max_value) const{
			if (value < min_value)
				value = min_value;
			else if (value > max_value)
				value = max_value;

			float lamda = (value - min_value) / (max_value - min_value);

			float coef = lamda * 6.0f;

			if (coef <= 1.0f)
				rgb[0] = 1.0f, rgb[1] = coef, rgb[2] = 0.0f;
			else if (coef <= 2.0f)
				rgb[0] = 1.0f - (coef - 1.0f), rgb[1] = 1.0f, rgb[2] = 0.0f;
			else if (coef <= 3.0f)
				rgb[0] = 0.0f, rgb[1] = 1.0f, rgb[2] = coef - 2.0f;
			else if (coef <= 4.0f)
				rgb[0] = 0.0f, rgb[1] = 1.0f - (coef - 3.0f), rgb[2] = 1.0f;
			else if (coef <= 5.0f)
				rgb[0] = coef - 4.0f, rgb[1] = 0.0f, rgb[2] = 1.0f;
			else
				rgb[0] = 1.0f, rgb[2] = 0.0f, rgb[2] = 1.0f - (coef - 5.0f);
		}

		__device__ __forceinline__ float int2float(int value) const
		{
			value = (value * 179426549 + 1300997) & 15487469;
			return float(value) / 15487469;
		}

		__device__ __forceinline__ void operator () () const
		{
				int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
				int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

				if (x >= dst.cols || y >= dst.rows)
					return;

				float3 v, n;
				v.x = vmap.ptr(y)[x];
				n.x = nmap.ptr(y)[x];

				PixelRGBA color;
				color.a = color.r = color.g = color.b = 0;

				if (!isnan(v.x) && !isnan(n.x))
				{
					v.y = vmap.ptr(y + dst.rows)[x];
					v.z = vmap.ptr(y + 2 * dst.rows)[x];

					n.y = nmap.ptr(y + dst.rows)[x];
					n.z = nmap.ptr(y + 2 * dst.rows)[x];

					float3 acc_vec = make_float3(0.f, 0.f, 0.f);
					{
						float3 vec = normalized(light.pos - v);
						float w = max(0.f, dot(vec, n));
						acc_vec.x += w * light.diffuse.x;
						acc_vec.y += w * light.diffuse.y;
						acc_vec.z += w * light.diffuse.z;
					}
					color.r = max(0, min(255, int(acc_vec.x*255.f)));
					color.g = max(0, min(255, int(acc_vec.y*255.f)));
					color.b = max(0, min(255, int(acc_vec.z*255.f)));
					color.a = 255;
				}

				dst.ptr(y)[x] = color;
			}
	};

	__global__ void generateImageKernel(const ImageGenerator ig)
	{
		ig();
	}

	void generateImage(const MapArr& vmap, const MapArr& nmap, ColorMap& dst, const LightSource& light)
	{
		ImageGenerator ig;
		ig.vmap = vmap;
		ig.nmap = nmap;
		ig.light = light;
		ig.dst = dst;

		dim3 block(ImageGenerator::CTA_SIZE_X, ImageGenerator::CTA_SIZE_Y);
		dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));

		generateImageKernel << <grid, block >> >(ig);
		cudaSafeCall(cudaGetLastError(), "generateImage");
	}

#pragma endregion

#pragma region --generate normal
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	struct NormalGenerator
	{
		enum
		{
			CTA_SIZE_X = 32, CTA_SIZE_Y = 8
		};

		PtrStep<float> nmap;
		Mat33 R;
		mutable PtrStepSz<PixelRGBA> dst;

		__device__ __forceinline__ void operator () () const
		{
			int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

			if (x >= dst.cols || y >= dst.rows)
				return;

			float3 n;
			n.x = nmap.ptr(y)[x];

			PixelRGBA color;
			color.a = color.r = color.g = color.b = 0;

			if (!isnan(n.x))
			{
				n.y = nmap.ptr(y + dst.rows)[x];
				n.z = nmap.ptr(y + 2 * dst.rows)[x];
				n = normalized(R*n);
				color.r = max(0, min(255, int(255 * (n.x * 0.5f + 0.5f))));
				color.g = max(0, min(255, int(255 * (n.y * 0.5f + 0.5f))));
				color.b = max(0, min(255, int(255 * (n.z * 0.5f + 0.5f))));
				color.a = 255;
			}

			dst.ptr(y)[x] = color;
		}
	};

	__global__ void generateNormalKernel(const NormalGenerator ig)
	{
		ig();
	}

	void generateNormalMap(const MapArr& nmap, ColorMap& dst, Mat33 R)
	{
		NormalGenerator ig;
		ig.nmap = nmap;
		ig.dst = dst;
		ig.R = R;


		dim3 block(NormalGenerator::CTA_SIZE_X, NormalGenerator::CTA_SIZE_Y);
		dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));

		generateNormalKernel << <grid, block >> >(ig);
		cudaSafeCall(cudaGetLastError(), "generateNormal");
	}

#pragma endregion


}