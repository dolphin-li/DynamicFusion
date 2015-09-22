#include "kinect_util.h"
#include "device_utils.h"
#include <set>
namespace dfusion
{
#pragma region --uniqe id
	bool is_cuda_pbo_vbo_id_used_push_new(unsigned int id)
	{
		static std::set<unsigned int> idset;
		if (idset.find(id) != idset.end())
			return true;
		idset.insert(id);
		return false;
	}
#pragma endregion

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

		dst.create(vmap.rows() / 3, vmap.cols());
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

#pragma region --bilateral
	const float sigma_color = 30;     //in mm
	const float sigma_space = 4.5;     // in pixels
	texture<depthtype, cudaTextureType2D, cudaReadModeElementType> g_bitex;
	__global__ void bilateralKernel( PtrStepSz<depthtype> dst, 
		float sigma_space2_inv_half, float sigma_color2_inv_half)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= dst.cols || y >= dst.rows)
			return;

		enum{
			R = 6
		};

		depthtype value = tex2D(g_bitex, x, y);

		float sum1 = 0;
		float sum2 = 0;
#pragma unroll
		for (int cy = -R; cy <= R; ++cy)
		{
#pragma unroll
			for (int cx = -R; cx <= R; ++cx)
			{
				depthtype tmp = tex2D(g_bitex, cx + x, cy + y);
				float space2 = cx*cx + cy*cy;
				float color2 = (value - tmp) * (value - tmp);
				float weight = __expf(-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));
				sum1 += tmp * weight;
				sum2 += weight;
			}
		}

		float res = sum1 / sum2;
		dst.ptr(y)[x] = (isnan(res) || isinf(res)) ? 0 : res;
	}
	
	void bilateralFilter(const DepthMap& src, DepthMap& dst)
	{
		dim3 block(32, 8);
		dim3 grid(divUp(src.cols(), block.x), divUp(src.rows(), block.y));

		dst.create(src.rows(), src.cols());

		// bind src to texture
		size_t offset;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<depthtype>();
		cudaBindTexture2D(&offset, &g_bitex, src.ptr(), &desc, src.cols(), src.rows(), src.step());
		assert(offset == 0);

		bilateralKernel << <grid, block >> >(dst, 0.5f / (sigma_space * sigma_space), 
			0.5f / (sigma_color * sigma_color));

		cudaSafeCall(cudaGetLastError());

		cudaUnbindTexture(&g_bitex);
	}
#pragma endregion

#pragma region --compute vmap, nmap
	__global__ void computeVmapKernel(const PtrStepSz<depthtype> depth, PtrStep<float> vmap, Intr intr)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u < depth.cols && v < depth.rows)
		{
			float z = depth.ptr(v)[u] * 0.001f; // load and convert: mm -> meters

			if (z > KINECT_NEAREST_METER)
			{
				float3 xyz = intr.uvd2xyz((float)u, (float)v, z);

				vmap.ptr(v)[u] = xyz.x;
				vmap.ptr(v + depth.rows)[u] = xyz.y;
				vmap.ptr(v + depth.rows * 2)[u] = xyz.z;
			}
			else
				vmap.ptr(v)[u] = numeric_limits<float>::quiet_NaN();

		}
	}

	__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= cols || v >= rows)
			return;

		if (u == cols - 1 || u == 0 || v == rows - 1 || v == 0)
		{
			nmap.ptr(v)[u] = numeric_limits<float>::quiet_NaN();
			return;
		}

		float3 v00, v01, v10;
		v00.x = vmap.ptr(v)[u];
		v01.x = vmap.ptr(v)[u + 1];
		v10.x = vmap.ptr(v - 1)[u];

		if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x))
		{
			v00.y = vmap.ptr(v + rows)[u];
			v01.y = vmap.ptr(v + rows)[u + 1];
			v10.y = vmap.ptr(v - 1 + rows)[u];

			v00.z = vmap.ptr(v + 2 * rows)[u];
			v01.z = vmap.ptr(v + 2 * rows)[u + 1];
			v10.z = vmap.ptr(v - 1 + 2 * rows)[u];

			float3 r = normalized(cross(v01 - v00, v10 - v00));

			nmap.ptr(v)[u] = r.x;
			nmap.ptr(v + rows)[u] = r.y;
			nmap.ptr(v + 2 * rows)[u] = r.z;
		}
		else
			nmap.ptr(v)[u] = numeric_limits<float>::quiet_NaN();
	}

	void createVMap(const Intr& intr, const DepthMap& depth, MapArr& vmap)
	{
		vmap.create(depth.rows() * 3, depth.cols());

		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(depth.cols(), block.x);
		grid.y = divUp(depth.rows(), block.y);

		computeVmapKernel << <grid, block >> >(depth, vmap, intr);
		cudaSafeCall(cudaGetLastError());
	}

	void createNMap(const MapArr& vmap, MapArr& nmap)
	{
		nmap.create(vmap.rows(), vmap.cols());

		int rows = vmap.rows() / 3;
		int cols = vmap.cols();

		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(cols, block.x);
		grid.y = divUp(rows, block.y);

		computeNmapKernel << <grid, block >> >(rows, cols, vmap, nmap);
		cudaSafeCall(cudaGetLastError());
	}
#pragma endregion

#pragma region --pydDown
	texture<depthtype, cudaTextureType2D, cudaReadModeElementType> g_pydtex;
	__global__ void pyrDownKernel(PtrStepSz<depthtype> dst, float sigma_color)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= dst.cols || y >= dst.rows)
			return;

		enum{
			R = 2
		};

		depthtype center = tex2D(g_pydtex, 2*x, 2*y); // src.ptr(2 * y)[2 * x];

		depthtype sum = 0;
		int count = 0;

#pragma unroll
		for (int cy = -R; cy <= R; ++cy)
		{
#pragma unroll
			for (int cx = -R; cx <= R; ++cx)
			{
				depthtype val = tex2D(g_pydtex, cx+2*x, cy+2*y);
				if (abs(val - center) < 3 * sigma_color)
				{
					sum += val;
					++count;
				}
			}
		}
		dst.ptr(y)[x] = sum / count;
	}

	void pyrDown(const DepthMap& src, DepthMap& dst)
	{
		dst.create(src.rows() / 2, src.cols() / 2);

		// bind src to texture
		size_t offset;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<depthtype>();
		cudaBindTexture2D(&offset, &g_pydtex, src.ptr(), &desc, src.cols(), src.rows(), src.step());
		assert(offset == 0);

		dim3 block(32, 8);
		dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));

		pyrDownKernel << <grid, block >> >(dst, sigma_color);
		cudaSafeCall(cudaGetLastError());

		cudaUnbindTexture(&g_pydtex);
	}

#pragma endregion

#pragma region --resize map
	template<bool normalize>
	__global__ void resizeMapKernel(int drows, int dcols, int srows, const PtrStep<float> input, PtrStep<float> output)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= dcols || y >= drows)
			return;

		const float qnan = numeric_limits<float>::quiet_NaN();

		int xs = x * 2;
		int ys = y * 2;

		float x00 = input.ptr(ys + 0)[xs + 0];
		float x01 = input.ptr(ys + 0)[xs + 1];
		float x10 = input.ptr(ys + 1)[xs + 0];
		float x11 = input.ptr(ys + 1)[xs + 1];

		if (isnan(x00) || isnan(x01) || isnan(x10) || isnan(x11))
		{
			output.ptr(y)[x] = qnan;
			return;
		}
		else
		{
			float3 n;

			n.x = (x00 + x01 + x10 + x11) / 4;

			float y00 = input.ptr(ys + srows + 0)[xs + 0];
			float y01 = input.ptr(ys + srows + 0)[xs + 1];
			float y10 = input.ptr(ys + srows + 1)[xs + 0];
			float y11 = input.ptr(ys + srows + 1)[xs + 1];

			n.y = (y00 + y01 + y10 + y11) / 4;

			float z00 = input.ptr(ys + 2 * srows + 0)[xs + 0];
			float z01 = input.ptr(ys + 2 * srows + 0)[xs + 1];
			float z10 = input.ptr(ys + 2 * srows + 1)[xs + 0];
			float z11 = input.ptr(ys + 2 * srows + 1)[xs + 1];

			n.z = (z00 + z01 + z10 + z11) / 4;

			if (normalize)
				n = normalized(n);

			output.ptr(y)[x] = n.x;
			output.ptr(y + drows)[x] = n.y;
			output.ptr(y + 2 * drows)[x] = n.z;
		}
	}

	template<bool normalize>
	void resizeMap(const MapArr& input, MapArr& output)
	{
		int in_cols = input.cols();
		int in_rows = input.rows() / 3;

		int out_cols = in_cols / 2;
		int out_rows = in_rows / 2;

		output.create(out_rows * 3, out_cols);

		dim3 block(32, 8);
		dim3 grid(divUp(out_cols, block.x), divUp(out_rows, block.y));
		resizeMapKernel<normalize> << < grid, block >> >(out_rows, out_cols, in_rows, input, output);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDeviceSynchronize());
	}

	void resizeVMap(const MapArr& input, MapArr& output)
	{
		resizeMap<false>(input, output);
	}

	void resizeNMap(const MapArr& input, MapArr& output)
	{
		resizeMap<true>(input, output);
	}
#pragma endregion

#pragma region --rigid transform
	__global__ void rigidTransformKernel(PtrStep<float> vmap, PtrStep<float> nmap, 
		Mat33 R, float3 t, int rows, int cols)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < cols && y < rows)
		{
			float3 v = make_float3(vmap(y, x), vmap(y + rows, x), vmap(y + rows * 2, x));
			float3 n = make_float3(nmap(y, x), nmap(y + rows, x), nmap(y + rows * 2, x));
			if (isnan(v.x) || isnan(n.x))
				return;

			v = R*v + t;
			n = R*n;
			vmap(y, x) = v.x;
			vmap(y + rows, x) = v.y;
			vmap(y + rows * 2, x) = v.z;
			nmap(y, x) = n.x;
			nmap(y + rows, x) = n.y;
			nmap(y + rows * 2, x) = n.z;
		}
	}

	void rigidTransform(MapArr& vmap, MapArr& nmap, Tbx::Transfo T)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		int cols = vmap.cols();
		int rows = vmap.rows() / 3;
		grid.x = divUp(cols, block.x);
		grid.y = divUp(rows, block.y);
		rigidTransformKernel << <grid, block >> >(vmap, nmap, convert(T.get_mat3()), 
			convert(T.get_translation()), rows, cols);
		cudaSafeCall(cudaGetLastError());
	}
#pragma endregion

#pragma region --rigid estimate
	typedef float float_type;

	template<int CTA_SIZE_, typename T>
	static __device__ __forceinline__ void reduce(volatile T* buffer)
	{
		int tid = Block::flattenedThreadId();
		T val = buffer[tid];

		if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = val + buffer[tid + 512]; __syncthreads(); }
		if (CTA_SIZE_ >= 512) { if (tid < 256) buffer[tid] = val = val + buffer[tid + 256]; __syncthreads(); }
		if (CTA_SIZE_ >= 256) { if (tid < 128) buffer[tid] = val = val + buffer[tid + 128]; __syncthreads(); }
		if (CTA_SIZE_ >= 128) { if (tid <  64) buffer[tid] = val = val + buffer[tid + 64]; __syncthreads(); }

		if (tid < 32){
			if (CTA_SIZE_ >= 64) { buffer[tid] = val = val + buffer[tid + 32]; }
			if (CTA_SIZE_ >= 32) { buffer[tid] = val = val + buffer[tid + 16]; }
			if (CTA_SIZE_ >= 16) { buffer[tid] = val = val + buffer[tid + 8]; }
			if (CTA_SIZE_ >= 8) { buffer[tid] = val = val + buffer[tid + 4]; }
			if (CTA_SIZE_ >= 4) { buffer[tid] = val = val + buffer[tid + 2]; }
			if (CTA_SIZE_ >= 2) { buffer[tid] = val = val + buffer[tid + 1]; }
		}
	}

	struct Combined
	{
		enum
		{
			CTA_SIZE_X = 32,
			CTA_SIZE_Y = 8,
			CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y
		};


		Mat33 Rcurr;
		float3 tcurr;

		PtrStep<float> vmap_curr;
		PtrStep<float> nmap_curr;

		Mat33 Rprev;
		Mat33 Rprev_inv;
		float3 tprev;

		Intr intr;

		PtrStep<float> vmap_prev;
		PtrStep<float> nmap_prev;

		float distThres;
		float angleThres;

		int cols;
		int rows;

		mutable PtrStep<float_type> gbuf;

		__device__ __forceinline__ bool search(int x, int y, float3& n, float3& d, float3& s) const
		{
			float3 ncurr;
			ncurr.x = nmap_curr.ptr(y)[x];
			ncurr.y = nmap_curr.ptr(y + rows)[x];
			ncurr.z = nmap_curr.ptr(y + 2 * rows)[x];

			if (isnan(ncurr.x))
				return (false);

			float3 vcurr;
			vcurr.x = vmap_curr.ptr(y)[x];
			vcurr.y = vmap_curr.ptr(y + rows)[x];
			vcurr.z = vmap_curr.ptr(y + 2 * rows)[x];

			float3 vcurr_g = Rcurr * vcurr + tcurr;
			float3 ncurr_g = Rcurr * ncurr;
			float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);	// prev camera coo space
			float3 ncurr_cp = Rprev_inv * ncurr_g;				// prev camera coo space

			float3 uvd = intr.xyz2uvd(vcurr_cp);
			int2 ukr = make_int2(__float2int_rn(uvd.x), __float2int_rn(uvd.y));

			// we use opengl coordinate, thus world.z should < 0
			if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z >= 0)
				return (false);

			float3 nprev;
			nprev.x = nmap_prev.ptr(ukr.y)[ukr.x];
			nprev.y = nmap_prev.ptr(ukr.y + rows)[ukr.x];
			nprev.z = nmap_prev.ptr(ukr.y + 2 * rows)[ukr.x];

			if (isnan(nprev.x))
				return (false);

			float3 vprev;
			vprev.x = vmap_prev.ptr(ukr.y)[ukr.x];
			vprev.y = vmap_prev.ptr(ukr.y + rows)[ukr.x];
			vprev.z = vmap_prev.ptr(ukr.y + 2 * rows)[ukr.x];

			float dist = norm(vprev - vcurr_cp);
			if (dist > distThres)
				return (false);

			float sine = norm(cross(ncurr_cp, nprev));
			if (sine >= angleThres)
				return (false);

			n = nprev;
			d = vprev;
			s = vcurr_cp;
			return (true);
		}

		__device__ __forceinline__ void operator () () const
		{
			int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

			float3 n, d, s;
			bool found_coresp = false;

			if (x < cols && y < rows)
				found_coresp = search(x, y, n, d, s);

			float row[7];

			if (found_coresp)
			{
				*(float3*)&row[0] = cross(s, n);
				*(float3*)&row[3] = n;
				row[6] = dot(n, d - s);
			}
			else
				row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

			__shared__ float_type smem[CTA_SIZE];
			int tid = Block::flattenedThreadId();

			int shift = 0;
			for (int i = 0; i < 6; ++i)        //rows
			{
#pragma unroll
				for (int j = i; j < 7; ++j)          // cols + b
				{
					__syncthreads();
					smem[tid] = row[i] * row[j];
					__syncthreads();

					reduce<CTA_SIZE>(smem);

					if (tid == 0)
						gbuf.ptr(shift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
				}
			}
		}
	};

	__global__ void combinedKernel(const Combined cs)
	{
		cs();
	}

	struct TranformReduction
	{
		enum
		{
			CTA_SIZE = 512,
			STRIDE = CTA_SIZE,

			B = 6, COLS = 6, ROWS = 6, DIAG = 6,
			UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
			TOTAL = UPPER_DIAG_MAT + B,

			GRID_X = TOTAL
		};

		PtrStep<float_type> gbuf;
		int length;
		mutable float_type* output;

		__device__ __forceinline__ void operator () () const
		{
			const float_type *beg = gbuf.ptr(blockIdx.x);
			const float_type *end = beg + length;

			int tid = threadIdx.x;

			float_type sum = 0.f;
			for (const float_type *t = beg + tid; t < end; t += STRIDE)
				sum += *t;

			__shared__ float_type smem[CTA_SIZE];

			smem[tid] = sum;
			__syncthreads();

			reduce<CTA_SIZE>(smem);

			if (tid == 0)
				output[blockIdx.x] = smem[0];
		}
	};

	__global__ void TransformEstimatorKernel2(const TranformReduction tr)
	{
		tr();
	}

	void estimateCombined(const Mat33& Rcurr, const float3& tcurr,
		const MapArr& vmap_curr, const MapArr& nmap_curr,
		const Mat33& Rprev, const float3& tprev, const Intr& intr,
		const MapArr& vmap_prev, const MapArr& nmap_prev,
		float distThres, float angleThres,
		DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf,
		float_type* matrixA_host, float_type* vectorB_host)
	{
		int cols = vmap_curr.cols();
		int rows = vmap_curr.rows() / 3;

		Combined cs;

		cs.Rcurr = Rcurr;
		cs.tcurr = tcurr;

		cs.vmap_curr = vmap_curr;
		cs.nmap_curr = nmap_curr;

		cs.Rprev = Rprev;
		cs.Rprev_inv = convert(convert(Rprev).inverse());
		cs.tprev = tprev;

		cs.intr = intr;

		cs.vmap_prev = vmap_prev;
		cs.nmap_prev = nmap_prev;

		cs.distThres = distThres;
		cs.angleThres = angleThres;

		cs.cols = cols;
		cs.rows = rows;

		//////////////////////////////

		dim3 block(Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y);
		dim3 grid(1, 1, 1);
		grid.x = divUp(cols, block.x);
		grid.y = divUp(rows, block.y);

		mbuf.create(TranformReduction::TOTAL);
		gbuf.create(TranformReduction::TOTAL, grid.x * grid.y);

		cs.gbuf = gbuf;

		combinedKernel << <grid, block >> >(cs);
		cudaSafeCall(cudaGetLastError());

		TranformReduction tr;
		tr.gbuf = gbuf;
		tr.length = grid.x * grid.y;
		tr.output = mbuf;

		TransformEstimatorKernel2 << <TranformReduction::TOTAL, TranformReduction::CTA_SIZE >> >(tr);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDeviceSynchronize());

		float_type host_data[TranformReduction::TOTAL];
		mbuf.download(host_data);

		int shift = 0;
		for (int i = 0; i < 6; ++i)  //rows
		for (int j = i; j < 7; ++j)    // cols + b
		{
			float_type value = host_data[shift++];
			if (j == 6)       // vector b
				vectorB_host[i] = value;
			else
				matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
		}
	}
#pragma endregion
}