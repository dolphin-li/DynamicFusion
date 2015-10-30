#include "kinect_util.h"
#include "device_utils.h"
#include <set>
namespace dfusion
{
//#define DEBUG_ASSIGN_BIG_ENERGY_TO_NO_CORR
//#define INVALID_DEPTH_VALUE 10

	__device__ __forceinline__ float3 read_float3_4(float4 a)
	{
		return make_float3(a.x, a.y, a.z);
	}

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
		cudaSafeCall(cudaGetLastError(), "calc_temperature_jet");
	}

#pragma endregion

#pragma region --generate image
	struct ImageGenerator
	{
		enum
		{
			CTA_SIZE_X = 32, CTA_SIZE_Y = 8
		};

		PtrStep<float4> vmap;
		PtrStep<float4> nmap;

		LightSource light;

		mutable PtrStepSz<PixelRGBA> dst;

		__device__ __forceinline__ void operator () () const
		{
			int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

			if (x >= dst.cols || y >= dst.rows)
				return;

			float4 v = vmap(y, x);
			float4 n = nmap(y, x);

			PixelRGBA color;
			color.a = color.r = color.g = color.b = 0;

			if (!isnan(v.x) && !isnan(n.x))
			{
				float3 acc_vec = make_float3(0.f, 0.f, 0.f);
				{
					float3 vec = normalized(light.pos - make_float3(v.x, v.y, v.z));
					float w = max(0.f, dot(vec, make_float3(n.x, n.y, n.z)));
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

		dst.create(vmap.rows(), vmap.cols());
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

		PtrStep<float4> nmap;
		Mat33 R;
		mutable PtrStepSz<PixelRGBA> dst;

		__device__ __forceinline__ void operator () () const
		{
			int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

			if (x >= dst.cols || y >= dst.rows)
				return;

			float4 n4 = nmap(y, x);
			float3 n = make_float3(n4.x, n4.y, n4.z);

			PixelRGBA color;
			color.a = color.r = color.g = color.b = 0;

			if (!isnan(n.x))
			{
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

	struct NormalGenerator1
	{
		enum
		{
			CTA_SIZE_X = 32, CTA_SIZE_Y = 8
		};

		PtrStep<float4> nmap;
		mutable PtrStepSz<PixelRGBA> dst;

		__device__ __forceinline__ void operator () () const
		{
			int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

			if (x >= dst.cols || y >= dst.rows)
				return;

			float4 n = nmap(y,x);

			PixelRGBA color;
			color.a = color.r = color.g = color.b = 0;

			if (!isnan(n.x))
			{
				color.r = max(0, min(255, int(255 * (n.x * 0.5f + 0.5f))));
				color.g = max(0, min(255, int(255 * (n.y * 0.5f + 0.5f))));
				color.b = max(0, min(255, int(255 * (n.z * 0.5f + 0.5f))));
				color.a = 255;
			}

			dst.ptr(y)[x] = color;
		}
	};

	__global__ void generateNormalKernel1(const NormalGenerator1 ig)
	{
		ig();
	}

	void generateNormalMap(const MapArr& nmap, ColorMap& dst, Mat33 R)
	{
		NormalGenerator ig;
		ig.nmap = nmap;
		ig.dst = dst;
		ig.R = R;

		dst.create(nmap.rows(), nmap.cols());

		dim3 block(NormalGenerator::CTA_SIZE_X, NormalGenerator::CTA_SIZE_Y);
		dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));

		generateNormalKernel << <grid, block >> >(ig);
		cudaSafeCall(cudaGetLastError(), "generateNormal");
	}

	void generateNormalMap(const MapArr& nmap, ColorMap& dst)
	{
		NormalGenerator1 ig;
		ig.nmap = nmap;
		ig.dst = dst;

		dst.create(nmap.rows(), nmap.cols());

		dim3 block(NormalGenerator1::CTA_SIZE_X, NormalGenerator1::CTA_SIZE_Y);
		dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));

		generateNormalKernel1 << <grid, block >> >(ig);
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

		cudaSafeCall(cudaGetLastError(), "bilateralKernel");

		cudaUnbindTexture(&g_bitex);
	}
#pragma endregion

#pragma region --compute vmap, nmap
	__global__ void computeVmapKernel(const PtrStepSz<depthtype> depth, PtrStep<float4> vmap, Intr intr)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u < depth.cols && v < depth.rows)
		{
			float z = depth.ptr(v)[u] * 0.001f; // load and convert: mm -> meters

			if (z > KINECT_NEAREST_METER)
			{
				float3 xyz = intr.uvd2xyz((float)u, (float)v, z);
				vmap(v, u) = make_float4(xyz.x, xyz.y, xyz.z, 1.f);
			}
			else
			{
#ifdef DEBUG_ASSIGN_BIG_ENERGY_TO_NO_CORR
				float3 xyz = intr.uvd2xyz((float)u, (float)v, INVALID_DEPTH_VALUE);
				vmap(v, u) = make_float4(xyz.x, xyz.y, xyz.z, 1.f);
#else
				vmap(v, u).x = numeric_limits<float>::quiet_NaN();
#endif
			}
		}
	}

	__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float4> vmap, PtrStep<float4> nmap)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= cols || v >= rows)
			return;

		if (u == cols - 1 || u == 0 || v == rows - 1 || v == 0)
		{
			nmap(v, u).x = numeric_limits<float>::quiet_NaN();
			return;
		}

		float3 v00, v01, v10;
		v00 = read_float3_4(vmap(v, u));
		v01 = read_float3_4(vmap(v, u + 1));
		v10 = read_float3_4(vmap(v - 1, u));

		if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x))
		{
			float3 r = normalized(cross(v01 - v00, v10 - v00));

			nmap(v, u) = make_float4(r.x, r.y, r.z, 0);
		}
		else
			nmap(v,u).x = numeric_limits<float>::quiet_NaN();
	}

	void createVMap(const Intr& intr, const DepthMap& depth, MapArr& vmap)
	{
		vmap.create(depth.rows(), depth.cols());

		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(depth.cols(), block.x);
		grid.y = divUp(depth.rows(), block.y);

		computeVmapKernel << <grid, block >> >(depth, vmap, intr);
		cudaSafeCall(cudaGetLastError(), "computeVmapKernel");
	}

	void createNMap(const MapArr& vmap, MapArr& nmap)
	{
		nmap.create(vmap.rows(), vmap.cols());

		int rows = vmap.rows();
		int cols = vmap.cols();

		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(cols, block.x);
		grid.y = divUp(rows, block.y);

		computeNmapKernel << <grid, block >> >(rows, cols, vmap, nmap);
		cudaSafeCall(cudaGetLastError(), "computeNmapKernel");
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
		cudaSafeCall(cudaGetLastError(), "pyrDownKernel");

		cudaUnbindTexture(&g_pydtex);
	}

#pragma endregion

#pragma region --resize map
	template<bool normalize>
	__global__ void resizeMapKernel(int drows, int dcols, int srows, 
		const PtrStep<float4> input, PtrStep<float4> output)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= dcols || y >= drows)
			return;

		const float qnan = numeric_limits<float>::quiet_NaN();

		int xs = x * 2;
		int ys = y * 2;

		float3 v00 = read_float3_4(input(ys + 0, xs + 0));
		float3 v01 = read_float3_4(input(ys + 0, xs + 1));
		float3 v10 = read_float3_4(input(ys + 1, xs + 0));
		float3 v11 = read_float3_4(input(ys + 1, xs + 1));

		if (isnan(v00.x) || isnan(v01.x) || isnan(v10.x) || isnan(v11.x))
		{
			output.ptr(y)[x].x = qnan;
			return;
		}
		else
		{
			float3 n = (v00 + v01 + v10 + v11) * 0.25f;

			if (normalize)
				n = normalized(n);

			output(y,x) = make_float4(n.x, n.y, n.z, 0);
		}
	}

	template<bool normalize>
	void resizeMap(const MapArr& input, MapArr& output)
	{
		int in_cols = input.cols();
		int in_rows = input.rows();

		int out_cols = in_cols / 2;
		int out_rows = in_rows / 2;

		output.create(out_rows, out_cols);

		dim3 block(32, 8);
		dim3 grid(divUp(out_cols, block.x), divUp(out_rows, block.y));
		resizeMapKernel<normalize> << < grid, block >> >(out_rows, out_cols, in_rows, input, output);
		cudaSafeCall(cudaGetLastError(), "resizeMapKernel");
		cudaSafeCall(cudaDeviceSynchronize(), "resizeMapKernel");
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
	__global__ void rigidTransformKernel(PtrStep<float4> vmap, PtrStep<float4> nmap, 
		Mat33 R, float3 t, int rows, int cols)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < cols && y < rows)
		{
			float3 v = read_float3_4(vmap(y, x));
			float3 n = read_float3_4(nmap(y, x));
			if (isnan(v.x) || isnan(n.x))
				return;

			v = R*v + t;
			n = R*n;

			vmap(y, x) = make_float4(v.x, v.y, v.z, 1.f);
			nmap(y, x) = make_float4(n.x, n.y, n.z, 0.f);
		}
	}

	void rigidTransform(MapArr& vmap, MapArr& nmap, Tbx::Transfo T)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		int cols = vmap.cols();
		int rows = vmap.rows();
		grid.x = divUp(cols, block.x);
		grid.y = divUp(rows, block.y);
		rigidTransformKernel << <grid, block >> >(vmap, nmap, convert(T.get_mat3()), 
			convert(T.get_translation()), rows, cols);
		cudaSafeCall(cudaGetLastError(), "rigidTransformKernel");
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

		PtrStep<float4> vmap_curr;
		PtrStep<float4> nmap_curr;

		Mat33 Rprev;
		Mat33 Rprev_inv;
		float3 tprev;

		Intr intr;

		PtrStep<float4> vmap_prev;
		PtrStep<float4> nmap_prev;

		float distThres;
		float angleThres;

		int cols;
		int rows;

		mutable PtrStep<float_type> gbuf;

		__device__ __forceinline__ bool search(int x, int y, float3& n, float3& d, float3& s) const
		{
			float3 ncurr = read_float3_4(nmap_curr(y, x));

			if (isnan(ncurr.x))
				return (false);

			float3 vcurr = read_float3_4(vmap_curr(y, x));
			float3 vcurr_g = Rcurr * vcurr + tcurr;
			float3 ncurr_g = Rcurr * ncurr;
			float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);	// prev camera coo space

			float3 uvd = intr.xyz2uvd(vcurr_cp);
			int2 ukr = make_int2(__float2int_rn(uvd.x), __float2int_rn(uvd.y));

			// we use opengl coordinate, thus world.z should < 0
			if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z >= 0)
				return (false);

			float3 nprev_g = Rprev * read_float3_4(nmap_prev(ukr.y, ukr.x));

			if (isnan(nprev_g.x))
				return (false);

			float3 vprev_g = Rprev * read_float3_4(vmap_prev(ukr.y, ukr.x)) + tprev;

			float dist = norm(vprev_g - vcurr_g);
			if (dist > distThres)
				return (false);

			float sine = norm(cross(ncurr_g, nprev_g));
			if (sine >= angleThres)
				return (false);

			n = nprev_g;
			d = vprev_g;
			s = vcurr_g;
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
		int rows = vmap_curr.rows();

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
		cudaSafeCall(cudaGetLastError(), "combinedKernel");

		TranformReduction tr;
		tr.gbuf = gbuf;
		tr.length = grid.x * grid.y;
		tr.output = mbuf;

		TransformEstimatorKernel2 << <TranformReduction::TOTAL, TranformReduction::CTA_SIZE >> >(tr);
		cudaSafeCall(cudaGetLastError(), "TransformEstimatorKernel2");
		cudaSafeCall(cudaDeviceSynchronize(), "TransformEstimatorKernel2");

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

#pragma region --error map
	__global__ void computeErrorMap_kernel(PtrStep<float4> vmap_live, PtrStep<float4> nmap_live, 
		PtrStep<float4> vmap_warp, PtrStep<float4> nmap_warp, PtrStep<PixelRGBA> errMap, 
		int cols, int rows, Intr intr, float range, float distThre, float angleThre_sin)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= cols || y >= rows)
			return;

		float3 nprev = read_float3_4(nmap_warp(y, x));
		float3 vprev = read_float3_4(vmap_warp(y, x));
		float3 uvd = intr.xyz2uvd(vprev);
		int2 ukr = make_int2(__float2int_rn(uvd.x), __float2int_rn(uvd.y));
		PixelRGBA color;
		color.a = color.r = 255;
		color.g = color.b = 0;

		if (!isnan(nprev.x) && ukr.x >= 0 && ukr.y >= 0 && ukr.x < cols && ukr.y < rows)
		{
			float3 ncurr = read_float3_4(nmap_live(ukr.y, ukr.x));
			float3 vcurr = read_float3_4(vmap_live(ukr.y, ukr.x));
			if (!isnan(ncurr.x))
			{
				float dist = norm(vprev - vcurr);
				float sine = norm(cross(ncurr, nprev));
				if (dist <= distThre && sine <= angleThre_sin)
				{
					float err = abs(dot(nprev, vprev - vcurr));
					err = min(1.f, err / range);
					color.r = color.g = color.b = err * 255;
				}
			}
		}

		errMap(y, x) = color;
	}

	void computeErrorMap(const MapArr& vmap_live, const MapArr& nmap_live,
		const MapArr& vmap_warp, const MapArr& nmap_warp, ColorMap& errMap, 
		Intr intr, float errMap_range, float distThre, float angleThre_sin)
	{
		int r = vmap_live.rows(), c = vmap_live.cols();

		if (r == 0 || c == 0)
			return;

		if (r != nmap_live.rows() || r != vmap_warp.rows() || r != nmap_warp.rows()
			|| c != nmap_live.cols() || c != vmap_warp.cols() || c != nmap_warp.cols())
			throw std::exception("error: size not matched in computeErrorMap()");

		if (errMap.cols() != c || errMap.rows() != r)
		{
			errMap.create(r, c);
		}

		dim3 block(32, 8);
		dim3 grid(divUp(c, block.x), divUp(r, block.y));
		
		computeErrorMap_kernel << <grid, block >> >(vmap_live, nmap_live, 
			vmap_warp, nmap_warp, errMap, c, r, intr, errMap_range, distThre, angleThre_sin);
		cudaSafeCall(cudaGetLastError(), "computeErrorMap");
	}
#pragma endregion
}