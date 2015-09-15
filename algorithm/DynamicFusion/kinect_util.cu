#include "kinect_util.h"

namespace dfusion
{
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
		jetRgb_d(v,u) = make_uchar4(255*(c[0] * val + c[1]), 255*(c[2] * val + c[3]), 255*(c[4] * val + c[5]), 255);
	}

	void calc_temperature_jet(PtrStepSz<depthtype> depth_d, PtrStepSz<uchar4> jetRgb_d, float shift, float div)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(depth_d.cols, block.x);
		grid.y = divUp(depth_d.rows, block.y);

		calcTemperatureJetKernel << <grid, block >> >(depth_d, jetRgb_d, shift, div);
		cudaSafeCall(cudaGetLastError());
	}
}