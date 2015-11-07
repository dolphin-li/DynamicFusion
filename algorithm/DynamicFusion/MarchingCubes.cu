#include "MarchingCubes.h"
#include "TsdfVolume.h"
#include "cudpp\thrust_wrapper.h"
#include "GpuMesh.h"
#include "device_utils.h"

namespace dfusion
{
#pragma region --marching cubes table data
	enum{
		TableSize = 256,
		TableSize2 = 16
	};
	__constant__ int g_edgeTable[TableSize] = {
		0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
		0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
		0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
		0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
		0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
		0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
		0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
		0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
		0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
		0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
		0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
		0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
		0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
		0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
		0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
		0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
		0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };
	__constant__  char g_triTable[TableSize][TableSize2] =
	{ { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
	{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
	{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
	{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
	{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
	{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
	{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
	{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
	{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
	{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
	{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
	{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
	{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
	{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
	{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
	{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
	{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
	{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
	{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
	{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
	{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
	{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
	{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
	{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
	{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
	{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
	{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
	{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
	{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
	{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
	{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
	{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
	{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
	{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
	{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
	{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
	{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
	{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
	{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
	{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
	{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
	{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
	{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
	{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
	{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
	{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
	{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
	{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
	{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
	{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
	{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
	{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
	{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
	{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
	{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
	{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
	{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
	{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
	{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
	{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
	{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
	{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
	{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
	{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
	{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
	{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
	{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
	{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
	{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
	{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
	{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
	{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
	{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
	{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
	{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
	{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
	{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
	{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
	{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
	{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
	{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
	{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
	{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
	{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
	{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
	{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
	{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
	{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
	{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
	{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
	{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
	{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
	{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 } };

	// number of vertices for each case above
	__constant__ char g_numVertsTable[TableSize] =
	{
		0,
		3,
		3,
		6,
		3,
		6,
		6,
		9,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		6,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		9,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		9,
		6,
		9,
		9,
		6,
		9,
		12,
		12,
		9,
		9,
		12,
		12,
		9,
		12,
		15,
		15,
		6,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		9,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		15,
		9,
		12,
		12,
		15,
		12,
		15,
		15,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		6,
		9,
		9,
		12,
		12,
		15,
		12,
		15,
		9,
		6,
		9,
		12,
		12,
		9,
		12,
		15,
		9,
		6,
		12,
		15,
		15,
		12,
		15,
		6,
		12,
		3,
		3,
		6,
		6,
		9,
		6,
		9,
		9,
		12,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		9,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		15,
		9,
		6,
		12,
		9,
		12,
		9,
		15,
		6,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		15,
		9,
		12,
		12,
		15,
		12,
		15,
		15,
		12,
		9,
		12,
		12,
		9,
		12,
		15,
		15,
		12,
		12,
		9,
		15,
		6,
		15,
		12,
		6,
		3,
		6,
		9,
		9,
		12,
		9,
		12,
		12,
		15,
		9,
		12,
		12,
		15,
		6,
		9,
		9,
		6,
		9,
		12,
		12,
		15,
		12,
		15,
		15,
		6,
		12,
		9,
		15,
		12,
		9,
		6,
		12,
		3,
		9,
		12,
		12,
		15,
		12,
		15,
		9,
		12,
		12,
		15,
		15,
		6,
		9,
		12,
		6,
		3,
		6,
		9,
		9,
		6,
		9,
		12,
		6,
		3,
		9,
		6,
		12,
		3,
		6,
		3,
		3,
		0,
	};

#pragma endregion

#pragma region --classifyVoxel
	__device__ int global_count = 0;
	__device__ int output_count;
	__device__ unsigned int blocks_done = 0;

	struct OccupiedVoxels
	{
		enum
		{
			CTA_SIZE_X = 32,
			CTA_SIZE_Y = 8,
			CTA_SIZE_Z = 2,
			CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y * CTA_SIZE_Z,
			WARPS_COUNT = CTA_SIZE / Warp::WARP_SIZE
		};

		mutable unsigned int* voxels_indeces;
		mutable unsigned int* vetexes_number;

		cudaTextureObject_t tex;
		MarchingCubes::Tile tile;
		float isoValue;
		float minWeights;

		__device__ __forceinline__ void operator () () const
		{
			const int tx = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			const int ty = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
			const int tz = threadIdx.z + blockIdx.z * CTA_SIZE_Z;
			const int x = (tx << tile.level) + tile.begin.x;
			const int y = (ty << tile.level) + tile.begin.y;
			const int z = (tz << tile.level) + tile.begin.z;
			const int rx = ((tile.end.x - tile.begin.x) >> tile.level);
			const int ry = ((tile.end.y - tile.begin.y) >> tile.level);
			const int s = (1 << tile.level);

			if (x >= tile.end.x || y >= tile.end.y || z >= tile.end.z)
				return;

			int ftid = Block::flattenedThreadId();
			int warp_id = Warp::id();
			int lane_id = Warp::laneId();

			volatile __shared__ int warps_buffer[WARPS_COUNT];

			float field[8];
			float2 tdata = unpack_tsdf(read_tsdf_texture(tex, x + 0, y + 0, z + 0));
			field[0] = tdata.x * (tdata.y >= minWeights);
			tdata = unpack_tsdf(read_tsdf_texture(tex, x + s, y + 0, z + 0));
			field[1] = tdata.x * (tdata.y >= minWeights);
			tdata = unpack_tsdf(read_tsdf_texture(tex, x + s, y + s, z + 0));
			field[2] = tdata.x * (tdata.y >= minWeights);
			tdata = unpack_tsdf(read_tsdf_texture(tex, x + 0, y + s, z + 0));
			field[3] = tdata.x * (tdata.y >= minWeights);
			tdata = unpack_tsdf(read_tsdf_texture(tex, x + 0, y + 0, z + s));
			field[4] = tdata.x * (tdata.y >= minWeights);
			tdata = unpack_tsdf(read_tsdf_texture(tex, x + s, y + 0, z + s));
			field[5] = tdata.x * (tdata.y >= minWeights);
			tdata = unpack_tsdf(read_tsdf_texture(tex, x + s, y + s, z + s));
			field[6] = tdata.x * (tdata.y >= minWeights);
			tdata = unpack_tsdf(read_tsdf_texture(tex, x + 0, y + s, z + s));
			field[7] = tdata.x * (tdata.y >= minWeights);

			int cubeindex = 0;
			if (field[0] && field[1] && field[2] && field[3] && field[4]
				&& field[5] && field[6] && field[7])// exactly 0 means no value, thus should be ignored
			{
				cubeindex |= (int(field[0] < isoValue) << 0);
				cubeindex |= (int(field[1] < isoValue) << 1);//  * 2;
				cubeindex |= (int(field[2] < isoValue) << 2);//  * 4;
				cubeindex |= (int(field[3] < isoValue) << 3);//  * 8;
				cubeindex |= (int(field[4] < isoValue) << 4);//  * 16;
				cubeindex |= (int(field[5] < isoValue) << 5);//  * 32;
				cubeindex |= (int(field[6] < isoValue) << 6);//  * 64;
				cubeindex |= (int(field[7] < isoValue) << 7);//  * 128;
			}

			int numVerts = g_numVertsTable[cubeindex];
			int total = __popc(__ballot(numVerts > 0));
			if (total)
			{
				if (lane_id == 0)
				{
					int old = atomicAdd(&global_count, total);
					warps_buffer[warp_id] = old;
				}

				int old_global_voxels_count = warps_buffer[warp_id];
				int offs = Warp::binaryExclScan(__ballot(numVerts > 0));
				if (old_global_voxels_count + offs < tile.max_num_activeVoxels && numVerts > 0)
				{
					voxels_indeces[old_global_voxels_count + offs] = ry*rx * tz + ty*rx + tx;
					vetexes_number[old_global_voxels_count + offs] = numVerts;
				}
			}
			/////////////////////////
			// prepare for future scans
			if (ftid == 0)
			{
				unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
				unsigned int value = atomicInc(&blocks_done, total_blocks);

				//last block
				if (value == total_blocks - 1)
				{
					output_count = min(tile.max_num_activeVoxels, global_count);
					blocks_done = 0;
					global_count = 0;
				}
			}
		} /* operator () */
	};
	__global__ void getOccupiedVoxelsKernel(const OccupiedVoxels ov) { ov(); }

	static unsigned int get_scanned_sum(unsigned int* d_ary, unsigned int* d_scan, int n)
	{
		if (n == 0)
			return 0;
		unsigned int lastElement, lastScanElement;
		cudaSafeCall(cudaMemcpy((void *)&lastElement,
			(void *)(d_ary + n- 1),
			sizeof(unsigned int), cudaMemcpyDeviceToHost),
			"get_scanned_sum 1");
		cudaSafeCall(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_scan + n - 1),
			sizeof(unsigned int), cudaMemcpyDeviceToHost),
			"get_scanned_sum 2");
		return lastElement + lastScanElement;
	}

	void MarchingCubes::classifyVoxel(Tile& tile)
	{
		int zero_mem = 0;
		cudaSafeCall(cudaMemcpyToSymbol(output_count, &zero_mem, sizeof(int)),
			"MarchingCubes::classifyVoxel 1");
		cudaSafeCall(cudaMemcpyToSymbol(global_count, &zero_mem, sizeof(int)),
			"MarchingCubes::classifyVoxel 2");
		cudaSafeCall(cudaMemcpyToSymbol(blocks_done, &zero_mem, sizeof(int)),
			"MarchingCubes::classifyVoxel 3");

		OccupiedVoxels ov;

		ov.voxels_indeces = m_compVoxelArray.ptr();
		ov.vetexes_number = m_voxelVerts.ptr();
		ov.tex = m_volTex;
		ov.tile = tile;
		ov.isoValue = m_param.marching_cube_isoValue;
		ov.minWeights = m_param.marchingCube_min_valied_weight;

		dim3 block(OccupiedVoxels::CTA_SIZE_X, OccupiedVoxels::CTA_SIZE_Y,
			OccupiedVoxels::CTA_SIZE_Z);
		dim3 grid(divUp((tile.end.x - tile.begin.x) >> tile.level, block.x), 
			divUp((tile.end.y - tile.begin.y) >> tile.level, block.y),
			divUp((tile.end.z - tile.begin.z) >> tile.level, block.z));

		getOccupiedVoxelsKernel << <grid, block >> >(ov);
		cudaSafeCall(cudaGetLastError(),
			"MarchingCubes::classifyVoxel getOccupiedVoxelsKernel");
		cudaSafeCall(cudaDeviceSynchronize(),
			"MarchingCubes::classifyVoxel getOccupiedVoxelsKernel");

		cudaSafeCall(cudaMemcpyFromSymbol(&tile.num_activeVoxels, output_count, sizeof(int)),
			"MarchingCubes::classifyVoxel 4");

		if (tile.num_activeVoxels == tile.max_num_activeVoxels)
		{
			printf("warning: memory limit achieved in marching cube, you may enlarge \
				   marching_cube_max_activeVoxel_ratio in Param()\n");
		}

		// scan to get total number of vertices
		thrust_wrapper::exclusive_scan(m_voxelVerts.ptr(), m_voxelVertsScan.ptr(), tile.num_activeVoxels);
		tile.nverts = get_scanned_sum(m_voxelVerts.ptr(), m_voxelVertsScan.ptr(), tile.num_activeVoxels);
	}
#pragma endregion


#pragma region --generate triangles
	enum{
		GEN_TRI_N_THREADS = 32
	};

	__device__ __forceinline__ float3 lerp(float3 a, float3 b, float t)
	{
		return a + t*(b - a);
	}
	__device__ __forceinline__ float4 lerp(float4 a, float4 b, float t)
	{
		return a + t*(b - a);
	}

	// compute interpolated vertex along an edge
	__device__ __forceinline__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
	{
		float t = (isolevel - f0) / (f1 - f0);
		return dfusion::lerp(p0, p1, t);
	}
	__device__ __forceinline__ float4 colorInterp(float isolevel, float4 p0, float4 p1, float f0, float f1)
	{
		float t = (isolevel - f0) / (f1 - f0);
		return dfusion::lerp(p0, p1, t) * COLOR_FUSION_BRIGHTNESS; // make it brighter
	}

	// calculate triangle normal
	__device__ __forceinline__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
	{
		float3 edge0 = *v1 - *v0;
		float3 edge1 = *v2 - *v0;
		// note - it's faster to perform normalization in vertex shader rather than here
		return normalize(cross(edge0, edge1));
	}

	// version that calculates flat surface normal for each triangle
	__global__ void generateTrianglesKernel(GpuMesh::PointType *pos, GpuMesh::PointType *norm,
		cudaTextureObject_t tex, MarchingCubes::Tile tile,
		unsigned int *compactedVoxelArray, unsigned int *numVertsScanned, float isoValue, float minWeights
#ifdef ENABLE_COLOR_FUSION
		,GpuMesh::PointType* color
#endif
		)
	{
		unsigned int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		unsigned int tid = __mul24(blockId, blockDim.x<<3) + threadIdx.x;

		const int rx = ((tile.end.x - tile.begin.x) >> tile.level);
		const int ry = ((tile.end.y - tile.begin.y) >> tile.level);
		const int s = (1 << tile.level);
		const float svsz = tile.voxelSize*s;
		const int rxry = rx*ry;

		for (int block_iter = 0; block_iter < 8; block_iter++, tid += blockDim.x)
		{
			// cannot return due to __syncthreads()
			if (tid < tile.num_activeVoxels)
			{
				unsigned int voxelId = compactedVoxelArray[tid];

				// compute position in 3d grid
				uint3 gridPos;
				gridPos.z = voxelId / rxry;
				gridPos.y = (voxelId - gridPos.z*rxry) / rx;
				gridPos.x = voxelId % rx;
				gridPos.x = tile.begin.x + (gridPos.x << tile.level);
				gridPos.y = tile.begin.y + (gridPos.y << tile.level);
				gridPos.z = tile.begin.z + (gridPos.z << tile.level);

				// calculate cell vertex positions
				float3 v[8];
				float field[8];
				v[0] = make_float3(tile.origion.x + gridPos.x * tile.voxelSize,
					tile.origion.y + gridPos.y * tile.voxelSize,
					tile.origion.z + gridPos.z * tile.voxelSize);
				v[1] = make_float3(v[0].x + svsz, v[0].y, v[0].z);
				v[2] = make_float3(v[0].x + svsz, v[0].y + svsz, v[0].z);
				v[3] = make_float3(v[0].x, v[0].y + svsz, v[0].z);
				v[4] = make_float3(v[0].x, v[0].y, v[0].z + svsz);
				v[5] = make_float3(v[0].x + svsz, v[0].y, v[0].z + svsz);
				v[6] = make_float3(v[0].x + svsz, v[0].y + svsz, v[0].z + svsz);
				v[7] = make_float3(v[0].x, v[0].y + svsz, v[0].z + svsz);

#ifdef ENABLE_COLOR_FUSION
				float4 c[8];
				float2 tdata;
				unpack_tsdf_vw_rgba(read_tsdf_texture(tex, gridPos.x + 0, gridPos.y + 0, gridPos.z + 0), tdata, c[0]);
				field[0] = tdata.x * (tdata.y >= minWeights);
				unpack_tsdf_vw_rgba(read_tsdf_texture(tex, gridPos.x + s, gridPos.y + 0, gridPos.z + 0), tdata, c[1]);
				field[1] = tdata.x * (tdata.y >= minWeights);
				unpack_tsdf_vw_rgba(read_tsdf_texture(tex, gridPos.x + s, gridPos.y + s, gridPos.z + 0), tdata, c[2]);
				field[2] = tdata.x * (tdata.y >= minWeights);
				unpack_tsdf_vw_rgba(read_tsdf_texture(tex, gridPos.x + 0, gridPos.y + s, gridPos.z + 0), tdata, c[3]);
				field[3] = tdata.x * (tdata.y >= minWeights);
				unpack_tsdf_vw_rgba(read_tsdf_texture(tex, gridPos.x + 0, gridPos.y + 0, gridPos.z + s), tdata, c[4]);
				field[4] = tdata.x * (tdata.y >= minWeights);
				unpack_tsdf_vw_rgba(read_tsdf_texture(tex, gridPos.x + s, gridPos.y + 0, gridPos.z + s), tdata, c[5]);
				field[5] = tdata.x * (tdata.y >= minWeights);
				unpack_tsdf_vw_rgba(read_tsdf_texture(tex, gridPos.x + s, gridPos.y + s, gridPos.z + s), tdata, c[6]);
				field[6] = tdata.x * (tdata.y >= minWeights);
				unpack_tsdf_vw_rgba(read_tsdf_texture(tex, gridPos.x + 0, gridPos.y + s, gridPos.z + s), tdata, c[7]);
				field[7] = tdata.x * (tdata.y >= minWeights);
#else
				float2 tdata = unpack_tsdf(read_tsdf_texture(tex, gridPos.x + 0, gridPos.y + 0, gridPos.z + 0));
				field[0] = tdata.x * (tdata.y >= minWeights);
				tdata = unpack_tsdf(read_tsdf_texture(tex, gridPos.x + s, gridPos.y + 0, gridPos.z + 0));
				field[1] = tdata.x * (tdata.y >= minWeights);
				tdata = unpack_tsdf(read_tsdf_texture(tex, gridPos.x + s, gridPos.y + s, gridPos.z + 0));
				field[2] = tdata.x * (tdata.y >= minWeights);
				tdata = unpack_tsdf(read_tsdf_texture(tex, gridPos.x + 0, gridPos.y + s, gridPos.z + 0));
				field[3] = tdata.x * (tdata.y >= minWeights);
				tdata = unpack_tsdf(read_tsdf_texture(tex, gridPos.x + 0, gridPos.y + 0, gridPos.z + s));
				field[4] = tdata.x * (tdata.y >= minWeights);
				tdata = unpack_tsdf(read_tsdf_texture(tex, gridPos.x + s, gridPos.y + 0, gridPos.z + s));
				field[5] = tdata.x * (tdata.y >= minWeights);
				tdata = unpack_tsdf(read_tsdf_texture(tex, gridPos.x + s, gridPos.y + s, gridPos.z + s));
				field[6] = tdata.x * (tdata.y >= minWeights);
				tdata = unpack_tsdf(read_tsdf_texture(tex, gridPos.x + 0, gridPos.y + s, gridPos.z + s));
				field[7] = tdata.x * (tdata.y >= minWeights);
#endif

				// recalculate flag, faster than store in global memory
				int cubeindex = 0;
				if (field[0] && field[1] && field[2] && field[3] && field[4]
					&& field[5] && field[6] && field[7])// exactly 0 means no value, thus should be ignored
				{
					cubeindex |= (int(field[0] < isoValue) << 0);
					cubeindex |= (int(field[1] < isoValue) << 1);//  * 2;
					cubeindex |= (int(field[2] < isoValue) << 2);//  * 4;
					cubeindex |= (int(field[3] < isoValue) << 3);//  * 8;
					cubeindex |= (int(field[4] < isoValue) << 4);//  * 16;
					cubeindex |= (int(field[5] < isoValue) << 5);//  * 32;
					cubeindex |= (int(field[6] < isoValue) << 6);//  * 64;
					cubeindex |= (int(field[7] < isoValue) << 7);//  * 128;
				}

				// find the vertices where the surface intersects the cube
				// use shared memory to avoid using local
				__shared__ float3 vertlist[12 * GEN_TRI_N_THREADS];

				vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
				vertlist[GEN_TRI_N_THREADS + threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
				vertlist[(GEN_TRI_N_THREADS * 2) + threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
				vertlist[(GEN_TRI_N_THREADS * 3) + threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
				vertlist[(GEN_TRI_N_THREADS * 4) + threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
				vertlist[(GEN_TRI_N_THREADS * 5) + threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
				vertlist[(GEN_TRI_N_THREADS * 6) + threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
				vertlist[(GEN_TRI_N_THREADS * 7) + threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
				vertlist[(GEN_TRI_N_THREADS * 8) + threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
				vertlist[(GEN_TRI_N_THREADS * 9) + threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
				vertlist[(GEN_TRI_N_THREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
				vertlist[(GEN_TRI_N_THREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
#ifdef ENABLE_COLOR_FUSION
				__shared__ float4 clorlist[12 * GEN_TRI_N_THREADS];

				clorlist[threadIdx.x] = colorInterp(isoValue, c[0], c[1], field[0], field[1]);
				clorlist[GEN_TRI_N_THREADS + threadIdx.x] = colorInterp(isoValue, c[1], c[2], field[1], field[2]);
				clorlist[(GEN_TRI_N_THREADS * 2) + threadIdx.x] = colorInterp(isoValue, c[2], c[3], field[2], field[3]);
				clorlist[(GEN_TRI_N_THREADS * 3) + threadIdx.x] = colorInterp(isoValue, c[3], c[0], field[3], field[0]);
				clorlist[(GEN_TRI_N_THREADS * 4) + threadIdx.x] = colorInterp(isoValue, c[4], c[5], field[4], field[5]);
				clorlist[(GEN_TRI_N_THREADS * 5) + threadIdx.x] = colorInterp(isoValue, c[5], c[6], field[5], field[6]);
				clorlist[(GEN_TRI_N_THREADS * 6) + threadIdx.x] = colorInterp(isoValue, c[6], c[7], field[6], field[7]);
				clorlist[(GEN_TRI_N_THREADS * 7) + threadIdx.x] = colorInterp(isoValue, c[7], c[4], field[7], field[4]);
				clorlist[(GEN_TRI_N_THREADS * 8) + threadIdx.x] = colorInterp(isoValue, c[0], c[4], field[0], field[4]);
				clorlist[(GEN_TRI_N_THREADS * 9) + threadIdx.x] = colorInterp(isoValue, c[1], c[5], field[1], field[5]);
				clorlist[(GEN_TRI_N_THREADS * 10) + threadIdx.x] = colorInterp(isoValue, c[2], c[6], field[2], field[6]);
				clorlist[(GEN_TRI_N_THREADS * 11) + threadIdx.x] = colorInterp(isoValue, c[3], c[7], field[3], field[7]);
#endif
				__syncthreads();

				// output triangle vertices
				unsigned int numVerts = g_numVertsTable[cubeindex];

				for (int i = 0; i < numVerts; i += 3)
				{
					unsigned int index = numVertsScanned[tid] + i;

					float3 *v[3];
#ifdef ENABLE_COLOR_FUSION
					float4 *c[3];
#endif
					for (int k = 0; k < 3; k++)
					{
						unsigned int edge = g_triTable[cubeindex][i + k];
						v[2-k] = &vertlist[(edge*GEN_TRI_N_THREADS) + threadIdx.x];
#ifdef ENABLE_COLOR_FUSION
						c[2-k] = &clorlist[(edge*GEN_TRI_N_THREADS) + threadIdx.x];
#endif
					}

					// calculate triangle surface normal
					float3 n = calcNormal(v[0], v[1], v[2]);

					if (index < tile.nverts - 2)
					{
						pos[index] = GpuMesh::to_point(*v[0]);
						norm[index] = GpuMesh::to_point(n);

						pos[index + 1] = GpuMesh::to_point(*v[1]);
						norm[index + 1] = GpuMesh::to_point(n);

						pos[index + 2] = GpuMesh::to_point(*v[2]);
						norm[index + 2] = GpuMesh::to_point(n);
#ifdef ENABLE_COLOR_FUSION
						color[index] = *c[0];
						color[index + 1] = *c[1];
						color[index + 2] = *c[2];
#endif
					}
				}// end for i
			}// end if tid < activeVoxels
		}// end for block_iter
	}

	void MarchingCubes::generateTriangles(const Tile& tile, GpuMesh& result)
	{
		result.create(tile.nverts);
		if (tile.nverts == 0)
			return;

		dim3 block(GEN_TRI_N_THREADS);
		dim3 grid(divUp(tile.num_activeVoxels, block.x<<3));

		result.lockVertsNormals();
		generateTrianglesKernel << <grid, block >> >(
			result.verts(), result.normals(),
			m_volTex, tile,
			m_compVoxelArray, m_voxelVertsScan, 
			m_param.marching_cube_isoValue,
			m_param.marchingCube_min_valied_weight
#ifdef ENABLE_COLOR_FUSION
			,result.colors()
#endif
			);
		cudaSafeCall(cudaGetLastError(), "generateTriangles");
		result.unlockVertsNormals();
	}
#pragma endregion
}