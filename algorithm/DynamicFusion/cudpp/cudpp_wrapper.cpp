#include "cudpp_wrapper.h"
#include <stdio.h>

#pragma comment(lib, "cudpp_hash64.lib")
#pragma comment(lib, "cudpp64.lib")

namespace cudpp_wrapper
{
#define CUDPP_CHECK(err, str) \
	if ((err) != CUDPPResult::CUDPP_SUCCESS){\
	printf("error: %s\n", str);\
	exit(-1); \
	}
	CUDPPHandle g_theCudpp;
	static CUDPPResult init_cudpp()
	{
		CUDPPResult r;
		r = cudppCreate(&g_theCudpp);
		CUDPP_CHECK(r, "cannot init cudpp!");
		return r;
	}
	CUDPPResult g_cudpp_initialized = init_cudpp();

	void exlusive_scan(unsigned int* d_in, unsigned int* d_out, int n)
	{
		CUDPPConfiguration config;
		config.op = CUDPP_ADD;
		config.datatype = CUDPP_UINT;
		config.algorithm = CUDPP_SCAN;
		config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

		// Run the scan
		CUDPPHandle scanplan;
		CUDPP_CHECK(cudppPlan(g_theCudpp, &scanplan, config, n, 1, 0), "scan plan create");
		CUDPP_CHECK(cudppScan(scanplan, d_out, d_in, n), "scan");
		CUDPP_CHECK(cudppDestroyPlan(scanplan), "scan plan destroy");
	}
}