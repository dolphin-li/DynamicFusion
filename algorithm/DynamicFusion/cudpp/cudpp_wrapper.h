#pragma once

#include "cudpp.h"
#include "cudpp_hash.h"

namespace cudpp_wrapper
{
	void exlusive_scan(unsigned int* d_in, unsigned int* d_out, int n);
}