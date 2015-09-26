#include <device_functions.h>
#include <driver_functions.h>
#include <vector_functions.h>
#include <cuda_fp16.h>
#include <channel_descriptor.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include <surface_types.h>
#include <surface_functions.h>
#include "helper_math.h"

#include <thrust\device_ptr.h>
#include <thrust\sort.h>
#include <thrust\scan.h>
#include <thrust\unique.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <map>
#include "thrust_wrapper.h"

namespace thrust_wrapper
{
	// cached_allocator: a simple allocator for caching allocation requests
	class cached_allocator
	{
	public:
		// just allocate bytes
		typedef char value_type;
		enum{
			BlockSize = 512
		};

		cached_allocator() {}

		~cached_allocator()
		{
			// free all allocations when cached_allocator goes out of scope
			free_all();
		}

		char *allocate(std::ptrdiff_t num_bytes)
		{
			char *result = 0;

			int nBlocks = (num_bytes+BlockSize-1)/BlockSize;

			// search the cache for a free block
			free_blocks_type::iterator free_block = free_blocks.find(nBlocks);

			if (free_block != free_blocks.end())
			{
				//std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

				// get the pointer
				result = free_block->second;

				// erase from the free_blocks map
				free_blocks.erase(free_block);
			}
			else
			{
				// no allocation of the right size exists
				// create a new one with cuda::malloc
				// throw if cuda::malloc can't satisfy the request
				try
				{
					//std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;

					// allocate memory and convert cuda::pointer to raw pointer
					result = thrust::cuda::malloc<char>(num_bytes).get();
				}
				catch (std::runtime_error &e)
				{
					throw;
				}
			}

			// insert the allocated pointer into the allocated_blocks map
			allocated_blocks.insert(std::make_pair(result, nBlocks));

			return result;
		}

		void deallocate(char *ptr, size_t n)
		{
			// erase the allocated block from the allocated blocks map
			allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
			std::ptrdiff_t num_bytes = iter->second;
			allocated_blocks.erase(iter);

			// insert the block into the free blocks map
			free_blocks.insert(std::make_pair(num_bytes, ptr));
		}

	private:
		typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
		typedef std::map<char *, std::ptrdiff_t>     allocated_blocks_type;

		free_blocks_type      free_blocks;
		allocated_blocks_type allocated_blocks;

		void free_all()
		{
			std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

			// deallocate all outstanding blocks in both lists
			for (free_blocks_type::iterator i = free_blocks.begin();
				i != free_blocks.end();
				++i)
			{
				// transform the pointer to cuda::pointer before calling cuda::free
				thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
			}

			for (allocated_blocks_type::iterator i = allocated_blocks.begin();
				i != allocated_blocks.end();
				++i)
			{
				// transform the pointer to cuda::pointer before calling cuda::free
				thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
			}
		}

	};

	cached_allocator g_allocator;

	void stable_sort_by_key(int* key_d, float4* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<float4> points_begin(value_d);
		thrust::stable_sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void stable_sort_by_key(float* key_d, int* value_d, int n)
	{
		thrust::device_ptr<float> key_begin(key_d);
		thrust::device_ptr<float> key_end(key_d + n);
		thrust::device_ptr<int> points_begin(value_d);
		thrust::stable_sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void sort_by_key(int* key_d, float4* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<float4> points_begin(value_d);
		thrust::sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void sort_by_key(float* key_d, int* value_d, int n)
	{
		thrust::device_ptr<float> key_begin(key_d);
		thrust::device_ptr<float> key_end(key_d + n);
		thrust::device_ptr<int> points_begin(value_d);
		thrust::sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void exclusive_scan(const int* in, int* out, int n)
	{
		thrust::device_ptr<int> in_begin((int*)in);
		thrust::device_ptr<int> in_end((int*)in + n);
		thrust::device_ptr<int> out_begin(out);
		thrust::exclusive_scan(thrust::cuda::par(g_allocator), in_begin, in_end, out_begin);
	}

	void inclusive_scan_by_key(int* key_d, float4* value_d, float4* dst_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<float4> points_begin(value_d);
		thrust::device_ptr<float4> dst_begin(dst_d);
		thrust::inclusive_scan_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin, dst_begin);
	}

	size_t unique_by_key(int* key_d, float4* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<float4> points_begin(value_d);
		auto ptr = thrust::unique_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
		return ptr.first - key_begin;
	}

	size_t unique_counting_by_key_copy(const int* key_d, int input_count_begin,
		int* out_key_d, int* out_value_d, int n)
	{
		thrust::device_ptr<int> key_begin((int*)key_d);
		thrust::device_ptr<int> key_end((int*)key_d + n);
		thrust::counting_iterator<int> input_value_begin(input_count_begin);
		thrust::device_ptr<int> out_key_begin(out_key_d);
		thrust::device_ptr<int> out_value_begin(out_value_d);
		auto ptr = thrust::unique_by_key_copy(thrust::cuda::par(g_allocator), key_begin, key_end, input_value_begin,
			out_key_begin, out_value_begin);
		return ptr.first - out_key_begin;
	}

	void scatter(const int* in_d, const int* map_d, int* out_d, int n)
	{
		thrust::device_ptr<int> first((int*)in_d);
		thrust::device_ptr<int> last((int*)in_d + n);
		thrust::device_ptr<int> map((int*)map_d);
		thrust::device_ptr<int> output(out_d);
		thrust::scatter(thrust::cuda::par(g_allocator), first, last, map, output);
	}

	void gather(const int* in_d, const int* map_d, int* out_d, int n)
	{
		thrust::device_ptr<int> first((int*)in_d);
		thrust::device_ptr<int> map((int*)map_d);
		thrust::device_ptr<int> map_last((int*)map_d + n);
		thrust::device_ptr<int> output(out_d);
		thrust::gather(thrust::cuda::par(g_allocator), map, map_last, first, output);
	}


	void gather(const float4* in_d, const int* map_d, float4* out_d, int n)
	{
		thrust::device_ptr<float4> first((float4*)in_d);
		thrust::device_ptr<int> map((int*)map_d);
		thrust::device_ptr<int> map_last((int*)map_d + n);
		thrust::device_ptr<float4> output(out_d);
		thrust::gather(thrust::cuda::par(g_allocator), map, map_last, first, output);
	}

	//! converts a float4 point (xyz) to a tuple of three float vals (used to separate the
	//! float4 input buffer into three arrays in the beginning of the tree build)
	struct pointxyz_to_px_py_pz
	{
		__device__
		thrust::tuple<float, float, float> operator()(const float4& val)
		{
			return thrust::make_tuple(val.x, val.y, val.z);
		}
	};

	void seperate_channels(const float4* xyzw, float* x, float* y, float* z, int n)
	{
		thrust::device_ptr<float4> xyzw_begin((float4*)xyzw);
		thrust::device_ptr<float4> xyzw_end((float4*)xyzw + n);
		thrust::device_ptr<float> xs(x);
		thrust::device_ptr<float> ys(y);
		thrust::device_ptr<float> zs(z);
		thrust::transform(thrust::cuda::par(g_allocator), xyzw_begin, xyzw_end,
			thrust::make_zip_iterator(thrust::make_tuple(xs, ys, zs)),
			pointxyz_to_px_py_pz());
	}

	void make_counting_array(int* ptr_d, int n, int begin)
	{
		thrust::device_ptr<int> data(ptr_d);
		thrust::counting_iterator<int> it(begin);
		thrust::copy(thrust::cuda::par(g_allocator), it, it + n, data);
	}

	void copy(int* dst, const int* src, int size)
	{
		thrust::device_ptr<int> src_begin((int*)src);
		thrust::device_ptr<int> src_end((int*)src + size);
		thrust::device_ptr<int> dst_begin(dst);
		thrust::copy(thrust::cuda::par(g_allocator), src_begin, src_end, dst_begin);
	}

	void copy(float* dst, const float* src, int size)
	{
		thrust::device_ptr<float> src_begin((float*)src);
		thrust::device_ptr<float> src_end((float*)src + size);
		thrust::device_ptr<float> dst_begin(dst);
		thrust::copy(thrust::cuda::par(g_allocator), src_begin, src_end, dst_begin);
	}

	void assign(int* ptr_d, int value, int n)
	{
		thrust::device_ptr<int> data_begin(ptr_d);
		thrust::device_ptr<int> data_end(ptr_d + n);
		thrust::fill(thrust::cuda::par(g_allocator), data_begin, data_end, value);
	}

	void assign(float* ptr_d, float value, int n)
	{
		thrust::device_ptr<float> data_begin(ptr_d);
		thrust::device_ptr<float> data_end(ptr_d + n);
		thrust::fill(thrust::cuda::par(g_allocator), data_begin, data_end, value);
	}

	void assign(float4* ptr_d, float4 value, int n)
	{
		thrust::device_ptr<float4> data_begin(ptr_d);
		thrust::device_ptr<float4> data_end(ptr_d + n);
		thrust::fill(thrust::cuda::par(g_allocator), data_begin, data_end, value);
	}

}