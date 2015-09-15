#pragma once

#include "Convolution_Helper.h"
#include "kdtree\AABB.h"
class ObjMesh;
namespace mpu
{
	template<typename T>
	class VolumeTemplate
	{
	public:
		VolumeTemplate(){}
		~VolumeTemplate(){ clear(); }

		//=================================================================
		void clear()
		{
			m_data.clear();
			m_resolution = 0;
		}
		void resize(unsigned short  x, unsigned short  y, unsigned short  z)
		{
			m_resolution = ldp::UShort3(x, y, z);
			m_data.resize(x*y*z);
		}
		void resize(ldp::UShort3 p)
		{
			resize(p[0], p[1], p[2]);
		}
		void fill(T v)
		{
			std::fill(m_data.begin(), m_data.end(), v);
		}

		//=================================================================
		/// data access methods
		ldp::UShort3 getResolution()const{ return m_resolution; }
		int stride_X()const{ return 1; }
		int stride_Y()const{ return m_resolution[0]; }
		int stride_Z()const{ return m_resolution[0] * m_resolution[1]; }

		bool contains(ldp::Int3 idx)const
		{
			return idx[0] < m_resolution[0] && idx[0] >= 0
				&& idx[1] < m_resolution[1] && idx[1] >= 0
				&& idx[2] < m_resolution[2] && idx[2] >= 0;
		}

		// get the value from the data array
		T& operator()(ldp::UShort3 p){ return (*this)(p[0], p[1], p[2]); }
		const T& operator()(ldp::UShort3 p)const { return (*this)(p[0], p[1], p[2]); }
		T& operator()(unsigned short  x, unsigned short  y, unsigned short  z)
		{
			return m_data[z*m_resolution[0] * m_resolution[1] + y*m_resolution[0] + x];
		}
		const T& operator()(unsigned short x, unsigned short  y, unsigned short  z)const
		{
			return m_data[z*m_resolution[0] * m_resolution[1] + y*m_resolution[0] + x];
		}

		T trilinear_at(ldp::Float3 p)const
		{
			ldp::Int3 p0 = p;
			ldp::Float3 d = p - ldp::Float3(p0);
			int inc[3] = { p0[0]<m_resolution[0] - 1, p0[1]<m_resolution[1] - 1, p0[2]<m_resolution[2] - 1 };

			const int y_stride = stride_Y();
			const int z_stride = stride_Z();

			const float* data_z0 = data() + z_stride * p0[2];
			const float* data_z1 = data_z0 + z_stride * inc[2];
			const float* data_z0y0 = data_z0 + y_stride * p0[1];
			const float* data_z0y1 = data_z0y0 + y_stride * inc[1];
			const float* data_z1y0 = data_z0y0 + z_stride * inc[2];
			const float* data_z1y1 = data_z1y0 + y_stride * inc[1];
			const int x0 = p0[0];
			const int x1 = x0 + inc[0];

			// ldp test: disable nans in trilinear interpolation
#if 0
			float v[8] = {data_z0y0[x0], data_z0y0[x1], data_z0y1[x0], data_z0y1[x1],
				data_z1y0[x0], data_z1y0[x1], data_z1y1[x0], data_z1y1[x1]};

			float w[8] = {(1 - d[0])*(1-d[1])*(1-d[2]),
				d[0]*(1-d[1])*(1-d[2]), 
				(1 - d[0])*d[1]*(1-d[2]), 
				d[0]*d[1]*(1-d[2]),
				(1 - d[0])*(1-d[1])*d[2], 
				d[0]*(1-d[1])*d[2], 
				(1 - d[0])*d[1]*d[2], 
				d[0]*d[1]*d[2]};

			float vwsum=0,wsum=0;
			for(int k=0; k<8; k++)
			if(!isnan(v[k]))
			{
				vwsum += v[k]*w[k];
				wsum += w[k];
			}
			return vwsum/wsum;
#else

			// first we interpolate along x-direction
			float c00 = data_z0y0[x0] * (1 - d[0]) + data_z0y0[x1] * d[0];
			float c10 = data_z0y1[x0] * (1 - d[0]) + data_z0y1[x1] * d[0];
			float c01 = data_z1y0[x0] * (1 - d[0]) + data_z1y0[x1] * d[0];
			float c11 = data_z1y1[x0] * (1 - d[0]) + data_z1y1[x1] * d[0];

			// then along y
			float c0 = c00 * (1 - d[1]) + c10 * d[1];
			float c1 = c01 * (1 - d[1]) + c11 * d[1];

			// finally along z
			float c = c0 * (1 - d[2]) + c1 * d[2];

			return c;
#endif
		}

		T *data(){ return m_data.data(); }
		const T *data()const{ return m_data.data(); }

		T *data_XYZ(ldp::UShort3 p){ return &(*this)(p); }
		const T *data_XYZ(ldp::UShort3 p)const{ return &(*this)(p); }
		T *data_XYZ(unsigned short x, unsigned short y, unsigned short z){ return &(*this)(x, y, z); }
		const T *data_XYZ(unsigned short x, unsigned short y, unsigned short z)const{ return &(*this)(x, y, z); }

		// fill in the patch with given data memory and given size/pos
		// dst must be of size patchSize*patchSize*patchSize
		template <int patchSize>
		void getPatch(T* dst, const ldp::UShort3& patch_x0y0z0)const
		{
			const T* src = data_XYZ(patch_x0y0z0);
			const int stride_z = stride_Z();
			const int stride_y = stride_Y();
			for (int z = 0; z < patchSize; z++)
			{
				const T* src_y = src;
				for (int y = 0; y < patchSize; y++)
				{
					for (int x = 0; x < patchSize; x++)
						*dst++ = src_y[x];
					src_y += stride_y;
				}
				src += stride_z;
			}
		}
		void getPatch(T* dst, const ldp::UShort3& patch_x0y0z0, int patchSize)const
		{
			switch (patchSize)
			{
			default:
				break;
			case 1:
				getPatch<1>(dst, patch_x0y0z0);
				break;
			case 2:
				getPatch<2>(dst, patch_x0y0z0);
				break;
			case 3:
				getPatch<3>(dst, patch_x0y0z0);
				break;
			case 4:
				getPatch<4>(dst, patch_x0y0z0);
				break;
			case 5:
				getPatch<5>(dst, patch_x0y0z0);
				break;
			case 6:
				getPatch<6>(dst, patch_x0y0z0);
				break;
			case 7:
				getPatch<7>(dst, patch_x0y0z0);
				break;
			case 8:
				getPatch<8>(dst, patch_x0y0z0);
				break;
			case 9:
				getPatch<9>(dst, patch_x0y0z0);
				break;
			case 10:
				getPatch<10>(dst, patch_x0y0z0);
				break;
			case 11:
				getPatch<11>(dst, patch_x0y0z0);
				break;
			}
		}
		void setPatch(const T* src, const ldp::UShort3& patch_x0y0z0, int patchSize)
		{
			T* dst = data_XYZ(patch_x0y0z0);
			const int stride_z = stride_Z();
			const int stride_y = stride_Y();
			for (int z = 0; z < patchSize; z++)
			{
				T* dst_y = dst;
				for (int y = 0; y < patchSize; y++)
				{
					for (int x = 0; x < patchSize; x++)
						dst_y[x] = *src++;
					dst_y += stride_y;
				}
				dst += stride_z;
			}
		}
		void fillPatch(T v, const ldp::UShort3& patch_x0y0z0, int patchSize)
		{
			T* dst = data_XYZ(patch_x0y0z0);
			const int stride_z = stride_Z();
			const int stride_y = stride_Y();
			for (int z = 0; z < patchSize; z++)
			{
				T* dst_y = dst;
				for (int y = 0; y < patchSize; y++)
				{
					for (int x = 0; x < patchSize; x++)
						dst_y[x] = v;
					dst_y += stride_y;
				}
				dst += stride_z;
			}
		}

		/// convolutions
		void convolve_max(int radius)
		{
			switch (radius)
			{
			default:
				throw std::exception("non-supported convolve-max radius!");
				break;
			case 0:
				return;
			case 1:
				return conv_helper::max_filter3<T, 3>(data(), m_resolution);
			case 2:
				return conv_helper::max_filter3<T, 5>(data(), m_resolution);
			case 3:
				return conv_helper::max_filter3<T, 7>(data(), m_resolution);
			case 4:
				return conv_helper::max_filter3<T, 9>(data(), m_resolution);
			case 5:
				return conv_helper::max_filter3<T, 11>(data(), m_resolution);
			case 6:
				return conv_helper::max_filter3<T, 13>(data(), m_resolution);
			case 7:
				return conv_helper::max_filter3<T, 15>(data(), m_resolution);
			case 8:
				return conv_helper::max_filter3<T, 17>(data(), m_resolution);
			case 9:
				return conv_helper::max_filter3<T, 19>(data(), m_resolution);
			case 10:
				return conv_helper::max_filter3<T, 21>(data(), m_resolution);
			}
		}
		void convolve_min(int radius)
		{
			switch (radius)
			{
			default:
				throw std::exception("non-supported convolve-max radius!");
				break;
			case 0:
				return;
			case 1:
				return conv_helper::min_filter3<T, 3>(data(), m_resolution);
			case 2:
				return conv_helper::min_filter3<T, 5>(data(), m_resolution);
			case 3:
				return conv_helper::min_filter3<T, 7>(data(), m_resolution);
			case 4:
				return conv_helper::min_filter3<T, 9>(data(), m_resolution);
			case 5:
				return conv_helper::min_filter3<T, 11>(data(), m_resolution);
			case 6:
				return conv_helper::min_filter3<T, 13>(data(), m_resolution);
			case 7:
				return conv_helper::min_filter3<T, 15>(data(), m_resolution);
			case 8:
				return conv_helper::min_filter3<T, 17>(data(), m_resolution);
			case 9:
				return conv_helper::min_filter3<T, 19>(data(), m_resolution);
			}
		}

		// the same with matlab conv(...,'same')
		void convolve(const float* kernel, int kernelSize)
		{
			switch (kernelSize)
			{
			default:
				throw std::exception("non-supported convolve kernelsize!");
				break;
			case 1:
				return conv_helper::conv3<T, 1>(data(), kernel, m_resolution);
			case 2:										
				return conv_helper::conv3<T, 2>(data(), kernel, m_resolution);
			case 3:										
				return conv_helper::conv3<T, 3>(data(), kernel, m_resolution);
			case 4:										
				return conv_helper::conv3<T, 4>(data(), kernel, m_resolution);
			case 5:									
				return conv_helper::conv3<T, 5>(data(), kernel, m_resolution);
			case 6:										
				return conv_helper::conv3<T, 6>(data(), kernel, m_resolution);
			case 7:										
				return conv_helper::conv3<T, 7>(data(), kernel, m_resolution);
			case 8:										
				return conv_helper::conv3<T, 8>(data(), kernel, m_resolution);
			case 9:										
				return conv_helper::conv3<T, 9>(data(), kernel, m_resolution);
			case 10:
				return conv_helper::conv3<T, 10>(data(), kernel, m_resolution);
			case 11:
				return conv_helper::conv3<T, 11>(data(), kernel, m_resolution);
			case 12:
				return conv_helper::conv3<T, 12>(data(), kernel, m_resolution);
			case 13:
				return conv_helper::conv3<T, 13>(data(), kernel, m_resolution);
			}
		}

		void mirrorExtendTo(VolumeTemplate<T>& rhs, int radius)const
		{
			rhs.resize(m_resolution[0] + radius * 2, m_resolution[1] + radius * 2, m_resolution[2] + radius * 2);
			for (int z = 0; z < rhs.m_resolution[2]; z++)
			{
				int src_z = abs(z - radius);
				if (src_z >= m_resolution[2])
					src_z = std::max(0, 2 * (int)m_resolution[2] - 2 - src_z);
				const T* src_z_ptr = stride_Z() * src_z + data();
				T* dst_z_ptr = rhs.stride_Z() * z + rhs.data();
				for (int y = 0; y < rhs.m_resolution[1]; y++)
				{
					int src_y = abs(y - radius);
					if (src_y >= m_resolution[1])
						src_y = std::max(0, 2 * (int)m_resolution[1] - 2 - src_y);
					const T* src_y_ptr = src_z_ptr + stride_Y() * src_y;
					T* dst_y_ptr = dst_z_ptr + rhs.stride_Y() * y;
					for (int x = 0; x < rhs.m_resolution[0]; x++)
					{
						int src_x = abs(x - radius);
						if (src_x >= m_resolution[0])
							src_x = std::max(0, 2 * (int)m_resolution[0] - 2 - src_x);
						dst_y_ptr[x] = src_y_ptr[src_x];
					}
				}// y
			}// z
		}

		void subVolumeTo(VolumeTemplate<T>& rhs, ldp::Int3 begin, ldp::Int3 end)const
		{
			for (int k = 0; k < 3; k++)
			{
				begin[k] = std::max(begin[k], 0);
				end[k] = std::min(end[k], (int)m_resolution[k]);
				if (end[k] <= begin[k])
				{
					rhs.clear();
					return;
				}
			}
			rhs.resize(ldp::Int3(end)-ldp::Int3(begin));
			for (int z = 0; z < rhs.m_resolution[2]; z++)
			{
				const T* src_z_ptr = stride_Z() * (z+begin[2]) + data();
				T* dst_z_ptr = rhs.stride_Z() * z + rhs.data();
				for (int y = 0; y < rhs.m_resolution[1]; y++)
				{
					const T* src_y_ptr = stride_Y() * (y + begin[1]) + src_z_ptr + begin[0];
					T* dst_y_ptr = rhs.stride_Y() * y + dst_z_ptr;
					for (int x = 0; x < rhs.m_resolution[0]; x++)
						dst_y_ptr[x] = src_y_ptr[x];
				}// y
			}// z
		}

		void subVolumeFrom(VolumeTemplate<T>& rhs, ldp::Int3 begin, ldp::Int3 end)
		{
			for (int k = 0; k < 3; k++)
			{
				begin[k] = std::max(begin[k], 0);
				end[k] = std::min(end[k], (int)m_resolution[k]);
				if (end[k] <= begin[k])
					return;
			}
			for (int z = 0; z < rhs.m_resolution[2]; z++)
			{
				T* dst_z_ptr = stride_Z() * (z + begin[2]) + data();
				const T* src_z_ptr = rhs.stride_Z() * z + rhs.data();
				for (int y = 0; y < rhs.m_resolution[1]; y++)
				{
					T* dst_y_ptr = stride_Y() * (y + begin[1]) + dst_z_ptr + begin[0];
					const T* src_y_ptr = rhs.stride_Y() * y + src_z_ptr;
					for (int x = 0; x < rhs.m_resolution[0]; x++)
						dst_y_ptr[x] = src_y_ptr[x];
				}// y
			}// z
		}

		virtual void save(const char* filename)const
		{
			FILE* pFile = fopen(filename, "wb");
			if (!pFile)
				throw std::exception(("error saving: " + std::string(filename)).c_str());
			save(pFile);
			fclose(pFile);
		}

		virtual void save(FILE* pFile)const
		{
			fwrite(&m_resolution, sizeof(m_resolution), 1, pFile);
			fwrite(data(), sizeof(T), m_resolution[0] * m_resolution[1] * m_resolution[2], pFile);
			printf("save volume, size %d, %d, %d\n",
				m_resolution[0], m_resolution[1], m_resolution[2]);
		}

		virtual void load(const char* filename)
		{
			FILE* pFile = fopen(filename, "rb");
			if (!pFile)
				throw std::exception(("error loading: " + std::string(filename)).c_str());
			load(pFile);
			fclose(pFile);
		}

		virtual void load(FILE* pFile)
		{
			fread(&m_resolution, sizeof(m_resolution), 1, pFile);
			resize(m_resolution);
			fread(data(), sizeof(T), m_resolution[0] * m_resolution[1] * m_resolution[2], pFile);
			printf("read volume, size %d, %d, %d\n",
				m_resolution[0], m_resolution[1], m_resolution[2]);
		}
	protected:
		ldp::UShort3 m_resolution;
		std::vector<T> m_data;
	};

	typedef VolumeTemplate<unsigned char> VolumeMask;

	class VolumeData : public VolumeTemplate<float>
	{
	public:
		enum VolumeType
		{
			VolumeTypeMpu = 0,
			VolumeTypeKinect
		};
		// not supported for volume larger than 2048*2948*1024
		struct CompactHalf
		{
			uint32_t idx;
			half_float::half val;
		};
	public:
		VolumeData();
		~VolumeData();

		//=================================================================
		void clear();
		void resize(unsigned short x, unsigned short y, unsigned short z){ VolumeTemplate<float>::resize(x,y,z); }
		void resize(ldp::UShort3 p){ VolumeTemplate<float>::resize(p); }
		void resize(ldp::UShort3 p, float voxelSize);
		void setBound(kdtree::AABB box){ m_boundingBox = box; }
		VolumeType getVolumeType()const{ return m_volumeType; }
		void setVolumeType(VolumeType t){ m_volumeType = t; }
		void subVolumeTo(VolumeData& rhs, ldp::Int3 begin, ldp::Int3 end, bool updateBound = true)const;
		void mirrorExtendTo(VolumeData& rhs, int radius)const;

		/**
		* IO
		* */

		// automatically load/save different file types
		// for kinect_volume or kvol, the default second parameter=0 is used
		void save(const char* filename)const;
		void save(FILE* pFile)const;
		void load(const char* filename);
		void load(FILE* pFile);

		// ".dvol"
		void saveFloat(const char* filename)const;
		void saveFloat(FILE* pFile)const;
		void loadFloat(const char* filename);
		void loadFloat(FILE* pFile);

		// ".hfdvol"
		void saveHalf(const char* filename)const;
		void saveHalf(FILE* pFile)const;
		void loadHalf(const char* filename);
		void loadHalf(FILE* pFile);

		// ".cvol"
		void saveCompactHalf(const char* filename)const;
		void saveCompactHalf(FILE* pFile)const;
		void loadCompactHalf(const char* filename);
		void loadCompactHalf(FILE* pFile);

		// ".kvol"
		void loadKinectFusionData(const char* filename, int weightThreshold=0);
		void loadKinectFusionData(FILE* pFile, int weightThreshold=0);

		// ".kinect_volume"
		void loadKinectFusionData_SDK(const char* filename, int weightThreshold=0);
		void loadKinectFusionData_SDK(FILE* pFile, int weightThreshold=0);
		void saveKinectFusionData_SDK(const char* filename)const;
		void saveKinectFusionData_SDK(FILE* pFile)const;

		//=================================================================
		/// data access methods
		float getVoxelSize()const{ return m_voxelSize; }
		kdtree::AABB getBound()const{ return m_boundingBox; }

		ldp::Float3 getVolumeIndexFromWorldPos(ldp::Float3 p)const
		{
			return (p-m_boundingBox.min)/getVoxelSize();
		}
		ldp::Float3 getWorldPosFromVolumeIndex(ldp::Float3 idx)const
		{
			return idx * getVoxelSize() + m_boundingBox.min;
		}

		// clip the values to the given region
		void clipValues(float minVal, float maxVal);

		float getValueScale()const;

		// currently supported exts:
		//	".dvol": float valued
		//  ".hfdvol": half-float valued
		//  ".cvol": half-float valued, compacted <idx, value> without saving the maximum.
		//  ".kinect_volume:" short-valued = weight+value
		static std::vector<std::string> getSupportedVolumeExts();
		static bool isSupportedVolumeExt(const char* filename);
	private:
		float m_voxelSize;
		kdtree::AABB m_boundingBox;
		VolumeType m_volumeType;
	};


	/// bwdist: the same with matlab's
	//		using the linear-time Euclidean distance transform method
	void bwdist(const VolumeMask& mask, VolumeData& distMap);
}