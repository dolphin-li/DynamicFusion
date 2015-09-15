#include "VolumeData.h"
#include <iostream>
#include <fstream>
#include "ObjMesh.h"
#include <queue>
#include <stack>
namespace mpu
{
	VolumeData::VolumeData()
	{
		clear();
	}

	VolumeData::~VolumeData()
	{
	}

	std::vector<std::string> VolumeData::getSupportedVolumeExts()
	{
		static const char* s_supported_volume_type[] =
		{
			".dvol",
			".kinect_volume",
			".hfdvol",
			".cvol"
		};
		static std::vector<std::string> exts(s_supported_volume_type, s_supported_volume_type + 4);
		return exts;
	}

	bool VolumeData::isSupportedVolumeExt(const char* filename)
	{
		const static int ns = 4;
		static const char* s_supported_volume_type[ns] =
		{
			".dvol",
			".kinect_volume",
			".hfdvol",
			".cvol"
		};

		std::string ext, dumy;
		fileparts(filename, dumy, dumy, ext);
		
		for (int i = 0; i < ns; i++)
		if (ext == s_supported_volume_type[i])
			return true;
		return false;
	}

	void VolumeData::clear()
	{
		m_data.clear();
		m_voxelSize = 0.f;
		m_resolution = 0;
		m_volumeType = VolumeTypeMpu;
	}

	void VolumeData::resize(ldp::UShort3 p, float voxelSize)
	{
		VolumeTemplate<float>::resize(p);
		m_voxelSize = voxelSize;
	}

	void VolumeData::save(const char* filename)const
	{
		std::string path, name, ext;
		fileparts(filename, path, name, ext);
		
		if (!isSupportedVolumeExt(ext.c_str()))
			throw std::exception(("not supported file extension: " + ext).c_str());

		FILE* pFile = fopen(filename, "wb");
		if (!pFile)
			throw std::exception((std::string("io error: saving file failed: ") + filename).c_str());

		if (ext == ".dvol")
			saveFloat(pFile);
		else if (ext == ".cvol")
			saveCompactHalf(pFile);
		else if (ext == ".hfdvol")
			saveHalf(pFile);
		else if (ext == ".kinect_volume")
			saveKinectFusionData_SDK(pFile);
		else if (ext == ".kvol")
			throw std::exception(("not supported saving format: " + ext).c_str());

		fclose(pFile);
	}

	void VolumeData::save(FILE* pFile)const
	{
		throw std::exception("not allowed calling");
	}

	void VolumeData::load(const char* filename)
	{
		std::string path, name, ext;
		fileparts(filename, path, name, ext);

		if (!isSupportedVolumeExt(ext.c_str()))
			throw std::exception(("not supported file extension: " + ext).c_str());

		FILE* pFile = fopen(filename, "rb");
		if (!pFile)
			throw std::exception((std::string("io error: saving file failed: ") + filename).c_str());

		if (ext == ".dvol")
			loadFloat(pFile);
		else if (ext == ".cvol")
			loadCompactHalf(pFile);
		else if (ext == ".hfdvol")
			loadHalf(pFile);
		else if (ext == ".kinect_volume")
			loadKinectFusionData_SDK(pFile);
		else if (ext == ".kvol")
			loadKinectFusionData(pFile);

		fclose(pFile);
	}

	void VolumeData::load(FILE* pFile)
	{
		throw std::exception("not allowed calling");
	}

	void VolumeData::saveFloat(const char* filename)const
	{
		FILE* pFile = fopen(filename, "wb");
		if (!pFile)
			throw std::exception((std::string("io error: saving file failed: ") + filename).c_str());

		saveFloat(pFile);
	}

	void VolumeData::saveFloat(FILE* pFile)const
	{
		fwrite(&m_resolution, sizeof(m_resolution), 1, pFile);
		fwrite(&m_voxelSize, sizeof(m_voxelSize), 1, pFile);
		fwrite(&m_boundingBox, sizeof(m_boundingBox), 1, pFile);
		fwrite(data(), sizeof(float), m_resolution[0] * m_resolution[1] * m_resolution[2], pFile);
		fwrite(&m_volumeType, sizeof(m_volumeType), 1, pFile);
		//printf("save volume, type %d, size %d, %d, %d\n", m_volumeType,
		//	m_resolution[0], m_resolution[1], m_resolution[2]);
	}

	void VolumeData::loadFloat(const char* filename)
	{
		FILE* pFile = fopen(filename, "rb");
		if (!pFile)
			throw std::exception((std::string("io error: loading file failed: ") + filename).c_str());

		loadFloat(pFile);

		fclose(pFile);
	}

	void VolumeData::loadFloat(FILE* pFile)
	{
		clear();

		fread(&m_resolution, sizeof(m_resolution), 1, pFile);
		VolumeTemplate<float>::resize(m_resolution);
		fread(&m_voxelSize, sizeof(m_voxelSize), 1, pFile);
		fread(&m_boundingBox, sizeof(m_boundingBox), 1, pFile);
		fread((char*)data(), sizeof(float), m_resolution[0] * m_resolution[1] * m_resolution[2], pFile);

		if (fread(&m_volumeType, sizeof(m_volumeType), 1, pFile) != 1)
		{
			printf("warning: no volume type provided, using the default: MPU\n");
			m_volumeType = VolumeTypeMpu;
		}

		printf("load volume, type %d, size %d, %d, %d, voxel size: %f\n", m_volumeType,
			m_resolution[0], m_resolution[1], m_resolution[2], m_voxelSize);
		printf("bounding box: %f %f %f, %f %f %f\n", m_boundingBox.min[0],
			m_boundingBox.min[1], m_boundingBox.min[2], m_boundingBox.max[0], 
			m_boundingBox.max[1], m_boundingBox.max[2]);
	}

	void VolumeData::saveHalf(const char* filename)const
	{
		FILE* pFile = fopen(filename, "wb");
		if (!pFile)
			throw std::exception((std::string("io error: saving file failed: ") + filename).c_str());

		saveHalf(pFile);

		fclose(pFile);
	}

	void VolumeData::saveHalf(FILE* pFile)const
	{
		fwrite(&m_resolution, sizeof(m_resolution), 1, pFile);
		fwrite(&m_voxelSize, sizeof(m_voxelSize), 1, pFile);
		fwrite(&m_boundingBox, sizeof(m_boundingBox), 1, pFile);
		std::vector<half_float::half> half_data(m_data.size());
		for (size_t i = 0; i < half_data.size(); i++)
			half_data[i] = m_data[i];
		fwrite(half_data.data(), sizeof(half_float::half), half_data.size(), pFile);
		fwrite(&m_volumeType, sizeof(m_volumeType), 1, pFile);
		//printf("save half volume, type %d, size %d, %d, %d\n", m_volumeType,
		//	m_resolution[0], m_resolution[1], m_resolution[2]);
	}

	void VolumeData::loadHalf(const char* filename)
	{
		FILE* pFile = fopen(filename, "rb");
		if (!pFile)
			throw std::exception((std::string("io error: loading file failed: ") + filename).c_str());

		loadHalf(pFile);

		fclose(pFile);
	}

	void VolumeData::loadHalf(FILE* pFile)
	{
		clear();

		fread(&m_resolution, sizeof(m_resolution), 1, pFile);
		VolumeTemplate<float>::resize(m_resolution);
		fread(&m_voxelSize, sizeof(m_voxelSize), 1, pFile);
		fread(&m_boundingBox, sizeof(m_boundingBox), 1, pFile);

		std::vector<half_float::half> half_data(m_data.size());
		int num = fread((char*)half_data.data(), sizeof(half_float::half), half_data.size(), pFile);
		if (num != half_data.size())
			throw std::exception(("data corruption: " + std::to_string(num) + " vs. "
			+ std::to_string(half_data.size())).c_str());
		for (size_t i = 0; i < half_data.size(); i++)
			m_data[i] = half_data[i];

		if (fread(&m_volumeType, sizeof(m_volumeType), 1, pFile) != 1)
		{
			printf("warning: no volume type provided, using the default: MPU\n");
			m_volumeType = VolumeTypeMpu;
		}

		printf("load half volume, type %d, size %d, %d, %d, voxel size: %f\n", m_volumeType,
			m_resolution[0], m_resolution[1], m_resolution[2], m_voxelSize);
		printf("bounding box: %f %f %f, %f %f %f\n", m_boundingBox.min[0],
			m_boundingBox.min[1], m_boundingBox.min[2], m_boundingBox.max[0],
			m_boundingBox.max[1], m_boundingBox.max[2]);
	}

	void VolumeData::saveCompactHalf(const char* filename)const
	{
		if (m_data.size() > UINT_MAX)
			throw std::exception("too large volume to be saved compactly!");

		FILE* pFile = fopen(filename, "wb");
		if (!pFile)
			throw std::exception((std::string("io error: saving file failed: ") + filename).c_str());

		saveCompactHalf(pFile);

		fclose(pFile);
	}

	void VolumeData::saveCompactHalf(FILE* pFile)const
	{
		fwrite(&m_resolution, sizeof(m_resolution), 1, pFile);
		fwrite(&m_voxelSize, sizeof(m_voxelSize), 1, pFile);
		fwrite(&m_boundingBox, sizeof(m_boundingBox), 1, pFile);

		float maxVal = -1e10f;
		for (int i = 0; i < m_data.size(); i++)
			maxVal = std::max(maxVal, m_data[i]);

		std::vector<CompactHalf> cdata;

		for (size_t i = 0; i < m_data.size(); i++)
		{
			float v = m_data[i];
			if (v < maxVal - m_voxelSize*0.1f)
			{
				CompactHalf ch;
				ch.idx = i;
				ch.val = v;
				cdata.push_back(ch);
			}
		}
		fwrite(&maxVal, sizeof(float), 1, pFile);
		int n = cdata.size();
		fwrite(&n, sizeof(int), 1, pFile);
		fwrite(cdata.data(), sizeof(CompactHalf), cdata.size(), pFile);
		fwrite(&m_volumeType, sizeof(m_volumeType), 1, pFile);
	}

	void VolumeData::loadCompactHalf(const char* filename)
	{
		FILE* pFile = fopen(filename, "rb");
		if (!pFile)
			throw std::exception((std::string("io error: loading file failed: ") + filename).c_str());

		loadCompactHalf(pFile);

		fclose(pFile);
	}

	void VolumeData::loadCompactHalf(FILE* pFile)
	{
		clear();

		fread(&m_resolution, sizeof(m_resolution), 1, pFile);
		VolumeTemplate<float>::resize(m_resolution);
		fread(&m_voxelSize, sizeof(m_voxelSize), 1, pFile);
		fread(&m_boundingBox, sizeof(m_boundingBox), 1, pFile);

		float maxVal = 0.f;
		fread(&maxVal, sizeof(float), 1, pFile);

		fill(maxVal);

		int n = 0;
		fread(&n, sizeof(int), 1, pFile);
		std::vector<CompactHalf> cdata(n);
		int num = fread((char*)cdata.data(), sizeof(CompactHalf), cdata.size(), pFile);
		if (num != cdata.size())
			throw std::exception(("data corruption: " + std::to_string(num) + " vs. "
			+ std::to_string(cdata.size())).c_str());

		for (size_t i = 0; i < cdata.size(); i++)
			m_data[cdata[i].idx] = cdata[i].val;

		if (fread(&m_volumeType, sizeof(m_volumeType), 1, pFile) != 1)
		{
			printf("warning: no volume type provided, using the default: MPU\n");
			m_volumeType = VolumeTypeMpu;
		}

		printf("load compact half volume, type %d, size %d, %d, %d, voxel size: %f\n", m_volumeType,
			m_resolution[0], m_resolution[1], m_resolution[2], m_voxelSize);
		printf("bounding box: %f %f %f, %f %f %f\n", m_boundingBox.min[0],
			m_boundingBox.min[1], m_boundingBox.min[2], m_boundingBox.max[0],
			m_boundingBox.max[1], m_boundingBox.max[2]);
	}

	void VolumeData::loadKinectFusionData(const char* filename, int weightThreshold)
	{
		FILE* pFile = fopen(filename, "rb");
		if (!pFile)
			throw std::exception((std::string("io error: loading file failed: ") + filename).c_str());

		loadKinectFusionData(pFile, weightThreshold);

		fclose(pFile);
	}
	
	void VolumeData::loadKinectFusionData(FILE* pFile, int weightThreshold)
	{
		clear();

		float voxelPerMeter;
		int nX, nY, nZ;
		fread(&m_voxelSize, sizeof(float), 1, pFile);
		fread(&nX, sizeof(int), 1, pFile);
		fread(&nY, sizeof(int), 1, pFile);
		fread(&nZ, sizeof(int), 1, pFile);
		std::vector<ldp::Short2> rawData;
		rawData.resize(nX*nY*nZ);

		if (nX*nY*nZ != fread(rawData.data(), sizeof(ldp::Short2), rawData.size(), pFile))
			throw std::exception("file corrupted");
		printf("kinect volume loaded: %d %d %d, %f\n", nX, nY, nZ, m_voxelSize);

		m_volumeType = VolumeTypeKinect;
		m_resolution = ldp::UShort3(nX, nY, nZ);
		VolumeTemplate<float>::resize(m_resolution);
		m_boundingBox.min = -m_voxelSize * ldp::Float3(m_resolution) / 2.f;
		m_boundingBox.max = m_voxelSize * ldp::Float3(m_resolution) / 2.f;

		const ldp::Short2* rawPtr = rawData.data();
		float* vPtr = data();
		for (int z = 0; z < m_resolution[2]; z++)
		{
			for (int y = 0; y < m_resolution[1]; y++)
			{
				for (int x = 0; x < m_resolution[0]; x++)
				{
					ldp::Short2 rawVal = *rawPtr++;
					float& vVar = *vPtr++;
					float dist = float(rawVal[0]) / float(MAXSHORT);

					if (rawVal[1] <= weightThreshold)
					{
						vVar = std::numeric_limits<float>::quiet_NaN();
						continue;
					}

					vVar = dist;
				}//x
			}//y
		}//z
	}

	void VolumeData::loadKinectFusionData_SDK(const char* filename, int weightThreshold)
	{
		FILE* pFile = fopen(filename, "rb");
		if (!pFile)
			throw std::exception((std::string("io error: loading file failed: ") + filename).c_str());

		loadKinectFusionData_SDK(pFile, weightThreshold);

		fclose(pFile);
	}

	void VolumeData::loadKinectFusionData_SDK(FILE* pFile, int weightThreshold)
	{
		clear();

		int nX, nY, nZ;
		fread(&m_voxelSize, sizeof(float), 1, pFile);
		m_voxelSize = 1.f / m_voxelSize;
		fread(&nX, sizeof(int), 1, pFile);
		fread(&nY, sizeof(int), 1, pFile);
		fread(&nZ, sizeof(int), 1, pFile);
		std::vector<short> rawData;
		rawData.resize(nX*nY*nZ);

		if (nX*nY*nZ != fread(rawData.data(), sizeof(short), rawData.size(), pFile))
			throw std::exception("file corrupted");
		printf("kinect volume loaded: %d %d %d, %f\n", nX, nY, nZ, m_voxelSize);

		m_volumeType = VolumeTypeKinect;
		m_resolution = ldp::UShort3(nX, nY, nZ);
		VolumeTemplate<float>::resize(m_resolution);
		m_boundingBox.min = -m_voxelSize * ldp::Float3(m_resolution) / 2.f;
		m_boundingBox.max = m_voxelSize * ldp::Float3(m_resolution) / 2.f;
		m_boundingBox.max[2] = -0.35f;
		m_boundingBox.min[2] = m_boundingBox.max[2] - float(m_resolution[2]) * m_voxelSize;

		const short* rawPtr = rawData.data();
		for (int z = 0; z < m_resolution[2]; z++)
		for (int y = 0; y < m_resolution[1]; y++)
		{
			float* vPtr = data_XYZ(0, m_resolution[1]-1-y, m_resolution[2]-1-z);
			for (int x = 0; x < m_resolution[0]; x++)
			{
				short rawVal = *rawPtr++;

				float& vVar = *vPtr++;
				int wRaw = (rawVal & 0xff);
				int depthRaw = -char((rawVal & 0xff00) >> 8);

				if (wRaw == weightThreshold)
				{
					vVar = std::numeric_limits<float>::quiet_NaN();
					continue;
				}

				vVar = std::min(1.f, std::max(-1.f, float(depthRaw) / float(0x7f)));
			}// x
		}//y,z
	}

	void VolumeData::saveKinectFusionData_SDK(const char* filename)const
	{
		FILE* pFile = fopen(filename, "wb");
		if (!pFile)
			throw std::exception((std::string("io error: creating file failed: ") + filename).c_str());

		saveKinectFusionData_SDK(pFile);

		fclose(pFile);
	}

	void VolumeData::saveKinectFusionData_SDK(FILE* pFile)const
	{
		float voxelPerMeter = 1.f / m_voxelSize;
		fwrite(&voxelPerMeter, sizeof(float), 1, pFile);
		ldp::Int3 res = m_resolution;
		fwrite(&res[0], sizeof(int), 1, pFile);
		fwrite(&res[1], sizeof(int), 1, pFile);
		fwrite(&res[2], sizeof(int), 1, pFile);
		std::vector<short> rawData;
		rawData.resize(m_resolution[0] * m_resolution[1]*m_resolution[2]);

		short* rawPtr = rawData.data();
		for (int z = 0; z < m_resolution[2]; z++)
		for (int y = 0; y < m_resolution[1]; y++)
		{
			const float* vPtr = data_XYZ(0, m_resolution[1] - 1 - y, m_resolution[2] - 1 - z);
			for (int x = 0; x < m_resolution[0]; x++)
			{
				short& rawVal = *rawPtr++;
				float vVar = std::min(1.f, std::max(-1.f, *vPtr++));

				if (std::isnan(vVar))
				{
					rawVal = 0x8000;
				}
				else
				{
					int depthRaw = -int(vVar * 0x7f);
					rawVal = ((depthRaw << 8) & 0xff00) + 0x7f;
				}
			}// x
		}//y,z
		fwrite(rawData.data(), sizeof(short), rawData.size(), pFile);
	}

	void VolumeData::clipValues(float minVal, float maxVal)
	{
		for (int i = 0; i < m_data.size(); i++)
			m_data[i] = std::max(minVal, std::min(maxVal, m_data[i]));
	}

	float VolumeData::getValueScale()const
	{
		if (m_data.size() == 0)
			return 0.f;
		float minVal = 1e10;
		float maxVal = -1e10;
		for (int i = 0; i < m_data.size(); i++)
		{
			float v = m_data[i];
			minVal = std::min(minVal, v);
			maxVal = std::max(maxVal, v);
		}
		return abs(maxVal - minVal);
	}

	void VolumeData::subVolumeTo(VolumeData& rhs, ldp::Int3 begin, ldp::Int3 end, bool updateBound)const
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
		VolumeTemplate<float>::subVolumeTo(rhs, begin, end);

		if (updateBound)
		{
			rhs.m_voxelSize = m_voxelSize;
			rhs.m_boundingBox.min = m_boundingBox.min + ldp::Float3(begin) * m_voxelSize;
			rhs.m_boundingBox.max = m_boundingBox.min + ldp::Float3(end) * m_voxelSize;
		}
		rhs.m_volumeType = m_volumeType;
	}

	void VolumeData::mirrorExtendTo(VolumeData& rhs, int radius)const
	{
		VolumeTemplate<float>::mirrorExtendTo(rhs, radius);

		rhs.m_voxelSize = m_voxelSize;
		rhs.m_boundingBox.min -= radius * m_voxelSize;
		rhs.m_boundingBox.max += radius * m_voxelSize;
		rhs.m_volumeType = m_volumeType;
	}

	//==========================================================================
	/// bwdist: the same with matlab's
	//		using the linear-time Euclidean distance transform method

	////////// Functions F and Sep for the SDT labelling
	inline float bwdist_sqr(float u)
	{
		return u*u;
	}
	inline int bwdist_F(float u, float i, float gi2)
	{
		return (u - i)*(u - i) + gi2;
	}
	inline int bwdist_Sep(float i, float u, float gi2, float gu2)
	{
		return (u*u - i*i + gu2 - gi2) / (2 * (u - i));
	}
	/////////

	void bwdist(const VolumeMask& mask, VolumeData& distMap)
	{
		VolumeData tmpXVolume, tmpXYVolume;
		const ldp::Int3 resolution = mask.getResolution();
		const int inf = resolution[0] + resolution[1] + resolution[2];

		bool distMapPreReady = (distMap.getResolution() == mask.getResolution());

		// phase x-----------------------------------------------------
		tmpXVolume.resize(mask.getResolution());

#pragma omp parallel for
		for (int z = 0; z < resolution[2]; z++)
		for (int y = 0; y < resolution[1]; y++)
		{
			if (mask(0, y, z) == 0)
				tmpXVolume(0, y, z) = distMapPreReady ? distMap(0, y, z) : 0;
			else
				tmpXVolume(0, y, z) = inf;

			// Forward scan
			for (int x = 1; x < resolution[0]; x++)
			{
				if (mask(x, y, z) == 0)
					tmpXVolume(x, y, z) = distMapPreReady ? distMap(0, y, z) : 0;
				else
					tmpXVolume(x, y, z) = 1 + tmpXVolume(x - 1, y, z);
			}

			//Backward scan
			for (int x = resolution[0] - 2; x >= 0; x--)
			if (tmpXVolume(x + 1, y, z) < tmpXVolume(x, y, z))
				tmpXVolume(x, y, z) = 1 + tmpXVolume(x + 1, y, z);
		}// end for y,z

		// phase y-----------------------------------------------------
		tmpXYVolume.resize(mask.getResolution());
#pragma omp parallel for
		for (int z = 0; z < resolution[2]; z++)
		{
			std::vector<float> s(resolution[1]), t(resolution[1]);
			for (int x = 0; x < resolution[0]; x++)
			{
				float q = 0, w = 0;
				s[0] = 0;
				t[0] = 0;

				//Forward Scan
				for (int u = 1; u < resolution[1]; u++)
				{
					while (q >= 0 && (bwdist_F(t[q], s[q], bwdist_sqr(tmpXVolume(x, s[q], z))) >
						bwdist_F(t[q], u, bwdist_sqr(tmpXVolume(x, u, z)))))
						q--;

					if (q < 0)
					{
						q = 0;
						s[0] = u;
					}
					else
					{
						w = 1 + bwdist_Sep(s[q], u, bwdist_sqr(tmpXVolume(x, s[q], z)),
							bwdist_sqr(tmpXVolume(x, u, z)));
						if (w < resolution[1])
						{
							q++;
							s[q] = u;
							t[q] = w;
						}
					}
				}

				//Backward Scan
				for (int u = resolution[1] - 1; u >= 0; --u)
				{
					tmpXYVolume(x, u, z) = bwdist_F(u, s[q], bwdist_sqr(tmpXVolume(x, s[q], z)));
					if (u == t[q])
						q--;
				}
			}// end for x
		}// end for z

		// phase z-----------------------------------------------------
		tmpXVolume.clear();
		distMap.resize(resolution);
		distMap.fill(0);
#pragma omp parallel for
		for (int y = 0; y < resolution[1]; y++)
		{
			std::vector<float> s(resolution[2]), t(resolution[2]);
			for (int x = 0; x < resolution[0]; x++)
			{
				float q = 0, w = 0;
				s[0] = 0;
				t[0] = 0;

				//Forward Scan
				for (int u = 1; u < resolution[2]; u++)
				{
					while (q >= 0 && (bwdist_F(t[q], s[q], tmpXYVolume(x, y, s[q])) >
						bwdist_F(t[q], u, tmpXYVolume(x, y, u))))
						q--;

					if (q < 0)
					{
						q = 0;
						s[0] = u;
					}
					else
					{
						w = 1 + bwdist_Sep(s[q], u, tmpXYVolume(x, y, s[q]), tmpXYVolume(x, y, u));
						if (w < resolution[2])
						{
							q++;
							s[q] = u;
							t[q] = w;
						}
					}
				}

				//Backward Scan
				for (int u = resolution[2] - 1; u >= 0; --u)
				{
					distMap(x, y, u) = sqrtf((float)bwdist_F(u, s[q], tmpXYVolume(x, y, s[q])));
					if (u == t[q])
						q--;
				}
			}// end for x
		}// end for y
	}

}// namespace mpu