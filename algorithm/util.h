#pragma once

#include <math.h>
#include "ldp_basic_mat.h"
#include <vector>
#include "eigen\Dense"
#include "eigen\SVD"
#include "eigen\Sparse"

class ObjMesh;
namespace ldp
{
	typedef double real;
	typedef Eigen::Matrix<real, -1, -1> Mat;
	typedef Eigen::Matrix<real, -1, 1> Vec;
	typedef Eigen::Matrix<float, -1, -1> Matf;
	typedef Eigen::Matrix<float, -1, 1> Vecf;
	typedef Eigen::SparseMatrix<real> SpMat;
	typedef Eigen::SparseMatrix<float> SpMatf;
	typedef Eigen::SparseVector<real> SpVec;
	typedef Eigen::SparseVector<float> SpVecf;
	typedef Eigen::LDLT<Mat> Solver;
	typedef Eigen::LDLT<Matf> Solverf;

	template<class T>
	class aligned_allocator
	{
	public:
		typedef size_t    size_type;
		typedef std::ptrdiff_t difference_type;
		typedef T*        pointer;
		typedef const T*  const_pointer;
		typedef T&        reference;
		typedef const T&  const_reference;
		typedef T         value_type;

		template<class U>
		struct rebind
		{
			typedef aligned_allocator<U> other;
		};

		pointer address(reference value) const
		{
			return &value;
		}

		const_pointer address(const_reference value) const
		{
			return &value;
		}

		aligned_allocator()
		{
		}

		aligned_allocator(const aligned_allocator&)
		{
		}

		template<class U>
		aligned_allocator(const aligned_allocator<U>&)
		{
		}

		~aligned_allocator()
		{
		}

		size_type max_size() const
		{
			return (std::numeric_limits<size_type>::max)();
		}

		pointer allocate(size_type num, const void* hint = 0)
		{
			if (num > size_t(-1) / sizeof(T))
				throw std::bad_alloc();
			return static_cast<pointer>(_aligned_malloc(num * sizeof(T), 16));
		}

		void construct(pointer p, const T& value)
		{
			::new(p)T(value);
		}

		void destroy(pointer p)
		{
			p->~T();
		}

		void deallocate(pointer p, size_type /*num*/)
		{
			_aligned_free(p);
		}

		bool operator!=(const aligned_allocator<T>&) const
		{
			return false;
		}

		bool operator==(const aligned_allocator<T>&) const
		{
			return true;
		}
	};

	struct MeshSelection
	{
		Eigen::aligned_allocator<float>a;
		int face_id;
		float w[4];
		ldp::Float3 tarPos;
		MeshSelection()
		{
			face_id = -1;
			w[0] = w[1] = w[2] = w[3] = 0;
			tarPos = 0;
		}
		MeshSelection(int fid, ldp::Float4 wts)
		{
			face_id = fid;
			for (int i = 0; i < 4; i++)
				w[i] = wts[i];
		}
		MeshSelection(const MeshSelection& r)
		{
			face_id = r.face_id;
			for (int i = 0; i < 4; i++)
				w[i] = r.w[i];
			tarPos = r.tarPos;
		}

		ldp::Float3 getPos(const ObjMesh& mesh)const;
		ldp::Float3 getPos(const ObjMesh& mesh, std::vector<ldp::Float3>& verts)const;
		void updatePos(const ObjMesh& mesh, const ldp::Float3& pos);
	};

	inline std::string fullfile(std::string path, std::string name)
	{
		if (path == "")
			return name;
		if (path.back() != '/' && path.back() != '\\')
			path.append("/");
		if (path != "" && name.size())
		{
			if (name[0] == '/' || name[0] == '\\')
				name = name.substr(1, name.size()-1);
		}
		return path + name;
	}

	inline std::string validWindowsPath(std::string oldpath)
	{
		for (int i = 0; i < oldpath.size(); i++)
		if (oldpath[i] == '/')
			oldpath[i] = '\\';
		return oldpath;
	}

	inline bool directoryExists(std::string path)
	{
		DWORD dwAttrib = GetFileAttributesA(path.c_str());

		return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
			(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
	}

	inline void mkdir(std::string path)
	{
		if (directoryExists(path))
			return;
		char a[1000];
		sprintf(a, "mkdir %s", validWindowsPath(path).c_str());
		system(a);
	}

	inline void fileparts(std::string fullfile, std::string& path, std::string& name, std::string& ext)
	{
		int pos = fullfile.find_last_of('/');
		if (pos >= fullfile.size())
			pos = fullfile.find_last_of('\\');
		if (pos >= fullfile.size())
		{
			path = "";
		}
		else
		{
			path = fullfile.substr(0, pos + 1);
			fullfile = fullfile.substr(pos + 1, fullfile.size());
		}


		int pos1 = fullfile.find_last_of('.');
		if (pos1 >= fullfile.size())
		{
			name = fullfile;
		}
		else
		{
			name = fullfile.substr(0, pos1);
			ext = fullfile.substr(pos1, fullfile.size());
		}
	}

	inline bool file_exist(const char* path)
	{
		FILE* pFile = fopen(path, "r");
		if (!pFile)
			return false;
		else
		{
			fclose(pFile);
			return true;
		}
	}

	// sample input:
	//	buffer = "label: abc"
	// sample output:
	//	buffer = "abc"
	//	return "label"
	inline std::string getLineLabel(std::string& buffer, char sep = ':')
	{
		std::string s;
		int pos = buffer.find_first_of(sep);
		if (pos < buffer.size())
		{
			s = buffer.substr(0, pos);
			buffer = buffer.substr(pos + 2); // ignore a space after :
		}
		return s;
	}

	// ext: E.G., "obj", ".off", "*", ".*"
	bool getAllFilesInDir(const std::string& path,
		std::vector<std::string>& names, std::string ext);

	inline int PointInPolygon(int nvert, const float *vertx, const float *verty, float testx, float testy)
	{
		int i, j, c = 0;
		for (i = 0, j = nvert - 1; i < nvert; j = i++) {
			if (((verty[i]>testy) != (verty[j]>testy)) &&
				(testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]))
				c = !c;
		}
		return c;
	}
	inline int PointInPolygon(int nvert, const ldp::Float2* verts, ldp::Float2 p)
	{
		int i, j, c = 0;
		for (i = 0, j = nvert - 1; i < nvert; j = i++) {
			if (((verts[i][1]>p[1]) != (verts[j][1]>p[1])) &&
				(p[0] < (verts[j][0] - verts[i][0]) * (p[1] - verts[i][1]) / 
				(verts[j][1] - verts[i][1]) + verts[i][1]))
				c = !c;
		}
		return c;
	}
	//--- solve for: Ax^2 + Bx + C = 0
	inline void QuadraticSolve(float A, float B, float C, float x[2])
	{
		if (abs(A) < abs(B) * 1e-8)
		{
			x[0] = x[1] = -C / B;
			return;
		}

		float delta = B*B - 4 * A*C;
		if (delta < 0)
			delta = 0;
		delta = sqrt(delta);

		//for numerical stability
		if (B >= 0)
		{
			x[0] = 2 * C / (-B - delta);
			x[1] = (-B - delta) / (2 * A);
		}
		else
		{
			x[0] = (-B + delta) / (2 * A);
			x[1] = (2 * C) / (-B + delta);
		}
	}

	//--- calculate the new coordinate inside the grid
	//--- get the solution (rx,ry) of the
	//the order of the 4 vertices should be
	//--- 0 2
	//    1 3
	//--- equation (1-rx)(1-ry)*(x1,y1) + (1-rx)ry*(x2,y2) + rx(1-ry)*(x3,y3) + rxry*(x4,y4) = (ax,ay)
	inline void GetInnerCoordinate(float &rx, float &ry, const float ax, const float ay,
		const float x1, const float y1, const float x2, const float y2,
		const float x3, const float y3, const float x4, const float y4)
	{
		// axy + bx + cy + d = 0
		float a1 = (x1 - x2 - x3 + x4); float b1 = (-x1 + x3); float c1 = (-x1 + x2); float d1 = (x1 - ax);
		float a2 = (y1 - y2 - y3 + y4); float b2 = (-y1 + y3); float c2 = (-y1 + y2); float d2 = (y1 - ay);

		float RY[2], RX[2];

		//solve for Ay*y + By + C = 0;
		float A = a1*c2 - a2*c1;
		float B = -a2*d1 + a1*d2 + b1*c2 - b2*c1;
		float C = b1*d2 - b2*d1;
		QuadraticSolve(A, B, C, RY);

		//x = -(c*y+d)/(a*y+b)
		for (int i = 0; i<2; i++)
		{
			float ayb1 = a1*RY[i] + b1;
			float ayb2 = a2*RY[i] + b2;
			if (fabs(ayb1)>fabs(ayb2))
				RX[i] = -(c1*RY[i] + d1) / ayb1;
			else
				RX[i] = -(c2*RY[i] + d2) / ayb2;
		}

		if (RX[0] * RX[0] + RY[0] * RY[0] < RX[1] * RX[1] + RY[1] * RY[1])
		{
			rx = RX[0];
			ry = RY[0];
		}
		else
		{
			rx = RX[1];
			ry = RY[1];
		}
	}


	//the order of the 4 vertices should be
	//0 3
	//1 2
	inline void CalcBilinearCoef(Float4& w, Float2 p, const Float2 v[4])
	{
		float rx, ry;
		GetInnerCoordinate(rx, ry, p[0], p[1], v[0][0], v[0][1], v[1][0], v[1][1], v[3][0], v[3][1], v[2][0], v[2][1]);
		w[0] = (1 - rx)*(1 - ry);
		w[1] = (1 - rx)*ry;
		w[3] = rx*(1 - ry);
		w[2] = rx*ry;
	}

	//the order of the 4 vertices should be
	//0 3
	//1 2
	inline Float2 BilinearInterpolate(Float4 w, const Float2 v[4])
	{
		return w[0] * v[0] + w[1] * v[1] + w[2] * v[2] + w[3] * v[3];
	}

	// w: the triangle coordinates of the projected points
	// return the projected points
	inline Float3 PointProjectToTri(ldp::Float3 p, ldp::Float3 v[3], ldp::Float3& w)
	{
		const int id_pe[3] = { 2, 0, 1 };
		const int id_ne[3] = { 1, 2, 0 };

		// projected to 2d
		ldp::Float3 Z = ldp::Float3(v[1] - v[0]).cross(v[2] - v[0]);
		ldp::Float3 X = v[1] - v[0];
		ldp::Float3 Y = Z.cross(X);
		ldp::Float2 v2[6];
		for (int k = 0; k < 3; k++)
			v2[k] = ldp::Float2((v[k] - v[0]).dot(X), (v[k] - v[0]).dot(Y));
		ldp::Float2 p2((p - v[0]).dot(X), (p - v[0]).dot(Y));

		// calc triangle coordinates
		float areaTotal = ldp::Float2(v2[1] - v2[0]).cross(v2[2] - v2[0]);
		for (int i = 0; i < 3; i++)
			w[i] = ldp::Float2(p2 - v2[id_ne[i]]).cross(p2 - v2[id_pe[i]]) / areaTotal;

		if (w[0] >= 0 && w[1] >= 0 && w[2] >= 0)
			return w[0] * v[0] + w[1] * v[1] + w[2] * v[2];

		// if not legal, then find the min-dist point inside.
		ldp::Float2 e[3];
		ldp::Float2 cand[6];
		int pos = 3;
		for (int i = 0; i < 3; i++)
		{
			e[i] = v2[id_ne[i]] - v2[i];
			ldp::Float2 pei = p2 - v2[i];
			float ti = e[i].dot(pei) / sqrt(e[i].sqrLength()*pei.sqrLength());
			if (ti >= 0 && ti < 1 && w[id_pe[i]]>=0)
				cand[pos++] = v2[i] + ti * e[i];
		}

		float minDist = FLT_MAX;
		int minId = -1;
		for (int k = 0; k < pos; k++)
		{
			float dist = sqrt((p2 - v2[k]).sqrLength());
			if (dist < minDist)
			{
				minDist = dist;
				minId = k;
			}
		}
		p2 = v2[minId];

		//recalc the coefs
		for (int i = 0; i < 3; i++)
			w[i] = ldp::Float2(p2 - v2[id_ne[i]]).cross(p2 - v2[id_pe[i]]) / areaTotal;	

		return w[0] * v[0] + w[1] * v[1] + w[2] * v[2];
	}

	inline ldp::Float3 calcTemperatureJet(float val)
	{
#undef min
#undef max

		const static float w[9][6] = 
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

		val = 8.f * std::min(1.f, std::max(0.f, val));
		const float* const c = w[int(val)];
		return ldp::Float3(c[0]*val+c[1], c[2]*val+c[3], c[4]*val+c[5]);
	}

	// kmeans algorithm:
	//	Data: data matrix, each column coresponding to a data point.
	// if useRandInit == fales
	//	then a best initialization will be guessed.
	void kmeans(const Mat& Data, int k, std::vector<int>& dataClusterId, 
		std::vector<Vec>& dataCenters,
		int nMaxIter=10, bool useRandInit = false, bool showInfo=false);


	class TimeStamp
	{
	public:
		explicit TimeStamp()
		{
			QueryPerformanceFrequency(&m_iFrequency);
			pFile = nullptr;
			Reset();
		}

		bool logToFile(const char* filename)
		{
			if (pFile)
				fclose(pFile);
			pFile = fopen(filename, "w");
			if (!pFile)
				return false;
			return true;
		}

		void flushLog()
		{
			if (pFile)
				fflush(pFile);
		}

		void closeLog()
		{
			if (pFile)
				fclose(pFile);
		}

		void Reset()
		{
			QueryPerformanceCounter(&m_iLast);
			m_iStart = m_iLast;
			m_szPrefix = "";
			if (pFile)
				fclose(pFile);
			pFile = nullptr;
		}

		void Prefix(const char* szPrefix)
		{
			m_szPrefix = szPrefix;
		}

		void Stamp(const char* szFormat, ...)
		{
			va_list args;
			va_start(args, szFormat);
			vsprintf_s(m_szMessage, szFormat, args);
			va_end(args);

			LARGE_INTEGER iNow = { 0 };
			QueryPerformanceCounter(&iNow);

			const float flDeltaLast = float(
				iNow.QuadPart - m_iLast.QuadPart) / m_iFrequency.QuadPart;
			const float flDeltaStart = float(
				iNow.QuadPart - m_iStart.QuadPart) / m_iFrequency.QuadPart;

			printf("%s[D=%.5fs, S=%.5fs] %s\n",
				m_szPrefix.c_str(), flDeltaLast, flDeltaStart, m_szMessage);

			if (pFile)
				fprintf(pFile, "%s[D=%.5fs, S=%.5fs] %s\n",
					m_szPrefix.c_str(), flDeltaLast, flDeltaStart, m_szMessage);

			m_iLast = iNow;
		}
	protected:
		LARGE_INTEGER m_iStart;
		LARGE_INTEGER m_iLast;
		LARGE_INTEGER m_iFrequency;
		std::string m_szPrefix;
		char m_szMessage[1024];
		FILE* pFile;
	};
}