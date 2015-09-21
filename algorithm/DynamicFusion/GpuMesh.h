#pragma once

#include "definations.h"

class ObjMesh;
class Camera;
namespace dfusion
{
	/** ************************************************************************
	* Mesh Stored on the GPU
	*	verts:		xyzxyzxyz...
	*	normals:	xyzxyzxyz...
	* NOTE1: buffers are allocated in openGL vbo
	*		to use buffers from cuda, call lockVertsNormals();
	*		after that, call unlockVertsNormals();
	* *************************************************************************/
	class GpuMesh
	{
	public:
		typedef float3 PointType;	
		__device__ __host__ __forceinline__ static float3 from_point(GpuMesh::PointType p)
		{
			return make_float3(p.x, p.y, p.z);
		}
		__device__ __host__ __forceinline__ static GpuMesh::PointType to_point(float3 p)
		{
			GpuMesh::PointType o;
			o.x = p.x;
			o.y = p.y;
			o.z = p.z;
			return o;
		}
	public:
		GpuMesh();
		GpuMesh(GpuMesh& rhs);
		~GpuMesh();

		// n: number of vertices
		void create(size_t n);
		void release();
		void copyFrom(GpuMesh& rhs);

		void lockVertsNormals();
		void unlockVertsNormals();
		PointType* verts(){ return m_verts_d; }
		const PointType* verts()const{ return m_verts_d; }
		PointType* normals(){ return m_normals_d; }
		const PointType* normals()const { return m_normals_d; }

		int num()const{ return m_num; }
		void toObjMesh(ObjMesh& omesh);

		// unlockVertsNormals is performed insided
		void renderToImg(const Camera& camera, LightSource light, ColorMap& img);
		void renderToDepth(const Camera& camera, DepthMap& img);
	protected:
		void createRenderer(int w, int h);
		void releaseRenderer();
		void copy_invert_y(const uchar4* gldata, ColorMap& img);
		void copy_gldepth_to_depthmap(const uchar4* gldata, DepthMap& depth, 
			float s1, float s2, float camNear);
	private:
		PointType* m_verts_d;
		PointType* m_normals_d;
		int m_num;
		int m_width;
		int m_height;

		unsigned int m_vbo_id;
		cudaGraphicsResource* m_cuda_res;

		unsigned int m_render_fbo_id;
		unsigned int m_render_texture_id;
		unsigned int m_render_depth_id;
		unsigned int m_render_fbo_pbo_id;
		cudaGraphicsResource* m_cuda_res_fbo;
	};
}