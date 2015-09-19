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
		GpuMesh();
		GpuMesh(GpuMesh& rhs);
		~GpuMesh();

		// n: number of vertices
		void create(size_t n);
		void release();
		void copyFrom(GpuMesh& rhs);

		void lockVertsNormals();
		void unlockVertsNormals();
		float3* verts(){ return m_verts_d; }
		const float3* verts()const{ return m_verts_d; }
		float3* normals(){ return m_normals_d; }
		const float3* normals()const { return m_normals_d; }

		int num()const{ return m_num; }
		void toObjMesh(ObjMesh& omesh);

		// unlockVertsNormals is performed insided
		void renderToImg(const Camera& camera, LightSource light, ColorMap& img);
	protected:
		void createRenderer(int w, int h);
		void releaseRenderer();
		void copy_invert_y(const uchar4* gldata, ColorMap& img);
	private:
		float3* m_verts_d;
		float3* m_normals_d;
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