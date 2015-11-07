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
	class WarpField;
	struct WarpNode;
	class Param;
	class GpuMesh
	{
	public:
		typedef float4 PointType;	
		__device__ __host__ __forceinline__ static float3 from_point(GpuMesh::PointType p)
		{
			return make_float3(p.x, p.y, p.z);
		}
		__device__ __host__ __forceinline__ static GpuMesh::PointType to_point(float3 p, float w=1.f)
		{
			GpuMesh::PointType o;
			o.x = p.x;
			o.y = p.y;
			o.z = p.z;
			o.w = w;
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
		PointType* colors(){ return m_colors_d; }
		const PointType* colors()const { return m_colors_d; }
		void setShowColor(bool c) { m_show_color = c; }

		int num()const{ return m_num; }
		void toObjMesh(ObjMesh& omesh);

		// unlockVertsNormals is performed insided
		void renderToImg(const Camera& camera, LightSource light, ColorMap& img, 
			const Param& param, const WarpField* warpField = nullptr,
			const MapArr* vmap_live = nullptr, const MapArr* vmap_warp = nullptr,
			const MapArr* nmap_live = nullptr, const MapArr* nmap_warp = nullptr,
			GpuMesh* canoMesh = nullptr,
			const float3* canoPosActive = nullptr,
			const KnnIdx* knnIdxActiveView = nullptr,
			const Intr* intr = nullptr, bool warp_nodes = true, bool no_rigid = false);
		void renderToDepth(const Camera& camera, DepthMap& img);

		// we assume self is warped by the warpField,
		// then the mesh is rendered with each pixel storing the canonical mesh verts/normals
		// given by canoMesh
		void renderToCanonicalMaps(const Camera& camera, 
			GpuMesh* canoMesh, DeviceArray2D<float4>& vmap, 
			DeviceArray2D<float4>& nmap);
	protected:
		void createRendererForWarpField(const WarpField* warpField);
		void releaseRendererForWarpField();
		void createRenderer(int w, int h);
		void releaseRenderer();
		void copy_invert_y(const float4* gldata, ColorMap& img);
		void copy_gldepth_to_depthmap(const float4* gldata, DepthMap& depth, 
			float s1, float s2, float camNear);
		void copy_canoview(const float4* gldata, DeviceArray2D<float4>& map);
		void copy_warp_node_to_gl_buffer(float4* gldata, const WarpField* warpField, 
			bool warp, bool single_level_graph);
		void copy_maps_to_gl_buffer(const MapArr& vmap_live, const MapArr& vmap_warp,
			const MapArr& nmap_live, const MapArr& nmap_warp,
			float4* gldata, const Param& param, Intr intr);
		void update_color_buffer_by_warpField(const WarpField* warpField, GpuMesh* canoMesh);
	private:
		PointType* m_verts_d;
		PointType* m_normals_d;
		PointType* m_colors_d;
		int m_num;
		int m_width;
		int m_height;
		int m_current_buffer_size;
		bool m_show_color;

		unsigned int m_vbo_id;
		cudaGraphicsResource* m_cuda_res;

		unsigned int m_render_fbo_id;
		unsigned int m_render_texture_id;
		unsigned int m_render_depth_id;
		unsigned int m_render_fbo_pbo_id;
		cudaGraphicsResource* m_cuda_res_fbo;

		unsigned int m_vbo_id_warpnodes;
		cudaGraphicsResource* m_cuda_res_warp;
	};
}