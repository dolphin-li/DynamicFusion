#ifndef __LDP_BMESH_H__
#define __LDP_BMESH_H__
/**
 *  THIS IS A C++ WRAPPER OF BLENDER BMESH DATA STRUCTURE.
 *  BY: DONGPING LI
 *	\file blender/bmesh/bmesh.h
 *  \ingroup bmesh
 *
 * \addtogroup bmesh BMesh
 *
 * \brief BMesh is a non-manifold boundary representation designed to replace the current, limited EditMesh structure,
 * solving many of the design limitations and maintenance issues of EditMesh.
 */
#include <vector>
#include "Renderable.h"
namespace ldp
{

class BMesh;
class BMVert;
class BMEdge;
class BMLoop;
class BMFace;
class BMIter;
struct BLI_mempool_chunk;
struct BLI_mempool_iter;
struct BLI_mempool;
enum 
{
	BM_VERT = 1,
	BM_EDGE = 2,
	BM_LOOP = 4,
	BM_FACE = 8
};

class BMesh : public Renderable
{
public:
	BMesh();
	BMesh(BMesh& other);
	~BMesh();
	BMesh& operator=(BMesh& rhs);
	/**
	* init from triangle meshes
	* @nverts
	* @verts: xyz-xyz-..., 3 * nverts
	* @nfaces
	* @faces: 012-012-..., 3 * nfaces
	* */
	void init_triangles(int nverts, float* verts, int nfaces, int *faces);
	/**
	* init from quad meshes
	* @nverts
	* @verts: xyz-xyz-..., 3 * nverts
	* @nfaces
	* @faces: 0123-0123-..., 4 * nfaces
	* */
	void init_quads(int nverts, float* verts, int nfaces, int *faces);
	/**
	* init from general polygon meshes
	* @nverts
	* @verts: xyz-xyz-..., 3 * nverts
	* @nfaces
	* @faces
	* @faceHeaders:
	*	index of begin of each polygon face. SIZE = nfaces+1
	* E.G.:
	*	faceaHeaders:	0,    3,         9
	*	faces:			0,1,2,0,2,3,4,9,7
	*	means 2 faces: (0,1,2), (0,2,3,4,9,7)
	* */
	void init_polygons(int nverts, float* verts, int nfaces, int *faces, int* faceHeaders);

	/**
	* init from a group of edges
	* @nverts
	* @verts: xyz-xyz-..., 3 * nverts
	* @nedges
	* @edges: ab-cd-ef-..., 2 * nedges
	*/
	void init_edges(int nverts, float* verts, int nedges, int *edges);

	/**
	* copy from other BMesh
	* */
	void init_bmesh(BMesh *bm_old);

	/**
	* set color material for rendering faces
	* */
	void set_default_face_color(ldp::Float3 color);

	/**
	* verts: 3*number, x.y.z, x.y.z
	* faces: 3*number, 0.1.2, 0.1.2
	* */
	void to_triangles(std::vector<float>& verts, std::vector<int>& faces);

	virtual void clear();

	bool save(const char* filename);
	bool load(const char* filename);
	/**
	* Render with opengl
	* @showType: same with in Renderable.h
	* */
	virtual void render(int showType, int frameIdx=0);
	virtual void renderConstColor(Float3 color)const{}
	virtual int getMeshType()const{ return TYPE_BMESH; }
	void set_render_point_size(float s){ m_render_point_size = s; }
	float get_render_point_size()const{ return m_render_point_size; }
	void set_render_line_width(float s){ m_render_line_width = s; }
	float get_render_line_width()const{ return m_render_line_width; }
	/**
	* Called in init_*(), you can call it as well.
	* */
	void updateNormal();

	/**
	* After per-element operation, the index may be dirty. So it should be updated before other index-related operation
	* */
	void updateIndex(int type = BM_VERT | BM_EDGE | BM_FACE);
	/**
	* find single source shortest path, Dijkstra method.
	* if disableIntersect==true, then the selected vertices will be ignored for new-path.
	* if useTopologyDist==true, then number of edges will be used instead of geometry-dist.
	* */
	bool find_shortest_path(BMVert* vbegin, BMVert** vends, int nvends, 
		std::vector<std::vector<BMVert*>> &paths, bool disableIntersect, bool useTopologyDist=false);
	/**
	* find single source , single destination shortest path, Dijkstra method.
	* if useTopologyDist==true, then number of edges will be used instead of geometry-dist.
	* NOTE: This is efficient than single-source, multi-destinations algorithm only if the path
	*		to be found is relatively short.
	* */
	bool find_shortest_path(BMVert* vbegin, BMVert* vend, std::vector<BMVert*> &paths, bool useTopologyDist=false);
	void select_shortest_path(BMVert* vbegin, BMVert* vend, bool disableIntersect, bool useTopologyDist=false);

	/**
	* Calculate Single-Source-All-Dst geodesic distances via Dijktra
	* if useTopologyDist, then all edge will be viewed as unit length
	* @paths_by_parent:
	*	the paths of the all found geodesic from dsts to vbegin
	*	To use it to find the path from vertex vdst to vbegin:
	*		int parent = vdst->getIndex();
	*		while(parent >= 0)
	*		{
	*			// do what you want with parent, the index
	*			parent = paths_by_parent[parent];
	*		}
	*		// finally, parent should be vbegin->getIndex()
	* @ vert_block_paths:
	*	a set of vertex ids, each act as a wall that blocks the geodesic paths
	* */
	bool calc_single_src_all_dst_geodesic(BMVert* vbegin, std::vector<float>& dists,
		std::vector<int>* paths_by_parent = 0,
		const std::vector<int>& vert_block_paths = std::vector<int>(),
		bool useTopologyDist = false);

	/**
	* find all verices that has a path linked to the src vertex
	* bSelect: 
	*	=true, do selection
	*	=false, remove selection
	* if useSelectedAsBarrier == true, then the selected vertices will block the further selection
	* if vmask != 0, then only those vmask[v] != 0 will be considered during searching
	* */
	bool find_linked_verts(BMVert* vsrc, std::vector<BMVert*> &linked, bool bSelect, 
		bool useSelectedAsBarrier, int* vmask = nullptr);
	void select_linked_verts(BMVert* vbegin, bool bSelect, bool useSelectedAsBarrier, int* vmask = nullptr);

	/**
	*
	* */
	BMVert* add_vert(float co[3]);
	BMEdge* add_edge(BMVert* v1, BMVert* v2);
	BMFace* add_face(BMVert** verts, int len);
	/**
	*
	* */
	void	split_edge(BMEdge* e);
	/**
	* Split input face into 2 faces according to given vertices
	* NOTE: 1. v1 and v2 must belong to f, or NULL will be returned.
	*		2. if v1 and v2 belong to the same edge, the a face with only 2 vertices will be created.
	* Return: 2 faces created, one pointed by f and one by return.
	* */
	BMFace*	split_faces(BMFace* f, BMVert* v1, BMVert* v2);
	/**
	* v1 -> v2
	* */
	void	merge_verts(BMVert* v1, BMVert* v2);
	/**
	* merge faces to a new face and return it.
	* */
	BMFace*	merge_faces(BMFace* f1, BMFace* f2);
	/**
	* f: input polygon face, newfaces: triangulated new faces, output, pass NULL if you don't want it.
	* */
	void	triangulate_face(BMFace *f, BMFace** newfaces);
	void	triangulate_selected();
	/**
	* remove given vertex and all edges/faces using it
	* */
	void	remove_vert(BMVert* v);
	/**
	* remove given edge and all faces using it
	* */
	void	remove_edge(BMEdge* e);
	/**
	* remove given face
	* */
	void	remove_face(BMFace* f);
	/**
	* remove given face, if all faces surrounding a vert are removed, then is the vert.
	* */
	void	remove_face_vert(BMFace* f);
	/**
	* type = BM_VERT or BM_FACE or BM_EDGE
	* */
	void	remove_all(int type);
	void	remove_selected(int type);
	/**
	* select a vert, if 2-verts of a edge is selected, then is the edge; if all verts of a face is selected, then is the face
	* */
	void	select_vert(BMVert* v, bool sel);
	/**
	* select a edge
	* */
	void 	select_edge(BMEdge* e, bool sel);
	/**
	* select a face
	* */
	void	select_face(BMFace* f, bool sel);

	/**
	* select the 1-ring vertices of given vertex
	* */
	void	select_onering(BMVert* v, bool sel);
	/**
	* type = BM_VERT or BM_FACE or BMEDGE
	* */
	void	select_all(int type, bool sel);
	void	select_inverse(int type);
	/**`
	* Queries By Index, Convenient but QUITE UITE ITE TE E...SLOW.
	* Time complex: O(N/512);
	* */
	BMVert* vert_at(int idx);
	BMEdge* edge_at(int idx);
	BMFace* face_at(int idx);
	/**
	* Queries: Vert Of Mesh.
	* BMIter iter;for(BMVert* v = vofm_begin(iter); v != vofm_end(iter); v=vofm_next(iter)){...}
	* */
	BMVert* vofm_begin(BMIter& iter);
	BMVert* vofm_next(BMIter& iter);
	BMVert* vofm_end(BMIter& iter);
	const BMVert* vofm_begin(BMIter& iter)const;
	const BMVert* vofm_next(BMIter& iter)const;
	const BMVert* vofm_end(BMIter& iter)const;
	int vofm_count()const{return totvert;}
	/**
	* Queries: Edge Of Mesh.
	* BMIter iter;for(BMEdge* e = eofm_begin(iter); e != eofm_end(iter); e=eofm_next(iter)){...}
	* */
	BMEdge* eofm_begin(BMIter& iter);
	BMEdge* eofm_next(BMIter& iter);
	BMEdge* eofm_end(BMIter& iter);
	const BMEdge* eofm_begin(BMIter& iter)const ;
	const BMEdge* eofm_next(BMIter& iter)const ;
	const BMEdge* eofm_end(BMIter& iter)const ;
	int eofm_count()const{return totedge;}
	/**
	* Queries: Face Of Mesh
	* BMIter iter;for(BMFace* f = fofm_begin(iter); f != fofm_end(iter); f=fofm_next(iter)){...}
	* */
	BMFace* fofm_begin(BMIter& iter);
	BMFace* fofm_next(BMIter& iter);
	BMFace* fofm_end(BMIter& iter);
	const BMFace* fofm_begin(BMIter& iter)const;
	const BMFace* fofm_next(BMIter& iter)const;
	const BMFace* fofm_end(BMIter& iter)const;
	int fofm_count()const{return totface;}
	/**
	* Queries: Vert Of Edge, the 2 verts of a given edge
	* */
	static BMVert* vofe_first(const BMEdge* e);
	static BMVert* vofe_last(const BMEdge* e);
	static int vofe_count(const BMEdge* e);
	/**
	* Queries: Vert Of Face
	* Useage: BMIter iter;iter.init(f); for(BMVert* v = voff_begin(iter); v != voff_end(iter); v=voff_next(iter)){...}
	* */
	static BMVert* voff_begin(BMIter& iter);
	static BMVert* voff_next(BMIter& iter);
	static BMVert* voff_end(BMIter& iter);
	static int voff_count(const BMFace* f);
	/**
	* Queries: Edge Of Vertex
	* Useage: BMIter iter; iter.init(v); for(BMEdge* e = eofv_begin(iter); e != eofv_end(iter); e=eofv_next(iter)){...}
	* */
	static BMEdge* eofv_begin(BMIter& iter);
	static BMEdge* eofv_next(BMIter& iter);
	static BMEdge* eofv_end(BMIter& iter);
	static int eofv_count(BMVert* v);
	/**
	* Returns the edge existing between v1 and v2, or NULL if there isn't one.
	*
	* \note multiple edges may exist between any two vertices, and therefore
	* this function only returns the first one found.
	* */
	static BMEdge* eofv_2(BMVert* v1, BMVert* v2);
	/**
	* Queries: Edge Of Face
	* Useage: BMIter iter; iter.init(f); for(BMEdge* e = voff_begin(iter); e != voff_end(iter); e=voff_next(iter)){...}
	* */
	static BMEdge* eoff_begin(BMIter& iter);
	static BMEdge* eoff_next(BMIter& iter);
	static BMEdge* eoff_end(BMIter& iter);
	static int eoff_count(BMFace* f);
	/**
	* Queries: Face Of Vert
	* Useage: BMIter iter; iter.init(v); for(BMFace* f = fofv_begin(iter); f != fofv_end(iter); f= fofv_next(iter)){...}
	* */
	static BMFace* fofv_begin(BMIter& iter);
	static BMFace* fofv_next(BMIter& iter);
	static BMFace* fofv_end(BMIter& iter);
	static int fofv_count(BMVert* v);
	/**
	* Queries: Face Of Edge
	* Useage: BMIter iter; iter.init(e); for(BMFace* f = fofe_begin(iter); f != fofe_end(iter); f= fofe_next(iter)){...}
	* */
	static BMFace* fofe_begin(BMIter& iter);
	static BMFace* fofe_next(BMIter& iter);
	static BMFace* fofe_end(BMIter& iter);
	static int fofe_count(BMEdge* e);
	/**
	* Queries: Loop Of Vert
	* Useage: BMIter iter; iter.init(v); for(BMLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
	* */
	static BMLoop* lofv_begin(BMIter& iter);
	static BMLoop* lofv_next(BMIter& iter);
	static BMLoop* lofv_end(BMIter& iter);
	static int lofv_count(BMVert* l);
	/**
	* Queries: Loop Of Edge
	* Useage: BMIter iter; iter.init(e); for(BMLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
	* */
	static BMLoop* lofe_begin(BMIter& iter);
	static BMLoop* lofe_next(BMIter& iter);
	static BMLoop* lofe_end(BMIter& iter);
	static int lofe_count(BMEdge* e);
	/**
	* Queries: Loop Of Face
	* Useage: BMIter iter; iter.init(f); for(BMLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
	* */
	static BMLoop* loff_begin(BMIter& iter);
	static BMLoop* loff_next(BMIter& iter);
	static BMLoop* loff_end(BMIter& iter);
	static int loff_count(BMFace* f);
	/**
	* Queries: Loops Of Loop
	* Useage: BMIter iter; iter.init(l); for(BMLoop* l2 = lofv_begin(iter); l2 != lofv_end(iter); l2=lofv_next(iter)){...}
	* */
	static BMLoop* lofl_begin(BMIter& iter);
	static BMLoop* lofl_next(BMIter& iter);
	static BMLoop* lofl_end(BMIter& iter);
	static int lofl_count(BMLoop* l);
public:
	const static int BM_LOOP_RADIAL_MAX = 10000;
	void			initMempool(int tv, int te, int tl, int tf);
private:
	//in bmesh_construct.h
	BMFace*			BM_face_create_quad_tri				(BMVert *v1,		BMVert *v2,			BMVert *v3, BMVert *v4);
	BMFace*			BM_face_create_quad_tri_v			(BMVert**verts,		int len);
	//in bmesh_core.h
	BMVert*			BM_vert_create						(const float co[3]);
	BMEdge*			BM_edge_create						(BMVert* v1,		BMVert* v2);
	BMFace*			BM_face_create						(BMVert **verts,	BMEdge **edges,		const int len);
	BMFace*			bm_face_create_internal				();
	BMFace*			BM_face_create_ngon					(BMVert* v1,		BMVert* v2, BMEdge** edges, int len);
	BMLoop*			bm_loop_create						(BMVert *v,			BMEdge *e,			BMFace *f);
	BMLoop*			bm_face_boundary_add				(BMFace*f,			BMVert* startv,		BMEdge* starte);
	BMVert*			bmesh_semv							(BMVert *tv,		BMEdge *e,			BMEdge **r_e);
	BMEdge*			bmesh_jekv							(BMEdge *ke,		BMVert *kv,			const short check_edge_double);
	BMFace*			bmesh_sfme							(BMFace *f,			BMVert *v1,			BMVert *v2,BMLoop **r_l);
	BMFace*			bmesh_jfke							(BMFace *f1, BMFace *f2, BMEdge *e);
	BMFace*			BM_face_split						(BMFace *f, BMVert *v1, BMVert *v2, BMLoop **r_l);
public:
	static void		bm_kill_only_vert(BMesh *bm, BMVert *v);
	static void		bm_kill_only_edge(BMesh *bm, BMEdge *e);
	static void		bm_kill_only_face(BMesh *bm, BMFace *f);
	static void		bm_kill_only_loop(BMesh *bm, BMLoop *l);
	static void		BM_face_edges_kill(BMesh *bm, BMFace *f);
	static void		BM_face_verts_kill(BMesh *bm, BMFace *f);
	static void		BM_face_kill(BMesh *bm, BMFace *f);
	static void		BM_edge_kill(BMesh *bm, BMEdge *e);
	static void		BM_vert_kill(BMesh *bm, BMVert *v);
	static int		bmesh_edge_separate(BMesh *bm, BMEdge *e, BMLoop *l_sep);//not implement yet..
	static int		BM_edge_splice(BMesh *bm, BMEdge *e, BMEdge *etarget);
	static int		BM_vert_splice(BMesh *bm, BMVert *v, BMVert *vtarget);
	static int		bmesh_vert_separate(BMesh *bm, BMVert *v, BMVert ***r_vout, int *r_vout_len);//not implement yet..
	static int		bmesh_loop_reverse(BMesh *bm, BMFace *f);
	static BMFace *	BM_faces_join(BMesh *bm, BMFace **faces, int totface, const short do_del);
	int				BM_vert_separate(BMesh *bm, BMVert *v, BMVert ***r_vout, int *r_vout_len,BMEdge **e_in, int e_in_len);
	//in bmesh_struct.h
	static bool		bmesh_disk_edge_append				(BMEdge* e,			BMVert* v);
	static BMEdge*	bmesh_disk_edge_next				(const BMEdge *e,	const BMVert *v);
	static BMEdge*	bmesh_disk_edge_prev				(const BMEdge *e,	const BMVert *v);
	static BMEdge*	bmesh_disk_edge_exists				(const BMVert *v1,	const BMVert *v2);
	static void		bmesh_disk_edge_remove				(BMEdge *e,			BMVert *v);
	static int		bmesh_disk_facevert_count			(const BMVert *v);
	static BMEdge*	bmesh_disk_faceedge_find_first		(BMEdge *e,			const BMVert *v	);
	static BMEdge*	bmesh_disk_faceedge_find_next		(BMEdge *e,			const BMVert *v	);
	static int		bmesh_disk_validate					(int len,			const BMEdge *e,	const BMVert *v);
	static int		bmesh_disk_count					(const BMVert *v);
	static int		bmesh_radial_validate				(int radlen,		const BMLoop *l	);
	static void		bmesh_radial_append					(BMEdge* e,			BMLoop* l);
	static int		bmesh_radial_length					(const BMLoop *l);
	static void		bmesh_radial_loop_remove			(BMLoop *l,			BMEdge *e);
	static BMLoop*	bmesh_radial_faceloop_find_first	(BMLoop *l,			const BMVert *v);
	static BMLoop*	bmesh_radial_faceloop_find_next		(BMLoop *l,			const BMVert *v);
	static int		bmesh_radial_face_find				(const BMEdge *e,	const BMFace *f);
	static int		bmesh_radial_facevert_count			(const BMLoop *l,	const BMVert *v);
	static int		bmesh_vert_in_edge					(const BMEdge *e,	const BMVert *v);
	static int		bmesh_verts_in_edge					(const BMVert *v1,	const BMVert *v2,	const BMEdge *e);
	static BMVert*	bmesh_edge_other_vert_get			(BMEdge *e,			const BMVert *v);
	static int		bmesh_edge_swapverts				(BMEdge *e,			BMVert *orig,		BMVert *newv);
	static int		bmesh_loop_validate					(const BMFace *f);
	static int		bm_loop_length						(BMLoop* l);
	//in bmesh_queries.h
	static int		BM_vert_in_face(BMFace *f, BMVert *v);
	static int		BM_verts_in_face(BMFace *f, BMVert **varr, int len);
	static int		BM_edge_in_face(BMFace *f, BMEdge *e);
	static int		BM_vert_in_edge(const BMEdge *e, const BMVert *v);
	static int		BM_verts_in_edge(BMVert *v1, BMVert *v2, BMEdge *e);
	static float	BM_edge_calc_length(BMEdge *e);
	static int		BM_edge_face_pair(BMEdge *e, BMFace **r_fa, BMFace **r_fb);
	static int		BM_edge_loop_pair(BMEdge *e, BMLoop **r_la, BMLoop **r_lb);
	static BMVert *	BM_edge_other_vert(BMEdge *e, BMVert *v);
	static BMLoop *	BM_face_other_edge_loop(BMFace *f, BMEdge *e, BMVert *v);
	static BMLoop *	BM_face_other_vert_loop(BMFace *f, BMVert *v_prev, BMVert *v);
	static BMLoop *	BM_loop_other_vert_loop(BMLoop *l, BMVert *v);
	static int		BM_vert_edge_count_nonwire(BMVert *v);
	static int		BM_vert_edge_count(BMVert *v);
	static int		BM_edge_face_count(BMEdge *e);
	static int		BM_vert_face_count(BMVert *v);
	static BMEdge *	BM_vert_other_disk_edge(BMVert *v, BMEdge *e);
	static int		BM_vert_is_wire(BMVert *v);
	static int		BM_edge_is_wire(BMEdge *e);
	static int		BM_vert_is_manifold(BMVert *v);
	static int		BM_edge_is_manifold(BMEdge *e);
	static int		BM_edge_is_boundary(BMEdge *e);
	static float	BM_loop_calc_face_angle(BMLoop *l);
	static void		BM_loop_calc_face_normal(BMLoop *l, float r_normal[3]);
	static void		BM_loop_calc_face_tangent(BMLoop *l, float r_tangent[3]);
	static float	BM_edge_calc_face_angle(BMEdge *e);
	static void		BM_edge_calc_face_tangent(BMEdge *e, BMLoop *e_loop, float r_tangent[3]);
	static float	BM_vert_calc_edge_angle(BMVert *v);
	static float	BM_vert_calc_shell_factor(BMVert *v);
	static BMEdge *	BM_edge_exists(BMVert *v1, BMVert *v2);
	static int		BM_face_exists_overlap(BMVert **varr, int len, BMFace **r_existface);
	static int		BM_face_exists(BMVert **varr, int len, BMFace **r_existface);
	static int		BM_face_exists_multi(BMVert **varr, BMEdge **earr, int len);
	static int		BM_face_exists_multi_edge(BMEdge **earr, int len);
	static int		BM_face_share_edge_count(BMFace *f1, BMFace *f2);
	static int		BM_edge_share_face_count(BMEdge *e1, BMEdge *e2);
	static int		BM_edge_share_vert_count(BMEdge *e1, BMEdge *e2);
	static BMVert *	BM_edge_share_vert(BMEdge *e1, BMEdge *e2);
	static BMLoop *	BM_face_vert_share_loop(BMFace *f, BMVert *v);
	static BMLoop *	BM_face_edge_share_loop(BMFace *f, BMEdge *e);
	static void		BM_edge_ordered_verts(BMEdge *edge, BMVert **r_v1, BMVert **r_v2);
	static void		BM_edge_ordered_verts_ex(BMEdge *edge, BMVert **r_v1, BMVert **r_v2,
		BMLoop *edge_loop);
	//bmesh_polygon.h
	static void		bm_face_calc_poly_normal(BMFace *f);
	static void		bm_face_calc_poly_normal_vertex_cos(BMFace *f, float n[3], float const (*vertexCos)[3]);
	static float	BM_face_calc_area(BMFace *f);
	static float	BM_face_calc_perimeter(BMFace *f);
	static void		BM_face_calc_center_bounds(BMFace *f, float center[3]);
	static void		BM_face_calc_center_mean(BMFace *f, float center[3]);
	static void		BM_face_normal_update(BMFace *f);
	static void		BM_face_normal_update_vcos(BMesh *bm, BMFace *f, float no[3], float const (*vertexCos)[3]);
	static void		BM_edge_normals_update(BMEdge *e);
	static void		BM_vert_normal_update(BMVert *v);
	static void		BM_vert_normal_update_all(BMVert *v);
	static void		BM_face_normal_flip(BMesh *bm, BMFace *f);
	static int		BM_face_point_inside_test(BMFace *f, const float co[3]);
	static int		bm_face_goodline(float const (*projectverts)[3], BMFace *f,
						int v1i, int v2i, int v3i, float polyArea);
	static BMLoop *	find_ear(BMFace *f, float (*verts)[3], const int nvert, const int use_beauty, float polyArea);
private:	
	/*element pools*/
	BLI_mempool* vpool, *fpool, *epool, *lpool;
	int totvert, totedge, totloop, totface;
	ldp::Float3 m_default_face_color;

	float m_render_point_size;
	float m_render_line_width;
};

class BMHeader {
	friend class BMIter;
	friend class BMesh;
public:
	BMHeader():index(0),htype(0),select(false),flag(0){}
	int getIndex()const{return index;}
	void setIndex(int i){index=i;}
	bool isSelect()const{return select;}
	void setSelect(bool s){select=s;}
private:
	int index; 
	char htype; /* element geometric type (verts/edges/loops/faces) */
	bool select;
	char flag;
};

class BMVert {
	friend class BMIter;
	friend class BMesh;
public:
	BMVert():e(0){}
	int getIndex()const{return head.getIndex();}
	void setIndex(int i){head.setIndex(i);}
	bool isSelect()const{return head.isSelect();}
	void setSelect(bool s){head.setSelect(s);}
private:
	BMHeader head;
public:
	float co[3];
	float no[3];
	BMEdge *e;
};

class BMDiskLink {
	friend class BMIter;
	friend class BMesh;
public:
	BMDiskLink():next(0),prev(0){}
private:
	BMEdge *next, *prev;
};

class BMEdge {
	friend class BMIter;
	friend class BMesh;
public:
	BMEdge():v1(0),v2(0),l(0){}
	int getIndex()const{return head.getIndex();}
	void setIndex(int i){head.setIndex(i);}
	bool isSelect()const{return head.isSelect();}
	void setSelect(bool s){head.setSelect(s);}
private:
	BMHeader head;
	BMLoop *l;
	/* disk cycle pointers */
	BMDiskLink v1_disk_link, v2_disk_link;
	BMVert *v1, *v2;
};

class BMLoop {
	friend class BMIter;
	friend class BMesh;
public:
	BMLoop():v(0),e(0),f(0),radial_next(0),radial_prev(0),next(0),prev(0){}
	int getIndex()const{return head.getIndex();}
	void setIndex(int i){head.setIndex(i);}
	bool isSelect()const{return head.isSelect();}
	void setSelect(bool s){head.setSelect(s);}
private:
	BMHeader head;
	/* circular linked list of loops which all use the same edge as this one '->e',
	 * but not necessarily the same vertex (can be either v1 or v2 of our own '->e') */
	BMLoop *radial_next, *radial_prev;

	/* these were originally commented as private but are used all over the code */
	/* can't use ListBase API, due to head */
	BMLoop *next, *prev; /* next/prev verts around the face */
	BMVert *v;
	BMEdge *e; /* edge, using verts (v, next->v) */
	BMFace *f;
};

class BMFace {
	friend class BMIter;
	friend class BMesh;
public:
	BMFace():l_first(0),len(0),mat_nr(0){}
	int getIndex()const{return head.getIndex();}
	void setIndex(int i){head.setIndex(i);}
	bool isSelect()const{return head.isSelect();}
	void setSelect(bool s){head.setSelect(s);}
private:
	BMHeader head;
	int len; /*includes all boundary loops*/
	BMLoop *l_first;
public:
	float no[3]; /*yes, we do store this here*/
	short mat_nr;
};



#pragma region --macros
#define TIME_EVAL(func) {ldp::tic(); func; ldp::toc(#func);}

#define BMESH_ALL_FACES(f, iter, mesh)\
	BMIter iter; \
for (BMFace* f = (mesh).fofm_begin(iter); f != (mesh).fofm_end(iter); f = (mesh).fofm_next(iter))

#define BMESH_ALL_VERTS(v, iter, mesh)\
	BMIter iter; \
for (BMVert* v = (mesh).vofm_begin(iter); v != (mesh).vofm_end(iter); v = (mesh).vofm_next(iter))

#define BMESH_ALL_EDGES(e, iter, mesh)\
	BMIter iter; \
for (BMEdge* e = (mesh).eofm_begin(iter); e != (mesh).eofm_end(iter); e = (mesh).eofm_next(iter))

#define BMESH_V_OF_F(v, f, iter, mesh)\
	BMIter iter; \
	iter.init(f); \
for (BMVert* v = (mesh).voff_begin(iter); v != (mesh).voff_end(iter); v = (mesh).voff_next(iter))

#define BMESH_F_OF_V(f, v, iter, mesh)\
	BMIter iter; \
	iter.init(v); \
for (BMFace* f = (mesh).fofv_begin(iter); f != (mesh).fofv_end(iter); f = (mesh).fofv_next(iter))

#define BMESH_E_OF_V(e, v, iter, mesh)\
	BMIter iter; \
	iter.init(v); \
for (BMEdge* e = (mesh).eofv_begin(iter); e != (mesh).eofv_end(iter); e = (mesh).eofv_next(iter))

#define BMESH_F_OF_E(f, e, iter, mesh)\
	BMIter iter; \
	iter.init(e); \
for (BMFace* f = (mesh).fofe_begin(iter); f != (mesh).fofe_end(iter); f = (mesh).fofe_next(iter))

#pragma endregion
//=============================================================================================

/* allow_iter allows iteration on this mempool.  note: this requires that the
 * first four bytes of the elements never contain the character string
 * 'free'.  use with care.*/

BLI_mempool *BLI_mempool_create(int esize, int totelem, int pchunk, int flag);
void *BLI_mempool_alloc(BLI_mempool *pool);
void *BLI_mempool_calloc(BLI_mempool *pool);
void  BLI_mempool_free(BLI_mempool *pool, void *addr);
void  BLI_mempool_destroy(BLI_mempool *pool);
int   BLI_mempool_count(BLI_mempool *pool);
void *BLI_mempool_findelem(BLI_mempool *pool, int index);

/** iteration stuff.  note: this may easy to produce bugs with **/
/*private structure*/
typedef struct BLI_mempool_iter {
	BLI_mempool *pool;
	struct BLI_mempool_chunk *curchunk;
	int curindex;
} BLI_mempool_iter;

/* flag */
enum {
	BLI_MEMPOOL_SYSMALLOC  = (1 << 0),
	BLI_MEMPOOL_ALLOW_ITER = (1 << 1)
};

void  BLI_mempool_iternew(BLI_mempool *pool, BLI_mempool_iter *iter);
void *BLI_mempool_iterstep(BLI_mempool_iter *iter);

//Iterators
class BMIter{
	friend class BMesh;
public:
	BMIter():count(0),firstvert(0),nextvert(0),
			firstedge(0),nextedge(0),
			firstloop(0),nextloop(0),ldata(0),l(0),
			firstpoly(0),nextpoly(0),
			c_vdata(0),c_edata(0),c_pdata(0)
			{}
	void clear();
	void init(BMLoop* l);
	void init(const BMVert* v);
	void init(const BMFace* f);
	void init(const BMEdge* e);
private:
	int count;
	BMVert *firstvert, *nextvert;
	BMEdge *firstedge, *nextedge;
	BMLoop *firstloop, *nextloop, *ldata, *l;
	BMFace *firstpoly, *nextpoly;
	const BMVert* c_vdata;
	const BMEdge* c_edata;
	const BMFace* c_pdata;
	BLI_mempool_iter pooliter;
};
};//namespace ldp


#endif//__LDP_BMESH_H__