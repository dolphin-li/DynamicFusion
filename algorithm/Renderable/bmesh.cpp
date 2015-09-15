#include "bmesh.h"
#include "Renderable.h"
#include <assert.h>
#include <glut.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <queue>
using namespace std;

#ifndef NULL
#define NULL 0
#endif
namespace ldp
{
typedef struct BMAllocTemplate {
	int totvert, totedge, totloop, totface;
} BMAllocTemplate;
BMAllocTemplate bm_mesh_chunksize_default = {512, 1024, 2048, 512};
const float VSELECT_COLOR[3]  = {175.f/255.f,175.f/255.f,48.f/255.f};
const float VNON_SEL_COLOR[3] = {0.0f,0.0f,0.0f};
const float FNON_SEL_COLOR[3] = {91.f/255.f, 97.f/255.f, 109.f/255.f};
const float FSELECT_COLOR[3] = {79.f/255.f, 56.f/255.f, 79.f/255.f};
/************************************************************************/
/* Small math functions /structures needed
/************************************************************************/
#include "bmesh_private.h"

/************************************************************************/
/* BMIter
/************************************************************************/
void BMIter::init(BMLoop* l)
{
	clear();
	ldata = l;
}
void BMIter::init(const BMVert* v)
{
	clear();
	c_vdata = v;
}
void BMIter::init(const BMFace* f)
{
	clear();
	c_pdata = f;
}
void BMIter::init(const BMEdge* e)
{
	clear();
	c_edata = e;
}
void BMIter::clear()
{
	firstvert=nextvert=0;
	firstedge=nextedge=0;
	firstloop=nextloop=ldata=l=0;
	firstpoly=nextpoly=0;
	c_vdata=0;
	c_edata=0;
	c_pdata=0;
}
/************************************************************************/
/* BMESH
/************************************************************************/
BMesh::BMesh()
{
	vpool = epool = fpool = lpool = 0;
	totvert = totedge = totloop = totface = 0;
	m_default_face_color = ldp::Float3(FNON_SEL_COLOR[0]
		, FNON_SEL_COLOR[1], FNON_SEL_COLOR[2]);

	m_render_point_size = 3;
	m_render_line_width = 1;
}

BMesh::BMesh(BMesh& other)
{
	vpool = epool = fpool = lpool = 0;
	totvert = totedge = totloop = totface = 0;
	m_default_face_color = ldp::Float3(FNON_SEL_COLOR[0]
		, FNON_SEL_COLOR[1], FNON_SEL_COLOR[2]);

	init_bmesh(&other);
}

BMesh::~BMesh()
{
	clear();
}

BMesh& BMesh::operator=(BMesh& rhs)
{
	this->init_bmesh(&rhs);
	return *this;
}

void BMesh::to_triangles(vector<float>& verts, vector<int>& faces)
{
	BMIter iter;
	for (BMFace* f=fofm_begin(iter); f!=fofm_end(iter); f=fofm_next(iter))
	{
		triangulate_face(f,0);
	}
	updateIndex();
	verts.reserve(this->totvert * 3);
	faces.reserve(this->totface * 3);

	for (BMVert* v=vofm_begin(iter); v!=vofm_end(iter); v=vofm_next(iter))
	{
		verts.push_back(v->co[0]);
		verts.push_back(v->co[1]);
		verts.push_back(v->co[2]);
	}

	for (BMFace* f=fofm_begin(iter); f!=fofm_end(iter); f=fofm_next(iter))
	{
		assert(f->len==3);
		BMIter fiter;
		fiter.init(f);
		for(BMVert* v=voff_begin(fiter); v!=voff_end(fiter); v=voff_next(fiter))
		{
			faces.push_back(v->getIndex());
		}
	}
}

void BMesh::init_bmesh(BMesh *bm_old)
{
	if(bm_old==0)	return;
	clear();
	BMVert **vtable = NULL;
	BMEdge **etable = NULL;
	BMIter iter, liter;
	int i,j;


	/* allocate a bmesh */
	initMempool(bm_old->totvert, bm_old->totedge, bm_old->totloop, bm_old->totface);
	vtable = (BMVert**)malloc(sizeof(BMVert *) * bm_old->totvert);
	etable = (BMEdge**)malloc(sizeof(BMEdge *) * bm_old->totedge);

	BMVert* v = bm_old->vofm_begin(iter);
	for (i = 0; v!=bm_old->vofm_end(iter); v = bm_old->vofm_next(iter), i++) {
		BMVert*v2 = BM_vert_create(v->co); /* copy between meshes so cant use 'example' argument */
		vtable[i] = v2;
		v->setIndex(i);
		v2->setIndex(i); /* set_inline */
		v2->setSelect(v->isSelect());
		copy_v3_v3(v2->no, v->no);
	}

	/* safety check */
	assert(i == bm_old->totvert);

	BMEdge* e = bm_old->eofm_begin(iter);
	for (i = 0; e!=bm_old->eofm_end(iter); e = bm_old->eofm_next(iter), i++) {
		BMEdge* e2 = BM_edge_create(vtable[e->v1->getIndex()],
			vtable[e->v2->getIndex()]);
		etable[i] = e2;
		e->setIndex(i);
		e2->setIndex(i); /* set_inline */
		e2->setSelect(e->isSelect());
	}

	/* safety check */
	assert(i == bm_old->totedge);

	BMFace* f = bm_old->fofm_begin(iter);
	for (i = 0; f!=bm_old->fofm_end(iter); f = bm_old->fofm_next(iter), i++) {
		f->setIndex(i);
		BMVert ** oldVerts = (BMVert**)malloc(f->len * sizeof(BMLoop*));
		BMVert ** newVerts = (BMVert**)malloc(f->len * sizeof(BMLoop*));
		BMEdge ** edges = (BMEdge**)malloc(f->len * sizeof(BMEdge*));

		liter.init(f);
		BMVert* v = bm_old->voff_begin(liter);
		for (j = 0; j < f->len; j++, v = bm_old->voff_next(liter)) {
			oldVerts[j] = v;
			newVerts[j] = vtable[v->getIndex()];
		}
		for (j=0; j<f->len; j++)
		{
			BMEdge* e= bm_old->eofv_2(oldVerts[j], oldVerts[(j+1)%f->len]);
			assert(e);
			edges[j] = etable[e->getIndex()];
		}

		BMFace *f2 = BM_face_create(newVerts, edges, f->len);
		if (!f2)
			continue;
		/* use totface in case adding some faces fails */
		f2->setIndex(this->totface - 1); /* set_inline */
		f2->setSelect(f->isSelect());

		copy_v3_v3(f2->no, f->no);

		free(oldVerts);
		free(newVerts);
		free(edges);
	}

	/* safety check */
	assert(i == bm_old->totface);

	free(etable);
	free(vtable);

	m_render_point_size = bm_old->m_render_point_size;
	m_render_line_width = bm_old->m_render_line_width;
}

void BMesh::init_triangles(int nverts, float* verts, int nfaces, int *faces)
{
	initMempool(nverts,nfaces*3/2,nfaces,nfaces);

	vector<BMVert*> savedVerts(nverts);
	for (int i=0,i3=0; i<nverts; i++,i3+=3)
	{
		savedVerts[i]=BM_vert_create(&verts[i3]);
		savedVerts[i]->head.index = i;
	}
	for (int i=0,i3=0; i<nfaces; i++,i3+=3)
	{
		BMFace* f = BM_face_create_quad_tri(savedVerts.at(faces[i3]),savedVerts.at(faces[i3+1]),
			savedVerts.at(faces[i3+2]),0);
		f->setIndex(i);
	}
	updateNormal();
	updateIndex();
}

void BMesh::init_quads(int nverts, float* verts, int nfaces, int *faces)
{
	initMempool(nverts,nfaces*4/2,nfaces,nfaces);

	vector<BMVert*> savedVerts(nverts);
	for (int i=0,i3=0; i<nverts; i++,i3+=3)
	{
		savedVerts[i]=BM_vert_create(&verts[i3]);
		savedVerts[i]->head.index = i;
	}
	for (int i=0,i4=0; i<nfaces; i++,i4+=4)
	{
		BMFace* f = BM_face_create_quad_tri(savedVerts.at(faces[i4]), savedVerts.at(faces[i4 + 1]),
			savedVerts.at(faces[i4 + 2]), savedVerts.at(faces[i4 + 3]));
		f->setIndex(i);
	}
	updateNormal();
	updateIndex();
}

void BMesh::init_polygons(int nverts, float* verts, int nfaces, int *faces, int* faceHeaders)
{
	if(nverts==0)	return;
	initMempool(nverts,nfaces*2,nfaces,nfaces);
	vector<BMVert*> savedVerts(nverts);
	vector<BMVert*> tmpVerts(nverts);
	for (int i=0,i3=0; i<nverts; i++,i3+=3)
	{
		savedVerts[i]=BM_vert_create(&verts[i3]);
		savedVerts[i]->head.index = i;
	}
	for (int i=0; i<nfaces; i++)
	{
		int ns = faceHeaders[i], ne = faceHeaders[i+1];
		for (int j=ns; j<ne; j++)
		{
			tmpVerts[j-ns] = savedVerts[faces[j]];
		}
		BMFace* f = add_face(&tmpVerts[0], ne - ns);
		f->setIndex(i);
	}
	updateNormal();
	updateIndex();
}

void BMesh::init_edges(int nverts, float* verts, int nedges, int *edges)
{
	initMempool(nverts, nedges, nedges, 0);

	vector<BMVert*> savedVerts(nverts);
	for (int i = 0, i3 = 0; i<nverts; i++, i3 += 3)
	{
		savedVerts[i] = BM_vert_create(&verts[i3]);
		savedVerts[i]->setIndex(i);
	}
	for (int i = 0, i2 = 0; i<nedges; i++, i2 += 2)
	{
		BMEdge* edge = BM_edge_create(savedVerts[edges[i2]], savedVerts[edges[i2+1]]);
		edge->setIndex(i);
	}
}

void BMesh::initMempool(int tv, int te, int tl, int tf)
{
	clear();
	if(tv>0)
	{
		vpool = BLI_mempool_create(sizeof(BMVert), tv, bm_mesh_chunksize_default.totvert, BLI_MEMPOOL_ALLOW_ITER);
		epool = BLI_mempool_create(sizeof(BMEdge), te, bm_mesh_chunksize_default.totedge, BLI_MEMPOOL_ALLOW_ITER);
		lpool = BLI_mempool_create(sizeof(BMLoop), tl, bm_mesh_chunksize_default.totloop, 0);
		fpool = BLI_mempool_create(sizeof(BMFace), tf, bm_mesh_chunksize_default.totface, BLI_MEMPOOL_ALLOW_ITER);
	}
}

void BMesh::clear()
{
	BLI_mempool_destroy(this->vpool);
	BLI_mempool_destroy(this->epool);
	BLI_mempool_destroy(this->lpool);
	BLI_mempool_destroy(this->fpool);
	vpool = epool = fpool = lpool = 0;
	totvert = totedge = totloop = totface = 0;
}

void BMesh::updateNormal()
{
	BMVert *v;
	BMFace *f;
	BMLoop *l;
	BMEdge *e;
	BMIter verts;
	BMIter faces;
	BMIter loops;
	BMIter edges;
	int index;
	float (*edgevec)[3];
	
	/* calculate all face normals */
	for(f=fofm_begin(faces); f!=fofm_end(faces); f=fofm_next(faces))
	{
		BM_face_normal_update(f);
	}
	
	/* Zero out vertex normals */
	for(v=vofm_begin(verts); v!=vofm_end(verts); v=vofm_next(verts))
	{
		zero_v3(v->no);
	}

	/* compute normalized direction vectors for each edge. directions will be
	 * used below for calculating the weights of the face normals on the vertex
	 * normals */
	index = 0;
	edgevec = (float(*)[3])malloc(sizeof(float) * 3 * this->totedge);
	for(e=eofm_begin(edges); e!=eofm_end(edges); e=eofm_next(edges))
	{
		e->setIndex(index);

		if (e->l) {
			sub_v3_v3v3(edgevec[index], e->v2->co, e->v1->co);
			normalize_v3(edgevec[index]);
		}
		else {
			/* the edge vector will not be needed when the edge has no radial */
		}

		index++;
	}

	/* add weighted face normals to vertices */
	for(f=fofm_begin(faces); f!=fofm_end(faces); f=fofm_next(faces))
	{
		loops.init(f);
		for(l=loff_begin(loops); l!=loff_end(loops); l=loff_next(loops)) 
		{
			float *e1diff, *e2diff;
			float dotprod;
			float fac;

			/* calculate the dot product of the two edges that
			 * meet at the loop's vertex */
			e1diff = edgevec[l->prev->e->getIndex()];
			e2diff = edgevec[l->e->getIndex()];
			dotprod = dot_v3v3(e1diff, e2diff);

			/* edge vectors are calculated from e->v1 to e->v2, so
			 * adjust the dot product if one but not both loops
			 * actually runs from from e->v2 to e->v1 */
			if ((l->prev->e->v1 == l->prev->v) ^ (l->e->v1 == l->v)) {
				dotprod = -dotprod;
			}

			fac = saacos(-dotprod);

			/* accumulate weighted face normal into the vertex's normal */
			madd_v3_v3fl(l->v->no, f->no, fac);
		}
	}
	
	/* normalize the accumulated vertex normals */
	for(v=vofm_begin(verts); v!=vofm_end(verts); v=vofm_next(verts))
	{
		if (normalize_v3(v->no) == 0.0f) {
			normalize_v3_v3(v->no, v->co);
		}
	}


	free(edgevec);
}

void BMesh::updateIndex(int type /* = BM_VERT | BM_EDGE | BM_FACE */)
{
	BMIter iter;
	int iiter;
	if(type & BM_VERT)
	{
		iiter=0;
		for (BMVert* v=vofm_begin(iter);v!=vofm_end(iter);v=vofm_next(iter),iiter++)
			v->setIndex(iiter);
	}
	if(type & BM_EDGE)
	{
		iiter = 0;
		for (BMEdge* e=eofm_begin(iter);e!=eofm_end(iter);e=eofm_next(iter),iiter++)
			e->setIndex(iiter);	
	}
	if(type & BM_FACE)
	{
		iiter = 0;
		for (BMFace* f=fofm_begin(iter);f!=fofm_end(iter);f=fofm_next(iter),iiter++)
			f->setIndex(iiter);	
	}
}

void BMesh::set_default_face_color(ldp::Float3 color)
{
	m_default_face_color = color;
}

void BMesh::render(int showType, int frameIdx)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glEnable(GL_COLOR_MATERIAL);
	if((showType & Renderable::SW_E) || (showType & Renderable::SW_V))
	{
		glEnable (GL_POLYGON_OFFSET_FILL); 
		glPolygonOffset (1., 1.); 
	}
	if(showType & Renderable::SW_F)
	{
		BMIter fiter;
		BMIter viter;
		BMFace* f;
		if(showType & Renderable::SW_SMOOTH)
			glColor3fv(m_default_face_color.ptr());
		else
			glColor3fv(m_default_face_color.ptr());
		for (f=fofm_begin(fiter);f!=fofm_end(fiter);f=fofm_next(fiter))
		{
			viter.init(f);
			glBegin(GL_POLYGON);
			if(showType & Renderable::SW_FLAT)
				glNormal3fv(f->no);
			for (BMVert* v=voff_begin(viter); v!=voff_end(viter); v=voff_next(viter))
			{
				if(showType & Renderable::SW_SMOOTH)
					glNormal3fv(v->no);
				if (f->isSelect())
					glColor3fv(FSELECT_COLOR);
				else
					glColor3fv(FNON_SEL_COLOR);
				glVertex3fv(v->co);
			}
			glEnd();
		}
	}
	if (showType & Renderable::SW_E)
	{
		BMIter eiter;
		BMEdge *e;
		BMVert *v1,*v2;
		glColor3f(0,0,0);
		glLineWidth(m_render_line_width);
		if (showType&SW_LIGHTING)
			glEnable(GL_LIGHTING);
		else
			glDisable(GL_LIGHTING);
		glBegin(GL_LINES);
		for (e=eofm_begin(eiter);e!=eofm_end(eiter);e=eofm_next(eiter))
		{
			v1 = vofe_first(e);
			v2 = vofe_last(e);
			if(v1->isSelect())
				glColor3fv(VSELECT_COLOR);
			else
				glColor3fv(VNON_SEL_COLOR);
			glNormal3fv(v1->no);
			glVertex3fv(v1->co);
			if(v2->isSelect())
				glColor3fv(VSELECT_COLOR);
			else
				glColor3fv(VNON_SEL_COLOR);
			glNormal3fv(v2->no);
			glVertex3fv(v2->co);
		}
		glEnd();
	}
	if (showType & Renderable::SW_V)
	{
		BMIter viter;
		BMVert *v;
		glColor3f(0,0,0);
		glPointSize(m_render_point_size);
		glDisable(GL_LIGHTING);
		glBegin(GL_POINTS);
		for (v=vofm_begin(viter);v!=vofm_end(viter);v=vofm_next(viter))
		{
			if(v->isSelect())
				glColor3fv(VSELECT_COLOR);
			else
				glColor3fv(VNON_SEL_COLOR);
			glNormal3fv(v->no);
			glVertex3fv(v->co);
		}
		glEnd();
	}

	glPopAttrib();
}

bool BMesh::save(const char* filename)
{
	FILE* pFile = fopen(filename, "w");
	if (!pFile)
		return false;

	fprintf(pFile, "verts: %d\n", totvert);
	fprintf(pFile, "edges: %d\n", totedge);
	fprintf(pFile, "faces: %d\n", totface);

	BMESH_ALL_VERTS(vptr, viter, *this)
	{
		fprintf(pFile, "v: %f %f %f vn: %f %f %f s: %d\n", vptr->co[0], vptr->co[1], vptr->co[2], 
			vptr->no[0], vptr->no[1], vptr->no[2], vptr->isSelect());
	}

	BMESH_ALL_EDGES(eptr, eiter, *this)
	{
		fprintf(pFile, "e: %d %d s: %d\n", eptr->v1->getIndex(), eptr->v2->getIndex(), eptr->isSelect());
	}

	BMESH_ALL_FACES(fptr, fiter, *this)
	{
		fprintf(pFile, "f: %d: ", this->voff_count(fptr));
		BMESH_V_OF_F(vptr, fptr, vfiter, *this)
		{
			fprintf(pFile, "%d ", vptr->getIndex());
		}
		fprintf(pFile, "s: %d\n", fptr->isSelect());
	}

	fclose(pFile);
	return true;
}

bool BMesh::load(const char* filename)
{
	FILE* pFile = fopen(filename, "r");
	if (!pFile)
		return false;

	clear();

	int nvert=0, nedge=0, nface=0;
	fscanf(pFile, "verts: %d\n", &nvert);
	fscanf(pFile, "edges: %d\n", &nedge);
	fscanf(pFile, "faces: %d\n", &nface);

	initMempool(nvert, nedge, nface, nface);

	vector<BMVert*> savedVerts(nvert);
	vector<BMVert*> tmpVerts(nvert);

	for (int i = 0; i<nvert; i++)
	{
		float co[3], no[3];
		int s;
		int r = fscanf(pFile, "v: %f %f %f vn: %f %f %f s: %d\n", &co[0], &co[1], &co[2],
			&no[0], &no[1], &no[2], &s);
		assert(r == 7);
		savedVerts[i] = BM_vert_create(co);
		savedVerts[i]->setIndex(i);
		savedVerts[i]->setSelect(s);
		for (int k = 0; k < 3; k++)
			savedVerts[i]->no[k] = no[k];
	}

	for (int i = 0; i < nedge; i++)
	{
		int v1 = 0, v2 = 0, s = 0;
		int r = fscanf(pFile, "e: %d %d s: %d\n", &v1, &v2, &s);
		assert(r == 3);
		BMEdge* edge = add_edge(savedVerts[v1], savedVerts[v2]);
		edge->setIndex(i);
		edge->setSelect(s);
	}


	for (int i = 0; i<nface; i++)
	{
		int nf = 0, s=0;
		int r = fscanf(pFile, "f: %d: ", &nf);
		assert(r == 1);
		for (int j = 0; j<nf; j++)
		{
			int jv = 0;
			r = fscanf(pFile, "%d ", &jv);
			assert(r == 1);
			tmpVerts[j] = savedVerts[jv];
		}
		r = fscanf(pFile, "s: %d\n", &s);
		assert(r == 1);
		if (nf)
		{
			BMFace* f = add_face(tmpVerts.data(), nf);
			f->setIndex(i);
			f->setSelect(s);
		}
	}
	
	fclose(pFile);
	
	return true;
}
/************************************************************************/
/* High Level functions
/************************************************************************/
struct GNode 
{
	int id;
	float val;
	GNode():val(0),id(0){}
	GNode(const GNode&rhs):val(rhs.val),id(rhs.id){}
	int getIndex()const{return id;}
	bool operator < (const GNode& rhs)const
	{
		return val < rhs.val;
	}
};
bool BMesh::find_shortest_path(BMVert* vbegin, BMVert** vends, int nvends, 
	std::vector<std::vector<BMVert*>> &paths, bool useMask, bool useTopologyDist)
{
	if(!vbegin || !vends) return false;
	vector<GNode> dist;
	vector<int> previous;
	vector<BMVert*> verts;
	LHeap<GNode> heap;
	BMIter viter;
	int nverts = vofm_count();
	int srcVertId=0;
	int iiter=0;

	verts.resize(nverts);
	for (BMVert* v=vofm_begin(viter);v!=vofm_end(viter);v=vofm_next(viter),iiter++)
	{
		verts[iiter] = v;
		v->setIndex(iiter);
		if(v==vbegin)srcVertId=iiter;
	}

	//Dijkstra initial
	dist.resize(nverts);
	previous.resize(nverts);
	for(int i=0; i<nverts; i++)
	{
		dist[i].id = i;
		dist[i].val = 1e15f;
		previous[i] = -1;
	}
	dist[srcVertId].val = 0;

	//Dijkstra main loop
	heap.init(&dist[0],dist.size());
	while(!heap.isEmpty())
	{
		GNode *u = heap.top();
		if(dist[u->id].val > 1e14f)
			break;
		heap.pop();

		//for each neighbor v of u
		BMVert *cv = verts[u->id];
		BMIter eiter;
		eiter.init(cv);
		for (BMEdge* e=eofv_begin(eiter); e!=eofv_end(eiter); e=eofv_next(eiter))
		{
			BMVert *v1 = vofe_first(e);
			if(v1 == cv) v1 = vofe_last(e);
			int vid = v1->getIndex();
			assert(vid != u->id);
			if(v1->isSelect() && useMask)
				continue;

			float disvv1 = 0;
			if(!useTopologyDist)
				disvv1 = len_v3v3(cv->co, v1->co);
			else
				disvv1 = 1;
			float alt = dist[u->id].val + disvv1;
			if (alt < dist[vid].val)
			{
				dist[vid].val = alt;
				previous[vid] = u->id;
				heap.updatePos(&dist[vid]);
			}
		}
	}//while !heap.isEmpty()

	//output
	paths.clear();
	paths.resize(nvends);
	for (int i=0; i<nvends; i++)
	{
		vector<BMVert*>& pathi = paths[i];
		int dstVertId = vends[i]->getIndex();
		if(useMask)//if useMask, then the final dist must be recalculated.
		{
			BMIter viter;
			viter.init(vends[i]);
			float disnow=1e15f;
			for(BMEdge* e=eofv_begin(viter); e!=eofv_end(viter); e=eofv_next(viter))
			{
				BMVert *v = BM_edge_other_vert(e, vends[i]);
				float alt = len_v3v3(vends[i]->co, v->co) + dist[v->getIndex()].val;
				if (alt < disnow)
				{
					disnow = alt;
					previous[dstVertId] = v->getIndex();
				}
			}
		}
		int prev = dstVertId;
		while(prev != -1){
			pathi.push_back(verts[prev]);
			prev = previous[prev];
		}
	}
	return true;
}

bool BMesh::find_shortest_path(BMVert* vbegin, BMVert* vend, std::vector<BMVert*> &paths,bool useTopologyDist)
{
	if(!vbegin || !vend) return false;
	vector<GNode> dist;
	vector<int> previous;
	vector<BMVert*> verts;
	LHeap<GNode> heap;
	BMIter viter;
	int nverts = vofm_count();
	int srcVertId=0;
	int iiter=0;

	verts.resize(nverts);
	for (BMVert* v=vofm_begin(viter);v!=vofm_end(viter);v=vofm_next(viter),iiter++)
	{
		verts[iiter] = v;
		v->setIndex(iiter);
		if(v==vbegin)srcVertId=iiter;
	}

	//Dijkstra initial
	dist.resize(nverts);
	previous.resize(nverts);
	for(int i=0; i<nverts; i++)
	{
		dist[i].id = i;
		dist[i].val = 1e15f;
		previous[i] = -1;
	}
	dist[srcVertId].val = 0;

	//Dijkstra main loop
	heap.init(&dist[0],dist.size());
	while(!heap.isEmpty())
	{
		GNode *u = heap.top();
		if(dist[u->id].val > 1e14f)
			break;
		heap.pop();

		//for each neighbor v of u
		BMVert *cv = verts[u->id];
		if(cv==vend)	break;
		BMIter eiter;
		eiter.init(cv);
		for (BMEdge* e=eofv_begin(eiter); e!=eofv_end(eiter); e=eofv_next(eiter))
		{
			BMVert *v1 = vofe_first(e);
			if(v1 == cv) v1 = vofe_last(e);
			int vid = v1->getIndex();
			assert(vid != u->id);

			float disvv1=0;
			if(!useTopologyDist)
				disvv1 = len_v3v3(cv->co, v1->co);
			else
				disvv1 = 1;
			float alt = dist[u->id].val + disvv1;
			if (alt < dist[vid].val)
			{
				dist[vid].val = alt;
				previous[vid] = u->id;
				heap.updatePos(&dist[vid]);
			}
		}
	}//while !heap.isEmpty()

	//output
	paths.clear();
	vector<BMVert*>& pathi = paths;
	int dstVertId = vend->getIndex();
	int prev = dstVertId;
	while(prev != -1){
		pathi.push_back(verts[prev]);
		prev = previous[prev];
	}
	if(paths.size()>1)
		if (paths[paths.size()-1]==vbegin)
			return true;
	return false;
}

void BMesh::select_shortest_path(BMVert* vbegin, BMVert* vend, bool useMask,bool useTopologyDist)
{
	if(vbegin==0 || vend==0)
		return;
	vector<vector<BMVert*>> paths;
	find_shortest_path(vbegin,&vend,1,paths, useMask, useTopologyDist);	
	for (unsigned int i=0; i<paths[0].size();i++)
	{
		BMVert *v = paths[0][i];
		select_vert(v,true);
	}
}

bool BMesh::calc_single_src_all_dst_geodesic(BMVert* vbegin, std::vector<float>& out_dists, 
	std::vector<int>* paths_by_parent,
	const std::vector<int>& vert_block_paths,
	bool useTopologyDist)
{
	if (!vbegin)
		return false;

	std::vector<int> should_vertex_be_block(vofm_count(), 0);
	for (int i = 0; i < vert_block_paths.size(); i++)
		should_vertex_be_block[vert_block_paths[i]] = 1;

	vector<GNode> dist;
	vector<int> previous;
	vector<BMVert*> verts;
	LHeap<GNode> heap;
	BMIter viter;
	int nverts = vofm_count();
	int srcVertId = 0;
	int iiter = 0;

	verts.resize(nverts);
	for (BMVert* v = vofm_begin(viter); v != vofm_end(viter); v = vofm_next(viter), iiter++)
	{
		verts[iiter] = v;
		v->setIndex(iiter);
		if (v == vbegin)srcVertId = iiter;
	}

	//Dijkstra initial
	dist.resize(nverts);
	previous.resize(nverts);
	for (int i = 0; i<nverts; i++)
	{
		dist[i].id = i;
		dist[i].val = 1e15f;
		previous[i] = -1;
	}
	dist[srcVertId].val = 0;

	//Dijkstra main loop
	heap.init(&dist[0], dist.size());
	while (!heap.isEmpty())
	{
		GNode *u = heap.top();
		if (dist[u->id].val > 1e14f)
			break;
		heap.pop();

		//for each neighbor v of u
		BMVert *cv = verts[u->id];
		BMIter eiter;
		eiter.init(cv);
		for (BMEdge* e = eofv_begin(eiter); e != eofv_end(eiter); e = eofv_next(eiter))
		{
			BMVert *v1 = vofe_first(e);
			if (v1 == cv) v1 = vofe_last(e);
			int vid = v1->getIndex();
			assert(vid != u->id);
			if (should_vertex_be_block[vid])
				continue;

			float disvv1 = len_v3v3(cv->co, v1->co);
			if (useTopologyDist)
				disvv1 = 1;
			float alt = dist[u->id].val + disvv1;
			if (alt < dist[vid].val)
			{
				dist[vid].val = alt;
				previous[vid] = u->id;
				heap.updatePos(&dist[vid]);
			}
		}
	}//while !heap.isEmpty()

	// if use block paths, then the final dists for the path verts 
	// should be recalculated
	for (int i_v = 0; i_v < vert_block_paths.size(); i_v++)
	{
		int vid = vert_block_paths[i_v];
		BMIter viter;
		viter.init(verts[vid]);
		float disnow = dist[vid].val;
		for (BMEdge* e = eofv_begin(viter); e != eofv_end(viter); e = eofv_next(viter))
		{
			BMVert *v = BM_edge_other_vert(e, verts[vid]);
			float alt = len_v3v3(verts[vid]->co, v->co) + dist[v->getIndex()].val;
			if (alt < disnow)
			{
				disnow = alt;
				previous[vid] = v->getIndex();
			}
		}
		dist[vid].val = disnow;
	}

	//output
	out_dists.resize(dist.size());
	for (int i = 0; i < dist.size(); i++)
		out_dists[i] = dist[i].val;

	if (paths_by_parent)
	{
		paths_by_parent->assign(previous.begin(), previous.end());
	}

	return true;
}

bool BMesh::find_linked_verts(BMVert* vsrc, std::vector<BMVert*> &linked, bool bSelect,
	bool useSelectedAsBarrier, int* vmask)
{
	if (vsrc == 0)
		return false;

	// vertices related to the selection
	linked.clear();
	std::vector<int> isVertVisited(this->vofm_count(), 0);
	std::queue<BMVert*> selectedVertIdx;
	selectedVertIdx.push(vsrc);
	isVertVisited[vsrc->getIndex()] = 1;

	if (vmask)
	for (int i = 0; i < totvert; i++)
	if (vmask[i] == 0)
		isVertVisited[i] = 1;

	// push all vertices related to the selection
	while (!selectedVertIdx.empty())
	{
		BMVert* v = selectedVertIdx.front();
		selectedVertIdx.pop();

		linked.push_back(v);

		BMESH_E_OF_V(e, v, eiter, *this)
		{
			BMVert* v1 = this->vofe_first(e);
			if (v1 == v)
				v1 = this->vofe_last(e);
			if (!isVertVisited[v1->getIndex()] && v1->isSelect() != bSelect)
			{
				isVertVisited[v1->getIndex()] = 1;
				selectedVertIdx.push(v1);
			}
		}
	}
	return true;
}

void BMesh::select_linked_verts(BMVert* vsrc, bool bSelect, bool useSelectedAsBarrier, int* vmask)
{
	if (vsrc == 0)
		return;
	vector<BMVert*> linked;
	find_linked_verts(vsrc, linked, bSelect, useSelectedAsBarrier, vmask);
	for (unsigned int i = 0; i<linked.size(); i++)
	{
		BMVert *v = linked[i];
		select_vert(v, bSelect);
	}
}

BMVert* BMesh::vert_at(int idx)
{
	return (BMVert*)BLI_mempool_findelem(this->vpool, idx);
}
BMEdge* BMesh::edge_at(int idx)
{
	return (BMEdge*)BLI_mempool_findelem(this->epool, idx);
}
BMFace* BMesh::face_at(int idx)
{
	return (BMFace*)BLI_mempool_findelem(this->fpool, idx);
}

//
void BMesh::split_edge(BMEdge * e)
{
	printf("Non Implemented method!\n");
	
}

BMFace*	BMesh::split_faces(BMFace* f, BMVert* v1, BMVert* v2)
{
	return bmesh_sfme(f,v1,v2,0);
}

struct EdgeSort
{
	bool operator()(const BMEdge* e1, const BMEdge* e2)const
	{
		BMVert * v11, *v12, *v21, *v22;
		v11 = BMesh::vofe_first(e1);
		v12 = BMesh::vofe_last(e1);
		v21 = BMesh::vofe_first(e2);
		v22 = BMesh::vofe_last(e2);
		if(v11 < v12) SWAP(BMVert*, v11,v12);
		if(v21 < v22) SWAP(BMVert*, v21,v22);
		return (v11<v21) || (v11==v21 && v12<v22);
	}
};

void BMesh::merge_verts(BMVert* v1, BMVert* v2)
{
	if(v1==v2)	return;
	BMEdge *edge = eofv_2(v1, v2);
	if(edge)	remove_edge(edge);
	BM_vert_splice(this, v1, v2);

	//Removing co-linear edges.
	BMIter eiter2;
	eiter2.init(v2);
	set<BMVert*> verts;
	vector<BMEdge*> edges;
	for(BMEdge* e2 = eofv_begin(eiter2); e2!=eofv_end(eiter2); e2=eofv_next(eiter2))
	{
		edges.push_back(e2);
	}

	sort(edges.begin(),edges.end(),EdgeSort());

	if(edges.size() >= 1)
	{
		for (unsigned int i=0; i<edges.size()-1;)
		{
			BMEdge* e1 = edges[i];
			if(e1->v1 == e1->v2)
				remove_edge(e1);
			for (unsigned int j=i+1; j<edges.size(); j++)
			{
				BMEdge* e2 = edges[j];
				if(e2->v1 == e2->v2)
					remove_edge(e2);
				int cc = int(e1->v1==e2->v1) + int(e1->v1==e2->v2) + int(e1->v2==e2->v1) + int(e1->v2==e2->v2);
				if (cc==2)
				{
					i=j+1;
					BM_edge_splice(this, e2,e1);
				}
				else
				{
					i=j;
					break;
				}
			}
		}//end for
	}//end if
}


BMFace* BMesh::merge_faces(BMFace*f1, BMFace*f2)
{
	if(f1==0 || f2==0)
		return 0;
	BMIter iter1,iter2;
	BMEdge *e=0;
	iter1.init(f1);
	iter2.init(f2);
	int count=0;
	for(BMEdge* e1=eoff_begin(iter1); e1!=eoff_end(iter1); e1=eoff_next(iter1))
	{
		for(BMEdge* e2=eoff_begin(iter2); e2!=eoff_end(iter2); e2=eoff_next(iter2))
		{
			if(e1==e2)
			{
				e=e1;
				count++;
			}
		}
	}
	if(count<=0 )
		return 0;
	return bmesh_jfke(f1,f2,e);
}
/**
* Queries: Vert Of Mesh.
* */
BMVert* BMesh::vofm_begin(BMIter& iter)
{
	if(!this->vpool)	return 0;
	BLI_mempool_iternew(this->vpool, &iter.pooliter);
	return (BMVert*)BLI_mempool_iterstep(&iter.pooliter);
}
BMVert* BMesh::vofm_next(BMIter& iter)
{
	return (BMVert*)BLI_mempool_iterstep(&iter.pooliter);
}
BMVert* BMesh::vofm_end(BMIter& iter)
{
	return 0;
}
const BMVert* BMesh::vofm_begin(BMIter& iter)const 
{
	if(!this->vpool)	return 0;
	BLI_mempool_iternew(this->vpool, &iter.pooliter);
	return (BMVert*)BLI_mempool_iterstep(&iter.pooliter);
}
const BMVert* BMesh::vofm_next(BMIter& iter)const 
{
	return (BMVert*)BLI_mempool_iterstep(&iter.pooliter);
}
const BMVert* BMesh::vofm_end(BMIter& iter)const 
{
	return 0;
}
/**
* Queries: Edge Of Mesh.
* */
BMEdge* BMesh::eofm_begin(BMIter& iter)
{
	if(!this->epool)	return 0;
	BLI_mempool_iternew(this->epool, &iter.pooliter);
	return (BMEdge*)BLI_mempool_iterstep(&iter.pooliter);
}
BMEdge* BMesh::eofm_next(BMIter& iter)
{
	return (BMEdge*)BLI_mempool_iterstep(&iter.pooliter);
}
BMEdge* BMesh::eofm_end(BMIter& iter)
{
	return 0;
}
const BMEdge* BMesh::eofm_begin(BMIter& iter)const 
{
	if(!this->epool)	return 0;
	BLI_mempool_iternew(this->epool, &iter.pooliter);
	return (BMEdge*)BLI_mempool_iterstep(&iter.pooliter);
}
const BMEdge* BMesh::eofm_next(BMIter& iter)const 
{
	return (BMEdge*)BLI_mempool_iterstep(&iter.pooliter);
}
const BMEdge* BMesh::eofm_end(BMIter& iter)const 
{
	return 0;
}
/**
* Queries: Face Of Mesh
* */
BMFace* BMesh::fofm_begin(BMIter& iter)
{
	if(!this->fpool)	return 0;
	BLI_mempool_iternew(this->fpool, &iter.pooliter);
	return (BMFace*)BLI_mempool_iterstep(&iter.pooliter);
}
BMFace* BMesh::fofm_next(BMIter& iter)
{
	return (BMFace*)BLI_mempool_iterstep(&iter.pooliter);
}
BMFace* BMesh::fofm_end(BMIter& iter)
{
	return 0;
}
const BMFace* BMesh::fofm_begin(BMIter& iter)const 
{
	if(!this->fpool)	return 0;
	BLI_mempool_iternew(this->fpool, &iter.pooliter);
	return (BMFace*)BLI_mempool_iterstep(&iter.pooliter);
}
const BMFace* BMesh::fofm_next(BMIter& iter)const 
{
	return (BMFace*)BLI_mempool_iterstep(&iter.pooliter);
}
const BMFace* BMesh::fofm_end(BMIter& iter)const 
{
	return 0;
}

//vert of edge
BMVert* BMesh::vofe_first(const BMEdge* e)
{
	return e->v1;
}
BMVert* BMesh::vofe_last(const BMEdge* e)
{
	return e->v2;
}
int BMesh::vofe_count(const BMEdge* e)
{
	return 2;
}

//vert of face
BMVert* BMesh::voff_begin(BMIter& iter)
{
	assert(iter.c_pdata);
	iter.l = iter.c_pdata->l_first;
	return iter.l ? iter.l->v : 0;
}
BMVert* BMesh::voff_next(BMIter& iter)
{
	if(iter.l)	iter.l=iter.l->next;
	if(iter.l==iter.c_pdata->l_first) iter.l=0;
	return iter.l ? iter.l->v : 0;
}
BMVert* BMesh::voff_end(BMIter& iter)
{
	return 0;
}
int BMesh::voff_count(const BMFace* f)
{
	return f->len;
}

//edge of vert
BMEdge* BMesh::eofv_begin(BMIter& iter)
{
	assert(iter.c_vdata);
	iter.nextedge = iter.c_vdata->e;
	return iter.c_vdata->e;
}
BMEdge* BMesh::eofv_next(BMIter& iter)
{
	if(iter.nextedge==0)	return 0;
	BMEdge* e = bmesh_disk_edge_next(iter.nextedge, iter.c_vdata);
	if(e==iter.c_vdata->e)	e=0;
	iter.nextedge = e;
	return e;
}
BMEdge* BMesh::eofv_end(BMIter& iter)
{
	return 0;
}
BMEdge* BMesh::eofv_2(BMVert* v1, BMVert* v2)
{
	BMIter iter;
	iter.init(v1);
	for(BMEdge* e = eofv_begin(iter); e!= eofv_end(iter); e=eofv_next(iter))
	{
		if(e->v1 == v2 || e->v2 == v2)
			return e;
	}
	return 0;
}
int BMesh::eofv_count(BMVert* v)
{
	BMIter iter;
	int count = 0;
	iter.init(v);
	for(BMEdge* e = eofv_begin(iter); e!= eofv_end(iter); e=eofv_next(iter))
	{
		count++;
	}
	return count;
}

//edge of face
BMEdge* BMesh::eoff_begin(BMIter& iter)
{
	iter.l = iter.c_pdata->l_first;
	return iter.l ? iter.l->e : 0;
}
BMEdge* BMesh::eoff_next(BMIter& iter)
{
	if(iter.l)	iter.l=iter.l->next;
	if(iter.l==iter.c_pdata->l_first) iter.l=0;
	return iter.l ? iter.l->e : 0;
}
BMEdge* BMesh::eoff_end(BMIter& iter)
{
	return 0;
}
int BMesh::eoff_count(BMFace* f)
{
	BMIter iter;
	int count = 0;
	iter.init(f);
	for(BMEdge* e = eoff_begin(iter); e!= eoff_end(iter); e=eoff_next(iter))
	{
		count++;
	}
	return count;
}

//face of vert
BMFace* BMesh::fofv_begin(BMIter& iter)
{
	assert(iter.c_vdata);
	iter.count=0;
	if(iter.c_vdata->e) iter.count=bmesh_disk_facevert_count(iter.c_vdata);
	if(iter.count)
	{
		iter.firstedge = bmesh_disk_faceedge_find_first(iter.c_vdata->e, iter.c_vdata);
		iter.nextedge = iter.firstedge;
		iter.firstloop = bmesh_radial_faceloop_find_first(iter.firstedge->l, iter.c_vdata);
		iter.nextloop = iter.firstloop;
		return iter.firstloop ? iter.firstloop->f : 0;
	}
	return 0;
}
BMFace* BMesh::fofv_next(BMIter& iter)
{
	BMLoop* current = iter.nextloop;
	if(iter.count && iter.nextloop)
	{
		iter.count--;
		iter.nextloop = bmesh_radial_faceloop_find_next(iter.nextloop, iter.c_vdata);
		if(iter.nextloop == iter.firstloop)
		{
			iter.nextedge = bmesh_disk_faceedge_find_next(iter.nextedge, iter.c_vdata);
			iter.firstloop = bmesh_radial_faceloop_find_first(iter.nextedge->l, iter.c_vdata);
			iter.nextloop = iter.firstloop;
		}
	}

	if(!iter.count) iter.nextloop = 0;
	return iter.nextloop ? iter.nextloop->f : 0;
}
BMFace* BMesh::fofv_end(BMIter& iter)
{
	return 0;
}
int BMesh::fofv_count(BMVert* v)
{
	BMIter iter;
	int count = 0;
	iter.init(v);
	for(BMFace* f = fofv_begin(iter); f!= fofv_end(iter); f=fofv_next(iter))
	{
		count++;
	}
	return count;
}

//face of edge
BMFace* BMesh::fofe_begin(BMIter& iter)
{
	assert(iter.c_edata);
	if(iter.c_edata->l)
	{
		iter.firstloop = iter.nextloop = iter.c_edata->l;
		return iter.nextloop->f;
	}
	return 0;
}
BMFace* BMesh::fofe_next(BMIter& iter)
{
	if(iter.nextloop)
		iter.nextloop = iter.nextloop->radial_next;
	if(iter.nextloop == iter.firstloop)	iter.nextloop = 0;
	return iter.nextloop ? iter.nextloop->f : 0;
}
BMFace* BMesh::fofe_end(BMIter& iter)
{
	return 0;
}
int BMesh::fofe_count(BMEdge* e)
{
	BMIter iter;
	int count = 0;
	iter.init(e);
	for(BMFace* f = fofe_begin(iter); f!= fofe_end(iter); f=fofe_next(iter))
	{
		count++;
	}
	return count;
}

/**
* Queries: Loop Of Vert
* Useage: BMIter iter; iter.init(v); for(BMLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
* */
BMLoop* BMesh::lofv_begin(BMIter& iter)
{
	assert(iter.c_vdata);
	iter.count = 0;
	if (iter.c_vdata->e)
		iter.count = bmesh_disk_facevert_count(iter.c_vdata);
	if (iter.count) {
		iter.firstedge = bmesh_disk_faceedge_find_first(iter.c_vdata->e, iter.c_vdata);
		iter.nextedge = iter.firstedge;
		iter.firstloop = bmesh_radial_faceloop_find_first(iter.firstedge->l, iter.c_vdata);
		iter.nextloop = iter.firstloop;
		return iter.nextloop;
	}
	return 0;
}
BMLoop* BMesh::lofv_next(BMIter& iter)
{
	if (iter.count) {
		iter.count--;
		iter.nextloop = bmesh_radial_faceloop_find_next(iter.nextloop, iter.c_vdata);
		if (iter.nextloop == iter.firstloop) {
			iter.nextedge = bmesh_disk_faceedge_find_next(iter.nextedge, iter.c_vdata);
			iter.firstloop = bmesh_radial_faceloop_find_first(iter.nextedge->l, iter.c_vdata);
			iter.nextloop = iter.firstloop;
		}
	}

	if (!iter.count) iter.nextloop = NULL;
	return iter.nextloop;
}
BMLoop* BMesh::lofv_end(BMIter& iter)
{
	return 0;
}
int BMesh::lofv_count(BMVert* v)
{
	BMIter iter;
	int count = 0;
	iter.init(v);
	for(BMLoop* l = lofv_begin(iter); l!= lofv_end(iter); l=lofv_next(iter))
	{
		count++;
	}
	return count;
}
/**
* Queries: Loop Of Edge
* Useage: BMIter iter; iter.init(e); for(BMLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
* */
BMLoop* BMesh::lofe_begin(BMIter& iter)
{
	assert(iter.c_edata);
	BMLoop *l;
	l = iter.c_edata->l;
	iter.firstloop = iter.nextloop = l;
	return l;
}
BMLoop* BMesh::lofe_next(BMIter& iter)
{
	if (iter.nextloop)
		iter.nextloop = iter.nextloop->radial_next;
	if (iter.nextloop == iter.firstloop)
		iter.nextloop = 0;
	return iter.nextloop;
}
BMLoop* BMesh::lofe_end(BMIter& iter)
{
	return 0;
}
int BMesh::lofe_count(BMEdge* e)
{
	BMIter iter;
	int count = 0;
	iter.init(e);
	for(BMLoop* l = lofe_begin(iter); l!= lofe_end(iter); l=lofe_next(iter))
	{
		count++;
	}
	return count;
}
/**
* Queries: Loop Of Face
* Useage: BMIter iter; iter.init(f); for(BMLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
* */
BMLoop* BMesh::loff_begin(BMIter& iter)
{
	iter.firstloop = iter.nextloop = iter.c_pdata->l_first;
	return iter.nextloop;
}
BMLoop* BMesh::loff_next(BMIter& iter)
{
	if (iter.nextloop) iter.nextloop = iter.nextloop->next;
	if (iter.nextloop == iter.firstloop) iter.nextloop = 0;
	return iter.nextloop;
}
BMLoop* BMesh::loff_end(BMIter& iter)
{
	return 0;
}
int BMesh::loff_count(BMFace* f)
{
	BMIter iter;
	int count = 0;
	iter.init(f);
	for(BMLoop* l = loff_begin(iter); l!= loff_end(iter); l=loff_next(iter))
	{
		count++;
	}
	return count;
}

//loops of loop
BMLoop* BMesh::lofl_begin(BMIter& iter)
{
	BMLoop *l;

	l = iter.ldata;
	iter.firstloop = l;
	iter.nextloop = iter.firstloop->radial_next;
	
	if (iter.nextloop == iter.firstloop)
		iter.nextloop = 0;
	return iter.nextloop;
}
BMLoop* BMesh::lofl_next(BMIter& iter)
{
	if (iter.nextloop)
		iter.nextloop = iter.nextloop->radial_next;
	if (iter.nextloop == iter.firstloop)
		iter.nextloop = 0;
	return iter.nextloop;
}
BMLoop* BMesh::lofl_end(BMIter& iter)
{
	return 0;
}
int BMesh::lofl_count(BMLoop* a_l)
{
	BMIter iter;
	int count = 0;
	iter.init(a_l);
	for(BMLoop* l = lofl_begin(iter); l!= lofl_end(iter); l=lofl_next(iter))
	{
		count++;
	}
	return count;
}

BMVert* BMesh::add_vert(float co[3])
{
	return BM_vert_create(co);
}
BMEdge* BMesh::add_edge(BMVert* v1, BMVert* v2)
{
	return BM_edge_create(v1,v2);
}
BMFace* BMesh::add_face(BMVert** verts, int len)
{
	if(len <= 0) return 0;
	BMEdge** edar = new BMEdge*[len];
	for (int i=0; i<len;i++)
	{
		edar[i] = BM_edge_create(verts[i],verts[(i+1)%len]);
	}
	BMFace *f = BM_face_create(verts, edar, len);
	delete edar;

	//select it.
	bool isfselect=true;
	BMIter viter;
	viter.init(f);
	for (BMVert* v1=voff_begin(viter); v1!=voff_end(viter); v1=voff_next(viter))
	{
		if(!v1->isSelect()){isfselect=false; break;}
	}
	if(isfselect) f->setSelect(true);
	else f->setSelect(false);

	return f;
}
/**
* remove given vertex and all edges/faces using it
* */
void BMesh::remove_vert(BMVert* v)
{
	if(v)
		BM_vert_kill(this,v);
}
/**
* remove given edge and all faces using it
* */
void BMesh::remove_edge(BMEdge* e)
{
	if(e)
		BM_edge_kill(this,e);
}
/**
* remove given face
* */
void BMesh::remove_face(BMFace* f)
{
	if(f)
		BM_face_kill(this,f);
}

void BMesh::remove_face_vert(BMFace* f)
{
	if(!f) return;
	vector<BMEdge *> es;
	vector<BMVert *> vs;
	BMIter iter;
	iter.init(f);
	for(BMEdge *e = eoff_begin(iter); e!=eoff_end(iter); e=eoff_next(iter))
		es.push_back(e);
	for(BMVert *v = voff_begin(iter); v!=voff_end(iter); v=voff_next(iter))
		vs.push_back(v);
	remove_face(f);
	for (unsigned int i=0; i<es.size(); i++)
	{
		if(fofe_count(es[i])==0)
			remove_edge(es[i]);
	}
	for (unsigned int i=0; i<vs.size(); i++)
	{
		if(eofv_count(vs[i])==0)
			remove_vert(vs[i]);
	}
}

void	BMesh::remove_all(int type)
{
	BMIter iter;
	if(type & BM_VERT)
	{
		for (BMVert* v=vofm_begin(iter); v!=vofm_end(iter); v=vofm_next(iter))
		{
			remove_vert(v);
		}
	}
	if(type & BM_EDGE)
	{
		for (BMEdge* e=eofm_begin(iter); e!=eofm_end(iter); e=eofm_next(iter))
		{
			remove_edge(e);
		}
	}
	if(type & BM_FACE)
	{
		for (BMFace* f=fofm_begin(iter); f!=fofm_end(iter); f=fofm_next(iter))
		{
			remove_face(f);
		}
	}
}
void	BMesh::remove_selected(int type)
{
	BMIter iter;
	if (type == BM_VERT)
	{
		for (BMVert* v=vofm_begin(iter); v!=vofm_end(iter); v=vofm_next(iter))
		{
			if(v->isSelect())
				remove_vert(v);
		}
	}
	if(type == (BM_VERT | BM_FACE))
	{
		for (BMFace* f=fofm_begin(iter); f!=fofm_end(iter); f=fofm_next(iter))
		{
			if(f->isSelect())
				remove_face_vert(f);
		}
	}
	if (type == BM_FACE)
	{
		for (BMFace* f = fofm_begin(iter); f != fofm_end(iter); f = fofm_next(iter))
		{
			if (f->isSelect())
				remove_face(f);
		}
	}
	if(type & BM_EDGE)
	{
		for (BMEdge* e=eofm_begin(iter); e!=eofm_end(iter); e=eofm_next(iter))
		{
			if(e->isSelect())
				remove_edge(e);
		}
	}
}

/**
* select a vert, if 2-verts of a edge is selected, then is the edge; if all verts of a face is selected, then is the face
* */
void BMesh::select_vert(BMVert* v, bool sel)
{
	if(!v) return;
	v->setSelect(sel);
	BMIter iter;
	iter.init(v);
	for(BMFace* f=fofv_begin(iter); f!=fofv_end(iter); f=fofv_next(iter))
	{
		bool isfselect=true;
		BMIter viter;
		viter.init(f);
		for (BMVert* v1=voff_begin(viter); v1!=voff_end(viter); v1=voff_next(viter))
		{
			if(!v1->isSelect()){isfselect=false; break;}
		}
		if(isfselect) f->setSelect(true);
		else f->setSelect(false);
	}
	for (BMEdge* e=eofv_begin(iter); e!=eofv_end(iter); e=eofv_next(iter))
	{
		if(vofe_first(e)->isSelect() && vofe_last(e)->isSelect())
			e->setSelect(true);
		else
			e->setSelect(false);
	}
}
/**
* select a edge, so is its 2 verts
* */
void BMesh::select_edge(BMEdge* e, bool sel)
{
	if(!e) return;
	e->setSelect(sel);
	vofe_first(e)->setSelect(sel);
	vofe_last(e)->setSelect(sel);
}
/**
* select a face, so is its all verts.
* */
void BMesh::select_face(BMFace* f, bool sel)
{
	if(!f) return;
	f->setSelect(sel);

	BMESH_V_OF_F(vptr, f, iter, *this)
	{
		vptr->setSelect(sel);
	}
}

void BMesh::select_onering(BMVert* v, bool sel)
{
	BMIter iter;
	iter.init(v);
	for (BMEdge* e=eofv_begin(iter); e!=eofv_end(iter); e=eofv_next(iter))
	{
		BMVert* v1;
		v1 = vofe_first(e);
		if(v1==v) v1=vofe_last(e);
		assert(v1!=v);
		select_vert(v1,sel);
	}
}

void	BMesh::select_all(int type, bool sel)
{
	BMIter iter;
	if(type & BM_VERT)
	{
		for (BMVert* v=vofm_begin(iter); v!=vofm_end(iter); v=vofm_next(iter))
		{
			select_vert(v,sel);
		}
	}
	if(type & BM_EDGE)
	{
		for (BMEdge* e=eofm_begin(iter); e!=eofm_end(iter); e=eofm_next(iter))
		{
			select_edge(e,sel);
		}
	}
	if(type & BM_FACE)
	{
		for (BMFace* f=fofm_begin(iter); f!=fofm_end(iter); f=fofm_next(iter))
		{
			select_face(f,sel);
		}
	}
}

void	BMesh::select_inverse(int type)
{
	BMIter iter;
	if(type & BM_VERT)
	{
		for (BMVert* v=vofm_begin(iter); v!=vofm_end(iter); v=vofm_next(iter))
		{
			select_vert(v,!v->isSelect());
		}
	}
	if(type & BM_EDGE)
	{
		for (BMEdge* e=eofm_begin(iter); e!=eofm_end(iter); e=eofm_next(iter))
		{
			select_edge(e,!e->isSelect());
		}
	}
	if(type & BM_FACE)
	{
		for (BMFace* f=fofm_begin(iter); f!=fofm_end(iter); f=fofm_next(iter))
		{
			select_face(f,!f->isSelect());
		}
	}
}
struct vec3
{
	float x,y,z;
	vec3(float* p):x(p[0]),y(p[1]),z(p[2]){}
	const float& operator [](int i)const{return ((float*)this)[i];}
	float& operator [](int i){return ((float*)this)[i];}
	float * ptr(){return &x;}
};
void	BMesh::triangulate_face(BMFace* f, BMFace** newfaces)
{
	int i, done, nvert, nf_i = 0;
	BMLoop *newl, *nextloop;
	BMLoop *l_iter;
	BMLoop *l_first;
	float polyArea = 0.f;
	vector<vec3> projectverts;

	/* copy vertex coordinates to vertspace arra */
	i = 0;
	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		projectverts.push_back(l_iter->v->co);
		l_iter->v->setIndex(i); /* set dirty! */
		i++;
	} while ((l_iter = l_iter->next) != l_first);

	calc_poly_normal(f->no, (float*)&projectverts[0], f->len);
	poly_rotate_plane(f->no, (float*)&projectverts[0], i);

	nvert = f->len;

	//calc_poly_plane(projectverts, i);
	for (i = 0; i < nvert; i++) {
		projectverts[i][2] = 0.0f;
		polyArea += cross_v2v2(projectverts[i].ptr(),projectverts[(i+1)%nvert].ptr());
	}

	done = 0;
	while (!done && f->len > 3) {
		done = 1;
		l_iter = find_ear(f, (float (*)[3])&projectverts[0], nvert, 1, polyArea);
		if (l_iter) {
			done = 0;
			f = BM_face_split(l_iter->f, l_iter->prev->v,l_iter->next->v,&newl);

			if (UNLIKELY(!f)) {
				fprintf(stderr, "%s: triangulator failed to split face! (bmesh internal error)\n", __func__);
				break;
			}

			copy_v3_v3(f->no, l_iter->f->no);
			
			if (newfaces) newfaces[nf_i++] = f;
		}
	}

	if (f->len > 3) {
		l_iter = BM_FACE_FIRST_LOOP(f);
		while (l_iter->f->len > 3) {
			nextloop = l_iter->next->next;
			f = BM_face_split(l_iter->f, l_iter->v, nextloop->v,&newl);
			if (!f) {
				printf("triangle fan step of triangulator failed.\n");

				/* NULL-terminate */
				if (newfaces) newfaces[nf_i] = NULL;
				return;
			}

			if (newfaces) newfaces[nf_i++] = f;
			l_iter = nextloop;
		}
	}
	
	/* NULL-terminate */
	if (newfaces) newfaces[nf_i] = NULL;
}

void	BMesh::triangulate_selected()
{
	BMIter iter;
	for (BMFace*f = fofm_begin(iter); f!=fofm_end(iter); f=fofm_next(iter))
	{
		if(f->isSelect())
		{
			vector<BMFace*> faces;
			faces.resize(f->len);
			memset(&faces[0], 0, f->len * sizeof(BMFace*));
			triangulate_face(f,&faces[0]);
			for (unsigned int i=0; i<faces.size(); i++)
			{
				if(faces[i])
					faces[i]->setSelect(true);
			}
		}
	}
}
/************************************************************************/
/* Low Level functions
/************************************************************************/
/**
 * \brief Make Quad/Triangle
 *
 * Creates a new quad or triangle from a list of 3 or 4 vertices.
 * \note The winding of the face is determined by the order
 * of the vertices in the vertex array.
 */
BMFace *BMesh::BM_face_create_quad_tri(BMVert *v1, BMVert *v2, BMVert *v3, BMVert *v4)
{
	BMVert *vtar[4] = {v1, v2, v3, v4};
	return BM_face_create_quad_tri_v(vtar, v4 ? 4 : 3);
}

BMFace* BMesh::BM_face_create_quad_tri_v(BMVert**verts,int len)
{
	BMFace *f = 0;
	int is_overlap = false;

	/* sanity check - debug mode only */
	if (len == 3) {
		assert(verts[0] != verts[1]);
		assert(verts[0] != verts[2]);
		assert(verts[1] != verts[2]);
	}
	else if (len == 4) {
		assert(verts[0] != verts[1]);
		assert(verts[0] != verts[2]);
		assert(verts[0] != verts[3]);

		assert(verts[1] != verts[2]);
		assert(verts[1] != verts[3]);

		assert(verts[2] != verts[3]);
	}
	else {
		assert(0);
	}

	/* make new face */
	if ((f == NULL) && (!is_overlap)) {
		BMEdge* edar[4];
		edar[0] = BM_edge_create(verts[0], verts[1]);
		edar[1] = BM_edge_create(verts[1], verts[2]);
		if (len == 4) {
			edar[2] = BM_edge_create(verts[2], verts[3]);
			edar[3] = BM_edge_create(verts[3], verts[0]);
		}
		else {
			edar[2] = BM_edge_create(verts[2], verts[0]);
		}

		f = BM_face_create(verts, edar, len);
	}

	return f;
}

BMVert* BMesh::BM_vert_create(const float co[3])
{
	BMVert *v = (BMVert*)BLI_mempool_calloc(this->vpool);
	v->head.index = -1;
	this->totvert++;
	v->head.htype = BM_VERT;
	v->head.select = false;

	/* 'v->no' is handled by BM_elem_attrs_copy */
	v->co[0] = co[0];
	v->co[1] = co[1];
	v->co[2] = co[2];

	return v;
}

BMFace* BMesh::BM_face_create(BMVert **verts, BMEdge **edges, const int len)
{
	if (len == 0)
		return 0;

	BMFace *f=bm_face_create_internal();
	BMLoop *l, *startl, *lastl;
	int i;

	startl = lastl = bm_face_boundary_add(f, verts[0], edges[0]);
	startl->v = verts[0];
	startl->e = edges[0];
	for (i = 1; i < len; i++)
	{
		l = bm_loop_create(verts[i],edges[i],f);
		l->f = f;
		bmesh_radial_append(edges[i],l);

		l->prev = lastl;
		lastl->next = l;
		lastl = l;
	}

	startl->prev = lastl;
	lastl->next = startl;
	f->len = len;
	return f;
}

BMFace* BMesh::bm_face_create_internal()
{
	BMFace *f;
	f = (BMFace*)BLI_mempool_calloc(fpool);
	f->head.index = -1;
	this->totface++;
	f->head.htype = BM_FACE;
	f->head.select = false;
	return f;
}

BMFace* BMesh::BM_face_create_ngon(BMVert* v1, BMVert* v2, BMEdge** edges, int len)
{
	vector<BMEdge*> edges2;
	vector<BMVert*> verts;
	edges2.reserve(BM_NGON_STACK_SIZE);
	verts.reserve(BM_NGON_STACK_SIZE);
	BMFace *f = NULL;
	BMEdge *e;
	BMVert *v, *ev1, *ev2;
	int i, /* j, */ v1found, reverse;

	/* this code is hideous, yeek.  I'll have to think about ways of
	 *  cleaning it up.  basically, it now combines the old BM_face_create_ngon
	 *  _and_ the old bmesh_mf functions, so its kindof smashed together
	 * - joeedh */

	if (!len || !v1 || !v2 || !edges)
		return NULL;

	ev1 = edges[0]->v1;
	ev2 = edges[0]->v2;

	if (v1 == ev2) {
		/* Swapping here improves performance and consistency of face
		 * structure in the special case that the edges are already in
		 * the correct order and winding */
		SWAP(BMVert *, ev1, ev2);
	}

	verts.push_back(ev1);
	v = ev2;
	e = edges[0];
	do {
		BMEdge *e2 = e;

		verts.push_back(v);
		edges2.push_back(e);

		do {
			e2 = bmesh_disk_edge_next(e2, v);
			if (e2 != e) {
				v = BM_edge_other_vert(e2, v);
				break;
			}
		} while (e2 != e);

		if (e2 == e)
			return 0; /* the edges do not form a closed loop */

		e = e2;
	} while (e != edges[0]);

	if (edges2.size() != len) {
		return 0; /* we didn't use all edges in forming the boundary loop */
	}

	/* ok, edges are in correct order, now ensure they are going
	 * in the correct direction */
	v1found = reverse = FALSE;
	for (i = 0; i < len; i++) {
		if (BM_vert_in_edge(edges2[i], v1)) {
			/* see if v1 and v2 are in the same edge */
			if (BM_vert_in_edge(edges2[i], v2)) {
				/* if v1 is shared by the *next* edge, then the winding
				 * is incorrect */
				if (BM_vert_in_edge(edges2[(i + 1) % len], v1)) {
					reverse = TRUE;
					break;
				}
			}

			v1found = TRUE;
		}

		if ((v1found == FALSE) && BM_vert_in_edge(edges2[i], v2)) {
			reverse = TRUE;
			break;
		}
	}

	if (reverse) {
		for (i = 0; i < len / 2; i++) {
			v = verts[i];
			verts[i] = verts[len - i - 1];
			verts[len - i - 1] = v;
		}
	}

	for (i = 0; i < len; i++) {
		edges2[i] = BM_edge_exists(verts[i], verts[(i + 1) % len]);
		if (!edges2[i]) {
			return 0;
		}
	}

	f = BM_face_create(&verts[0], &edges2[0], len);

	return f;
}

BMEdge* BMesh::BM_edge_create(BMVert* v1, BMVert* v2)
{
	BMEdge *e;
	if (e = eofv_2(v1, v2))
		return e;
	e = (BMEdge*)BLI_mempool_calloc(epool);
	e->head.index = -1; /* set_ok_invalid */
	this->totedge++;
	e->head.htype = BM_EDGE;
	e->v1 = v1;
	e->v2 = v2;
	e->head.select = false;

	bmesh_disk_edge_append(e, e->v1);
	bmesh_disk_edge_append(e, e->v2);
	return e;
}

BMLoop* BMesh::bm_loop_create(BMVert *v, BMEdge *e, BMFace *f)
{
	BMLoop *l = NULL;

	l = (BMLoop*)BLI_mempool_calloc(this->lpool);
	l->next = l->prev = NULL;
	l->v = v;
	l->e = e;
	l->f = f;
	l->radial_next = l->radial_prev = NULL;
	l->head.htype = BM_LOOP;
	l->head.select = false;
	this->totloop++;
	return l;
}

BMLoop* BMesh::bm_face_boundary_add(BMFace*f, BMVert* startv, BMEdge* starte)
{
	BMLoop *l = bm_loop_create(startv, starte, f);
	bmesh_radial_append(starte, l);
	f->l_first = l;
	l->f = f;
	return l;
}

/**
 * \brief Split Edge Make Vert (SEMV)
 *
 * Takes \a e edge and splits it into two, creating a new vert.
 * \a tv should be one end of \a e : the newly created edge
 * will be attached to that end and is returned in \a r_e.
 *
 * \par Examples:
 *
 *                     E
 *     Before: OV-------------TV
 *
 *                 E       RE
 *     After:  OV------NV-----TV
 *
 * \return The newly created BMVert pointer.
 */
BMVert *BMesh::bmesh_semv(BMVert *tv, BMEdge *e, BMEdge **r_e)
{
	BMLoop *nextl;
	BMEdge *ne;
	BMVert *nv, *ov;
	int i, edok, valence1 = 0, valence2 = 0;

	assert(bmesh_vert_in_edge(e, tv) != FALSE);

	ov = bmesh_edge_other_vert_get(e, tv);

	valence1 = bmesh_disk_count(ov);

	valence2 = bmesh_disk_count(tv);

	nv = BM_vert_create(tv->co);
	ne = BM_edge_create(nv, tv);

	bmesh_disk_edge_remove(ne, tv);
	bmesh_disk_edge_remove(ne, nv);

	/* remove e from tv's disk cycle */
	bmesh_disk_edge_remove(e, tv);

	/* swap out tv for nv in e */
	bmesh_edge_swapverts(e, tv, nv);

	/* add e to nv's disk cycle */
	bmesh_disk_edge_append(e, nv);

	/* add ne to nv's disk cycle */
	bmesh_disk_edge_append(ne, nv);

	/* add ne to tv's disk cycle */
	bmesh_disk_edge_append(ne, tv);

	/* verify disk cycle */
	edok = bmesh_disk_validate(valence1, ov->e, ov);
	BMESH_ASSERT(edok != FALSE);
	edok = bmesh_disk_validate(valence2, tv->e, tv);
	BMESH_ASSERT(edok != FALSE);
	edok = bmesh_disk_validate(2, nv->e, nv);
	BMESH_ASSERT(edok != FALSE);

	/* Split the radial cycle if present */
	nextl = e->l;
	e->l = NULL;
	if (nextl) {
		BMLoop *nl, *l;
		int radlen = bmesh_radial_length(nextl);
		int first1 = 0, first2 = 0;

		/* Take the next loop. Remove it from radial. Split it. Append to appropriate radials */
		while (nextl) {
			l = nextl;
			l->f->len++;
			nextl = nextl != nextl->radial_next ? nextl->radial_next : NULL;
			bmesh_radial_loop_remove(l, NULL);

			nl = bm_loop_create(NULL, NULL, l->f);
			nl->prev = l;
			nl->next = (l->next);
			nl->prev->next = nl;
			nl->next->prev = nl;
			nl->v = nv;

			/* assign the correct edge to the correct loop */
			if (bmesh_verts_in_edge(nl->v, nl->next->v, e)) {
				nl->e = e;
				l->e = ne;

				/* append l into ne's rad cycle */
				if (!first1) {
					first1 = 1;
					l->radial_next = l->radial_prev = NULL;
				}

				if (!first2) {
					first2 = 1;
					l->radial_next = l->radial_prev = NULL;
				}
				
				bmesh_radial_append(nl->e, nl);
				bmesh_radial_append(l->e, l);
			}
			else if (bmesh_verts_in_edge(nl->v, nl->next->v, ne)) {
				nl->e = ne;
				l->e = e;

				/* append l into ne's rad cycle */
				if (!first1) {
					first1 = 1;
					l->radial_next = l->radial_prev = NULL;
				}

				if (!first2) {
					first2 = 1;
					l->radial_next = l->radial_prev = NULL;
				}

				bmesh_radial_append(nl->e, nl);
				bmesh_radial_append(l->e, l);
			}

		}

		/* verify length of radial cycle */
		edok = bmesh_radial_validate(radlen, e->l);
		BMESH_ASSERT(edok != FALSE);
		edok = bmesh_radial_validate(radlen, ne->l);
		BMESH_ASSERT(edok != FALSE);

		/* verify loop->v and loop->next->v pointers for e */
		for (i = 0, l = e->l; i < radlen; i++, l = l->radial_next) {
			BMESH_ASSERT(l->e == e);
			//BMESH_ASSERT(l->radial_next == l);
			BMESH_ASSERT(!(l->prev->e != ne && l->next->e != ne));

			edok = bmesh_verts_in_edge(l->v, l->next->v, e);
			BMESH_ASSERT(edok != FALSE);
			BMESH_ASSERT(l->v != l->next->v);
			BMESH_ASSERT(l->e != l->next->e);
		}
		/* verify loop->v and loop->next->v pointers for ne */
		for (i = 0, l = ne->l; i < radlen; i++, l = l->radial_next) {
			BMESH_ASSERT(l->e == ne);
			// BMESH_ASSERT(l->radial_next == l);
			BMESH_ASSERT(!(l->prev->e != e && l->next->e != e));
			edok = bmesh_verts_in_edge(l->v, l->next->v, ne);
			BMESH_ASSERT(edok != FALSE);
			BMESH_ASSERT(l->v != l->next->v);
			BMESH_ASSERT(l->e != l->next->e);
		}
	}
	if (r_e) *r_e = ne;
	return nv;
}

/**
 * \brief Join Edge Kill Vert (JEKV)
 *
 * Takes an edge \a ke and pointer to one of its vertices \a kv
 * and collapses the edge on that vertex.
 *
 * \par Examples:
 *
 *     Before:         OE      KE
 *                   ------- -------
 *                   |     ||      |
 *                  OV     KV      TV
 *
 *
 *     After:              OE
 *                   ---------------
 *                   |             |
 *                  OV             TV
 *
 * \par Restrictions:
 * KV is a vertex that must have a valance of exactly two. Furthermore
 * both edges in KV's disk cycle (OE and KE) must be unique (no double edges).
 *
 * \return The resulting edge, NULL for failure.
 *
 * \note This euler has the possibility of creating
 * faces with just 2 edges. It is up to the caller to decide what to do with
 * these faces.
 */
BMEdge *BMesh::bmesh_jekv(BMEdge *ke, BMVert *kv, const short check_edge_double)
{
	BMEdge *oe;
	BMVert *ov, *tv;
	BMLoop *killoop, *l;
	int len, radlen = 0, halt = 0, i, valence1, valence2, edok;

	if (bmesh_vert_in_edge(ke, kv) == 0) {
		return NULL;
	}

	len = bmesh_disk_count(kv);
	
	if (len == 2) {
		oe = bmesh_disk_edge_next(ke, kv);
		tv = bmesh_edge_other_vert_get(ke, kv);
		ov = bmesh_edge_other_vert_get(oe, kv);
		halt = bmesh_verts_in_edge(kv, tv, oe); /* check for double edge */
		
		if (halt) {
			return NULL;
		}
		else {
			BMEdge *e_splice;

			/* For verification later, count valence of ov and t */
			valence1 = bmesh_disk_count(ov);
			valence2 = bmesh_disk_count(tv);

			if (check_edge_double) {
				e_splice = BM_edge_exists(tv, ov);
			}

			/* remove oe from kv's disk cycle */
			bmesh_disk_edge_remove(oe, kv);
			/* relink oe->kv to be oe->tv */
			bmesh_edge_swapverts(oe, kv, tv);
			/* append oe to tv's disk cycle */
			bmesh_disk_edge_append(oe, tv);
			/* remove ke from tv's disk cycle */
			bmesh_disk_edge_remove(ke, tv);

			/* deal with radial cycle of ke */
			radlen = bmesh_radial_length(ke->l);
			if (ke->l) {
				/* first step, fix the neighboring loops of all loops in ke's radial cycle */
				for (i = 0, killoop = ke->l; i < radlen; i++, killoop = killoop->radial_next) {
					/* relink loops and fix vertex pointer */
					if (killoop->next->v == kv) {
						killoop->next->v = tv;
					}

					killoop->next->prev = killoop->prev;
					killoop->prev->next = killoop->next;
					if (BM_FACE_FIRST_LOOP(killoop->f) == killoop) {
						BM_FACE_FIRST_LOOP(killoop->f) = killoop->next;
					}
					killoop->next = NULL;
					killoop->prev = NULL;

					/* fix len attribute of face */
					killoop->f->len--;
				}
				/* second step, remove all the hanging loops attached to ke */
				radlen = bmesh_radial_length(ke->l);

				if (LIKELY(radlen)) {
					BMLoop **loops = new BMLoop*[radlen];

					killoop = ke->l;

					/* this should be wrapped into a bme_free_radial function to be used by bmesh_KF as well... */
					for (i = 0; i < radlen; i++) {
						loops[i] = killoop;
						killoop = killoop->radial_next;
					}
					for (i = 0; i < radlen; i++) {
						this->totloop--;
						BLI_mempool_free(this->lpool, loops[i]);
					}
					delete loops;
				}

				/* Validate radial cycle of oe */
				edok = bmesh_radial_validate(radlen, oe->l);
				BMESH_ASSERT(edok != FALSE);
			}

			/* deallocate edg */
			bm_kill_only_edge(this, ke);

			/* deallocate verte */
			bm_kill_only_vert(this, kv);

			/* Validate disk cycle lengths of ov, tv are unchanged */
			edok = bmesh_disk_validate(valence1, ov->e, ov);
			BMESH_ASSERT(edok != FALSE);
			edok = bmesh_disk_validate(valence2, tv->e, tv);
			BMESH_ASSERT(edok != FALSE);

			/* Validate loop cycle of all faces attached to oe */
			for (i = 0, l = oe->l; i < radlen; i++, l = l->radial_next) {
				BMESH_ASSERT(l->e == oe);
				edok = bmesh_verts_in_edge(l->v, l->next->v, oe);
				BMESH_ASSERT(edok != FALSE);
				edok = bmesh_loop_validate(l->f);
				BMESH_ASSERT(edok != FALSE);
			}

			if (check_edge_double) {
				if (e_splice) {
					/* removes e_splice */
					BM_edge_splice(this, e_splice, oe);
				}
			}
			return oe;
		}
	}
	return NULL;
}

/**
 * \brief Split Face Make Edge (SFME)
 *
 * Takes as input two vertices in a single face. An edge is created which divides the original face
 * into two distinct regions. One of the regions is assigned to the original face and it is closed off.
 * The second region has a new face assigned to it.
 *
 * \par Examples:
 *
 *     Before:               After:
 *      +--------+           +--------+
 *      |        |           |        |
 *      |        |           |   f1   |
 *     v1   f1   v2          v1======v2
 *      |        |           |   f2   |
 *      |        |           |        |
 *      +--------+           +--------+
 *
 * \note the input vertices can be part of the same edge. This will
 * result in a two edged face. This is desirable for advanced construction
 * tools and particularly essential for edge bevel. Because of this it is
 * up to the caller to decide what to do with the extra edge.
 *
 * \note If \a holes is NULL, then both faces will lose
 * all holes from the original face.  Also, you cannot split between
 * a hole vert and a boundary vert; that case is handled by higher-
 * level wrapping functions (when holes are fully implemented, anyway).
 *
 * \note that holes represents which holes goes to the new face, and of
 * course this requires removing them from the existing face first, since
 * you cannot have linked list links inside multiple lists.
 *
 * \return A BMFace pointer
 */
BMFace *BMesh::bmesh_sfme(BMFace *f, BMVert *v1, BMVert *v2,BMLoop **r_l)
{
	BMFace *f2;
	BMLoop *l_iter, *l_first;
	BMLoop *v1loop = NULL, *v2loop = NULL, *f1loop = NULL, *f2loop = NULL;
	BMEdge *e;
	int i, len, f1len, f2len, first_loop_f1;

	/* verify that v1 and v2 are in face */
	len = f->len;
	for (i = 0, l_iter = BM_FACE_FIRST_LOOP(f); i < len; i++, l_iter = l_iter->next) {
		if (l_iter->v == v1) v1loop = l_iter;
		else if (l_iter->v == v2) v2loop = l_iter;
	}

	if (!v1loop || !v2loop) {
		return NULL;
	}

	/* allocate new edge between v1 and v2 */
	e = BM_edge_create(v1, v2);

	f2 = bm_face_create_internal();
	f1loop = bm_loop_create(v2, e, f);
	f2loop = bm_loop_create(v1, e, f2);

	f1loop->prev = v2loop->prev;
	f2loop->prev = v1loop->prev;
	v2loop->prev->next = f1loop;
	v1loop->prev->next = f2loop;

	f1loop->next = v1loop;
	f2loop->next = v2loop;
	v1loop->prev = f1loop;
	v2loop->prev = f2loop;

	/* find which of the faces the original first loop is in */
	l_iter = l_first = f1loop;
	first_loop_f1 = 0;
	do {
		if (l_iter == f->l_first)
			first_loop_f1 = 1;
	} while ((l_iter = l_iter->next) != l_first);

	if (first_loop_f1) {
		/* original first loop was in f1, find a suitable first loop for f2
		 * which is as similar as possible to f1. the order matters for tools
		 * such as duplifaces. */
		if (f->l_first->prev == f1loop)
			f2->l_first = f2loop->prev;
		else if (f->l_first->next == f1loop)
			f2->l_first = f2loop->next;
		else
			f2->l_first = f2loop;
	}
	else {
		/* original first loop was in f2, further do same as above */
		f2->l_first = f->l_first;

		if (f->l_first->prev == f2loop)
			f->l_first = f1loop->prev;
		else if (f->l_first->next == f2loop)
			f->l_first = f1loop->next;
		else
			f->l_first = f1loop;
	}

	/* validate both loop */
	/* I don't know how many loops are supposed to be in each face at this point! FIXME */

	/* go through all of f2's loops and make sure they point to it properly */
	l_iter = l_first = BM_FACE_FIRST_LOOP(f2);
	f2len = 0;
	do {
		l_iter->f = f2;
		f2len++;
	} while ((l_iter = l_iter->next) != l_first);

	/* link up the new loops into the new edges radial */
	bmesh_radial_append(e, f1loop);
	bmesh_radial_append(e, f2loop);

	f2->len = f2len;

	f1len = 0;
	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		f1len++;
	} while ((l_iter = l_iter->next) != l_first);

	f->len = f1len;

	if (r_l) *r_l = f2loop;
	
	return f2;
}

/**
 * \brief Face Split
 *
 * Split a face along two vertices. returns the newly made face, and sets
 * the \a r_l member to a loop in the newly created edge.
 *
 * \param bm The bmesh
 * \param f the original face
 * \param v1, v2 vertices which define the split edge, must be different
 * \param r_l pointer which will receive the BMLoop for the split edge in the new face
 * \param example Edge used for attributes of splitting edge, if non-NULL
 * \param nodouble Use an existing edge if found
 *
 * \return Pointer to the newly created face representing one side of the split
 * if the split is successful (and the original original face will be the
 * other side). NULL if the split fails.
 */
BMFace *BMesh::BM_face_split(BMFace *f, BMVert *v1, BMVert *v2, BMLoop **r_l)
{
	BMFace *nf;

	assert(v1 != v2);

	nf = bmesh_sfme(f, v1, v2, r_l);

	return nf;
}

/**
 * \brief Join Face Kill Edge (JFKE)
 *
 * Takes two faces joined by a single 2-manifold edge and fuses them together.
 * The edge shared by the faces must not be connected to any other edges which have
 * Both faces in its radial cycle
 *
 * \par Examples:
 *
 *           A                   B
 *      +--------+           +--------+
 *      |        |           |        |
 *      |   f1   |           |   f1   |
 *     v1========v2 = Ok!    v1==V2==v3 == Wrong!
 *      |   f2   |           |   f2   |
 *      |        |           |        |
 *      +--------+           +--------+
 *
 * In the example A, faces \a f1 and \a f2 are joined by a single edge,
 * and the euler can safely be used.
 * In example B however, \a f1 and \a f2 are joined by multiple edges and will produce an error.
 * The caller in this case should call #bmesh_jekv on the extra edges
 * before attempting to fuse \a f1 and \a f2.
 *
 * \note The order of arguments decides whether or not certain per-face attributes are present
 * in the resultant face. For instance vertex winding, material index, smooth flags, etc are inherited
 * from \a f1, not \a f2.
 *
 * \return A BMFace pointer
 */
BMFace *BMesh::bmesh_jfke(BMFace *f1, BMFace *f2, BMEdge *e)
{
	BMLoop *l_iter, *f1loop = NULL, *f2loop = NULL;
	int newlen = 0, i, f1len = 0, f2len = 0, edok;

	/* can't join a face to itself */
	if (f1 == f2) {
		return NULL;
	}

	/* validate that edge is 2-manifold edge */
	if (!BM_edge_is_manifold(e)) {
		return NULL;
	}

	/* verify that e is in both f1 and f2 */
	f1len = f1->len;
	f2len = f2->len;

	if (!((f1loop = BM_face_edge_share_loop(f1, e)) &&
	      (f2loop = BM_face_edge_share_loop(f2, e))))
	{
		return NULL;
	}

	/* validate direction of f2's loop cycle is compatible */
	if (f1loop->v == f2loop->v) {
		return NULL;
	}

	/* validate that for each face, each vertex has another edge in its disk cycle that is
	 * not e, and not shared. */
	if (bmesh_radial_face_find(f1loop->next->e, f2) ||
	    bmesh_radial_face_find(f1loop->prev->e, f2) ||
	    bmesh_radial_face_find(f2loop->next->e, f1) ||
	    bmesh_radial_face_find(f2loop->prev->e, f1) )
	{
		return NULL;
	}

	/* validate only one shared edge */
	if (BM_face_share_edge_count(f1, f2) > 1) {
		return NULL;
	}

	/* validate no internal join */
	for (i = 0, l_iter = BM_FACE_FIRST_LOOP(f1); i < f1len; i++, l_iter = l_iter->next) {
		l_iter->v->head.flag = 0;
	}
	for (i = 0, l_iter = BM_FACE_FIRST_LOOP(f2); i < f2len; i++, l_iter = l_iter->next) {
		l_iter->v->head.flag = 0;
	}

	for (i = 0, l_iter = BM_FACE_FIRST_LOOP(f1); i < f1len; i++, l_iter = l_iter->next) {
		if (l_iter != f1loop) {
			l_iter->v->head.flag = 1;
		}
	}
	for (i = 0, l_iter = BM_FACE_FIRST_LOOP(f2); i < f2len; i++, l_iter = l_iter->next) {
		if (l_iter != f2loop) {
			/* as soon as a duplicate is found, bail out */
			if (l_iter->v->head.flag==1) {
				return NULL;
			}
		}
	}

	/* join the two loop */
	f1loop->prev->next = f2loop->next;
	f2loop->next->prev = f1loop->prev;
	
	f1loop->next->prev = f2loop->prev;
	f2loop->prev->next = f1loop->next;
	
	/* if f1loop was baseloop, make f1loop->next the base. */
	if (BM_FACE_FIRST_LOOP(f1) == f1loop)
		BM_FACE_FIRST_LOOP(f1) = f1loop->next;

	/* increase length of f1 */
	f1->len += (f2->len - 2);

	/* make sure each loop points to the proper face */
	newlen = f1->len;
	for (i = 0, l_iter = BM_FACE_FIRST_LOOP(f1); i < newlen; i++, l_iter = l_iter->next)
		l_iter->f = f1;
	
	/* remove edge from the disk cycle of its two vertices */
	bmesh_disk_edge_remove(f1loop->e, f1loop->e->v1);
	bmesh_disk_edge_remove(f1loop->e, f1loop->e->v2);
	
	/* deallocate edge and its two loops as well as f2 */
	BLI_mempool_free(this->epool, f1loop->e);
	this->totedge--;
	BLI_mempool_free(this->lpool, f1loop);
	this->totloop--;
	BLI_mempool_free(this->lpool, f2loop);
	this->totloop--;
	BLI_mempool_free(this->fpool, f2);
	this->totface--;

	/* validate the new loop cycle */
	edok = bmesh_loop_validate(f1);
	BMESH_ASSERT(edok != FALSE);
	
	return f1;
}

/************************************************************************/
/* static functions
/************************************************************************/

BMFace *BMesh::BM_faces_join(BMesh *bm, BMFace **faces, int totface, const short do_del)
{
//	BMFace *f, *newf;
//	BMLoop *l_iter;
//	BMLoop *l_first;
//	BMVert *v1 = NULL, *v2 = NULL;
//	vector<BMEdge*> edges;  
//	vector<BMEdge*> deledges;
//	vector<BMVert*> delverts;
//	edges.reserve(BM_NGON_STACK_SIZE);
//	deledges.reserve(BM_NGON_STACK_SIZE);
//	delverts.reserve(BM_NGON_STACK_SIZE);
//	const char *err = NULL;
//	int i, tote = 0;
//
//	if (UNLIKELY(!totface)) {
//		BMESH_ASSERT(0);
//		return NULL;
//	}
//
//	if (totface == 1)
//		return faces[0];
//
//	for (i = 0; i < totface; i++)
//		faces[i]->head.flag = 1;
//
//	for (i = 0; i < totface; i++) {
//		f = faces[i];
//		l_iter = l_first = BM_FACE_FIRST_LOOP(f);
//		do {
//			int rlen = 0;
//			BMLoop *l2 = l_iter;
//			do {
//				if (UNLIKELY(!l2)) {
//					BMESH_ASSERT(0);
//					rlen=0;
//					break;
//				}
//				l2 = l2->radial_next;
//				if (UNLIKELY(rlen >= BM_LOOP_RADIAL_MAX)) {
//					BMESH_ASSERT(0);
//					rlen = 0;
//					break;
//				}
//				rlen+= (l2->f->head.flag==1);
//			} while (l2 != l_iter);
//
//			if (rlen > 2) {
//				err = "Input faces do not form a contiguous manifold region";
//				goto error;
//			}
//			else if (rlen == 1) {
//				edges.push_back(l_iter->e);
//
//				if (!v1) {
//					v1 = l_iter->v;
//					v2 = BM_edge_other_vert(l_iter->e, l_iter->v);
//				}
//				tote++;
//			}
//			else if (rlen == 2) {
//				int d1, d2;
//
//				d1 = 0;
//				d2 = 0;
//
//				/* don't remove an edge it makes up the side of another face
//					* else this will remove the face as well - campbell */
//				if (BM_edge_face_count(l_iter->e) <= 2) {
//					if (do_del) {
//						deledges.push_back(l_iter->e);
//					}
//				}
//				
//				if (d1) {
//					if (do_del) {
//						delverts.push_back(l_iter->e->v1);
//					}
//				}
//
//				if (d2) {
//					if (do_del) {
//						delverts.push_back(l_iter->e->v2);
//					}
//				}
//			}
//		} while ((l_iter = l_iter->next) != l_first);
//	}
//
//	/* create region face */
//	if(edges.size() > 0)
//		newf = bm->BM_face_create_ngon(v1, v2, &edges[0], tote);
//	else
//		newf = 0;
//	if (!newf) {
//		err = "Invalid boundary region to join faces";
//		goto error;
//	}
//
//	/* copy over loop data */
//	l_iter = l_first = BM_FACE_FIRST_LOOP(newf);
//	do {
//		BMLoop *l2 = l_iter->radial_next;
//
//		do {
//			l2 = l2->radial_next;
//		} while (l2 != l_iter);
//
//		if (l2 != l_iter) {
//			/* I think this is correct */
//			if (l2->v != l_iter->v) {
//				l2 = l2->next;
//			}
//
//		}
//	} while ((l_iter = l_iter->next) != l_first);
//	
//
//
//
//	/* update loop face pointer */
//	l_iter = l_first = BM_FACE_FIRST_LOOP(newf);
//	do {
//		l_iter->f = newf;
//	} while ((l_iter = l_iter->next) != l_first);
//
//	/* delete old geometry */
//	if (do_del) {
//		for (unsigned int i = 0; i < deledges.size(); i++) {
//			BM_edge_kill(bm, deledges[i]);
//		}
//
//		for (unsigned int i = 0; i < delverts.size(); i++) {
//			BM_vert_kill(bm, delverts[i]);
//		}
//	}
//	else {
//		/* otherwise we get both old and new faces */
//		for (int i = 0; i < totface; i++) {
//			BM_face_kill(bm, faces[i]);
//		}
//	}
//
//	return newf;
//error:
//	printf("%s: %s\n", __func__, err);
	return 0;
}

bool BMesh::bmesh_disk_edge_append(BMEdge* e, BMVert* v)
{
	if (!v->e) {
		BMDiskLink *dl1 = BM_DISK_EDGE_LINK_GET(e, v);

		v->e = e;
		dl1->next = dl1->prev = e;
	}
	else {
		BMDiskLink *dl1, *dl2, *dl3;

		dl1 = BM_DISK_EDGE_LINK_GET(e, v);
		dl2 = BM_DISK_EDGE_LINK_GET(v->e, v);
		dl3 = dl2->prev ? BM_DISK_EDGE_LINK_GET(dl2->prev, v) : NULL;

		dl1->next = v->e;
		dl1->prev = dl2->prev;

		dl2->prev = e;
		if (dl3)
			dl3->next = e;
	}

	return true;
}

void BMesh::bmesh_radial_append(BMEdge* e, BMLoop* l)
{
	if (e->l == 0)
	{
		e->l = l;
		l->radial_next = l->radial_prev = l;
	}
	else
	{
		l->radial_prev = e->l;
		l->radial_next = e->l->radial_next;

		e->l->radial_next->radial_prev = l;
		e->l->radial_next = l;

		e->l = l;
	}

	if (l->e && l->e != e)
	{
		assert(0);
	}

	l->e = e;
}

int BMesh::bmesh_radial_length(const BMLoop *l)
{
	const BMLoop *l_iter = l;
	int i = 0;

	if (!l)
		return 0;

	do {
		if (!l_iter) {
			/* radial cycle is broken (not a circulat loop) */
			assert(0);
			return 0;
		}

		i++;
		if (i >= BM_LOOP_RADIAL_MAX) {
			assert(0);
			return -1;
		}
	} while ((l_iter = l_iter->radial_next) != l);

	return i;
}

/**
 * \brief Next Disk Edge
 *
 *	Find the next edge in a disk cycle
 *
 *	\return Pointer to the next edge in the disk cycle for the vertex v.
 */
BMEdge *BMesh::bmesh_disk_edge_next(const BMEdge *e, const BMVert *v)
{
	if (v == e->v1)
		return e->v1_disk_link.next;
	if (v == e->v2)
		return e->v2_disk_link.next;
	return NULL;
}

BMEdge *BMesh::bmesh_disk_edge_prev(const BMEdge *e, const BMVert *v)
{
	if (v == e->v1)
		return e->v1_disk_link.prev;
	if (v == e->v2)
		return e->v2_disk_link.prev;
	return NULL;
}

BMEdge *BMesh::bmesh_disk_edge_exists(const BMVert *v1, const BMVert *v2)
{
	BMEdge *e_iter, *e_first;
	
	if (v1->e) {
		e_first = e_iter = v1->e;

		do {
			if (bmesh_verts_in_edge(v1, v2, e_iter)) {
				return e_iter;
			}
		} while ((e_iter = bmesh_disk_edge_next(e_iter, v1)) != e_first);
	}
	
	return NULL;
}

int BMesh::bmesh_disk_count(const BMVert *v)
{
	if (v->e) {
		BMEdge *e_first, *e_iter;
		int count = 0;

		e_iter = e_first = v->e;

		do {
			if (!e_iter) {
				return 0;
			}

			if (count >= (1 << 20)) {
				printf("bmesh error: infinite loop in disk cycle!\n");
				return 0;
			}
			count++;
		} while ((e_iter = bmesh_disk_edge_next(e_iter, v)) != e_first);
		return count;
	}
	else {
		return 0;
	}
}

int BMesh::bmesh_disk_validate(int len, const BMEdge *e, const BMVert *v)
{
	const BMEdge *e_iter;

	if (!BM_vert_in_edge(e, v))
		return 0;
	if (bmesh_disk_count(v) != len || len == 0)
		return 0;

	e_iter = e;
	do {
		if (len != 1 && bmesh_disk_edge_prev(e_iter, v) == e_iter) {
			return 0;
		}
	} while ((e_iter = bmesh_disk_edge_next(e_iter, v)) != e);

	return 0;
}

/**
 *	MISC utility functions.
 */

int BMesh::bmesh_vert_in_edge(const BMEdge *e, const BMVert *v)
{
	if (e->v1 == v || e->v2 == v) return 1;
	return 0;
}
int BMesh::bmesh_verts_in_edge(const BMVert *v1, const BMVert *v2, const BMEdge *e)
{
	if (e->v1 == v1 && e->v2 == v2) return 1;
	else if (e->v1 == v2 && e->v2 == v1) return 1;
	return 0;
}

BMVert *BMesh::bmesh_edge_other_vert_get(BMEdge *e, const BMVert *v)
{
	if (e->v1 == v) {
		return e->v2;
	}
	else if (e->v2 == v) {
		return e->v1;
	}
	return 0;
}

int BMesh::bmesh_edge_swapverts(BMEdge *e, BMVert *orig, BMVert *newv)
{
	if (e->v1 == orig) {
		e->v1 = newv;
		e->v1_disk_link.next = e->v1_disk_link.prev = 0;
		return 1;
	}
	else if (e->v2 == orig) {
		e->v2 = newv;
		e->v2_disk_link.next = e->v2_disk_link.prev = 0;
		return 1;
	}
	return 0;
}

void BMesh::bmesh_disk_edge_remove(BMEdge *e, BMVert *v)
{
	BMDiskLink *dl1, *dl2;

	dl1 = BM_DISK_EDGE_LINK_GET(e, v);
	if (dl1->prev) {
		dl2 = BM_DISK_EDGE_LINK_GET(dl1->prev, v);
		dl2->next = dl1->next;
	}

	if (dl1->next) {
		dl2 = BM_DISK_EDGE_LINK_GET(dl1->next, v);
		dl2->prev = dl1->prev;
	}

	if (v->e == e)
		v->e = (e != dl1->next) ? dl1->next : NULL;

	dl1->next = dl1->prev = NULL;
}

/**
 * \brief DISK COUNT FACE VERT
 *
 * Counts the number of loop users
 * for this vertex. Note that this is
 * equivalent to counting the number of
 * faces incident upon this vertex
 */
int BMesh::bmesh_disk_facevert_count(const BMVert *v)
{
	/* is there an edge on this vert at all */
	if (v->e) {
		BMEdge *e_first, *e_iter;
		int count = 0;

		/* first, loop around edge */
		e_first = e_iter = v->e;
		do {
			if (e_iter->l) {
				count += bmesh_radial_facevert_count(e_iter->l, v);
			}
		} while ((e_iter = bmesh_disk_edge_next(e_iter, v)) != e_first);
		return count;
	}
	else {
		return 0;
	}
}

/**
 * \brief FIND FIRST FACE EDGE
 *
 * Finds the first edge in a vertices
 * Disk cycle that has one of this
 * vert's loops attached
 * to it.
 */
BMEdge *BMesh::bmesh_disk_faceedge_find_first(BMEdge *e, const BMVert *v)
{
	BMEdge *searchedge = NULL;
	searchedge = e;
	do {
		if (searchedge->l && bmesh_radial_facevert_count(searchedge->l, v)) {
			return searchedge;
		}
	} while ((searchedge = bmesh_disk_edge_next(searchedge, v)) != e);

	return NULL;
}

BMEdge *BMesh::bmesh_disk_faceedge_find_next(BMEdge *e, const BMVert *v)
{
	BMEdge *searchedge = NULL;
	searchedge = bmesh_disk_edge_next(e, v);
	do {
		if (searchedge->l && bmesh_radial_facevert_count(searchedge->l, v)) {
			return searchedge;
		}
	} while ((searchedge = bmesh_disk_edge_next(searchedge, v)) != e);
	return e;
}

/*****radial cycle functions, e.g. loops surrounding edges**** */
int BMesh::bmesh_radial_validate(int radlen, const BMLoop *l)
{
	const BMLoop *l_iter = l;
	int i = 0;
	
	if (bmesh_radial_length(l) != radlen)
		return FALSE;

	do {
		if (UNLIKELY(!l_iter)) {
			BMESH_ASSERT(0);
			return FALSE;
		}
		
		if (l_iter->e != l->e)
			return FALSE;
		if (l_iter->v != l->e->v1 && l_iter->v != l->e->v2)
			return FALSE;
		
		if (UNLIKELY(i > BM_LOOP_RADIAL_MAX)) {
			BMESH_ASSERT(0);
			return FALSE;
		}
		
		i++;
	} while ((l_iter = l_iter->radial_next) != l);

	return TRUE;
}

/**
 * \brief BMESH RADIAL REMOVE LOOP
 *
 * Removes a loop from an radial cycle. If edge e is non-NULL
 * it should contain the radial cycle, and it will also get
 * updated (in the case that the edge's link into the radial
 * cycle was the loop which is being removed from the cycle).
 */
void BMesh::bmesh_radial_loop_remove(BMLoop *l, BMEdge *e)
{
	/* if e is non-NULL, l must be in the radial cycle of e */
	if (UNLIKELY(e && e != l->e)) {
		assert(0);
	}

	if (l->radial_next != l) {
		if (e && l == e->l)
			e->l = l->radial_next;

		l->radial_next->radial_prev = l->radial_prev;
		l->radial_prev->radial_next = l->radial_next;
	}
	else {
		if (e) {
			if (l == e->l) {
				e->l = NULL;
			}
			else {
				BMESH_ASSERT(0);
			}
		}
	}

	/* l is no longer in a radial cycle; empty the links
	 * to the cycle and the link back to an edge */
	l->radial_next = l->radial_prev = NULL;
	l->e = NULL;
}


/**
 * \brief BME RADIAL FIND FIRST FACE VERT
 *
 * Finds the first loop of v around radial
 * cycle
 */
BMLoop *BMesh::bmesh_radial_faceloop_find_first(BMLoop *l, const BMVert *v)
{
	BMLoop *l_iter;
	l_iter = l;
	do {
		if (l_iter->v == v) {
			return l_iter;
		}
	} while ((l_iter = l_iter->radial_next) != l);
	return NULL;
}

BMLoop *BMesh::bmesh_radial_faceloop_find_next(BMLoop *l, const BMVert *v)
{
	BMLoop *l_iter;
	l_iter = l->radial_next;
	do {
		if (l_iter->v == v) {
			return l_iter;
		}
	} while ((l_iter = l_iter->radial_next) != l);
	return l;
}

int BMesh::bmesh_radial_face_find(const BMEdge *e, const BMFace *f)
{
	const BMLoop *l_iter;
	int i, len;

	len = bmesh_radial_length(e->l);
	for (i = 0, l_iter = e->l; i < len; i++, l_iter = l_iter->radial_next) {
		if (l_iter->f == f)
			return TRUE;
	}
	return FALSE;
}

/**
 * \brief RADIAL COUNT FACE VERT
 *
 * Returns the number of times a vertex appears
 * in a radial cycle
 */
int BMesh::bmesh_radial_facevert_count(const BMLoop *l, const BMVert *v)
{
	const BMLoop *l_iter=l;
	int count = 0;
	do {
		if (l_iter->v == v) {
			count++;
		}
	} while ((l_iter = l_iter->radial_next) != l);

	return count;
}

/*****loop cycle functions, e.g. loops surrounding a face**** */
int BMesh::bmesh_loop_validate(const BMFace *f)
{
	int i;
	int len = f->len;
	BMLoop *l_iter, *l_first;

	l_first = BM_FACE_FIRST_LOOP(f);

	if (l_first == NULL) {
		return FALSE;
	}

	/* Validate that the face loop cycle is the length specified by f->len */
	for (i = 1, l_iter = l_first->next; i < len; i++, l_iter = l_iter->next) {
		if ((l_iter->f != f) ||
		    (l_iter == l_first))
		{
			return FALSE;
		}
	}
	if (l_iter != l_first) {
		return FALSE;
	}

	/* Validate the loop->prev links also form a cycle of length f->len */
	for (i = 1, l_iter = l_first->prev; i < len; i++, l_iter = l_iter->prev) {
		if (l_iter == l_first) {
			return FALSE;
		}
	}
	if (l_iter != l_first) {
		return FALSE;
	}

	return TRUE;
}

int BMesh::bm_loop_length(BMLoop *l)
{
	BMLoop *l_first = l;
	int i = 0;

	do {
		i++;
	} while ((l = l->next) != l_first);

	return i;
}


int	BMesh::bmesh_loop_reverse(BMesh* bm,BMFace* f)
{
	BMLoop *l_first = f->l_first;

	BMLoop *l_iter, *oldprev, *oldnext;
	vector<BMEdge*> edar;
	edar.reserve(BM_NGON_STACK_SIZE);

	int i, j, edok, len = 0;

	len = bm_loop_length(l_first);

	for (i = 0, l_iter = l_first; i < len; i++, l_iter = l_iter->next) {
		BMEdge *curedge = l_iter->e;
		bmesh_radial_loop_remove(l_iter, curedge);
		edar.push_back(curedge);
	}

	/* actually reverse the loop */
	for (i = 0, l_iter = l_first; i < len; i++) {
		oldnext = l_iter->next;
		oldprev = l_iter->prev;
		l_iter->next = oldprev;
		l_iter->prev = oldnext;
		l_iter = oldnext;
	}

	if (len == 2) { /* two edged face */
		/* do some verification here! */
		l_first->e = edar[1];
		l_first->next->e = edar[0];
	}
	else {
		for (i = 0, l_iter = l_first; i < len; i++, l_iter = l_iter->next) {
			edok = 0;
			for (j = 0; j < len; j++) {
				edok = bmesh_verts_in_edge(l_iter->v, l_iter->next->v, edar[j]);
				if (edok) {
					l_iter->e = edar[j];
					break;
				}
			}
		}
	}
	/* rebuild radia */
	for (i = 0, l_iter = l_first; i < len; i++, l_iter = l_iter->next)
		bmesh_radial_append(l_iter->e, l_iter);

	return 1;
}


#define BM_OVERLAP (1 << 13)

/**
 * Returns whether or not a given vertex is
 * is part of a given edge.
 */
int BMesh::BM_vert_in_edge(const BMEdge *e, const BMVert *v)
{
	return bmesh_vert_in_edge(e, v);
}

/**
 * \brief Other Loop in Face Sharing an Edge
 *
 * Finds the other loop that shares \a v with \a e loop in \a f.
 *
 *     +----------+
 *     |          |
 *     |    f     |
 *     |          |
 *     +----------+ <-- return the face loop of this vertex.
 *     v --> e
 *     ^     ^ <------- These vert args define direction
 *                      in the face to check.
 *                      The faces loop direction is ignored.
 *
 */
BMLoop *BMesh::BM_face_other_edge_loop(BMFace *f, BMEdge *e, BMVert *v)
{
	BMLoop *l_iter;
	BMLoop *l_first;

	/* we could loop around the face too, but turns out this uses a lot
	 * more iterations (approx double with quads, many more with 5+ ngons) */
	l_iter = l_first = e->l;

	do {
		if (l_iter->e == e && l_iter->f == f) {
			break;
		}
	} while ((l_iter = l_iter->radial_next) != l_first);
	
	return l_iter->v == v ? l_iter->prev : l_iter->next;
}

/**
 * \brief Other Loop in Face Sharing a Vertex
 *
 * Finds the other loop in a face.
 *
 * This function returns a loop in \a f that shares an edge with \a v
 * The direction is defined by \a v_prev, where the return value is
 * the loop of what would be 'v_next'
 *
 *
 *     +----------+ <-- return the face loop of this vertex.
 *     |          |
 *     |    f     |
 *     |          |
 *     +----------+
 *     v_prev --> v
 *     ^^^^^^     ^ <-- These vert args define direction
 *                      in the face to check.
 *                      The faces loop direction is ignored.
 *
 * \note \a v_prev and \a v _implicitly_ define an edge.
 */
BMLoop *BMesh::BM_face_other_vert_loop(BMFace *f, BMVert *v_prev, BMVert *v)
{
	BMIter liter;
	BMLoop *l_iter;

	assert(BM_edge_exists(v_prev, v) != NULL);
	liter.init(f);
	for (l_iter = lofv_begin(liter); l_iter!=lofv_end(liter); l_iter=lofv_next(liter))
	{
		if(l_iter->f == f) break;
	}

	if (l_iter) {
		if (l_iter->prev->v == v_prev) {
			return l_iter->next;
		}
		else if (l_iter->next->v == v_prev) {
			return l_iter->prev;
		}
		else {
			/* invalid args */
			assert(0);
			return NULL;
		}
	}
	else {
		/* invalid args */
		assert(0);
		return NULL;
	}
}

/**
 * \brief Other Loop in Face Sharing a Vert
 *
 * Finds the other loop that shares \a v with \a e loop in \a f.
 *
 *     +----------+ <-- return the face loop of this vertex.
 *     |          |
 *     |          |
 *     |          |
 *     +----------+ <-- This vertex defines the direction.
 *           l    v
 *           ^ <------- This loop defines both the face to search
 *                      and the edge, in combination with 'v'
 *                      The faces loop direction is ignored.
 */

BMLoop *BMesh::BM_loop_other_vert_loop(BMLoop *l, BMVert *v)
{
	BMEdge *e = l->e;
	BMVert *v_prev = BM_edge_other_vert(e, v);
	if (l->v == v) {
		if (l->prev->v == v_prev) {
			return l->next;
		}
		else {
			assert(l->next->v == v_prev);

			return l->prev;
		}
	}
	else {
		assert(l->v == v_prev);

		if (l->prev->v == v) {
			return l->prev->prev;
		}
		else {
			assert(l->next->v == v);
			return l->next->next;
		}
	}
}

/**
 * Returns TRUE if the vertex is used in a given face.
 */

int BMesh::BM_vert_in_face(BMFace *f, BMVert *v)
{
	BMLoop *l_iter, *l_first;
	l_iter = l_first = f->l_first;
	do {
		if (l_iter->v == v) {
			return TRUE;
		}
	} while ((l_iter = l_iter->next) != l_first);

	return FALSE;
}

/**
 * Compares the number of vertices in an array
 * that appear in a given face
 */
int BMesh::BM_verts_in_face(BMFace *f, BMVert **varr, int len)
{
	BMLoop *l_iter, *l_first;

	int count = 0;

	l_iter = l_first = f->l_first;
	do {
		for (int i=0; i<len; i++)
		{
			if (varr[i] == l_iter->v)
			{
				count ++;
			}
		}

	} while ((l_iter = l_iter->next) != l_first);

	return count;
}

/**
 * Returns whether or not a given edge is is part of a given face.
 */
int BMesh::BM_edge_in_face(BMFace *f, BMEdge *e)
{
	BMLoop *l_iter;
	BMLoop *l_first;

	l_iter = l_first = f->l_first;

	do {
		if (l_iter->e == e) {
			return TRUE;
		}
	} while ((l_iter = l_iter->next) != l_first);

	return FALSE;
}

/**
 * Returns whether or not two vertices are in
 * a given edge
 */
int BMesh::BM_verts_in_edge(BMVert *v1, BMVert *v2, BMEdge *e)
{
	return bmesh_verts_in_edge(v1, v2, e);
}

/**
 * Given a edge and one of its vertices, returns
 * the other vertex.
 */
BMVert *BMesh::BM_edge_other_vert(BMEdge *e, BMVert *v)
{
	return bmesh_edge_other_vert_get(e, v);
}

/**
 * The function takes a vertex at the center of a fan and returns the opposite edge in the fan.
 * All edges in the fan must be manifold, otherwise return NULL.
 *
 * \note This could (probably) be done more effieiently.
 */
BMEdge *BMesh::BM_vert_other_disk_edge(BMVert *v, BMEdge *e_first)
{
	BMLoop *l_a;
	int tot = 0;
	int i;

	assert(BM_vert_in_edge(e_first, v));

	l_a = e_first->l;
	do {
		l_a = BM_loop_other_vert_loop(l_a, v);
		l_a = BM_vert_in_edge(l_a->e, v) ? l_a : l_a->prev;
		if (BM_edge_is_manifold(l_a->e)) {
			l_a = l_a->radial_next;
		}
		else {
			return NULL;
		}

		tot++;
	} while (l_a != e_first->l);

	/* we know the total, now loop half way */
	tot /= 2;
	i = 0;

	l_a = e_first->l;
	do {
		if (i == tot) {
			l_a = BM_vert_in_edge(l_a->e, v) ? l_a : l_a->prev;
			return l_a->e;
		}

		l_a = BM_loop_other_vert_loop(l_a, v);
		l_a = BM_vert_in_edge(l_a->e, v) ? l_a : l_a->prev;
		if (BM_edge_is_manifold(l_a->e)) {
			l_a = l_a->radial_next;
		}
		/* this wont have changed from the previous loop */


		i++;
	} while (l_a != e_first->l);

	return NULL;
}

/**
 * Returms edge length
 */
float BMesh::BM_edge_calc_length(BMEdge *e)
{
	return len_v3v3(e->v1->co, e->v2->co);
}

/**
 * Utility function, since enough times we have an edge
 * and want to access 2 connected faces.
 *
 * \return TRUE when only 2 faces are found.
 */
int BMesh::BM_edge_face_pair(BMEdge *e, BMFace **r_fa, BMFace **r_fb)
{
	BMLoop *la, *lb;

	if ((la = e->l) &&
	    (lb = la->radial_next) &&
	    (lb->radial_next == la))
	{
		*r_fa = la->f;
		*r_fb = lb->f;
		return TRUE;
	}
	else {
		*r_fa = NULL;
		*r_fb = NULL;
		return FALSE;
	}
}

/**
 * Utility function, since enough times we have an edge
 * and want to access 2 connected loops.
 *
 * \return TRUE when only 2 faces are found.
 */
int BMesh::BM_edge_loop_pair(BMEdge *e, BMLoop **r_la, BMLoop **r_lb)
{
	BMLoop *la, *lb;

	if ((la = e->l) &&
	    (lb = la->radial_next) &&
	    (lb->radial_next == la))
	{
		*r_la = la;
		*r_lb = lb;
		return TRUE;
	}
	else {
		*r_la = NULL;
		*r_lb = NULL;
		return FALSE;
	}
}

/**
 *	Returns the number of edges around this vertex.
 */
int BMesh::BM_vert_edge_count(BMVert *v)
{
	return bmesh_disk_count(v);
}

int BMesh::BM_vert_edge_count_nonwire(BMVert *v)
{
	int count = 0;
	BMIter eiter;
	BMEdge *edge;
	eiter.init(v);
	for (edge=eofv_begin(eiter); edge!=eofv_end(eiter); edge=eofv_next(eiter))
	{
		if(edge->l)	count++;
	}
	return count;
}
/**
 *	Returns the number of faces around this edge
 */
int BMesh::BM_edge_face_count(BMEdge *e)
{
	int count = 0;

	if (e->l) {
		BMLoop *l_iter;
		BMLoop *l_first;

		l_iter = l_first = e->l;

		do {
			count++;
		} while ((l_iter = l_iter->radial_next) != l_first);
	}

	return count;
}

/**
 *	Returns the number of faces around this vert
 */
int BMesh::BM_vert_face_count(BMVert *v)
{
	int count = 0;
	BMLoop *l;
	BMIter iter;
	iter.init(v);
	for(l=lofv_begin(iter); l!= lofv_end(iter); l=lofv_next(iter))
		count++;
	return count;
}

/**
 * Tests whether or not the vertex is part of a wire edge.
 * (ie: has no faces attached to it)
 */
int BMesh::BM_vert_is_wire(BMVert *v)
{
	BMEdge *curedge;

	if (v->e == NULL) {
		return FALSE;
	}
	
	curedge = v->e;
	do {
		if (curedge->l) {
			return FALSE;
		}

		curedge = bmesh_disk_edge_next(curedge, v);
	} while (curedge != v->e);

	return TRUE;
}

/**
 * Tests whether or not the edge is part of a wire.
 * (ie: has no faces attached to it)
 */
int BMesh::BM_edge_is_wire(BMEdge *e)
{
	return (e->l) ? FALSE : TRUE;
}

/**
 * A vertex is non-manifold if it meets the following conditions:
 * 1: Loose - (has no edges/faces incident upon it).
 * 2: Joins two distinct regions - (two pyramids joined at the tip).
 * 3: Is part of a an edge with more than 2 faces.
 * 4: Is part of a wire edge.
 */
int BMesh::BM_vert_is_manifold(BMVert *v)
{
	BMEdge *e, *oe;
	BMLoop *l;
	int len, count, flag;

	if (v->e == NULL) {
		/* loose vert */
		return FALSE;
	}

	/* count edges while looking for non-manifold edges */
	len = 0;
	oe = e = v->e;
	do {
		/* loose edge or edge shared by more than two faces,
		 * edges with 1 face user are OK, otherwise we could
		 * use BM_edge_is_manifold() here */
		if (e->l == NULL || bmesh_radial_length(e->l) > 2) {
			return FALSE;
		}
		len++;
	} while ((e = bmesh_disk_edge_next(e, v)) != oe);

	count = 1;
	flag = 1;
	e = NULL;
	oe = v->e;
	l = oe->l;
	while (e != oe) {
		l = (l->v == v) ? l->prev : l->next;
		e = l->e;
		count++; /* count the edges */

		if (flag && l->radial_next == l) {
			/* we've hit the edge of an open mesh, reset once */
			flag = 0;
			count = 1;
			oe = e;
			e = NULL;
			l = oe->l;
		}
		else if (l->radial_next == l) {
			/* break the loop */
			e = oe;
		}
		else {
			l = l->radial_next;
		}
	}

	if (count < len) {
		/* vert shared by multiple regions */
		return FALSE;
	}

	return TRUE;
}

/**
 * Tests whether or not this edge is manifold.
 * A manifold edge has exactly 2 faces attached to it.
 */

#if 1 /* fast path for checking manifold */
int BMesh::BM_edge_is_manifold(BMEdge *e)
{
	const BMLoop *l = e->l;
	return (l && (l->radial_next != l) &&             /* not 0 or 1 face users */
	             (l->radial_next->radial_next == l)); /* 2 face users */
}
#else
int BM_edge_is_manifold(BMEdge *e)
{
	int count = BM_edge_face_count(e);
	if (count == 2) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}
#endif

/**
 * Tests whether or not an edge is on the boundary
 * of a shell (has one face associated with it)
 */

#if 1 /* fast path for checking boundary */
int BMesh::BM_edge_is_boundary(BMEdge *e)
{
	const BMLoop *l = e->l;
	return (l && (l->radial_next == l));
}
#else
int BM_edge_is_boundary(BMEdge *e)
{
	int count = BM_edge_face_count(e);
	if (count == 1) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}
#endif

/**
 *  Counts the number of edges two faces share (if any)
 */
int BMesh::BM_face_share_edge_count(BMFace *f1, BMFace *f2)
{
	BMLoop *l_iter;
	BMLoop *l_first;
	int count = 0;
	
	l_iter = l_first = BM_FACE_FIRST_LOOP(f1);
	do {
		if (bmesh_radial_face_find(l_iter->e, f2)) {
			count++;
		}
	} while ((l_iter = l_iter->next) != l_first);

	return count;
}

/**
 *	Test if e1 shares any faces with e2
 */
int BMesh::BM_edge_share_face_count(BMEdge *e1, BMEdge *e2)
{
	BMLoop *l;
	BMFace *f;

	if (e1->l && e2->l) {
		l = e1->l;
		do {
			f = l->f;
			if (bmesh_radial_face_find(e2, f)) {
				return TRUE;
			}
			l = l->radial_next;
		} while (l != e1->l);
	}
	return FALSE;
}

/**
 *	Tests to see if e1 shares a vertex with e2
 */
int BMesh::BM_edge_share_vert_count(BMEdge *e1, BMEdge *e2)
{
	return (e1->v1 == e2->v1 ||
	        e1->v1 == e2->v2 ||
	        e1->v2 == e2->v1 ||
	        e1->v2 == e2->v2);
}

/**
 *	Return the shared vertex between the two edges or NULL
 */
BMVert *BMesh::BM_edge_share_vert(BMEdge *e1, BMEdge *e2)
{
	if (BM_vert_in_edge(e2, e1->v1)) {
		return e1->v1;
	}
	else if (BM_vert_in_edge(e2, e1->v2)) {
		return e1->v2;
	}
	else {
		return NULL;
	}
}

/**
 * \brief Return the Loop Shared by Face and Vertex
 *
 * Finds the loop used which uses \a v in face loop \a l
 *
 * \note currenly this just uses simple loop in future may be speeded up
 * using radial vars
 */
BMLoop *BMesh::BM_face_vert_share_loop(BMFace *f, BMVert *v)
{
	BMLoop *l_first;
	BMLoop *l_iter;

	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		if (l_iter->v == v) {
			return l_iter;
		}
	} while ((l_iter = l_iter->next) != l_first);

	return NULL;
}

/**
 * \brief Return the Loop Shared by Face and Edge
 *
 * Finds the loop used which uses \a e in face loop \a l
 *
 * \note currenly this just uses simple loop in future may be speeded up
 * using radial vars
 */
BMLoop *BMesh::BM_face_edge_share_loop(BMFace *f, BMEdge *e)
{
	BMLoop *l_first;
	BMLoop *l_iter;

	l_iter = l_first = e->l;
	do {
		if (l_iter->f == f) {
			return l_iter;
		}
	} while ((l_iter = l_iter->radial_next) != l_first);

	return NULL;
}

/**
 * Returns the verts of an edge as used in a face
 * if used in a face at all, otherwise just assign as used in the edge.
 *
 * Useful to get a deterministic winding order when calling
 * BM_face_create_ngon() on an arbitrary array of verts,
 * though be sure to pick an edge which has a face.
 *
 * \note This is infact quite a simple check, mainly include this function so the intent is more obvious.
 * We know these 2 verts will _always_ make up the loops edge
 */
void BMesh::BM_edge_ordered_verts_ex(BMEdge *edge, BMVert **r_v1, BMVert **r_v2,
                              BMLoop *edge_loop)
{
	assert(edge_loop->e == edge);
	*r_v1 = edge_loop->v;
	*r_v2 = edge_loop->next->v;
}

void BMesh::BM_edge_ordered_verts(BMEdge *edge, BMVert **r_v1, BMVert **r_v2)
{
	BM_edge_ordered_verts_ex(edge, r_v1, r_v2, edge->l);
}

/**
 * Calculates the angle between the previous and next loops
 * (angle at this loops face corner).
 *
 * \return angle in radians
 */
float BMesh::BM_loop_calc_face_angle(BMLoop *l)
{
	return angle_v3v3v3(l->prev->v->co,
	                    l->v->co,
	                    l->next->v->co);
}

/**
 * \brief BM_loop_calc_face_normal
 *
 * Calculate the normal at this loop corner or fallback to the face normal on straignt lines.
 *
 * \param bm The BMesh
 * \param l The loop to calculate the normal at
 * \param r_normal Resulting normal
 */
void BMesh::BM_loop_calc_face_normal(BMLoop *l, float r_normal[3])
{
	float e1[3],e2[3];
	sub_v3_v3v3(e1, l->v->co, l->prev->v->co);
	sub_v3_v3v3(e2, l->next->v->co, l->v->co);
	cross_v3_v3v3(r_normal,e1,e2);
	float len = len_v3(r_normal);
	if(len != 0.f)	
	{
		mul_v3_v3fl(r_normal, r_normal, 1.f/len);
		return;
	}
	else 
	{
		copy_v3_v3(r_normal, l->f->no);
	}
}

/**
 * \brief BM_loop_calc_face_tangent
 *
 * Calculate the tangent at this loop corner or fallback to the face normal on straignt lines.
 * This vector always points inward into the face.
 *
 * \param bm The BMesh
 * \param l The loop to calculate the tangent at
 * \param r_tangent Resulting tangent
 */
void BMesh::BM_loop_calc_face_tangent(BMLoop *l, float r_tangent[3])
{
	float v_prev[3];
	float v_next[3];

	sub_v3_v3v3(v_prev, l->prev->v->co, l->v->co);
	sub_v3_v3v3(v_next, l->v->co, l->next->v->co);

	normalize_v3(v_prev);
	normalize_v3(v_next);

	if (compare_v3v3(v_prev, v_next, FLT_EPSILON) == FALSE) {
		float dir[3];
		float nor[3]; /* for this purpose doesn't need to be normalized */
		add_v3_v3v3(dir, v_prev, v_next);
		cross_v3_v3v3(nor, v_prev, v_next);
		cross_v3_v3v3(r_tangent, dir, nor);
	}
	else {
		/* prev/next are the same - compare with face normal since we don't have one */
		cross_v3_v3v3(r_tangent, v_next, l->f->no);
	}

	normalize_v3(r_tangent);
}

/**
 * \brief BMESH EDGE/FACE ANGLE
 *
 *  Calculates the angle between two faces.
 *  Assumes the face normals are correct.
 *
 * \return angle in radians
 */
float BMesh::BM_edge_calc_face_angle(BMEdge *e)
{
	if (BM_edge_is_manifold(e)) {
		BMLoop *l1 = e->l;
		BMLoop *l2 = e->l->radial_next;
		return angle_normalized_v3v3(l1->f->no, l2->f->no);
	}
	else {
		return 90.0f/180.0f * M_PI;
	}
}

/**
 * \brief BMESH EDGE/FACE TANGENT
 *
 * Calculate the tangent at this loop corner or fallback to the face normal on straignt lines.
 * This vector always points inward into the face.
 *
 * \brief BM_edge_calc_face_tangent
 * \param e
 * \param e_loop The loop to calculate the tangent at,
 * used to get the face and winding direction.
 */

void BMesh::BM_edge_calc_face_tangent(BMEdge *e, BMLoop *e_loop, float r_tangent[3])
{
	float tvec[3];
	BMVert *v1, *v2;
	BM_edge_ordered_verts_ex(e, &v1, &v2, e_loop);

	sub_v3_v3v3(tvec, v1->co, v2->co); /* use for temp storage */
	/* note, we could average the tangents of both loops,
	 * for non flat ngons it will give a better direction */
	cross_v3_v3v3(r_tangent, tvec, e_loop->f->no);
	normalize_v3(r_tangent);
}

/**
 * \brief BMESH VERT/EDGE ANGLE
 *
 * Calculates the angle a verts 2 edges.
 *
 * \returns the angle in radians
 */
float BMesh::BM_vert_calc_edge_angle(BMVert *v)
{
	BMEdge *e1, *e2;

	/* saves BM_vert_edge_count(v) and and edge iterator,
	 * get the edges and count them both at once */

	if ((e1 = v->e) &&
	    (e2 =  bmesh_disk_edge_next(e1, v)) &&
	    /* make sure we come full circle and only have 2 connected edges */
	    (e1 == bmesh_disk_edge_next(e2, v)))
	{
		BMVert *v1 = BM_edge_other_vert(e1, v);
		BMVert *v2 = BM_edge_other_vert(e2, v);

		return M_PI - angle_v3v3v3(v1->co, v->co, v2->co);
	}
	else {
		return M_PI * 0.5f;
	}
}

/**
 * \note this isn't optimal to run on an array of verts,
 * see 'solidify_add_thickness' for a function which runs on an array.
 */
float BMesh::BM_vert_calc_shell_factor(BMVert *v)
{
	BMIter iter;
	BMLoop *l;
	float accum_shell = 0.0f;
	float accum_angle = 0.0f;
	iter.init(v);
	for (l=lofv_begin(iter); l!=lofv_end(iter); l=lofv_end(iter))
	{
		const float face_angle = BM_loop_calc_face_angle(l);
		accum_shell += shell_angle_to_dist(angle_normalized_v3v3(v->no, l->f->no)) * face_angle;
		accum_angle += face_angle;
	}

	return accum_shell / accum_angle;
}

/**
 * Returns the edge existing between v1 and v2, or NULL if there isn't one.
 *
 * \note multiple edges may exist between any two vertices, and therefore
 * this function only returns the first one found.
 */
BMEdge *BMesh::BM_edge_exists(BMVert *v1, BMVert *v2)
{
	return eofv_2(v1,v2);
}

/**
 * Given a set of vertices \a varr, find out if
 * all those vertices overlap an existing face.
 *
 * \note Making a face here is valid but in some cases you wont want to
 * make a face thats part of another.
 *
 * \returns TRUE for overlap
 *
 */
int BMesh::BM_face_exists_overlap(BMVert **varr, int len, BMFace **r_overlapface)
{
	int i, amount;
	for (i = 0; i < len; i++) {
		BMIter viter;
		viter.init(varr[i]);
		BMFace *f;
		for (f=fofv_begin(viter); f!=fofv_end(viter); f=fofv_next(viter))
		{
			amount = BM_verts_in_face(f, varr, len);
			if (amount >= len) {
				if (r_overlapface) {
					*r_overlapface = f;
				}
				return TRUE;
			}
		}
	}

	if (r_overlapface) {
		*r_overlapface = NULL;
	}

	return FALSE;
}

/**
 * Given a set of vertices (varr), find out if
 * there is a face with exactly those vertices
 * (and only those vertices).
 */
int BMesh::BM_face_exists(BMVert **varr, int len, BMFace **r_existface)
{
	int i, amount;

	for (i = 0; i < len; i++) {
		BMIter viter;
		viter.init(varr[i]);
		BMFace *f;
		for (f=fofv_begin(viter); f!=fofv_end(viter); f=fofv_next(viter))
		{
			amount = BM_verts_in_face(f, varr, len);
			if (amount == len && amount == f->len) {
				if (r_existface) {
					*r_existface = f;
				}
				return TRUE;
			}
		}
	}

	if (r_existface) {
		*r_existface = NULL;
	}
	return FALSE;
}


/**
 * Given a set of vertices and edges (\a varr, \a earr), find out if
 * all those vertices are filled in by existing faces that _only_ use those vertices.
 *
 * This is for use in cases where creating a face is possible but would result in
 * many overlapping faces.
 *
 * An example of how this is used: when 2 tri's are selected that share an edge,
 * pressing Fkey would make a new overlapping quad (without a check like this)
 *
 * \a earr and \a varr can be in any order, however they _must_ form a closed loop.
 */
int BMesh::BM_face_exists_multi(BMVert **varr, BMEdge **earr, int len)
{
	//BMFace *f;
	//BMEdge *e;
	//BMVert *v;
	//int ok;
	//int tot_tag;

	//BMIter fiter;
	//BMIter viter;

	//int i;

	//for (i = 0; i < len; i++) {
	//	/* save some time by looping over edge faces rather then vert faces
	//	 * will still loop over some faces twice but not as many */
	//	BM_ITER_ELEM (f, &fiter, earr[i], BM_FACES_OF_EDGE) {
	//		BM_elem_flag_disable(f, BM_ELEM_INTERNAL_TAG);
	//		BM_ITER_ELEM (v, &viter, f, BM_VERTS_OF_FACE) {
	//			BM_elem_flag_disable(v, BM_ELEM_INTERNAL_TAG);
	//		}
	//	}

	//	/* clear all edge tags */
	//	BM_ITER_ELEM (e, &fiter, varr[i], BM_EDGES_OF_VERT) {
	//		BM_elem_flag_disable(e, BM_ELEM_INTERNAL_TAG);
	//	}
	//}

	///* now tag all verts and edges in the boundary array as true so
	// * we can know if a face-vert is from our array */
	//for (i = 0; i < len; i++) {
	//	BM_elem_flag_enable(varr[i], BM_ELEM_INTERNAL_TAG);
	//	BM_elem_flag_enable(earr[i], BM_ELEM_INTERNAL_TAG);
	//}


	///* so! boundary is tagged, everything else cleared */


	///* 1) tag all faces connected to edges - if all their verts are boundary */
	//tot_tag = 0;
	//for (i = 0; i < len; i++) {
	//	BM_ITER_ELEM (f, &fiter, earr[i], BM_FACES_OF_EDGE) {
	//		if (!BM_elem_flag_test(f, BM_ELEM_INTERNAL_TAG)) {
	//			ok = TRUE;
	//			BM_ITER_ELEM (v, &viter, f, BM_VERTS_OF_FACE) {
	//				if (!BM_elem_flag_test(v, BM_ELEM_INTERNAL_TAG)) {
	//					ok = FALSE;
	//					break;
	//				}
	//			}

	//			if (ok) {
	//				/* we only use boundary verts */
	//				BM_elem_flag_enable(f, BM_ELEM_INTERNAL_TAG);
	//				tot_tag++;
	//			}
	//		}
	//		else {
	//			/* we already found! */
	//		}
	//	}
	//}

	//if (tot_tag == 0) {
	//	/* no faces use only boundary verts, quit early */
	//	return FALSE;
	//}

	///* 2) loop over non-boundary edges that use boundary verts,
	// *    check each have 2 tagges faces connected (faces that only use 'varr' verts) */
	//ok = TRUE;
	//for (i = 0; i < len; i++) {
	//	BM_ITER_ELEM (e, &fiter, varr[i], BM_EDGES_OF_VERT) {

	//		if (/* non-boundary edge */
	//		    BM_elem_flag_test(e, BM_ELEM_INTERNAL_TAG) == FALSE &&
	//		    /* ...using boundary verts */
	//		    BM_elem_flag_test(e->v1, BM_ELEM_INTERNAL_TAG) == TRUE &&
	//		    BM_elem_flag_test(e->v2, BM_ELEM_INTERNAL_TAG) == TRUE)
	//		{
	//			int tot_face_tag = 0;
	//			BM_ITER_ELEM (f, &fiter, e, BM_FACES_OF_EDGE) {
	//				if (BM_elem_flag_test(f, BM_ELEM_INTERNAL_TAG)) {
	//					tot_face_tag++;
	//				}
	//			}

	//			if (tot_face_tag != 2) {
	//				ok = FALSE;
	//				break;
	//			}

	//		}
	//	}

	//	if (ok == FALSE) {
	//		break;
	//	}
	//}

	//return ok;
printf("%s: not implemented!\n",__func__);
return 0;
}

/* same as 'BM_face_exists_multi' but built vert array from edges */
int BMesh::BM_face_exists_multi_edge(BMEdge **earr, int len)
{
	BMVert **varr;
	varr = new BMVert*[len];

	int ok;
	int i, i_next;

	ok = TRUE;
	for (i = len - 1, i_next = 0; i_next < len; (i = i_next++)) {
		if (!(varr[i] = BM_edge_share_vert(earr[i], earr[i_next]))) {
			ok = FALSE;
			break;
		}
	}

	if (ok == FALSE) {
		BMESH_ASSERT(0);
		free(varr);
		return FALSE;
	}

	ok = BM_face_exists_multi(varr, earr, len);
	delete varr;
	return ok;
}


/**
 * \brief TEST EDGE SIDE and POINT IN TRIANGLE
 *
 * Point in triangle tests stolen from scanfill.c.
 * Used for tessellator
 */

static short testedgesidef(const float v1[2], const float v2[2], const float v3[2])
{
	/* is v3 to the right of v1 - v2 ? With exception: v3 == v1 || v3 == v2 */
	double inp;

	//inp = (v2[cox] - v1[cox]) * (v1[coy] - v3[coy]) + (v1[coy] - v2[coy]) * (v1[cox] - v3[cox]);
	inp = (v2[0] - v1[0]) * (v1[1] - v3[1]) + (v1[1] - v2[1]) * (v1[0] - v3[0]);

	if (inp < 0.0) {
		return FALSE;
	}
	else if (inp == 0) {
		if (v1[0] == v3[0] && v1[1] == v3[1]) return FALSE;
		if (v2[0] == v3[0] && v2[1] == v3[1]) return FALSE;
	}
	return TRUE;
}

/**
 * \brief COMPUTE POLY NORMAL
 *
 * Computes the normal of a planar
 * polygon See Graphics Gems for
 * computing newell normal.
 */
static void calc_poly_normal(float normal[3], float verts[][3], int nverts)
{
	float const *v_prev = verts[nverts - 1];
	float const *v_curr = verts[0];
	float n[3] = {0.0f};
	int i;

	/* Newell's Method */
	for (i = 0; i < nverts; v_prev = v_curr, v_curr = verts[++i]) {
		add_newell_cross_v3_v3v3(n, v_prev, v_curr);
	}

	if (UNLIKELY(normalize_v3_v3(normal, n) == 0.0f)) {
		normal[2] = 1.0f; /* other axis set to 0.0 */
	}
}

/**
 * \brief COMPUTE POLY NORMAL (BMFace)
 *
 * Same as #calc_poly_normal but operates directly on a bmesh face.
 */
void BMesh::bm_face_calc_poly_normal(BMFace *f)
{
	BMLoop *l_first = f->l_first;
	BMLoop *l_iter  = l_first;
	float const *v_prev = l_first->prev->v->co;
	float const *v_curr = l_first->v->co;
	float n[3] = {0.0f};

	/* Newell's Method */
	do {
		add_newell_cross_v3_v3v3(n, v_prev, v_curr);

		l_iter = l_iter->next;
		v_prev = v_curr;
		v_curr = l_iter->v->co;

	} while (l_iter != l_first);

	if (UNLIKELY(normalize_v3_v3(f->no, n) == 0.0f)) {
		f->no[2] = 1.0f; /* other axis set to 0.0 */
	}
}

/**
 * \brief COMPUTE POLY NORMAL (BMFace)
 *
 * Same as #calc_poly_normal and #bm_face_calc_poly_normal
 * but takes an array of vertex locations.
 */
void BMesh::bm_face_calc_poly_normal_vertex_cos(BMFace *f, float n[3],
                                                float const (*vertexCos)[3])
{
	BMLoop *l_first = BM_FACE_FIRST_LOOP(f);
	BMLoop *l_iter  = l_first;
	float const *v_prev = vertexCos[l_first->prev->v->getIndex()];
	float const *v_curr = vertexCos[l_first->v->getIndex()];

	zero_v3(n);

	/* Newell's Method */
	do {
		add_newell_cross_v3_v3v3(n, v_prev, v_curr);

		l_iter = l_iter->next;
		v_prev = v_curr;
		v_curr = vertexCos[l_iter->v->getIndex()];
	} while (l_iter != l_first);

	if (UNLIKELY(normalize_v3(n) == 0.0f)) {
		n[2] = 1.0f; /* other axis set to 0.0 */
	}
}

/**
 * get the area of the face
 */
float BMesh::BM_face_calc_area(BMFace *f)
{
	BMLoop *l;
	BMIter iter;
	float *verts;
	float normal[3];
	float area;
	int i;

	verts = (float*)malloc(f->len * sizeof(float)*3);

	iter.init(f);
	for (i=0,l = loff_begin(iter); l!=loff_end(iter); l=loff_next(iter),i++)
		copy_v3_v3(&verts[i*3], l->v->co);

	if (f->len == 3) {
		area = area_tri_v3(&verts[0], &verts[3], &verts[6]);
	}
	else if (f->len == 4) {
		area = area_quad_v3(&verts[0], &verts[3], &verts[6], &verts[9]);
	}
	else {
		calc_poly_normal(normal, verts, f->len);
		area = area_poly_v3(f->len, (float(*)[3])verts, normal);
	}
	free(verts);
	return area;
}

/**
 * compute the perimeter of an ngon
 */
float BMesh::BM_face_calc_perimeter(BMFace *f)
{
	BMLoop *l_iter, *l_first;
	float perimeter = 0.0f;

	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		perimeter += len_v3v3(l_iter->v->co, l_iter->next->v->co);
	} while ((l_iter = l_iter->next) != l_first);

	return perimeter;
}

/**
 * computes center of face in 3d.  uses center of bounding box.
 */
void BMesh::BM_face_calc_center_bounds(BMFace *f, float r_cent[3])
{
	BMLoop *l_iter;
	BMLoop *l_first;
	float min[3], max[3];

	INIT_MINMAX(min, max);

	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		DO_MINMAX(l_iter->v->co, min, max);
	} while ((l_iter = l_iter->next) != l_first);

	mid_v3_v3v3(r_cent, min, max);
}

/**
 * computes the center of a face, using the mean average
 */
void BMesh::BM_face_calc_center_mean(BMFace *f, float r_cent[3])
{
	BMLoop *l_iter;
	BMLoop *l_first;

	zero_v3(r_cent);

	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		add_v3_v3(r_cent, l_iter->v->co);
	} while ((l_iter = l_iter->next) != l_first);

	if (f->len)
		mul_v3_fl(r_cent, 1.0f / (float) f->len);
}

/**
 * updates face and vertex normals incident on an edge
 */
void BMesh::BM_edge_normals_update(BMEdge *e)
{
	BMIter iter;
	BMFace *f;
	iter.init(e);
	for (f=fofe_begin(iter); f!=fofe_end(iter); f=fofe_next(iter))
		BM_face_normal_update(f);
	BM_vert_normal_update(e->v1);
	BM_vert_normal_update(e->v2);
}

/**
 * update a vert normal (but not the faces incident on it)
 */
void BMesh::BM_vert_normal_update(BMVert *v)
{
	/* TODO, we can normalize each edge only once, then compare with previous edge */

	BMIter liter;
	BMLoop *l;
	float vec1[3], vec2[3], fac;
	int len = 0;

	zero_v3(v->no);
	liter.init(v);
	for(l=lofv_begin(liter);l!=lofv_end(liter);l=lofv_next(liter))
	{
		/* Same calculation used in BM_mesh_normals_update */
		sub_v3_v3v3(vec1, l->v->co, l->prev->v->co);
		sub_v3_v3v3(vec2, l->next->v->co, l->v->co);
		normalize_v3(vec1);
		normalize_v3(vec2);

		fac = saacos(-dot_v3v3(vec1, vec2));

		madd_v3_v3fl(v->no, l->f->no, fac);

		len++;
	}

	if (len) {
		normalize_v3(v->no);
	}
}

void BMesh::BM_vert_normal_update_all(BMVert *v)
{
	BMIter iter;
	BMFace *f;
	iter.init(v);
	for(f=fofv_begin(iter);f!=fofv_end(iter);f=fofv_next(iter))
	{
		BM_face_normal_update(f);
	}

	BM_vert_normal_update(v);
}

/**
 * \brief BMESH UPDATE FACE NORMAL
 *
 * Updates the stored normal for the
 * given face. Requires that a buffer
 * of sufficient length to store projected
 * coordinates for all of the face's vertices
 * is passed in as well.
 */

void BMesh::BM_face_normal_update(BMFace *f)
{
	BMLoop *l;

	/* common cases first */
	switch (f->len) {
		case 4:
		{
			const float *co1 = (l = BM_FACE_FIRST_LOOP(f))->v->co;
			const float *co2 = (l = l->next)->v->co;
			const float *co3 = (l = l->next)->v->co;
			const float *co4 = (l->next)->v->co;

			normal_quad_v3(f->no, co1, co2, co3, co4);
			break;
		}
		case 3:
		{
			const float *co1 = (l = BM_FACE_FIRST_LOOP(f))->v->co;
			const float *co2 = (l = l->next)->v->co;
			const float *co3 = (l->next)->v->co;

			normal_tri_v3(f->no, co1, co2, co3);
			break;
		}
		case 0:
		{
			zero_v3(f->no);
			break;
		}
		default:
		{
			bm_face_calc_poly_normal(f);
			break;
		}
	}
}
/* exact same as 'bmesh_face_normal_update' but accepts vertex coords */
void BMesh::BM_face_normal_update_vcos(BMesh *bm, BMFace *f, float no[3],
                                float const (*vertexCos)[3])
{
	BMLoop *l;

	/* common cases first */
	switch (f->len) {
		case 4:
		{
			const float *co1 = vertexCos[(l = BM_FACE_FIRST_LOOP(f))->v->getIndex()];
			const float *co2 = vertexCos[(l = l->next)->v->getIndex()];
			const float *co3 = vertexCos[(l = l->next)->v->getIndex()];
			const float *co4 = vertexCos[(l->next)->v->getIndex()];

			normal_quad_v3(no, co1, co2, co3, co4);
			break;
		}
		case 3:
		{
			const float *co1 = vertexCos[(l = BM_FACE_FIRST_LOOP(f))->v->getIndex()];
			const float *co2 = vertexCos[(l = l->next)->v->getIndex()];
			const float *co3 = vertexCos[(l->next)->v->getIndex()];

			normal_tri_v3(no, co1, co2, co3);
			break;
		}
		case 0:
		{
			zero_v3(no);
			break;
		}
		default:
		{
			bm_face_calc_poly_normal_vertex_cos(f, no, vertexCos);
			break;
		}
	}
}

/**
 * \brief Face Flip Normal
 *
 * Reverses the winding of a face.
 * \note This updates the calculated normal.
 */
void BMesh::BM_face_normal_flip(BMesh *bm, BMFace *f)
{
	bmesh_loop_reverse(bm, f);
	negate_v3(f->no);
}

/* detects if two line segments cross each other (intersects).
 * note, there could be more winding cases then there needs to be. */
static int linecrossesf(const float v1[2], const float v2[2], const float v3[2], const float v4[2])
{

#define GETMIN2_AXIS(a, b, ma, mb, axis)   \
	{                                      \
		ma[axis] = MIN2(a[axis], b[axis]); \
		mb[axis] = MAX2(a[axis], b[axis]); \
	} (void)0

#define GETMIN2(a, b, ma, mb)          \
	{                                  \
		GETMIN2_AXIS(a, b, ma, mb, 0); \
		GETMIN2_AXIS(a, b, ma, mb, 1); \
	} (void)0

#define EPS (FLT_EPSILON * 15)

	int w1, w2, w3, w4, w5 /*, re */;
	float mv1[2], mv2[2], mv3[2], mv4[2];
	
	/* now test winding */
	w1 = testedgesidef(v1, v3, v2);
	w2 = testedgesidef(v2, v4, v1);
	w3 = !testedgesidef(v1, v2, v3);
	w4 = testedgesidef(v3, v2, v4);
	w5 = !testedgesidef(v3, v1, v4);
	
	if (w1 == w2 && w2 == w3 && w3 == w4 && w4 == w5) {
		return TRUE;
	}
	
	GETMIN2(v1, v2, mv1, mv2);
	GETMIN2(v3, v4, mv3, mv4);
	
	/* do an interval test on the x and y axes */
	/* first do x axis */
	if (abs(v1[1] - v2[1]) < EPS &&
	    abs(v3[1] - v4[1]) < EPS &&
	    abs(v1[1] - v3[1]) < EPS)
	{
		return (mv4[0] >= mv1[0] && mv3[0] <= mv2[0]);
	}

	/* now do y axis */
	if (abs(v1[0] - v2[0]) < EPS &&
	    abs(v3[0] - v4[0]) < EPS &&
	    abs(v1[0] - v3[0]) < EPS)
	{
		return (mv4[1] >= mv1[1] && mv3[1] <= mv2[1]);
	}

	return FALSE;

#undef GETMIN2_AXIS
#undef GETMIN2
#undef EPS

}
static void axis_dominant_v3(int *axis_a, int *axis_b, const float axis[3])
{
	const float xn = fabsf(axis[0]);
	const float yn = fabsf(axis[1]);
	const float zn = fabsf(axis[2]);

	if      (zn >= xn && zn >= yn) { *axis_a= 0; *axis_b = 1; }
	else if (yn >= xn && yn >= zn) { *axis_a= 0; *axis_b = 2; }
	else                           { *axis_a= 1; *axis_b = 2; }
}
/**
 *  BM POINT IN FACE
 *
 * Projects co onto face f, and returns true if it is inside
 * the face bounds.
 *
 * \note this uses a best-axis projection test,
 * instead of projecting co directly into f's orientation space,
 * so there might be accuracy issues.
 */
int BMesh::BM_face_point_inside_test(BMFace *f, const float co[3])
{
	int ax, ay;
	float co2[2], cent[2] = {0.0f, 0.0f}, out[2] = {FLT_MAX * 0.5f, FLT_MAX * 0.5f};
	BMLoop *l_iter;
	BMLoop *l_first;
	int crosses = 0;
	float onepluseps = 1.0f + (float)FLT_EPSILON * 150.0f;
	
	if (dot_v3v3(f->no, f->no) <= FLT_EPSILON * 10)
		BM_face_normal_update(f);
	
	/* find best projection of face XY, XZ or YZ: barycentric weights of
	 * the 2d projected coords are the same and faster to compute
	 *
	 * this probably isn't all that accurate, but it has the advantage of
	 * being fast (especially compared to projecting into the face orientation)
	 */
	axis_dominant_v3(&ax, &ay, f->no);

	co2[0] = co[ax];
	co2[1] = co[ay];
	
	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		cent[0] += l_iter->v->co[ax];
		cent[1] += l_iter->v->co[ay];
	} while ((l_iter = l_iter->next) != l_first);
	
	mul_v2_fl(cent, 1.0f / (float)f->len);
	
	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		float v1[2], v2[2];
		
		v1[0] = (l_iter->prev->v->co[ax] - cent[ax]) * onepluseps + cent[ax];
		v1[1] = (l_iter->prev->v->co[ay] - cent[ay]) * onepluseps + cent[ay];
		
		v2[0] = (l_iter->v->co[ax] - cent[ax]) * onepluseps + cent[ax];
		v2[1] = (l_iter->v->co[ay] - cent[ay]) * onepluseps + cent[ay];
		
		crosses += linecrossesf(v1, v2, co2, out) != 0;
	} while ((l_iter = l_iter->next) != l_first);
	
	return crosses % 2 != 0;
}

int BMesh::bm_face_goodline(float const (*projectverts)[3], BMFace *f,
	int v1i, int v2i, int v3i, float polyArea)
{
	BMLoop *l_iter;
	BMLoop *l_first;
	float v1[3], v2[3], v3[3], pv1[3], pv2[3];
	int i;

	copy_v3_v3(v1, projectverts[v1i]);
	copy_v3_v3(v2, projectverts[v2i]);
	copy_v3_v3(v3, projectverts[v3i]);
	
	if (testedgesidef(v1, v2, v3) && polyArea>0) {
		return FALSE;
	}

	//for (i = 0; i < nvert; i++) {
	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		i = l_iter->v->getIndex();
		if (i == v1i || i == v2i || i == v3i) {
			continue;
		}
		
		copy_v3_v3(pv1, projectverts[l_iter->v->getIndex()]);
		copy_v3_v3(pv2, projectverts[l_iter->next->v->getIndex()]);
		
		//if (linecrossesf(pv1, pv2, v1, v3)) return FALSE;

		if (isect_point_tri_v2(pv1, v1, v2, v3) ||
		    isect_point_tri_v2(pv1, v3, v2, v1))
		{
			return FALSE;
		}
	} while ((l_iter = l_iter->next) != l_first);
	return TRUE;
}

/**
 * \brief Find Ear
 *
 * Used by tessellator to find
 * the next triangle to 'clip off'
 * of a polygon while tessellating.
 *
 * \param use_beauty Currently only applies to quads, can be extended later on.
 */
BMLoop *BMesh::find_ear(BMFace *f, float (*verts)[3], const int nvert, const int use_beauty, float polyArea)
{
	BMLoop *bestear = NULL;

	BMLoop *l_iter;
	BMLoop *l_first;

	//if (f->len == 4) {
	//	BMLoop *larr[4];
	//	int i = 0;

	//	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	//	do {
	//		larr[i] = l_iter;
	//		i++;
	//	} while ((l_iter = l_iter->next) != l_first);

	//	/* pick 0/1 based on best lenth */
	//	bestear = larr[(((len_squared_v3v3(larr[0]->v->co, larr[2]->v->co) >
	//	                  len_squared_v3v3(larr[1]->v->co, larr[3]->v->co))) != use_beauty)];

	//}
	//else 
	{
		BMVert *v1, *v2, *v3;
		float pv1[3], pv2[3], maxcosa=-1e15f;

		/* float angle, bestangle = 180.0f; */
		int isear /*, i = 0 */;

		l_iter = l_first = BM_FACE_FIRST_LOOP(f);
		do {
			isear = TRUE;

			v1 = l_iter->prev->v;
			v2 = l_iter->v;
			v3 = l_iter->next->v;

			if (BM_edge_exists(v1, v3)) {
				isear = FALSE;
			}
			else if (!bm_face_goodline((float const (*)[3])verts, f,
			                           v1->getIndex(), v2->getIndex(), v3->getIndex(),polyArea
									   ))
			{
				isear = FALSE;
			}

			if (isear) 
			{
				sub_v3_v3v3(pv1,v2->co,v1->co);
				sub_v3_v3v3(pv2,v2->co,v3->co);
				normalize_v3(pv1);
				normalize_v3(pv2);
				float cosa = dot_v3v3(pv1,pv2);
				if(cosa > maxcosa)
				{
					maxcosa=cosa;
					bestear = l_iter;
				}
			}
		} while ((l_iter = l_iter->next) != l_first);
	}

	return bestear;
}


/**
 * low level function, only frees the vert,
 * doesn't change or adjust surrounding geometry
 */
void BMesh::bm_kill_only_vert(BMesh *bm, BMVert *v)
{
	bm->totvert--;
	BLI_mempool_free(bm->vpool, v);
}

/**
 * low level function, only frees the edge,
 * doesn't change or adjust surrounding geometry
 */
void BMesh::bm_kill_only_edge(BMesh *bm, BMEdge *e)
{
	bm->totedge--;
	BLI_mempool_free(bm->epool, e);
}

/**
 * low level function, only frees the face,
 * doesn't change or adjust surrounding geometry
 */
void BMesh::bm_kill_only_face(BMesh *bm, BMFace *f)
{
	bm->totface--;
	BLI_mempool_free(bm->fpool, f);
}

/**
 * low level function, only frees the loop,
 * doesn't change or adjust surrounding geometry
 */
void BMesh::bm_kill_only_loop(BMesh *bm, BMLoop *l)
{
	bm->totloop--;
	BLI_mempool_free(bm->lpool, l);
}

/**
 * kills all edges associated with \a f, along with any other faces containing
 * those edges
 */
void BMesh::BM_face_edges_kill(BMesh *bm, BMFace *f)
{
	vector<BMEdge*> edges;
	edges.reserve(BM_NGON_STACK_SIZE);
	BMLoop *l_iter;
	BMLoop *l_first;
	
	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		edges.push_back(l_iter->e);
	} while ((l_iter = l_iter->next) != l_first);
	
	for (unsigned i = 0; i < edges.size(); i++) {
		BM_edge_kill(bm, edges[i]);
	}
}

/**
 * kills all verts associated with \a f, along with any other faces containing
 * those vertices
 */
void BMesh::BM_face_verts_kill(BMesh *bm, BMFace *f)
{
	vector<BMVert*> verts;
	verts.reserve(BM_LOOPS_OF_FACE);
	BMLoop *l_iter;
	BMLoop *l_first;

	l_iter = l_first = BM_FACE_FIRST_LOOP(f);
	do {
		verts.push_back(l_iter->v);
	} while ((l_iter = l_iter->next) != l_first);

	for (unsigned i = 0; i < verts.size(); i++) {
		BM_vert_kill(bm, verts[i]);
	}
}

void BMesh::BM_face_kill(BMesh *bm, BMFace *f)
{
	if (f->l_first)
	{
		BMLoop *l_iter, *l_next, *l_first;
		l_iter = l_first = f->l_first;
		do {
			l_next = l_iter->next;

			bmesh_radial_loop_remove(l_iter, l_iter->e);
			bm_kill_only_loop(bm, l_iter);

		} while ((l_iter = l_next) != l_first);
	}
	bm_kill_only_face(bm, f);
}
/**
 * kills \a e and all faces that use it.
 */
void BMesh::BM_edge_kill(BMesh *bm, BMEdge *e)
{

	bmesh_disk_edge_remove(e, e->v1);
	bmesh_disk_edge_remove(e, e->v2);

	if (e->l) {
		BMLoop *l = e->l, *lnext, *startl = e->l;

		do {
			lnext = l->radial_next;
			if (lnext->f == l->f) {
				BM_face_kill(bm, l->f);
				break;
			}
			
			BM_face_kill(bm, l->f);

			if (l == lnext)
				break;
			l = lnext;
		} while (l != startl);
	}
	
	bm_kill_only_edge(bm, e);
}

/**
 * kills \a v and all edges that use it.
 */
void BMesh::BM_vert_kill(BMesh *bm, BMVert *v)
{
	if (v->e) {
		BMEdge *e, *nexte;
		
		e = v->e;
		while (v->e) {
			nexte = bmesh_disk_edge_next(e, v);
			BM_edge_kill(bm, e);
			e = nexte;
		}
	}

	bm_kill_only_vert(bm, v);
}

/**
 * \brief Splice Vert
 *
 * Merges two verts into one (\a v into \a vtarget).
 *
 * \return Success
 */
int BMesh::BM_vert_splice(BMesh *bm, BMVert *v, BMVert *vtarget)
{
	BMEdge *e;
	BMLoop *l;
	BMIter liter;

	/* verts already spliced */
	if (v == vtarget) {
		return FALSE;
	}

	/* retarget all the loops of v to vtarget */
	liter.init(v);
	for (l=lofv_begin(liter); l!=lofv_end(liter);l=lofv_next(liter)) {
		l->v = vtarget;
	}

	/* move all the edges from v's disk to vtarget's disk */
	while ((e = v->e)) {
		bmesh_disk_edge_remove(e, v);
		bmesh_edge_swapverts(e, v, vtarget);
		bmesh_disk_edge_append(e, vtarget);
	}

	/* v is unused now, and can be killed */
	BM_vert_kill(bm, v);

	return TRUE;
}

/**
 * \brief Splice Edge
 *
 * Splice two unique edges which share the same two vertices into one edge.
 *
 * \return Success
 *
 * \note Edges must already have the same vertices.
 */
int BMesh::BM_edge_splice(BMesh *bm, BMEdge *e, BMEdge *etarget)
{
	BMLoop *l;

	if (!BM_vert_in_edge(e, etarget->v1) || !BM_vert_in_edge(e, etarget->v2)) {
		/* not the same vertices can't splice */
		return FALSE;
	}

	while (e->l) {
		l = e->l;
		assert(BM_vert_in_edge(etarget, l->v));
		assert(BM_vert_in_edge(etarget, l->next->v));
		bmesh_radial_loop_remove(l, e);
		bmesh_radial_append(etarget, l);
	}

	assert(bmesh_radial_length(e->l) == 0);

	/* removes from disks too */
	BM_edge_kill(bm, e);

	return TRUE;
}
//====================================================================================================


}//end namespace ldp


