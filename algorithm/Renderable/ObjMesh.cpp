#define _CRT_SECURE_NO_WARNINGS
#include "ObjMesh.h"
#include <stdio.h>
#include <glut.h>
#include <algorithm>
#include "CFreeImage.h"
#include "bmesh.h"
#include <queue>
using namespace ldp;
using namespace std;
#undef min
#undef max

#define WHITESPACE " \t\n\r"

ObjMesh::obj_material ObjMesh::default_material;

ObjMesh::ObjMesh():Renderable()
{
	scene_filename[0] = 0;
	material_filename[0] = 0;
	_name = "ObjMesh";
	_fast_view_should_update = true;
	_fast_view_last_showType = 0;
	m_bmesh = 0;
	m_bmesh_triagulate = false;
}

ObjMesh::ObjMesh(const ObjMesh& rhs)
{
	cloneFrom(&rhs);
}

void ObjMesh::cloneFrom(const ObjMesh* rhs)
{
	if (rhs == this)
		return;
	vertex_list.assign(rhs->vertex_list.begin(), rhs->vertex_list.end());
	vertex_normal_list.assign(rhs->vertex_normal_list.begin(), rhs->vertex_normal_list.end());
	vertex_color_list.assign(rhs->vertex_color_list.begin(), rhs->vertex_color_list.end());
	face_normal_list.assign(rhs->face_normal_list.begin(), rhs->face_normal_list.end());
	vertex_texture_list.assign(rhs->vertex_texture_list.begin(), rhs->vertex_texture_list.end());
	face_list.assign(rhs->face_list.begin(), rhs->face_list.end());
	material_list.assign(rhs->material_list.begin(), rhs->material_list.end());
	vertex_is_selected.assign(rhs->vertex_is_selected.begin(), rhs->vertex_is_selected.end());

	for (int i = 0; i < 2; i++)
		boundingBox[i] = rhs->boundingBox[i];
	strcpy_s(scene_filename, rhs->scene_filename);
	strcpy_s(material_filename, rhs->material_filename);

	// reconstruct bmesh, prevent multi-reference.
	m_bmesh = 0;
}

ObjMesh& ObjMesh::operator=(const ObjMesh& rhs)
{
	cloneFrom(&rhs);
	return *this;
}

ObjMesh::~ObjMesh()
{
	clear();
}

void ObjMesh::clear()
{
	scene_filename[0] = 0;
	material_filename[0] = 0;
	vertex_list.clear();
	vertex_normal_list.clear();
	vertex_color_list.clear();
	vertex_texture_list.clear();
	face_list.clear();
	material_list.clear();
	face_normal_list.clear();
	boundingBox[0] = boundingBox[1] = 0;

	_fast_view_verts.clear();
	_fast_view_normals.clear();
	_fast_view_texcoords.clear();
	_fast_view_should_update = true;

	if (m_bmesh)
	{
		delete m_bmesh;
		m_bmesh = 0;
	}
	m_bmeshVerts.clear();
}

void ObjMesh::renderConstColor(Float3 color)const
{
	if(!_isEnabled)
		return;
	if(vertex_list.size() == 0)
		return;
	if(face_list.size() == 0)
		return;

	glDisable(GL_LIGHTING);
	glPushAttrib(GL_COLOR_WRITEMASK);
	const obj_face *faces = &face_list[0];
	const Float3* vertices = &vertex_list[0];
	int nfaces = face_list.size();
	int faceNum = faces[0].vertex_count;
	glColor3fv(color.ptr());
	if(faceNum == 3)
		glBegin(GL_TRIANGLES);
	else
		glBegin(GL_QUADS);
	for(int i=0; i<nfaces; i++)
	{
		if(faces[i].vertex_count != faceNum)
		{
			glEnd();
			if(faces[i].vertex_count == 3)
				glBegin(GL_TRIANGLES);
			else
				glBegin(GL_QUADS);
			faceNum = faces[i].vertex_count;
		}
		const int* vidx = faces[i].vertex_index;
		for(int j=0; j<faces[i].vertex_count; j++)
		{
			glVertex3fv((const float*)&vertices[vidx[j]]);
		}
	}
	glEnd();

	glPopAttrib();
	glEnable(GL_LIGHTING);
}

void ObjMesh::render(int showType, int frameIndex)
{
	if(!_isEnabled)
		return;
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	if(vertex_list.size() == 0)
		return;
	if(face_list.size() == 0)
		return;
	if(face_normal_list.size() == 0 || vertex_normal_list.size() == 0)
		updateNormals();
	if (vertex_is_selected.size() != vertex_list.size())
		vertex_is_selected.resize(vertex_list.size(), 0);
	if (vertex_color_list.size() != vertex_list.size())
		vertex_color_list.resize(vertex_list.size(), 0.8);


	const Float3* vertices = &vertex_list[0];
	const Float3* vnormals = &vertex_normal_list[0];
	const Float3* fnormals = &face_normal_list[0];
	const obj_face *faces = &face_list[0];
	const obj_material *mats = 0;
	int nfaces = face_list.size();
	int nverts = vertex_list.size();
	if(material_list.size()>0)
		mats = &material_list[0];

	if (showType & SW_LIGHTING)
	{
		glEnable(GL_LIGHTING);
	}
	else{
		glDisable(GL_LIGHTING);
	}

	if(showType & SW_E)
	{
		glEnable (GL_POLYGON_OFFSET_FILL); 
		glPolygonOffset (1., 1.); 
	}

	if(showType & SW_F)
	{
		renderFaces(showType);
	}
	if(showType & SW_E)
	{
		glDisable(GL_LIGHTING);
		glColor3f(0.2,0.3,0.4);
		glLineWidth( 1.0f );
		glBegin(GL_LINES);
		for(int i=0; i<nfaces; i++)
		{
			const int* vidx = faces[i].vertex_index;
			int nfc = faces[i].vertex_count;
			for(int j=0; j<nfc; j++)
			{
				if (vertex_is_selected[vidx[j]])
					glColor3f(1, 1, 0);
				else
					glColor3f(0.2, 0.3, 0.6);
				glVertex3fv((const float*)&vertices[vidx[j]]);
				if (vertex_is_selected[vidx[(j + 1) % nfc]])
					glColor3f(1, 1, 0);
				else
					glColor3f(0.2, 0.3, 0.6);
				glVertex3fv((const float*)&vertices[vidx[(j+1)%nfc]]);
			}
		}
		glEnd();
		glColor3f(1, 1, 1);
	}
	if(showType & SW_V)
	{
		glDisable(GL_LIGHTING);
		glPointSize(5);
		glColor3f(1,1,0);

		glBegin(GL_POINTS);
		for (int i = 0; i < nverts; i++)
		{
			if (vertex_is_selected[i])
				glColor3f(1, 1, 0);
			else
				glColor3f(0.2, 0.3, 0.6);
			glVertex3fv(vertices[i].ptr());
		}
		glEnd();

		glColor3f(1,1,1);
		glPointSize(1);
	}
	
	glDisable (GL_POLYGON_OFFSET_FILL); 

	glPopAttrib();

	_fast_view_last_showType = showType;
}

void ObjMesh::renderFaces(int showType)const
{
	if (_fast_view_last_showType != showType)
	{
		_fast_view_should_update = true;
		_fast_view_last_showType = showType;
	}
	if (_fast_view_should_update)
		generate_fast_view_tri_face_by_group(showType);

	bool enableTexture = ((showType & SW_TEXTURE) != 0) && (vertex_texture_list.size() > 0);
	bool enableColor = ((showType & SW_COLOR) != 0) && (vertex_color_list.size() > 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	if (enableTexture)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	if (enableColor)
		glEnableClientState(GL_COLOR_ARRAY);
	
	int nS = material_list.size();
	if (nS == 0) nS = 1;

	for (int i = 0; i < nS; i++)
	{
		if (material_list.size() == 0)
			default_material.drawMat(enableTexture);
		else
			material_list[i].drawMat(enableTexture);
		if (_fast_view_verts[i].size() == 0)
			continue;
		glVertexPointer(3, GL_FLOAT, 0, _fast_view_verts[i].data());
		glNormalPointer(GL_FLOAT, 0, _fast_view_normals[i].data());
		if (enableTexture)
			glTexCoordPointer(2, GL_FLOAT, 0, _fast_view_texcoords[i].data());
		if (enableColor)
			glColorPointer(3, GL_FLOAT, 0, _fast_view_colors[i].data());
		glDrawArrays(GL_TRIANGLES, 0, _fast_view_verts[i].size());
	}

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}

void ObjMesh::obj_material::drawMat(int isTextureEnabled)const
{
	float		color[4];

	color[0] = amb[0];
	color[1] = amb[1];
	color[2] = amb[2];
	color[3] = trans;
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);

	color[0] = (float) diff[0];
	color[1] = (float) diff[1];
	color[2] = (float) diff[2];
	color[3] = (float) trans;
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);

	color[0] = (float) spec[0];
	color[1] = (float) spec[1];
	color[2] = (float) spec[2];
	color[3] = (float) trans;
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);


	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 
		(float) (shiny));

	//draw texture
	if (isTextureEnabled && texture_id>0)
	{
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, texture_id);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, &image[0]);
	}
	else{
		glDisable(GL_TEXTURE_2D);
	}
}

void ObjMesh::obj_material::generateTextures()
{
	if (image.size() != image_width * image_height * image_channel)
	{
		return;
	}
	glGenTextures(1, &texture_id);

	if(texture_id == 0)
		printf("Get OpenGL contex failed!\n");

	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glBindTexture(GL_TEXTURE_2D, 0);
}

void ObjMesh::generate_fast_view_tri_face_by_group(int showType)const
{
	int nS = material_list.size();
	if (nS == 0) nS = 1;

	_fast_view_verts.clear();
	_fast_view_normals.clear();
	_fast_view_texcoords.clear();
	_fast_view_colors.clear();
	_fast_view_verts.resize(nS);
	_fast_view_normals.resize(nS);
	_fast_view_texcoords.resize(nS);
	_fast_view_colors.resize(nS);
	for (int i = 0; i < nS; i++)
	{
		_fast_view_verts[i].reserve(face_list.size() * 6);
		_fast_view_normals[i].reserve(face_list.size() * 6);
		_fast_view_texcoords[i].reserve(face_list.size() * 6);
		_fast_view_colors[i].reserve(face_list.size() * 6);
	}

	const static ldp::Int3 order[2] = {ldp::Int3(0,1,2), ldp::Int3(0,2,3)};
	for (int i = 0; i < face_list.size(); i++)
	{
		const obj_face& f = face_list[i];
		int iS = 0;
		if (material_list.size() != 0) iS = f.material_index;

		std::vector<ldp::Float3>& verts = _fast_view_verts[iS];
		std::vector<ldp::Float3>& normals = _fast_view_normals[iS];
		std::vector<ldp::Float2>& tex_coords = _fast_view_texcoords[iS];
		std::vector<ldp::Float3>& colors = _fast_view_colors[iS];

		for (int j = 0; j <= f.vertex_count-3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				int p = order[j][k];
				verts.push_back(vertex_list[f.vertex_index[p]]);
				if (showType & SW_SMOOTH)
					normals.push_back(vertex_normal_list[f.normal_index[p]]);
				else
					normals.push_back(face_normal_list[i]);
				if (vertex_texture_list.size())
				{
					if (f.texture_index[p] >= 0)
						tex_coords.push_back(vertex_texture_list[f.texture_index[p]]);
					else
						tex_coords.push_back(0.f);
				}
				if (vertex_color_list.size())
					colors.push_back(vertex_color_list[f.vertex_index[p]]);
			}
		}
	}//end for all faces
	_fast_view_should_update = false;
}

int ObjMesh::loadObj(const char* filename, bool isNormalGen, bool isNormalize)
{
	gtime_t t1 = gtime_now();

	clear();

	FILE* obj_file_stream;
	int current_material = -1; 
	char *current_token = NULL;
	char current_line[OBJ_LINE_SIZE];
	int line_number = 0;
	// open scene
	obj_file_stream = fopen( filename, "r");
	if(obj_file_stream == 0)
	{
		fprintf(stderr, "Error reading file: %s\n", filename);
		return 0;
	}

	//get name
	strcpy(scene_filename, filename);
	_name = scene_filename;
	int pos1 = _name.find_last_of("\\");
	if(!(pos1>=0 && pos1<(int)_name.size()))
		pos1 = 0;
	int pos2 = _name.find_last_of("/");
	if(!(pos2>=0 && pos2<(int)_name.size()))
		pos2 = 0;
	int pos = std::max(pos1, pos2);
	if (pos) pos++;
	_name = _name.substr(pos, _name.size());

	//parser loop
	while( fgets(current_line, OBJ_LINE_SIZE, obj_file_stream) )
	{
		current_token = strtok( current_line, " \t\n\r");
		line_number++;
		
		//skip comments
		if( current_token == NULL || current_token[0] == '#')
			continue;

		//parse objects
		else if( strcmp(current_token, "v")==0 ) //process vertex
		{
			Float3 v;
			v[0] = (float)atof( strtok(NULL, WHITESPACE));
			v[1] = (float)atof( strtok(NULL, WHITESPACE));
			v[2] = (float)atof( strtok(NULL, WHITESPACE));
			vertex_list.push_back(v);
		}
		
		else if( strcmp(current_token, "vn") == 0 ) //process vertex normal
		{
			Float3 v;
			v[0] = (float)atof( strtok(NULL, WHITESPACE));
			v[1] = (float)atof( strtok(NULL, WHITESPACE));
			v[2] = (float)atof( strtok(NULL, WHITESPACE));
			vertex_normal_list.push_back(v);
		}
		
		else if( strcmp(current_token, "vt") == 0) //process vertex texture
		{
			Float2 v;
			v[0] = (float)atof( strtok(NULL, WHITESPACE));
			v[1] = (float)atof( strtok(NULL, WHITESPACE));
			vertex_texture_list.push_back(v);
		}
		
		else if( strcmp(current_token, "f") == 0) //process face
		{
			int vertex_count;
			obj_face face;
	
			vertex_count = obj_parse_vertex_index(face.vertex_index, face.texture_index, face.normal_index);
			face.vertex_count = vertex_count;
			face.material_index = current_material;
			face_list.push_back(face);
		}
		
		else if( strcmp(current_token, "usemtl") == 0) // usemtl
		{
			char *mtok = strtok(NULL, "\n");
			current_material = -1;
			for(int i=0; i<(int)material_list.size(); i++)
			{
				int tl = strlen(mtok);
				int tr = strlen(material_list[i].name);
				while(material_list[i].name[tr-1]==10) {
					material_list[i].name[tr-1]=0;
					tr--;
				}
				if(strncmp(material_list[i].name, mtok, tl)==0 && tl==tr)
				{
					current_material = i;
				} 
			}
		}
		
		else if( strcmp(current_token, "mtllib") == 0 ) // mtllib
		{
			strncpy(material_filename, strtok(NULL, WHITESPACE), OBJ_FILENAME_LENGTH);
			std::string fullmat = filename;
			int pos1 = fullmat.find_last_of("\\");
			if(!(pos1>=0 && pos1<(int)fullmat.size()))
				pos1 = 0;
			int pos2 = fullmat.find_last_of("/");
			if(!(pos2>=0 && pos2<(int)fullmat.size()))
				pos2 = 0;
			int pos = std::max(pos1, pos2);
			if (pos) pos++;
			fullmat = fullmat.substr(0, pos);
			fullmat.append(material_filename);
			//parse mtl file
			obj_parse_mtl_file(fullmat.c_str());
			continue;
		}
		else
		{
			printf("Unknown command '%s' in scene code at line %i: \"%s\".\n",
					current_token, line_number, current_line);
		}
	}

	fclose(obj_file_stream);

	gtime_t t2 = gtime_now();

	printf("ObjLoaded: \n");
	printf("\tnumber of vertices:%d\n", vertex_list.size());
	printf("\tnumber of normals:%d\n", vertex_normal_list.size());
	printf("\tnumber of tex_uvs:%d\n", vertex_texture_list.size());
	printf("\tnumber of faces:%d\n", face_list.size());
	printf("\tnumber of materials:%d\n", material_list.size());
	printf("Time cost:%f\n", gtime_seconds(t1, t2));

	if(isNormalGen || vertex_normal_list.size() == 0)
	{
		updateNormals();
	}

	vertex_is_selected.resize(vertex_list.size(), 0);
	vertex_color_list.resize(vertex_list.size(), 0.8);

	updateBoundingBox();

	if(isNormalize)
		normalizeModel();

	updateBoundingBox();
	
	return 1;
}

int ObjMesh::loadOff(const char* filename, bool isNormalize)
{
	gtime_t t1 = gtime_now();

	clear();

	FILE* off_file_stream = fopen(filename, "r");

	if (off_file_stream == 0)
	{
		fprintf(stderr, "Error reading file: %s\n", filename);
		return 0;
	}

	//get name
	strcpy(scene_filename, filename);
	_name = scene_filename;
	int pos1 = _name.find_last_of("\\");
	if (!(pos1 >= 0 && pos1<(int)_name.size()))
		pos1 = 0;
	int pos2 = _name.find_last_of("/");
	if (!(pos2 >= 0 && pos2<(int)_name.size()))
		pos2 = 0;
	int pos = std::max(pos1, pos2);
	if (pos) pos++;
	_name = _name.substr(pos, _name.size());

	//parser loop
	fscanf(off_file_stream, "OFF\n");
	int nFaces = 0, nVerts = 0, nEdges = 0;
	fscanf(off_file_stream, "%d %d %d\n", &nVerts, &nFaces, &nEdges);

	vertex_list.resize(nVerts);
	face_list.resize(nFaces);

	for (int iVert = 0; iVert < nVerts; iVert++)
	{
		fscanf(off_file_stream, "%f %f %f\n", &vertex_list[iVert][0],
			&vertex_list[iVert][1], &vertex_list[iVert][2]);
	}// iVert

	for (int iFace = 0; iFace < nFaces; iFace++)
	{
		int nFv = 0;
		fscanf(off_file_stream, "%d", &nFv);
		if (nFv >= MAX_VERT_COUNT)
		{
			printf("error in loading off file: max number of verts per face is too large: %d\n", nFv);
			return 0;
		}
		obj_face &f = face_list[iFace];
		f.vertex_count = nFv;
		f.material_index = -1;
		for (int iFv = 0; iFv < nFv; iFv++)
			fscanf(off_file_stream, " %d", &f.vertex_index[iFv]);
		fscanf(off_file_stream, "\n");
	}// end for iFace

	fclose(off_file_stream);

	gtime_t t2 = gtime_now();

	printf("OffLoaded: \n");
	printf("\tnumber of vertices:%d\n", vertex_list.size());
	printf("\tnumber of faces:%d\n", face_list.size());
	printf("Time cost:%f\n", gtime_seconds(t1, t2));

	updateNormals();
	vertex_is_selected.resize(vertex_list.size(), 0);
	vertex_color_list.resize(vertex_list.size(), 0.8);
	updateBoundingBox();
	if (isNormalize)
		normalizeModel();
	updateBoundingBox();

	return 1;
}

void ObjMesh::updateBoundingBox()
{
	boundingBox[0] = 1e15f;
	boundingBox[1] = -1e15f;
	for(int i=0; i<(int)vertex_list.size(); i++)
	{
		Float3 v = vertex_list[i];
		boundingBox[1][0] = max(v[0], boundingBox[1][0]);
		boundingBox[1][1] = max(v[1], boundingBox[1][1]);
		boundingBox[1][2] = max(v[2], boundingBox[1][2]);

		boundingBox[0][0] = min(v[0], boundingBox[0][0]);
		boundingBox[0][1] = min(v[1], boundingBox[0][1]);
		boundingBox[0][2] = min(v[2], boundingBox[0][2]);
	}
}

void ObjMesh::updateNormals()
{
	if(face_normal_list.size() != face_list.size())
		face_normal_list.resize(face_list.size());
	if(vertex_normal_list.size() != vertex_list.size())
		vertex_normal_list.resize(vertex_list.size());
	for(int i=0; i<(int)vertex_normal_list.size(); i++)
	{
		vertex_normal_list[i] = 0;
	}
	for(int i=0; i<(int)face_list.size(); i++)
	{
		obj_face &f = face_list[i];
		Float3 v = 0;
		for(int j=0; j<=f.vertex_count-3; j++)
		{
			int j1 = (j+1) % f.vertex_count;
			int j2 = (j+2) % f.vertex_count;
			v += ldp::Float3(vertex_list[f.vertex_index[j1]]-vertex_list[f.vertex_index[j]]).cross(
				vertex_list[f.vertex_index[j2]]-vertex_list[f.vertex_index[j]]);
		}
		for(int j=0; j<f.vertex_count; j++)
		{
			vertex_normal_list[f.vertex_index[j]] += v;
			f.normal_index[j] = f.vertex_index[j];
		}
		if(v.length() != 0)
			face_normal_list[i] = v.normalizeLocal();
	}
	for(int i=0; i<(int)vertex_normal_list.size(); i++)
	{
		if(vertex_normal_list[i].length() != 0)
			vertex_normal_list[i].normalizeLocal();
	}
	_fast_view_should_update = true;
}

void ObjMesh::normalizeModel()
{
	Float3 rg = boundingBox[1] - boundingBox[0];
	Float3 center = 0.5f * (boundingBox[1] + boundingBox[0]);
	float diag = max(rg[0], max(rg[1], rg[2]));
	if(diag == 0)
		return;
	for(int i=0; i<(int)vertex_list.size(); i++)
	{
		Float3 &v = vertex_list[i];
		v = (v - center) / diag;
	}
	_fast_view_should_update = true;
}

BMesh* ObjMesh::get_bmesh(bool triangulate)
{
	if (m_bmesh && m_bmesh_triagulate == triangulate)
		return m_bmesh;
	if (m_bmesh)
		delete m_bmesh;
	m_bmesh = new BMesh();
	m_bmeshVerts.clear();

	std::vector<ldp::Int3> faces;

	if (triangulate)
	{
		// init bmesh
		const static ldp::Int3 od[2] = { ldp::Int3(0, 1, 2), ldp::Int3(0, 2, 3) };
		for (int i = 0; i < face_list.size(); i++)
		{
			const ObjMesh::obj_face& f = face_list[i];
			for (int j = 0; j <= f.vertex_count - 3; j++)
			{
				faces.push_back(ldp::Int3(f.vertex_index[od[j][0]],
					f.vertex_index[od[j][1]], f.vertex_index[od[j][2]]));
			}
		}

		m_bmesh->init_triangles(vertex_list.size(), (float*)vertex_list.data(),
			faces.size(), (int*)faces.data());
	}//end if triangulate
	else
	{
		std::vector<int> faces;
		std::vector<int> faceHeaders;

		faceHeaders.push_back(0);
		for (int i_face = 0; i_face < face_list.size(); i_face++)
		{
			const ObjMesh::obj_face& face = face_list[i_face];
			for (int i_v = 0; i_v < face.vertex_count; i_v++)
				faces.push_back(face.vertex_index[i_v]);
			faceHeaders.push_back(faces.size());
		}

		m_bmesh->init_polygons(vertex_list.size(), (float*)vertex_list.data(),
			face_list.size(), faces.data(), faceHeaders.data());
	}

	BMESH_ALL_VERTS(v, viter, *m_bmesh)
	{
		m_bmeshVerts.push_back(v);
	}
	return m_bmesh;
}

void ObjMesh::setSelection(const std::vector<int>& selectedIds)
{
	BMesh* bmesh = get_bmesh(m_bmesh_triagulate);
	bmesh->select_all(BM_VERT, false);

	for (int i_s = 0; i_s < selectedIds.size(); i_s++)
	{
		int vid = selectedIds[i_s];
		bmesh->select_vert(m_bmeshVerts[vid], true);
	}
	update_selection_via_bmesh();
	_fast_view_should_update = true;
}

void ObjMesh::getSelection(std::vector<int>& selectedIds)const
{
	selectedIds.clear();
	for (int i = 0; i < vertex_is_selected.size(); i++)
	if (vertex_is_selected[i])
		selectedIds.push_back(i);
}

bool ObjMesh::isVertexSelected(int i)
{
	if (vertex_is_selected.size() == 0)
		update_selection_via_bmesh();
	return vertex_is_selected[i];
}

void ObjMesh::update_selection_via_bmesh()
{
	if (vertex_is_selected.size() != vertex_list.size())
		vertex_is_selected.resize(vertex_list.size());

	BMesh* bmesh = get_bmesh(m_bmesh_triagulate);
	BMESH_ALL_VERTS(v, iter, *bmesh)
	{
		vertex_is_selected[v->getIndex()] = v->isSelect();
	}
	_fast_view_should_update = true;
}

void ObjMesh::selectSingleVertex(int vert_id, VertexSelectOP op)
{
	BMesh* bmesh = get_bmesh(m_bmesh_triagulate);

	switch (op)
	{
	case ObjMesh::Select_OnlyGiven:
		bmesh->select_all(BM_VERT, false);
		bmesh->select_vert(m_bmeshVerts[vert_id], true);
		break;
	case ObjMesh::Select_Union:
		bmesh->select_vert(m_bmeshVerts[vert_id], true);
		break;
	case ObjMesh::Select_Remove:
		bmesh->select_vert(m_bmeshVerts[vert_id], false);
		break;
	default:
		break;
	}

	update_selection_via_bmesh();
}

void ObjMesh::selectLinkedVertices(int vert_id, VertexSelectOP op)
{
	BMesh* bmesh = get_bmesh(m_bmesh_triagulate);

	switch (op)
	{
	case ObjMesh::Select_OnlyGiven:
		bmesh->select_all(BM_VERT, false);
		bmesh->select_linked_verts(m_bmeshVerts[vert_id], true, false);
		break;
	case ObjMesh::Select_Union:
		bmesh->select_linked_verts(m_bmeshVerts[vert_id], true, false);
		break;
	case ObjMesh::Select_Remove:
		bmesh->select_linked_verts(m_bmeshVerts[vert_id], false, false);
		break;
	default:
		break;
	}

	update_selection_via_bmesh();
}

void ObjMesh::selectAll()
{
	BMesh* bmesh = get_bmesh(m_bmesh_triagulate);
	bmesh->select_all(BM_VERT, true);
	update_selection_via_bmesh();
}

void ObjMesh::selectNone()
{
	BMesh* bmesh = get_bmesh(m_bmesh_triagulate);
	bmesh->select_all(BM_VERT, false);
	update_selection_via_bmesh();
}

void ObjMesh::selectShortestPath(int vert_id_1, int vert_id_2, bool disablePathIntersect)
{
	BMesh* bmesh = get_bmesh(m_bmesh_triagulate);
	bmesh->select_shortest_path(m_bmeshVerts[vert_id_1], m_bmeshVerts[vert_id_2], disablePathIntersect);
	update_selection_via_bmesh();
}

void ObjMesh::selectInnerRegion(int vert_id)
{
	BMesh* bmesh = get_bmesh(m_bmesh_triagulate);
	bmesh->select_linked_verts(m_bmeshVerts[vert_id], true, true);
	update_selection_via_bmesh();
}

bool ObjMesh::hasSelectedVert()const
{
	for (int i = 0; i < vertex_is_selected.size(); i++)
	if (vertex_is_selected[i])
		return true;
	return false;
}

void ObjMesh::getSubMesh(const std::vector<int>& validVertexIdx, ObjMesh* subMesh,
	std::vector<int>* faceIdToValidFaceId)const
{
	if (subMesh == 0)
	{
		printf("error: null pointer provided!");
		assert(0);
		return;
	}

	std::vector<int> validIdSorted(validVertexIdx.begin(), validVertexIdx.end());
	std::sort(validIdSorted.begin(), validIdSorted.end());

	// construct the output mesh
	subMesh->clear();
	for (int i = 0; i < validVertexIdx.size(); i++)
	{
		int id = validVertexIdx[i];
		if (id < 0 || id >= vertex_list.size())
		{
			printf("error: not valid vertex id: %d", id);
			assert(0);
			return;
		}
		subMesh->vertex_list.push_back(vertex_list[id]);
		subMesh->vertex_color_list.push_back(vertex_color_list[id]);
	}

	std::vector<int> vertIdxToValidIdx(vertex_list.size(), -1);
	for (int i = 0; i < validVertexIdx.size(); i++)
		vertIdxToValidIdx[validVertexIdx[i]] = i;

	if (faceIdToValidFaceId)
		faceIdToValidFaceId->resize(face_list.size(), -1);
	for (int i_face = 0; i_face < face_list.size(); i_face++)
	{
		ObjMesh::obj_face f = face_list[i_face];
		bool valid = true;
		for (int i = 0; i < f.vertex_count; i++)
		{
			if (!std::binary_search(validIdSorted.begin(), validIdSorted.end(), f.vertex_index[i]))
			{
				valid = false;
				break;
			}
		}
		if (valid)
		{
			for (int i = 0; i < f.vertex_count; i++)
				f.vertex_index[i] = vertIdxToValidIdx[f.vertex_index[i]];
			subMesh->face_list.push_back(f);
			if (faceIdToValidFaceId)
				(*faceIdToValidFaceId)[i_face] = subMesh->face_list.size() - 1;
		}
	}

	subMesh->updateNormals();
	subMesh->updateBoundingBox();
	subMesh->selectNone();
}

void ObjMesh::saveObj(const char* path)const
{
	FILE* pFile = fopen(path, "w");
	if (!pFile)
		throw std::exception((std::string("Open file failed: ") + path).c_str());

	if (material_list.size() > 0)
		fprintf(pFile, "mtllib %s\n", material_filename);
	fprintf(pFile, "#number of vertices: %d\n", vertex_list.size());
	for (int i = 0; i < vertex_list.size(); i++)
	{
		ldp::Float3 v = vertex_list[i];
		fprintf(pFile, "v %f %f %f\n", v[0], v[1], v[2]);
	}
	fprintf(pFile, "#number of normals: %d\n", vertex_normal_list.size());
	for (int i = 0; i < vertex_normal_list.size(); i++)
	{
		ldp::Float3 v = vertex_normal_list[i];
		fprintf(pFile, "vn %f %f %f\n", v[0], v[1], v[2]);
	}
	fprintf(pFile, "#number of texcoords: %d\n", vertex_texture_list.size());
	for (int i = 0; i < vertex_texture_list.size(); i++)
	{
		ldp::Float2 v = vertex_texture_list[i];
		fprintf(pFile, "vt %f %f\n", v[0], v[1]);
	}

	int last_mat_id = -1;
	for (int i = 0; i < face_list.size(); i++)
	{
		const ObjMesh::obj_face& f = face_list[i];
		if (f.material_index != last_mat_id && f.material_index < material_list.size())
		{
			last_mat_id = f.material_index;
			fprintf(pFile, "usemtl %s\n", material_list[f.material_index].name);
		}

		fprintf(pFile, "f ");
		for (int k = 0; k < f.vertex_count; k++)
		{
			fprintf(pFile, "%d/", f.vertex_index[k]+1);
			if (f.texture_index[k] >= 0 && f.texture_index[k] < vertex_texture_list.size())
				fprintf(pFile, "%d", f.texture_index[k]+1);
			fprintf(pFile, "/");
			if (f.normal_index[k] >= 0 && f.normal_index[k] < vertex_normal_list.size())
				fprintf(pFile, "%d", f.normal_index[k]+1);
			if (k != f.vertex_count-1)
				fprintf(pFile, " ");
		}
		fprintf(pFile, "\n");
	}

	fclose(pFile);
}

int ObjMesh::obj_parse_vertex_index(int *vertex_index, int *texture_index, int *normal_index)const
{
	char *temp_str;
	char *token;
	int vertex_count = 0;
	
	while( (token = strtok(NULL, WHITESPACE)) != NULL)
	{
		if(texture_index != NULL)
			texture_index[vertex_count] = -1;
		if(normal_index != NULL)
		normal_index[vertex_count] = -1;

		vertex_index[vertex_count] = atoi( token ) - 1;
		
		if(strstr(token, "//") != 0)  //normal only
		{
			temp_str = strchr(token, '/');
			temp_str++;
			normal_index[vertex_count] = atoi( ++temp_str ) - 1;
		}
		else if(strstr(token, "/") != 0)
		{
			temp_str = strchr(token, '/');
			texture_index[vertex_count] = atoi( ++temp_str ) - 1;

			if(strstr(temp_str, "/") != 0)
			{
				temp_str = strchr(temp_str, '/');
				normal_index[vertex_count] = atoi( ++temp_str ) - 1;
			}
		}
		
		vertex_count++;
	}

	return vertex_count;
}

int ObjMesh::obj_parse_mtl_file(const char *filename)
{
	int line_number = 0;
	char *current_token;
	char current_line[OBJ_LINE_SIZE];
	char material_open = 0;
	obj_material *current_mtl = 0;
	FILE *mtl_file_stream;

	// open scene
	mtl_file_stream = fopen( filename, "r");
	if(mtl_file_stream == 0)
	{
		fprintf(stderr, "Error reading file: %s\n", filename);
		return 0;
	}
		
	material_list.clear();

	while( fgets(current_line, OBJ_LINE_SIZE, mtl_file_stream) )
	{
		current_token = strtok( current_line, " \t\n\r");
		line_number++;
		
		//skip comments
		if( current_token == NULL || strcmp(current_token, "//")==0 || strcmp(current_token, "#")==0)
			continue;
		

		//start material
		else if( strcmp(current_token, "newmtl")==0)
		{
			material_open = 1;
			material_list.push_back(obj_material());
			current_mtl = &material_list[material_list.size()-1];
			
			// get the name
			strncpy(current_mtl->name, strtok(NULL, "\n"), MATERIAL_NAME_SIZE);
		}
		
		//ambient
		else if( strcmp(current_token, "Ka")==0 && material_open)
		{
			current_mtl->amb[0] = (float)atof( strtok(NULL, " \t"));
			current_mtl->amb[1] = (float)atof( strtok(NULL, " \t"));
			current_mtl->amb[2] = (float)atof( strtok(NULL, " \t"));
		}

		//diff
		else if( strcmp(current_token, "Kd")==0 && material_open)
		{
			current_mtl->diff[0] = (float)atof( strtok(NULL, " \t"));
			current_mtl->diff[1] = (float)atof( strtok(NULL, " \t"));
			current_mtl->diff[2] = (float)atof( strtok(NULL, " \t"));
		}
		
		//specular
		else if( strcmp(current_token, "Ks")==0 && material_open)
		{
			current_mtl->spec[0] = (float)atof( strtok(NULL, " \t"));
			current_mtl->spec[1] = (float)atof( strtok(NULL, " \t"));
			current_mtl->spec[2] = (float)atof( strtok(NULL, " \t"));
		}
		//shiny
		else if( strcmp(current_token, "Ns")==0 && material_open)
		{
			current_mtl->shiny = (float)atof( strtok(NULL, " \t"));
		}
		//transparent
		else if( (strcmp(current_token, "Tr")==0 || strcmp(current_token, "d")==0) && material_open)
		{
			current_mtl->trans = (float)atof( strtok(NULL, " \t"));
		}
		//reflection
		else if( strcmp(current_token, "r")==0 && material_open)
		{
			current_mtl->reflect = (float)atof(strtok(NULL, " \t"));
		}
		else if( strcmp(current_token, "ra")==0 && material_open)
		{
			current_mtl->refract = (float)atof(strtok(NULL, " \t"));
		}
		//glossy
		else if( strcmp(current_token, "sharpness")==0 && material_open)
		{
			current_mtl->glossy = (float)atof(strtok(NULL, " \t"));
		}
		//refract index
		else if( strcmp(current_token, "Ni")==0 && material_open)
		{
			current_mtl->refract_index = (float)atof(strtok(NULL, " \t"));
		}
		// illumination type
		else if( strcmp(current_token, "illum")==0 && material_open)
		{
		}
		// texture map
		else if( (	strcmp(current_token, "map_Ka")==0 ||
					strcmp(current_token, "map_Kd")==0	)
			&& material_open)
		{
			strncpy(current_mtl->texture_filename, strtok(NULL, " \t"), OBJ_FILENAME_LENGTH);
			//remove ' '
			for (int i=strlen(current_mtl->texture_filename)-1; i>=0; i--)
			{
				if(current_mtl->texture_filename[i]!= ' '
					&&current_mtl->texture_filename[i]!= '\n')
					break;
				current_mtl->texture_filename[i] = 0;
			}

			//to full path
			std::string fullmat = filename;
			int pos1 = fullmat.find_last_of("\\");
			if(!(pos1>=0 && pos1<(int)fullmat.size()))
				pos1 = 0;
			int pos2 = fullmat.find_last_of("/");
			if(!(pos2>=0 && pos2<(int)fullmat.size()))
				pos2 = 0;
			int pos = std::max(pos1, pos2);
			if (pos) pos++;
			fullmat = fullmat.substr(0, pos);
			fullmat.append(current_mtl->texture_filename);

			CFreeImage img;
			if (img.load(fullmat.c_str()))
			{
				current_mtl->image_width = img.width();
				current_mtl->image_height = img.height();
				current_mtl->image_channel = 4;
				current_mtl->image.resize(img.width() * img.height() * 4);
				unsigned char* mtlimg = &current_mtl->image[0];
				const unsigned char* srcimg = img.getBits();
				int len = img.width() * img.height();
				for (int i=0; i<len; i++)
				{
					if (img.nChannels() == 1)
					{
						*mtlimg++ = srcimg[i];
						*mtlimg++ = srcimg[i];
						*mtlimg++ = srcimg[i];
						*mtlimg++ = srcimg[i];
					}
					else if (img.nChannels() == 3)
					{
						*mtlimg++ = srcimg[i*3+2];
						*mtlimg++ = srcimg[i*3+1];
						*mtlimg++ = srcimg[i*3];
						*mtlimg++ = 255;
					}
					else if (img.nChannels() == 4)
					{
						*mtlimg++ = srcimg[i*4+2];
						*mtlimg++ = srcimg[i*4+1];
						*mtlimg++ = srcimg[i*4];
						*mtlimg++ = srcimg[i*4+3];
					}
				}

				current_mtl->generateTextures();
			}
			else{
				fprintf(stderr, "Load Texture failed: %s\n", fullmat.c_str());
			}
		}
		else
		{
			fprintf(stderr, "Unknown command '%s' in material file %s at line %i:\n\t%s\n",
					current_token, filename, line_number, current_line);
			//return 0;
		}
	}
	fclose(mtl_file_stream);

	return 1;

}

