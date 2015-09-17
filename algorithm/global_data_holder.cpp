#include "global_data_holder.h"
#include <fstream>
GlobalDataHolder g_dataholder;
using namespace ldp;

void GlobalDataHolder::init()
{
	m_kinect.InitKinect(1);
	m_depth_h.resize(dfusion::KINECT_WIDTH*dfusion::KINECT_HEIGHT);
	m_depth_d.create(dfusion::KINECT_HEIGHT, dfusion::KINECT_WIDTH);

	m_view_normalmap = false;
	m_lights.pos = make_float3(0, 0, 0);
	m_lights.amb = make_float3(0.3, 0.3, 0.3);
	m_lights.diffuse = make_float3(0.8, 0.8, 0.8);
}

void GlobalDataHolder::saveDepth(const std::vector<dfusion::depthtype>& depth_h, std::string filename)
{
	if (depth_h.size() != dfusion::KINECT_WIDTH*dfusion::KINECT_HEIGHT)
		throw std::exception("saveDepth: size not matched!");

	std::ofstream stm(filename, std::ios_base::binary);
	if (stm.fail())
		throw std::exception(("save failed: "+ filename).c_str());

	int w = dfusion::KINECT_WIDTH;
	int h = dfusion::KINECT_HEIGHT;
	stm.write((const char*)&w, sizeof(int));
	stm.write((const char*)&h, sizeof(int));
	stm.write((const char*)depth_h.data(), depth_h.size()*sizeof(dfusion::depthtype));

	stm.close();
}

void GlobalDataHolder::loadDepth(std::vector<dfusion::depthtype>& depth_h, std::string filename)
{
	std::ifstream stm(filename, std::ios_base::binary);
	if (stm.fail())
		throw std::exception(("load failed: " + filename).c_str());

	depth_h.resize(dfusion::KINECT_WIDTH*dfusion::KINECT_HEIGHT);
	std::fill(depth_h.begin(), depth_h.end(), 0);
	int w = 0, h = 0;
	stm.read((char*)&w, sizeof(int));
	stm.read((char*)&h, sizeof(int));
	if (w != dfusion::KINECT_WIDTH || h != dfusion::KINECT_HEIGHT)
		throw std::exception("loadDepth: size not matched!");
	stm.read((char*)depth_h.data(), depth_h.size()*sizeof(dfusion::depthtype));

	stm.close();
}

void GlobalDataHolder::meshCopy(const dfusion::GpuMesh& gmesh, ObjMesh& mesh)
{
	mesh.clear();

	mesh.vertex_list.resize(gmesh.verts.size());
	mesh.vertex_normal_list.resize(gmesh.normals.size());
	gmesh.verts.download((float3*)mesh.vertex_list.data());
	gmesh.normals.download((float3*)mesh.vertex_normal_list.data());

	mesh.face_list.resize(mesh.vertex_list.size() / 3);
	for (size_t fid = 0; fid < mesh.face_list.size(); fid++)
	{
		ObjMesh::obj_face &f = mesh.face_list[fid];
		f.material_index = -1;
		f.vertex_count = 3;
		f.vertex_index[0] = fid * 3;
		f.vertex_index[1] = fid * 3 + 1;
		f.vertex_index[2] = fid * 3 + 2;
	}
}