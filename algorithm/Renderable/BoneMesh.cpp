#include "BoneMesh.h"
#include "util.h"
#include <fstream>
#include <algorithm>
#include "Error.h"


inline ldp::Float3 transFromMat4(const ldp::Mat4f& M)
{
	return ldp::Float3(M(0,3), M(1,3), M(2,3))/M(3,3);
}

ldp::Mat4f BoneMesh::BoneVertex::getCombinedTM(int frameId)const
{
	// accumulate weights of leafs
	ldp::Mat4f TM = 0.f;
	if (nodes.size() == 0)
		return TM;

	if (!nodes[0]->isUseDualQuat())
	{
		for (size_t i = 0; i < nodes.size(); i++)
			TM += weights[i] * nodes[i]->getGlobalTM(frameId) * nodes[i]->getInitGlobalTM().inv();
	}
	else
	{
		ldp::DualQuaternionF dq;
		for (size_t i = 0; i < nodes.size(); i++)
		{
			ldp::DualQuaternionF dq_i = nodes[i]->getDualQuat(frameId);
			dq += weights[i] * dq_i;
		}

		dq.getTransform(TM);
	}

	return TM;
}

ldp::Float3 BoneMesh::BoneVertex::getGlobalPos(int frameId)const
{
	ldp::Float4 l(initGlobalPos[0], initGlobalPos[1], initGlobalPos[2], 1);
	ldp::Float4 p = getCombinedTM(frameId) * l;
	return ldp::Float3(p[0], p[1], p[2]) / p[3];
}

BoneMesh::BoneMesh() : ObjMesh()
{
	m_hasBones = false;
	m_frameId = -1;
	m_numFrames = 0;
}

BoneMesh::~BoneMesh()
{
	clear();
}

BoneMesh::BoneMesh(const BoneMesh& rhs)
{
	cloneFrom(&rhs);
}

void BoneMesh::cloneFrom(const ObjMesh* rhs)
{
	ObjMesh::cloneFrom(rhs);
	if (rhs == this)
		return;

	if (rhs->getMeshType() != TYPE_BONE_MESH)
		return;

	BoneMesh* bmesh = (BoneMesh*)rhs;

	m_hasBones = bmesh->m_hasBones;
	m_frameId = bmesh->m_frameId;
	m_numFrames = bmesh->m_numFrames;

	m_nodes.assign(bmesh->m_nodes.begin(), bmesh->m_nodes.end());
	updateNodeSet();
	m_boneVertices.assign(bmesh->m_boneVertices.begin(), bmesh->m_boneVertices.end());
	updateBoneVertexSet();

	for (int i_nd = 0; i_nd < m_nodes.size(); i_nd++)
	{
		Node& nd = m_nodes[i_nd];
		if (nd.parent)
			nd.parent = getNodeById(nd.parent->id);
		for (int j = 0; j < nd.children.size(); j++)
			nd.children[j] = getNodeById(nd.children[j]->id);
	}

	for (int i_v = 0; i_v < m_boneVertices.size(); i_v++)
	{
		BoneVertex& v = m_boneVertices[i_v];
		for (int j = 0; j < v.nodes.size(); j++)
			v.nodes[j] = getNodeById(v.nodes[j]->id);
	}

	for (int i_r = 0; i_r < m_pRootNodes.size(); i_r++)
		m_pRootNodes[i_r] = getNodeById(m_pRootNodes[i_r]->id);
}

void BoneMesh::clear()
{
	ObjMesh::clear();
	m_pRootNodes.clear();
	m_nodes.clear();
	m_boneVertices.clear();
	m_nodesSet.clear(); 
	m_boneVertsSet.clear();
	m_frameId = -1;
	m_numFrames = 0;
}

void BoneMesh::setUseDualQuaternion(bool use)
{
	for (size_t i = 0; i < m_nodes.size(); i++)
		m_nodes[i].setUseDualQuat(use);
	updateVertexPosByBones(m_frameId);
}

void BoneMesh::updateVertexPosByBones(int frameId)
{
	if (frameId < 0 || frameId >= m_numFrames)
		return;
	for (int i_r = 0; i_r < m_pRootNodes.size(); i_r++)
		m_pRootNodes[i_r]->setGlobalTM(m_pRootNodes[i_r]->getGlobalTM(frameId), frameId);
	
	#pragma omp parallel for
	for (int i = 0; i < m_boneVertices.size(); i++)
	{
		int id = m_boneVertices[i].id;
		vertex_list[id] = m_boneVertices[i].getGlobalPos(frameId);
	}
	
	updateNormals();
}

void BoneMesh::clearAllFrames()
{
	for (int i = 0; i < m_nodes.size(); i++)
		m_nodes[i].clearAllFrames();
	m_numFrames = 0;
	m_frameId = -1;
}

void BoneMesh::addFrames(int num)
{
	for (int i = 0; i < m_nodes.size(); i++)
		m_nodes[i].addFrames(num);
	m_numFrames += num;
}

void BoneMesh::setFrameId(int id)
{
	if (id < 0 || id >= m_numFrames)
		return;
	m_frameId = id;
	updateVertexPosByBones(m_frameId);
}

static void getCommonParentNodeFromSet(BoneMesh::NodeSetConst& nodes)
{
	// 
	std::vector<const BoneMesh::Node*> curNodes(nodes.begin(), nodes.end());
	for (int i = 0; i < curNodes.size(); i++)
	{
		const BoneMesh::Node* nd = curNodes[i];
		for (int j = 0; j < nd->getNumChild(); j++)
		{
			const BoneMesh::Node* cnd = nd->getChild(j);
			BoneMesh::NodeSetConst::iterator it = nodes.find(cnd);
			if (it != nodes.end())
				nodes.erase(it);
		}
	}

	if (nodes.size() == 1)
		return;

	if (nodes.empty())
	{
		throw std::exception("cycle detected in the bone hierachy!");
	}

	// 
	BoneMesh::NodeSetConst parents;
	for (BoneMesh::NodeSetConst::iterator it = nodes.begin(); it != nodes.end(); ++it)
	{
		if ((*it)->getParent())
			parents.insert((*it)->getParent());
	}

	if (!parents.empty())
	{
		nodes.swap(parents);
		if (nodes.size() > 1)
			getCommonParentNodeFromSet(nodes);
	}
}

static void getAllChildren(const BoneMesh::Node* root, BoneMesh::NodeSetConst& nodeSet)
{
	for (int i = 0; i < root->getNumChild(); i++)
	{
		nodeSet.insert(root->getChild(i));
		getAllChildren(root->getChild(i), nodeSet);
	}
}

void BoneMesh::getSubMesh(const std::vector<int>& validVertexIdx,
	ObjMesh* subMesh, std::vector<int>* faceIdToValidFaceId)const
{
	ObjMesh::getSubMesh(validVertexIdx, subMesh, faceIdToValidFaceId);

	if (subMesh->getMeshType() != Renderable::TYPE_BONE_MESH || !hasBones())
		return;
	BoneMesh * boneSubMesh = (BoneMesh*)subMesh;

	std::vector<int> vertIdToValidVertId(vertex_list.size(), -1);
	for (int i = 0; i < validVertexIdx.size(); i++)
		vertIdToValidVertId[validVertexIdx[i]] = i;

	// get common root nodes
	NodeSetConst validNodeSet;
	NodeSetConst rootNodeSet;
	for (int i = 0; i < validVertexIdx.size(); i++)
	{
		const BoneVertex * v = getBoneVertexById(validVertexIdx[i]);
		for (int j = 0; j < v->nodes.size(); j++)
		{
			validNodeSet.insert(v->nodes[j]);
			rootNodeSet.insert(v->nodes[j]);
		}
	}

	getCommonParentNodeFromSet(rootNodeSet);
	
	for (NodeSetConst::iterator it = rootNodeSet.begin(); it != rootNodeSet.end(); ++it)
	{
		if (validNodeSet.find(*it) == validNodeSet.end())
			validNodeSet.insert(*it);
	}

	// all nodes
	boneSubMesh->m_nodes.clear();
	boneSubMesh->m_nodes.reserve(validNodeSet.size());
	for (NodeSetConst::iterator it = validNodeSet.begin(); it != validNodeSet.end(); ++it)
		boneSubMesh->m_nodes.push_back(**it);

	boneSubMesh->updateNodeSet();

	// all bone vertex
	boneSubMesh->m_boneVertices.clear();
	for (int i = 0; i < validVertexIdx.size(); i++)
	{
		BoneVertex v = *getBoneVertexById(validVertexIdx[i]);
		v.id = i;
		boneSubMesh->m_boneVertices.push_back(v);
	}
	boneSubMesh->updateBoneVertexSet();

	// children-parent relationship
	for (int i_nd = 0; i_nd < boneSubMesh->m_nodes.size(); i_nd++)
	{
		Node& node = boneSubMesh->m_nodes[i_nd];
		std::vector<Node*> tmpChildren;
		node.children.swap(tmpChildren);
		node.childrenIds.clear();
		for (int i = 0; i < tmpChildren.size(); i++)
		{
			if (validNodeSet.find(tmpChildren[i]) != validNodeSet.end())
			{
				node.children.push_back(boneSubMesh->getNodeById(tmpChildren[i]->id));
				node.childrenIds.push_back(node.children.back()->id);
			}
		}
		node.parent = boneSubMesh->getNodeById(node.parent->id);
	}

	// vertex-bone relationship
	for (int i_v = 0; i_v < boneSubMesh->m_boneVertices.size(); i_v++)
	{
		BoneVertex& v = boneSubMesh->m_boneVertices[i_v];
		std::vector<Node*> tmpNodes;
		tmpNodes.swap(v.nodes);
		v.nodeIds.clear();
		for (int j = 0; j < tmpNodes.size(); j++)
		{
			Node* nd = boneSubMesh->getNodeById(tmpNodes[j]->id);
			if (nd)
			{
				v.nodes.push_back(nd);
				v.nodeIds.push_back(nd->id);
			}
		}
	}

	// root nodes
	boneSubMesh->m_pRootNodes.clear();
	for (int i = 0; i < boneSubMesh->m_nodes.size(); i++)
	if (boneSubMesh->m_nodes[i].parent == 0)
		boneSubMesh->m_pRootNodes.push_back(&boneSubMesh->m_nodes[i]);

	boneSubMesh->m_hasBones = true;
	boneSubMesh->m_frameId = m_frameId;
	boneSubMesh->m_numFrames = m_numFrames;
}

int BoneMesh::loadObj(const char* objname, bool isNormalGen, bool isNormalize)
{
	clear();

	std::string path, name, ext;
	ldp::fileparts(objname, path, name, ext);

	std::string skelname = ldp::fullfile(path, name + ".skeleton");
	int suc = ObjMesh::loadObj(objname, isNormalGen, isNormalize);
	if (!suc)
		return 0;

	suc = loadSkeleton(skelname.c_str());
	if (!suc)
		m_hasBones = false;
	else
		suc = checkData();
	if (!suc)
		m_hasBones = false;
	else
		m_hasBones = true;

	if (m_hasBones)
	{
		m_frameId = 0;
		m_numFrames = 0;
		if (m_nodes.size() > 0)
			m_numFrames = m_nodes[0].globalTMs.size();
		updateVertexPosByBones(m_frameId);
	}

	return 1;
}

void BoneMesh::saveObj(const char* objname)const
{
	ObjMesh::saveObj(objname);

	std::string path, name, ext;
	ldp::fileparts(objname, path, name, ext);

	std::string skelName = ldp::fullfile(path, name + ".skeleton");
	FILE* pFile = fopen(skelName.c_str(), "w");
	if (pFile == 0)
		throw std::exception((std::string("error in open file: " + skelName).c_str()));

	fprintf(pFile, "Multiple Frame: %d\n", m_numFrames);
	for (int i_b = 0; i_b < m_nodes.size(); i_b++)
	{
		const Node& node = m_nodes[i_b];
		fprintf(pFile, "New Bone:\n");
		fprintf(pFile, "bone_name: %s\n", node.name.c_str());
		fprintf(pFile, "bone_id: %d\n", node.id);
		fprintf(pFile, "child_num: %d\n", node.children.size());
		for (int i_c = 0; i_c < node.children.size(); i_c++)
			fprintf(pFile, "child_id: %d\n", node.children[i_c]->id);
		fprintf(pFile, "Init-TM:\n");
		for (int y = 0; y < 4; y++)
		{
			fprintf(pFile, "%f %f %f %f\n", node.initGlobalTM(y, 0), node.initGlobalTM(y, 1), node.initGlobalTM(y, 2),
				node.initGlobalTM(y, 3));
		}
		fprintf(pFile, "Local-TM:\n");
		ldp::Mat4f localTM = node.getLocalTM(0);
		for (int y = 0; y < 4; y++)
		{
			fprintf(pFile, "%f %f %f %f\n", localTM(y, 0), localTM(y, 1), localTM(y, 2),
				localTM(y, 3));
		}
		fprintf(pFile, "Global-TM:\n");
		for (int i_frame = 0; i_frame < m_numFrames; i_frame++)
		{
			for (int y = 0; y < 4; y++)
			{
				fprintf(pFile, "%f %f %f %f\n", 
					node.globalTMs[i_frame](y, 0),
					node.globalTMs[i_frame](y, 1),
					node.globalTMs[i_frame](y, 2),
					node.globalTMs[i_frame](y, 3));
			}
			if (i_frame != m_numFrames - 1)
				fprintf(pFile, "\n");
		}
	}

	for (int i_v = 0; i_v < m_boneVertices.size(); i_v++)
	{
		const BoneVertex& v = m_boneVertices[i_v];
		fprintf(pFile, "vertex: %d 0.0 0.0 0.0\n", v.id);
		for (int k = 0; k < v.nodes.size(); k++)
		{
			fprintf(pFile, "bone_id_w: %d %f\n", v.nodes[k]->id, v.weights[k]);
		}
	}

	fclose(pFile);
}

void BoneMesh::setBones(const std::vector<Node>& nodes, const std::vector<BoneVertex>& boneVerts)
{
	if (nodes.size() == 0)
		return;
	m_numFrames = nodes[0].getNumFrames();

	m_nodes.assign(nodes.begin(), nodes.end());
	updateNodeSet();

	// find all root nodes: nodes that do not have parents
	m_pRootNodes.clear();
	for (int i_nd = 0; i_nd < m_nodes.size(); i_nd++)
	{
		Node& nd = m_nodes[i_nd];
		if (nd.parent == NULL)
		{
			m_pRootNodes.push_back(&nd);
			nd.root_virtual_parent_GlobalTM = nd.root_virtual_parent_GlobalTM;
		}
	}

	// make sure the bone-vertices are unique, then build a set.
	m_boneVertices.assign(boneVerts.begin(), boneVerts.end());
	updateBoneVertexSet();

	// update pointers
	for (int i_node = 0; i_node < m_nodes.size(); i_node++)
	{
		Node& node = m_nodes[i_node];
		if (node.parent)
			node.parent = getNodeById(node.parent->id);

		if (node.getNumFrames() != m_numFrames)
		{
			ldp::Logger::error("error: un matched bones for each frame!\n");
			throw std::exception("error: un matched bones for each frame!\n");
		}

		node.children.resize(node.childrenIds.size());
		for (int i_cld = 0; i_cld < node.children.size(); i_cld++)
			node.children[i_cld] = getNodeById(node.childrenIds[i_cld]);
	}
	for (int i_vert = 0; i_vert < m_boneVertices.size(); i_vert++)
	{
		BoneVertex& vert = m_boneVertices[i_vert];
		vert.nodes.resize(vert.nodeIds.size());
		assert(vert.nodeIds.size() == vert.weights.size());
		for (int i_nd = 0; i_nd < vert.nodeIds.size(); i_nd++)
			vert.nodes[i_nd] = getNodeById(vert.nodeIds[i_nd]);
	}

	//checkData();

	updateVertexPosByBones(0);

	m_hasBones = true;
}

int BoneMesh::loadSkeleton(const char* skelname)
{
	std::ifstream stream(skelname);
	if (stream.fail())
		return 0;

	std::string lineBuffer;
	while (std::getline(stream, lineBuffer))
	{
		std::string lineLabel = getLabel(lineBuffer);
		if (lineLabel == "Multiple Frame")
		{
			int r = sscanf(lineBuffer.c_str(), "%d", &m_numFrames);
			if (r != 1)
			{
				ldp::Logger::error("read in multi-frame configuration failed!\n");
				return 0;
			}
			printf("number of frames: %d\n", m_numFrames);
		}
		if (lineLabel == "New Modifier")
		{
			// skip the next 2 lines
			std::getline(stream, lineBuffer);
			std::getline(stream, lineBuffer);
		}
		else if (lineLabel == "New Bone")
		{
			m_nodes.push_back(Node());
			m_nodes.back().parent = NULL;
			readInOneNode(stream, m_nodes.back());
		}
		else if (lineLabel == "vertex")
		{
			m_boneVertices.push_back(BoneVertex());
			ldp::Float3 p;
			int r = sscanf(lineBuffer.c_str(), "%d %f %f %f", &m_boneVertices.back().id, &p[0], &p[1], &p[2]);
			if (r != 4)
			{
				ldp::Logger::error("read in vertex id failed!\n");
				return 0;
			}
			// ignore this p, it is not consistent with the skin.
			//m_boneVertices.back().initGlobalPos = p;
		}
		else if (lineLabel == "bone_id_w")
		{
			BoneVertex& v = m_boneVertices.back();
			int id;
			float w;
			int r = sscanf(lineBuffer.c_str(), "%d %f", &id, &w);
			if (r != 2)
			{
				ldp::Logger::error("read in vertex weights failed!\n");
				return 0;
			}
			if (w != 0)
			{
				v.nodeIds.push_back(id);
				v.weights.push_back(w);
			}
		}
	}

	// make sure the nodes are unique, then build a set.
	updateNodeSet();

	for (int i_nd = 0; i_nd < m_nodes.size(); i_nd++)
	{
		Node& nd = m_nodes[i_nd];
		nd.children.resize(nd.childrenIds.size(), NULL);
		for (int i_child = 0; i_child < nd.childrenIds.size(); i_child++)
		{
			Node cnd;
			cnd.id = nd.childrenIds[i_child];
			NodeSet::iterator it = m_nodesSet.find(&cnd);
			if (it != m_nodesSet.end())
			{
				nd.children[i_child] = *it;
				(*it)->parent = &nd;
			}
			else
			{
				//ldp::Logger::warning("NULL children referenced!\n");
			}
		}

		// remove null reference
		std::vector<int> cids(nd.childrenIds.begin(), nd.childrenIds.end());
		std::vector<Node*> cptrs(nd.children.begin(), nd.children.end());
		nd.children.clear();
		nd.childrenIds.clear();
		for (int i = 0; i < cptrs.size(); i++)
		{
			if (cptrs[i])
			{
				nd.children.push_back(cptrs[i]);
				nd.childrenIds.push_back(cids[i]);
			}
		}
	}

	// find all root nodes: nodes that do not have parents
	for (int i_nd = 0; i_nd < m_nodes.size(); i_nd++)
	{
		Node& nd = m_nodes[i_nd];
		if (nd.parent == NULL)
		{
			m_pRootNodes.push_back(&nd);
		}
	}

	// make sure the bone-vertices are unique, then build a set.
	updateBoneVertexSet();

	for (int i_v = 0; i_v < m_boneVertices.size(); i_v++)
	{
		BoneVertex& v = m_boneVertices[i_v];
		Node pnd;
		v.nodes.resize(v.nodeIds.size());
		for (int k = 0; k < v.nodeIds.size(); k++)
		{
			pnd.id = v.nodeIds[k];
			NodeSet::iterator it = m_nodesSet.find(&pnd);
			v.nodes[k] = *it;
			assert((*it)->id == v.nodeIds[k]);

		}
	}

	// calculate local vertex position and find leaf nodes
	for (int i_v = 0; i_v < m_boneVertices.size(); i_v++)
	{
		BoneVertex& v = m_boneVertices[i_v];

		// calc local position
		ldp::Mat4f MT;
		for (int i_nd = 0; i_nd < v.nodes.size(); i_nd++)
			MT += v.weights[i_nd] * (v.nodes[i_nd]->getGlobalTM(0) * v.nodes[i_nd]->getInitGlobalTM().inv());
		ldp::Float4 gv(vertex_list[v.id][0], vertex_list[v.id][1], vertex_list[v.id][2], 1);
		gv = MT.inv() * gv;
		v.initGlobalPos = ldp::Float3(gv[0], gv[1], gv[2]);
	}

	return 1;
}

void BoneMesh::updateNodeSet()
{
	// build child-parent ponter relationship
	float dataScale = (boundingBox[1] - boundingBox[0]).length();
	float eps = 1e-4f * dataScale;
	m_nodesSet.clear();
	for (int i_nd = 0; i_nd < m_nodes.size(); i_nd++)
	{
		NodeSet::iterator it = m_nodesSet.find(&m_nodes[i_nd]);
		if (it != m_nodesSet.end())
		{
			ldp::Mat4f R1 = (*it)->getGlobalTM(0);
			ldp::Mat4f R2 = m_nodes[i_nd].getGlobalTM(0);
			if ((R1 - R2).norm() > eps)
				ldp::Logger::warning("Conflicated duplicated nodes: %d\n", (*it)->id);
		}
		m_nodesSet.insert(&m_nodes[i_nd]);
	}
	std::vector<Node> tmpNodes;
	for (NodeSet::iterator it = m_nodesSet.begin(); it != m_nodesSet.end(); ++it)
		tmpNodes.push_back(**it);
	m_nodesSet.clear();
	m_nodes.assign(tmpNodes.begin(), tmpNodes.end());
	for (int i_nd = 0; i_nd < m_nodes.size(); i_nd++)
		m_nodesSet.insert(&m_nodes[i_nd]);
}

void BoneMesh::updateBoneVertexSet()
{
	std::sort(m_boneVertices.begin(), m_boneVertices.end());
	std::vector<BoneVertex>::iterator it = std::unique(m_boneVertices.begin(), m_boneVertices.end());
	m_boneVertices.resize(it - m_boneVertices.begin());

	m_boneVertsSet.clear();
	for (int i_v = 0; i_v < m_boneVertices.size(); i_v++)
		m_boneVertsSet.insert(&m_boneVertices[i_v]);
}

int BoneMesh::checkData()
{
	return 1;
	float dataScale = (boundingBox[1] - boundingBox[0]).length();
	float eps = 1e-4f * dataScale;

	// check local-global vertex positions
	for (int i_v = 0; i_v < m_boneVertices.size(); i_v++)
	{
		const BoneVertex& v = m_boneVertices[i_v];
		ldp::Float3 gvpos = v.getGlobalPos(0);
		ldp::Float3 gvpos_init = vertex_list[v.id];
		float dist = (gvpos - gvpos_init).length();
		if (dist > eps)
		{
			ldp::Logger::error("Input skeleton data not proper: local vertex check failed\
							   :\n%f %f %f\n%f %f %f\n", gvpos_init[0], gvpos_init[1], gvpos_init[2],
							   gvpos[0], gvpos[1], gvpos[2]);
		}
	}

	return 1;
}

int BoneMesh::readInOneNode(std::ifstream& stream, Node& node)
{
	std::string lineBuffer;
	std::string label;
	int ret;

	// node name
	std::getline(stream, lineBuffer);
	label = getLabel(lineBuffer);
	node.name = lineBuffer;

	// node id
	std::getline(stream, lineBuffer);
	label = getLabel(lineBuffer);
	ret = sscanf(lineBuffer.c_str(), "%d", &node.id);
	if (ret != 1)
	{
		ldp::Logger::error("read in node id failed!");
		return 0;
	}

	// child id
	std::getline(stream, lineBuffer);
	label = getLabel(lineBuffer);
	int child_num = 0;
	ret = sscanf(lineBuffer.c_str(), "%d", &child_num);
	if (ret != 1 || label != "child_num")
	{
		ldp::Logger::error("read in child num failed!");
		return 0;
	}
	node.childrenIds.resize(child_num, -1);
	for (int i = 0; i < child_num; i++)
	{
		std::getline(stream, lineBuffer);
		label = getLabel(lineBuffer);
		int child_id = 0;
		ret = sscanf(lineBuffer.c_str(), "%d", &child_id);
		if (ret != 1 || label != "child_id")
		{
			ldp::Logger::error("read in child id failed!");
			return 0;
		}
		node.childrenIds[i] = child_id;
	}

	// init global TM
	std::getline(stream, lineBuffer);
	assert(getLabel(lineBuffer) == "Init-TM");
	for (int k = 0; k < 4; k++)
	{
		std::getline(stream, lineBuffer);
		ret = sscanf(lineBuffer.c_str(), "%f %f %f %f", &node.initGlobalTM(k, 0),
			&node.initGlobalTM(k, 1), &node.initGlobalTM(k, 2), &node.initGlobalTM(k, 3));
		if (ret != 4)
		{
			ldp::Logger::error("read in node globalTM failed!");
			return 0;
		}
	}

	// local TM
	std::getline(stream, lineBuffer);
	assert(getLabel(lineBuffer) == "Local-TM");
	ldp::Mat4f localTM;
	for (int k = 0; k < 4; k++)
	{
		std::getline(stream, lineBuffer);
		ret = sscanf(lineBuffer.c_str(), "%f %f %f %f", &localTM(k,0),
			&localTM(k, 1), &localTM(k, 2), &localTM(k, 3));
		if (ret != 4)
		{
			ldp::Logger::error("read in node localTM failed!");
			return 0;
		}
	}

	// global TM
	std::getline(stream, lineBuffer);
	assert(getLabel(lineBuffer) == "Global-TM");
	if (m_numFrames <= 0)
	{
		node.globalTMs.resize(1);
		for (int k = 0; k < 4; k++)
		{
			std::getline(stream, lineBuffer);
			ret = sscanf(lineBuffer.c_str(), "%f %f %f %f", &node.globalTMs[0](k, 0),
				&node.globalTMs[0](k, 1), &node.globalTMs[0](k, 2), &node.globalTMs[0](k, 3));
			if (ret != 4)
			{
				ldp::Logger::error("read in node globalTM failed!");
				return 0;
			}
		}
	}
	else
	{
		node.globalTMs.resize(m_numFrames);
		for (int i_frame = 0; i_frame < m_numFrames; i_frame++)
		{
			for (int k = 0; k < 4; k++)
			{
				std::getline(stream, lineBuffer);
				ret = sscanf(lineBuffer.c_str(), "%f %f %f %f", &node.globalTMs[i_frame](k, 0),
					&node.globalTMs[i_frame](k, 1), &node.globalTMs[i_frame](k, 2), &node.globalTMs[i_frame](k, 3));
				if (ret != 4)
				{
					ldp::Logger::error("read in node globalTM failed!");
					return 0;
				}
			}
			if (i_frame != m_numFrames - 1)
				std::getline(stream, lineBuffer);
		}
	}

	// a hack here for the .max file data
	node.root_virtual_parent_GlobalTM = node.getGlobalTM(0) * localTM.inv();

	return 1;
}

std::string BoneMesh::getLabel(std::string& lineBuffer)const
{
	size_t pos = lineBuffer.find_first_of(':');
	if (pos == lineBuffer.size())
		return "";
	std::string lb = lineBuffer.substr(0, pos);
	lineBuffer = lineBuffer.substr(pos+1);
	pos = lineBuffer.find_first_not_of(' ');
	if (pos < lineBuffer.size())
		lineBuffer = lineBuffer.substr(pos);
	return lb;
}