#pragma once

#include <set>
#include "ObjMesh.h"
#include "ldpMat\Quaternion.h"

class BoneMesh : public ObjMesh
{
public:
	struct BoneVertex;
	class Node
	{
		friend class BoneMesh;
	protected:
		int id;
		Node* parent;
		std::vector<Node*> children;
		std::vector<int> childrenIds;
		std::string name;
		std::vector<ldp::Mat4f> globalTMs;
		ldp::Mat4f root_virtual_parent_GlobalTM;
		ldp::Mat4f initGlobalTM;

		// new part: using dual-quaternion for interpolation
		bool useDualQuat;
		std::vector<ldp::DualQuaternionF> dualQuat_G_mult_Inv_InitGs;//one-to-one to a rotation + a translation
	public:
		Node()
		{
			id = -1;
			parent = 0;
			root_virtual_parent_GlobalTM.eye();
			initGlobalTM.eye();
			useDualQuat = false;
		}

		int getNumFrames()const{ return globalTMs.size(); }
		void clearAllFrames()
		{
			globalTMs.clear();
			dualQuat_G_mult_Inv_InitGs.clear();
		}
		void addFrames(int num=1)
		{
			for (int i = 0; i < num; i++)
			{
				globalTMs.push_back(ldp::Mat4f().eye());
				if (useDualQuat)
					dualQuat_G_mult_Inv_InitGs.push_back(ldp::DualQuaternionF().setIdentity());
			}
		}

		const char* getName()const{ return name.c_str(); }

		int getNumChild()const{return int(children.size());}

		Node* getChild(int i){ return children[i]; }
		void setChild(int i, Node* n){ children[i] = n; }

		const Node* getChild(int i)const{return children[i];}

		const Node* getParent()const{return parent;}
		Node* getParent(){return parent;}

		int getId()const{return id;}

		ldp::Mat4f getLocalTM(int frameId)const
		{
			return getParentGlobalTM(frameId).inv() * getGlobalTM(frameId);
		}

		ldp::Mat4f getGlobalTM(int frameId)const
		{
			return globalTMs[frameId];
		}

		ldp::Mat4f getParentGlobalTM(int frameId)const
		{
			ldp::Mat4f G;
			if (parent)
				G = parent->getGlobalTM(frameId);
			else
				G = root_virtual_parent_GlobalTM;
			return G;
		}

		ldp::Mat4f getInitGlobalTM()const{ return initGlobalTM; }

		void setLocalTM(const ldp::Mat4f& L, int frameId)
		{
			setGlobalTM(getParentGlobalTM(frameId)*L, frameId);
		}

		void setGlobalTM(const ldp::Mat4f& G, int frameId)
		{
			// the order is important
			// we must firstly update children and then update self
			for (int i = 0; i < children.size(); i++)
				children[i]->setGlobalTM(G*children[i]->getLocalTM(frameId), frameId);

			setGlobalTM_no_update(G, frameId);
		}

		void setGlobalTM_no_update(const ldp::Mat4f& G, int frameId)
		{
			if (frameId < 0 || frameId >= globalTMs.size())
				return;
			globalTMs[frameId] = G;

			if (useDualQuat)
				dualQuat_G_mult_Inv_InitGs[frameId].setFromTransform(globalTMs[frameId]*initGlobalTM.inv());
		}

		void setInitGlobalTM_no_update(const ldp::Mat4f& G)
		{
			initGlobalTM = G;
		}

		void setParent(Node* parent)
		{
			this->parent = parent;
		}

		void setChildren(const std::vector<Node*>& children)
		{
			this->children.assign(children.begin(), children.end());
			this->childrenIds.resize(this->children.size(), -1);
			for (int i = 0; i < this->children.size(); i++)
				this->childrenIds[i] = this->children[i]->id;
		}
		void addChild(Node* child)
		{
			if (child)
			{
				this->children.push_back(child);
				this->childrenIds.push_back(child->id);
			}
		}

		void setID(int id)
		{
			this->id = id;
		}

		bool operator < (const Node& r)const
		{
			return id < r.id;
		}

		bool isUseDualQuat()const{ return useDualQuat; }
		void setUseDualQuat(bool use)
		{ 
			useDualQuat = use; 
			if (useDualQuat)
			{
				dualQuat_G_mult_Inv_InitGs.resize(globalTMs.size());
				for (int i = 0; i < globalTMs.size(); i++)
					dualQuat_G_mult_Inv_InitGs[i].setFromTransform(getGlobalTM(i)*initGlobalTM.inv());
			}
		}
		const ldp::DualQuaternionF& getDualQuat(int frameId)const
		{ 
			return dualQuat_G_mult_Inv_InitGs[frameId]; 
		}
		void setDualQuat(const ldp::DualQuaternionF& dq, int frameId)
		{
			dualQuat_G_mult_Inv_InitGs[frameId] = dq;
			ldp::Mat4f M;
			dq.getTransform(M);
			setGlobalTM(M*initGlobalTM, frameId);
		}		
		void setDualQuat_notUpdateTM(const ldp::DualQuaternionF& dq, int frameId)
		{
			dualQuat_G_mult_Inv_InitGs[frameId] = dq;
		}
	};
	
	struct NodePtrCmp
	{
		bool operator()(const Node* l, const Node* r)const
		{
			return l->id < r->id;
		}
	};
	
	class BoneVertex
	{
		friend class BoneMesh;
	public:
		int id;
		std::vector<Node*> nodes;
		std::vector<int> nodeIds;
		std::vector<float> weights;
		ldp::Float3 initGlobalPos;

		ldp::Float3 getInitGlobalPos()const{ return initGlobalPos; }

		ldp::Mat4f getCombinedTM(int frameId)const;

		ldp::Float3 getGlobalPos(int frameId)const;

		bool operator < (const BoneVertex& r)const
		{
			return id < r.id;
		}
		bool operator == (const BoneVertex& r)const
		{
			return id == r.id;
		}
	};

	struct VertexPtrCmp
	{
		bool operator()(const BoneVertex* l, const BoneVertex* r)const
		{
			return l->id < r->id;
		}
	};

	typedef std::set<Node*, NodePtrCmp> NodeSet;
	typedef std::set<const Node*, NodePtrCmp> NodeSetConst;
	typedef std::set<BoneVertex*, VertexPtrCmp> BoneVertexSet;
	typedef std::set<const BoneVertex*, VertexPtrCmp> BoneVertexSetConst;

public:
	BoneMesh();
	~BoneMesh();

	BoneMesh(const BoneMesh& rhs);

	int loadObj(const char* objname, bool isNormalGen, bool isNormalize);
	virtual void saveObj(const char* objname)const;
	bool hasBones()const{ return m_hasBones; }
	virtual int getMeshType()const{ return TYPE_BONE_MESH; }
	virtual void getSubMesh(const std::vector<int>& validVertexIdx,
		ObjMesh* subMesh, std::vector<int>* faceIdToValidFaceId = 0)const;

	void updateVertexPosByBones(int frameId);

	Node* getNodeById(int id)
	{
		Node tmp;
		tmp.id = id;
		std::set<Node*, NodePtrCmp>::iterator it = m_nodesSet.find(&tmp);
		if (it != m_nodesSet.end())
			return *it;
		return NULL;
	}
	const Node* getNodeById(int id)const
	{
		Node tmp;
		tmp.id = id;
		std::set<Node*, NodePtrCmp>::iterator it = m_nodesSet.find(&tmp);
		if (it != m_nodesSet.end())
			return *it;
		return NULL;
	}
	BoneVertex* getBoneVertexById(int id)
	{
		BoneVertex tmp;
		tmp.id = id;
		std::set<BoneVertex*, VertexPtrCmp>::iterator it = m_boneVertsSet.find(&tmp);
		if (it != m_boneVertsSet.end())
			return *it;
		return NULL;
	}
	const BoneVertex* getBoneVertexById(int id)const
	{
		BoneVertex tmp;
		tmp.id = id;
		std::set<BoneVertex*, VertexPtrCmp>::iterator it = m_boneVertsSet.find(&tmp);
		if (it != m_boneVertsSet.end())
			return *it;
		return NULL;
	}

	Node* getNode(int arrayPos){ return &m_nodes[arrayPos]; }
	const Node* getNode(int arrayPos)const{ return &m_nodes[arrayPos]; }
	int getNumNodes()const{ return m_nodes.size(); }

	BoneVertex* getBoneVertex(int arrayPos){ return &m_boneVertices[arrayPos]; }
	const BoneVertex* getBoneVertex(int arrayPos)const{ return &m_boneVertices[arrayPos]; }
	int getNumBoneVertex()const{ return m_boneVertices.size(); }

	void clear();

	virtual void cloneFrom(const ObjMesh* rhs);
	int checkData();

	void setBones(const std::vector<Node>& nodes, const std::vector<BoneVertex>& boneVerts);

	void setUseDualQuaternion(bool use);

	void clearAllFrames();
	void addFrames(int num);
	int getFrameId()const{ return m_frameId; }
	void setFrameId(int id);
protected:
	int loadSkeleton(const char* skelname);
	std::string getLabel(std::string& lineBuffer)const;
	int readInOneNode(std::ifstream& stream, Node& node);
	void updateNodeSet();
	void updateBoneVertexSet();

private:
	bool m_hasBones;
	NodeSet m_nodesSet;
	BoneVertexSet m_boneVertsSet;
	int m_frameId;
	int m_numFrames;

	std::vector<Node*> m_pRootNodes;
	std::vector<Node> m_nodes;
	std::vector<BoneVertex> m_boneVertices;
};

