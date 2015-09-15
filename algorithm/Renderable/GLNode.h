#ifndef __NODE_H121__
#define __NODE_H121__

#include <list>
#include <string>
#include <vector>
#include "Renderable.h"
using namespace std;
using namespace ldp;
class AbstractMesh;
class GLNode :public Renderable
{
public:
	GLNode(void);
	~GLNode(void);

	/**
	* showType = MESH::SW_V,...
	* */
	void render(int showType, int frameIndex=0);
	void renderConstColor(ldp::Float3 color)const;
	void renderChildrenAsIndex(int indexBegin)const;
	virtual int getMeshType()const{return TYPE_NODE;}

	void attach(GLNode& node);
	void attach(Renderable* mesh);

	void detachLastNode();
	void detachLastMesh();

	void detachAll();
	void detachMesh(int i);

	int getNumChildMesh()const;
	int getNumChildNode()const;
	Renderable* getMesh(int i);
	Renderable* getLastMesh();

	GLNode* getNode(int i);
	GLNode* getLastNode();


	int getNumAllMeshes()const;//meshes of mine and my child-nodes

	//recursivly make index from meshes of mine and my child-nodes.
	void getMeshWithGlobalIndex(int index, Renderable** me, GLNode** fatherNode);

	void setTransMat(const ldp::Mat4f& tMat);
	ldp::Mat4f getTransMat()const;

	GLNode *getFatherNode();
private:
	void recurRenderIndex(int &ib)const;
	void recurGetMeshWithIdx(int &idx, Renderable** out, GLNode** fatherNode);
protected:
	list<GLNode> _childNodes;
	vector<Renderable*> _meshes;

	ldp::Mat4f _myTrans;
	GLNode *_fatherNode;
};

#endif //__NODE_H__
