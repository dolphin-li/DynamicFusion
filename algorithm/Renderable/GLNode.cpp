#include "GLNode.h"
#include "Error.h"
#include <glut.h>
#include <stack>

using namespace ldp;

GLNode::GLNode(void):Renderable()
{
	_myTrans.eye();
	_fatherNode = 0;
}

GLNode::~GLNode(void)
{

}

GLNode* GLNode::getFatherNode()
{
	return _fatherNode;
}

void GLNode::setTransMat(const Mat4f& tMat)
{
	_myTrans = tMat;
}

Mat4f GLNode::getTransMat()const
{
	return _myTrans;
}

void GLNode::renderChildrenAsIndex(int indexBegin)const
{
	recurRenderIndex(indexBegin);
}

void GLNode::recurRenderIndex(int &indexBegin)const
{
	glPushMatrix();
	glMultMatrixf(_myTrans.ptr());
	vector<Renderable*>::const_iterator it2=_meshes.begin();
	for(; it2!=_meshes.end(); it2++)
	{
		(*it2)->renderConstColor(Float3(indexBegin));
		indexBegin++;
	}
	glPopMatrix();

	list<GLNode>::const_iterator it1=_childNodes.begin();
	for(; it1!=_childNodes.end(); it1++)
	{
		glPushMatrix();
		glMultMatrixf(_myTrans.ptr());
		it1->recurRenderIndex(indexBegin);
		glPopMatrix();
	}
}

void GLNode::renderConstColor(Float3 color)const
{
	glPushMatrix();
	glMultMatrixf(_myTrans.ptr());
	vector<Renderable*>::const_iterator it2=_meshes.begin();
	for(; it2!=_meshes.end(); it2++)
		(*it2)->renderConstColor(color);
	glPopMatrix();

	list<GLNode>::const_iterator it1=_childNodes.begin();
	for(; it1!=_childNodes.end(); it1++)
	{
		glPushMatrix();
		glMultMatrixf(_myTrans.ptr());
		it1->renderConstColor(color);
		glPopMatrix();
	}

}

void GLNode::render(int showType, int frameIndex)
{
	if (!isEnabled())
		return;
	glPushMatrix();
	glMultMatrixf(_myTrans.ptr());
	vector<Renderable*>::iterator it2=_meshes.begin();
	for(; it2!=_meshes.end(); it2++)
		(*it2)->render(showType,frameIndex);
	glPopMatrix();

	list<GLNode>::iterator it1=_childNodes.begin();
	for(; it1!=_childNodes.end(); it1++)
	{
		glPushMatrix();
		glMultMatrixf(_myTrans.ptr());
		it1->render(showType,frameIndex);
		glPopMatrix();
	}
}

void GLNode::attach(GLNode &node)
{
	node._fatherNode = this;
	_childNodes.push_back(node);
}

void GLNode::attach(Renderable *mesh)
{
	_meshes.push_back(mesh);
}

void GLNode::detachLastMesh()
{
	_meshes.pop_back();
}

void GLNode::detachLastNode()
{
	_childNodes.rbegin()->_fatherNode = 0;
	_childNodes.pop_back();
}

void GLNode::detachAll()
{
	_meshes.clear();

	list<GLNode>::iterator it1=_childNodes.begin();
		for(; it1!=_childNodes.end(); it1++)
			it1->_fatherNode = 0;

	_childNodes.clear();
}

Renderable* GLNode::getMesh(int i)
{
	if(i>=0 && i<(int)_meshes.size())
	{
		return _meshes.at(i);
	}
	ldp::Logger::warning("%d out of meshes index when get\n",i);
	return 0;
}
GLNode* GLNode::getNode(int i)
{
	int idx = 0;
	list<GLNode>::iterator it1=_childNodes.begin();
	for(; it1!=_childNodes.end(); it1++){
		if(idx == i)
			return &(*it1);
		idx++;
	}
	ldp::Logger::warning("%d out of meshes index when get\n",i);
	return 0;
}

void GLNode::detachMesh(int i)
{
	if(i>=0 && i<(int)_meshes.size())
	{
		_meshes.erase(_meshes.begin()+i);
		return;
	}
	ldp::Logger::warning("%d out of meshes index when detach\n",i);
}

Renderable* GLNode::getLastMesh()
{
	if(_meshes.size()>0)
	{
		vector<Renderable*>::iterator it2=_meshes.end();
		it2--;
		if(*it2) return *it2;
	}
	ldp::Logger::warning("Empty Node!");
	return 0;
}

GLNode* GLNode::getLastNode()
{
	if(_childNodes.size()>0)
	{
		return &(*_childNodes.rbegin());
	}
	ldp::Logger::warning("Empty Node!");
	return 0;
}

int GLNode::getNumChildMesh()const
{
	return _meshes.size();
}

int GLNode::getNumChildNode()const
{
	return _childNodes.size();
}

int GLNode::getNumAllMeshes()const
{
	int sum = 0;
	sum += getNumChildMesh();
	list<GLNode>::const_iterator it1=_childNodes.begin();
	for(; it1!=_childNodes.end(); it1++)
	{
		sum += it1->getNumAllMeshes();
	}
	return sum;
}

void GLNode::getMeshWithGlobalIndex(int idx, Renderable** me, GLNode** father)
{
	*me = 0;
	*father = 0;
	recurGetMeshWithIdx(idx, me, father);
}

void GLNode::recurGetMeshWithIdx(int& idx, Renderable** out, GLNode** fatherNode)
{
	if(idx < getNumChildMesh()){
		*out =  getMesh(idx);
		*fatherNode = this;
	}
	if(*out) return;
	idx -= getNumChildMesh();
	list<GLNode>::iterator it1=_childNodes.begin();
	for(; it1!=_childNodes.end(); it1++)
	{
		it1->recurGetMeshWithIdx(idx, out, fatherNode);
		if(*out) return;
	}	
}
