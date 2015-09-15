#ifndef __ABSTRACTMESH_H__
#define __ABSTRACTMESH_H__

#include <vector>
#include <string>
#include "Renderable.h"
using namespace ldp;
using namespace std;

class TriMesh;
class AbstractMesh: public Renderable
{
friend class ObjLoader;
friend class OffLoader;
public:
	const static Float3 SELECT_COLOR;
	const static Float3 NON_SEL_COLOR;
protected:
	string name;
	Float3 boundingBox[2];//boudingBox[0]=min, boundingBox[1]=max
	vector<Float3>vertices;
	vector<Float3>colors;
	vector<Float3>normals;//vertex normals
	vector<int> isVertSelected;
public:
	AbstractMesh(void);
	virtual ~AbstractMesh(void);

	virtual void clear(){
		vertices.clear();
		normals.clear();
		colors.clear();
		isVertSelected.clear();
	}

	virtual void toTriMesh(TriMesh* store)const{}//convert a general mesh to a triangle mesh

	virtual void renderConstColor(Float3 color)const{}
	virtual void render(int showType, int frameIndex=0){}
	virtual void generateNormals(){}
	virtual void getFaces(vector<int>& store)const{}//3*int for trimesh and 4*int for quadmesh
	virtual void getEdges(vector<Int2>& store)const{}
	virtual void getFaceNormals(vector<Float3>& store)const{}//get face normals
	virtual void generateEdges(){}
	virtual void setFaces(vector<Int3>& input){}
	virtual int getMeshType()const{return TYPE_ABSTRACT;}

	virtual Float3* getVertexPointer(){return &vertices[0];}
	virtual Float3* getNormalPointer(){return &normals[0];}
	virtual Int2* getEdgePointer(){return 0;}
	virtual int* getFacePointer(){return 0;}
	virtual Float3* getFaceNormalPointer(){return 0;}

	bool isVertexSelected(int i)const{return isVertSelected[i]==1;}
	void setVertexSelected(int i, bool select);
	void getVertices(vector<Float3>& store)const;
	void setVertices(const vector<Float3>& input, bool isNormalsGen=true);
	void getNormals(vector<Float3>& store)const;//get vertex normals

	//getters for single primitives
	void getVertex(int idx, Float3& store)const{
		store = vertices[idx];
	}
	void getNormal(int idx, Float3& store)const{
		store = normals[idx];
	}
	void getBoundingBox(Float3 bBox[2])const{
		bBox[0] = boundingBox[0];
		bBox[1] = boundingBox[1];
	}
	int getNumVerts()const{
		return vertices.size();
	}
	virtual int getNumFaces()const{return -1;}//this should be down by children
	virtual int getNumEdges()const{return -1;}
	virtual void getFace(int idx, Int4 &store)const{}
	virtual void getFace(int idx, Int3 &store)const{}
	virtual void getEdge(int idx, Int2 &store)const{}
	
	//setters for single primitives
	void setVertex(int idx,const Float3& input){
		vertices[idx] = input;
	}
	void setNormal(int idx,const Float3& input){
		normals[idx] = input;
	}
	void setBoundingBox(const Float3 bBox[2]){
		boundingBox[0] = bBox[0];
		boundingBox[1] = bBox[1];
	}


	virtual void setFace(int idx,const Int4 &input){}
	virtual void setFace(int idx,const Int3 &input){}
	virtual void setEdge(int idx,const Int2 &input){}
	
};


#endif //_ABSTRACTMESH_H__
