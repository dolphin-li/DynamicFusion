#include "AbstractMesh.h"
#include "Error.h"

const Float3 AbstractMesh::SELECT_COLOR = Float3(175.f/255.f,175.f/255.f,48.f/255.f);
const Float3 AbstractMesh::NON_SEL_COLOR = Float3(0.0f,0.0f,0.0f);
AbstractMesh::AbstractMesh(void):Renderable()
{
}

AbstractMesh::~AbstractMesh(void)
{
}

void AbstractMesh::getNormals(vector<Float3>& store)const
{
	if(normals.size()==0)
	{
		Logger::warning("Empty Normals, In TriMesh::getNormals()\n");
		return;
	}
	if(store.size() != this->normals.size())
		store.resize(normals.size());
	memcpy(&store[0], &normals[0], normals.size()*sizeof(normals[0]));
}
void AbstractMesh::getVertices(vector<Float3>& store)const
{
	if(vertices.size()==0)
	{
		Logger::warning("Empty Mesh, In TriMesh::getVertices()\n");
		return;
	}
	if(store.size() != this->vertices.size())
		store.resize(vertices.size());
	memcpy(&store[0], &vertices[0], vertices.size()*sizeof(vertices[0]));
}

void AbstractMesh::setVertices(const vector<Float3>& input, bool isNormalsGen)
{
	if(input.size() != this->vertices.size())
	{
		vertices.resize(input.size());
		isVertSelected.resize(input.size());
		colors.resize(input.size());
	}
	memcpy(&vertices[0], &input[0], vertices.size()*sizeof(vertices[0]));
	memset(&isVertSelected[0], 0, isVertSelected.size()*sizeof(isVertSelected[0]));
	for(int i=0; i<(int)colors.size(); i++)
		colors[i] = AbstractMesh::NON_SEL_COLOR;

	if(isNormalsGen)
	{
		generateNormals();
	}
}

void AbstractMesh::setVertexSelected(int i, bool select)
{
	isVertSelected[i]=(select ? 1:0);
	if(isVertSelected[i]) colors[i] = SELECT_COLOR;
	else colors[i] = NON_SEL_COLOR;
}