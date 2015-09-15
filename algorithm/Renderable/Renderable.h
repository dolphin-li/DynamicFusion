#ifndef __RENDERABLE_H__
#define __RENDERABLE_H__

#include "assert.h"
#include "ldp_basic_mat.h"
#include <string>
using ldp::Float3;
class Renderable
{
public:
	Renderable()
	{
		_isEnabled = true;
		_isSelected = false;
		_name = "Renderable";
	}
	virtual ~Renderable()
	{

	}
public:
	const static int TYPE_GENERAL = 0x10000000;
	const static int TYPE_ABSTRACT = 0x10000001;
	const static int TYPE_TRIMESH = 0x10000002;
	const static int TYPE_QUADMESH = 0x10000003;
	const static int TYPE_POINTCLOUD = 0x10000004;
	const static int TYPE_OBJMESH = 0x10000005;
	const static int TYPE_NODE = 0x10000006;
	const static int TYPE_DEPTHIMAGE = 0x10000007;
	const static int TYPE_COLORIMAGE = 0x10000008;
	const static int TYPE_BMESH = 0x10000009;
	const static int TYPE_BONE_MESH = 0x10000010;

	//show type
	const static int SW_V = 0x00000001;
	const static int SW_E = 0x00000002;
	const static int SW_F = 0x00000004;
	const static int SW_N = 0x00000008;
	const static int SW_FLAT = 0x00000010;
	const static int SW_SMOOTH = 0x00000020;
	const static int SW_TEXTURE = 0x00000040;
	const static int SW_LIGHTING = 0x00000080;
	const static int SW_COLOR = 0x00000100;
public:
	virtual void render(int showType, int frameIndex=0) =0;

	virtual void renderConstColor(Float3 color)const = 0;

	virtual int getMeshType()const{return TYPE_GENERAL;}

	virtual void clear(){assert(0 && "your child class should overload clear()");}

	bool isEnabled()const{
		return _isEnabled;
	}

	void setEnabled(bool enable){
		_isEnabled = enable;
	}

	bool isSelected()const{
		return _isSelected;
	}

	void setSelected(bool enable){
		_isSelected = enable;
	}

	void setName(const char* name)
	{
		_name = name;
	}

	const char* getName()const
	{
		return _name.c_str();
	}
protected:
	bool _isEnabled;
	bool _isSelected;
	std::string _name;
};





#endif