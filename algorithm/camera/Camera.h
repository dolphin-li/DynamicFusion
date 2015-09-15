#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "../ldpMat/ldp_basic_mat.h"
using namespace ldp;
class Camera
{
public:
	Camera(void);
	Camera(const Camera& r);
	~Camera(void);

	/**
	* Apply the transform matrix to current glContex
	* */
	void apply()const;

	void reset();
	/**
	* Directly set transform matrix to the camera.
	* The corresponding frustum infos are not changed.
	* */
	void setModelViewMatrix(const Mat4f& rhs);
	Mat4f getModelViewMatrix()const;

	void setProjectionMatrix(const Mat4f& rhs);
	Mat4f getProjectionMatrix()const;

	/**
	* Convinient method, change direction of camera to (target-location)
	* !Note: this method doesn't change frustum infomations, e.g. frustumLeft
	*  but change the modelView matrix
	* */
	void lookAt(const Float3& location,const Float3& target, const Float3& up);

	Float3 getWorldCoords(const Float3& screenCoords)const;
	Float3 getScreenCoords(const Float3& worldCoords)const;

	/**
	* Some getters and setters
	* */
	void setViewPort(float vLeft, float vRight, float vTop, float vBottom);
	float getViewPortLeft()const;
	float getViewPortRight()const;
	float getViewPortTop()const;
	float getViewPortBottom()const;

	void setFrustum(float fLeft, float fRight, float fTop, float fBottom,float fNear, float fFar );
	float getFrustumLeft()const;
	float getFrustumRight()const;
	float getFrustumTop()const;
	float getFrustumBottom()const;
	float getFrustumNear()const;
	float getFrustumFar()const;

	void setLocation(const Float3& location);
	Float3 getLocation()const;
	void setDirection(const Float3& direction);
	Float3 getDirection()const;
	void setUp(const Float3& up);
	Float3 getUp()const;
	void setScalar(const Float3& scalar);
	Float3 getScalar()const;

	void scale(const Float3& scalar);
	void translate(const Float3& transvec);

	//isOrtho=false is set in this method
	void setPerspective(float fov, float aspect, float fNear, float fFar);
	float getFov()const;
	float getAspect()const;

	/**
	* Arcball related
	* */
	void arcballClick(Float2 pos);//call me when button clicked
	void arcballDrag(Float2 pos);//call me when mouse dragged.

	/**
	* Enable/Disable ortho
	* */
	void enableOrtho(bool enable);
	bool isOrthoEnabled()const;

	/**
	* Interperlation
	* */
	const Camera& interpolateModelViewMatrixWith(const Camera& rhs, float s);

	/**
	* Save/Load
	* */
	bool save(const char* fileName)const;
	bool load(const char* fileName);
	bool save(FILE* pFile)const;
	bool load(FILE* pFile);

protected:
	bool isOrtho;
	Mat4f projection;
	Mat4f modelView;
	Mat4f modelViewProjection;
	Mat4f invModelViewProjection;

	float viewPortLeft;
	float viewPortRight;
	float viewPortTop;
	float viewPortBottom;

	//arcball related
	Float3 stVec;//saved clicked vector
	Mat3f lastRot;
protected:
	/**
	* Update the transform matrix
	* */
	Float3 arcballSphereMap(Float2 pos) const;
};

#endif //__CAMERA_H__
