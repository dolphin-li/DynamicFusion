#include "Camera.h"
#include <gl/gl.h>
#include <gl/glu.h>
#include "fmath.h"
#include "../ldpMat/Quaternion.h"
Camera::Camera(void)
{
	isOrtho = false;

	viewPortLeft = 0.0f;
	viewPortRight = 1.0f;
	viewPortTop = 1.0f;
	viewPortBottom = 0.0f;
	modelView.eye();
	modelViewProjection.eye();
	projection.eye();
	invModelViewProjection.eye();
}

Camera::Camera(const Camera& r)
{
	isOrtho = r.isOrtho;
	viewPortLeft = r.viewPortLeft;
	viewPortRight = r.viewPortRight;
	viewPortTop = r.viewPortTop;
	viewPortBottom = r.viewPortBottom;
	modelView = r.modelView;
	modelViewProjection = r.modelViewProjection;
	invModelViewProjection = r.invModelViewProjection;
	projection = r.projection;
	stVec = r.stVec;
	lastRot = r.lastRot;
}

Camera::~Camera(void)
{
}

void Camera::reset()
{
	viewPortLeft = 0.0f;
	viewPortRight = 1.0f;
	viewPortTop = 1.0f;
	viewPortBottom = 0.0f;
	modelView.eye();
	modelViewProjection.eye();
	projection.eye();
	invModelViewProjection.eye();
}

void Camera::apply()const
{
	glViewport((int)viewPortLeft,(int)viewPortTop,(int)(viewPortRight-viewPortLeft),(int)(viewPortBottom-viewPortTop));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMultMatrixf(projection.ptr());
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMultMatrixf(modelView.ptr());
}

void Camera::setModelViewMatrix(const Mat4f &rhs)
{
	modelView = rhs;

	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

Mat4f Camera::getModelViewMatrix()const
{
	return modelView;
}

void Camera::setProjectionMatrix(const Mat4f& rhs)
{
	projection = rhs;

	//others
	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

Mat4f Camera::getProjectionMatrix()const
{
	return projection;
}

void Camera::lookAt(const Float3& location, const Float3& target, const Float3 &up)
{
	Float3 sc = getScalar();
	Float3 f = target - location;
	f.normalizeLocal();
	Float3 s = f.cross(up).normalizeLocal();
	Float3 u = s.cross(f);

	modelView.eye();
	modelView(0, 0) = s[0];		modelView(0, 1) = s[1];		modelView(0, 2) = s[2];
	modelView(1, 0) = u[0];		modelView(1, 1) = u[1];		modelView(1, 2) = u[2];
	modelView(2, 0) = -f[0];	modelView(2, 1) = -f[1];	modelView(2, 2) = -f[2];

	//translate -location
	setLocation(location);

	setScalar(sc);

	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

void Camera::scale(const Float3& scalar)
{
	setScalar(getScalar()*scalar);
}

void Camera::translate(const Float3& transvec)
{
	Float3 v = modelView.getRotationPart().inv() * transvec;
	setLocation(getLocation() + v);
}


/**
* Some getters and setters
* */
void Camera::setViewPort(float vLeft, float vRight, float vTop, float vBottom)
{
	this->viewPortLeft = vLeft;
	this->viewPortRight = vRight;
	this->viewPortTop = vTop;
	this->viewPortBottom = vBottom;
}


float Camera::getViewPortLeft()const
{
	return viewPortLeft;
}

float Camera::getViewPortRight()const
{
	return viewPortRight;
}

float Camera::getViewPortTop()const
{
	return viewPortTop;
}

float Camera::getViewPortBottom()const
{
	return viewPortBottom;
}

void Camera::setFrustum(float fLeft, float fRight, float fTop, float fBottom,float fNear, float fFar )
{
	if(isOrtho)
	{
		projection.zeros();
		Mat4f& M=projection;
		M(0,0) = 2/(fRight - fLeft);			M(0,3) = (fRight+fLeft)/(fLeft-fRight);
		M(1,1) = 2/(fTop - fBottom);			M(1,3) = (fTop+fBottom)/(fBottom-fTop);
		M(2,2) = 2/(fNear-fFar);				M(2,3) = (fFar+fNear)/(fNear-fFar);
		M(3,3) = 1.f;
	}
	else
	{
		projection.zeros();
		Mat4f& M=projection;
		float left=fLeft,right=fRight,bottom=fBottom,top=fTop,zNear=fNear,zFar=fFar;
		if(fmath::cmpf(left, right) || fmath::cmpf(top, bottom) || fmath::cmpf(zNear,zFar)
			|| zNear <= 0 || zFar<=0)
		{
			printf("Error Input in frustum, invalid values!\n");
			return;
		}
		const float r_width = 1.0f/(right - left);
		const float r_height = 1.0f/(top - bottom);
		const float r_depth = 1.0f/(zNear - zFar);
		const float x = zNear * r_width * 2;
		const float y = zNear * r_height * 2;
		const float A = (right + left) * r_width;
		const float B = (top + bottom) * r_height;
		const float C = (zFar + zNear) * r_depth;
		const float D = (zFar * zNear * r_depth)*2;
		M(0,0) = x;		M(0,2) = A;		M(1,1) = y;
		M(1,2) = B;		M(2,2) = C;		M(2,3) = D;
		M(3,2) = -1.0f;	M(3,3) = 0;
	}
	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

void Camera::setPerspective(float fov, float aspect, float fNear, float fFar)
{
	const float top = fNear * tanf(fov * fmath::DEG_TO_RAD * 0.5f);
	const float bottom = -top;
	const float left = bottom * aspect;
	const float right = -left;
	setFrustum(left,right,top,bottom,fNear,fFar);
}

float Camera::getFov()const
{
	return atanf(getFrustumTop()/getFrustumNear()) / (fmath::DEG_TO_RAD*0.5f);
}

float Camera::getAspect()const
{
	return getFrustumLeft() / getFrustumBottom();
}

float Camera::getFrustumLeft()const
{
	const Mat4f& M = projection;
	if(isOrthoEnabled())
		return (-1	-	M(0,3))/M(0,0);
	else
		return M(2,3)/(M(2,2)-1) * (M(0,2)-1) / M(0,0);
}

float Camera::getFrustumRight()const
{
	const Mat4f& M = projection;
	if(isOrthoEnabled())
		return (1	-	M(0,3))/M(0,0);
	else
		return M(2,3)/(M(2,2)-1) * (M(0,2)+1) / M(0,0);
}

float Camera::getFrustumTop()const
{
	const Mat4f& M = projection;
	if(isOrthoEnabled())
		return (1	-	M(1,3))/M(1,1);
	else
		return M(2,3)/(M(2,2)-1) * (M(1,2)+1) / M(1,1);
}

float Camera::getFrustumBottom()const
{
	const Mat4f& M = projection;
	if(isOrthoEnabled())
		return (-1	-	M(1,3))/M(1,1);
	else
		return M(2,3)/(M(2,2)-1) * (M(1,2)-1) / M(1,1);
}

float Camera::getFrustumNear()const
{
	const Mat4f& M = projection;
	if(isOrthoEnabled())
		return (1	+	M(2,3))/M(2,2);
	else
		return M(2,3)/(M(2,2)-1);
}

float Camera::getFrustumFar()const
{
	const Mat4f& M = projection;
	if(isOrthoEnabled())
		return (-1	+	M(2,3))/M(2,2);
	else
		return M(2,3)/(M(2,2)+1);
}

void Camera::setLocation(const Float3& location)
{
	modelView[12]  = -location[0] * modelView[0] - location[1]*modelView[4] - location[2]*modelView[8];
	modelView[13]  = -location[0] * modelView[1] - location[1]*modelView[5] - location[2]*modelView[9];
	modelView[14]  = -location[0] * modelView[2] - location[1]*modelView[6] - location[2]*modelView[10];
	modelView[15] += -location[0] * modelView[3] - location[1]*modelView[7] - location[2]*modelView[11];
	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

Float3 Camera::getLocation()const
{
	return modelView.getRotationPart().inv() * Float3(-modelView(0,3), -modelView(1,3), -modelView(2,3));
}

void Camera::setDirection(const Float3& direction)
{
	if (direction.length()==0)
	{
		return;
	}
	Float3 dv = direction.normalize();
	modelView(2,0) = -dv[0];
	modelView(2,1) = -dv[1];
	modelView(2,2) = -dv[2];
	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

Float3 Camera::getDirection()const
{
	return Float3(-modelView(2,0),-modelView(2,1),-modelView(2,2));
}
void Camera::setUp(const Float3& up)
{
	if (up.length()==0)
	{
		return;
	}
	Float3 dv = up.normalize();
	modelView(1,0) = dv[0];
	modelView(1,1) = dv[1];
	modelView(1,2) = dv[2];
	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

Float3 Camera::getUp()const
{
	return Float3(modelView(1,0),modelView(1,1),modelView(1,2));
}

void Camera::setScalar(const Float3 &scalar)
{
	Float3 v = getScalar();
	v = scalar / v;
	for (int i=0; i<3; i++)
	{
		for (int j=0; j<3; j++)
		{
			modelView(i,j) *= v[j];
		}
	}
	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

Float3 Camera::getScalar()const
{
	Float3 v;
	v[0] = Float3(modelView(0,0), modelView(1,0), modelView(2,0)).length();
	v[1] = Float3(modelView(0,1), modelView(1,1), modelView(2,1)).length();
	v[2] = Float3(modelView(0,2), modelView(1,2), modelView(2,2)).length();
	return v;
}

/**
* Enable/Disable ortho
* */
void Camera::enableOrtho(bool enable)
{
	setFrustum(getFrustumLeft(),getFrustumRight(),getFrustumTop(),getFrustumBottom(),getFrustumNear(),getFrustumFar());
	isOrtho = enable;
}

bool Camera::isOrthoEnabled()const
{
	return isOrtho;
}

Float3 Camera::getWorldCoords(const Float3 &screenCoords)const
{
	float width = viewPortRight - viewPortLeft;
	float height = viewPortBottom - viewPortTop;
	Float4 v((screenCoords[0]-viewPortLeft)/width*2-1,
		(screenCoords[1]-viewPortTop)/height*2-1,
		screenCoords[2]*2-1,1.f);
	v = invModelViewProjection * v;
	v /= v[3];
	return Float3(v[0],v[1],v[2]);
}

Float3 Camera::getScreenCoords(const Float3 &worldCoords)const
{
	float width = viewPortRight - viewPortLeft;
	float height = viewPortBottom - viewPortTop;
	Float4 v(worldCoords[0], worldCoords[1], worldCoords[2],1.f);
	v = modelViewProjection * v;
	v /= v[3];
	Float3 store;
    store[0] = ( v[0] + 1 ) * width / 2   + viewPortLeft;
    store[1] = ( v[1] + 1 ) * height / 2  + viewPortTop;
    store[2] = ( v[2] + 1 ) / 2;

	store[0] = (int)floor(store[0]);
	store[1] = (int)floor(store[1]);
	return store;
}


Float3 Camera::arcballSphereMap(Float2 pos)const
{
    float len;
	float width = fabs(viewPortRight - viewPortLeft);
	float height = fabs(viewPortBottom - viewPortTop);
	float adjustWidth = 1.0f / (width * 0.5f);
	float adjustHeight = 1.0f / (height * 0.5f);
	Float3 store;

    //Adjust point coords and scale down to range of [-1 ... 1]
    pos[0]  =        (pos[0] * adjustWidth)  - 1.0f;
    pos[1]  = 1.0f - (pos[1] * adjustHeight);

    //Compute the square of the length of the vector to the point from the center
	len = pos.sqrLength();	

	//If the point is mapped outside of the sphere... (length > radius squared)
    if (len > 1.0f)
    {
        float norm = 1.0f / sqrtf(len);;
        //Return the "normalized" vector, a point on the sphere
		store[0] = pos[0] * norm;
		store[1] = pos[1] * norm;
		store[2] = 0.0f;
    }
    else    //Else it's on the inside
    {
        //Return a vector to a point mapped inside the sphere sqrt(radius squared - length)
        store[0] = pos[0];
        store[1] = pos[1];
        store[2] = sqrtf(1.f - len);
    }	
	return store;
}

void Camera::arcballClick(Float2 pos)
{
	stVec = arcballSphereMap(pos);
	lastRot.eye();
}

void Camera::arcballDrag(Float2 pos)
{
	if (lastRot.det() == 0)
		return;

	//get rotation quaternion
	Float3 enVec = arcballSphereMap(pos);
	QuaternionF newRot;
	Float3 perp = stVec.cross(enVec);
	if(perp.length() > 1e-5f)
		newRot.fromAngleAxis(2*acos(stVec.dot(enVec)), perp);
	else
		newRot.setIdentity();

	Mat3f thisRot = newRot.toRotationMatrix3();
	Mat3f M = thisRot * lastRot.inv() * modelView.getRotationPart();
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			modelView(i,j) = M(i,j);
	lastRot = thisRot;

	// update transformation matrix
	modelViewProjection = projection * modelView;
	invModelViewProjection = modelViewProjection.inv();
}

const Camera&  Camera::interpolateModelViewMatrixWith(const Camera& rhs, float s)
{
	QuaternionF q1, q2, q;
	Mat3f R1, R2, R;

	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			R1(i,j) = modelView(i,j);
			R2(i,j) = rhs.modelView(i,j);
		}
	}

	q1.fromRotationMatrix(R1);
	q2.fromRotationMatrix(R2);

	//quternion interpolation
	float theta = acos(q1.dot(q2));
	float ws = sin(theta);
	float w1 = sin( (1.f-s)*theta )/ws;
	float w2 = sin( s*theta )/ws;
	q = w1*q1 + w2*q2;
	q.normalize();
	R = q.toRotationMatrix3();

	//reconstruct
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			modelView(i,j) = R(i,j);

	modelView(0,3) = (1.f-s) * modelView(0,3) + s * rhs.modelView(0,3);
	modelView(1,3) = (1.f-s) * modelView(1,3) + s * rhs.modelView(1,3);
	modelView(2,3) = (1.f-s) * modelView(2,3) + s * rhs.modelView(2,3);
	modelView(3,3) = (1.f-s) * modelView(3,3) + s * rhs.modelView(3,3);

	return *this;
}

bool Camera::save(const char* fileName)const
{
	FILE* pFile=NULL;
	int err = fopen_s(&pFile, fileName, "w");
	if(err) return false;

	if (!save(pFile))
		return false;

	fclose(pFile);
	return true;
}
bool Camera::save(FILE* pFile)const
{
	// the orthogonal flag
	fprintf_s(pFile, "Orthogonal:%d\n", (int)isOrtho);

	// the projection matrix
	fprintf_s(pFile, "Projection Matrix:\n");
	for (int i = 0; i<4; i++)
	{
		for (int j = 0; j<4; j++)
			fprintf_s(pFile, "%f ", projection(i, j));
		fprintf_s(pFile, "\n");
	}

	// the model view matrix
	fprintf_s(pFile, "Model View Matrix:\n");
	for (int i = 0; i<4; i++)
	{
		for (int j = 0; j<4; j++)
			fprintf_s(pFile, "%f ", modelView(i, j));
		fprintf_s(pFile, "\n");
	}

	// the viewport
	fprintf_s(pFile, "Viewport:\n");
	fprintf_s(pFile, "%f %f %f %f\n", viewPortLeft, viewPortRight, viewPortTop, viewPortBottom);

	return true;
}
bool Camera::load(const char* fileName)
{
	FILE* pFile=NULL;
	int err = fopen_s(&pFile, fileName, "r");
	if(err) return false;

	if (!load(pFile))
		return false;

	fclose(pFile);
	return true;
}
bool Camera::load(FILE* pFile)
{
	// the orthogonal flag
	int dummy = 0;
	fscanf_s(pFile, "Orthogonal:%d\n", &dummy);
	isOrtho = (dummy == 1);

	// the projection matrix
	fscanf_s(pFile, "Projection Matrix:\n");
	for (int i = 0; i<4; i++)
	{
		for (int j = 0; j<4; j++)
			fscanf_s(pFile, "%f ", &projection(i, j));
		fscanf_s(pFile, "\n");
	}

	// the model view matrix
	fscanf_s(pFile, "Model View Matrix:\n");
	for (int i = 0; i<4; i++)
	{
		for (int j = 0; j<4; j++)
			fscanf_s(pFile, "%f ", &modelView(i, j));
		fscanf_s(pFile, "\n");
	}
	setModelViewMatrix(modelView);

	// the viewport
	fscanf_s(pFile, "Viewport:\n");
	fscanf_s(pFile, "%f %f %f %f\n", &viewPortLeft, &viewPortRight, &viewPortTop, &viewPortBottom);

	return true;
}