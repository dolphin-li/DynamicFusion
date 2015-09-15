/************************************************************************/
/* NOTE: THIS file is only for bmesh.cpp, you cannot include it directly
/************************************************************************/
#if defined(_MSC_VER)
#  define __func__ __FUNCTION__
#endif
#define BM_DISK_EDGE_LINK_GET(e, v)  (                                        \
	((v) == ((BMEdge *)(e))->v1) ?                                            \
	&((e)->v1_disk_link) :                                                \
	&((e)->v2_disk_link)                                                  \
	)
#define MAKE_ID(a,b,c,d) ( (int)(d)<<24 | (int)(c)<<16 | (b)<<8 | (a) )
#define FREEWORD MAKE_ID('f', 'r', 'e', 'e')
#define MEMPOOL_ELEM_SIZE_MIN (sizeof(void *) * 2)
#define MIN2(x,y)               ( (x)<(y) ? (x) : (y) )
#define MIN3(x,y,z)             MIN2( MIN2((x),(y)) , (z) )
#define MIN4(x,y,z,a)           MIN2( MIN2((x),(y)) , MIN2((z),(a)) )

#define MAX2(x,y)               ( (x)>(y) ? (x) : (y) )
#define MAX3(x,y,z)             MAX2( MAX2((x),(y)) , (z) )
#define MAX4(x,y,z,a)           MAX2( MAX2((x),(y)) , MAX2((z),(a)) )
#define INIT_MINMAX(min, max) {                                               \
	(min)[0]= (min)[1]= (min)[2]= 1.0e30f;                                \
	(max)[0]= (max)[1]= (max)[2]= -1.0e30f;                               \
}
#define INIT_MINMAX2(min, max) {                                              \
	(min)[0]= (min)[1]= 1.0e30f;                                          \
	(max)[0]= (max)[1]= -1.0e30f;                                         \
} (void)0
#define DO_MIN(vec, min) {                                                    \
	if( (min)[0]>(vec)[0] ) (min)[0]= (vec)[0];                           \
	if( (min)[1]>(vec)[1] ) (min)[1]= (vec)[1];                           \
	if( (min)[2]>(vec)[2] ) (min)[2]= (vec)[2];                           \
} (void)0
#define DO_MAX(vec, max) {                                                    \
	if( (max)[0]<(vec)[0] ) (max)[0]= (vec)[0];                           \
	if( (max)[1]<(vec)[1] ) (max)[1]= (vec)[1];                           \
	if( (max)[2]<(vec)[2] ) (max)[2]= (vec)[2];                           \
} (void)0
#define DO_MINMAX(vec, min, max) {                                            \
	if( (min)[0]>(vec)[0] ) (min)[0]= (vec)[0];                           \
	if( (min)[1]>(vec)[1] ) (min)[1]= (vec)[1];                           \
	if( (min)[2]>(vec)[2] ) (min)[2]= (vec)[2];                           \
	if( (max)[0]<(vec)[0] ) (max)[0]= (vec)[0];                           \
	if( (max)[1]<(vec)[1] ) (max)[1]= (vec)[1];                           \
	if( (max)[2]<(vec)[2] ) (max)[2]= (vec)[2];                           \
} (void)0
#define DO_MINMAX2(vec, min, max) {                                           \
	if( (min)[0]>(vec)[0] ) (min)[0]= (vec)[0];                           \
	if( (min)[1]>(vec)[1] ) (min)[1]= (vec)[1];                           \
	if( (max)[0]<(vec)[0] ) (max)[0]= (vec)[0];                           \
	if( (max)[1]<(vec)[1] ) (max)[1]= (vec)[1];                           \
} (void)0
#ifndef SWAP
#define SWAP(type, a, b)	{ type sw_ap; sw_ap=(a); (a)=(b); (b)=sw_ap; }
#endif
#define BM_FACE_FIRST_LOOP(p) ((p)->l_first)
#define UNLIKELY(e) (e)
#define LIKELY(e) (e)
#define BMESH_ASSERT(e) assert(e)
#undef NULL
#undef TRUE
#undef FALSE
#define TRUE 1
#define FALSE 0
#define NULL 0
#ifndef M_PI
#define M_PI        3.14159265358979323846f
#endif
#define SMALL_NUMBER 1e-8f
#define MINLINE static __forceinline
#define BM_NGON_STACK_SIZE 32
#define BM_LOOPS_OF_FACE 11
#ifndef M_SQRT2
#define M_SQRT2     1.41421356237309504880
#endif
#define ISECT_LINE_LINE_COLINEAR	-1
#define ISECT_LINE_LINE_NONE		 0
#define ISECT_LINE_LINE_EXACT		 1
#define ISECT_LINE_LINE_CROSS		 2
#define SET_INT_IN_POINTER(i)    ((void *)(intptr_t)(i))
#define GET_INT_FROM_POINTER(i)  ((int)(intptr_t)(i))

MINLINE void mul_v3_fl(float r[3], float f)
{
	r[0] *= f;
	r[1] *= f;
	r[2] *= f;
}
MINLINE void mul_v2_fl(float r[3], float f)
{
	r[0] *= f;
	r[1] *= f;
}

MINLINE void mul_v3_v3fl(float r[3], const float a[3], float f)
{
	r[0] = a[0] * f;
	r[1] = a[1] * f;
	r[2] = a[2] * f;
}
MINLINE void mul_v3_v3(float r[3], const float a[3])
{
	r[0] *= a[0];
	r[1] *= a[1];
	r[2] *= a[2];
}
MINLINE void sub_v3_v3(float r[3], const float a[3])
{
	r[0] -= a[0];
	r[1] -= a[1];
	r[2] -= a[2];
}
MINLINE void madd_v3_v3fl(float r[3], const float a[3], float f)
{
	r[0] += a[0] * f;
	r[1] += a[1] * f;
	r[2] += a[2] * f;
}

MINLINE void madd_v3_v3v3(float r[3], const float a[3], const float b[3])
{
	r[0] += a[0] * b[0];
	r[1] += a[1] * b[1];
	r[2] += a[2] * b[2];
}

MINLINE void sub_v3_v3v3(float r[3], const float a[3], const float b[3])
{
	r[0] = a[0] - b[0];
	r[1] = a[1] - b[1];
	r[2] = a[2] - b[2];
}
MINLINE void add_v3_v3(float r[3], const float a[3])
{
	r[0] += a[0];
	r[1] += a[1];
	r[2] += a[2];
}
MINLINE void madd_v3_v3v3fl(float r[3], const float a[3], const float b[3], float f)
{
	r[0] = a[0] + b[0] * f;
	r[1] = a[1] + b[1] * f;
	r[2] = a[2] + b[2] * f;
}

MINLINE void madd_v3_v3v3v3(float r[3], const float a[3], const float b[3], const float c[3])
{
	r[0] = a[0] + b[0] * c[0];
	r[1] = a[1] + b[1] * c[1];
	r[2] = a[2] + b[2] * c[2];
}

MINLINE void add_v3_v3v3(float r[3], const float a[3], const float b[3])
{
	r[0] = a[0] + b[0];
	r[1] = a[1] + b[1];
	r[2] = a[2] + b[2];
}
MINLINE void mid_v3_v3v3(float v[3], const float v1[3], const float v2[3])
{
	v[0] = 0.5f * (v1[0] + v2[0]);
	v[1] = 0.5f * (v1[1] + v2[1]);
	v[2] = 0.5f * (v1[2] + v2[2]);
}
MINLINE void negate_v3(float r[3])
{
	r[0] = -r[0];
	r[1] = -r[1];
	r[2] = -r[2];
}

MINLINE void negate_v3_v3(float r[3], const float a[3])
{
	r[0] = -a[0];
	r[1] = -a[1];
	r[2] = -a[2];
}

MINLINE int compare_v3v3(const float v1[3], const float v2[3], const float limit)
{
	if (fabsf(v1[0] - v2[0]) < limit)
		if (fabsf(v1[1] - v2[1]) < limit)
			if (fabsf(v1[2] - v2[2]) < limit)
				return 1;

	return 0;
}
MINLINE void copy_v3_v3(float r[3], const float a[3])
{
	r[0] = a[0];
	r[1] = a[1];
	r[2] = a[2];
}
MINLINE float saasin(float fac)
{
	if (fac <= -1.0f) return (float)-M_PI / 2.0f;
	else if (fac >= 1.0f) return (float)M_PI / 2.0f;
	else return (float)asin(fac);
}
MINLINE float saacos(float fac)
{
	if (fac <= -1.0f) return (float)M_PI;
	else if (fac >= 1.0f) return 0.0;
	else return (float)acos(fac);
}
MINLINE void zero_v3(float r[3])
{
	r[0] = 0.0f;
	r[1] = 0.0f;
	r[2] = 0.0f;
}

MINLINE void cross_v3_v3v3(float r[3], const float a[3], const float b[3])
{
	r[0] = a[1] * b[2] - a[2] * b[1];
	r[1] = a[2] * b[0] - a[0] * b[2];
	r[2] = a[0] * b[1] - a[1] * b[0];
}
MINLINE float cross_v2v2(const float a[2], const float b[2])
{
	return a[0] * b[1] - a[1] * b[0];
}
MINLINE float dot_v3v3(const float* x, const float* y)
{
	return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
}
MINLINE float len_v3(const float* d)
{
	return sqrtf(dot_v3v3(d,d));
}
MINLINE float len_v3v3(const float* a, const float*b)
{
	float c[3];
	sub_v3_v3v3(c,a,b);
	return len_v3(c);
}
MINLINE float normalize_v3_v3(float r[3], const float a[3])
{
	float d = dot_v3v3(a, a);

	/* a larger value causes normalize errors in a
	 * scaled down models with camera xtreme close */
	if (d > 1.0e-35f) {
		d = sqrtf(d);
		mul_v3_v3fl(r, a, 1.0f / d);
	}
	else {
		zero_v3(r);
		d = 0.0f;
	}

	return d;
}
MINLINE double normalize_v3_d(double n[3])
{
	double d = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];

	/* a larger value causes normalize errors in a
	 * scaled down models with camera xtreme close */
	if (d > 1.0e-35) {
		double mul;

		d = sqrt(d);
		mul = 1.0 / d;

		n[0] *= mul;
		n[1] *= mul;
		n[2] *= mul;
	}
	else {
		n[0] = n[1] = n[2] = 0;
		d = 0.0;
	}

	return d;
}
MINLINE float normalize_v3(float n[3])
{
	return normalize_v3_v3(n, n);
}
MINLINE float angle_normalized_v3v3(const float v1[3], const float v2[3])
{
	/* this is the same as acos(dot_v3v3(v1, v2)), but more accurate */
	if (dot_v3v3(v1, v2) < 0.0f) {
		float vec[3];

		vec[0] = -v2[0];
		vec[1] = -v2[1];
		vec[2] = -v2[2];

		return (float)M_PI - 2.0f * (float)saasin(len_v3v3(vec, v1) / 2.0f);
	}
	else
		return 2.0f * (float)saasin(len_v3v3(v2, v1) / 2.0f);
}
MINLINE float angle_v3v3v3(const float v1[3], const float v2[3], const float v3[3])
{
	float vec1[3], vec2[3];

	sub_v3_v3v3(vec1, v2, v1);
	sub_v3_v3v3(vec2, v2, v3);
	normalize_v3(vec1);
	normalize_v3(vec2);

	return angle_normalized_v3v3(vec1, vec2);
}
MINLINE void add_newell_cross_v3_v3v3(float n[3], const float v_prev[3], const float v_curr[3])
{
	n[0] += (v_prev[1] - v_curr[1]) * (v_prev[2] + v_curr[2]);
	n[1] += (v_prev[2] - v_curr[2]) * (v_prev[0] + v_curr[0]);
	n[2] += (v_prev[0] - v_curr[0]) * (v_prev[1] + v_curr[1]);
}
MINLINE float shell_angle_to_dist(const float angle)
{
	return (angle < SMALL_NUMBER) ? 1.0f : fabsf(1.0f / cosf(angle));
}
MINLINE void unit_qt(float q[4])
{
	q[0] = 1.0f;
	q[1] = q[2] = q[3] = 0.0f;
}

void axis_angle_to_quat(float q[4], const float axis[3], float angle)
{
	float nor[3];
	float si;

	if (normalize_v3_v3(nor, axis) == 0.0f) {
		unit_qt(q);
		return;
	}

	angle /= 2;
	si = (float)sin(angle);
	q[0] = (float)cos(angle);
	q[1] = nor[0] * si;
	q[2] = nor[1] * si;
	q[3] = nor[2] * si;
}

MINLINE float line_point_side_v2(const float l1[2], const float l2[2], const float pt[2])
{
	return (((l1[0] - pt[0]) * (l2[1] - pt[1])) -
		((l2[0] - pt[0]) * (l1[1] - pt[1])));
}

MINLINE float len_squared_v3v3(const float a[3], const float b[3])
{
	float d[3];

	sub_v3_v3v3(d, b, a);
	return dot_v3v3(d, d);
}
MINLINE void mul_v3_m3v3(float r[3], float M[3][3], float a[3])
{
	r[0] = M[0][0] * a[0] + M[1][0] * a[1] + M[2][0] * a[2];
	r[1] = M[0][1] * a[0] + M[1][1] * a[1] + M[2][1] * a[2];
	r[2] = M[0][2] * a[0] + M[1][2] * a[1] + M[2][2] * a[2];
}

MINLINE void mul_m3_v3(float M[3][3], float r[3])
{
	float tmp[3];

	mul_v3_m3v3(tmp, M, r);
	copy_v3_v3(r, tmp);
}

void quat_to_mat3(float m[][3], const float q[4])
{
	double q0, q1, q2, q3, qda, qdb, qdc, qaa, qab, qac, qbb, qbc, qcc;

	q0 = M_SQRT2 * (double)q[0];
	q1 = M_SQRT2 * (double)q[1];
	q2 = M_SQRT2 * (double)q[2];
	q3 = M_SQRT2 * (double)q[3];

	qda = q0 * q1;
	qdb = q0 * q2;
	qdc = q0 * q3;
	qaa = q1 * q1;
	qab = q1 * q2;
	qac = q1 * q3;
	qbb = q2 * q2;
	qbc = q2 * q3;
	qcc = q3 * q3;

	m[0][0] = (float)(1.0 - qbb - qcc);
	m[0][1] = (float)(qdc + qab);
	m[0][2] = (float)(-qdb + qac);

	m[1][0] = (float)(-qdc + qab);
	m[1][1] = (float)(1.0 - qaa - qcc);
	m[1][2] = (float)(qda + qbc);

	m[2][0] = (float)(qdb + qac);
	m[2][1] = (float)(-qda + qbc);
	m[2][2] = (float)(1.0 - qaa - qbb);
}

void cent_tri_v3(float cent[3], const float v1[3], const float v2[3], const float v3[3])
{
	cent[0] = 0.33333f * (v1[0] + v2[0] + v3[0]);
	cent[1] = 0.33333f * (v1[1] + v2[1] + v3[1]);
	cent[2] = 0.33333f * (v1[2] + v2[2] + v3[2]);
}

void cent_quad_v3(float cent[3], const float v1[3], const float v2[3], const float v3[3], const float v4[3])
{
	cent[0] = 0.25f * (v1[0] + v2[0] + v3[0] + v4[0]);
	cent[1] = 0.25f * (v1[1] + v2[1] + v3[1] + v4[1]);
	cent[2] = 0.25f * (v1[2] + v2[2] + v3[2] + v4[2]);
}

float normal_tri_v3(float n[3], const float v1[3], const float v2[3], const float v3[3])
{
	float n1[3], n2[3];

	n1[0] = v1[0] - v2[0];
	n2[0] = v2[0] - v3[0];
	n1[1] = v1[1] - v2[1];
	n2[1] = v2[1] - v3[1];
	n1[2] = v1[2] - v2[2];
	n2[2] = v2[2] - v3[2];
	n[0] = n1[1] * n2[2] - n1[2] * n2[1];
	n[1] = n1[2] * n2[0] - n1[0] * n2[2];
	n[2] = n1[0] * n2[1] - n1[1] * n2[0];

	return normalize_v3(n);
}

float normal_quad_v3(float n[3], const float v1[3], const float v2[3], const float v3[3], const float v4[3])
{
	/* real cross! */
	float n1[3], n2[3];

	n1[0] = v1[0] - v3[0];
	n1[1] = v1[1] - v3[1];
	n1[2] = v1[2] - v3[2];

	n2[0] = v2[0] - v4[0];
	n2[1] = v2[1] - v4[1];
	n2[2] = v2[2] - v4[2];

	n[0] = n1[1] * n2[2] - n1[2] * n2[1];
	n[1] = n1[2] * n2[0] - n1[0] * n2[2];
	n[2] = n1[0] * n2[1] - n1[1] * n2[0];

	return normalize_v3(n);
}

float area_tri_v2(const float v1[2], const float v2[2], const float v3[2])
{
	return 0.5f * fabsf((v1[0] - v2[0]) * (v2[1] - v3[1]) + (v1[1] - v2[1]) * (v3[0] - v2[0]));
}

float area_tri_signed_v2(const float v1[2], const float v2[2], const float v3[2])
{
	return 0.5f * ((v1[0] - v2[0]) * (v2[1] - v3[1]) + (v1[1] - v2[1]) * (v3[0] - v2[0]));
}

/* only convex Quadrilaterals */
float area_quad_v3(const float v1[3], const float v2[3], const float v3[3], const float v4[3])
{
	float len, vec1[3], vec2[3], n[3];

	sub_v3_v3v3(vec1, v2, v1);
	sub_v3_v3v3(vec2, v4, v1);
	cross_v3_v3v3(n, vec1, vec2);
	len = normalize_v3(n);

	sub_v3_v3v3(vec1, v4, v3);
	sub_v3_v3v3(vec2, v2, v3);
	cross_v3_v3v3(n, vec1, vec2);
	len += normalize_v3(n);

	return (len / 2.0f);
}

/* Triangles */
float area_tri_v3(const float v1[3], const float v2[3], const float v3[3])
{
	float len, vec1[3], vec2[3], n[3];

	sub_v3_v3v3(vec1, v3, v2);
	sub_v3_v3v3(vec2, v1, v2);
	cross_v3_v3v3(n, vec1, vec2);
	len = normalize_v3(n);

	return (len / 2.0f);
}

float area_poly_v3(int nr, float verts[][3], const float normal[3])
{
	float x, y, z, area, max;
	float *cur, *prev;
	int a, px = 0, py = 1;

	/* first: find dominant axis: 0==X, 1==Y, 2==Z
	 * don't use 'axis_dominant_v3()' because we need max axis too */
	x = fabsf(normal[0]);
	y = fabsf(normal[1]);
	z = fabsf(normal[2]);
	max = MAX3(x, y, z);
	if (max == y) py = 2;
	else if (max == x) {
		px = 1;
		py = 2;
	}

	/* The Trapezium Area Rule */
	prev = verts[nr - 1];
	cur = verts[0];
	area = 0;
	for (a = 0; a < nr; a++) {
		area += (cur[px] - prev[px]) * (cur[py] + prev[py]);
		prev = verts[a];
		cur = verts[a + 1];
	}

	return fabsf(0.5f * area / max);
}

/* intersect Line-Line, shorts */
int isect_line_line_v2_int(const int v1[2], const int v2[2], const int v3[2], const int v4[2])
{
	float div, labda, mu;

	div = (float)((v2[0] - v1[0]) * (v4[1] - v3[1]) - (v2[1] - v1[1]) * (v4[0] - v3[0]));
	if (div == 0.0f) return ISECT_LINE_LINE_COLINEAR;

	labda = ((float)(v1[1] - v3[1]) * (v4[0] - v3[0]) - (v1[0] - v3[0]) * (v4[1] - v3[1])) / div;

	mu = ((float)(v1[1] - v3[1]) * (v2[0] - v1[0]) - (v1[0] - v3[0]) * (v2[1] - v1[1])) / div;

	if (labda >= 0.0f && labda <= 1.0f && mu >= 0.0f && mu <= 1.0f) {
		if (labda == 0.0f || labda == 1.0f || mu == 0.0f || mu == 1.0f) return ISECT_LINE_LINE_EXACT;
		return ISECT_LINE_LINE_CROSS;
	}
	return ISECT_LINE_LINE_NONE;
}

/* intersect Line-Line, floats */
int isect_line_line_v2(const float v1[2], const float v2[2], const float v3[2], const float v4[2])
{
	float div, labda, mu;

	div = (v2[0] - v1[0]) * (v4[1] - v3[1]) - (v2[1] - v1[1]) * (v4[0] - v3[0]);
	if (div == 0.0f) return ISECT_LINE_LINE_COLINEAR;

	labda = ((float)(v1[1] - v3[1]) * (v4[0] - v3[0]) - (v1[0] - v3[0]) * (v4[1] - v3[1])) / div;

	mu = ((float)(v1[1] - v3[1]) * (v2[0] - v1[0]) - (v1[0] - v3[0]) * (v2[1] - v1[1])) / div;

	if (labda >= 0.0f && labda <= 1.0f && mu >= 0.0f && mu <= 1.0f) {
		if (labda == 0.0f || labda == 1.0f || mu == 0.0f || mu == 1.0f) return ISECT_LINE_LINE_EXACT;
		return ISECT_LINE_LINE_CROSS;
	}
	return ISECT_LINE_LINE_NONE;
}

/* point in tri */

int isect_point_tri_v2(const float pt[2], const float v1[2], const float v2[2], const float v3[2])
{
	if (line_point_side_v2(v1, v2, pt) >= 0.0f) {
		if (line_point_side_v2(v2, v3, pt) >= 0.0f) {
			if (line_point_side_v2(v3, v1, pt) >= 0.0f) {
				return 1;
			}
		}
	}
	else {
		if (!(line_point_side_v2(v2, v3, pt) >= 0.0f)) {
			if (!(line_point_side_v2(v3, v1, pt) >= 0.0f)) {
				return -1;
			}
		}
	}

	return 0;
}

/* point in quad - only convex quads */
int isect_point_quad_v2(const float pt[2], const float v1[2], const float v2[2], const float v3[2], const float v4[2])
{
	if (line_point_side_v2(v1, v2, pt) >= 0.0f) {
		if (line_point_side_v2(v2, v3, pt) >= 0.0f) {
			if (line_point_side_v2(v3, v4, pt) >= 0.0f) {
				if (line_point_side_v2(v4, v1, pt) >= 0.0f) {
					return 1;
				}
			}
		}
	}
	else {
		if (!(line_point_side_v2(v2, v3, pt) >= 0.0f)) {
			if (!(line_point_side_v2(v3, v4, pt) >= 0.0f)) {
				if (!(line_point_side_v2(v4, v1, pt) >= 0.0f)) {
					return -1;
				}
			}
		}
	}

	return 0;
}


/**
 * \brief COMPUTE POLY NORMAL
 *
 * Computes the normal of a planar
 * polygon See Graphics Gems for
 * computing newell normal.
 */
static void calc_poly_normal(float normal[3], float* verts, int nverts)
{
	float const *v_prev = &verts[3*(nverts - 1)];
	float const *v_curr = &verts[0];
	float n[3] = {0.0f};
	int i;

	/* Newell's Method */
	for (i = 0; i < nverts; v_prev = v_curr, v_curr = &verts[3*(++i)]) {
		add_newell_cross_v3_v3v3(n, v_prev, v_curr);
	}

	if (UNLIKELY(normalize_v3_v3(normal, n) == 0.0f)) {
		normal[2] = 1.0f; /* other axis set to 0.0 */
	}
}
/**
 * COMPUTE POLY PLANE
 *
 * Projects a set polygon's vertices to
 * a plane defined by the average
 * of its edges cross products
 */
void calc_poly_plane(float (*verts)[3], const int nverts)
{
	
	float avgc[3], norm[3], mag, avgn[3];
	float *v1, *v2, *v3;
	int i;
	
	if (nverts < 3)
		return;

	zero_v3(avgn);
	zero_v3(avgc);

	for (i = 0; i < nverts; i++) {
		v1 = verts[i];
		v2 = verts[(i + 1) % nverts];
		v3 = verts[(i + 2) % nverts];
		normal_tri_v3(norm, v1, v2, v3);

		add_v3_v3(avgn, norm);
	}

	if (UNLIKELY(normalize_v3(avgn) == 0.0f)) {
		avgn[2] = 1.0f;
	}
	
	for (i = 0; i < nverts; i++) {
		v1 = verts[i];
		mag = dot_v3v3(v1, avgn);
		madd_v3_v3fl(v1, avgn, -mag);
	}
}

/**
 * \brief BM LEGAL EDGES
 *
 * takes in a face and a list of edges, and sets to NULL any edge in
 * the list that bridges a concave region of the face or intersects
 * any of the faces's edges.
 */
static void shrink_edgef(float v1[3], float v2[3], const float fac)
{
	float mid[3];

	mid_v3_v3v3(mid, v1, v2);

	sub_v3_v3v3(v1, v1, mid);
	sub_v3_v3v3(v2, v2, mid);

	mul_v3_fl(v1, fac);
	mul_v3_fl(v2, fac);

	add_v3_v3v3(v1, v1, mid);
	add_v3_v3v3(v2, v2, mid);
}


/**
 * \brief POLY ROTATE PLANE
 *
 * Rotates a polygon so that it's
 * normal is pointing towards the mesh Z axis
 */
void poly_rotate_plane(const float normal[3], float (*verts), const int nverts)
{

	float up[3] = {0.0f, 0.0f, 1.0f}, axis[3], q[4];
	float mat[3][3];
	double angle;
	int i;

	cross_v3_v3v3(axis, normal, up);

	angle = saacos(dot_v3v3(normal, up));

	if (angle == 0.0) return;

	axis_angle_to_quat(q, axis, (float)angle);
	quat_to_mat3(mat, q);

	for (i = 0; i < nverts; i++)
		mul_m3_v3(mat, &verts[i*3]);
}
/************************************************************************/
/* BLI_mempool
/************************************************************************/

typedef struct Link
{
	struct Link *next,*prev;
} Link;

/* use this when it is not worth defining a custom one... */
typedef struct LinkData
{
	struct LinkData *next, *prev;
	void *data;
} LinkData;

/* never change the size of this! genfile.c detects pointerlen with it */
typedef struct ListBase 
{
	void *first, *last;
} ListBase;

typedef struct BLI_freenode {
	struct BLI_freenode *next;
	int freeword; /* used to identify this as a freed node */
} BLI_freenode;

typedef struct BLI_mempool_chunk {
	struct BLI_mempool_chunk *next, *prev;
	void *data;
} BLI_mempool_chunk;

struct BLI_mempool {
	struct ListBase chunks;
	int esize;         /* element size in bytes */
	int csize;         /* chunk size in bytes */
	int pchunk;        /* number of elements per chunk */
	int flag;
	/* keeps aligned to 16 bits */

	BLI_freenode *free;    /* free element list. Interleaved into chunk datas. */
	int totalloc, totused; /* total number of elements allocated in total,
	                        * and currently in use */
};

/* Ripped this from blender.c */
void BLI_movelisttolist(ListBase *dst, ListBase *src)
{
	if (src->first==NULL) return;

	if (dst->first==NULL) {
		dst->first= src->first;
		dst->last= src->last;
	}
	else {
		((Link *)dst->last)->next= (Link*)src->first;
		((Link *)src->first)->prev= (Link*)dst->last;
		dst->last= src->last;
	}
	src->first= src->last= NULL;
}

void BLI_addhead(ListBase *listbase, void *vlink)
{
	Link *link= (Link*)vlink;

	if (link == NULL) return;
	if (listbase == NULL) return;

	link->next = (Link*)listbase->first;
	link->prev = NULL;

	if (listbase->first) ((Link *)listbase->first)->prev = link;
	if (listbase->last == NULL) listbase->last = link;
	listbase->first = link;
}

void BLI_addtail(ListBase *listbase, void *vlink)
{
	Link *link= (Link*)vlink;

	if (link == NULL) return;
	if (listbase == NULL) return;

	link->next = NULL;
	link->prev = (Link*)listbase->last;

	if (listbase->last) ((Link *)listbase->last)->next = link;
	if (listbase->first == NULL) listbase->first = link;
	listbase->last = link;
}

void BLI_remlink(ListBase *listbase, void *vlink)
{
	Link *link= (Link*)vlink;

	if (link == NULL) return;
	if (listbase == NULL) return;

	if (link->next) link->next->prev = link->prev;
	if (link->prev) link->prev->next = link->next;

	if (listbase->last == link) listbase->last = link->prev;
	if (listbase->first == link) listbase->first = link->next;
}

int BLI_findindex(const ListBase *listbase, void *vlink)
{
	Link *link= NULL;
	int number= 0;

	if (listbase == NULL) return -1;
	if (vlink == NULL) return -1;

	link= (Link*)listbase->first;
	while (link) {
		if (link == vlink)
			return number;

		number++;
		link= link->next;
	}

	return -1;
}

int BLI_remlink_safe(ListBase *listbase, void *vlink)
{
	if (BLI_findindex(listbase, vlink) != -1) {
		BLI_remlink(listbase, vlink);
		return 1;
	}
	else {
		return 0;
	}
}

void BLI_insertlink(ListBase *listbase, void *vprevlink, void *vnewlink)
{
	Link *prevlink= (Link*)vprevlink;
	Link *newlink= (Link*)vnewlink;

	/* newlink comes after prevlink */
	if (newlink == NULL) return;
	if (listbase == NULL) return;

	/* empty list */
	if (listbase->first == NULL) { 

		listbase->first= newlink;
		listbase->last= newlink;
		return;
	}

	/* insert before first element */
	if (prevlink == NULL) {	
		newlink->next= (Link*)listbase->first;
		newlink->prev= NULL;
		newlink->next->prev= newlink;
		listbase->first= newlink;
		return;
	}

	/* at end of list */
	if (listbase->last== prevlink) 
		listbase->last = newlink;

	newlink->next= prevlink->next;
	prevlink->next= newlink;
	if (newlink->next) newlink->next->prev= newlink;
	newlink->prev= prevlink;
}


void BLI_insertlinkafter(ListBase *listbase, void *vprevlink, void *vnewlink)
{
	Link *prevlink= (Link*)vprevlink;
	Link *newlink= (Link*)vnewlink;

	/* newlink before nextlink */
	if (newlink == NULL) return;
	if (listbase == NULL) return;

	/* empty list */
	if (listbase->first == NULL) { 
		listbase->first= newlink;
		listbase->last= newlink;
		return;
	}

	/* insert at head of list */
	if (prevlink == NULL) {	
		newlink->prev = NULL;
		newlink->next = (Link*)listbase->first;
		((Link *)listbase->first)->prev = newlink;
		listbase->first = newlink;
		return;
	}

	/* at end of list */
	if (listbase->last == prevlink) 
		listbase->last = newlink;

	newlink->next = prevlink->next;
	newlink->prev = prevlink;
	prevlink->next = newlink;
	if (newlink->next) newlink->next->prev = newlink;
}
/* This uses insertion sort, so NOT ok for large list */
void BLI_sortlist(ListBase *listbase, int (*cmp)(void *, void *))
{
	Link *current = NULL;
	Link *previous = NULL;
	Link *next = NULL;

	if (cmp == NULL) return;
	if (listbase == NULL) return;

	if (listbase->first != listbase->last) {
		for (previous = (Link*)listbase->first, current = previous->next; current; current = next) {
			next = current->next;
			previous = current->prev;

			BLI_remlink(listbase, current);

			while (previous && cmp(previous, current) == 1)
			{
				previous = previous->prev;
			}

			BLI_insertlinkafter(listbase, previous, current);
		}
	}
}


void BLI_insertlinkbefore(ListBase *listbase, void *vnextlink, void *vnewlink)
{
	Link *nextlink= (Link*)vnextlink;
	Link *newlink= (Link*)vnewlink;

	/* newlink before nextlink */
	if (newlink == NULL) return;
	if (listbase == NULL) return;

	/* empty list */
	if (listbase->first == NULL) { 
		listbase->first= newlink;
		listbase->last= newlink;
		return;
	}

	/* insert at end of list */
	if (nextlink == NULL) {	
		newlink->prev= (Link*)listbase->last;
		newlink->next= NULL;
		((Link *)listbase->last)->next= newlink;
		listbase->last= newlink;
		return;
	}

	/* at beginning of list */
	if (listbase->first== nextlink) 
		listbase->first = newlink;

	newlink->next= nextlink;
	newlink->prev= nextlink->prev;
	nextlink->prev= newlink;
	if (newlink->prev) newlink->prev->next= newlink;
}


void BLI_freelist(ListBase *listbase)
{
	Link *link, *next;

	if (listbase == NULL) 
		return;

	link= (Link*)listbase->first;
	while (link) {
		next= link->next;
		free(link);
		link= next;
	}

	listbase->first= NULL;
	listbase->last= NULL;
}

int BLI_countlist(const ListBase *listbase)
{
	Link *link;
	int count = 0;

	if (listbase) {
		link = (Link*)listbase->first;
		while (link) {
			count++;
			link= link->next;
		}
	}
	return count;
}

void *BLI_findlink(const ListBase *listbase, int number)
{
	Link *link = NULL;

	if (number >= 0) {
		link = (Link*)listbase->first;
		while (link != NULL && number != 0) {
			number--;
			link = link->next;
		}
	}

	return link;
}

void *BLI_rfindlink(const ListBase *listbase, int number)
{
	Link *link = NULL;

	if (number >= 0) {
		link = (Link*)listbase->last;
		while (link != NULL && number != 0) {
			number--;
			link = link->prev;
		}
	}

	return link;
}

void *BLI_findstring(const ListBase *listbase, const char *id, const int offset)
{
	Link *link= NULL;
	const char *id_iter;

	if (listbase == NULL) return NULL;

	for (link= (Link*)listbase->first; link; link= link->next) {
		id_iter= ((const char *)link) + offset;

		if (id[0] == id_iter[0] && strcmp(id, id_iter)==0) {
			return link;
		}
	}

	return NULL;
}
/* same as above but find reverse */
void *BLI_rfindstring(const ListBase *listbase, const char *id, const int offset)
{
	Link *link= NULL;
	const char *id_iter;

	if (listbase == NULL) return NULL;

	for (link= (Link*)listbase->last; link; link= link->prev) {
		id_iter= ((const char *)link) + offset;

		if (id[0] == id_iter[0] && strcmp(id, id_iter)==0) {
			return link;
		}
	}

	return NULL;
}

void *BLI_findstring_ptr(const ListBase *listbase, const char *id, const int offset)
{
	Link *link= NULL;
	const char *id_iter;

	if (listbase == NULL) return NULL;

	for (link= (Link*)listbase->first; link; link= link->next) {
		/* exact copy of BLI_findstring(), except for this line */
		id_iter= *((const char **)(((const char *)link) + offset));

		if (id[0] == id_iter[0] && strcmp(id, id_iter)==0) {
			return link;
		}
	}

	return NULL;
}
/* same as above but find reverse */
void *BLI_rfindstring_ptr(const ListBase *listbase, const char *id, const int offset)
{
	Link *link= NULL;
	const char *id_iter;

	if (listbase == NULL) return NULL;

	for (link= (Link*)listbase->last; link; link= link->prev) {
		/* exact copy of BLI_rfindstring(), except for this line */
		id_iter= *((const char **)(((const char *)link) + offset));

		if (id[0] == id_iter[0] && strcmp(id, id_iter)==0) {
			return link;
		}
	}

	return NULL;
}

int BLI_findstringindex(const ListBase *listbase, const char *id, const int offset)
{
	Link *link= NULL;
	const char *id_iter;
	int i= 0;

	if (listbase == NULL) return -1;

	link= (Link*)listbase->first;
	while (link) {
		id_iter= ((const char *)link) + offset;

		if (id[0] == id_iter[0] && strcmp(id, id_iter)==0)
			return i;
		i++;
		link= link->next;
	}

	return -1;
}

BLI_mempool *BLI_mempool_create(int esize, int totelem, int pchunk, int flag)
{
	BLI_mempool *pool = NULL;
	BLI_freenode *lasttail = NULL, *curnode = NULL;
	int i, j, maxchunks;
	char *addr;

	/* allocate the pool structure */
	pool = (BLI_mempool *)malloc(sizeof(BLI_mempool));


	/* set the elem size */
	if (esize < MEMPOOL_ELEM_SIZE_MIN) {
		esize = MEMPOOL_ELEM_SIZE_MIN;
	}

	if (flag & BLI_MEMPOOL_ALLOW_ITER) {
		pool->esize = MAX2(esize, sizeof(BLI_freenode));
	}
	else {
		pool->esize = esize;
	}

	pool->flag = flag;
	pool->pchunk = pchunk;
	pool->csize = esize * pchunk;
	pool->chunks.first = pool->chunks.last = NULL;
	pool->totused = 0;
	pool->totalloc = 0;

	maxchunks = totelem / pchunk + 1;
	if (maxchunks == 0) {
		maxchunks = 1;
	}

	/* allocate the actual chunks */
	for (i = 0; i < maxchunks; i++) {
		BLI_mempool_chunk *mpchunk;

		mpchunk = (BLI_mempool_chunk*)malloc(sizeof(BLI_mempool_chunk));
		mpchunk->data = malloc(pool->csize);


		mpchunk->next = mpchunk->prev = NULL;
		BLI_addtail(&(pool->chunks), mpchunk);

		if (i == 0) {
			pool->free = (BLI_freenode*)mpchunk->data; /* start of the list */
			if (pool->flag & BLI_MEMPOOL_ALLOW_ITER) {
				pool->free->freeword = FREEWORD;
			}
		}

		/* loop through the allocated data, building the pointer structures */
		for (addr = (char*)mpchunk->data, j = 0; j < pool->pchunk; j++) {
			curnode = ((BLI_freenode *)addr);
			addr += pool->esize;
			curnode->next = (BLI_freenode *)addr;
			if (pool->flag & BLI_MEMPOOL_ALLOW_ITER) {
				if (j != pool->pchunk - 1)
					curnode->next->freeword = FREEWORD;
				curnode->freeword = FREEWORD;
			}
		}
		/* final pointer in the previously allocated chunk is wrong */
		if (lasttail) {
			lasttail->next = (BLI_freenode*)mpchunk->data;
			if (pool->flag & BLI_MEMPOOL_ALLOW_ITER) {
				lasttail->freeword = FREEWORD;
			}
		}

		/* set the end of this chunks memoryy to the new tail for next iteration */
		lasttail = curnode;

		pool->totalloc += pool->pchunk;
	}
	/* terminate the list */
	curnode->next = NULL;
	return pool;
}

void *BLI_mempool_alloc(BLI_mempool *pool)
{
	void *retval = NULL;

	pool->totused++;

	if (!(pool->free)) {
		BLI_freenode *curnode = NULL;
		char *addr;
		int j;

		/* need to allocate a new chunk */
		BLI_mempool_chunk *mpchunk;

		mpchunk       = (BLI_mempool_chunk*)malloc(sizeof(BLI_mempool_chunk));
		mpchunk->data = malloc(pool->csize);


		mpchunk->next = mpchunk->prev = NULL;
		BLI_addtail(&(pool->chunks), mpchunk);

		pool->free = (BLI_freenode*)mpchunk->data; /* start of the list */

		if (pool->flag & BLI_MEMPOOL_ALLOW_ITER) {
			pool->free->freeword = FREEWORD;
		}

		for (addr = (char*)mpchunk->data, j = 0; j < pool->pchunk; j++) {
			curnode = ((BLI_freenode *)addr);
			addr += pool->esize;
			curnode->next = (BLI_freenode *)addr;

			if (pool->flag & BLI_MEMPOOL_ALLOW_ITER) {
				curnode->freeword = FREEWORD;
				if (j != pool->pchunk - 1)
					curnode->next->freeword = FREEWORD;
			}
		}
		curnode->next = NULL; /* terminate the list */

		pool->totalloc += pool->pchunk;
	}

	retval = pool->free;

	if (pool->flag & BLI_MEMPOOL_ALLOW_ITER) {
		pool->free->freeword = 0x7FFFFFFF;
	}

	pool->free = pool->free->next;
	//memset(retval, 0, pool->esize);
	return retval;
}

void *BLI_mempool_calloc(BLI_mempool *pool)
{
	void *retval = BLI_mempool_alloc(pool);
	memset(retval, 0, pool->esize);
	return retval;
}

/* doesnt protect against double frees, don't be stupid! */
void BLI_mempool_free(BLI_mempool *pool, void *addr)
{
	BLI_freenode *newhead = (BLI_freenode*)addr;

	if (pool->flag & BLI_MEMPOOL_ALLOW_ITER) {
		newhead->freeword = FREEWORD;
	}

	newhead->next = pool->free;
	pool->free = newhead;

	pool->totused--;

	/* nothing is in use; free all the chunks except the first */
	if (pool->totused == 0) {
		BLI_freenode *curnode = NULL;
		char *tmpaddr = NULL;
		int i;

		BLI_mempool_chunk *mpchunk = NULL;
		BLI_mempool_chunk *first = (BLI_mempool_chunk*)pool->chunks.first;

		BLI_remlink(&pool->chunks, first);

		for (mpchunk = (BLI_mempool_chunk *)pool->chunks.first; mpchunk; mpchunk = mpchunk->next) {
			free(mpchunk->data);
		}
		BLI_freelist(&(pool->chunks));


		BLI_addtail(&pool->chunks, first);
		pool->totalloc = pool->pchunk;

		pool->free = (BLI_freenode*)first->data; /* start of the list */
		for (tmpaddr = (char*)first->data, i = 0; i < pool->pchunk; i++) {
			curnode = ((BLI_freenode *)tmpaddr);
			tmpaddr += pool->esize;
			curnode->next = (BLI_freenode *)tmpaddr;
		}
		curnode->next = NULL; /* terminate the list */
	}
}

int BLI_mempool_count(BLI_mempool *pool)
{
	return pool->totused;
}

void *BLI_mempool_findelem(BLI_mempool *pool, int index)
{
	if (!(pool->flag & BLI_MEMPOOL_ALLOW_ITER)) {
		fprintf(stderr, "%s: Error! you can't iterate over this mempool!\n", __func__);
		return NULL;
	}
	else if ((index >= 0) && (index < pool->totused)) {
		/* we could have some faster mem chunk stepping code inline */
		BLI_mempool_iter iter;
		void *elem;
		BLI_mempool_iternew(pool, &iter);
		for (elem = BLI_mempool_iterstep(&iter); index-- != 0; elem = BLI_mempool_iterstep(&iter)) {
			/* do nothing */
		};
		return elem;
	}

	return NULL;
}

void BLI_mempool_iternew(BLI_mempool *pool, BLI_mempool_iter *iter)
{
	if (!(pool->flag & BLI_MEMPOOL_ALLOW_ITER)) {
		fprintf(stderr, "%s: Error! you can't iterate over this mempool!\n", __func__);
		iter->curchunk = NULL;
		iter->curindex = 0;

		return;
	}

	iter->pool = pool;
	iter->curchunk = (BLI_mempool_chunk*)pool->chunks.first;
	iter->curindex = 0;
}

#if 0
/* unoptimized, more readable */

static void *bli_mempool_iternext(BLI_mempool_iter *iter)
{
	void *ret = NULL;

	if (!iter->curchunk || !iter->pool->totused) return NULL;

	ret = ((char *)iter->curchunk->data) + iter->pool->esize * iter->curindex;

	iter->curindex++;

	if (iter->curindex >= iter->pool->pchunk) {
		iter->curchunk = iter->curchunk->next;
		iter->curindex = 0;
	}

	return ret;
}

void *BLI_mempool_iterstep(BLI_mempool_iter *iter)
{
	BLI_freenode *ret;

	do {
		ret = bli_mempool_iternext(iter);
	} while (ret && ret->freeword == FREEWORD);

	return ret;
}

#else

/* optimized version of code above */

void *BLI_mempool_iterstep(BLI_mempool_iter *iter)
{
	BLI_freenode *ret;

	if (UNLIKELY(iter->pool->totused == 0)) {
		return NULL;
	}

	do {
		if (LIKELY(iter->curchunk)) {
			ret = (BLI_freenode *)(((char *)iter->curchunk->data) + iter->pool->esize * iter->curindex);
		}
		else {
			return NULL;
		}

		if (UNLIKELY(++iter->curindex >= iter->pool->pchunk)) {
			iter->curindex = 0;
			iter->curchunk = iter->curchunk->next;
		}
	} while (ret->freeword == FREEWORD);

	return ret;
}

#endif

void BLI_mempool_destroy(BLI_mempool *pool)
{
	if(pool==0)	return;
	BLI_mempool_chunk *mpchunk = NULL;

	for (mpchunk = (BLI_mempool_chunk *)pool->chunks.first; mpchunk; mpchunk = mpchunk->next) {
		free(mpchunk->data);
	}
	BLI_freelist(&(pool->chunks));
	free(pool);

}

/************************************************************************/
/* Heap for bmesh
/************************************************************************/
struct HeapNode {
	void *ptr;
	float value;
	int index;
};

struct Heap {
	unsigned int size;
	unsigned int bufsize;
	HeapNode *freenodes;
	HeapNode *nodes;
	HeapNode **tree;
};
typedef	void	(*HeapFreeFP)(void *ptr);
#define HEAP_PARENT(i) ((i - 1) >> 1)
#define HEAP_LEFT(i)   ((i << 1) + 1)
#define HEAP_RIGHT(i)  ((i << 1) + 2)
#define HEAP_COMPARE(a, b) (a->value < b->value)
#define HEAP_EQUALS(a, b) (a->value == b->value)
#define HEAP_SWAP(heap, i, j) \
{                                                                             \
	SWAP(int, heap->tree[i]->index, heap->tree[j]->index);                    \
	SWAP(HeapNode *, heap->tree[i], heap->tree[j]);                           \
}

/***/

Heap *BLI_heap_new(void)
{
	Heap *heap = (Heap *)malloc(sizeof(Heap));
	heap->bufsize = 1;
	heap->tree = (HeapNode **)malloc(sizeof(HeapNode *));

	return heap;
}

void BLI_heap_free(Heap *heap, HeapFreeFP ptrfreefp)
{
	unsigned int i;

	if (ptrfreefp)
		for (i = 0; i < heap->size; i++)
			ptrfreefp(heap->tree[i]->ptr);

	free(heap->tree);
	free(heap);
}

static void BLI_heap_down(Heap *heap, int i)
{
	while (1) {
		int size = heap->size, smallest;
		int l = HEAP_LEFT(i);
		int r = HEAP_RIGHT(i);

		smallest = ((l < size) && HEAP_COMPARE(heap->tree[l], heap->tree[i])) ? l : i;

		if ((r < size) && HEAP_COMPARE(heap->tree[r], heap->tree[smallest]))
			smallest = r;

		if (smallest == i)
			break;

		HEAP_SWAP(heap, i, smallest);
		i = smallest;
	}
}

static void BLI_heap_up(Heap *heap, int i)
{
	while (i > 0) {
		int p = HEAP_PARENT(i);

		if (HEAP_COMPARE(heap->tree[p], heap->tree[i]))
			break;

		HEAP_SWAP(heap, p, i);
		i = p;
	}
}

HeapNode *BLI_heap_insert(Heap *heap, float value, void *ptr)
{
	HeapNode *node;

	if ((heap->size + 1) > heap->bufsize) {
		int newsize = heap->bufsize * 2;
		HeapNode **newtree;

		newtree = (HeapNode **)malloc(newsize * sizeof(*newtree));
		memcpy(newtree, heap->tree, sizeof(HeapNode *) * heap->size);
		free(heap->tree);

		heap->tree = newtree;
		heap->bufsize = newsize;
	}

	if (heap->freenodes) {
		node = heap->freenodes;
		heap->freenodes = (HeapNode *)(((HeapNode *)heap->freenodes)->ptr);
	}
	else
		node = (HeapNode *)malloc(sizeof *node);

	node->value = value;
	node->ptr = ptr;
	node->index = heap->size;

	heap->tree[node->index] = node;

	heap->size++;

	BLI_heap_up(heap, heap->size - 1);

	return node;
}

int BLI_heap_empty(Heap *heap)
{
	return (heap->size == 0);
}

int BLI_heap_size(Heap *heap)
{
	return heap->size;
}

HeapNode *BLI_heap_top(Heap *heap)
{
	return heap->tree[0];
}

void *BLI_heap_popmin(Heap *heap)
{
	void *ptr = heap->tree[0]->ptr;

	heap->tree[0]->ptr = heap->freenodes;
	heap->freenodes = heap->tree[0];

	if (heap->size == 1)
		heap->size--;
	else {
		HEAP_SWAP(heap, 0, heap->size-1);
		heap->size--;

		BLI_heap_down(heap, 0);
	}

	return ptr;
}

void BLI_heap_remove(Heap *heap, HeapNode *node)
{
	int i = node->index;

	while (i > 0) {
		int p = HEAP_PARENT(i);

		HEAP_SWAP(heap, p, i);
		i = p;
	}

	BLI_heap_popmin(heap);
}

float BLI_heap_node_value(HeapNode *node)
{
	return node->value;
}

void *BLI_heap_node_ptr(HeapNode *node)
{
	return node->ptr;
}

/************************************************************************/
/* Hash for bmesh
/************************************************************************/
typedef unsigned int	(*GHashHashFP)		(const void *key);
typedef int				(*GHashCmpFP)		(const void *a, const void *b);
typedef	void			(*GHashKeyFreeFP)	(void *key);
typedef void			(*GHashValFreeFP)	(void *val);

typedef struct Entry {
	struct Entry *next;

	void *key, *val;
} Entry;

typedef struct GHash {
	GHashHashFP	hashfp;
	GHashCmpFP	cmpfp;

	Entry **buckets;
	struct BLI_mempool *entrypool;
	int nbuckets, nentries, cursize;
} GHash;

typedef struct GHashIterator {
	GHash *gh;
	int curBucket;
	struct Entry *curEntry;
} GHashIterator;
typedef struct GHashPair {
	const void *first;
	int second;
} GHashPair;

unsigned int hashsizes[] = {
	5, 11, 17, 37, 67, 131, 257, 521, 1031, 2053, 4099, 8209, 
	16411, 32771, 65537, 131101, 262147, 524309, 1048583, 2097169, 
	4194319, 8388617, 16777259, 33554467, 67108879, 134217757, 
	268435459
};

/***/

GHash *BLI_ghash_new(GHashHashFP hashfp, GHashCmpFP cmpfp, const char *info)
{
	GHash *gh = (GHash*)malloc(sizeof(*gh));
	gh->hashfp = hashfp;
	gh->cmpfp = cmpfp;
	gh->entrypool = BLI_mempool_create(sizeof(Entry), 64, 64, 0);

	gh->cursize = 0;
	gh->nentries = 0;
	gh->nbuckets = hashsizes[gh->cursize];

	gh->buckets = (Entry**)malloc(gh->nbuckets * sizeof(*gh->buckets));
	memset(gh->buckets, 0, gh->nbuckets * sizeof(*gh->buckets));

	return gh;
}

int BLI_ghash_size(GHash *gh)
{
	return gh->nentries;
}

void BLI_ghash_insert(GHash *gh, void *key, void *val)
{
	unsigned int hash = gh->hashfp(key) % gh->nbuckets;
	Entry *e = (Entry*)BLI_mempool_alloc(gh->entrypool);

	e->key = key;
	e->val = val;
	e->next = gh->buckets[hash];
	gh->buckets[hash] = e;

	if (++gh->nentries > (float)gh->nbuckets / 2) {
		Entry **old = gh->buckets;
		int i, nold = gh->nbuckets;

		gh->nbuckets = hashsizes[++gh->cursize];
		gh->buckets = (Entry**)malloc(gh->nbuckets * sizeof(*gh->buckets));
		memset(gh->buckets, 0, gh->nbuckets * sizeof(*gh->buckets));

		for (i = 0; i < nold; i++) {
			for (e = old[i]; e;) {
				Entry *n = e->next;

				hash = gh->hashfp(e->key) % gh->nbuckets;
				e->next = gh->buckets[hash];
				gh->buckets[hash] = e;

				e = n;
			}
		}

		free(old);
	}
}

void *BLI_ghash_lookup(GHash *gh, const void *key)
{
	if (gh) {
		unsigned int hash = gh->hashfp(key) % gh->nbuckets;
		Entry *e;

		for (e = gh->buckets[hash]; e; e = e->next)
			if (gh->cmpfp(key, e->key) == 0)
				return &e->val;
	}
	return NULL;
}

int BLI_ghash_remove(GHash *gh, void *key, GHashKeyFreeFP keyfreefp, GHashValFreeFP valfreefp)
{
	unsigned int hash = gh->hashfp(key) % gh->nbuckets;
	Entry *e;
	Entry *p = NULL;

	for (e = gh->buckets[hash]; e; e = e->next) {
		if (gh->cmpfp(key, e->key) == 0) {
			Entry *n = e->next;

			if (keyfreefp)
				keyfreefp(e->key);
			if (valfreefp)
				valfreefp(e->val);
			BLI_mempool_free(gh->entrypool, e);

			/* correct but 'e' isn't used before return */
			/* e= n; *//*UNUSED*/
			if (p)
				p->next = n;
			else
				gh->buckets[hash] = n;

			--gh->nentries;
			return 1;
		}
		p = e;
	}

	return 0;
}

int BLI_ghash_haskey(GHash *gh, void *key)
{
	unsigned int hash = gh->hashfp(key) % gh->nbuckets;
	Entry *e;

	for (e = gh->buckets[hash]; e; e = e->next)
		if (gh->cmpfp(key, e->key) == 0)
			return 1;

	return 0;
}

void BLI_ghash_free(GHash *gh, GHashKeyFreeFP keyfreefp, GHashValFreeFP valfreefp)
{
	int i;

	if (keyfreefp || valfreefp) {
		for (i = 0; i < gh->nbuckets; i++) {
			Entry *e;

			for (e = gh->buckets[i]; e;) {
				Entry *n = e->next;

				if (keyfreefp) keyfreefp(e->key);
				if (valfreefp) valfreefp(e->val);

				e = n;
			}
		}
	}

	free(gh->buckets);
	BLI_mempool_destroy(gh->entrypool);
	gh->buckets = NULL;
	gh->nentries = 0;
	gh->nbuckets = 0;
	free(gh);
}

/***/

GHashIterator *BLI_ghashIterator_new(GHash *gh)
{
	GHashIterator *ghi = (GHashIterator *)malloc(sizeof(*ghi));
	ghi->gh = gh;
	ghi->curEntry = NULL;
	ghi->curBucket = -1;
	while (!ghi->curEntry) {
		ghi->curBucket++;
		if (ghi->curBucket == ghi->gh->nbuckets)
			break;
		ghi->curEntry = ghi->gh->buckets[ghi->curBucket];
	}
	return ghi;
}
void BLI_ghashIterator_init(GHashIterator *ghi, GHash *gh)
{
	ghi->gh = gh;
	ghi->curEntry = NULL;
	ghi->curBucket = -1;
	while (!ghi->curEntry) {
		ghi->curBucket++;
		if (ghi->curBucket == ghi->gh->nbuckets)
			break;
		ghi->curEntry = ghi->gh->buckets[ghi->curBucket];
	}
}
void BLI_ghashIterator_free(GHashIterator *ghi)
{
	free(ghi);
}

void *BLI_ghashIterator_getKey(GHashIterator *ghi)
{
	return ghi->curEntry ? ghi->curEntry->key : NULL;
}
void *BLI_ghashIterator_getValue(GHashIterator *ghi)
{
	return ghi->curEntry ? ghi->curEntry->val : NULL;
}

void BLI_ghashIterator_step(GHashIterator *ghi)
{
	if (ghi->curEntry) {
		ghi->curEntry = ghi->curEntry->next;
		while (!ghi->curEntry) {
			ghi->curBucket++;
			if (ghi->curBucket == ghi->gh->nbuckets)
				break;
			ghi->curEntry = ghi->gh->buckets[ghi->curBucket];
		}
	}
}
int BLI_ghashIterator_isDone(GHashIterator *ghi)
{
	return !ghi->curEntry;
}

/***/

unsigned int BLI_ghashutil_ptrhash(const void *key)
{
	return (unsigned int)(intptr_t)key;
}
int BLI_ghashutil_ptrcmp(const void *a, const void *b)
{
	if (a == b)
		return 0;
	else
		return (a < b) ? -1 : 1;
}

unsigned int BLI_ghashutil_inthash(const void *ptr)
{
	uintptr_t key = (uintptr_t)ptr;

	key += ~(key << 16);
	key ^=  (key >>  5);
	key +=  (key <<  3);
	key ^=  (key >> 13);
	key += ~(key <<  9);
	key ^=  (key >> 17);

	return (unsigned int)(key & 0xffffffff);
}

int BLI_ghashutil_intcmp(const void *a, const void *b)
{
	if (a == b)
		return 0;
	else
		return (a < b) ? -1 : 1;
}

unsigned int BLI_ghashutil_strhash(const void *ptr)
{
	const char *s = (const char*)ptr;
	unsigned int i = 0;
	unsigned char c;

	while ((c = *s++)) {
		i = i * 37 + c;
	}

	return i;
}
int BLI_ghashutil_strcmp(const void *a, const void *b)
{
	return strcmp((const char*)a, (const char*)b);
}

GHashPair *BLI_ghashutil_pairalloc(const void *first, int second)
{
	GHashPair *pair = (GHashPair *)malloc(sizeof(GHashPair));
	pair->first = first;
	pair->second = second;
	return pair;
}

unsigned int BLI_ghashutil_pairhash(const void *ptr)
{
	const GHashPair *pair = (const GHashPair *)ptr;
	unsigned int hash = BLI_ghashutil_ptrhash(pair->first);
	return hash ^ BLI_ghashutil_inthash(SET_INT_IN_POINTER(pair->second));
}

int BLI_ghashutil_paircmp(const void *a, const void *b)
{
	const GHashPair *A = (const GHashPair *)a;
	const GHashPair *B = (const GHashPair *)b;

	int cmp = BLI_ghashutil_ptrcmp(A->first, B->first);
	if (cmp == 0)
		return BLI_ghashutil_intcmp(SET_INT_IN_POINTER(A->second), SET_INT_IN_POINTER(B->second));
	return cmp;
}

void BLI_ghashutil_pairfree(void *ptr)
{
	free((void*)ptr);
}

/************************************************************************/
/* Template Priority Queue
/************************************************************************/
/**
* A minimal heap.
* Two things must be implemented by T:
* operator <, for heap setup.
* function getIndex(), get the original position of elements, for heap position update.
* The index of T must be 0~size.
* */

template<class T>
class LHeap
{
private:
	vector<T*> _heap;		
	vector<int> _pos;
	int size;
	int capacity;
public:
	LHeap()
	{
		_heap.clear();
		_pos.clear();
		size = 0;
		capacity = 0;
	}
	~LHeap()
	{
		release();
	}

	LHeap& init(T* src, int nSrc)
	{
		release();
		capacity = nSrc;
		size = nSrc;
		_heap.resize(size);
		_pos.resize(size);
		for(int i=0; i< size; i++)
		{
			_heap[i]= &src[i];
			_pos[_heap[i]->getIndex()] = i;
		}
		//make heap
		for(int i=((size>>1)-1); i>=0; i--)
		{
			heapDown(i);
		}
		return *this;
	}

	void release()
	{
		size = 0;
		capacity = 0;
		_heap.clear();
		_pos.clear();
	}

	//push elements with insertion sort.
	void push(T* t)			
	{
		assert(size < capacity);
		_heap[size]= t;
		_pos[_heap[size]->getIndex()]=size;
		heapUp(size);
		size++;
	}

	//get the top of the heap
	T* top()
	{
		assert(size > 0);
		return _heap[0];
	}

	//pop the top of heap
	T* pop()
	{
		assert(size > 0);
		T * minElement = _heap[0];
		--size;
		move_heap(size,0);
		if(size>0)
			heapDown(0);
		return minElement;
	}

	bool isEmpty()const
	{
		return size <= 0;
	}

	int getSize()const
	{
		return size;
	}

	bool isInHeap(T* edge)const
	{
		if(!edge)
			return false;
		return _pos[edge->getIndex()]<size && _pos[edge->getIndex()]>=0;
	}

	//I don't check whether "edge" is contained here, so make sure the parameter "edge" is valide.
	T* remove(T* edge)
	{
		assert(size > 0);
		assert(edge->getIndex() >= 0 && edge->getIndex() < capacity);
		int p = _pos[edge->getIndex()];
		T * minElement = _heap[p];
		--size;
		move_heap(size,p);
		if(size>p)
			heapDown(p);
		return minElement;
	}

	//I don't check whether "edge" is contained here, so make sure the parameter "edge" is valide.
	void updatePos(T* edge)
	{
		assert(edge->getIndex() >= 0 && edge->getIndex() < capacity);
		int p = _pos[edge->getIndex()];
		heapUp(p);
		heapDown(p);
	}

private:
	int father(const int& i)const
	{
		return (i-1) >> 1;
	}
	int child1(const int& i)const
	{
		return (i<<1) + 1;
	}
	int child2(const int& i)const
	{
		return (i<<1) + 2;
	}
	void move_heap(int startId, int endId)
	{
		_heap[endId] = _heap[startId];
		_pos[_heap[startId]->getIndex()] = endId;
	}
	void move_heap(T* s, int endId)
	{
		_heap[endId] = s;
		_pos[s->getIndex()] = endId;
	}
	void heapUp(const int& i)
	{
		assert(i>=0 && i<size);
		int k = i;
		T  *ths = _heap[i];
		while(k!=0 && *ths < *_heap[father(k)])
		{
			move_heap(father(k),k);
			k = father(k);
		}
		move_heap(ths, k);
	}
	void heapDown(const int& i)
	{
		assert(i>=0 && i<size);
		int child = i;
		int k = i;
		T  *ths = _heap[i];
		for(k = i; child1(k) < size; k = child)
		{
			//find the smaller child.
			child = child1(k);
			if(child < size-1 && *_heap[child+1] < *_heap[child])
				child++;
			//down one level.
			if(!(*ths < *_heap[child]))
				move_heap(child,k);
			else
				break;
		}
		move_heap(ths, k);
	}
};