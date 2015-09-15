#include "Primitive.h"

#include <algorithm>

#undef min
#undef max

namespace ldp
{
	namespace kdtree
	{

		// -----------------------------------------------------------
		// Texture class implementation
		// -----------------------------------------------------------


		Texture::Texture(const Texture& rhs)
		{
			m_Bitmap.assign(rhs.m_Bitmap.begin(), rhs.m_Bitmap.end());
			m_Width = rhs.m_Width;
			m_Height = rhs.m_Height;
			if (rhs.m_fileName == 0)
				m_fileName[0] = 0;
			else
				strcpy_s(m_fileName, rhs.m_fileName);
		}

		Texture::Texture(Color* a_Bitmap, int a_Width, int a_Height, const char* fileName)
		{
			Init(a_Bitmap, a_Width, a_Height, fileName);
		}

		void Texture::Init(Color* a_Bitmap, int a_Width, int a_Height, const char* fileName)
		{
			m_Bitmap.clear();
			m_Width = a_Width;
			m_Height = a_Height;
			if (fileName == 0)
				m_fileName[0] = 0;
			else
				strcpy_s(m_fileName, fileName);
			m_Bitmap.resize(a_Width*a_Height);
			memcpy(m_Bitmap.data(), a_Bitmap, a_Width*a_Height*sizeof(Color));
		}


		Color Texture::GetTexel(real a_U, real a_V)
		{
			if (m_Bitmap.size() == 0) return Color(1, 1, 1, 1);
			// fetch a bilinearly filtered texel
			real fu = (a_U - floor(a_U)) * (m_Width - 1);
			real fv = (a_V - floor(a_V)) * (m_Height - 1);
			int u1 = std::min(m_Width - 1, std::max(0, (int)fu));
			int v1 = std::min(m_Height - 1, std::max(0, (int)fv));
			int u2 = std::min(m_Width - 1, std::max(0, u1 + 1));
			int v2 = std::min(m_Height - 1, std::max(0, v1 + 1));
			// calculate fractional parts of u and v
			real fracu = fu - floor(fu);
			real fracv = fv - floor(fv);
			// calculate weight factors
			real w1 = (1 - fracu) * (1 - fracv);
			real w2 = fracu * (1 - fracv);
			real w3 = (1 - fracu) * fracv;
			real w4 = fracu *  fracv;
			// fetch four texels
			Color c1 = m_Bitmap[u1 + v1 * m_Width];
			Color c2 = m_Bitmap[u2 + v1 * m_Width];
			Color c3 = m_Bitmap[u1 + v2 * m_Width];
			Color c4 = m_Bitmap[u2 + v2 * m_Width];
			// scale and sum the four colors
			return c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4;
		}

		// -----------------------------------------------------------
		// Material class implementation
		// -----------------------------------------------------------

		Material::Material() :
			m_Color(Color(0.2f, 0.2f, 0.2f, 1.0f)),
			m_Refl(0), m_Diff(0.2f), m_Spec(0.8f),
			m_RIndex(1.5f), m_DRefl(0.8f),
			m_UScale(1.0f), m_VScale(1.0f), m_Shinny(10.f)
		{
			m_Texture.Init(0, 0, 0, 0);
			m_name = "";
		}

		Material::Material(const Material& r)
		{
			m_Color = r.m_Color;
			m_Refl = r.m_Refl;
			m_Refr = r.m_Refr;
			m_Diff = r.m_Diff;
			m_Spec = r.m_Spec;
			m_DRefl = r.m_DRefl;
			m_RIndex = r.m_RIndex;
			m_Texture = r.m_Texture;
			m_UScale = r.m_UScale;
			m_VScale = r.m_VScale;
			m_RUScale = r.m_RUScale;
			m_RVScale = r.m_RVScale;
			m_Shinny = r.m_Shinny;
			m_name = r.m_name;
		}

		void Material::SetUVScale(real a_UScale, real a_VScale)
		{
			m_UScale = a_UScale;
			m_VScale = a_VScale;
			m_RUScale = 1.0f / a_UScale;
			m_RVScale = 1.0f / a_VScale;
		}

		void Material::SetParameters(real a_Refl, real a_Refr, Color& a_Col,
			real a_Diff, real a_Spec, real a_Shinny, real a_DRefl, const char* name)
		{
			m_Refl = a_Refl;
			m_Refr = a_Refr;
			m_Color = a_Col;
			m_Diff = a_Diff;
			m_Spec = a_Spec;
			m_Shinny = a_Shinny;
			m_DRefl = a_DRefl;

			if (name == 0)
				m_name = "";
			else
				m_name = name;
		}

		// -----------------------------------------------------------
		// Primitive methods
		// -----------------------------------------------------------

		Primitive::Primitive(int a_Type, vector3& a_Centre, real a_Radius) :Primitive()
		{
			m_Centre = a_Centre;
			m_SqRadius = a_Radius * a_Radius;
			m_Radius = a_Radius;
			m_RRadius = 1.0f / a_Radius;
			m_Type = a_Type;
			m_Material = NULL;
			// set vectors for texture mapping
			m_Vn = vector3(0, 1, 0);
			m_Ve = vector3(1, 0, 0);
			m_Vc = m_Vn.cross(m_Ve);
		}

		Primitive::Primitive(int a_Type, vector3* a_V1, vector3* a_V2, vector3* a_V3) :Primitive()
		{
			m_Type = a_Type;
			m_Material = 0;
			m_Vertex[0] = a_V1;
			m_Vertex[1] = a_V2;
			m_Vertex[2] = a_V3;
			// init precomp
			vector3 A = *m_Vertex[0];
			vector3 B = *m_Vertex[1];
			vector3 C = *m_Vertex[2];
			vector3 c = B - A;
			vector3 b = C - A;
			m_faceNormal = c.cross(b);

			// acceleration
			static const int waldModulo[4] = { 1, 2, 0, 1 };
			k = 0;
			/* Determine the largest projection axis */
			for (int j = 0; j<3; j++) {
				if (std::abs(m_faceNormal[j]) > std::abs(m_faceNormal[k]))
					k = j;
			}

			uint32_t u = waldModulo[k],
				v = waldModulo[k + 1];
			const Float n_k = m_faceNormal[k],
				denom = b[u] * c[v] - b[v] * c[u];

			assert(denom != 0);

			/* Pre-compute intersection calculation constants */
			n_u = m_faceNormal[u] / n_k;
			n_v = m_faceNormal[v] / n_k;
			n_d = (*a_V1).dot(m_faceNormal) / n_k;
			b_nu = b[u] / denom;
			b_nv = -b[v] / denom;
			a_u = A[u];
			a_v = A[v];
			c_nu = c[v] / denom;
			c_nv = -c[u] / denom;

			// finalize normal
			m_faceNormal.normalizeLocal();

			m_Normal[0] = 0;
			m_Normal[1] = 0;
			m_Normal[2] = 0;

			m_texCoords[0] = 0;
			m_texCoords[1] = 0;
			m_texCoords[2] = 0;
		}


		Primitive::Primitive(int a_Type, vector3* a_V1, vector3* a_V2, vector3* a_V3,
			vector3* a_N1, vector3* a_N2, vector3* a_N3,
			Vec2* a_T1, Vec2* a_T2, Vec2* a_T3) :Primitive(a_Type, a_V1, a_V2, a_V3)
		{

			if (a_N1 != NULL)
				m_Normal[0] = a_N1;
			if (a_N2 != NULL)
				m_Normal[1] = a_N2;
			if (a_N3 != NULL)
				m_Normal[2] = a_N3;

			if (a_T1 != NULL)
				m_texCoords[0] = a_T1;
			if (a_T2 != NULL)
				m_texCoords[1] = a_T2;
			if (a_T3 != NULL)
				m_texCoords[2] = a_T3;
		}

		Primitive::~Primitive()
		{
		}

		unsigned int modulo[] = { 1, 2, 0, 1, 2 };
		int Primitive::Intersect(const Ray& a_Ray, real& a_Dist, int thread_id)
		{
			if (m_Type == SPHERE)
			{
				vector3 v = a_Ray.o - m_Centre;
				real b = -DOT(v, a_Ray.d);
				real c = DOT(v, v) - m_SqRadius;
				real det = b*b - c;
				int retval = MISS;
				if (det > 0)
				{
					det = sqrt(det);

					if (b > 0)
					{
						real i1 = c / (b + det);
						real i2 = b + det;
						if (i1 < 0)
						{
							if (i2 < a_Dist)
							{
								a_Dist = i2;
								retval = INPRIM;
							}
						}
						else
						{
							if (i1 < a_Dist)
							{
								a_Dist = i1;
								retval = HIT;
							}
						}
					}
				}
				return retval;
			}
			else
			{
				real o_u, o_v, o_k, d_u, d_v, d_k;
				o_u = a_Ray.o[modulo[k + 0]];
				o_v = a_Ray.o[modulo[k + 1]];
				o_k = a_Ray.o[modulo[k + 2]];
				d_u = a_Ray.d[modulo[k + 0]];
				d_v = a_Ray.d[modulo[k + 1]];
				d_k = a_Ray.d[modulo[k + 2]];

				/* Calculate the plane intersection (Typo in the thesis?) */
				real t = (n_d - o_u*n_u - o_v*n_v - o_k) /
					(d_u * n_u + d_v * n_v + d_k);

				if (t < 0 || t > a_Dist) return MISS;

				/* Calculate the projected plane intersection point */
				const Float hu = o_u + t * d_u - a_u;
				const Float hv = o_v + t * d_v - a_v;

				/* In barycentric coordinates */
				real u = m_U[thread_id] = hv * b_nu + hu * b_nv;
				real v = m_V[thread_id] = hu * c_nu + hv * c_nv;
				if (!(u >= 0 && v >= 0 && u + v <= 1.0f)) return MISS;

				a_Dist = t;
				return (DOT(a_Ray.d, m_faceNormal) > 0) ? INPRIM : HIT;
			}
		}

		vector3 Primitive::GetNormal(vector3& a_Pos, int thread_id)
		{
			if (m_Type == SPHERE)
			{
				return (a_Pos - m_Centre) * m_RRadius;
			}
			else
			{
				vector3 N1, N2, N3;
				if (m_Normal[0] == 0)
					N1 = N2 = N3 = m_faceNormal;
				else
				{
					N1 = *(m_Normal[0]);
					N2 = *(m_Normal[1]);
					N3 = *(m_Normal[2]);
				}
				vector3 N = N1 + m_U[thread_id] * (N2 - N1) + m_V[thread_id] * (N3 - N1);
				NORMALIZE(N);
				return N;
			}
		}

		Color Primitive::GetColor(vector3& a_Pos, int thread_id)
		{
			if (m_Light)
				return m_Light->GetColor();

			Color retval;
			if (!m_Material->GetTexture()) retval = m_Material->GetColor(); else
			{
				if (m_Type == SPHERE)
				{
					vector3 vp = (a_Pos - m_Centre) * m_RRadius;
					real phi = acos(-DOT(vp, m_Vn));
					real u, v = phi * m_Material->GetVScaleReci() * (1.0f / PI);
					real theta = (acos(DOT(m_Ve, vp) / sin(phi))) * (2.0f / PI);
					if (DOT(m_Vc, vp) >= 0) u = (1.0f - theta) * m_Material->GetUScaleReci();
					else u = theta * m_Material->GetUScaleReci();
					retval = m_Material->GetTexture()->GetTexel(u, v) * m_Material->GetColor();
				}
				else
				{
					if (m_texCoords[0] != NULL && m_texCoords[1] != NULL && m_texCoords[2] != NULL)
					{
						Vec2 uv[3];
						for (int k = 0; k < 3; k++)
							uv[k] = *m_texCoords[k];
						real u = uv[0][0] + m_U[thread_id] * (uv[1][0] - uv[0][0]) + m_V[thread_id] * (uv[2][0] - uv[0][0]);
						real v = uv[0][1] + m_U[thread_id] * (uv[1][1] - uv[0][1]) + m_V[thread_id] * (uv[2][1] - uv[0][1]);
						retval = m_Material->GetTexture()->GetTexel(u, v) * m_Material->GetColor();
					}
					else
					{
						retval = m_Material->GetColor();
					}
				}
			}
			return retval;
		}

#define FINDMINMAX( x0, x1, x2, min, max ) \
	min = max = x0; if (x1<min) min = x1; if (x1>max) max = x1; if (x2<min) min = x2; if (x2>max) max = x2;
		// X-tests
#define AXISTEST_X01( a, b, fa, fb )											\
	p0 = a * v0[1] - b * v0[2], p2 = a * v2[1] - b * v2[2]; \
		if (p0 < p2) { min = p0; max = p2; }\
 else { min = p2; max = p0; }			\
 rad = fa * a_BoxHalfsize[1] + fb * a_BoxHalfsize[2];				\
		if (min > rad || max < -rad) return 0;
#define AXISTEST_X2( a, b, fa, fb )												\
	p0 = a * v0[1] - b * v0[2], p1 = a * v1[1] - b * v1[2];	\
		if (p0 < p1) { min = p0; max = p1; }\
	else { min = p1; max = p0; }			\
	rad = fa * a_BoxHalfsize[1] + fb * a_BoxHalfsize[2];				\
		if (min>rad || max < -rad) return 0;
		// Y-tests
#define AXISTEST_Y02( a, b, fa, fb )											\
	p0 = -a * v0[0] + b * v0[2], p2 = -a * v2[0] + b * v2[2]; \
		if (p0 < p2) { min = p0; max = p2; }\
	else { min = p2; max = p0; }			\
	rad = fa * a_BoxHalfsize[0] + fb * a_BoxHalfsize[2];				\
		if (min > rad || max < -rad) return 0;
#define AXISTEST_Y1( a, b, fa, fb )												\
	p0 = -a * v0[0] + b * v0[2], p1 = -a * v1[0] + b * v1[2]; \
		if (p0 < p1) { min = p0; max = p1; }\
	else { min = p1; max = p0; }			\
	rad = fa * a_BoxHalfsize[0] + fb * a_BoxHalfsize[2];				\
		if (min > rad || max < -rad) return 0;
		// Z-tests
#define AXISTEST_Z12( a, b, fa, fb )											\
	p1 = a * v1[0] - b * v1[1], p2 = a * v2[0] - b * v2[1]; \
		if (p2 < p1) { min = p2; max = p1; }\
	else { min = p1; max = p2; }			\
	rad = fa * a_BoxHalfsize[0] + fb * a_BoxHalfsize[1];				\
		if (min > rad || max < -rad) return 0;
#define AXISTEST_Z0( a, b, fa, fb )												\
	p0 = a * v0[0] - b * v0[1], p1 = a * v1[0] - b * v1[1];	\
		if (p0 < p1) { min = p0; max = p1; }\
	else { min = p1; max = p0; }			\
	rad = fa * a_BoxHalfsize[0] + fb * a_BoxHalfsize[1];				\
		if (min > rad || max < -rad) return 0;

		bool Primitive::PlaneBoxOverlap(vector3& a_Normal, vector3& a_Vert, vector3& a_MaxBox)
		{
			vector3 vmin, vmax;
			for (int q = 0; q < 3; q++)
			{
				real v = a_Vert[q];
				if (a_Normal[q] > 0.0f)
				{
					vmin[q] = -a_MaxBox[q] - v;
					vmax[q] = a_MaxBox[q] - v;
				}
				else
				{
					vmin[q] = a_MaxBox[q] - v;
					vmax[q] = -a_MaxBox[q] - v;
				}
			}
			if (DOT(a_Normal, vmin) > real(0.0)) return false;
			if (DOT(a_Normal, vmax) >= real(0.0)) return true;
			return false;
		}

		bool Primitive::IntersectTriBox(vector3& a_BoxCentre, vector3& a_BoxHalfsize, vector3& a_V0, vector3& a_V1, vector3& a_V2)
		{
			vector3 v0, v1, v2, normal, e0, e1, e2;
			real min, max, p0, p1, p2, rad, fex, fey, fez;
			v0 = a_V0 - a_BoxCentre;
			v1 = a_V1 - a_BoxCentre;
			v2 = a_V2 - a_BoxCentre;
			e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;
			fex = fabs(e0[0]);
			fey = fabs(e0[1]);
			fez = fabs(e0[2]);
			AXISTEST_X01(e0[2], e0[1], fez, fey);
			AXISTEST_Y02(e0[2], e0[0], fez, fex);
			AXISTEST_Z12(e0[1], e0[0], fey, fex);
			fex = fabs(e1[0]);
			fey = fabs(e1[1]);
			fez = fabs(e1[2]);
			AXISTEST_X01(e1[2], e1[1], fez, fey);
			AXISTEST_Y02(e1[2], e1[0], fez, fex);
			AXISTEST_Z0(e1[1], e1[0], fey, fex);
			fex = fabs(e2[0]);
			fey = fabs(e2[1]);
			fez = fabs(e2[2]);
			AXISTEST_X2(e2[2], e2[1], fez, fey);
			AXISTEST_Y1(e2[2], e2[0], fez, fex);
			AXISTEST_Z12(e2[1], e2[0], fey, fex);
			FINDMINMAX(v0[0], v1[0], v2[0], min, max);
			if (min > a_BoxHalfsize[0] || max < -a_BoxHalfsize[0]) return false;
			FINDMINMAX(v0[1], v1[1], v2[1], min, max);
			if (min > a_BoxHalfsize[1] || max < -a_BoxHalfsize[1]) return false;
			FINDMINMAX(v0[2], v1[2], v2[2], min, max);
			if (min > a_BoxHalfsize[2] || max < -a_BoxHalfsize[2]) return false;
			normal = e0.cross(e1);
			if (!PlaneBoxOverlap(normal, v0, a_BoxHalfsize)) return false;
			return true;
		}

		bool Primitive::IntersectSphereBox(vector3& a_Centre, AABB& a_Box)
		{
			real dmin = 0;
			vector3 spos = a_Centre;
			vector3 bpos = a_Box.getCorner(0);
			vector3 bsize = a_Box.getExtents();
			for (int i = 0; i < 3; i++)
			{
				if (spos[i] < bpos[i])
				{
					dmin = dmin + (spos[i] - bpos[i]) * (spos[i] - bpos[i]);
				}
				else if (spos[i] > (bpos[i] + bsize[i]))
				{
					dmin = dmin + (spos[i] - (bpos[i] + bsize[i])) * (spos[i] - (bpos[i] + bsize[i]));
				}
			}
			return (dmin <= m_SqRadius);
		}

		bool Primitive::IntersectBox(AABB& a_Box)
		{
			if (m_Type == SPHERE)
			{
				return IntersectSphereBox(m_Centre, a_Box);
			}
			else
			{
				return IntersectTriBox(a_Box.getCenter(), vector3(a_Box.getExtents() * 0.5f),
					*m_Vertex[0], *m_Vertex[1], *m_Vertex[2]);
			}
		}

		void Primitive::CalculateRange(real& a_Pos1, real& a_Pos2, int a_Axis)
		{
			if (m_Type == SPHERE)
			{
				a_Pos1 = m_Centre[a_Axis] - m_Radius;
				a_Pos2 = m_Centre[a_Axis] + m_Radius;
			}
			else
			{
				vector3 pos1 = *m_Vertex[0];
				a_Pos1 = pos1[a_Axis], a_Pos2 = pos1[a_Axis];
				for (int i = 1; i < 3; i++)
				{
					vector3 pos = *m_Vertex[i];
					if (pos[a_Axis] < a_Pos1) a_Pos1 = pos[a_Axis];
					if (pos[a_Axis] > a_Pos2) a_Pos2 = pos[a_Axis];
				}
			}
		}


		// -----------------------------------------------------------
		// Light class implementation
		// -----------------------------------------------------------

		Light::Light(int a_Type, vector3& a_P1, vector3& a_P2, vector3& a_P3, Color& a_Color)
		{
			m_Type = a_Type;
			m_Color = a_Color;
			m_Grid = new vector3[16];
			m_Grid[0] = vector3(1, 2, 0);
			m_Grid[1] = vector3(3, 3, 0);
			m_Grid[2] = vector3(2, 0, 0);
			m_Grid[3] = vector3(0, 1, 0);
			m_Grid[4] = vector3(2, 3, 0);
			m_Grid[5] = vector3(0, 3, 0);
			m_Grid[6] = vector3(0, 0, 0);
			m_Grid[7] = vector3(2, 2, 0);
			m_Grid[8] = vector3(3, 1, 0);
			m_Grid[9] = vector3(1, 3, 0);
			m_Grid[10] = vector3(1, 0, 0);
			m_Grid[11] = vector3(3, 2, 0);
			m_Grid[12] = vector3(2, 1, 0);
			m_Grid[13] = vector3(3, 0, 0);
			m_Grid[14] = vector3(1, 1, 0);
			m_Grid[15] = vector3(0, 2, 0);
			m_CellX = (a_P2 - a_P1) * 0.25f;
			m_CellY = (a_P3 - a_P1) * 0.25f;
			for (int i = 0; i < 16; i++)
				m_Grid[i] = m_Grid[i][0] * m_CellX + m_Grid[i][1] * m_CellY + a_P1;
			m_Pos = a_P1 + 2.f * m_CellX + 2.f * m_CellY;

			m_bAsPrimitive = true;
		}
	}
}