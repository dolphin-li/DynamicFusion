#pragma once

#include "common.h"
#include <vector>
#include "Ray.h"
#include "AABB.h"
namespace ldp
{
	namespace kdtree
	{
		class Texture
		{
		public:
			Texture(){ m_Width = 0; m_Height = 0; m_fileName[0] = 0; m_Bitmap.clear(); }
			Texture(const Texture& rhs);
			Texture(Color* a_Bitmap, int a_Width, int a_Height, const char* fileName);
			void Init(Color* a_Bitmap, int a_Width, int a_Height, const char* fileName);
			Color* GetBitmap() { return m_Bitmap.data(); }
			Color GetTexel(real a_U, real a_V);
			const char* GetFileName()const{ return m_fileName; }
			int GetWidth() { return m_Width; }
			int GetHeight() { return m_Height; }
		private:
			std::vector<Color> m_Bitmap;
			char m_fileName[MAX_PATH];
			int m_Width, m_Height;
		};

		// -----------------------------------------------------------
		// Material class definition
		// -----------------------------------------------------------
		class Light;
		class Material
		{
		public:
			Material();
			Material(const Material& r);
			void SetColor(Color& a_Color) { m_Color = a_Color; }
			Color GetColor()const { return m_Color; }
			void SetDiffuse(real a_Diff) { m_Diff = a_Diff; }
			void SetSpecular(real a_Spec) { m_Spec = a_Spec; }
			void SetReflection(real a_Refl) { m_Refl = a_Refl; }
			void SetRefraction(real a_Refr) { m_Refr = a_Refr; }
			void SetParameters(real a_Refl, real a_Refr, Color& a_Col, real a_Diff,
				real a_Spec, real a_Shinny, real a_DRefl, const char* name);
			real GetSpecular()const { return m_Spec; }
			real GetDiffuse()const { return m_Diff; }
			real GetReflection()const { return m_Refl; }
			real GetRefraction()const { return m_Refr; }
			void SetRefrIndex(real a_Refr) { m_RIndex = a_Refr; }
			real GetRefrIndex()const { return m_RIndex; }
			void SetDiffuseRefl(real a_DRefl) { m_DRefl = a_DRefl; }
			real GetDiffuseRefl()const { return m_DRefl; }
			void SetTexture(Color* a_Bitmap, int a_Width, int a_Height, const char* fileName)
			{
				m_Texture.Init(a_Bitmap, a_Width, a_Height, fileName);
			}
			Texture* GetTexture() { return &m_Texture; }
			void SetUVScale(real a_UScale, real a_VScale);
			real GetUScale()const { return m_UScale; }
			real GetVScale()const { return m_VScale; }
			real GetUScaleReci()const { return m_RUScale; }
			real GetVScaleReci()const { return m_RVScale; }
			const char* GetMaterialName()const { return m_name.c_str(); }
			void SetMaterialName(const char* name){ m_name = name; }
			void SetShinny(real s){ m_Shinny = s; }
			real GetShinny()const{ return m_Shinny; }
		private:
			Color m_Color;
			real m_Refl, m_Refr;
			real m_Diff, m_Spec, m_Shinny;
			real m_DRefl;
			real m_RIndex;
			Texture m_Texture;
			real m_UScale, m_VScale, m_RUScale, m_RVScale;
			std::string m_name;
		};

		// -----------------------------------------------------------
		// Primitive class definition
		// -----------------------------------------------------------
		class Light
		{
		public:
			enum
			{
				POINT = 1,
				AREA
			};
			Light(int a_Type, vector3& a_Pos, Color& a_Color)
				: m_Type(a_Type), m_Pos(a_Pos), m_Color(a_Color), m_Grid(0), m_bAsPrimitive(false) {};
			Light(int a_Type, vector3& a_P1, vector3& a_P2, vector3& a_P3, Color& a_Color);
			vector3 GetPos()const { return m_Pos; }
			vector3 GetCellX()const { return m_CellX; }
			vector3 GetCellY()const { return m_CellY; }
			vector3 GetGrid(int a_Idx)const { return m_Grid[a_Idx]; }
			vector3 GetCorner(int i)const
			{
				Vec3 v;
				switch (i)
				{
				default:
					break;
				case 0:
					v = m_Pos - m_CellX*real(2) - m_CellY*real(2);
					break;
				case 1:
					v = m_Pos + m_CellX*real(2) - m_CellY*real(2);
					break;
				case 2:
					v = m_Pos + m_CellX*real(2) + m_CellY*real(2);
					break;
				case 3:
					v = m_Pos - m_CellX*real(2) + m_CellY*real(2);
					break;
				}
				return v;
			}
			Color GetColor()const { return m_Color; }
			int GetType()const  { return m_Type; }
			bool AsPrimitive()const { return m_bAsPrimitive; }
			void SetAsPrimitive(bool en){ m_bAsPrimitive = en; }
		private:
			vector3 m_Pos, m_CellX, m_CellY;
			Color m_Color;
			int m_Type;
			bool m_bAsPrimitive;
			vector3* m_Grid;
		};

		class Primitive
		{
		public:
			int index;
			typedef Vec3 vector3;

			enum
			{
				SPHERE = 1,
				TRIANGLE
			};
			Primitive() { memset(this, 0, sizeof(*this)); };
			Primitive(int a_Type, vector3& a_Centre, real a_Radius);
			Primitive(int a_Type, vector3* a_V1, vector3* a_V2, vector3* a_V3);
			Primitive(int a_Type, vector3* a_V1, vector3* a_V2, vector3* a_V3,
				vector3* a_N1, vector3* a_N2, vector3* a_N3,
				Vec2* a_T1, Vec2* a_T2, Vec2* a_T3);
			~Primitive();
			Material* GetMaterial() { return m_Material; }
			void SetMaterial(Material* a_Mat) { m_Material = a_Mat; }
			int GetType() { return m_Type; }
			int Intersect(const Ray& a_Ray, real& a_Dist, int thread_id);
			bool IntersectBox(AABB& a_Box);
			void CalculateRange(real& a_Pos1, real& a_Pos2, int a_Axis);
			vector3 GetNormal(vector3& a_Pos, int thread_id);
			Color GetColor(vector3& a_Pos, int thread_id);
			// triangle-box intersection stuff
			bool PlaneBoxOverlap(vector3& a_Normal, vector3& a_Vert, vector3& a_MaxBox);
			bool IntersectTriBox(vector3& a_BoxCentre, vector3& a_BoxHalfsize, vector3& a_V0, vector3& a_V1, vector3& a_V2);
			bool IntersectSphereBox(vector3& a_Centre, AABB& a_Box);
			// sphere primitive methods
			vector3& GetCentre() { return m_Centre; }
			real GetSqRadius() { return m_SqRadius; }
			real GetRadius() { return m_Radius; }
			// triangle primitive methods
			vector3 GetFaceNormal() { return m_faceNormal; }
			vector3* GetVertex(int a_Idx) { return m_Vertex[a_Idx]; }
			vector3* GetNormal(int a_Idx)
			{
				if (m_Normal[a_Idx] == 0)
					return &m_faceNormal;
				return m_Normal[a_Idx];
			}
			Vec2* GetTexCoord(int a_Idx) { return m_texCoords[a_Idx]; }
			void SetVertex(int a_Idx, vector3* a_Vertex) { m_Vertex[a_Idx] = a_Vertex; }
			void SetNormal(int a_Idx, vector3* a_Normal) { m_Normal[a_Idx] = a_Normal; }
			void SetTexCoord(int a_Idx, Vec2* a_tex) { m_texCoords[a_Idx] = a_tex; }
			Light* GetLight(){ return m_Light; }
			void SetLight(Light* l){ m_Light = l; }

			FINLINE AABB getAABB()const
			{
				if (m_Type == SPHERE)
				{
					return AABB(m_Centre - m_Radius, m_Centre + m_Radius);
				}
				else
				{
					AABB ab(*m_Vertex[0]);
					for (int k = 0; k < 3; k++)
					{
						ab.min[k] = std::min(ab.min[k], (*m_Vertex[0])[k]);
						ab.max[k] = std::max(ab.max[k], (*m_Vertex[0])[k]);
						ab.min[k] = std::min(ab.min[k], (*m_Vertex[1])[k]);
						ab.max[k] = std::max(ab.max[k], (*m_Vertex[1])[k]);
						ab.min[k] = std::min(ab.min[k], (*m_Vertex[2])[k]);
						ab.max[k] = std::max(ab.max[k], (*m_Vertex[2])[k]);
					}
					return ab;
				}
			}

			FINLINE AABB getClippedAABB(const AABB& r)const
			{
				AABB ab = getAABB();
				ab.clip(r);
				return ab;
			}
			// data members
		private:
			Material* m_Material;							// 4
			int m_Type;										// 4
			Light* m_Light;
			// unified data for primitives
			union
			{
				// sphere
				struct
				{
					vector3 m_Centre;
					real m_SqRadius, m_Radius, m_RRadius;
					vector3 m_Ve, m_Vn, m_Vc;
				};
				// triangle
				struct
				{
					vector3 m_faceNormal;
					vector3 *m_Normal[3];
					vector3 *m_Vertex[3];
					Vec2 *m_texCoords[3];
					real m_U[MAX_THREADS], m_V[MAX_THREADS];

					// for acceleration
					real n_u, n_v, n_d;
					int k;
					real a_u;
					real a_v;
					real b_nu;
					real b_nv;
					real c_nu;
					real c_nv;
				};
			};
		};
	}
}