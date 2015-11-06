#include "CpuGaussNewton.h"
#include "WarpField.h"
#include "LMSolver.h"
#include <helper_math.h>
#include <set>
#include "ldp_basic_mat.h"
#include <queue>
namespace dfusion
{
#define ENABLE_DEBUG_DUMP_MATRIX_EACH_ITER
#define SHOW_LAST_NODE_INFO

#define USE_ROBUST_HUBER_PENALTY
#define USE_ROBUST_TUKEY_PENALTY
#define ENABLE_ANTIPODALITY
#define ENABLE_DIALG_HESSIAN
#define ENABLE_DOUBLE_EDGE
	inline float3 read_float3_from_4(float4 p)
	{
		return make_float3(p.x, p.y, p.z);
	}
	inline float norm(float3 a)
	{
		return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
	}


	struct EigenContainter
	{
		typedef float real;
		typedef Eigen::Matrix<real, -1, 1> Vec;
		typedef Eigen::Matrix<real, 6, 6> Mat6;
		typedef Eigen::Matrix<real, -1, -1> Mat;
		typedef Eigen::SparseMatrix<real, Eigen::ColMajor> SpMat;

		// 6-twist of exponential map of dual-quaternion
		Vec x_;
		SpMat jac_, jact_;
		std::vector<Mat6> m_diagBlocks;
		int nodesInLevel0_;

		std::vector<KnnIdx> vmapKnn_;
		std::vector<KnnIdx> nodesKnn_;
		std::vector<float4> nodesVw_;
		std::vector<float4> vmap_cano_;
		std::vector<float4> nmap_cano_;
		std::vector<float4> vmap_warp_;
		std::vector<float4> nmap_warp_;
		std::vector<float4> vmap_live_;
		std::vector<float4> nmap_live_;
		std::vector<int> map_c2l_corr_;
		std::vector<int> row_index_of_pixel_;
		std::vector<int> pixel_index_of_row_;
		std::vector<int> coo_pos_of_pixel_;
		int nPixel_rows_;
		int nPixel_cooPos_;
		int imgWidth_;
		int imgHeight_;
		Param param_;
		Intr intr_;
		Tbx::Transfo Tlw_;
		std::vector<Eigen::Triplet<real>> m_cooSys;


		void checkNanAndInf(const Vec& vec, const Vec& fx, const Vec& g, const SpMat& H)
		{
			for (size_t i = 0; i < vec.size(); i++)
			{
				if (isnan(vec[i]) || isinf(vec[i]))
				{
					printf("warning: nan/inf found: %d=%f; pixelRow: %d\n", i, vec[i], nPixel_rows_);
					dumpSparseMatrix(jac_, "D:/ana.txt");
					dumpSparseMatrix(H, "D:/H.txt");
					dumpVec(fx, "D:/fx.txt");
					dumpVec(g, "D:/g.txt");
					dumpVec(vec, "D:/x.txt");
					system("pause");
				}
			}
		}

		void checkNanAndInf(const SpMat& H)
		{
			for (int r = 0; r < H.outerSize(); r++)
			{
				int rs = H.outerIndexPtr()[r];
				int re = H.outerIndexPtr()[r + 1];
				for (int c = rs; c < re; c++)
				{
					real v = H.valuePtr()[c];
					if (isnan(v) || isinf(v))
					{
						printf("warning: nan/inf found: %d %d=%f; pixelRow: %d\n", r, 
							H.innerIndexPtr()[c], v, nPixel_rows_);
						dumpSparseMatrix(jac_, "D:/ana.txt");
						dumpSparseMatrix(H, "D:/H.txt");
						system("pause");
					}
				}
			}
		}

		void checkLinearSolver(const SpMat& A, const Vec& x, const Vec& b)
		{
			real err = (A*x - b).norm() / (b.norm() + 1e-5);
			if (err > 1e-1)
			{
				printf("linear solver error: %f = %f/%f\n", err, (A*x - b).norm(), b.norm());
				dumpSparseMatrix(jac_, "D:/ana.txt");
				dumpSparseMatrix(A, "D:/H.txt");
				dumpVec(x, "D:/x.txt");
				dumpVec(b, "D:/g.txt");
				system("pause");
				throw std::exception("linear solver failed");
			}
		}

		void CalcCorr()
		{
			map_c2l_corr_.resize(imgWidth_*imgHeight_);
			std::fill(map_c2l_corr_.begin(), map_c2l_corr_.end(), -1);

			for (int y = 0; y < imgHeight_; y++)
			{
				for (int x = 0; x < imgWidth_; x++)
				{
					float3 pwarp = read_float3_from_4(vmap_warp_[y*imgWidth_ + x]);
					float3 nwarp = read_float3_from_4(nmap_warp_[y*imgWidth_ + x]);		
					
					if (isnan(nwarp.x))
						continue;

					float3 uvd = intr_.xyz2uvd(pwarp);
					int2 ukr = make_int2(uvd.x+0.5, uvd.y+0.5);

					// we use opengl coordinate, thus world.z should < 0
					if (ukr.x < 0 || ukr.y < 0 || ukr.x >= imgWidth_ || ukr.y >= imgHeight_ || pwarp.z >= 0)
						continue;

					float3 plive = read_float3_from_4(vmap_live_[ukr.y*imgWidth_ + ukr.x]);
					float3 nlive = read_float3_from_4(nmap_live_[ukr.y*imgWidth_ + ukr.x]);
					if (isnan(nlive.x))
						continue;

					float dist = norm(pwarp - plive);
					if (dist > param_.fusion_nonRigid_distThre || isnan(dist))
						continue;

					float sine = norm(cross(nwarp, nlive));
					if (sine >= param_.fusion_nonRigid_angleThreSin || isnan(sine))
						continue;

					map_c2l_corr_[y*imgWidth_ + x] = ukr.y*imgWidth_ + ukr.x;
				}// x
			}// y
		}

		real Optimize(Vec& xStart, int nMaxIter, bool showInfo = true)
		{
			//define jacobi structure
			DefineJacobiStructure(jac_, jact_);

			SpMat JacTJac;
			Vec fx(jac_.rows()), h(jac_.cols()), g(jac_.cols()), fx1(jac_.rows());

#ifndef ENABLE_DIALG_HESSIAN
			CalcHessian(JacTJac);
			Eigen::SparseLU<SpMat> solver;
			solver.analyzePattern(JacTJac);
#endif

			//Gauss-Newton Optimization
			for (int iter = 0; iter<nMaxIter; iter++)
			{
				CalcJacobiFunc(xStart, jac_, jact_);	//J

				CalcHessian(JacTJac);
				CalcEnergyFunc(xStart, fx);	//f

				//solve: J'J h =  - J' f(x)
				g = jact_ * (-fx);
				checkNanAndInf(JacTJac);

#ifndef ENABLE_DIALG_HESSIAN
				solver.factorize(JacTJac);
				h = solver.solve(g);
#else
				blockSolve(JacTJac, h, g);
#endif
				checkNanAndInf(h, fx, g, JacTJac);
				checkLinearSolver(JacTJac, h, g);

				real old_energy = evaluateTotalEnergy(xStart);
				real new_energy = 0;
				real h_0 = h[0];
				real normv = xStart.norm();
				real alpha = 1;
				if (param_.fusion_GaussNewton_fixedStep <= 0.f)
				{
					for (; alpha > 1e-15; alpha *= 0.5)
					{
						Vec x = xStart + h;
						new_energy = evaluateTotalEnergy(x);
						if (new_energy > old_energy)
						{
							h = h * 0.5;
						}
						if (new_energy < old_energy)
						{
							xStart = x;
							break;
						}
						else
							new_energy = 0;
					}
				}
				else
				{
					alpha = param_.fusion_GaussNewton_fixedStep;
					h *= alpha;
					xStart += h;
				}

				real normh = h.norm();
				if (showInfo)
					printf("Gauss-Newton %f: %d %f: %f->%f [0]=%f, %f\n", alpha, iter, normh / (1e-6+normv),
					old_energy, new_energy, xStart[0], h_0);

				if (normh < (normv + real(1e-6)) * real(1e-6))
					break;
			}
			return evaluateTotalEnergy(xStart);
		}

		// should be called each time after CalcHessian()
		void blockSolve(const SpMat& H, Vec& x, const Vec& rhs)
		{
			// level0 term LLt
			std::vector<Eigen::Triplet<real>> L0invsys, L0L0tinvsys;
			L0invsys.reserve(m_diagBlocks.size() * 36);
			L0L0tinvsys.reserve(m_diagBlocks.size() * 36);
			std::vector<Mat6> L0L0tinvBlocks(m_diagBlocks.size());
			for (size_t i = 0; i < m_diagBlocks.size(); i++)
			{
				Mat6 L = m_diagBlocks[i].llt().matrixL();
				Mat6 Linv = L.triangularView<Eigen::Lower>().solve(Mat6::Identity());
				L0L0tinvBlocks[i] = Linv.transpose()*Linv;
				int rc = i * 6;
				for (int y = 0; y < 6; y++)
				for (int x = 0; x <= y; x++)
					L0invsys.push_back(Eigen::Triplet<real>(rc + y, rc + x, Linv(y, x)));
				for (int y = 0; y < 6; y++)
				for (int x = 0; x < 6; x++)
					L0L0tinvsys.push_back(Eigen::Triplet<real>(rc + y, rc + x, L0L0tinvBlocks[i](y, x)));
			}
			SpMat L0inv, L0L0tinv;
			L0inv.resize(m_diagBlocks.size() * 6, m_diagBlocks.size() * 6);
			L0inv.setFromTriplets(L0invsys.begin(), L0invsys.end());
			L0L0tinv.resize(m_diagBlocks.size() * 6, m_diagBlocks.size() * 6);
			L0L0tinv.setFromTriplets(L0L0tinvsys.begin(), L0L0tinvsys.end());

			int res = H.rows() - nodesInLevel0_ * 6;
			// 
			if (res > 0)
			{
				SpMat BtD = H.bottomRows(H.rows() - nodesInLevel0_ * 6);
				SpMat::ColsBlockXpr Bt = BtD.leftCols(nodesInLevel0_ * 6);
				SpMat::ColsBlockXpr D = BtD.rightCols(BtD.cols() - Bt.cols());
				Mat Q = D - (Bt * L0L0tinv) * Bt.transpose();
				Mat L1 = Q.llt().matrixL();

				// solve for Lu=b
				Vec u;
				u.resize(H.rows());
				u.topRows(L0inv.rows()) = L0inv * rhs.topRows(L0inv.cols());
				u.bottomRows(L1.rows()) = L1.triangularView<Eigen::Lower>().solve(
					rhs.bottomRows(L1.rows()) - Bt * (L0L0tinv * rhs.topRows(L0L0tinv.cols())).eval());

				// solve for Lt x = u
				x.resize(H.cols());
				x.topRows(L0inv.cols()) = L0inv.transpose() * u.topRows(L0inv.rows())
					- L0L0tinv.transpose() * (Bt.transpose() *
					L1.triangularView<Eigen::Lower>().transpose().solve(u.bottomRows(L1.cols()))).eval();
				x.bottomRows(L1.cols()) = L1.triangularView<Eigen::Lower>().transpose().solve(u.bottomRows(L1.cols()));
#ifdef ENABLE_DEBUG_DUMP_MATRIX_EACH_ITER
				{
					static int a = 0;
					{
						std::string name = ("D:/tmp/cpu_Q_" + std::to_string(a) + ".txt").c_str();
						dumpMat(Q, name.c_str());
					}
					{
						std::string name = ("D:/tmp/cpu_u_" + std::to_string(a) + ".txt").c_str();
						dumpVec(u, name.c_str());
					}
					{
						std::string name = ("D:/tmp/cpu_h_" + std::to_string(a) + ".txt").c_str();
						dumpVec(x, name.c_str());
					}
					{
						std::string name = ("D:/tmp/cpu_B_" + std::to_string(a) + ".txt").c_str();
						dumpSparseMatrix(Bt.transpose(), name.c_str());
					}

					a++;
				}
#endif
			}
			else
			{
				x = L0L0tinv * rhs;
#ifdef ENABLE_DEBUG_DUMP_MATRIX_EACH_ITER
				{
					static int a = 0;
					{
						std::string name = ("D:/tmp/cpu_h_" + std::to_string(a) + ".txt").c_str();
						dumpVec(x, name.c_str());
					}

					a++;
				}
#endif
			}
		}

		inline real data_term_energy(real f)const
		{
#ifndef USE_ROBUST_TUKEY_PENALTY
			// the penalty function is ||f||^2/2, thus a single term is the gradient f.
			return 0.5f*f*f;
#else
			// the robust Tukey penelty gradient
			real c = param_.fusion_psi_data;
			if (abs(f) <= c)
				return c*c / 6.f *(1 - pow(1 - ldp::sqr(f / param_.fusion_psi_data), 3));
			else
				return c*c / 6.f;
#endif
		}

		inline real data_term_penalty(real f)const
		{
#ifndef USE_ROBUST_TUKEY_PENALTY
			// the penalty function is ||f||^2/2, thus a single term is the gradient f.
			return f;
#else
			// the robust Tukey penelty gradient
			if (abs(f) <= param_.fusion_psi_data)
				return f * ldp::sqr(1 - ldp::sqr(f / param_.fusion_psi_data));
			else
				return 0;
#endif
		}

		inline real reg_term_energy(Tbx::Vec3 f)const
		{
#ifndef USE_ROBUST_HUBER_PENALTY
			return f.dot(f)*0.5f;
#else
			// the robust Huber penelty gradient
			real s = 0;
			real norm = f.norm();
			if (norm < param_.fusion_psi_reg)
				s = norm * norm * 0.5f;
			else
				s = param_.fusion_psi_reg*(norm - param_.fusion_psi_reg*0.5f);
			return s;
#endif
		}
		
		inline Tbx::Vec3 reg_term_penalty(Tbx::Vec3 f)const
		{
#ifndef USE_ROBUST_HUBER_PENALTY
			return f;
#else
			// the robust Huber penelty gradient
			Tbx::Vec3 df;
			real norm = f.norm();
			if (norm < param_.fusion_psi_reg)
				df = f;
			else for(int k = 0; k < 3; k++)
				df[k] = f[k] * param_.fusion_psi_reg / norm;
			return df;
#endif
		}
		
		inline Tbx::Transfo outer_product(Tbx::Vec3 n, Tbx::Point3 v)const
		{
			return Tbx::Transfo(
				n.x*v.x, n.x*v.y, n.x*v.z, n.x,
				n.y*v.x, n.y*v.y, n.y*v.z, n.y,
				n.z*v.x, n.z*v.y, n.z*v.z, n.z,
				0, 0, 0, 0
				);
		}

		inline Tbx::Dual_quat_cu p_qk_p_alpha_func(Tbx::Dual_quat_cu dq, int i)
		{
			Tbx::Vec3 t, r;
			float b, c, n;
			Tbx::Quat_cu q0(0,0,0,0), q1=dq.get_non_dual_part();
			switch (i)
			{
			case 0:
				dq.to_twist(r, t);
				n = r.norm();
				if (n > Tbx::Dual_quat_cu::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);
					q0.coeff0 = -r.x * b;
					q0.coeff1 = b + r.x*r.x*c;
					q0.coeff2 = r.x*r.y*c;
					q0.coeff3 = r.x*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 1;
					q0.coeff2 = 0;
					q0.coeff3 = 0;
				}

				q1.coeff0 = (t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = (t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = (t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 1:
				dq.to_twist(r, t);
				n = r.norm(); 
				if (n > Tbx::Dual_quat_cu::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);
					q0.coeff0 = -r.y * b;
					q0.coeff1 = r.y*r.x*c;
					q0.coeff2 = b + r.y*r.y*c;
					q0.coeff3 = r.y*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 0;
					q0.coeff2 = 1;
					q0.coeff3 = 0;
				}

				q1.coeff0 = (t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = (t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = (t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 2:
				dq.to_twist(r, t);
				n = r.norm();
				if (n > Tbx::Dual_quat_cu::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);

					q0.coeff0 = -r.z * b;
					q0.coeff1 = r.z*r.x*c;
					q0.coeff2 = r.z*r.y*c;
					q0.coeff3 = b + r.z*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 0;
					q0.coeff2 = 0;
					q0.coeff3 = 1;
				}

				q1.coeff0 = ( t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = ( t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = ( t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 3:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff1, q1.coeff0, -q1.coeff3, q1.coeff2))*0.5;
			case 4:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff2, q1.coeff3, q1.coeff0, -q1.coeff1))*0.5;
			case 5:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff3, -q1.coeff2, q1.coeff1, q1.coeff0))*0.5;
			default:
				throw std::exception("p_qk_p_alpha_func: out of range");
				break;
			}
		}

		inline float trace_AtB(Tbx::Transfo A, Tbx::Transfo B)
		{
			float sum = 0;
			for (int i = 0; i < 16; i++)
				sum += A[i] * B[i];
			return sum;
		}

		inline Tbx::Transfo p_SE3_p_dq_func(Tbx::Dual_quat_cu dq, int i)
		{
			Tbx::Quat_cu q0 = dq.get_non_dual_part();
			Tbx::Quat_cu q1 = dq.get_dual_part();
			real x0 = q0.i(), y0 = q0.j(), z0 = q0.k(), w0 = q0.w();
			real x1 = q1.i(), y1 = q1.j(), z1 = q1.k(), w1 = q1.w();
			switch (i)
			{
			case 0:
				return Tbx::Transfo(
					0, -z0, y0, x1,
					z0, 0, -x0, y1,
					-y0, x0, 0, z1,
					0, 0, 0, 0) * 2;
			case 1:
				return Tbx::Transfo(
					0, y0, z0, -w1,
					y0, -2*x0, -w0, -z1,
					z0, w0, -2*x0, y1,
					0, 0, 0, 0) * 2;
			case 2:
				return Tbx::Transfo(
					-2*y0, x0, w0, z1,
					x0, 0, z0, -w1,
					-w0, z0, -2*y0, -x1,
					0, 0, 0, 0) * 2;
			case 3:
				return Tbx::Transfo(
					-2*z0, -w0, x0, -y1,
					w0, -2*z0, y0, x1,
					x0, y0, 0, -w1,
					0, 0, 0, 0) * 2;
			case 4:
				return Tbx::Transfo(
					0, 0, 0, -x0,
					0, 0, 0, -y0,
					0, 0, 0, -z0,
					0, 0, 0, 0) * 2;
			case 5:
				return Tbx::Transfo(
					0, 0, 0, w0,
					0, 0, 0, z0,
					0, 0, 0, -y0,
					0, 0, 0, 0) * 2;
			case 6:
				return Tbx::Transfo(
					0, 0, 0, -z0,
					0, 0, 0, w0,
					0, 0, 0, x0,
					0, 0, 0, 0) * 2;
			case 7:
				return Tbx::Transfo(
					0, 0, 0, y0,
					0, 0, 0, -x0,
					0, 0, 0, w0,
					0, 0, 0, 0) * 2;
			default:
				throw std::exception("index out of range");
				return Tbx::Transfo::identity();
			}
		}

		inline Tbx::Transfo p_SE3_p_alpha_func(Tbx::Dual_quat_cu dq, int i)
		{
			Tbx::Transfo T = Tbx::Transfo::empty();
			Tbx::Dual_quat_cu p_dq_p_alphai = p_qk_p_alpha_func(dq, i);
			for (int j = 0; j < 8; j++)
			{
				T = T + p_SE3_p_dq_func(dq, j)*p_dq_p_alphai[j];
			}
			return T;
		}

		void DefineJacobiStructure(SpMat& jac, SpMat& jact)
		{
			enum {VarPerNode = 6};
			const int nNodes = x_.size() / 6;
			const int nPixel = imgHeight_ * imgWidth_;

			m_cooSys.clear();
			coo_pos_of_pixel_.resize(nPixel);
			row_index_of_pixel_.resize(nPixel);
			pixel_index_of_row_.resize(nPixel);
			std::fill(row_index_of_pixel_.begin(), row_index_of_pixel_.end(), -1);
			std::fill(pixel_index_of_row_.begin(), pixel_index_of_row_.end(), -1);
			std::fill(coo_pos_of_pixel_.begin(), coo_pos_of_pixel_.end(), -1);

			// data term
			int nRow = 0;
			for (int iPixel = 0; iPixel < nPixel; iPixel++)
			{
				int corrPixel = map_c2l_corr_[iPixel];
				if (corrPixel < 0)
					continue;

				coo_pos_of_pixel_[iPixel] = m_cooSys.size();

				bool valid = false;
				KnnIdx knn = vmapKnn_[iPixel];
				for (int k = 0; k < KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						valid = true;
						for (int t = 0; t < VarPerNode; t++)
						{
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, 
							knnNodeId * VarPerNode + t, 0));
						}
					}
				}
				if (valid)
				{
					row_index_of_pixel_[iPixel] = nRow;
					pixel_index_of_row_[nRow] = iPixel;
					nRow++;
				}
			}// end for iPixel
			nPixel_rows_ = nRow;
			nPixel_cooPos_ = m_cooSys.size();

			// reg term
			for (int iNode = 0; iNode < nNodes; iNode++)
			{
				KnnIdx knn = nodesKnn_[iNode];
				for (int k = 0; k < KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						for (int ixyz = 0; ixyz < 3; ixyz++, nRow++)
						for (int t = 0; t < VarPerNode; t++)
						{
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, iNode * VarPerNode + t, 0));
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, knnNodeId * VarPerNode + t, 0));
						}
#ifdef ENABLE_DOUBLE_EDGE
						for (int ixyz = 0; ixyz < 3; ixyz++, nRow++)
						for (int t = 0; t < VarPerNode; t++)
						{
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, iNode * VarPerNode + t, 0));
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, knnNodeId * VarPerNode + t, 0));
						}
#endif
					}
				}
			}// end for iNode

			jac.resize(nRow, nNodes * VarPerNode);
			if (m_cooSys.size())
				jac.setFromTriplets(m_cooSys.begin(), m_cooSys.end());
			jact = jac.transpose();
		}

		void CalcEnergyFunc(const Vec& x, Vec& f)const
		{
			int nNodes = x.size() / 6;

			// data term
#pragma omp parallel for
			for (int iPixel = 0; iPixel < map_c2l_corr_.size(); iPixel++)
			{
				int nRow = row_index_of_pixel_[iPixel];
				if (nRow < 0)
					continue;

				int corrPixel = map_c2l_corr_[iPixel];

				Tbx::Point3 v(convert(read_float3_from_4(vmap_cano_[iPixel])));
				Tbx::Vec3 n = convert(read_float3_from_4(nmap_cano_[iPixel]));
				Tbx::Point3 vl(convert(read_float3_from_4(vmap_live_[corrPixel])));

				KnnIdx knn = vmapKnn_[iPixel];
				Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				Tbx::Dual_quat_cu dq0;
				for (int k = 0; k < KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						Tbx::Dual_quat_cu dq;
						Tbx::Vec3 r(x[knnNodeId * 6], x[knnNodeId * 6 + 1], x[knnNodeId * 6 + 2]);
						Tbx::Vec3 t(x[knnNodeId * 6 + 3], x[knnNodeId * 6 + 4], x[knnNodeId * 6 + 5]);
						Tbx::Point3 nodesV(convert(read_float3_from_4(nodesVw_[knnNodeId])));
						float invNodesW = nodesVw_[knnNodeId].w; 
						dq.from_twist(r, t);
						// note: we store inv radius as vw.w, thus using * instead of / here
						float wk = exp(-(v - nodesV).dot(v - nodesV)*(2 * invNodesW * invNodesW));
						if (k == 0)
							dq0 = dq;
						if (dq0.get_non_dual_part().dot(dq.get_non_dual_part()) < 0)
							wk = -wk;
						dq_blend = dq_blend + dq*wk;
					}
				}

				if (dq_blend.get_non_dual_part().norm() < Tbx::Dual_quat_cu::epsilon())
					f[nRow] = 0;
				else
				{
					dq_blend.normalize();
					v = Tlw_*(dq_blend.transform(v));
					n = Tlw_*(dq_blend.rotate(n));
					f[nRow] = data_term_penalty(n.dot(v - vl));
				}
			}// end for iPixel

			// reg term
			int nRow = nPixel_rows_;
			const float lambda = param_.fusion_lambda;
			for (int iNode = 0; iNode < nNodes; iNode++)
			{
				KnnIdx knn = nodesKnn_[iNode];
				Tbx::Dual_quat_cu dqi;
				Tbx::Vec3 vi = convert(read_float3_from_4(nodesVw_[iNode]));
				Tbx::Vec3 ri(x[iNode * 6], x[iNode * 6 + 1], x[iNode * 6 + 2]);
				Tbx::Vec3 ti(x[iNode * 6 + 3], x[iNode * 6 + 4], x[iNode * 6 + 5]);
				dqi.from_twist(ri, ti);

				for (int k = 0; k < KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						Tbx::Vec3 rj(x[knnNodeId * 6], x[knnNodeId * 6 + 1], x[knnNodeId * 6 + 2]);
						Tbx::Vec3 tj(x[knnNodeId * 6 + 3], x[knnNodeId * 6 + 4], x[knnNodeId * 6 + 5]);
						real alpha_ij = max(1 / nodesVw_[iNode].w, 1 / nodesVw_[knnNodeId].w);

						Tbx::Dual_quat_cu dqj;
						dqj.from_twist(rj, tj);
						Tbx::Vec3 vj = convert(read_float3_from_4(nodesVw_[knnNodeId]));
						Tbx::Vec3 val = dqi.transform(Tbx::Point3(vi)) - dqj.transform(Tbx::Point3(vi));
						val = reg_term_penalty(val);

						real ww = sqrt(lambda * alpha_ij);
						f[nRow++] = val.x * ww;
						f[nRow++] = val.y * ww;
						f[nRow++] = val.z * ww;
#ifdef ENABLE_DOUBLE_EDGE
						Tbx::Vec3 val1 = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
						val1 = reg_term_penalty(val1);

						f[nRow++] = val1.x * ww;
						f[nRow++] = val1.y * ww;
						f[nRow++] = val1.z * ww;
#endif
					}
				}
			}// end for iNode
#ifdef ENABLE_DEBUG_DUMP_MATRIX_EACH_ITER
			{
				static int a = 0;
				{
					std::string name = ("D:/tmp/cpu_f_" + std::to_string(a) + ".txt").c_str();
					dumpVec(f, name.c_str());
				}
				{
					std::string name = ("D:/tmp/cpu_gd_" + std::to_string(a) + ".txt").c_str();
					dumpVec(jact_.leftCols(nPixel_rows_) * f.topRows(nPixel_rows_), name.c_str());
				}
				{
					std::string name = ("D:/tmp/cpu_g_" + std::to_string(a) + ".txt").c_str();
					dumpVec(-jact_* f, name.c_str());
				}

				a++;
			}
#endif
		}

		real evaluateTotalEnergy(const Vec& x)const
		{
			int nNodes = x.size() / 6;

			double total_energy = 0;

			// data term
//#pragma omp parallel for
			for (int iPixel = 0; iPixel < map_c2l_corr_.size(); iPixel++)
			{
				int nRow = row_index_of_pixel_[iPixel];
				if (nRow < 0)
					continue;

				int corrPixel = map_c2l_corr_[iPixel];

				Tbx::Point3 v(convert(read_float3_from_4(vmap_cano_[iPixel])));
				Tbx::Vec3 n = convert(read_float3_from_4(nmap_cano_[iPixel]));
				Tbx::Point3 vl(convert(read_float3_from_4(vmap_live_[corrPixel])));

				KnnIdx knn = vmapKnn_[iPixel];
				Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				Tbx::Dual_quat_cu dq0;
				for (int k = 0; k < KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						Tbx::Dual_quat_cu dq;
						Tbx::Vec3 r(x[knnNodeId * 6], x[knnNodeId * 6 + 1], x[knnNodeId * 6 + 2]);
						Tbx::Vec3 t(x[knnNodeId * 6 + 3], x[knnNodeId * 6 + 4], x[knnNodeId * 6 + 5]);
						Tbx::Point3 nodesV(convert(read_float3_from_4(nodesVw_[knnNodeId])));
						float invNodesW = nodesVw_[knnNodeId].w;
						dq.from_twist(r, t);
						// note: we store inv radius as vw.w, thus using * instead of / here
						float wk = exp(-(v - nodesV).dot(v - nodesV)*(2 * invNodesW * invNodesW));
#ifdef ENABLE_ANTIPODALITY
						if (k == 0)
							dq0 = dq;
						if (dq0.get_non_dual_part().dot(dq.get_non_dual_part()) < 0)
							wk = -wk;
#endif
						dq_blend = dq_blend + dq*wk;
					}
				}

				if (dq_blend.get_non_dual_part().norm() < Tbx::Dual_quat_cu::epsilon())
					continue;

				dq_blend.normalize();
				v = Tlw_*(dq_blend.transform(v));
				n = Tlw_*(dq_blend.rotate(n));
				total_energy += data_term_energy(n.dot(v - vl));
			}// end for iPixel

			// reg term
			int nRow = nPixel_rows_;
			const float lambda = param_.fusion_lambda;
			for (int iNode = 0; iNode < nNodes; iNode++)
			{
				KnnIdx knn = nodesKnn_[iNode];
				Tbx::Dual_quat_cu dqi;
				Tbx::Vec3 vi = convert(read_float3_from_4(nodesVw_[iNode]));
				Tbx::Vec3 ri(x[iNode * 6], x[iNode * 6 + 1], x[iNode * 6 + 2]);
				Tbx::Vec3 ti(x[iNode * 6 + 3], x[iNode * 6 + 4], x[iNode * 6 + 5]);
				dqi.from_twist(ri, ti);

				for (int k = 0; k < KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						Tbx::Vec3 rj(x[knnNodeId * 6], x[knnNodeId * 6 + 1], x[knnNodeId * 6 + 2]);
						Tbx::Vec3 tj(x[knnNodeId * 6 + 3], x[knnNodeId * 6 + 4], x[knnNodeId * 6 + 5]);
						real alpha_ij = max(1 / nodesVw_[iNode].w, 1 / nodesVw_[knnNodeId].w);

						Tbx::Dual_quat_cu dqj;
						dqj.from_twist(rj, tj);
						Tbx::Vec3 vj = convert(read_float3_from_4(nodesVw_[knnNodeId]));
						Tbx::Vec3 val = dqi.transform(Tbx::Point3(vi)) - dqj.transform(Tbx::Point3(vi));
						total_energy += reg_term_energy(val) * lambda * alpha_ij;
#ifdef ENABLE_DOUBLE_EDGE
						Tbx::Vec3 val1 = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
						total_energy += reg_term_energy(val1) * lambda * alpha_ij;
#endif
					}
				}
			}// end for iNode

			return total_energy;
		}

		void CalcHessian(SpMat& H)
		{
#ifdef ENABLE_DIALG_HESSIAN
			// reg term, use full representation
			SpMat jacReg = jac_.bottomRows(jac_.rows() - nPixel_rows_);

			if (jacReg.rows()>0)
				H = jacReg.transpose()*jacReg;
			else
			{
				H = jact_ * jac_ * 0.f;
			}

			// data term, only compute diag block
			SpMat jacData = jac_.topRows(nPixel_rows_).eval();
			const int nNodes = jacData.cols() / 6;

			m_diagBlocks.resize(nodesInLevel0_);
			for (int iNode = 0; iNode < nodesInLevel0_; iNode++)
			{
				SpMat jacDataBlock = jacData.middleCols(iNode * 6, 6);
				m_diagBlocks[iNode] = (jacDataBlock.transpose() * jacDataBlock).eval().toDense();
			}// end for ix

			// traverse H to adding blocks on
			for (int row = 0; row < H.outerSize(); row ++)
			{
				int rs = H.outerIndexPtr()[row];
				int re = H.outerIndexPtr()[row + 1];
				int iNode = row / 6;

				if (iNode >= nodesInLevel0_)
					continue;

				int blockRowShift = row - iNode*6;
				Mat6& block = m_diagBlocks[iNode];
				for (int c = rs; c < re; c++)
				{
					int col = H.innerIndexPtr()[c];
					int blockColShift = col - iNode*6;
					if (blockColShift < 6 && blockColShift >= 0)
					{
						H.valuePtr()[c] += block(blockRowShift, blockColShift);
						block(blockRowShift, blockColShift) = H.valuePtr()[c]; // keep this value for later usage
					}
				}
			}
#ifdef ENABLE_DEBUG_DUMP_MATRIX_EACH_ITER
			{
				static int a = 0;
				{
					std::string name = ("D:/tmp/cpu_Jr_" + std::to_string(a) + ".txt").c_str();
					dumpSparseMatrix(jacReg, name.c_str());
				}
				{
					std::string name = ("D:/tmp/cpu_Jrt_" + std::to_string(a) + ".txt").c_str();
					dumpSparseMatrix(jacReg.transpose(), name.c_str());
				}
				{
					std::string name = ("D:/tmp/cpu_Hd_"+std::to_string(a)+".txt").c_str();
					FILE*pFile = fopen(name.c_str(), "w");
					for(int i=0; i<m_diagBlocks.size(); i++)
					{
						for (int y = 0; y < 6; y++)
						for (int x = 0; x < 6; x++)
							fprintf(pFile, "%ef ", m_diagBlocks[i](y, x));
						fprintf(pFile, "\n");
					}
					fclose(pFile);
				}
				a++;
			}
#endif

			// ldp debug
#ifdef ENABLE_DEBUG_DUMP_MATRIX_EACH_ITER
			{
				static int a = 0;
				dumpSparseMatrix(jacReg.transpose()*jacReg, ("D:/tmp/regH"+std::to_string(a)+".txt").c_str());
				dumpSparseMatrix(H, ("D:/tmp/H" + std::to_string(a) + ".txt").c_str());
				a++;
			}
#endif
#else
			H = jact_ * jac_;
#endif
			// ldp test: add small reg value to prevent singular
			Eigen::Diagonal<SpMat> diag(H);
			diag += Vec(diag.size()).setConstant(param_.fusion_GaussNewton_diag_regTerm);
		}

		int FindColIndices(const SpMat& A, int cid, int* vidx, int* ridx)
		{
			int ns = A.outerIndexPtr()[cid], ne = A.outerIndexPtr()[cid + 1];
			int k = 0;
			for (int i = ns; i<ne; i++, k++)
			{
				ridx[k] = A.innerIndexPtr()[i];
				vidx[k] = i;
			}
			return k;
		}

		void CalcJacobiFunc(const Vec& pTest, SpMat& jac, SpMat& jact)
		{
			//CalcJacobiFuncNumeric(pTest, jac, jact);
			CalcJacobiFuncAnalytic(pTest, jac, jact);
#if 0
			// debug
			CalcJacobiFuncNumeric(pTest, jac, jact);
			dumpSparseMatrix(jac, "D:/num.txt");
			CalcJacobiFuncAnalytic(pTest, jac, jact);
			dumpSparseMatrix(jac, "D:/ana.txt");
			system("pause");
			// end debug
#endif
		}

		void CalcJacobiFuncAnalytic(const Vec& pTest, SpMat& jac, SpMat& jact)
		{
			enum { VarPerNode = 6 };
			const int nNodes = x_.size() / 6;
			const int nPixel = imgHeight_ * imgWidth_;

			for (size_t i = 0; i < m_cooSys.size(); i++)
				m_cooSys[i] = Eigen::Triplet<real>(m_cooSys[i].row(), m_cooSys[i].col(), 0);

			// data term    ========================================================
#pragma omp parallel for
			for (int iPixel = 0; iPixel < nPixel; iPixel++)
			{
				int nRow = row_index_of_pixel_[iPixel];
				if (nRow < 0)
					continue;

				int cooPos = coo_pos_of_pixel_[iPixel];

				int corrPixel = map_c2l_corr_[iPixel];
				Tbx::Point3 v(convert(read_float3_from_4(vmap_cano_[iPixel])));
				Tbx::Vec3 n = convert(read_float3_from_4(nmap_cano_[iPixel]));
				Tbx::Point3 vl(convert(read_float3_from_4(vmap_live_[corrPixel])));

				KnnIdx knn = vmapKnn_[iPixel];
				Tbx::Dual_quat_cu dq(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				Tbx::Dual_quat_cu dqk[KnnK];
				real wk[KnnK];
				for (int knnK = 0; knnK < KnnK; knnK++)
				{
					int knnNodeId = knn_k(knn, knnK);
					if (knnNodeId < nNodes)
					{
						Tbx::Vec3 r(pTest[knnNodeId * 6], pTest[knnNodeId * 6 + 1], pTest[knnNodeId * 6 + 2]);
						Tbx::Vec3 t(pTest[knnNodeId * 6 + 3], pTest[knnNodeId * 6 + 4], pTest[knnNodeId * 6 + 5]);
						Tbx::Point3 nodesV(convert(read_float3_from_4(nodesVw_[knnNodeId])));
						float invNodesW = nodesVw_[knnNodeId].w;
						dqk[knnK].from_twist(r, t);
						// note: we store inv radius as vw.w, thus using * instead of / here
						wk[knnK] = exp(-(v - nodesV).dot(v - nodesV)*(2 * invNodesW * invNodesW));
						if (dqk[0].get_non_dual_part().dot(dqk[knnK].get_non_dual_part()) < 0)
							wk[knnK] = -wk[knnK];
						dq = dq + dqk[knnK] * wk[knnK];
					}// end if (knnNodeId < nNodes)
				}// ebd fir knnK

				if (dq.get_non_dual_part().norm() < std::numeric_limits<float>::epsilon())
					continue;

				Tbx::Dual_quat_cu dq_bar = dq;
				real norm_dq_bar = dq_bar.get_non_dual_part().norm();
				real norm_dq_bar3 = norm_dq_bar*norm_dq_bar*norm_dq_bar;
				dq.normalize();
				Tbx::Transfo T = Tlw_*dq.to_transformation();

				// paitial_f_partial_T
				Tbx::Transfo nvt = outer_product(n, v);
				Tbx::Transfo vlnt = outer_product(n, vl).transpose();
				Tbx::Transfo p_f_p_T = T*(nvt + nvt.transpose()) - vlnt;

				for (int knnK = 0; knnK < KnnK; knnK++)
				{
					int knnNodeId = knn_k(knn, knnK);
					if (knnNodeId < nNodes)
					{
						real p_f_p_alpha[6];
						// partial_T_partial_alphak
						for (int ialpha = 0; ialpha < 6; ialpha++)
						{
							Tbx::Transfo p_T_p_alphak = Tbx::Transfo::empty();
							Tbx::Dual_quat_cu p_qk_p_alpha = p_qk_p_alpha_func(dqk[knnK], ialpha);
							for (int idq = 0; idq < 8; idq++)
							{
								// partial_SE3_partial_dqi
								Tbx::Transfo p_SE3_p_dqi = p_SE3_p_dq_func(dq, idq);
								real dq_bar_i = dq_bar[idq];

								// partial_dqi_partial_alphak
								real p_dqi_p_alphak = 0;
								for (int j = 0; j < 8; j++)
								{
									// partial_dqi_partial_qkj
									real dq_bar_j = dq_bar[j];
									real p_dqi_p_qkj = wk[knnK] / norm_dq_bar * (idq == j);
									if (j < 4)
										p_dqi_p_qkj -= wk[knnK] / norm_dq_bar3*dq_bar_i*dq_bar_j;

									// partial_qkj_partial_alphak
									real p_qkj_p_alphak = p_qk_p_alpha[j];

									p_dqi_p_alphak += p_dqi_p_qkj * p_qkj_p_alphak;
								}// end for j

								p_T_p_alphak += p_SE3_p_dqi * p_dqi_p_alphak;
							}// end for idq
							p_T_p_alphak = Tlw_ * p_T_p_alphak;

							p_f_p_alpha[ialpha] = trace_AtB(p_f_p_T, p_T_p_alphak);

							// debug, check nan
							if (isnan(p_f_p_alpha[ialpha]) || isinf(p_f_p_alpha[ialpha])
								)
							{
								printf("warning: nan/inf found: %d %d,%d = %f\n", 
									nRow, knnNodeId, ialpha, p_f_p_alpha[ialpha]);
								printf("v: %f %f %f\n", v.x, v.y, v.z);
								printf("n: %f %f %f\n", n.x, n.y, n.z);
								printf("vl: %f %f %f\n", vl.x, vl.y, vl.z);
								printf("dq_bar: %f %f %f %f\n", dq_bar.get_non_dual_part().w(),
									dq_bar.get_non_dual_part().i(), dq_bar.get_non_dual_part().j(), 
									dq_bar.get_non_dual_part().k());
								printf("w: %f %f %f %f\n", wk[0], wk[1], wk[2], wk[3]);
								printf("p_f_p_T:\n");
								p_f_p_T.print();
								printf("p_T_p_alphak:\n");
								p_T_p_alphak.print();
								printf("p_qk_p_alpha:\n");
								p_qk_p_alpha.to_transformation().print();
								system("pause");
							}

							// write to jacobi
							m_cooSys.at(cooPos++) = Eigen::Triplet<real>(nRow,
								knnNodeId * VarPerNode + ialpha, p_f_p_alpha[ialpha]);
						}// end for ialpha

					}// end if knnNodeId < nNodes
				}// end for knnK
			}// end for iPixel
			
			// Reg term    ======================================================================
			int nRow = nPixel_rows_;
			int cooPos = nPixel_cooPos_;
			const float lambda = param_.fusion_lambda;
			for (int iNode = 0; iNode < nNodes; iNode++)
			{
				KnnIdx knn = nodesKnn_[iNode];
				Tbx::Dual_quat_cu dqi;
				Tbx::Point3 vi(convert(read_float3_from_4(nodesVw_[iNode])));
				Tbx::Vec3 ri(pTest[iNode * 6], pTest[iNode * 6 + 1], pTest[iNode * 6 + 2]);
				Tbx::Vec3 ti(pTest[iNode * 6 + 3], pTest[iNode * 6 + 4], pTest[iNode * 6 + 5]);
				dqi.from_twist(ri, ti);

				for (int knnK = 0; knnK < KnnK; knnK++)
				{
					int knnNodeId = knn_k(knn, knnK);
					if (knnNodeId < nNodes)
					{
						Tbx::Vec3 rj(pTest[knnNodeId * 6], pTest[knnNodeId * 6 + 1], pTest[knnNodeId * 6 + 2]);
						Tbx::Vec3 tj(pTest[knnNodeId * 6 + 3], pTest[knnNodeId * 6 + 4], pTest[knnNodeId * 6 + 5]);
						real alpha_ij = max(1 / nodesVw_[iNode].w, 1 / nodesVw_[knnNodeId].w);
						Tbx::Dual_quat_cu dqj;
						dqj.from_twist(rj, tj);
						Tbx::Point3 vj(convert(read_float3_from_4(nodesVw_[knnNodeId])));

						for (int ialpha = 0; ialpha < VarPerNode; ialpha++)
						{
							Tbx::Transfo p_Ti_p_alpha = p_SE3_p_alpha_func(dqi, ialpha);
							Tbx::Transfo p_Tj_p_alpha = p_SE3_p_alpha_func(dqj, ialpha);

							// partial_h_partial_alpha
							Tbx::Vec3 p_h_p_alphai_j, p_h_p_alphaj_j;
							Tbx::Vec3 p_h_p_alphai_i, p_h_p_alphaj_i;
							for (int ixyz = 0; ixyz < 3; ixyz++)
							{
								p_h_p_alphai_j[ixyz] = p_Ti_p_alpha(ixyz, 0) * vj.x + p_Ti_p_alpha(ixyz, 1) * vj.y
									+ p_Ti_p_alpha(ixyz, 2) * vj.z + p_Ti_p_alpha(ixyz, 3);
								p_h_p_alphaj_j[ixyz] = -(p_Tj_p_alpha(ixyz, 0) * vj.x + p_Tj_p_alpha(ixyz, 1) * vj.y
									+ p_Tj_p_alpha(ixyz, 2) * vj.z + p_Tj_p_alpha(ixyz, 3));
								p_h_p_alphai_i[ixyz] = p_Ti_p_alpha(ixyz, 0) * vi.x + p_Ti_p_alpha(ixyz, 1) * vi.y
									+ p_Ti_p_alpha(ixyz, 2) * vi.z + p_Ti_p_alpha(ixyz, 3);
								p_h_p_alphaj_i[ixyz] = -(p_Tj_p_alpha(ixyz, 0) * vi.x + p_Tj_p_alpha(ixyz, 1) * vi.y
									+ p_Tj_p_alpha(ixyz, 2) * vi.z + p_Tj_p_alpha(ixyz, 3));
							}

							// partial_psi_partial_alpha
							real ww = sqrt(lambda * alpha_ij);

							Tbx::Vec3 p_psi_p_alphai_j = p_h_p_alphai_j * ww;
							Tbx::Vec3 p_psi_p_alphaj_j = p_h_p_alphaj_j * ww;
							Tbx::Vec3 p_psi_p_alphai_i = p_h_p_alphai_i * ww;
							Tbx::Vec3 p_psi_p_alphaj_i = p_h_p_alphaj_i * ww;

							if (isnan(p_psi_p_alphai_j[0]) || isinf(p_psi_p_alphai_j[0]))
							{
								printf("warning: nan/inf in reg term jacobi: %d,%d=%f\n", 
									iNode, ialpha, p_psi_p_alphai_j[0]);
								printf("lambda=%f, alpha_ij=%f\n", lambda, alpha_ij);
								printf("p_T_p_alpha:\n");
								p_Ti_p_alpha.print();
								printf("vi: %f %f %f\n", vi.x, vi.y, vi.z);
								printf("vj: %f %f %f\n", vj.x, vj.y, vj.z);
							}

							for (int ixyz = 0; ixyz < 3; ixyz++)
							{
								m_cooSys[cooPos++] = Eigen::Triplet<real>(nRow + ixyz,
									iNode * VarPerNode + ialpha, 
									p_psi_p_alphai_i[ixyz]);
								m_cooSys[cooPos++] = Eigen::Triplet<real>(nRow + ixyz,
									knnNodeId * VarPerNode + ialpha,
									p_psi_p_alphaj_i[ixyz]);
							}
#ifdef ENABLE_DOUBLE_EDGE
							for (int ixyz = 0; ixyz < 3; ixyz++)
							{
								m_cooSys[cooPos++] = Eigen::Triplet<real>(nRow + 3 + ixyz,
									iNode * VarPerNode + ialpha,
									p_psi_p_alphai_j[ixyz]);
								m_cooSys[cooPos++] = Eigen::Triplet<real>(nRow + 3 + ixyz,
									knnNodeId * VarPerNode + ialpha,
									p_psi_p_alphaj_j[ixyz]);
							}
#endif
						}// end for ialpha
						nRow += 3;
#ifdef ENABLE_DOUBLE_EDGE
						nRow += 3;
#endif
					}// end if knnNode valid
				}// end for knnK
			}// end for iNode
	
			if (m_cooSys.size())
				jac.setFromTriplets(m_cooSys.begin(), m_cooSys.end());
			jact = jac.transpose();
		}

		void CalcJacobiFuncNumeric(const Vec& pTest, SpMat& jac, SpMat& jact)
		{
			Vec p = pTest;
			const int nobs = jac.rows();
			const int nvars = jac.cols();

			register int i, j, jj, k;
			register real d;
			int ii, m, *jcol, *varlist, *coldone, forw;
			int *vidxs, *ridxs;
			real *tmpd;
			real delta, delta_mul;

			/* retrieve problem-specific information passed in *dat */
			Vec hx(nobs), hxx(nobs);
			forw = 1;
			delta_mul = real(1e-3);
			delta = real(1e-5);

			CalcEnergyFunc(p, hx);//hx = f(p)

			jcol = (int *)malloc(nobs*sizeof(int)); /* keeps track of measurements influenced by the set of variables currently in "varlist" below */
			for (i = 0; i<nobs; ++i) jcol[i] = -1;

			vidxs = (int *)malloc(2 * nobs*sizeof(int));
			ridxs = vidxs + nobs;

			varlist = (int *)malloc(nvars*sizeof(int)); /* stores indices of J's columns which are computed with the same "func" call */
			coldone = (int *)malloc(nvars*sizeof(int)); /* keeps track of J's columns which have been already computed */
			memset(coldone, 0, nvars*sizeof(int)); /* initialize to zero */

			tmpd = (real *)malloc(nvars*sizeof(real));

			for (j = 0; j<nvars; ++j)
			{
				real scl;

				if (coldone[j]) continue; /* column j already computed */

				//for(i=0; i<nobs; ++i) jcol[i]=-1;
				k = FindColIndices(jac, j, vidxs, ridxs);
				for (i = 0; i<k; ++i) jcol[ridxs[i]] = j;
				varlist[0] = j; m = 1; coldone[j] = 1;

				for (jj = j + 1; jj<nvars; ++jj)
				{
					if (coldone[jj]) continue; /* column jj already computed */

					k = FindColIndices(jac, jj, vidxs, ridxs);
					for (i = 0; i<k; ++i)
					if (jcol[ridxs[i]] != -1) goto nextjj;

					if (k == 0) { coldone[jj] = 1; continue; } /* all zeros column, ignore */

					/* column jj does not clash with previously considered ones, mark it */
					for (i = 0; i<k; ++i) jcol[ridxs[i]] = jj;
					varlist[m++] = jj; coldone[jj] = 1;

				nextjj:
					continue;
				}

				for (k = 0; k<m; ++k)
				{
					/* determine d=max(SPLM_DELTA_SCALE*|p[varlist[k]]|, delta), see HZ */
					d = delta_mul*p[varlist[k]]; // force evaluation
					d = fabs(d);
					if (d<delta) d = delta;

					tmpd[varlist[k]] = d;
					p[varlist[k]] += d;
				}

				CalcEnergyFunc(p, hxx);// hxx=f(p+d)

				if (forw)
				{
					for (k = 0; k<m; ++k)
						p[varlist[k]] -= tmpd[varlist[k]]; /* restore */

					scl = 1.0;
				}
				else
				{ // central
					for (k = 0; k<m; ++k)
						p[varlist[k]] -= 2 * tmpd[varlist[k]];

					CalcEnergyFunc(p, hx);// hx=f(p-d)

					for (k = 0; k<m; ++k)
						p[varlist[k]] += tmpd[varlist[k]]; /* restore */

					scl = 0.5; // 1./2.
				}

				for (k = 0; k<m; ++k)
				{
					d = tmpd[varlist[k]];
					d = scl / d; /* invert so that divisions can be carried out faster as multiplications */

					jj = FindColIndices(jac, varlist[k], vidxs, ridxs);
					for (i = 0; i<jj; ++i)
					{
						ii = ridxs[i];
						jac.valuePtr()[vidxs[i]] = (hxx[ii] - hx[ii])*d;
						jcol[ii] = -1; /* restore */
					}
				}
			}//end for jj

			//calc 
			jact = jac.transpose();

			free(tmpd);
			free(coldone);
			free(varlist);
			free(vidxs);
			free(jcol);
		}

		static void dumpSparseMatrix(const SpMat& A, const char* filename)
		{
			FILE* pFile = fopen(filename, "w");
			if (!pFile)
				throw std::exception("dumpSparseMatrix: create file failed!");
			for (int r = 0; r < A.outerSize(); r++)
			{
				int rs = A.outerIndexPtr()[r];
				int re = A.outerIndexPtr()[r+1];
				for (int c = rs; c < re; c++)
					fprintf(pFile, "%d %d %ef\n", A.innerIndexPtr()[c], r, A.valuePtr()[c]);
			}
			fclose(pFile);
		}

		static void dumpVec(const Vec& A, const char* filename)
		{
			FILE* pFile = fopen(filename, "w");
			if (!pFile)
				throw std::exception("dumpVec: create file failed!");
			for (int r = 0; r < A.size(); r++)
			{
				fprintf(pFile, "%ef\n", A[r]);
			}
			fclose(pFile);
		}

		static void dumpMat(const Mat& A, const char* filename)
		{
			FILE* pFile = fopen(filename, "w");
			if (!pFile)
				throw std::exception("dumpMat: create file failed!");
			for (int r = 0; r < A.rows(); r++)
			{
				for (int c = 0; c < A.cols(); c++)
					fprintf(pFile, "%ef ", A(r, c));
				fprintf(pFile, "\n");
			}
			fclose(pFile);
		}
	};

	CpuGaussNewton::CpuGaussNewton()
	{
		m_egc = new EigenContainter();
	}

	CpuGaussNewton::~CpuGaussNewton()
	{
		delete m_egc;
	}

	static void checkGraph(const std::vector<KnnIdx>& knnGraph, int nNodes)
	{
		std::vector<ldp::Int2> edges;
		for (int i = 0; i < knnGraph.size(); i++)
		{
			KnnIdx knn = knnGraph[i];
			for (int k = 0; k < KnnK; k++)
			{
				int nb = knn_k(knn, k);
				if (nb < nNodes)
				{
					edges.push_back(ldp::Int2(i, nb));
					edges.push_back(ldp::Int2(nb, i));
				}
			}// k
		}// i
		std::sort(edges.begin(), edges.end());

		std::vector<int> edgeHeader(nNodes + 1, 0);
		for (int i = 1; i < edges.size(); i++)
		{
			if (edges[i][0] != edges[i - 1][0])
				edgeHeader[edges[i][0]] = i;
		}
		edgeHeader[nNodes] = edges.size();

		// find indepedent components
		std::set<int> verts;
		for (int i = 0; i < nNodes; i++)
			verts.insert(i);
		std::vector<std::vector<int>> components;

		while (!verts.empty())
		{
			components.push_back(std::vector<int>());
			std::vector<int>& cp = components.back();

			auto set_iter = verts.begin();
			std::queue<int> queue;
			queue.push(*set_iter);
			verts.erase(set_iter);

			while (!queue.empty())
			{
				const int v = queue.front();
				queue.pop();
				cp.push_back(v);

				for (int i = edgeHeader[v]; i < edgeHeader[v + 1]; i++)
				{
					const int v1 = edges[i][1];
					if (edges[i][0] != v)
						throw std::exception("edgeHeader error in find independent componnet debugging!");
					set_iter = verts.find(v1);
					if (set_iter != verts.end())
					{
						queue.push(v1);
						verts.erase(set_iter);
					}
				}// end for i
			}// end while
		}

		printf("Totally %d compoents:\n", components.size());
		for (int i = 0; i < components.size(); i++)
			printf("\t components %d, size %d\n", i, components[i].size());

		// dump
		FILE* pFile = fopen("D:/tmp/knnGraph_components.txt", "w");
		if (pFile)
		{
			for (int i = 0; i < components.size(); i++)
			{
				fprintf(pFile, "components %d, size %d\n", i, components[i].size());
				std::sort(components[i].begin(), components[i].end());
				for (int j = 0; j < components[i].size(); j++)
					fprintf(pFile, "%d\n", components[i][j]);
			}
			fclose(pFile);
		}
	}

	void CpuGaussNewton::init(WarpField* pWarpField, const MapArr& vmap_cano, const MapArr& nmap_cano,
		Param param, Intr intr)
	{
		m_pWarpField = pWarpField;
		m_egc->imgWidth_ = vmap_cano.cols();
		m_egc->imgHeight_ = vmap_cano.rows();
		m_egc->param_ = param;
		m_egc->intr_ = intr;
		m_egc->Tlw_ = pWarpField->get_rigidTransform();
		m_egc->nodesInLevel0_ = pWarpField->getNumNodesInLevel(0);

		// copy maps to cpu
		m_egc->vmap_cano_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		m_egc->nmap_cano_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		vmap_cano.download(m_egc->vmap_cano_.data(), m_egc->imgWidth_*sizeof(float4));
		nmap_cano.download(m_egc->nmap_cano_.data(), m_egc->imgWidth_*sizeof(float4));

		// extract knn map and copy to cpu
		m_pWarpField->extract_knn_for_vmap(vmap_cano, m_vmapKnn);
		m_egc->vmapKnn_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		m_vmapKnn.download(m_egc->vmapKnn_.data(), m_egc->imgWidth_*sizeof(KnnIdx));

		//extract nodes info and copy to cpu
		m_pWarpField->extract_nodes_info(m_nodesKnn, m_twist, m_vw);
		m_egc->nodesKnn_.resize(m_nodesKnn.size());
		m_nodesKnn.download(m_egc->nodesKnn_.data());

		m_egc->nodesVw_.resize(m_vw.size());
		m_vw.download(m_egc->nodesVw_.data());

		std::vector<float> tmpx;
		m_twist.download(tmpx);
		m_egc->x_.resize(m_twist.size());
		for (int i = 0; i < tmpx.size(); i++)
			m_egc->x_[i] = tmpx[i];

		// debug: checking the deformation graph
#ifdef ENABLE_DEBUG_DUMP_MATRIX_EACH_ITER
		checkGraph(m_egc->nodesKnn_, pWarpField->getNumAllNodes());
		{
			FILE* pFile = fopen("D:/tmp/knnGraph.txt", "w");
			if (pFile)
			{
				for (int i = 0; i < m_egc->nodesKnn_.size(); i++)
				{
					KnnIdx knn = m_egc->nodesKnn_[i];
					for (int k = 0; k < KnnK; k++)
					if (knn_k(knn, k) < m_egc->nodesKnn_.size())
						fprintf(pFile, "%d %d\n", i, knn_k(knn, k));
				}
				fclose(pFile);
			}
		}
#endif
	}

	void CpuGaussNewton::findCorr(const MapArr& vmap_live, const MapArr& nmap_live,
		const MapArr& vmap_warp, const MapArr& nmap_warp)
	{
		m_egc->vmap_live_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		m_egc->nmap_live_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		vmap_live.download(m_egc->vmap_live_.data(), m_egc->imgWidth_*sizeof(float4));
		nmap_live.download(m_egc->nmap_live_.data(), m_egc->imgWidth_*sizeof(float4));
		m_egc->vmap_warp_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		m_egc->nmap_warp_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		vmap_warp.download(m_egc->vmap_warp_.data(), m_egc->imgWidth_*sizeof(float4));
		nmap_warp.download(m_egc->nmap_warp_.data(), m_egc->imgWidth_*sizeof(float4));
		m_egc->CalcCorr();
	}

	void CpuGaussNewton::debug_set_init_x(const float* x, int n)
	{
		if (n != m_egc->x_.size())
		{
			printf("debug_set_init_x: size not matched: %d %d\n", n, m_egc->x_.size());
			throw std::exception();
		}
		for (int i = 0; i < n; i++)
			m_egc->x_[i] = x[i];
	}

	void CpuGaussNewton::solve(bool factor_rigid_out)
	{
		if (m_egc->x_.size() == 0)
			return;
#ifdef SHOW_LAST_NODE_INFO
		printf("last node before opt.: %f %f %f, %f %f %f\n", 
			m_egc->x_[m_egc->x_.size() - 6],
			m_egc->x_[m_egc->x_.size() - 5],
			m_egc->x_[m_egc->x_.size() - 4],
			m_egc->x_[m_egc->x_.size() - 3],
			m_egc->x_[m_egc->x_.size() - 2],
			m_egc->x_[m_egc->x_.size() - 1]);
#endif

		// pre initialization for all level!=0 nodes
#if 0
		std::vector<std::vector<int>> levelNbsToFiner(m_egc->nodesKnn_.size());
		for (int iNode = 0; iNode < m_egc->nodesKnn_.size(); iNode++)
		{
			WarpField::KnnIdx knn = m_egc->nodesKnn_[iNode];
			for (int k = 0; k < WarpField::KnnK; k++)
			{
				if (knn_k(knn, k) < m_egc->nodesKnn_.size())
					levelNbsToFiner.at(knn_k(knn, k)).push_back(iNode);
			}
		}

		// sequential...
		for (int iNode = 0; iNode < m_egc->nodesKnn_.size(); iNode++)
		{
			const std::vector<int>& nbs = levelNbsToFiner.at(iNode);
			if (nbs.size() == 0)
				continue;

			Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
			for (int k = 0; k < nbs.size(); k++)
			{
				int id = nbs[k]*6;
				Tbx::Vec3 r(m_egc->x_[id + 0], m_egc->x_[id + 1], m_egc->x_[id + 2]);
				Tbx::Vec3 t(m_egc->x_[id + 3], m_egc->x_[id + 4], m_egc->x_[id + 5]);

				Tbx::Dual_quat_cu dq;
				dq.from_twist(r, t);
				dq_blend = dq_blend + dq;
			}
			dq_blend.normalize();
			Tbx::Vec3 r, t;
			dq_blend.to_twist(r, t);
			m_egc->x_[iNode*6 + 0] = r.x;
			m_egc->x_[iNode*6 + 1] = r.y;
			m_egc->x_[iNode*6 + 2] = r.z;
			m_egc->x_[iNode*6 + 3] = t.x;
			m_egc->x_[iNode*6 + 4] = t.y;
			m_egc->x_[iNode*6 + 5] = t.z;
		}
#ifdef SHOW_LAST_NODE_INFO
		printf("last node after pre.: %f %f %f, %f %f %f\n",
			m_egc->x_[m_egc->x_.size() - 6],
			m_egc->x_[m_egc->x_.size() - 5],
			m_egc->x_[m_egc->x_.size() - 4],
			m_egc->x_[m_egc->x_.size() - 3],
			m_egc->x_[m_egc->x_.size() - 2],
			m_egc->x_[m_egc->x_.size() - 1]);
#endif
#endif
		m_egc->Optimize(m_egc->x_, m_egc->param_.fusion_GaussNewton_maxIter);

		if (factor_rigid_out)
		{
			Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0,0,0,0), Tbx::Quat_cu(0,0,0,0));
			for (int i = 0; i < m_egc->x_.size(); i += 6)
			{
				Tbx::Vec3 r(m_egc->x_[i + 0], m_egc->x_[i + 1], m_egc->x_[i + 2]);
				Tbx::Vec3 t(m_egc->x_[i + 3], m_egc->x_[i + 4], m_egc->x_[i + 5]);
				Tbx::Dual_quat_cu dq;
				dq.from_twist(r, t);
				dq_blend = dq_blend + dq;
			}
			dq_blend.normalize();

			Tbx::Dual_quat_cu invDq_blend = dq_blend.conjugate();

			for (int i = 0; i < m_egc->x_.size(); i += 6)
			{
				Tbx::Vec3 r(m_egc->x_[i + 0], m_egc->x_[i + 1], m_egc->x_[i + 2]);
				Tbx::Vec3 t(m_egc->x_[i + 3], m_egc->x_[i + 4], m_egc->x_[i + 5]);
				Tbx::Dual_quat_cu dq;
				dq.from_twist(r, t);
				dq = invDq_blend * dq;
				dq.to_twist(r, t);

				m_egc->x_[i + 0] = r.x;
				m_egc->x_[i + 1] = r.y;
				m_egc->x_[i + 2] = r.z;
				m_egc->x_[i + 3] = t.x;
				m_egc->x_[i + 4] = t.y;
				m_egc->x_[i + 5] = t.z;
			}
			m_pWarpField->set_rigidTransform(m_pWarpField->get_rigidTransform()*dq_blend.to_transformation());
		}

#ifdef SHOW_LAST_NODE_INFO
		printf("last node after opt.: %f %f %f, %f %f %f\n",
			m_egc->x_[m_egc->x_.size() - 6],
			m_egc->x_[m_egc->x_.size() - 5],
			m_egc->x_[m_egc->x_.size() - 4],
			m_egc->x_[m_egc->x_.size() - 3],
			m_egc->x_[m_egc->x_.size() - 2],
			m_egc->x_[m_egc->x_.size() - 1]);
#endif


		std::vector<float> tmpx(m_egc->x_.size());
		for (int i = 0; i < tmpx.size(); i++)
			tmpx[i] = m_egc->x_[i];

		m_twist.upload(tmpx.data(), m_egc->x_.size());
		m_pWarpField->update_nodes_via_twist(m_twist);
	}
}