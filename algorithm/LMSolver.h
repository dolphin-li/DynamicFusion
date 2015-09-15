#pragma once

#include <eigen\Core>
#include <eigen\Dense>
#include <eigen\Svd>
#include <eigen\Sparse>

/**
* Levenberg-Marquardt Method for dense fixed-size equations
* min_x { f(x)^2 }, with f_(m,1) and x_(n,1)
* */
template<int M, int N>
class CSmallLMSolver
{
public:
	typedef double real;
	typedef Eigen::Matrix<real,M,1> VecM;
	typedef Eigen::Matrix<real,N,1> VecN;
	typedef Eigen::Matrix<real,M,N> MatMN;
	typedef Eigen::Matrix<real,N,M> MatNM;
	typedef Eigen::Matrix<real,N,N> MatNN;
#define CSmallLMSolver_Max(a,b) (a)>(b) ? (a) : (b);
public:

	real Optimize(VecN& xStart, int nMaxIter)
	{
		const static real eps_1 = real(1e-12);
		const static real eps_2 = real(1e-12);
		const static real tau = real(1e-3);
		MatMN Jac;
		MatNM JacT;
		MatNN JacTJac;
		VecM f;	//f(x)
		VecN g;	//g=-J'*f
		VecN h;  //(J'J + mu * I) h = g
		VecN xnew, x;
		VecM fnew;	//f(xnew)
		Eigen::LDLT<MatNN> solver;

		//initial settings
		x = xStart;
		CalcJacobiFunc(x, Jac);	//J
		JacT = Jac.transpose();
		CalcEnergyFunc(x, f);	//f
		g = JacT * (-f);	//g
		if(g.norm() <= eps_1)
			return true;
		JacTJac = JacT*Jac;
		solver.compute(JacTJac);
		h = solver.solve(g);	//J'Jh = g

		//find the diag of J'J
		real v=real(2), mu=real(0);
		for(int i=0; i<JacTJac.rows(); i++)
			mu = CSmallLMSolver_Max(mu, JacTJac(i,i));
		mu *= tau;

		for(int iter=0; iter < nMaxIter; iter++)
		{
			//+mu*I
			for(int i=0; i<JacTJac.rows(); i++)
				JacTJac(i,i) += mu;
	
			//solve
			solver.compute(JacTJac);
			h = solver.solve(g);

			//-mu*I
			for(int i=0; i<JacTJac.rows(); i++)
				JacTJac(i,i) -= mu;

			if(h.norm() <= eps_2 * (x.norm() + eps_2))
				break;

			xnew = x + h;	
			CalcEnergyFunc(xnew, fnew);	//fnew

			//dL = L(0) - L(h)
			real dL = h.dot(mu*h+g);
			real dF = f.dot(f) - fnew.dot(fnew);
			real rho = dF/dL;

			if(rho > 0)
			{
				x = xnew;	//x
				f = fnew;	//f
				CalcJacobiFunc(x, Jac);	//J
				JacT = Jac.transpose();
				g = JacT * (-f);	//g	
				if(g.norm() <= eps_1)
					break;
				JacTJac = JacT * Jac;	

				mu *= CSmallLMSolver_Max(real(1./3.), real(1-pow(2*rho-1,3)));
				v = 2;
			}
			else
			{
				mu *= v;
				v *= 2;
			}
		}//end for iter

		xStart = x;
		return f.dot(f);
	}


protected:
	virtual void CalcEnergyFunc(const VecN& x, VecM& fx)=0;
	
	//default: calc jacobi matrix numerically with forward diff
	virtual void CalcJacobiFunc(VecN& x, MatMN& jac)
	{
		const static real delta = real(1e-8);

		VecM fx, fxp;
		CalcEnergyFunc(x, fx);
		for(int j=0; j<N; j++)
		{
			real d=real(1e-4)*x[j]; // force evaluation
			d=fabs(d);
			if(d<delta) d=delta;

			x[j] += d;
			CalcEnergyFunc(x, fxp);
			x[j] -= d;

			d = real(1.0)/d;
			for(int i=0; i<M; i++)
			{
				jac(i,j) = (fxp[i]-fx[i])*d;
			}

		}//end for j
	}
};


/**
* Levenberg-Marquardt Method for dense fix-variable equations
* min_x { f(x)^2 }, with f_(m,1) and x_(n,1)
* */
template<int N>
class CFixVarLMSolver
{
public:
	typedef double real;
	typedef Eigen::Matrix<real,-1,1> VecM;
	typedef Eigen::Matrix<real,N,1> VecN;
	typedef Eigen::Matrix<real,-1,N> MatMN;
	typedef Eigen::Matrix<real,N,-1> MatNM;
	typedef Eigen::Matrix<real,N,N> MatNN;
#define CSmallLMSolver_Max(a,b) (a)>(b) ? (a) : (b);
public:

	real Optimize(VecN& xStart, int nMaxIter)
	{
		const static real eps_1 = real(1e-12);
		const static real eps_2 = real(1e-12);
		const static real tau = real(1e-3);
		MatMN Jac(M,N);
		MatNM JacT(N,M);
		MatNN JacTJac;
		VecM f(M);	//f(x)
		VecN g;	//g=-J'*f
		VecN h;  //(J'J + mu * I) h = g
		VecN xnew, x;
		VecM fnew(M);	//f(xnew)
		Eigen::LDLT<MatNN> solver;

		//initial settings
		x = xStart;
		CalcJacobiFunc(x, Jac);	//J
		JacT = Jac.transpose();
		CalcEnergyFunc(x, f);	//f
		g = JacT * (-f);	//g
		if(g.norm() <= eps_1)
			return true;
		JacTJac = JacT*Jac;
		solver.compute(JacTJac);
		h = solver.solve(g);	//J'Jh = g

		//find the diag of J'J
		real v=real(2), mu=real(0);
		for(int i=0; i<JacTJac.rows(); i++)
			mu = CSmallLMSolver_Max(mu, JacTJac(i,i));
		mu *= tau;

		for(int iter=0; iter < nMaxIter; iter++)
		{
			//+mu*I
			for(int i=0; i<JacTJac.rows(); i++)
				JacTJac(i,i) += mu;
	
			//solve
			solver.compute(JacTJac);
			h = solver.solve(g);

			//-mu*I
			for(int i=0; i<JacTJac.rows(); i++)
				JacTJac(i,i) -= mu;

			if(h.norm() <= eps_2 * (x.norm() + eps_2))
				break;

			xnew = x + h;	
			CalcEnergyFunc(xnew, fnew);	//fnew

			//dL = L(0) - L(h)
			real dL = h.dot(mu*h+g);
			real dF = f.dot(f) - fnew.dot(fnew);
			real rho = dF/dL;

			if(rho > 0)
			{
				x = xnew;	//x
				f = fnew;	//f
				CalcJacobiFunc(x, Jac);	//J
				JacT = Jac.transpose();
				g = JacT * (-f);	//g	
				if(g.norm() <= eps_1)
					break;
				JacTJac = JacT * Jac;	

				mu *= CSmallLMSolver_Max(real(1./3.), real(1-pow(2*rho-1,3)));
				v = 2;
			}
			else
			{
				mu *= v;
				v *= 2;
			}
		}//end for iter

		xStart = x;

		//return the final energy
		return ldp::sqr( f.norm() );
	}


protected:
	virtual void CalcEnergyFunc(const VecN& x, VecM& fx)=0;
	
	//default: calc jacobi matrix numerically with forward diff
	virtual void CalcJacobiFunc(VecN& x, MatMN& jac)
	{
		const static real delta = real(1e-8);

		VecM fx(M), fxp(M);
		CalcEnergyFunc(x, fx);
		for(int j=0; j<N; j++)
		{
			real d=real(1e-4)*x[j]; // force evaluation
			d=fabs(d);
			if(d<delta) d=delta;

			x[j] += d;
			CalcEnergyFunc(x, fxp);
			x[j] -= d;

			d = real(1.0)/d;
			for(int i=0; i<M; i++)
			{
				jac(i,j) = (fxp[i]-fx[i])*d;
			}

		}//end for j
	}
protected:
	int M;
};


/**
* Levenberg-Marquardt Method for dense equations
* min_x { f(x)^2 }, with f_(m,1) and x_(n,1)
* */
class CDenseLMSolver
{
public:
	typedef double real;
	typedef Eigen::Matrix<real,-1,1> DVec;
	typedef Eigen::Matrix<real,-1,-1> DMat;
#define CSmallLMSolver_Max(a,b) (a)>(b) ? (a) : (b);
public:
	CDenseLMSolver()
	{
		useBound = false;
	}

	real Optimize(DVec& xStart, int nMaxIter, bool isInfoShow=false)
	{
		const static real eps_1 = real(1e-12);
		const static real eps_2 = real(1e-12);
		const static real tau = real(1e-3);
		DMat Jac(M,N);
		DMat JacT(N,M);
		DMat JacTJac(N,N);
		DVec f(M);	//f(x)
		DVec g(N);	//g=-J'*f
		DVec h(N);  //(J'J + mu * I) h = g
		DVec xnew(N), x(N);
		DVec fnew(M);	//f(xnew)
		Eigen::LDLT<DMat> solver;

		//initial settings
		x = xStart;
		CalcJacobiFunc(x, Jac);	//J
		JacT = Jac.transpose();
		CalcEnergyFunc(x, f);	//f
		g = JacT * (-f);	//g
		if(g.norm() <= eps_1)
			return true;
		JacTJac = JacT*Jac;
		solver.compute(JacTJac);
		h = solver.solve(g);	//J'Jh = g

		//find the diag of J'J
		real v=real(2), mu=real(0);
		for(int i=0; i<JacTJac.rows(); i++)
			mu = CSmallLMSolver_Max(mu, JacTJac(i,i));
		mu *= tau;

		for(int iter=0; iter < nMaxIter; iter++)
		{
			//+mu*I
			for(int i=0; i<JacTJac.rows(); i++)
				JacTJac(i,i) += mu;
	
			//solve
			solver.compute(JacTJac);
			h = solver.solve(g);

			//-mu*I
			for(int i=0; i<JacTJac.rows(); i++)
				JacTJac(i,i) -= mu;

			if(h.norm() <= eps_2 * (x.norm() + eps_2))
				break;

			xnew = x + h;

			if (useBound)
			{
				xnew = xnew.cwiseMin(x_upper);
				xnew = xnew.cwiseMax(x_lower);
				h = xnew - h;
			}

			CalcEnergyFunc(xnew, fnew);	//fnew

			//dL = L(0) - L(h)
			real dL = h.dot(mu*h+g);
			real dF = f.dot(f) - fnew.dot(fnew);
			real rho = dF/dL;

			if(rho > 0)
			{
				x = xnew;	//x
				f = fnew;	//f
				CalcJacobiFunc(x, Jac);	//J
				JacT = Jac.transpose();
				g = JacT * (-f);	//g	
				if(g.norm() <= eps_1)
					break;
				JacTJac = JacT * Jac;	

				mu *= CSmallLMSolver_Max(real(1./3.), real(1-pow(2*rho-1,3)));
				v = 2;
			}
			else
			{
				mu *= v;
				v *= 2;
			}

			if (isInfoShow && iter % 10 == 0)
				printf("iter: %d, energy: %f, dif: %f\n", iter, sqrt(f.dot(f)), h.norm()/x.norm());
		}//end for iter

		xStart = x;
		return f.dot(f);
	}

	void SetBound(DVec& xMin, DVec& xMax)
	{
		x_lower = xMin;
		x_upper = xMax;
		useBound = true;
	}

protected:
	virtual void CalcEnergyFunc(const DVec& x, DVec& fx)=0;
	
	//default: calc jacobi matrix numerically with forward diff
	virtual void CalcJacobiFunc(DVec& x, DMat& jac)
	{
		const static real delta = real(1e-8);

		DVec fx(M), fxp(M);
		CalcEnergyFunc(x, fx);
		
		for(int j=0; j<N; j++)
		{
			real d=real(1e-4)*x[j]; // force evaluation
			d=fabs(d);
			if(d<delta) d=delta;

			x[j] += d;
			CalcEnergyFunc(x, fxp);
			x[j] -= d;

			d = real(1.0)/d;
			for(int i=0; i<M; i++)
			{
				jac(i,j) = (fxp[i]-fx[i])*d;
			}

		}//end for j
	}
protected:
	int M;
	int N;
	bool useBound;
	DVec x_lower, x_upper;
};


class CSparseLMSolver
{
public:
	typedef double real;
	typedef Eigen::Matrix<real,-1,1> Vec;
	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> SpMat;
#define CSmallLMSolver_Max(a,b) (a)>(b) ? (a) : (b);
public:
	real Optimize(Vec& xStart, int nMaxIter, bool showInfo=true)
	{
		const static real eps_1 = real(1e-12);
		const static real eps_2 = real(1e-12);
		const static real tau = real(1e-3);

		//define jacobi structure
		DefineJacobiStructure(m_jacobi, m_jacobiT);

		SpMat JacTJac;
		Vec f(m_jacobi.rows());	//f(x)
		Vec g(m_jacobi.cols());	//g=-J'*f
		Vec h(m_jacobi.cols());  //(J'J + mu * I) h = g
		Eigen::VectorXi diagPos(m_jacobi.cols());
		Vec diagKept(m_jacobi.cols());
		Vec xnew(xStart.size());
		Vec fnew(m_jacobi.rows());	//f(xnew)
	
		//define structure of J'J
		JacTJac = m_jacobiT * m_jacobiT.transpose();
		Eigen::SimplicialCholesky<SpMat> solver(JacTJac);

		//initial settings
		CalcJacobiFunc(xStart,m_jacobi, m_jacobiT);	//J
		CalcEnergyFunc(xStart, f);	//f
		g = m_jacobiT * (-f);	//g
		if(g.norm() <= eps_1)
			return f.dot(f);
		FastAtAGivenStructure(m_jacobi, m_jacobiT, JacTJac);
		//JacTJac = m_jacobiT * m_jacobiT.transpose();
		solver.factorize(JacTJac);
		h = solver.solve(g);	//J'Jh = g

		//find the diag of J'J
		real v=real(2), mu=real(0);
		Eigen::Diagonal<SpMat> diag(JacTJac);
		for(int i=0; i<diag.size(); i++)
			mu = CSmallLMSolver_Max(mu, diag[i]);
		mu *= tau;

		for(int iter=0; iter < nMaxIter; iter++)
		{
			//+mu*I
			diag += Vec(diag.size()).setConstant(mu);
	
			//solve
			solver.factorize(JacTJac);
			h = solver.solve(g);

			//-mu*I
			diag -= Vec(diag.size()).setConstant(mu);

			if(h.norm() <= eps_2 * (xStart.norm() + eps_2))
				break;

			xnew = xStart + h;	
			CalcEnergyFunc(xnew, fnew);	//fnew

			//dL = L(0) - L(h)
			real dL = h.dot(mu*h+g);
			real dF = f.dot(f) - fnew.dot(fnew);
			real rho = dF/dL;

			if(rho > 0)
			{
				xStart = xnew;	//x
				f = fnew;	//f
				CalcJacobiFunc(xStart,m_jacobi, m_jacobiT);	//J
				g = m_jacobiT * (-f);	//g	
				if(g.norm() <= eps_1)
					break;
				FastAtAGivenStructure(m_jacobi, m_jacobiT, JacTJac);
				//JacTJac = m_jacobiT * m_jacobiT.transpose();	

				mu *= CSmallLMSolver_Max(real(1./3.), real(1-pow(2*rho-1,3)));
				v = 2;
			}
			else
			{
				mu *= v;
				v *= 2;
			}
			if((iter+1)%10==0 && showInfo)
				printf("L-M:%d  dif: %ef  energy: %ef  mu:%ef\n", iter, h.norm()/(eps_2+xStart.norm()), rho, mu);
		}//end for iter

		return f.dot(f);
	}

protected:
	//define the structure of jac and jacT
	virtual void DefineJacobiStructure(SpMat& jac, SpMat& jacT)=0;

	virtual void CalcEnergyFunc(const Vec& x, Vec& fx)=0;
	
	//default: calc jacobi matrix numerically with forward diff
	//Note that structure of both jac and jacT should be given
	//children classes should fill both jac and jacT
	virtual void CalcJacobiFunc(Vec& pTest, SpMat& jac, SpMat& jacT)
	{
		Vec p = pTest;
		const int nobs = jac.rows();
		const int nvars = jac.cols();

		register int i, j, jj, k;
		register real d;
		int ii, m, *jcol, *varlist, *coldone, forw;
		int *vidxs, *ridxs;
		real *tmpd;
		real delta;

		/* retrieve problem-specific information passed in *dat */
		Vec hx(nobs), hxx(nobs);
		forw=1;
		delta=real(1e-8);

		CalcEnergyFunc(p, hx);//hx = f(p)

		jcol=(int *)malloc(nobs*sizeof(int)); /* keeps track of measurements influenced by the set of variables currently in "varlist" below */
		for(i=0; i<nobs; ++i) jcol[i]=-1;

		vidxs=(int *)malloc(2*nobs*sizeof(int));
		ridxs=vidxs+nobs;

		varlist=(int *)malloc(nvars*sizeof(int)); /* stores indices of J's columns which are computed with the same "func" call */
		coldone=(int *)malloc(nvars*sizeof(int)); /* keeps track of J's columns which have been already computed */
		memset(coldone, 0, nvars*sizeof(int)); /* initialize to zero */

		tmpd=(real *)malloc(nvars*sizeof(real));

		for(j=0; j<nvars; ++j)
		{
			real scl;

			if(coldone[j]) continue; /* column j already computed */

			//for(i=0; i<nobs; ++i) jcol[i]=-1;
			k=FindColIndices(jac, j, vidxs, ridxs);
			for(i=0; i<k; ++i) jcol[ridxs[i]]=j;
			varlist[0]=j; m=1; coldone[j]=1;

			for(jj=j+1; jj<nvars; ++jj)
			{
				if(coldone[jj]) continue; /* column jj already computed */

				k=FindColIndices(jac, jj, vidxs, ridxs);
				for(i=0; i<k; ++i)
				if(jcol[ridxs[i]]!=-1) goto nextjj;

				if(k==0) { coldone[jj]=1; continue; } /* all zeros column, ignore */

				/* column jj does not clash with previously considered ones, mark it */
				for(i=0; i<k; ++i) jcol[ridxs[i]]=jj;
				varlist[m++]=jj; coldone[jj]=1;

			nextjj:
				continue;
			}

			for(k=0; k<m; ++k)
			{
				/* determine d=max(SPLM_DELTA_SCALE*|p[varlist[k]]|, delta), see HZ */
				d=real(1e-4)*p[varlist[k]]; // force evaluation
				d=fabs(d);
				if(d<delta) d=delta;

				tmpd[varlist[k]]=d;
				p[varlist[k]]+=d;
			}

			CalcEnergyFunc(p, hxx);// hxx=f(p+d)

			if(forw)
			{
				for(k=0; k<m; ++k)
				p[varlist[k]]-=tmpd[varlist[k]]; /* restore */

				scl=1.0;
			}
			else
			{ // central
				for(k=0; k<m; ++k)
				p[varlist[k]]-=2*tmpd[varlist[k]];

				CalcEnergyFunc(p, hx);// hx=f(p-d)

				for(k=0; k<m; ++k)
				p[varlist[k]]+=tmpd[varlist[k]]; /* restore */

				scl=0.5; // 1./2.
			}

			for(k=0; k<m; ++k)
			{
				d=tmpd[varlist[k]];
				d=scl/d; /* invert so that divisions can be carried out faster as multiplications */

				jj=FindColIndices(jac, varlist[k], vidxs, ridxs);
				for(i=0; i<jj; ++i)
				{
					ii=ridxs[i];
					jac.valuePtr()[vidxs[i]]=(hxx[ii]-hx[ii])*d;
					jcol[ii]=-1; /* restore */
				}
			}
		}//end for jj

		//calc 
		FastTransGivenStructure(jac, jacT);

		free(tmpd);
		free(coldone);
		free(varlist);
		free(vidxs);
		free(jcol);
	}

protected:
	int FindColIndices(const SpMat& A, int cid, int* vidx, int* ridx)
	{
		int ns = A.outerIndexPtr()[cid], ne = A.outerIndexPtr()[cid+1];
		int k=0;
		for(int i=ns; i<ne; i++,k++)
		{
			ridx[k] = A.innerIndexPtr()[i];
			vidx[k] = i;
		}
		return k;
	}
	void FastTransGivenStructure(const Eigen::SparseMatrix<real>& A, Eigen::SparseMatrix<real>& At)
	{
		Eigen::VectorXi positions(At.outerSize());
		for(int i=0; i<At.outerSize(); i++)
			positions[i] = At.outerIndexPtr()[i];
		for (int j=0; j<A.outerSize(); ++j)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it)
			{
				int i = it.index();
				int pos = positions[i]++;
				At.valuePtr()[pos] = it.value();
			}
		}
	}
	void FastAtAGivenStructure(const Eigen::SparseMatrix<real>& A, const Eigen::SparseMatrix<real>& At, Eigen::SparseMatrix<real>& AtA)
	{
		const static int nThread = 1;
		//omp_set_num_threads(nThread);

		Eigen::VectorXd Tmps[nThread];
		Eigen::VectorXi Marks[nThread];
		for(int i=0; i<nThread; i++)
		{
			Tmps[i].resize(AtA.innerSize());
			Marks[i].resize(AtA.innerSize());
			Marks[i].setZero();
		}

		//#pragma omp parallel for
		for(int j=0; j<AtA.outerSize(); j++)
		{
			int tid = 0;//omp_get_thread_num();
			Eigen::VectorXd& Tmp = Tmps[tid];
			Eigen::VectorXi& Mark = Marks[tid];
			for(Eigen::SparseMatrix<double>::InnerIterator it_A(A, j); it_A; ++it_A)
			{
				int k = it_A.index();
				double v_A = it_A.value();

				for(Eigen::SparseMatrix<double>::InnerIterator it_At(At, k); it_At; ++it_At)
				{
					int i = it_At.index();
					double v_At = it_At.value();
					if(!Mark[i])
					{
						Mark[i] = 1;
						Tmp[i] = v_A*v_At;
					}
					else
						Tmp[i] += v_A*v_At;
				}//end for it_At
			}//end for it_A

			for(Eigen::SparseMatrix<double>::InnerIterator it(AtA, j); it; ++it)
			{
				int i = it.index();
				it.valueRef() = Tmp[i];
				Mark[i] = 0;
			}
		}//end for i
	}
protected:
	SpMat m_jacobi;
	SpMat m_jacobiT;
};

class CDenseNewtonSolver :public CDenseLMSolver
{
public:
	real Optimize(DVec& xStart, int nMaxIter, bool showInfo = false)
	{
		DMat jac(M, N);
		DMat JacTJac(N,N);
		DVec fx(M), fx1(M), h(N), g(N);
		Eigen::LDLT<DMat> solver;

		//Gauss-Newton Optimization
		for (int iter = 0; iter<nMaxIter; iter++)
		{
			CalcJacobiFunc(xStart, jac);	//J
			JacTJac = jac.transpose() * jac;

			CalcEnergyFunc(xStart, fx);	//f

			//solve: J'J h =  - J' f(x)
			g = jac.transpose() * (-fx);
			solver.compute(JacTJac);
			h = solver.solve(g);

			real normv = xStart.norm();
			for (real alpha = 1; alpha > 1e-15; alpha *= 0.5)
			{
				DVec x = xStart + h;
				if (useBound)
				{
					x = x.cwiseMin(x_upper);
					x = x.cwiseMax(x_lower);
				}
				CalcEnergyFunc(x, fx1);	//f
				if (fx1.dot(fx1) > fx.dot(fx))
					h = h * 0.5;
				else
				{
					xStart = x;
					break;
				}
			}

			real normh = h.norm();

			if (showInfo)
				printf("Gauss-Newton: %d -- %f, energy: %f\n", iter, normh / normv, sqrt(fx.dot(fx)));

			if (normh < (normv + real(1e-6)) * real(1e-6))
				break;
		}

		return fx.dot(fx);
	}
};

class CSparseNewtonSolver:public CSparseLMSolver
{
public:
	real Optimize(Vec& xStart, int nMaxIter, bool showInfo=true)
	{
		//define jacobi structure
		DefineJacobiStructure(m_jacobi, m_jacobiT);

		SpMat JacTJac;
		Vec fx(m_jacobi.rows()), h(m_jacobi.cols()), g(m_jacobi.cols()), fx1(m_jacobi.rows());

		//define structure of J'J
		JacTJac = m_jacobiT * m_jacobi;
		Eigen::SimplicialCholesky<SpMat> solver;
		solver.analyzePattern(JacTJac.triangularView<Eigen::Lower>());

		//Gauss-Newton Optimization
		for(int iter=0; iter<nMaxIter; iter++)
		{
			CalcJacobiFunc(xStart, m_jacobi, m_jacobiT);	//J

			//JacTJac = m_jacobiT * m_jacobi;//J'J
			FastAtAGivenStructure(m_jacobi, m_jacobiT, JacTJac);
			
			CalcEnergyFunc(xStart, fx);	//f

			//solve: J'J h =  - J' f(x)
			g = m_jacobiT * (-fx);
			solver.factorize(JacTJac.triangularView<Eigen::Lower>());
			h = solver.solve(g);

			real normv = xStart.norm();
			double old_energy = fx.dot(fx);
			for (real alpha = 1; alpha > 1e-15; alpha *= 0.5)
			{
				Vec x = xStart + h;
				CalcEnergyFunc(x, fx1);	//f
				double new_energy = fx1.dot(fx1);
				if (new_energy > old_energy)
					h = h * 0.5;
				else
				{
					xStart = x;
					break;
				}
			}
			real normh = h.norm();

			if(showInfo)
				printf("Gauss-Newton: %d -- %f\n", iter, normh/normv);

			if(normh < (normv+real(1e-6)) * real(1e-6))
				break;
		}

		return fx.dot(fx);
	}
};