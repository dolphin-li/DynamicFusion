//	******************************************************
//	Class BFGSFit
//	Solve the x in min{(Ax-B)^2} using BFGS
//	Author: Chen Cao	10/17/2013
//	******************************************************

#pragma once

// Include common
#include "program.h"

// Include Eigen
#include <Eigen/Core>
#include <vector>

namespace ldp
{
	class BFGSFit : public Program
	{
	public:
		// Constructor, initialize
		BFGSFit								( int n, double* x, double* lb, double* ub, long int* btype,
											int m = defaultm, int maxiter = defaultmaxiter,
											double factr = defaultfactr, double pgtol = defaultpgtol );
		void initialize						();

		// Access functions
		void setMatrix						( const Eigen::MatrixXf& A, const Eigen::VectorXf& B );

		// Re-implemented functions
		virtual double computeObjective		( int n, double* x );
		virtual void computeGradient		( int n, double* x, double* g );
		virtual void iterCallback			( int t, double* x, double f );

	protected:
		// Basic members
		std::vector<double>					mX;			// Object
		std::vector<double>					mG;			// Gradient

		const Eigen::MatrixXf*				mA_P;		// Matrix A's pointer
		const Eigen::VectorXf*				mB_P;		// Vector B's pointer

		Eigen::VectorXf						mAx_B;		// A * X - B
		Eigen::MatrixXf						mAt;		// At
		Eigen::VectorXf						mAtAx_B;	// At*(A*X-B)
	};
}
