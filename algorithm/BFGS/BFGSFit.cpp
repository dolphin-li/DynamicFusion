//	******************************************************
//	Class BFGSFit
//	Solve the x in min{(Ax-B)^2} using BFGS
//	Author: Chen Cao	10/17/2013
//	******************************************************

#include "BFGSFit.h"

using namespace ldp;

// <-----------------------Public Functions----------------------->

//////////////////////////////////////////////////////////////////////////
// <Constructor, initialize>
BFGSFit::BFGSFit( int n, double* x, double* lb, double* ub, long int* btype, 
				 int m /* = defaultm */, int maxiter /* = defaultmaxiter */, 
				 double factr /* = defaultfactr */, double pgtol /* = defaultpgtol */ )
				 : Program( n, x, lb, ub, btype, m, maxiter, factr, pgtol )
{
	initialize();
}

void BFGSFit::initialize()
{
	mX					.clear();
	mG					.clear();

	mA_P				= NULL;
	mB_P				= NULL;

	mAx_B				.setZero( 0 );
	mAt					.setZero( 0, 0 );
	mAtAx_B				.setZero( 0 );
}
// </Constructor, initialize>
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// <Access functions>
void BFGSFit::setMatrix( const Eigen::MatrixXf& A, const Eigen::VectorXf& B )
{
	mA_P = &A;
	mB_P = &B;

	mAt = mA_P->transpose();
}
// </Access functions>
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// <Re-implemented functions>
double BFGSFit::computeObjective( int n, double* x )
{
	Eigen::VectorXf xVec( n );
	for( int i=0; i<n; i++ )	xVec[ i ] = x[ i ];

	mAx_B = (*mA_P) * xVec - (*mB_P);
	double ret = mAx_B.dot( mAx_B );

	return ret;
}

void BFGSFit::computeGradient( int n, double* x, double* g )
{
	Eigen::VectorXf xVec( n );
	for( int i=0; i<n; i++ )	xVec[ i ] = x[ i ];

	mAx_B = (*mA_P) * xVec - (*mB_P);
	mAtAx_B = mAt * mAx_B;

	for( int i=0; i<n; i++ )	g[ i ] = mAtAx_B[ i ];
}

void BFGSFit::iterCallback( int t, double* x, double f )
{

}
// </Re-implemented functions>
//////////////////////////////////////////////////////////////////////////

// </-----------------------Public Functions----------------------->