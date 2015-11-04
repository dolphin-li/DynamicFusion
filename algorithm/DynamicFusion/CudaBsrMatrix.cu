#include "CudaBsrMatrix.h"
#include "CudaDiagBlockMatrix.h"

typedef CudaBsrMatrix::Range Range;

__global__ void CudaBsrMatrix_set(int n, float* ptr, float val)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
		ptr[i] = val;
}

__global__ void CudaBsrMatrix_scale_add(int n, float* ptr, float alpha, float beta)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
		ptr[i] = alpha * ptr[i] + beta;
}

__global__ void CudaBsrMatrix_fill_increment_1_n(int* data, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
		data[i] = i;
}

__global__ void CudaBsrMatrix_setRowFromBlockedCsrRowPtr(const int* csrRowPtr, 
	int* bsrRowPtr, int blockInRows, int rowsPerBlock, int elementsPerBlock)
{
	int iBlockRow = threadIdx.x + blockIdx.x * blockDim.x;
	if (iBlockRow <= blockInRows)
		bsrRowPtr[iBlockRow] = csrRowPtr[iBlockRow*rowsPerBlock]/elementsPerBlock;
}

__global__ void CudaBsrMatrix_transpose_fill_value_by_bid(const int* blockIds, const float* srcValues,
	float* dstValues, int blockSize_RxC, int blockR, int blockC, int nnz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nnz)
	{
		int dBlock = i / blockSize_RxC;
		int dShift = i - dBlock * blockSize_RxC;
		int dR = dShift / blockC;
		int dC = dShift - dR * blockC;
		int sBlock = blockIds[dBlock];
		int sShift = dC*blockR + dR;
		dstValues[i] = srcValues[sBlock * blockSize_RxC + sShift];
	}
}

template<int colsPerBlock>
__global__ void CudaBsrMatrix_Mv(cudaTextureObject_t bsrRowPtr, 
	cudaTextureObject_t bsrColIdx, cudaTextureObject_t bsrValue, 
	const float* x, float* y, float alpha, float beta, int nRow,
	int rowsPerBlock)
{
	int iRow = threadIdx.x + blockIdx.x * blockDim.x;
	if (iRow < nRow)
	{
		int iBlockRow = iRow / rowsPerBlock;
		int rowShift = iRow - iBlockRow * rowsPerBlock;
		int blockColPosBegin = 0;
		tex1Dfetch(&blockColPosBegin, bsrRowPtr, iBlockRow);
		int blockColPosEnd = 0;
		tex1Dfetch(&blockColPosEnd, bsrRowPtr, iBlockRow + 1);

		float sum = 0.f;
		for (int bIdx = blockColPosBegin; bIdx < blockColPosEnd; ++bIdx)
		{
			int iBlockCol = 0;
			tex1Dfetch(&iBlockCol, bsrColIdx, bIdx);
			iBlockCol *= colsPerBlock;
			int valIdx = (bIdx * rowsPerBlock + rowShift) * colsPerBlock;
			for (int c = 0; c < colsPerBlock; c++)
			{
				float val = 0.f;
				tex1Dfetch(&val, bsrValue, valIdx + c);
				sum += x[iBlockCol + c] * val;
			}
		}
		y[iRow] = alpha * sum + beta * y[iRow];
	}
}

template<bool LowerInsteadOfFull, bool Trans>
__device__ __forceinline__ float CudaBsrMatrix_rightMultDiag_1(int blockBeginLeft, int blockColResult,
	int rowPerBlock, int colsPerBlock, int elePerBlock, int rowShiftResult, int colShiftResult,
	cudaTextureObject_t leftValue, cudaTextureObject_t rightValue)
{
	return 0.f;
}

template<>
__device__ __forceinline__ float CudaBsrMatrix_rightMultDiag_1<false, false>(int blockBeginLeft, int blockColResult,
	int rowPerBlock, int colsPerBlock, int elePerBlock, int rowShiftResult, int colShiftResult,
	cudaTextureObject_t leftValue, cudaTextureObject_t rightValue)
{
	float sum = 0.f;
	blockBeginLeft = blockBeginLeft * elePerBlock + rowShiftResult * colsPerBlock;
	blockColResult = blockColResult * elePerBlock + colShiftResult * 1;
	for (int k = 0; k < colsPerBlock; k++)
	{
		float valLeft = 0.f, valRight = 0.f;
		tex1Dfetch(&valLeft, leftValue, blockBeginLeft + k);
		tex1Dfetch(&valRight, rightValue, blockColResult + k*colsPerBlock);
		sum += valLeft * valRight;
	}
	return sum;
}

template<>
__device__ __forceinline__ float CudaBsrMatrix_rightMultDiag_1<true, false>(int blockBeginLeft, int blockColResult,
	int rowPerBlock, int colsPerBlock, int elePerBlock, int rowShiftResult, int colShiftResult,
	cudaTextureObject_t leftValue, cudaTextureObject_t rightValue)
{
	float sum = 0.f;
	blockBeginLeft = blockBeginLeft * elePerBlock + rowShiftResult * colsPerBlock;
	blockColResult = blockColResult * elePerBlock + colShiftResult * 1;
	for (int k = colShiftResult; k < colsPerBlock; k++)
	{
		float valLeft = 0.f, valRight = 0.f;
		tex1Dfetch(&valLeft, leftValue, blockBeginLeft + k);
		tex1Dfetch(&valRight, rightValue, blockColResult + k*colsPerBlock);
		sum += valLeft * valRight;
	}
	return sum;
}

template<>
__device__ __forceinline__ float CudaBsrMatrix_rightMultDiag_1<false, true>(int blockBeginLeft, int blockColResult,
	int rowPerBlock, int colsPerBlock, int elePerBlock, int rowShiftResult, int colShiftResult,
	cudaTextureObject_t leftValue, cudaTextureObject_t rightValue)
{
	float sum = 0.f;
	blockBeginLeft = blockBeginLeft * elePerBlock + rowShiftResult * colsPerBlock;
	blockColResult = blockColResult * elePerBlock + colShiftResult * colsPerBlock;
	for (int k = 0; k < colsPerBlock; k++)
	{
		float valLeft = 0.f, valRight = 0.f;
		tex1Dfetch(&valLeft, leftValue, blockBeginLeft + k);
		tex1Dfetch(&valRight, rightValue, blockColResult + k);
		sum += valLeft * valRight;
	}
	return sum;
}

template<>
__device__ __forceinline__ float CudaBsrMatrix_rightMultDiag_1<true, true>(int blockBeginLeft, int blockColResult,
	int rowPerBlock, int colsPerBlock, int elePerBlock, int rowShiftResult, int colShiftResult,
	cudaTextureObject_t leftValue, cudaTextureObject_t rightValue)
{
	float sum = 0.f;
	blockBeginLeft = blockBeginLeft * elePerBlock + rowShiftResult * colsPerBlock;
	blockColResult = blockColResult * elePerBlock + colShiftResult * colsPerBlock;
	for (int k = 0; k <= colShiftResult; k++)
	{
		float valLeft = 0.f, valRight = 0.f;
		tex1Dfetch(&valLeft, leftValue, blockBeginLeft + k);
		tex1Dfetch(&valRight, rightValue, blockColResult + k);
		sum += valLeft * valRight;
	}
	return sum;
}

template<bool LowerInsteadOfFull, bool Trans>
__global__ void CudaBsrMatrix_rightMultDiag(
	const int* bsrRowPtr, const int* bsrRowPtr_coo,
	cudaTextureObject_t bsrColIdx, cudaTextureObject_t bsrValue,
	cudaTextureObject_t x, float* y, float alpha, float beta,
	int rowsPerBlock, int colsPerBlock, int nnz)
{
	int posResult = threadIdx.x + blockIdx.x * blockDim.x;
	if (posResult >= nnz)
		return;
	int elePerBlock = rowsPerBlock * colsPerBlock;
	int posResultBlock = posResult / elePerBlock;
	int shiftResult = posResult - posResultBlock * elePerBlock;
	int rowShiftResult = shiftResult / colsPerBlock;
	int colShiftResult = shiftResult - rowShiftResult * colsPerBlock;

	int blockRowResult = bsrRowPtr_coo[posResultBlock];
	int blockColResult = 0;
	tex1Dfetch(&blockColResult, bsrColIdx, posResultBlock);

	int blockBeginLeft = bsrRowPtr[blockRowResult];
	int blockEndLeft = bsrRowPtr[blockRowResult + 1];

	// binary search diag blocks: blockColResult
	while (blockBeginLeft < blockEndLeft)
	{
		int imid = ((blockBeginLeft + blockEndLeft) >> 1);
		int b = 0;
		tex1Dfetch(&b, bsrColIdx, imid);
		if (b < blockColResult)
			blockBeginLeft = imid + 1;
		else
			blockEndLeft = imid;
	}

	int b = 0;
	tex1Dfetch(&b, bsrColIdx, blockBeginLeft);
	float sum = 0.f;
	if (b == blockColResult && blockBeginLeft == blockEndLeft)
	{
		sum = CudaBsrMatrix_rightMultDiag_1<LowerInsteadOfFull, Trans>(
			blockBeginLeft, blockColResult, rowsPerBlock, colsPerBlock, elePerBlock,
			rowShiftResult, colShiftResult, bsrValue, x);
	}

	// write the result
	y[posResult] = alpha * sum + beta * y[posResult];
}

__global__ void CudaBsrMatrix_Range_multBsrT_value(
	const int* bsrRowPtrA, cudaTextureObject_t bsrColIdxA, cudaTextureObject_t valueA, int rangeColBeginA,
	const int* bsrRowPtrB, cudaTextureObject_t bsrColIdxB, cudaTextureObject_t valueB, int rangeColBeginB,
	const int* bsrRowPtrC_coo, const int* bsrColIdxC, float* valueC, 
	int rowsPerBlockA, int colsPerBlockA, int rowsPerBlockB, int nnzC
	)
{
	int innzC = threadIdx.x + blockIdx.x * blockDim.x;
	if (innzC >= nnzC)
		return;
	const int elePerBlockC = rowsPerBlockA * rowsPerBlockB;
	int innzBlockC = innzC / elePerBlockC;
	int innzShiftC = innzC - innzBlockC * elePerBlockC;
	int rowShiftC = innzShiftC / rowsPerBlockB;
	int colShiftC = innzShiftC - rowShiftC * rowsPerBlockB;
	int rowBlockC = bsrRowPtrC_coo[innzBlockC];
	int colBlockC = bsrColIdxC[innzBlockC];

	int blockBeginA = bsrRowPtrA[rowBlockC];
	int blockEndA = bsrRowPtrA[rowBlockC + 1];
	int blockBeginB = bsrRowPtrB[colBlockC];
	int blockEndB = bsrRowPtrB[colBlockC + 1];

	float sum = 0.f;
	for (int i0 = blockBeginA, i1 = blockBeginB; i0 < blockEndA && i1 < blockEndB;)
	{
		int colBlockA = 0, colBlockB = 0;
		tex1Dfetch(&colBlockA, bsrColIdxA, i0);
		tex1Dfetch(&colBlockB, bsrColIdxB, i1);
		colBlockA -= rangeColBeginA;
		colBlockB -= rangeColBeginB;
		if (colBlockA == colBlockB)
		{
			int pos0 = (i0*colsPerBlockA + rowShiftC)*rowsPerBlockA;
			int pos1 = (i1*rowsPerBlockB + colShiftC)*colsPerBlockA;
			for (int k = 0; k < colsPerBlockA; k++)
			{
				float v1 = 0.f, v2 = 0.f;
				tex1Dfetch(&v1, valueA, pos0 + k);
				tex1Dfetch(&v2, valueB, pos1 + k);
				sum += v1 * v2;
			}
			i0++;
			i1++;
		}

		i0 += (colBlockA < colBlockB);
		i1 += (colBlockA > colBlockB);
	}// i
	valueC[innzC] = sum;
}

void CudaBsrMatrix::fill_increment_1_n(int* data, int n)
{
	if (n == 0)
		return;
	CudaBsrMatrix_fill_increment_1_n << <divUp(n, CTA_SIZE), CTA_SIZE >> >(
		data, n);
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::fill_increment_1_n");
}

void CudaBsrMatrix::transpose_fill_values_by_blockId(const int* blockIds, const CudaBsrMatrix& t)
{
	CudaBsrMatrix_transpose_fill_value_by_bid << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		blockIds, t.value(), value(), rowsPerBlock() * colsPerBlock(), rowsPerBlock(), 
		colsPerBlock(), nnz());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::CudaBsrMatrix_transpose_fill_value_by_bid");
}

CudaBsrMatrix& CudaBsrMatrix::operator = (float constVal)
{
	if (nnz() == 0)
		return *this;
	if (constVal == 0.f)
	{
		cudaSafeCall(cudaMemset(m_values.ptr(), 0, nnz()*m_values.elem_size),
			"CudaBsrMatrix::operator = 0");
	}
	else
	{
		CudaBsrMatrix_set << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(nnz(), m_values.ptr(), constVal);
		cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::operator = constVal");
	}
	return *this;
}

CudaBsrMatrix& CudaBsrMatrix::operator = (const CudaBsrMatrix& rhs)
{
	m_cusparseHandle = rhs.m_cusparseHandle;
	resize(rhs.blocksInRow(), rhs.blocksInCol(), rhs.rowsPerBlock(), rhs.colsPerBlock());
	resize_nnzBlocks(rhs.nnzBlocks());

	cudaSafeCall(cudaMemcpy(bsrRowPtr(), rhs.bsrRowPtr(), 
		(1+rhs.blocksInRow())*sizeof(int), cudaMemcpyDeviceToDevice),
		"CudaBsrMatrix::operator =, cpy bsrRowPtr");
	cudaSafeCall(cudaMemcpy(bsrRowPtr_coo(), rhs.bsrRowPtr_coo(), 
		rhs.nnzBlocks()*sizeof(int), cudaMemcpyDeviceToDevice),
		"CudaBsrMatrix::operator =, cpy bsrRowPtr_coo");
	cudaSafeCall(cudaMemcpy(bsrColIdx(), rhs.bsrColIdx(),
		rhs.nnzBlocks()*sizeof(int), cudaMemcpyDeviceToDevice),
		"CudaBsrMatrix::operator =, cpy bsrColIdx");
	cudaSafeCall(cudaMemcpy(value(), rhs.value(),
		rhs.nnz()*sizeof(float), cudaMemcpyDeviceToDevice),
		"CudaBsrMatrix::operator =, cpy value");
	return *this;
}

CudaBsrMatrix& CudaBsrMatrix::axpy(float alpha, float beta)
{
	if (nnz() == 0)
		return *this;
	CudaBsrMatrix_scale_add << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		nnz(), value(), alpha, beta);
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::axpy");
	return *this;
}

void CudaBsrMatrix::setValue(const float* val_d)
{
	cudaSafeCall(cudaMemcpy(value(), val_d, nnz()*sizeof(float),
		cudaMemcpyDeviceToDevice), "CudaBsrMatrix::setValue");
}

void CudaBsrMatrix::Mv(const float* x, float* y, float alpha, float beta)const
{
	if (rows() == 0 || cols() == 0)
		return;
	switch (colsPerBlock())
	{
	case 0:
		break;
	case 1:
		CudaBsrMatrix_Mv<1> << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
			bsrRowPtrTexture(), bsrColIdxTexture(), valueTexture(), x, y, alpha, beta,
			rows(), rowsPerBlock());
		break;
	case 2:
		CudaBsrMatrix_Mv<2> << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
			bsrRowPtrTexture(), bsrColIdxTexture(), valueTexture(), x, y, alpha, beta,
			rows(), rowsPerBlock());
		break;
	case 3:
		CudaBsrMatrix_Mv<3> << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
			bsrRowPtrTexture(), bsrColIdxTexture(), valueTexture(), x, y, alpha, beta,
			rows(), rowsPerBlock());
		break;
	case 4:
		CudaBsrMatrix_Mv<4> << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
			bsrRowPtrTexture(), bsrColIdxTexture(), valueTexture(), x, y, alpha, beta,
			rows(), rowsPerBlock());
		break;
	case 5:
		CudaBsrMatrix_Mv<5> << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
			bsrRowPtrTexture(), bsrColIdxTexture(), valueTexture(), x, y, alpha, beta,
			rows(), rowsPerBlock());
		break;
	case 6:
		CudaBsrMatrix_Mv<6> << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
			bsrRowPtrTexture(), bsrColIdxTexture(), valueTexture(), x, y, alpha, beta,
			rows(), rowsPerBlock());
		break;
	default:
		throw std::exception("non-supported block size!");
	}
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::Mv");
}

void CudaBsrMatrix::rightMultDiag_structure(const CudaDiagBlockMatrix& x, CudaBsrMatrix& y)const
{
	if (cols() != x.rows())
		throw std::exception("CudaBsrMatrix::rightMultDiag_structure: block size not matched");
	if (x.blockSize() != colsPerBlock() || x.blockSize() != rowsPerBlock())
		throw std::exception("CudaBsrMatrix::rightMultDiag_structure: matrix size not matched");
	y = *this;
	y = 0;
}

void CudaBsrMatrix::rightMultDiag_value(const CudaDiagBlockMatrix& x, CudaBsrMatrix& y,
	bool useLowerInsteadOfFull_x, bool trans_x, float alpha, float beta)const
{
	if (cols() != x.rows())
		throw std::exception("CudaBsrMatrix::rightMultDiag_value: block size not matched");
	if (x.blockSize() != colsPerBlock() || x.blockSize() != rowsPerBlock())
		throw std::exception("CudaBsrMatrix::rightMultDiag_value: matrix size not matched");
	if (cols() != y.cols() || rows() != y.rows())
		throw std::exception("CudaBsrMatrix::rightMultDiag_value: y not matched, call rightMultDiag_structure()!");

	if (useLowerInsteadOfFull_x == true && trans_x == true)
		CudaBsrMatrix_rightMultDiag<true, true> << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtr(), bsrRowPtr_coo(), bsrColIdxTexture(), valueTexture(), 
		x.getTexture(), y.value(), alpha, beta, rowsPerBlock(), colsPerBlock(), nnz());
	if (useLowerInsteadOfFull_x == true && trans_x == false)
		CudaBsrMatrix_rightMultDiag<true, false> << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtr(), bsrRowPtr_coo(), bsrColIdxTexture(), valueTexture(),
		x.getTexture(), y.value(), alpha, beta, rowsPerBlock(), colsPerBlock(), nnz());
	if (useLowerInsteadOfFull_x == false && trans_x == false)
		CudaBsrMatrix_rightMultDiag<false, false> << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtr(), bsrRowPtr_coo(), bsrColIdxTexture(), valueTexture(),
		x.getTexture(), y.value(), alpha, beta, rowsPerBlock(), colsPerBlock(), nnz());
	if (useLowerInsteadOfFull_x == false && trans_x == true)
		CudaBsrMatrix_rightMultDiag<false, true> << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtr(), bsrRowPtr_coo(), bsrColIdxTexture(), valueTexture(),
		x.getTexture(), y.value(), alpha, beta, rowsPerBlock(), colsPerBlock(), nnz());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::rightMultDiag_value");
}

void CudaBsrMatrix::setRowFromBlockedCsrRowPtr(const int* csrRowPtr)
{
	if (blocksInRow() == 0)
		return;
	
	beginConstructRowPtr();
	CudaBsrMatrix_setRowFromBlockedCsrRowPtr << <divUp(blocksInRow(), CTA_SIZE), CTA_SIZE >> >(
		csrRowPtr, bsrRowPtr(), blocksInRow(), rowsPerBlock(), rowsPerBlock()*colsPerBlock());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::setRowFromBlockedCsrRowPtr");
	endConstructRowPtr();
}

void CudaBsrMatrix::Range::multBsr_value(const Range& B, CudaBsrMatrix& C, float alpha)
{
	throw std::exception("CudaBsrMatrix::Range::multBsr_value(): not implemented");
}

void CudaBsrMatrix::Range::multBsrT_value(const Range& B, CudaBsrMatrix& C, float alpha)
{
	if (A == nullptr || B.A == nullptr)
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): null pointer exception");
	if (blocksInCol() != B.blocksInCol())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): matrix size not matched");
	if (colsPerBlock() != B.colsPerBlock())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): block size not matched");
	if (blocksInRow() != C.blocksInRow() || B.blocksInRow() != C.blocksInCol()
		|| rowsPerBlock() != C.rowsPerBlock() || B.rowsPerBlock() != C.colsPerBlock())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): result size not matched");
	if (C.nnzBlocks() == 0)
		return;

	CudaBsrMatrix_Range_multBsrT_value << <divUp(C.nnz(), CTA_SIZE), CTA_SIZE >> >(
		A->bsrRowPtr()+blockRowBegin, A->bsrColIdxTexture(), A->valueTexture(), blockColBegin,
		B.A->bsrRowPtr()+B.blockRowBegin, B.A->bsrColIdxTexture(), B.A->valueTexture(), B.blockColBegin,
		C.bsrRowPtr_coo(), C.bsrColIdx(), C.value(),
		rowsPerBlock(), colsPerBlock(), B.rowsPerBlock(), C.nnz()
		);
}