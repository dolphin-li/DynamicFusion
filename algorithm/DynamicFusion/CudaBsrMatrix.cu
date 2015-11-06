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

__global__ void CudaBsrMatrix_scale_add_diag(int nRows, float* ptr, 
	const int* bsrRowPtr, cudaTextureObject_t bsrColIdx, 
	int blockSize, float alpha, float beta)
{
	int iRow = threadIdx.x + blockIdx.x * blockDim.x;
	if (iRow >= nRows)
		return;

	int iBlockRow = iRow / blockSize;
	int rowBlockBegin = bsrRowPtr[iBlockRow];
	int rowBlockEnd = bsrRowPtr[iBlockRow + 1];

	for (int c = rowBlockBegin; c < rowBlockEnd; c++)
	{
		int iBlockCol = 0;
		tex1Dfetch(&iBlockCol, bsrColIdx, c);
		if (iBlockCol == iBlockRow)
		{
			int rowShift = iRow - iBlockRow * blockSize;
			int pos = (c * blockSize + rowShift) * blockSize + rowShift;
			ptr[pos] = alpha * ptr[pos] + beta;
		}
	}
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
	const int* bsrRowPtrA, cudaTextureObject_t bsrColIdxA, cudaTextureObject_t valueA, 
	int rangeColBeginA, int rangeColEndA,
	const int* bsrRowPtrB, cudaTextureObject_t bsrColIdxB, cudaTextureObject_t valueB, 
	int rangeColBeginB, int rangeColEndB,
	const int* bsrRowPtrD, cudaTextureObject_t bsrColIdxD, cudaTextureObject_t valueD,
	int rangeColBeginD, int rangeColEndD,
	const int* bsrRowPtrC_coo, const int* bsrColIdxC, float* valueC, 
	int rowsPerBlockA, int colsPerBlockA, int rowsPerBlockB, int nnzC, float alpha, float beta
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

	// A*B
	float sum = 0.f;
	for (int i0 = blockBeginA, i1 = blockBeginB; i0 < blockEndA && i1 < blockEndB;)
	{
		int colBlockA = 0, colBlockB = 0;
		tex1Dfetch(&colBlockA, bsrColIdxA, i0);
		tex1Dfetch(&colBlockB, bsrColIdxB, i1);
		if (colBlockA >= rangeColEndA || colBlockB >= rangeColEndB)
			break;
		colBlockA -= rangeColBeginA;
		colBlockB -= rangeColBeginB;
		if (colBlockA == colBlockB && colBlockA >= 0)
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

		i0 += (colBlockA < colBlockB) || (colBlockA < 0);
		i1 += (colBlockA > colBlockB) || (colBlockB < 0);
	}// i


	// D
	float D_val = 0.f;
	if (bsrRowPtrD)
	{
		int blockBeginD = bsrRowPtrD[rowBlockC];
		int blockEndD = bsrRowPtrD[rowBlockC + 1];
		int colBlockD = 0;
		for (int c = blockBeginD; c < blockEndD && colBlockD < rangeColEndD; c++)
		{
			tex1Dfetch(&colBlockD, bsrColIdxD, c);
			if (colBlockD - rangeColBeginD == colBlockC)
			{
				tex1Dfetch(&D_val, valueD, (c * rowsPerBlockA + rowShiftC) * rowsPerBlockB + colShiftC);
				break;
			}
		}
	}// end if bsrRowPtrD

	valueC[innzC] = alpha * sum + beta * D_val;
}

__global__ void CudaBsrMatrix_Range_multBsrT_addDiag_value(
	const int* bsrRowPtrA, cudaTextureObject_t bsrColIdxA, cudaTextureObject_t valueA,
	int rangeColBeginA, int rangeColEndA,
	const int* bsrRowPtrB, cudaTextureObject_t bsrColIdxB, cudaTextureObject_t valueB,
	int rangeColBeginB, int rangeColEndB,
	cudaTextureObject_t valueD,
	const int* bsrRowPtrC_coo, const int* bsrColIdxC, float* valueC,
	int rowsPerBlockA, int colsPerBlockA, int rowsPerBlockB, int nnzC, float alpha, float beta
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

	// A*B
	float sum = 0.f;
	for (int i0 = blockBeginA, i1 = blockBeginB; i0 < blockEndA && i1 < blockEndB;)
	{
		int colBlockA = 0, colBlockB = 0;
		tex1Dfetch(&colBlockA, bsrColIdxA, i0);
		tex1Dfetch(&colBlockB, bsrColIdxB, i1);
		if (colBlockA >= rangeColEndA || colBlockB >= rangeColEndB)
			break;
		colBlockA -= rangeColBeginA;
		colBlockB -= rangeColBeginB;
		if (colBlockA == colBlockB && colBlockA >= 0)
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

		i0 += (colBlockA < colBlockB) || (colBlockA < 0);
		i1 += (colBlockA > colBlockB) || (colBlockB < 0);
	}// i


	// D
	float D_val = 0.f;
	if (valueD && rowBlockC == colBlockC)
		tex1Dfetch(&D_val, valueD, (rowBlockC * rowsPerBlockA + rowShiftC) * rowsPerBlockB + colShiftC);

	valueC[innzC] = alpha * sum + beta * D_val;
}

__global__ void CudaBsrMatrix_Range_AAt_blockDiags(
	const int* bsrRowPtrA, cudaTextureObject_t bsrColIdxA, cudaTextureObject_t valueA, 
	int rangeColBeginA, int rangeColEndA,
	float* diag, int rowsPerBlockA, int colsPerBlockA, int nnzDiag, 
	bool useLowerInsteadOfFull, float alpha, float beta
	)
{
	int innzDiag = threadIdx.x + blockIdx.x * blockDim.x;
	if (innzDiag >= nnzDiag)
		return;
	int blockDiagSz = rowsPerBlockA*rowsPerBlockA;
	int iBlockDiag = innzDiag / blockDiagSz;
	int shift = innzDiag - iBlockDiag*blockDiagSz;
	int rowShift = shift / colsPerBlockA;
	int colShift = shift - rowShift * colsPerBlockA;
	if (useLowerInsteadOfFull && rowShift < colShift)
		return;

	int row0 = bsrRowPtrA[iBlockDiag];

	int row0_begin = (row0*rowsPerBlockA + rowShift) * colsPerBlockA;
	const int row_blocks = bsrRowPtrA[iBlockDiag + 1] - row0;
	int row1_begin = (row0*rowsPerBlockA + colShift) * colsPerBlockA;

	int blockSzA = rowsPerBlockA * colsPerBlockA;
	float sum = 0;
	int colBlock = 0;
	for (int iBlocks = 0; iBlocks < row_blocks && colBlock < rangeColEndA; 
		iBlocks++, row0_begin += blockSzA, row1_begin += blockSzA)
	{
		tex1Dfetch(&colBlock, bsrColIdxA, row0 + iBlocks);
		if (colBlock < rangeColBeginA)
			continue;
		for (int i = 0; i < colsPerBlockA; i++)
		{
			float v1 = 0.f, v2 = 0.f;
			tex1Dfetch(&v1, valueA, row0_begin + i);
			tex1Dfetch(&v2, valueA, row1_begin + i);
			sum += v1 * v2;
		}
	}

	diag[innzDiag] = alpha*sum + beta*diag[innzDiag];
}

__global__ void CudaBsrMatrix_subRows_structure_rptr(const int* bsrRowPtrFull, int* bsrRowPtrSub,
	int rowBegin, int num)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i <= num)// <= because size=num+1 in bsr row
	{
		bsrRowPtrSub[i] = bsrRowPtrFull[i + rowBegin] - bsrRowPtrFull[rowBegin];
	}
}

__global__ void CudaBsrMatrix_subRows_structure_cidx(cudaTextureObject_t bsrRowPtrFull, 
	const int* bsrColIdxFull, int* bsrColIdxSub, int rowBegin, int nnzBlocks)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nnzBlocks)
	{
		int colBegin = 0;
		tex1Dfetch(&colBegin, bsrRowPtrFull, rowBegin);

		bsrColIdxSub[i] = bsrColIdxFull[i + colBegin];
	}
}

__global__ void CudaBsrMatrix_subRows_value(cudaTextureObject_t bsrRowPtrFull,
	const float* valueFull, float* valueSub, int rowBegin, int nnz, int blockSize2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nnz)
	{
		int nnzBegin = 0;
		tex1Dfetch(&nnzBegin, bsrRowPtrFull, rowBegin);
		nnzBegin *= blockSize2;

		valueSub[i] = valueFull[i + nnzBegin];
	}
}

__global__ void CudaBsrMatrix_toCsr_structure_rptr(cudaTextureObject_t bsrRowPtr,
	int* csrRowPtr, int bsrBlockRow, int bsrBlockCol, int nCsrRows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int ib = i / bsrBlockRow;
	if (i < nCsrRows)
	{
		int shift = i - ib*bsrBlockRow;
		int bsr = 0, bsr1=0;
		tex1Dfetch(&bsr, bsrRowPtr, ib);
		tex1Dfetch(&bsr1, bsrRowPtr, ib+1);
		csrRowPtr[i] = (bsr*bsrBlockRow + (bsr1-bsr)*shift) * bsrBlockCol;
	}
	if (i == nCsrRows)
	{
		int bsr = 0;
		tex1Dfetch(&bsr, bsrRowPtr, ib);
		csrRowPtr[i] = bsr*bsrBlockRow * bsrBlockCol;
	}
}

__global__ void CudaBsrMatrix_toCsr_structure_cidx(
	cudaTextureObject_t bsrRowPtr, cudaTextureObject_t bsrColIdx,
	const int* csrRowPtr_coo, const int* csrRowPtr, int* csrColIdx, 
	int bsrBlockRow, int bsrBlockCol, int nCsrNNZ)
{
	int innz = threadIdx.x + blockIdx.x * blockDim.x;
	if (innz < nCsrNNZ)
	{
		int iRow = csrRowPtr_coo[innz];
		int iBlockRow = iRow / bsrBlockRow;
		int colShiftOfRow = innz - csrRowPtr[iRow];
		int blockColShiftOfRow = colShiftOfRow / bsrBlockCol;

		int iBlock = 0;
		tex1Dfetch(&iBlock, bsrRowPtr, iBlockRow);
		iBlock += blockColShiftOfRow;

		int cshift = colShiftOfRow - blockColShiftOfRow * bsrBlockCol;

		int bc = 0;
		tex1Dfetch(&bc, bsrColIdx, iBlock);
		csrColIdx[innz] = bc * bsrBlockCol + cshift;
	}
}

__global__ void CudaBsrMatrix_toCsr_structure_val(
	cudaTextureObject_t bsrRowPtr, cudaTextureObject_t bsrValue,
	const int* csrRowPtr_coo, const int* csrRowPtr, float* csrValue,
	int bsrBlockRow, int bsrBlockCol, int nCsrNNZ)
{
	int innz = threadIdx.x + blockIdx.x * blockDim.x;
	if (innz < nCsrNNZ)
	{
		int iRow = csrRowPtr_coo[innz];
		int iBlockRow = iRow / bsrBlockRow;
		int colShiftOfRow = innz - csrRowPtr[iRow];
		int blockColShiftOfRow = colShiftOfRow / bsrBlockCol;

		int iBlock = 0;
		tex1Dfetch(&iBlock, bsrRowPtr, iBlockRow);
		iBlock += blockColShiftOfRow;

		int rshift = iRow - iBlockRow * bsrBlockRow;
		int cshift = colShiftOfRow - blockColShiftOfRow * bsrBlockCol;

		tex1Dfetch(&csrValue[innz], bsrValue, (iBlock*bsrBlockRow+rshift)*bsrBlockCol + cshift);
	}
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
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::transpose_fill_values_by_blockId(): symbolic matrix cannot touch values");
	CudaBsrMatrix_transpose_fill_value_by_bid << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		blockIds, t.value(), value(), rowsPerBlock() * colsPerBlock(), rowsPerBlock(), 
		colsPerBlock(), nnz());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::CudaBsrMatrix_transpose_fill_value_by_bid");
}

CudaBsrMatrix& CudaBsrMatrix::operator = (float constVal)
{
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::operator =: symbolic matrix cannot touch values");
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
	m_symbolic = rhs.m_symbolic;
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
	if (!isSymbolic())
		cudaSafeCall(cudaMemcpy(value(), rhs.value(),
			rhs.nnz()*sizeof(float), cudaMemcpyDeviceToDevice),
			"CudaBsrMatrix::operator =, cpy value");
	return *this;
}

CudaBsrMatrix& CudaBsrMatrix::axpy(float alpha, float beta)
{
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::axpy(): symbolic matrix cannot touch values");
	if (nnz() == 0)
		return *this;
	CudaBsrMatrix_scale_add << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		nnz(), value(), alpha, beta);
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::axpy");
	return *this;
}

CudaBsrMatrix& CudaBsrMatrix::axpy_diag(float alpha, float beta)
{
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::axpy_diag(): symbolic matrix cannot touch values");
	if (rowsPerBlock() != colsPerBlock() || blocksInRow() != blocksInCol())
		throw std::exception("CudaBsrMatrix::axpy_diag(): only square matrix supported");
	if (nnz() == 0)
		return *this;
	CudaBsrMatrix_scale_add_diag << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
		rows(), value(), bsrRowPtr(), bsrColIdxTexture(), rowsPerBlock(), alpha, beta);
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::axpy_diag");
	return *this;
}

void CudaBsrMatrix::setValue(const float* val_d)
{
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::setValue(): symbolic matrix cannot touch values");
	cudaSafeCall(cudaMemcpy(value(), val_d, nnz()*sizeof(float),
		cudaMemcpyDeviceToDevice), "CudaBsrMatrix::setValue");
}

void CudaBsrMatrix::Mv(const float* x, float* y, float alpha, float beta)const
{
	if (rows() == 0 || cols() == 0)
		return;
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::Mv(): symbolic matrix cannot touch values");
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
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::rightMultDiag_value(): symbolic matrix cannot touch values");
	if (cols() != x.rows())
		throw std::exception("CudaBsrMatrix::rightMultDiag_value: block size not matched");
	if (x.blockSize() != colsPerBlock() || x.blockSize() != rowsPerBlock())
		throw std::exception("CudaBsrMatrix::rightMultDiag_value: matrix size not matched");
	if (cols() != y.cols() || rows() != y.rows())
		throw std::exception("CudaBsrMatrix::rightMultDiag_value: y not matched, call rightMultDiag_structure()!");

	if (nnz() == 0)
		return;

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

void CudaBsrMatrix::Range::multBsr_value(const Range& B, CudaBsrMatrix& C, float alpha, 
	const Range* D, float beta)const
{
	throw std::exception("CudaBsrMatrix::Range::multBsr_value(): not implemented");
}

void CudaBsrMatrix::Range::multBsrT_value(const Range& B, CudaBsrMatrix& C, float alpha,
	const Range* D, float beta)const
{
	if (A == nullptr || B.A == nullptr)
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): null pointer exception");
	if (D)
	if (D->A == nullptr)
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): null pointer exception");
	if (A->isSymbolic() || B.A->isSymbolic())
		throw std::exception("CudaBsrMatrix::multBsrT_value(): symbolic matrix cannot touch values");
	if (D)
	{
		if (D->A->isSymbolic())
			throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): symbolic matrix cannot touch values");
		if (C.blocksInRow() != D->blocksInRow() || C.blocksInCol() != D->blocksInCol()
			|| C.rowsPerBlock() != D->rowsPerBlock() || C.colsPerBlock() != D->colsPerBlock())
			throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): D size not matched");
	}
	if (blocksInCol() != B.blocksInCol())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): matrix size not matched");
	if (colsPerBlock() != B.colsPerBlock())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): block size not matched");
	if (blocksInRow() != C.blocksInRow() || B.blocksInRow() != C.blocksInCol()
		|| rowsPerBlock() != C.rowsPerBlock() || B.rowsPerBlock() != C.colsPerBlock())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value(): result size not matched");
	if (C.nnzBlocks() == 0)
		return;

	const int* D_rptr = nullptr;
	cudaTextureObject_t D_cidx = 0, D_val = 0;
	int D_cbegin = 0, D_cend = 0;
	if (D)
	{
		D_rptr = D->A->bsrRowPtr() +D->blockRowBegin;
		D_cidx = D->A->bsrColIdxTexture();
		D_val = D->A->valueTexture();
		D_cbegin = D->blockColBegin;
		D_cend = D->blockColEnd;
	}

	CudaBsrMatrix_Range_multBsrT_value << <divUp(C.nnz(), CTA_SIZE), CTA_SIZE >> >(
		A->bsrRowPtr()+blockRowBegin, A->bsrColIdxTexture(), A->valueTexture(), 
		blockColBegin, blockColEnd,
		B.A->bsrRowPtr()+B.blockRowBegin, B.A->bsrColIdxTexture(), B.A->valueTexture(), 
		B.blockColBegin, B.blockColEnd,
		D_rptr, D_cidx, D_val, D_cbegin, D_cend,
		C.bsrRowPtr_coo(), C.bsrColIdx(), C.value(),
		rowsPerBlock(), colsPerBlock(), B.rowsPerBlock(), C.nnz(), alpha, beta
		);
}

void CudaBsrMatrix::Range::multBsrT_addDiag_value(const Range& B, CudaBsrMatrix& C, float alpha,
	const CudaDiagBlockMatrix* D, float beta)const
{
	if (A == nullptr || B.A == nullptr)
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value()1: null pointer exception");
	if (A->isSymbolic() || B.A->isSymbolic())
		throw std::exception("CudaBsrMatrix::multBsrT_value()1: symbolic matrix cannot touch values");
	if (blocksInCol() != B.blocksInCol())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value()1: matrix size not matched");
	if (colsPerBlock() != B.colsPerBlock())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value()1: block size not matched");
	if (blocksInRow() != C.blocksInRow() || B.blocksInRow() != C.blocksInCol()
		|| rowsPerBlock() != C.rowsPerBlock() || B.rowsPerBlock() != C.colsPerBlock())
		throw std::exception("CudaBsrMatrix::Range::multBsrT_value()1: result size not matched");
	if (D)
	{
		if (C.blocksInRow() != D->numBlocks() || C.blocksInCol() != D->numBlocks())
			throw std::exception("CudaBsrMatrix::Range::multBsrT_value()1: D size not matched");
		if (C.rowsPerBlock() != D->blockSize() || C.colsPerBlock() != D->blockSize())
			throw std::exception("CudaBsrMatrix::Range::multBsrT_value()1: D block not matched");
	}
	if (C.nnzBlocks() == 0)
		return;

	cudaTextureObject_t D_val = 0;
	if (D)
		D_val = D->getTexture();

	CudaBsrMatrix_Range_multBsrT_addDiag_value << <divUp(C.nnz(), CTA_SIZE), CTA_SIZE >> >(
		A->bsrRowPtr() + blockRowBegin, A->bsrColIdxTexture(), A->valueTexture(),
		blockColBegin, blockColEnd,
		B.A->bsrRowPtr() + B.blockRowBegin, B.A->bsrColIdxTexture(), B.A->valueTexture(),
		B.blockColBegin, B.blockColEnd,
		D_val,
		C.bsrRowPtr_coo(), C.bsrColIdx(), C.value(),
		rowsPerBlock(), colsPerBlock(), B.rowsPerBlock(), C.nnz(), alpha, beta
		);
}

void CudaBsrMatrix::Range::AAt_blockDiags(CudaDiagBlockMatrix& C,
	bool lowerInsteadOfFull, float alpha, float beta)const
{
	if (A == nullptr)
		throw std::exception("CudaBsrMatrix::Range::AAt_blockDiags(): null pointer exception");
	if (A->isSymbolic())
		throw std::exception("CudaBsrMatrix::AAt_blockDiags(): symbolic matrix cannot touch values");
	if (blocksInRow() != C.numBlocks())
		throw std::exception("CudaBsrMatrix::Range::AAt_blockDiags(): matrix size not matched");
	if (rowsPerBlock() != C.blockSize())
		throw std::exception("CudaBsrMatrix::Range::AAt_blockDiags(): block size not matched");
	if (A->nnzBlocks() == 0)
		return;


	CudaBsrMatrix_Range_AAt_blockDiags << <divUp(C.nnz(), CTA_SIZE), CTA_SIZE >> >(
		A->bsrRowPtr() + blockRowBegin, A->bsrColIdxTexture(), A->valueTexture(), 
		blockColBegin, blockColEnd,
		C.value(), rowsPerBlock(), colsPerBlock(), C.nnz(), 
		lowerInsteadOfFull, alpha, beta
		);
}

void CudaBsrMatrix::subRows_structure(CudaBsrMatrix& S, int blockRowBegin, int blockRowEnd)const
{
	blockRowBegin = std::max(0, blockRowBegin);
	blockRowEnd = std::min(blocksInRow(), blockRowEnd);
	if (blockRowBegin >= blockRowEnd)
	{
		S.resize(0, 0, rowsPerBlock(), colsPerBlock());
		return;
	}

	// rows
	S.resize(blockRowEnd - blockRowBegin, blocksInCol(), rowsPerBlock(), colsPerBlock());
	S.beginConstructRowPtr();
	CudaBsrMatrix_subRows_structure_rptr << <divUp(S.blocksInRow(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtr(), S.bsrRowPtr(), blockRowBegin, S.blocksInRow());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix_subRows_structure_rptr");
	S.endConstructRowPtr();

	// cols
	CudaBsrMatrix_subRows_structure_cidx << <divUp(S.nnzBlocks(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtrTexture(), bsrColIdx(), S.bsrColIdx(), blockRowBegin, S.nnzBlocks());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix_subRows_structure_cidx");
}

void CudaBsrMatrix::subRows_value(CudaBsrMatrix& S, int blockRowBegin, int blockRowEnd)const
{
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::AAt_blockDiags(): symbolic matrix cannot touch values");
	blockRowBegin = std::max(0, blockRowBegin);
	blockRowEnd = std::min(blocksInRow(), blockRowEnd);
	if (S.blocksInRow() != blockRowEnd - blockRowBegin ||
		S.blocksInCol() != blocksInCol() ||
		S.rowsPerBlock() != rowsPerBlock() ||
		S.colsPerBlock() != colsPerBlock())
		throw std::exception("CudaBsrMatrix::subRows_value: size not matched");

	CudaBsrMatrix_subRows_value << <divUp(S.nnz(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtrTexture(), value(), S.value(), blockRowBegin, S.nnz(),
		S.rowsPerBlock() * S.colsPerBlock());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix_subRows_value");

}

void CudaBsrMatrix::toCsr_structure(CudaBsrMatrix& B)const
{
	B.m_symbolic = isSymbolic();
	B.resize(rows(), cols(), 1, 1);
	B.resize_nnzBlocks(0);

	if (rows() == 0 || nnz() == 0)
		return;

	// 1. rptr
	B.beginConstructRowPtr();
	CudaBsrMatrix_toCsr_structure_rptr << <divUp(rows()+1, CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtrTexture(), B.bsrRowPtr(), rowsPerBlock(), colsPerBlock(), rows());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix_toCsr_structure_rptr");
	B.endConstructRowPtr(nnz());

	// 2. cidx
	CudaBsrMatrix_toCsr_structure_cidx << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtrTexture(), bsrColIdxTexture(),
		B.bsrRowPtr_coo(), B.bsrRowPtr(), B.bsrColIdx(),
		rowsPerBlock(), colsPerBlock(), nnz());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix_toCsr_structure_cidx");

}

void CudaBsrMatrix::toCsr_value(CudaBsrMatrix& B)const
{
	if (isSymbolic() || B.isSymbolic())
		throw std::exception("CudaBsrMatrix::toCsr_value(): symbolic cannot touch values");
	if (B.rows() != rows() || B.cols() != cols() || B.rowsPerBlock() != 1 || B.colsPerBlock() != 1)
		throw std::exception("CudaBsrMatrix::toCsr_value(): size of B not matched");
	if (rows() == 0 || nnz() == 0)
		return;

	CudaBsrMatrix_toCsr_structure_val << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		bsrRowPtrTexture(), valueTexture(),
		B.bsrRowPtr_coo(), B.bsrRowPtr(), B.value(),
		rowsPerBlock(), colsPerBlock(), nnz());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix_toCsr_structure_cidx");
}