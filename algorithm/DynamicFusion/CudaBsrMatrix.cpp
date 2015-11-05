#include "CudaBsrMatrix.h"
#include "cudpp\ModerGpuWrapper.h"

static void cusparseCheck(cusparseStatus_t st, const char* msg = nullptr)
{
	if (CUSPARSE_STATUS_SUCCESS != st)
	{
		printf("cusparse error[%d]: %s", st, msg);
		throw std::exception(msg);
	}
}

CudaBsrMatrix::CudaBsrMatrix(cusparseHandle_t handle, bool symbolic)
{
	m_cusparseHandle = handle;
	m_symbolic = symbolic;
	m_blocksInRow = 0;
	m_blocksInCol = 0;
	m_rowsPerBlock = 0;
	m_colsPerBlock = 0;
	m_nnzBlocks = 0;

	m_tex_values = 0;
	m_tex_bsrRowPtr = 0;
	m_tex_bsrRowPtr_coo = 0;
	m_tex_bsrColIdx = 0;
	cusparseCheck(cusparseCreateMatDescr(&m_desc));
	cusparseCheck(cusparseCreateCsrgemm2Info(&m_csrgemm2info));
}

CudaBsrMatrix::~CudaBsrMatrix()
{
	clear();
	cusparseCheck(cusparseDestroyMatDescr(m_desc));
	cusparseCheck(cusparseDestroyCsrgemm2Info(m_csrgemm2info));
}

void CudaBsrMatrix::clear()
{
	m_cusparseHandle = nullptr;
	m_blocksInRow = 0;
	m_blocksInCol = 0;
	m_rowsPerBlock = 0;
	m_colsPerBlock = 0;
	m_nnzBlocks = 0;

	if (m_tex_values)
		cudaSafeCall(cudaDestroyTextureObject(m_tex_values), "CudaBsrMatrix::clear(), destroy tex 1");
	if (m_tex_bsrRowPtr)
		cudaSafeCall(cudaDestroyTextureObject(m_tex_bsrRowPtr), "CudaBsrMatrix::clear(), destroy tex 2");
	if (m_tex_bsrRowPtr_coo)
		cudaSafeCall(cudaDestroyTextureObject(m_tex_bsrRowPtr_coo), "CudaBsrMatrix::clear(), destroy tex 3");
	if (m_tex_bsrColIdx)
		cudaSafeCall(cudaDestroyTextureObject(m_tex_bsrColIdx), "CudaBsrMatrix::clear(), destroy tex 4");
	m_tex_values = 0;
	m_tex_bsrRowPtr = 0;
	m_tex_bsrRowPtr_coo = 0;
	m_tex_bsrColIdx = 0;

	m_bsrRowPtr.release();
	m_bsrColIdx.release();
	m_bsrRowPtr_coo.release();
	m_values.release();
	m_helperBuffer.release();
}

template<class T>
static void bindLinearTex(T* ptr, int sizeBytes, cudaTextureObject_t& old)
{
	if (old)
		cudaSafeCall(cudaDestroyTextureObject(old), "CudaBsrMatrix::bindTexture::Destory");
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = ptr;
	texRes.res.linear.sizeInBytes = sizeBytes;
	texRes.res.linear.desc = cudaCreateChannelDesc<T>();
	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = 0;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;
	cudaSafeCall(cudaCreateTextureObject(&old, &texRes, &texDescr, NULL),
		"CudaBsrMatrix::bindTexture");
}

void CudaBsrMatrix::resize(int blocksInRow, int blocksInCol, int rowsPerBlock, int colsPerBlock)
{
	m_blocksInRow = blocksInRow;
	m_blocksInCol = blocksInCol;
	m_rowsPerBlock = rowsPerBlock;
	m_colsPerBlock = colsPerBlock;

	if (m_blocksInRow + 1 > m_bsrRowPtr.size())
	{
		m_bsrRowPtr.create((m_blocksInRow + 1) * 1.2);
		bindLinearTex(m_bsrRowPtr.ptr(), m_bsrRowPtr.sizeBytes(), m_tex_bsrRowPtr);
	}
}

void CudaBsrMatrix::resize_nnzBlocks(int nnzBlocks)
{
	m_nnzBlocks = nnzBlocks;
	if (m_nnzBlocks > m_bsrColIdx.size())
	{
		if (!m_symbolic)
			m_values.create(nnz()*1.2);
		m_bsrColIdx.create(m_nnzBlocks*1.2);
		m_bsrRowPtr_coo.create(m_nnzBlocks*1.2);
		if (!m_symbolic)
			bindLinearTex(m_values.ptr(), m_values.sizeBytes(), m_tex_values);
		bindLinearTex(m_bsrColIdx.ptr(), m_bsrColIdx.sizeBytes(), m_tex_bsrColIdx);
		bindLinearTex(m_bsrRowPtr_coo.ptr(), m_bsrRowPtr_coo.sizeBytes(), m_tex_bsrRowPtr_coo);
	}
}

void CudaBsrMatrix::resize(int blocksInRow, int blocksInCol, int sizePerBlock)
{
	resize(blocksInRow, blocksInCol, sizePerBlock, sizePerBlock);
}

void CudaBsrMatrix::beginConstructRowPtr()
{
	if (m_blocksInRow == 0)
		return;
}

void CudaBsrMatrix::endConstructRowPtr()
{
	if (m_blocksInRow == 0)
		return;

	int nnzBlocks = 0;
	cudaSafeCall(cudaMemcpy(&nnzBlocks, bsrRowPtr() + blocksInRow(), sizeof(int),
		cudaMemcpyDeviceToHost), "CudaBsrMatrix::endConstructRowPtr: copy nnz");

	endConstructRowPtr(nnzBlocks);
}

void CudaBsrMatrix::endConstructRowPtr(int nnzBlocks)
{
	resize_nnzBlocks(nnzBlocks);
	cusparseCheck(cusparseXcsr2coo(m_cusparseHandle, bsrRowPtr(), nnzBlocks,
		blocksInRow(), bsrRowPtr_coo(), CUSPARSE_INDEX_BASE_ZERO));
}

void CudaBsrMatrix::transposeStructureTo(CudaBsrMatrix& rhs)const
{
	rhs.m_cusparseHandle = m_cusparseHandle;
	rhs.resize(blocksInCol(), blocksInRow(), colsPerBlock(), rowsPerBlock());
	rhs.resize_nnzBlocks(nnzBlocks());
	if (nnzBlocks() == 0)
		return;
	cudaSafeCall(cudaMemcpy(rhs.bsrRowPtr_coo(), bsrColIdx(), nnzBlocks()*sizeof(int), 
		cudaMemcpyDeviceToDevice), "CudaBsrMatrix::transposeStructureTo, 1");
	cudaSafeCall(cudaMemcpy(rhs.bsrColIdx(), bsrRowPtr_coo(), nnzBlocks()*sizeof(int),
		cudaMemcpyDeviceToDevice), "CudaBsrMatrix::transposeStructureTo, 2");

	modergpu_wrapper::mergesort_by_key(rhs.bsrRowPtr_coo(), rhs.bsrColIdx(), rhs.nnzBlocks());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::transposeStructureTo 3");
	cusparseCheck(cusparseXcoo2csr(m_cusparseHandle,
		rhs.bsrRowPtr_coo(), rhs.nnzBlocks(), rhs.blocksInRow(),
		rhs.bsrRowPtr(), CUSPARSE_INDEX_BASE_ZERO));
}

void CudaBsrMatrix::transposeValueTo(CudaBsrMatrix& rhs)const
{
	if (isSymbolic() || rhs.isSymbolic())
		throw std::exception("CudaBsrMatrix::transposeValueTo(): symbolic matrix cannot touch values!");
	rhs.m_cusparseHandle = m_cusparseHandle;
	rhs.resize(m_blocksInCol, m_blocksInRow, m_colsPerBlock, m_rowsPerBlock);
	rhs.resize_nnzBlocks(m_nnzBlocks);
	if (nnzBlocks() == 0)
		return;
	cudaSafeCall(cudaMemcpy(rhs.bsrRowPtr_coo(), bsrColIdx(), nnzBlocks()*sizeof(int),
		cudaMemcpyDeviceToDevice), "CudaBsrMatrix::transposeValueTo, 1");

	int *hptr = (int*)get_helper_buffer(m_bsrColIdx.sizeBytes());
	fill_increment_1_n(hptr, m_nnzBlocks);
	modergpu_wrapper::mergesort_by_key(rhs.bsrRowPtr_coo(), hptr, rhs.nnzBlocks());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::transposeValueTo 2");

	rhs.transpose_fill_values_by_blockId(hptr, *this);
}

void CudaBsrMatrix::setRowFromBsrRowPtr(const int* bsrRowPtr)
{
	if (blocksInRow() == 0)
		return;
	beginConstructRowPtr();
	cudaSafeCall(cudaMemcpy(m_bsrRowPtr, bsrRowPtr, (1+blocksInRow())*m_bsrRowPtr.elem_size,
		cudaMemcpyDeviceToDevice));
	endConstructRowPtr();
}

void CudaBsrMatrix::fromCsr(const int* csrRowPtr, const int* csrColIdx, const float* csrValue)
{
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::fromCsr(): symbolic matrix cannot touch values!");
	if (blocksInRow() == 0)
		return;

	int bufferSizeInBytes = 0;
	cusparseCheck(cusparseScsr2gebsr_bufferSize(m_cusparseHandle, CUSPARSE_DIRECTION_ROW, rows(), cols(),
		m_desc, csrValue, csrRowPtr, csrColIdx, rowsPerBlock(), colsPerBlock(), &bufferSizeInBytes));
	char* hptr = get_helper_buffer(bufferSizeInBytes);

	// 1. rows
	beginConstructRowPtr();
	cusparseCheck(cusparseXcsr2gebsrNnz(m_cusparseHandle, CUSPARSE_DIRECTION_ROW, rows(), cols(), m_desc,
		csrRowPtr, csrColIdx, m_desc, bsrRowPtr(), rowsPerBlock(), colsPerBlock(), 
		&m_nnzBlocks, hptr));
	endConstructRowPtr(m_nnzBlocks);

	// 2. cols & values
	cusparseCheck(cusparseScsr2gebsr(m_cusparseHandle, CUSPARSE_DIRECTION_ROW, rows(), cols(), m_desc,
		csrValue, csrRowPtr, csrColIdx, m_desc, value(), bsrRowPtr(), bsrColIdx(),
		rowsPerBlock(), colsPerBlock(), hptr));
}

void CudaBsrMatrix::toCsr(DeviceArray<int>& csrRowPtr, DeviceArray<int>& csrColIdx, 
	DeviceArray<float>& csrValue)const
{
	if (isSymbolic())
		throw std::exception("CudaBsrMatrix::toCsr(): symbolic cannot touch values");
	if (csrRowPtr.size() < rows() + 1)
		csrRowPtr.create(rows() + 1);
	if (csrColIdx.size() < nnz())
		csrColIdx.create(nnz());
	if (csrValue.size() < nnz())
		csrValue.create(nnz());
	cusparseCheck(cusparseSgebsr2csr(m_cusparseHandle, CUSPARSE_DIRECTION_ROW, blocksInRow(), 
		blocksInCol(), m_desc, value(), bsrRowPtr(), bsrColIdx(), rowsPerBlock(), colsPerBlock(), 
		m_desc, csrValue, csrRowPtr, csrColIdx));
}

void CudaBsrMatrix::multBsr_structure(const CudaBsrMatrix& B, CudaBsrMatrix& C, const CudaBsrMatrix* D)const
{
	range().multBsr_structure(B, C, D);
}

void CudaBsrMatrix::multBsr_value(const CudaBsrMatrix& B, CudaBsrMatrix& C, float alpha,
	const CudaBsrMatrix* D, float beta)const
{
	range().multBsr_value(B.range(), C, alpha, D == nullptr ? nullptr : &D->range(), beta);
}

void CudaBsrMatrix::multBsrT_value(const CudaBsrMatrix& B, CudaBsrMatrix& C, float alpha,
	const CudaBsrMatrix* D, float beta)const
{
	range().multBsrT_value(B.range(), C, alpha, D==nullptr ? nullptr : &D->range(), beta);
}

void CudaBsrMatrix::multBsrT_addDiag_value(const CudaBsrMatrix& B, CudaBsrMatrix& C, float alpha,
	const CudaDiagBlockMatrix* D, float beta)const
{
	range().multBsrT_addDiag_value(B.range(), C, alpha, D, beta);
}

void CudaBsrMatrix::AAt_blockDiags(CudaDiagBlockMatrix& C, bool lowerInsteadOfFull,
	float alpha, float beta)const
{
	range().AAt_blockDiags(C, lowerInsteadOfFull, alpha, beta);
}

void CudaBsrMatrix::dump(std::string name)const
{
	std::vector<int> bhr, bhc;
	std::vector<float> hv;

	m_bsrRowPtr.download(bhr);
	m_bsrColIdx.download(bhc);
	if (!isSymbolic())
		m_values.download(hv);

	FILE* pFile = fopen(name.c_str(), "w");
	if (pFile)
	{
		for (int br = 0; br < m_blocksInRow; br++)
		{
			int rowbegin = br*m_rowsPerBlock;
			int bcb = bhr[br], bce = bhr[br + 1];
			for (int bic = bcb; bic < bce; bic++)
			{
				int colbegin = bhc[bic] * m_colsPerBlock;
				int valbegin = bic * m_colsPerBlock * m_rowsPerBlock;
				for (int r = 0; r < m_rowsPerBlock; r++)
				for (int c = 0; c < m_colsPerBlock; c++)
				{
					if (!isSymbolic())
						fprintf(pFile, "%d %d %ef\n", rowbegin + r, colbegin + c, hv[valbegin++]);
					else
						fprintf(pFile, "%d %d %ef\n", rowbegin + r, colbegin + c, 1.f);
				}
			}
			// if an empty row, fill diag with zero
			// this is for the convinience when exporting to matlab
			if (bcb == bce)
			for (int r = 0; r < m_rowsPerBlock; r++)
				fprintf(pFile, "%d %d %ef\n", rowbegin + r, rowbegin + r, 0);
		}
		fclose(pFile);
	}
}

void CudaBsrMatrix::Range::multBsr_structure(const CudaBsrMatrix& B, 
	CudaBsrMatrix& C, const CudaBsrMatrix* D)const
{
	if (A == nullptr)
		throw std::exception("CudaBsrMatrix::Range::multBsr_structure(): nullpointer exception");
	if (blockColBegin != 0 || blockColEnd != A->blocksInCol()
		|| blockRowBegin != 0 || blockRowEnd != A->blocksInRow())
		throw std::exception("CudaBsrMatrix::Range::multBsr_structure(): ranges not supported now");
	if (cols() != B.rows())
		throw std::exception("CudaBsrMatrix::Range::multBsr_structure(): matrix size not matched");
	if (colsPerBlock() != B.rowsPerBlock())
		throw std::exception("CudaBsrMatrix::Range::multBsr_structure(): block size not matched");
	if (D)
	{
		if (A->rowsPerBlock() != D->rowsPerBlock() || B.colsPerBlock() != D->colsPerBlock()
			|| A->blocksInRow() != D->blocksInRow() || B.blocksInCol() != D->blocksInCol())
			throw std::exception("CudaBsrMatrix::Range::multBsr_structure(): D size not matched!");
	}

	C.resize(blocksInRow(), B.blocksInCol(), rowsPerBlock(), B.colsPerBlock());

	// 0. working buffer
	float alpha = 1.f, beta = 1.f;
	float* pAlpha = &alpha, *pBeta = nullptr;
	int nnzD = 0;
	const int* D_rptr = nullptr, *D_cidx = nullptr;
	const float* D_val = nullptr;
	if (D)
	{
		pBeta = &beta;
		nnzD = D->nnzBlocks();
		D_rptr = D->bsrRowPtr();
		D_cidx = D->bsrColIdx();
	}

	size_t nBufferBytes = 0;
	cusparseCheck( cusparseScsrgemm2_bufferSizeExt(A->m_cusparseHandle,
		blocksInRow(), B.blocksInCol(), blocksInCol(), pAlpha,
		A->m_desc, A->nnzBlocks(), A->bsrRowPtr(), A->bsrColIdx(),
		B.m_desc, B.nnzBlocks(), B.bsrRowPtr(), B.bsrColIdx(),
		pBeta, B.m_desc, nnzD, D_rptr, D_cidx, A->m_csrgemm2info, &nBufferBytes));

	char* workBufer = A->get_helper_buffer(nBufferBytes);

	// 1. construct rows
	C.beginConstructRowPtr();

	int cnnz = 0;
	cusparseCheck(cusparseXcsrgemm2Nnz(A->m_cusparseHandle, 
		blocksInRow(), B.blocksInCol(), blocksInCol(),
		A->m_desc, A->nnzBlocks(), A->bsrRowPtr(), A->bsrColIdx(),
		B.m_desc, B.nnzBlocks(), B.bsrRowPtr(), B.bsrColIdx(),
		B.m_desc, nnzD, D_rptr, D_cidx,
		C.m_desc, C.bsrRowPtr(), &cnnz, A->m_csrgemm2info, workBufer));

	C.endConstructRowPtr(cnnz);

	// 2. construct cols
	// NOTE: cusparse calculates values together with colIdx
	// here we only want colIdx, the values calculated here is invalid since we use bsr format
	cusparseCheck(cusparseScsrgemm2(A->m_cusparseHandle, 
		blocksInRow(), B.blocksInCol(), blocksInCol(), pAlpha,
		A->m_desc, A->nnzBlocks(), nullptr, A->bsrRowPtr(), A->bsrColIdx(),
		B.m_desc, B.nnzBlocks(), nullptr, B.bsrRowPtr(), B.bsrColIdx(), pBeta,
		B.m_desc, nnzD, nullptr, D_rptr, D_cidx,
		C.m_desc, nullptr, C.bsrRowPtr(), C.bsrColIdx(), A->m_csrgemm2info, workBufer));
}

char* CudaBsrMatrix::get_helper_buffer(int nBytes)const
{
	if (m_helperBuffer.sizeBytes() < nBytes)
		m_helperBuffer.create(nBytes*1.2);
	return (char*)m_helperBuffer.ptr();
}


