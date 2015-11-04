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

CudaBsrMatrix::CudaBsrMatrix(cusparseHandle_t handle)
{
	m_cusparseHandle = handle;
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
}

CudaBsrMatrix::~CudaBsrMatrix()
{
	clear();
	cusparseCheck(cusparseDestroyMatDescr(m_desc));
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
		m_values.create(nnz()*1.2);
		m_bsrColIdx.create(m_nnzBlocks*1.2);
		m_bsrRowPtr_coo.create(m_nnzBlocks*1.2);
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
	rhs.m_cusparseHandle = m_cusparseHandle;
	rhs.resize(m_blocksInCol, m_blocksInRow, m_colsPerBlock, m_rowsPerBlock);
	rhs.resize_nnzBlocks(m_nnzBlocks);
	cudaSafeCall(cudaMemcpy(rhs.bsrRowPtr_coo(), bsrColIdx(), nnzBlocks()*sizeof(int),
		cudaMemcpyDeviceToDevice), "CudaBsrMatrix::transposeValueTo, 1");

	if (m_helperBuffer.size() < m_bsrColIdx.size())
		m_helperBuffer.create(m_bsrColIdx.size());
	fill_increment_1_n(m_helperBuffer.ptr(), m_nnzBlocks);
	modergpu_wrapper::mergesort_by_key(rhs.bsrRowPtr_coo(), m_helperBuffer.ptr(), rhs.nnzBlocks());
	cudaSafeCall(cudaGetLastError(), "CudaBsrMatrix::transposeValueTo 2");

	rhs.transpose_fill_values_by_blockId(m_helperBuffer.ptr(), *this);
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
	if (blocksInRow() == 0)
		return;

	int bufferSizeInBytes = 0;
	cusparseCheck(cusparseScsr2gebsr_bufferSize(m_cusparseHandle, CUSPARSE_DIRECTION_ROW, rows(), cols(),
		m_desc, csrValue, csrRowPtr, csrColIdx, rowsPerBlock(), colsPerBlock(), &bufferSizeInBytes));
	if (bufferSizeInBytes > m_helperBuffer.sizeBytes())
		m_helperBuffer.create(bufferSizeInBytes / m_helperBuffer.elem_size);

	// 1. rows
	beginConstructRowPtr();
	cusparseCheck(cusparseXcsr2gebsrNnz(m_cusparseHandle, CUSPARSE_DIRECTION_ROW, rows(), cols(), m_desc,
		csrRowPtr, csrColIdx, m_desc, bsrRowPtr(), rowsPerBlock(), colsPerBlock(), 
		&m_nnzBlocks, m_helperBuffer.ptr()));
	endConstructRowPtr(m_nnzBlocks);

	// 2. cols & values
	cusparseCheck(cusparseScsr2gebsr(m_cusparseHandle, CUSPARSE_DIRECTION_ROW, rows(), cols(), m_desc,
		csrValue, csrRowPtr, csrColIdx, m_desc, value(), bsrRowPtr(), bsrColIdx(),
		rowsPerBlock(), colsPerBlock(), m_helperBuffer.ptr()));
}

void CudaBsrMatrix::toCsr(DeviceArray<int>& csrRowPtr, DeviceArray<int>& csrColIdx, DeviceArray<float>& csrValue)
{
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

void CudaBsrMatrix::multBsr_structure(const CudaBsrMatrix& B, CudaBsrMatrix& C)
{
	if (cols() != B.rows())
		throw std::exception("CudaBsrMatrix::multBsr_structure(): matrix size not matched");
	if (colsPerBlock() != B.rowsPerBlock())
		throw std::exception("CudaBsrMatrix::multBsr_structure(): block size not matched");

	C.resize(blocksInRow(), B.blocksInCol(), rowsPerBlock(), B.colsPerBlock());

	// 1. construct rows
	C.beginConstructRowPtr();

	int cnnz = 0;
	cusparseCheck( cusparseXcsrgemmNnz(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		blocksInRow(), B.blocksInCol(), blocksInCol(),
		m_desc, nnzBlocks(), bsrRowPtr(), bsrColIdx(),
		B.m_desc, B.nnzBlocks(), B.bsrRowPtr(), B.bsrColIdx(),
		C.m_desc, C.bsrRowPtr(), &cnnz) );

	C.endConstructRowPtr(cnnz);

	// 2. construct cols
	// NOTE: cusparse calculates values together with colIdx
	// here we only want colIdx, the values calculated here is invalid since we use bsr format
	cusparseCheck(cusparseScsrgemm(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		blocksInRow(), B.blocksInCol(), blocksInCol(),
		m_desc, nnzBlocks(), value(), bsrRowPtr(), bsrColIdx(),
		B.m_desc, B.nnzBlocks(), B.value(), B.bsrRowPtr(), B.bsrColIdx(),
		C.m_desc, C.value(), C.bsrRowPtr(), C.bsrColIdx()));
}

void CudaBsrMatrix::multBsr_value(const CudaBsrMatrix& B, CudaBsrMatrix& C, float alpha)
{
	range().multBsr_value(B.range(), C, alpha);
}

void CudaBsrMatrix::multBsrT_value(const CudaBsrMatrix& B, CudaBsrMatrix& C, float alpha)
{
	range().multBsrT_value(B.range(), C, alpha);
}

void CudaBsrMatrix::dump(std::string name)const
{
	std::vector<int> bhr, bhc;
	std::vector<float> hv;

	m_bsrRowPtr.download(bhr);
	m_bsrColIdx.download(bhc);
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
					fprintf(pFile, "%d %d %ef\n", rowbegin + r, colbegin + c, hv[valbegin++]);
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

