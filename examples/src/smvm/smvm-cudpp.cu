/* -----------------------------------------------------------------------------
 *
 * Module    : SMVM
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/


#include "smvm.h"
#include <cudpp.h>

template <typename T> CUDPPDatatype getType();
template <> CUDPPDatatype getType<float>() { return CUDPP_FLOAT; }
template <> CUDPPDatatype getType<unsigned int>() { return CUDPP_UINT; }


/*
 * Sparse matrix-dense vector multiply. Hook directly into the CUDPP
 * implementation.
 */
template <typename T>
void smvm_cudpp
(
    float               *d_y,
    const float         *d_x,
    const float         *h_data,
    const unsigned int  *h_rowPtr,
    const unsigned int  *h_colIdx,
    const unsigned int  num_rows,
    const unsigned int  num_nonzeros
)
{
    CUDPPConfiguration cp;
    CUDPPHandle        sm;

    cp.datatype  = getType<T>();
    cp.options   = 0;
    cp.algorithm = CUDPP_SPMVMULT;

    cudppSparseMatrix(&sm, cp, num_nonzeros, num_rows, h_data, h_rowPtr, h_colIdx);
    cudppSparseMatrixVectorMultiply(sm, d_y, d_x);

    cudppDestroySparseMatrix(sm);
}


// -----------------------------------------------------------------------------
// Instances
// -----------------------------------------------------------------------------

void smvm_cudpp_f(float *d_y, float *d_x, float *h_data, unsigned int *h_rowPtr, unsigned int *h_colIdx, unsigned int num_rows, unsigned int num_nonzeros)
{
    smvm_cudpp<float>(d_y, d_x, h_data, h_rowPtr, h_colIdx, num_rows, num_nonzeros);
}

