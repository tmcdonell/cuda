/* -----------------------------------------------------------------------------
 *
 * Module    : SMVM
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/

#ifndef __SMVM_H__
#define __SMVM_H__

/*
 * Optimised for Tesla C1060 (compute 1.3)
 * Maximum performance for your card may be achieved with different values.
 *
 * http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls
 */
#define MAX_THREADS             128
#define MAX_BLOCKS_PER_SM       8
#define MAX_BLOCKS              (MAX_BLOCKS_PER_SM * 30)
#define WARP_SIZE               32

#ifdef __cplusplus
extern "C" {
#endif

void smvm_csr_f(float *d_y, float *d_x, float *d_data, unsigned int *d_rowPtr, unsigned int *d_colIdx, unsigned int num_rows);
void smvm_cudpp_f(float *d_y, float *d_x, float *h_data, unsigned int *h_rowPtr, unsigned int *h_colIdx, unsigned int num_rows, unsigned int num_nonzeros);

#ifdef __cplusplus
}
#endif
#endif
