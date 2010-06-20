/*
 * Extra bits for CUDA bindings
 */

#ifndef C_STUBS_H
#define C_STUBS_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t
cudaConfigureCallSimple(int gx, int gy, int bx, int by, int bz, size_t sharedMem, cudaStream_t stream);

const char*
cudaGetErrorStringWrapper(cudaError_t error);

CUresult
cuTexRefSetAddress2DSimple(CUtexref tex, CUarray_format fmt, int chn, CUdeviceptr dptr, int width, int height, int pitch);

#ifdef __cplusplus
}
#endif
#endif
