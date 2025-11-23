/*
 * Extra bits for CUDA bindings
 */

#ifndef C_STUBS_H
#define C_STUBS_H

#ifdef __MINGW32__
#include <host_defines.h>
#undef CUDARTAPI
#define CUDARTAPI __stdcall
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#if CUDART_VERSION >= 7000
cudaError_t cudaLaunchKernel_simple(const void *func, unsigned int gridX, unsigned int gridY, unsigned int gridZ, unsigned int blockX, unsigned int blockY, unsigned int blockZ, void **args, size_t sharedMem, cudaStream_t stream);
#else
cudaError_t cudaConfigureCall_simple(unsigned int gridX, unsigned int gridY, unsigned int blockX, unsigned int blockY, unsigned int blockZ, size_t sharedMem, cudaStream_t stream);
#endif

#ifdef __cplusplus
}
#endif
#endif
