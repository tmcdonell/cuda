/*
 * Extra bits for CUDA binding
 */

#include "cbits/stubs.h"

#if CUDART_VERSION >= 7000
cudaError_t cudaLaunchKernel_simple(const void *func, unsigned int gridX, unsigned int gridY, unsigned int gridZ, unsigned int blockX, unsigned int blockY, unsigned int blockZ, void **args, size_t sharedMem, cudaStream_t stream)
{
    dim3 gridDim  = {gridX, gridY, gridZ};
    dim3 blockDim = {blockX, blockY, blockZ};

    return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
#else
cudaError_t cudaConfigureCall_simple(unsigned int gridX, unsigned int gridY, unsigned int blockX, unsigned int blockY, unsigned int blockZ, size_t sharedMem, cudaStream_t stream)
{
    dim3 gridDim  = {gridX, gridY, 1};
    dim3 blockDim = {blockX,blockY,blockZ};

    return cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}
#endif
