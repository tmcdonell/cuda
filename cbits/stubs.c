/*
 * Extra bits for CUDA binding
 */

#include "cbits/stubs.h"


cudaError_t
cudaConfigureCallSimple(int gx, int gy, int bx, int by, int bz, size_t sharedMem, cudaStream_t stream)
{
    dim3 gridDim  = {gx,gy,0};
    dim3 blockDim = {bx,by,bz};

    return cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

