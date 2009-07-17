/*
 * Test output of device properties
 */

#include <stdio.h>
#include <cuda_runtime_api.h>

int main(int argc, char **argv)
{
    struct cudaDeviceProp d;

    cudaGetDeviceProperties(&d, 0);
    printf("name = %s\n", d.name);
    printf("computeCapability = (%d,%d)\n", d.major, d.minor);
    printf("totalGlobalMem = %lu\n", d.totalGlobalMem);
    printf("totalConstMem = %ld\n", d.totalConstMem);
    printf("sharedMemPerBlock = %ld\n", d.sharedMemPerBlock);
    printf("regsPerBlock = %d\n", d.regsPerBlock);
    printf("warpSize = %d\n", d.warpSize);
    printf("maxThreadsPerBlock = %d\n", d.maxThreadsPerBlock);
    printf("maxThreadsDim = (%d,%d,%d)\n", d.maxThreadsDim[0], d.maxThreadsDim[1], d.maxThreadsDim[2]);
    printf("maxGridSize = (%d,%d,%d)\n", d.maxGridSize[0], d.maxGridSize[1], d.maxGridSize[2]);
    printf("clockRate = %d\n", d.clockRate);
    printf("multiProcessorCount = %d\n", d.multiProcessorCount);
    printf("memPitch = %ld\n", d.memPitch);
    printf("textureAlignment = %ld\n", d.textureAlignment);
    printf("computeMode = %d\n", d.computeMode);
    printf("deviceOverlap = %d\n", d.deviceOverlap);
    printf("kernelExecTimeoutEnabled = %d\n", d.kernelExecTimeoutEnabled);
    printf("integrated = %d\n", d.integrated);
    printf("canMapHostMemory = %d\n", d.canMapHostMemory);

    return 0;
}
