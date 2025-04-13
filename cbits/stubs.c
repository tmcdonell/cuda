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

CUresult cuMemcpy2DHtoD(CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY, void* srcHost, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY, unsigned int widthInBytes, unsigned int height)
{
    CUDA_MEMCPY2D desc;

    // source array (host)
    // srcStart = (void*)((char*)srcHost + srcY * srcPitch + srcXInBytes)
    desc.srcMemoryType  = CU_MEMORYTYPE_HOST;
    desc.srcHost        = srcHost;
    desc.srcXInBytes    = srcXInBytes;
    desc.srcY           = srcY;
    desc.srcPitch       = srcPitch;

    // destination array (device)
    // dstStart = dstDevice + dstY * dstPitch + dstXInBytes
    desc.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    desc.dstDevice      = dstDevice;
    desc.dstXInBytes    = dstXInBytes;
    desc.dstY           = dstY;
    desc.dstPitch       = dstPitch;

    // transfer description
    desc.WidthInBytes   = widthInBytes;
    desc.Height         = height;

    return cuMemcpy2D(&desc);
}

CUresult cuMemcpy2DHtoDAsync(CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY, void* srcHost, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY, unsigned int widthInBytes, unsigned int height, CUstream hStream)
{
    CUDA_MEMCPY2D desc;

    // source array (host)
    // srcStart = (void*)((char*)srcHost + srcY * srcPitch + srcXInBytes)
    desc.srcMemoryType  = CU_MEMORYTYPE_HOST;
    desc.srcHost        = srcHost;
    desc.srcXInBytes    = srcXInBytes;
    desc.srcY           = srcY;
    desc.srcPitch       = srcPitch;

    // destination array (device)
    // dstStart = dstDevice + dstY * dstPitch + dstXInBytes
    desc.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    desc.dstDevice      = dstDevice;
    desc.dstXInBytes    = dstXInBytes;
    desc.dstY           = dstY;
    desc.dstPitch       = dstPitch;

    // transfer description
    desc.WidthInBytes   = widthInBytes;
    desc.Height         = height;

    return cuMemcpy2DAsync(&desc, hStream);
}

CUresult cuMemcpy2DDtoH(void* dstHost, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY, CUdeviceptr srcDevice, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY, unsigned int widthInBytes, unsigned int height)
{
    CUDA_MEMCPY2D desc;

    // source array (device)
    // srcStart = srcDevice + srcY * srcPitch + srcXInBytes
    desc.srcMemoryType  = CU_MEMORYTYPE_DEVICE;
    desc.srcDevice      = srcDevice;
    desc.srcXInBytes    = srcXInBytes;
    desc.srcY           = srcY;
    desc.srcPitch       = srcPitch;

    // destination array (host)
    // dstStart = (void*)((char*)dstHost + dstY * dstPitch + dstXInBytes)
    desc.dstMemoryType  = CU_MEMORYTYPE_HOST;
    desc.dstHost        = dstHost;
    desc.dstXInBytes    = dstXInBytes;
    desc.dstY           = dstY;
    desc.dstPitch       = dstPitch;

    // transfer description
    desc.WidthInBytes   = widthInBytes;
    desc.Height         = height;

    return cuMemcpy2D(&desc);
}

CUresult cuMemcpy2DDtoHAsync(void* dstHost, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY, CUdeviceptr srcDevice, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY, unsigned int widthInBytes, unsigned int height, CUstream hStream)
{
    CUDA_MEMCPY2D desc;

    // source array (device)
    // srcStart = srcDevice + srcY * srcPitch + srcXInBytes
    desc.srcMemoryType  = CU_MEMORYTYPE_DEVICE;
    desc.srcDevice      = srcDevice;
    desc.srcXInBytes    = srcXInBytes;
    desc.srcY           = srcY;
    desc.srcPitch       = srcPitch;

    // destination array (host)
    // dstStart = (void*)((char*)dstHost + dstY * dstPitch + dstXInBytes)
    desc.dstMemoryType  = CU_MEMORYTYPE_HOST;
    desc.dstHost        = dstHost;
    desc.dstXInBytes    = dstXInBytes;
    desc.dstY           = dstY;
    desc.dstPitch       = dstPitch;

    // transfer description
    desc.WidthInBytes   = widthInBytes;
    desc.Height         = height;

    return cuMemcpy2DAsync(&desc, hStream);
}

CUresult cuMemcpy2DDtoD(CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY, CUdeviceptr srcDevice, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY, unsigned int widthInBytes, unsigned int height)
{
    CUDA_MEMCPY2D desc;

    // source array (device)
    // srcStart = srcDevice + srcY * srcPitch + srcXInBytes
    desc.srcMemoryType  = CU_MEMORYTYPE_DEVICE;
    desc.srcDevice      = srcDevice;
    desc.srcXInBytes    = srcXInBytes;
    desc.srcY           = srcY;
    desc.srcPitch       = srcPitch;

    // destination array (device)
    // dstStart = dstDevice + dstY * dstPitch + dstXInBytes
    desc.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    desc.dstDevice      = dstDevice;
    desc.dstXInBytes    = dstXInBytes;
    desc.dstY           = dstY;
    desc.dstPitch       = dstPitch;

    // transfer description
    desc.WidthInBytes   = widthInBytes;
    desc.Height         = height;

    return cuMemcpy2D(&desc);
}

CUresult cuMemcpy2DDtoDAsync(CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY, CUdeviceptr srcDevice, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY, unsigned int widthInBytes, unsigned int height, CUstream hStream)
{
    CUDA_MEMCPY2D desc;

    // source array (device)
    // srcStart = srcDevice + srcY * srcPitch + srcXInBytes
    desc.srcMemoryType  = CU_MEMORYTYPE_DEVICE;
    desc.srcDevice      = srcDevice;
    desc.srcXInBytes    = srcXInBytes;
    desc.srcY           = srcY;
    desc.srcPitch       = srcPitch;

    // destination array (device)
    // dstStart = dstDevice + dstY * dstPitch + dstXInBytes
    desc.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    desc.dstDevice      = dstDevice;
    desc.dstXInBytes    = dstXInBytes;
    desc.dstY           = dstY;
    desc.dstPitch       = dstPitch;

    // transfer description
    desc.WidthInBytes   = widthInBytes;
    desc.Height         = height;

    return cuMemcpy2DAsync(&desc, hStream);
}


#if CUDA_VERSION >= 3020
/*
 * Extra exports for CUDA-3.2
 */
CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
    return cuDeviceTotalMem_v2(bytes, dev);
}

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    return cuCtxCreate_v2(pctx, flags, dev);
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name)
{
    return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr)
{
    return cuMemFree_v2(dptr);
}

CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total)
{
    return cuMemGetInfo_v2(free, total);
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr)
{
    return cuMemGetAddressRange_v2(pbase, psize, dptr);
}

CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags)
{
    return cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    return cuMemsetD16_v2(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    return cuMemsetD32_v2(dstDevice, ui, N);
}
#endif

#if CUDA_VERSION >= 4000
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx)
{
    return cuCtxDestroy_v2(ctx);
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx)
{
    return cuCtxPopCurrent_v2(pctx);
}

CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx)
{
    return cuCtxPushCurrent_v2(ctx);
}

CUresult CUDAAPI cuStreamDestroy(CUstream hStream)
{
    return cuStreamDestroy_v2(hStream);
}

CUresult CUDAAPI cuEventDestroy(CUevent hEvent)
{
    return cuEventDestroy_v2(hEvent);
}
#endif

#if CUDA_VERSION >= 6050
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags)
{
    return cuMemHostRegister_v2(p, bytesize, Flags);
}

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut)
{
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    return cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
}
#endif

#if CUDA_VERSION >= 10000
CUresult CUDAAPI cuGraphAddHostNode_simple(CUgraphNode *phGraphNode, CUgraph hGraph, CUgraphNode *dependencies, size_t numDependencies, CUhostFn fn, void* userData)
{
    CUDA_HOST_NODE_PARAMS nodeParams;
    nodeParams.fn       = fn;
    nodeParams.userData = userData;

    return cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, &nodeParams);
}

CUresult CUDAAPI cuGraphAddKernelNode_simple(CUgraphNode *phGraphNode, CUgraph hGraph, CUgraphNode *dependencies, size_t numDependencies, CUfunction func, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, void** kernelParams)
{
    CUDA_KERNEL_NODE_PARAMS nodeParams;
    nodeParams.func           = func;
    nodeParams.gridDimX       = gridDimX;
    nodeParams.gridDimY       = gridDimY;
    nodeParams.gridDimZ       = gridDimZ;
    nodeParams.blockDimX      = blockDimX;
    nodeParams.blockDimY      = blockDimY;
    nodeParams.blockDimZ      = blockDimZ;
    nodeParams.sharedMemBytes = sharedMemBytes;
    nodeParams.kernelParams   = kernelParams;
    nodeParams.extra          = NULL;

    return cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, &nodeParams);
}

CUresult CUDAAPI cuGraphAddMemcpyNode_simple(CUgraphNode *phGraphNode, CUgraph hGraph, CUgraphNode *dependencies, size_t numDependencies, CUcontext ctx, size_t srcXInBytes, size_t srcY, size_t srcZ, size_t srcLOD, CUmemorytype srcMemoryType, const void *srcPtr, size_t srcPitch, size_t srcHeight, size_t dstXInBytes, size_t dstY, size_t dstZ, size_t dstLOD, CUmemorytype dstMemoryType, void *dstPtr, size_t dstPitch, size_t dstHeight, size_t widthInBytes, size_t height, size_t depth)
{
    CUDA_MEMCPY3D copyParams;

    copyParams.srcXInBytes   = srcXInBytes;
    copyParams.srcY          = srcY;
    copyParams.srcZ          = srcZ;
    copyParams.srcLOD        = srcLOD;
    copyParams.srcMemoryType = srcMemoryType;
    copyParams.srcHost       = (void*) srcPtr;
    copyParams.srcDevice     = (CUdeviceptr) srcPtr;
    copyParams.srcArray      = (CUarray) srcPtr;
    copyParams.reserved0     = NULL;
    copyParams.srcPitch      = srcPitch;
    copyParams.srcHeight     = srcHeight;

    copyParams.dstXInBytes   = dstXInBytes;
    copyParams.dstY          = dstY;
    copyParams.dstZ          = dstZ;
    copyParams.dstLOD        = dstLOD;
    copyParams.dstMemoryType = dstMemoryType;
    copyParams.dstHost       = (void*) dstPtr;
    copyParams.dstDevice     = (CUdeviceptr) dstPtr;
    copyParams.dstArray      = (CUarray) dstPtr;
    copyParams.reserved1     = NULL;
    copyParams.dstPitch      = dstPitch;
    copyParams.dstHeight     = dstHeight;

    copyParams.WidthInBytes  = widthInBytes;
    copyParams.Height        = height;
    copyParams.Depth         = depth;

    return cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, &copyParams, ctx);
}

CUresult CUDAAPI cuGraphAddMemsetNode_simple(CUgraphNode *phGraphNode, CUgraph hGraph, CUgraphNode *dependencies, size_t numDependencies, CUcontext ctx, CUdeviceptr dst, unsigned int elementSize, size_t height, size_t pitch, unsigned int value, size_t width)
{
    CUDA_MEMSET_NODE_PARAMS memsetParams;
    memsetParams.dst         = dst;
    memsetParams.elementSize = elementSize;
    memsetParams.height      = height;
    memsetParams.pitch       = pitch;
    memsetParams.value       = value;
    memsetParams.width       = width;

    return cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, &memsetParams, ctx);
}
#endif

#if CUDA_VERSION >= 10010
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode)
{
    return cuStreamBeginCapture_v2(hStream, mode);
}

CUresult CUDAAPI cuGraphExecKernelNodeSetParams_simple(CUgraphExec hGraphExec, CUgraphNode hNode, CUfunction func, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, void** kernelParams)
{
    CUDA_KERNEL_NODE_PARAMS nodeParams;
    nodeParams.func           = func;
    nodeParams.gridDimX       = gridDimX;
    nodeParams.gridDimY       = gridDimY;
    nodeParams.gridDimZ       = gridDimZ;
    nodeParams.blockDimX      = blockDimX;
    nodeParams.blockDimY      = blockDimY;
    nodeParams.blockDimZ      = blockDimZ;
    nodeParams.sharedMemBytes = sharedMemBytes;
    nodeParams.kernelParams   = kernelParams;
    nodeParams.extra          = NULL;

    return cuGraphExecKernelNodeSetParams(hGraphExec, hNode, &nodeParams);
}
#endif

