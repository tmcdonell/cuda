/*
 * Extra bits for CUDA binding
 */

#include "cbits/stubs.h"


cudaError_t
cudaConfigureCallSimple(int gx, int gy, int bx, int by, int bz, size_t sharedMem, cudaStream_t stream)
{
    dim3 gridDim  = {gx,gy,1};
    dim3 blockDim = {bx,by,bz};

    return cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

const char*
cudaGetErrorStringWrapper(cudaError_t error)
{
    return cudaGetErrorString(error);
}

CUresult
cuTexRefSetAddress2DSimple(CUtexref tex, CUarray_format fmt, int chn, CUdeviceptr dptr, int width, int height, int pitch)
{
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format      = fmt;
    desc.NumChannels = chn;
    desc.Width       = width;
    desc.Height      = height;

    return cuTexRefSetAddress2D(tex, &desc, dptr, pitch);
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

CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)
{
    return cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);
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
