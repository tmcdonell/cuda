/*
 * Extra bits for CUDA binding
 */

#include "cbits/stubs.h"


cudaError_t
cudaConfigureCallSimple
(
    int gridX,  int gridY,
    int blockX, int blockY, int blockZ,
    size_t sharedMem,
    cudaStream_t stream
)
{
    dim3 gridDim  = {gridX, gridY, 1};
    dim3 blockDim = {blockX,blockY,blockZ};

    return cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

const char*
cudaGetErrorStringWrapper(cudaError_t error)
{
    return cudaGetErrorString(error);
}

CUresult
cuTexRefSetAddress2DSimple
(
    CUtexref tex,
    CUarray_format format,
    unsigned int numChannels,
    CUdeviceptr dptr,
    size_t width,
    size_t height,
    size_t pitch
)
{
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format      = format;
    desc.NumChannels = numChannels;
    desc.Width       = width;
    desc.Height      = height;

    return cuTexRefSetAddress2D(tex, &desc, dptr, pitch);
}

CUresult
cuMemcpy2DHtoD
(
    CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY,
    void* srcHost,         unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY,
    unsigned int widthInBytes,
    unsigned int height
)
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

CUresult
cuMemcpy2DHtoDAsync
(
    CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY,
    void* srcHost,         unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY,
    unsigned int widthInBytes,
    unsigned int height,
    CUstream hStream
)
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

CUresult
cuMemcpy2DDtoH
(
    void* dstHost,         unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY,
    CUdeviceptr srcDevice, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY,
    unsigned int widthInBytes,
    unsigned int height
)
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

CUresult
cuMemcpy2DDtoHAsync
(
    void* dstHost,         unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY,
    CUdeviceptr srcDevice, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY,
    unsigned int widthInBytes,
    unsigned int height,
    CUstream hStream
)
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

CUresult
cuMemcpy2DDtoD
(
    CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY,
    CUdeviceptr srcDevice, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY,
    unsigned int widthInBytes,
    unsigned int height
)
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

CUresult
cuMemcpy2DDtoDAsync
(
    CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int dstXInBytes, unsigned int dstY,
    CUdeviceptr srcDevice, unsigned int srcPitch, unsigned int srcXInBytes, unsigned int srcY,
    unsigned int widthInBytes,
    unsigned int height,
    CUstream hStream
)
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

#if CUDA_VERSION >= 6050
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags)
{
    return cuMemHostRegister_v2(p, bytesize, Flags);
}
#endif

