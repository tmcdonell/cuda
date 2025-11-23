/* -----------------------------------------------------------------------------
 *
 * Module    : Control
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/

#include "utils.h"
#include <cuda_runtime_api.h>

typedef void* KernelPointer;

/*
 * Stolen from the RadixSort example from the CUDA SDK.
 */

void
computeNumCTAs(KernelPointer kernel, int smemDynamicBytes, bool bManualCoalesce)
{
    int                 deviceID = -1;
    cudaDeviceProp      devprop;
    cudaFuncAttributes  attr;

    assert(cudaSuccess == cudaGetDevice(&deviceID));
    assert(cudaSuccess == cudaGetDeviceProperties(&devprop, deviceID));
    assert(cudaSuccess == cudaFuncGetAttributes(&attr, (const char*)kernel));

    /*
     * Determine the maximum number of CTAs that can be run simultaneously for
     * each kernel. This is equivalent to the calculation done in the CUDA
     * Occupancy Calculator spreadsheet
     */
    const unsigned int regAllocationUnit      = (devprop.major < 2 && devprop.minor < 2) ? 256 : 512; // in registers
    const unsigned int warpAllocationMultiple = 2;
    const unsigned int smemAllocationUnit     = 512;                                                  // in bytes
    const unsigned int maxThreadsPerSM        = bManualCoalesce ? 768 : 1024;                         // sm_12 GPUs increase threads/SM to 1024
    const unsigned int maxBlocksPerSM         = 8;

    size_t numWarps;
    numWarps = multiple(RadixSort::CTA_SIZE, devprop.warpSize); // Number of warps (round up to nearest whole multiple of warp size)
    numWarps = ceiling(numWarps, warpAllocationMultiple);       // Round up to warp allocation multiple

    size_t regsPerCTA;
    regsPerCTA = attr.numRegs * devprop.warpSize * numWarps;    // Number of regs is (regs per thread * number of warps * warp size)
    regsPerCTA = ceiling(regsPerCTA, regAllocationUnit);        // Round up to multiple of register allocation unit size

    size_t smemBytes = attr.sharedSizeBytes + smemDynamicBytes;
    size_t smemPerCTA = ceiling(smemBytes, smemAllocationUnit);

    size_t ctaLimitRegs    = regsPerCTA > 0 ? devprop.regsPerBlock      / regsPerCTA : maxBlocksPerSM;
    size_t ctaLimitSMem    = smemPerCTA > 0 ? devprop.sharedMemPerBlock / smemPerCTA : maxBlocksPerSM;
    size_t ctaLimitThreads =                  maxThreadsPerSM           / RadixSort::CTA_SIZE;

    unsigned int numSMs  = devprop.multiProcessorCount;
    unsigned int maxCTAs = numSMs * std::min<size_t>(ctaLimitRegs, std::min<size_t>(ctaLimitSMem, std::min<size_t>(ctaLimitThreads, maxBlocksPerSM)));
//    setNumCTAs(kernel, maxCTAs);
}


