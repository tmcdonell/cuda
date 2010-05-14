--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Device
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Common device functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Device
  where

#include <cuda.h>

import Data.Int


-- |
-- The compute mode the device is currently in
--
{# enum CUcomputemode as ComputeMode { }
    with prefix="CU_COMPUTEMODE" deriving (Eq, Show) #}

-- |
-- The properties of a compute device
--
data DeviceProperties = DeviceProperties
  {
    deviceName               :: String,         -- ^ Identifier
    computeCapability        :: Double,         -- ^ Supported compute capability
    totalGlobalMem           :: Int64,          -- ^ Available global memory on the device in bytes
    totalConstMem            :: Int64,          -- ^ Available constant memory on the device in bytes
    sharedMemPerBlock        :: Int64,          -- ^ Available shared memory per block in bytes
    regsPerBlock             :: Int,            -- ^ 32-bit registers per block
    warpSize                 :: Int,            -- ^ Warp size in threads (SIMD width)
    maxThreadsPerBlock       :: Int,            -- ^ Max number of threads per block
    maxBlockSize             :: (Int,Int,Int),  -- ^ Max size of each dimension of a block
    maxGridSize              :: (Int,Int,Int),  -- ^ Max size of each dimension of a grid
#if CUDA_VERSION >= 3000
    maxTextureDim1D          :: Int,            -- ^ Maximum texture dimensions
    maxTextureDim2D          :: (Int,Int),
    maxTextureDim3D          :: (Int,Int,Int),
#endif
    clockRate                :: Int,            -- ^ Clock frequency in kilohertz
    multiProcessorCount      :: Int,            -- ^ Number of multiprocessors on the device
    memPitch                 :: Int64,          -- ^ Max pitch in bytes allowed by memory copies
    textureAlignment         :: Int64,          -- ^ Alignment requirement for textures
    computeMode              :: ComputeMode,
    deviceOverlap            :: Bool,           -- ^ Device can concurrently copy memory and execute a kernel
#if CUDA_VERSION >= 3000
    concurrentKernels        :: Bool,           -- ^ Device can possibly execute multiple kernels concurrently
    eccEnabled               :: Bool,           -- ^ Device supports and has enabled error correction
#endif
    kernelExecTimeoutEnabled :: Bool,           -- ^ Whether there is a runtime limit on kernels
    integrated               :: Bool,           -- ^ As opposed to discrete
    canMapHostMemory         :: Bool            -- ^ Device can use pinned memory
  }
  deriving (Show)

