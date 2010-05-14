--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Analysis.Device
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Common device functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Analysis.Device
  (
    ComputeMode(..), DeviceProperties(..), DeviceResources(..), Allocation(..),
    resources
  )
  where

#include <cuda.h>

import Data.Int
import Data.Maybe


-- |
-- The compute mode the device is currently in
--
{# enum CUcomputemode as ComputeMode
    { underscoreToCase }
    with prefix="CU_COMPUTEMODE" deriving (Eq, Show) #}

-- |
-- GPU compute capability
--
type Compute = Double

-- |
-- The properties of a compute device
--
data DeviceProperties = DeviceProperties
  {
    deviceName               :: String,         -- ^ Identifier
    computeCapability        :: Compute,        -- ^ Supported compute capability
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


-- GPU Hardware Resources
--
data Allocation      = Warp | Block
data DeviceResources = DeviceResources
  {
    threadsPerWarp     :: Int,          -- ^ Warp size
    threadsPerMP       :: Int,          -- ^ Maximum number of in-flight threads on a multiprocessor
    threadBlocksPerMP  :: Int,          -- ^ Maximum number of thread blocks resident on a multiprocessor
    warpsPerMP         :: Int,          -- ^ Maximum number of in-flight warps per multiprocessor
    sharedMemPerMP     :: Int,          -- ^ Total amount of shared memory per multiprocessor (bytes)
    sharedMemAllocUnit :: Int,          -- ^ Shared memory allocation unit size (bytes)
    regFileSize        :: Int,          -- ^ Total number of registers in a multiprocessor
    regAllocUnit       :: Int,          -- ^ Register allocation unit size
    regAllocWarp       :: Int,          -- ^ Register allocation granularity for warps
    allocation         :: Allocation    -- ^ How multiprocessor resources are divided
  }


-- |
-- Extract some additional hardware resource limitations for a given device
--
resources :: DeviceProperties -> DeviceResources
resources dev = fromMaybe err (lookup compute gpuData)
  where
    compute = computeCapability dev
    err     = error $  "No data for device: "
                    ++ shows (deviceName dev) " (compute " ++ shows compute ")"

    gpuData =
      [(1.0, DeviceResources 32  768 8 24 16384 512  8192 256 2 Block)
      ,(1.1, DeviceResources 32  768 8 24 16384 512  8192 256 2 Block)
      ,(1.2, DeviceResources 32 1024 8 32 16384 512 16384 512 2 Block)
      ,(1.3, DeviceResources 32 1024 8 32 16384 512 16384 512 2 Block)
      ,(2.0, DeviceResources 32 1536 8 48 49152 128 32768  64 1 Warp)
      ]

