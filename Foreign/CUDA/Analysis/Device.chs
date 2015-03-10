--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Analysis.Device
-- Copyright : [2009..2014] Trevor L. McDonell
-- License   : BSD
--
-- Common device functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Analysis.Device
  (
    Compute(..), ComputeMode(..),
    DeviceProperties(..), DeviceResources(..), Allocation(..), PCI(..),
    deviceResources
  )
  where

#include "cbits/stubs.h"

import Data.Int
import Debug.Trace


-- |
-- The compute mode the device is currently in
--
{# enum CUcomputemode as ComputeMode
    { underscoreToCase }
    with prefix="CU_COMPUTEMODE" deriving (Eq, Show) #}

-- |
-- GPU compute capability, major and minor revision number respectively.
--
data Compute = Compute !Int !Int
  deriving Eq

instance Show Compute where
  show (Compute major minor) = show major ++ "." ++ show minor

instance Ord Compute where
  compare (Compute m1 n1) (Compute m2 n2) =
    case compare m1 m2 of
      EQ -> compare n1 n2
      x  -> x

{--
cap :: Int -> Int -> Double
cap a 0 = fromIntegral a
cap a b = let a' = fromIntegral a in
            let b' = fromIntegral b in
            a' + b' / max 10 (10^ ((ceiling . logBase 10) b' :: Int))
--}

-- |
-- The properties of a compute device
--
data DeviceProperties = DeviceProperties
  {
    deviceName                  :: !String,             -- ^ Identifier
    computeCapability           :: !Compute,            -- ^ Supported compute capability
    totalGlobalMem              :: !Int64,              -- ^ Available global memory on the device in bytes
    totalConstMem               :: !Int64,              -- ^ Available constant memory on the device in bytes
    sharedMemPerBlock           :: !Int64,              -- ^ Available shared memory per block in bytes
    regsPerBlock                :: !Int,                -- ^ 32-bit registers per block
    warpSize                    :: !Int,                -- ^ Warp size in threads (SIMD width)
    maxThreadsPerBlock          :: !Int,                -- ^ Max number of threads per block
#if CUDA_VERSION >= 4000
    maxThreadsPerMultiProcessor :: !Int,                -- ^ Max number of threads per multiprocessor
#endif
    maxBlockSize                :: !(Int,Int,Int),      -- ^ Max size of each dimension of a block
    maxGridSize                 :: !(Int,Int,Int),      -- ^ Max size of each dimension of a grid
#if CUDA_VERSION >= 3000
    maxTextureDim1D             :: !Int,                -- ^ Maximum texture dimensions
    maxTextureDim2D             :: !(Int,Int),
    maxTextureDim3D             :: !(Int,Int,Int),
#endif
    clockRate                   :: !Int,                -- ^ Clock frequency in kilohertz
    multiProcessorCount         :: !Int,                -- ^ Number of multiprocessors on the device
    memPitch                    :: !Int64,              -- ^ Max pitch in bytes allowed by memory copies
#if CUDA_VERSION >= 4000
    memBusWidth                 :: !Int,                -- ^ Global memory bus width in bits
    memClockRate                :: !Int,                -- ^ Peak memory clock frequency in kilohertz
#endif
    textureAlignment            :: !Int64,              -- ^ Alignment requirement for textures
    computeMode                 :: !ComputeMode,
    deviceOverlap               :: !Bool,               -- ^ Device can concurrently copy memory and execute a kernel
#if CUDA_VERSION >= 3000
    concurrentKernels           :: !Bool,               -- ^ Device can possibly execute multiple kernels concurrently
    eccEnabled                  :: !Bool,               -- ^ Device supports and has enabled error correction
#endif
#if CUDA_VERSION >= 4000
    asyncEngineCount            :: !Int,                -- ^ Number of asynchronous engines
    cacheMemL2                  :: !Int,                -- ^ Size of the L2 cache in bytes
    tccDriverEnabled            :: !Bool,               -- ^ Whether this is a Tesla device using the TCC driver
    pciInfo                     :: !PCI,                -- ^ PCI device information for the device
#endif
    kernelExecTimeoutEnabled    :: !Bool,               -- ^ Whether there is a runtime limit on kernels
    integrated                  :: !Bool,               -- ^ As opposed to discrete
    canMapHostMemory            :: !Bool,               -- ^ Device can use pinned memory
#if CUDA_VERSION >= 4000
    unifiedAddressing           :: !Bool                -- ^ Device shares a unified address space with the host
#endif
  }
  deriving (Show)


data PCI = PCI
  {
    busID       :: !Int,                -- ^ PCI bus ID of the device
    deviceID    :: !Int,                -- ^ PCI device ID
    domainID    :: !Int                 -- ^ PCI domain ID
  }
  deriving (Show)


-- GPU Hardware Resources
--
data Allocation      = Warp | Block
data DeviceResources = DeviceResources
  {
    threadsPerWarp      :: !Int,        -- ^ Warp size
    threadsPerMP        :: !Int,        -- ^ Maximum number of in-flight threads on a multiprocessor
    threadBlocksPerMP   :: !Int,        -- ^ Maximum number of thread blocks resident on a multiprocessor
    warpsPerMP          :: !Int,        -- ^ Maximum number of in-flight warps per multiprocessor
    coresPerMP          :: !Int,        -- ^ Number of SIMD arithmetic units per multiprocessor
    sharedMemPerMP      :: !Int,        -- ^ Total amount of shared memory per multiprocessor (bytes)
    sharedMemAllocUnit  :: !Int,        -- ^ Shared memory allocation unit size (bytes)
    regFileSize         :: !Int,        -- ^ Total number of registers in a multiprocessor
    regAllocUnit        :: !Int,        -- ^ Register allocation unit size
    regAllocWarp        :: !Int,        -- ^ Register allocation granularity for warps
    regPerThread        :: !Int,        -- ^ Maximum number of registers per thread
    allocation          :: !Allocation  -- ^ How multiprocessor resources are divided
  }


-- |
-- Extract some additional hardware resource limitations for a given device.
--
deviceResources :: DeviceProperties -> DeviceResources
deviceResources = resources . computeCapability
  where
    -- This is mostly extracted from tables in the CUDA occupancy calculator.
    --
    resources compute = case compute of
      Compute 1 0 -> DeviceResources 32  768  8 24   8  16384 512   8192 256 2 124 Block  -- Tesla G80
      Compute 1 1 -> DeviceResources 32  768  8 24   8  16384 512   8192 256 2 124 Block  -- Tesla G8x
      Compute 1 2 -> DeviceResources 32 1024  8 32   8  16384 512  16384 512 2 124 Block  -- Tesla G9x
      Compute 1 3 -> DeviceResources 32 1024  8 32   8  16384 512  16384 512 2 124 Block  -- Tesla GT200
      Compute 2 0 -> DeviceResources 32 1536  8 48  32  49152 128  32768  64 2  63 Warp   -- Fermi GF100
      Compute 2 1 -> DeviceResources 32 1536  8 48  48  49152 128  32768  64 2  63 Warp   -- Fermi GF10x
      Compute 3 0 -> DeviceResources 32 2048 16 64 192  49152 256  65536 256 4  63 Warp   -- Kepler GK10x
      Compute 3 2 -> DeviceResources 32 2048 16 64 192  49152 256  65536 256 4 255 Warp   -- Jetson TK1 (speculative)
      Compute 3 5 -> DeviceResources 32 2048 16 64 192  49152 256  65536 256 4 255 Warp   -- Kepler GK11x
      Compute 3 7 -> DeviceResources 32 2048 16 64 192 114688 256 131072 256 4 266 Warp   -- Kepler GK21x
      Compute 5 0 -> DeviceResources 32 2048 32 64 128  65536 256  65536 256 4 255 Warp   -- Maxwell GM10x
      Compute 5 2 -> DeviceResources 32 2048 32 64 128  98304 256  65536 256 4 255 Warp   -- Maxwell GM20x

      -- Something might have gone wrong, or the library just needs to be
      -- updated for the next generation of hardware, in which case we just want
      -- to pick a sensible default and carry on.
      --
      -- This is slightly dodgy as the warning message is coming from pure code.
      -- However, it should be OK because all library functions run in IO, so it
      -- is likely the user code is as well.
      --
      _           -> trace warning $ resources (Compute 3 0)
        where warning = unlines [ "*** Warning: Unknown CUDA device compute capability: " ++ show compute
                                , "*** Please submit a bug report at https://github.com/tmcdonell/cuda/issues" ]

