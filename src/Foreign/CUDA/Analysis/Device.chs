--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Analysis.Device
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- Common device functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Analysis.Device (

    Compute(..), ComputeMode(..),
    DeviceProperties(..), DeviceResources(..), Allocation(..), PCI(..),
    deviceResources,
    describe

) where

#include "cbits/stubs.h"

import Data.Int
import Text.Show.Describe

import Debug.Trace


-- |
-- The compute mode the device is currently in
--
{# enum CUcomputemode as ComputeMode
    { underscoreToCase }
    with prefix="CU_COMPUTEMODE" deriving (Eq, Show) #}

instance Describe ComputeMode where
  describe Default          = "Multiple contexts are allowed on the device simultaneously"
#if CUDA_VERSION < 8000
  describe Exclusive        = "Only one context used by a single thread can be present on this device at a time"
#endif
  describe Prohibited       = "No contexts can be created on this device at this time"
  describe ExclusiveProcess = "Only one context used by a single process can be present on this device at a time"


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
    deviceName                    :: !String          -- ^ Identifier
  , computeCapability             :: !Compute         -- ^ Supported compute capability
  , totalGlobalMem                :: !Int64           -- ^ Available global memory on the device in bytes
  , totalConstMem                 :: !Int64           -- ^ Available constant memory on the device in bytes
  , sharedMemPerBlock             :: !Int64           -- ^ Available shared memory per block in bytes
  , regsPerBlock                  :: !Int             -- ^ 32-bit registers per block
  , warpSize                      :: !Int             -- ^ Warp size in threads (SIMD width)
  , maxThreadsPerBlock            :: !Int             -- ^ Maximum number of threads per block
#if CUDA_VERSION >= 4000
  , maxThreadsPerMultiProcessor   :: !Int             -- ^ Maximum number of threads per multiprocessor
#endif
  , maxBlockSize                  :: !(Int,Int,Int)   -- ^ Maximum size of each dimension of a block
  , maxGridSize                   :: !(Int,Int,Int)   -- ^ Maximum size of each dimension of a grid
#if CUDA_VERSION >= 3000
  , maxTextureDim1D               :: !Int             -- ^ Maximum texture dimensions
  , maxTextureDim2D               :: !(Int,Int)
  , maxTextureDim3D               :: !(Int,Int,Int)
#endif
  , clockRate                     :: !Int             -- ^ Clock frequency in kilohertz
  , multiProcessorCount           :: !Int             -- ^ Number of multiprocessors on the device
  , memPitch                      :: !Int64           -- ^ Maximum pitch in bytes allowed by memory copies
#if CUDA_VERSION >= 4000
  , memBusWidth                   :: !Int             -- ^ Global memory bus width in bits
  , memClockRate                  :: !Int             -- ^ Peak memory clock frequency in kilohertz
#endif
  , textureAlignment              :: !Int64           -- ^ Alignment requirement for textures
  , computeMode                   :: !ComputeMode
  , deviceOverlap                 :: !Bool            -- ^ Device can concurrently copy memory and execute a kernel
#if CUDA_VERSION >= 3000
  , concurrentKernels             :: !Bool            -- ^ Device can possibly execute multiple kernels concurrently
  , eccEnabled                    :: !Bool            -- ^ Device supports and has enabled error correction
#endif
#if CUDA_VERSION >= 4000
  , asyncEngineCount              :: !Int             -- ^ Number of asynchronous engines
  , cacheMemL2                    :: !Int             -- ^ Size of the L2 cache in bytes
  , pciInfo                       :: !PCI             -- ^ PCI device information for the device
  , tccDriverEnabled              :: !Bool            -- ^ Whether this is a Tesla device using the TCC driver
#endif
  , kernelExecTimeoutEnabled      :: !Bool            -- ^ Whether there is a runtime limit on kernels
  , integrated                    :: !Bool            -- ^ As opposed to discrete
  , canMapHostMemory              :: !Bool            -- ^ Device can use pinned memory
#if CUDA_VERSION >= 4000
  , unifiedAddressing             :: !Bool            -- ^ Device shares a unified address space with the host
#endif
#if CUDA_VERSION >= 5050
  , streamPriorities              :: !Bool            -- ^ Device supports stream priorities
#endif
#if CUDA_VERSION >= 6000
  , globalL1Cache                 :: !Bool            -- ^ Device supports caching globals in L1 cache
  , localL1Cache                  :: !Bool            -- ^ Device supports caching locals in L1 cache
  , managedMemory                 :: !Bool            -- ^ Device supports allocating managed memory on this system
  , multiGPUBoard                 :: !Bool            -- ^ Device is on a multi-GPU board
  , multiGPUBoardGroupID          :: !Int             -- ^ Unique identifier for a group of devices associated with the same board
#endif
#if CUDA_VERSION >= 8000
  , preemption                    :: !Bool            -- ^ Device supports compute pre-emption
  , singleToDoublePerfRatio       :: !Int             -- ^ Ratio of single precision performance (in floating-point operations per second) to double precision performance
#endif
#if CUDA_VERSION >= 9000
  , cooperativeLaunch             :: !Bool            -- ^ Device supports launching cooperative kernels
  , cooperativeLaunchMultiDevice  :: !Bool            -- ^ Device can participate in cooperative multi-device kernels
#endif
  }
  deriving (Show)


data PCI = PCI
  {
    busID       :: !Int,      -- ^ PCI bus ID of the device
    deviceID    :: !Int,      -- ^ PCI device ID
    domainID    :: !Int       -- ^ PCI domain ID
  }
  deriving (Show)


-- GPU Hardware Resources
--
-- These are either taken from the CUDA occupancy calculator, or the CUDA
-- wikipedia entry: <https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications>
--
data Allocation      = Warp | Block
  deriving Show

data DeviceResources = DeviceResources
  { threadsPerWarp          :: !Int         -- ^ Warp size
  , coresPerMP              :: !Int         -- ^ Number of SIMD arithmetic units per multiprocessor
  , warpsPerMP              :: !Int         -- ^ Maximum number of in-flight warps per multiprocessor
  , threadsPerMP            :: !Int         -- ^ Maximum number of in-flight threads on a multiprocessor
  , threadBlocksPerMP       :: !Int         -- ^ Maximum number of thread blocks resident on a multiprocessor
  , sharedMemPerMP          :: !Int         -- ^ Total amount of shared memory per multiprocessor (bytes)
  , maxSharedMemPerBlock    :: !Int         -- ^ Maximum amount of shared memory per thread block (bytes)
  , regFileSizePerMP        :: !Int         -- ^ Total number of registers in a multiprocessor
  , maxRegPerBlock          :: !Int         -- ^ Maximum number of registers per block
  , regAllocUnit            :: !Int         -- ^ Register allocation unit size
  , regAllocationStyle      :: !Allocation  -- ^ How multiprocessor resources are divided (register allocation granularity)
  , maxRegPerThread         :: !Int         -- ^ Maximum number of registers per thread
  , sharedMemAllocUnit      :: !Int         -- ^ Shared memory allocation unit size (bytes)
  , warpAllocUnit           :: !Int         -- ^ Warp allocation granularity
  , warpRegAllocUnit        :: !Int         -- ^ Warp register allocation granularity
  , maxGridsPerDevice       :: !Int         -- ^ Maximum number of resident grids per device (concurrent kernels)
  }
  deriving Show


-- |
-- Extract some additional hardware resource limitations for a given device.
--
deviceResources :: DeviceProperties -> DeviceResources
deviceResources = resources . computeCapability
  where
    -- This is mostly extracted from tables in the CUDA occupancy calculator.
    -- For Compute 8.7 and up, the Occupancy Calculator is hidden in Nsight Compute.
    --
    resources compute = case compute of
      Compute 1 0 -> resources (Compute 1 1)      -- Tesla G80
      Compute 1 1 -> DeviceResources              -- Tesla G8x
        { threadsPerWarp        = 32
        , coresPerMP            = 8
        , warpsPerMP            = 24
        , threadsPerMP          = 768
        , threadBlocksPerMP     = 8
        , sharedMemPerMP        = 16384
        , maxSharedMemPerBlock  = 16384
        , regFileSizePerMP      = 8192
        , maxRegPerBlock        = 8192
        , regAllocUnit          = 256
        , regAllocationStyle    = Block
        , maxRegPerThread       = 124
        , sharedMemAllocUnit    = 512
        , warpAllocUnit         = 2
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 1
        }
      Compute 1 2 ->  resources (Compute 1 3)     -- Tesla G9x
      Compute 1 3 -> (resources (Compute 1 1))    -- Tesla GT200
        { threadsPerMP          = 1024
        , warpsPerMP            = 32
        , regFileSizePerMP      = 16384
        , maxRegPerBlock        = 16384
        , regAllocUnit          = 512
        }

      Compute 2 0 -> DeviceResources              -- Fermi GF100
        { threadsPerWarp        = 32
        , coresPerMP            = 32
        , warpsPerMP            = 48
        , threadsPerMP          = 1536
        , threadBlocksPerMP     = 8
        , sharedMemPerMP        = 49152
        , maxSharedMemPerBlock  = 49152
        , regFileSizePerMP      = 32768
        , maxRegPerBlock        = 32768
        , regAllocUnit          = 64
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 63
        , sharedMemAllocUnit    = 128
        , warpAllocUnit         = 2
        , warpRegAllocUnit      = 64
        , maxGridsPerDevice     = 16
        }
      Compute 2 1 -> (resources (Compute 2 0))    -- Fermi GF10x
        { coresPerMP            = 48
        }

      Compute 3 0 -> DeviceResources              -- Kepler GK10x
        { threadsPerWarp        = 32
        , coresPerMP            = 192
        , warpsPerMP            = 64
        , threadsPerMP          = 2048
        , threadBlocksPerMP     = 16
        , sharedMemPerMP        = 49152
        , maxSharedMemPerBlock  = 49152
        , regFileSizePerMP      = 65536
        , maxRegPerBlock        = 65536
        , regAllocUnit          = 256
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 63
        , sharedMemAllocUnit    = 256
        , warpAllocUnit         = 4
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 16
        }
      Compute 3 2 -> (resources (Compute 3 5))    -- Jetson TK1
        { maxRegPerBlock        = 32768
        , maxGridsPerDevice     = 4
        }
      Compute 3 5 -> (resources (Compute 3 0))    -- Kepler GK11x
        { maxRegPerThread       = 255
        , maxGridsPerDevice     = 32
        }
      Compute 3 7 -> (resources (Compute 3 5))    -- Kepler GK21x
        { sharedMemPerMP        = 114688
        , regFileSizePerMP      = 131072
        }

      Compute 5 0 -> DeviceResources              -- Maxwell GM10x
        { threadsPerWarp        = 32
        , coresPerMP            = 128
        , warpsPerMP            = 64
        , threadsPerMP          = 2048
        , threadBlocksPerMP     = 32
        , sharedMemPerMP        = 65536
        , maxSharedMemPerBlock  = 49152
        , regFileSizePerMP      = 65536
        , maxRegPerBlock        = 65536
        , regAllocUnit          = 256
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 255
        , sharedMemAllocUnit    = 256
        , warpAllocUnit         = 4
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 32
        }
      Compute 5 2 -> (resources (Compute 5 0))    -- Maxwell GM20x
        { sharedMemPerMP        = 98304
        , maxRegPerBlock        = 32768
        , warpAllocUnit         = 2
        }
      Compute 5 3 -> (resources (Compute 5 0))    -- Maxwell GM20B
        { maxRegPerBlock        = 32768
        , warpAllocUnit         = 2
        , maxGridsPerDevice     = 16
        }

      Compute 6 0 -> DeviceResources              -- Pascal GP100
        { threadsPerWarp        = 32
        , coresPerMP            = 64
        , warpsPerMP            = 64
        , threadsPerMP          = 2048
        , threadBlocksPerMP     = 32
        , sharedMemPerMP        = 65536
        , maxSharedMemPerBlock  = 49152
        , regFileSizePerMP      = 65536
        , maxRegPerBlock        = 65536
        , regAllocUnit          = 256
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 255
        , sharedMemAllocUnit    = 256
        , warpAllocUnit         = 2
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 128
        }
      Compute 6 1 -> (resources (Compute 6 0))    -- Pascal GP10x
        { coresPerMP            = 128
        , sharedMemPerMP        = 98304
        , warpAllocUnit         = 4
        , maxGridsPerDevice     = 32
        }
      Compute 6 2 -> (resources (Compute 6 0))    -- Pascal GP10B
        { coresPerMP            = 128
        , warpsPerMP            = 128
        , threadBlocksPerMP     = 4096
        , maxRegPerBlock        = 32768
        , warpAllocUnit         = 4
        , maxGridsPerDevice     = 16
        }

      Compute 7 0 -> DeviceResources              -- Volta GV100
        { threadsPerWarp        = 32
        , coresPerMP            = 64
        , warpsPerMP            = 64
        , threadsPerMP          = 2048
        , threadBlocksPerMP     = 32
        , sharedMemPerMP        = 98304           -- of 128KB
        , maxSharedMemPerBlock  = 98304
        , regFileSizePerMP      = 65536
        , maxRegPerBlock        = 65536
        , regAllocUnit          = 256
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 255
        , sharedMemAllocUnit    = 256
        , warpAllocUnit         = 4
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 128
        }

      Compute 7 2 -> (resources (Compute 7 0))    -- Volta GV10B
        { maxGridsPerDevice     = 16
        , maxSharedMemPerBlock  = 49152
        }

      Compute 7 5 -> (resources (Compute 7 0))    -- Turing TU1xx
        { warpsPerMP            = 32
        , threadBlocksPerMP     = 16
        , threadsPerMP          = 1024
        , maxGridsPerDevice     = 128
        , sharedMemPerMP        = 65536           -- of 96KB
        , maxSharedMemPerBlock  = 65536
        }

      Compute 8 0 -> DeviceResources              -- Ampere GA100
        { threadsPerWarp        = 32
        , coresPerMP            = 64
        , warpsPerMP            = 64
        , threadsPerMP          = 2048
        , threadBlocksPerMP     = 32
        , sharedMemPerMP        = 167936          -- of 192KB
        , maxSharedMemPerBlock  = 167936
        , regFileSizePerMP      = 65536
        , maxRegPerBlock        = 65536
        , regAllocUnit          = 256
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 255
        , sharedMemAllocUnit    = 128
        , warpAllocUnit         = 4
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 128
        }

      Compute 8 6 -> (resources (Compute 8 0))    -- Ampere GA102
        { warpsPerMP            = 48
        , threadsPerMP          = 1536
        , threadBlocksPerMP     = 16
        , sharedMemPerMP        = 102400
        , maxSharedMemPerBlock  = 102400
        }

      Compute 8 7 -> DeviceResources              
        { threadsPerWarp        = 32
        , coresPerMP            = 64
        , warpsPerMP            = 48
        , threadsPerMP          = 1536
        , threadBlocksPerMP     = 16
        , sharedMemPerMP        = 167936          
        , maxSharedMemPerBlock  = 167936
        , regFileSizePerMP      = 65536
        , maxRegPerBlock        = 65536
        , regAllocUnit          = 256
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 255
        , sharedMemAllocUnit    = 128
        , warpAllocUnit         = 4
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 128
        }

      Compute 8 9 -> DeviceResources              
        { threadsPerWarp        = 32
        , coresPerMP            = 64 
        , warpsPerMP            = 48
        , threadsPerMP          = 1536
        , threadBlocksPerMP     = 24
        , sharedMemPerMP        = 102400       
        , maxSharedMemPerBlock  = 102400
        , regFileSizePerMP      = 65536
        , maxRegPerBlock        = 65536
        , regAllocUnit          = 256
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 255
        , sharedMemAllocUnit    = 128
        , warpAllocUnit         = 4
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 128
        }

      Compute 9 0 -> trace "*** Warning: Compute Capability 9.0 has Thread Block Clusters, which this occupancy calculation might not support" $ DeviceResources
        { threadsPerWarp        = 32
        , coresPerMP            = 64
        , warpsPerMP            = 64
        , threadsPerMP          = 1536
        , threadBlocksPerMP     = 32
        , sharedMemPerMP        = 233472        
        , maxSharedMemPerBlock  = 233472
        , regFileSizePerMP      = 65536
        , maxRegPerBlock        = 65536
        , regAllocUnit          = 256
        , regAllocationStyle    = Warp
        , maxRegPerThread       = 255
        , sharedMemAllocUnit    = 128
        , warpAllocUnit         = 4
        , warpRegAllocUnit      = 256
        , maxGridsPerDevice     = 128
        }


      -- Something might have gone wrong, or the library just needs to be
      -- updated for the next generation of hardware, in which case we just want
      -- to pick a sensible default and carry on.
      --
      -- This is slightly dodgy as the warning message is coming from pure code.
      -- However, it should be OK because all library functions run in IO, so it
      -- is likely the user code is as well.
      --
      _           -> trace warning $ resources (Compute 6 0) -- TODO: is this still a sensible default?
        where warning = unlines [ "*** Warning: Unknown CUDA device compute capability: " ++ show compute
                                , "*** Please submit a bug report at https://github.com/tmcdonell/cuda/issues" ]

