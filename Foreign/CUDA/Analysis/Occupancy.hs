--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Analysis.Occupancy
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Occupancy calculations for CUDA kernels
--
-- <http://developer.download.nvidia.com/compute/cuda/3_0/sdk/docs/CUDA_Occupancy_calculator.xls>
--
-- /Determining Registers Per Thread and Shared Memory Per Block/
--
-- To determine the number of registers used per thread in your kernel, simply
-- compile the kernel code using the option
--
-- > --ptxas-options=-v
--
-- to nvcc.  This will output information about register, local memory, shared
-- memory, and constant memory usage for each kernel in the @.cu@ file.
-- Alternatively, you can compile with the @-cubin@ option to nvcc.  This will
-- generate a @.cubin@ file, which you can open in a text editor.  Look for the
-- @code@ section with your kernel's name.  Within the curly braces (@{ ... }@)
-- for that code block, you will see a line with @reg = X@, where @x@ is the
-- number of registers used by your kernel.  You can also see the amount of
-- shared memory used as @smem = Y@.  However, if your kernel declares any
-- external shared memory that is allocated dynamically, you will need to add
-- the number in the @.cubin@ file to the amount you dynamically allocate at run
-- time to get the correct shared memory usage.
--
-- /Notes About Occupancy/
--
-- Higher occupancy does not necessarily mean higher performance.  If a kernel
-- is not bandwidth bound, then increasing occupancy will not necessarily
-- increase performance.  If a kernel invocation is already running at least one
-- thread block per multiprocessor in the GPU, and it is bottlenecked by
-- computation and not by global memory accesses, then increasing occupancy may
-- have no effect.  In fact, making changes just to increase occupancy can have
-- other effects, such as additional instructions, spills to local memory (which
-- is off chip), divergent branches, etc.  As with any optimization, you should
-- experiment to see how changes affect the *wall clock time* of the kernel
-- execution.  For bandwidth bound applications, on the other hand, increasing
-- occupancy can help better hide the latency of memory accesses, and therefore
-- improve performance.
--


module Foreign.CUDA.Analysis.Occupancy
  (
    calcOccupancy, Compute, Occupancy(..)
  )
  where

import Data.Maybe


-- GPU Hardware Resources
type Compute    = Double                -- ^ Compute capability

data Allocation = Warp | Block
data Resources  = Resources
  {
    threadsPerWarp     :: Int,          -- ^ Warp size
   _threadsPerMP       :: Int,          -- ^ Maximum number of in-flight threads on a multiprocessor
    threadBlocksPerMP  :: Int,          -- ^ Maximum number of thread blocks resident on a multiprocessor
    warpsPerMP         :: Int,          -- ^ Maximum number of in-flight warps per multiprocessor
    sharedMemPerMP     :: Int,          -- ^ Total amount of shared memory per multiprocessor (bytes)
    sharedMemAllocUnit :: Int,          -- ^ Shared memory allocation unit size (bytes)
    regFileSize        :: Int,          -- ^ Total number of registers in a multiprocessor
    regAllocUnit       :: Int,          -- ^ Register allocation unit size
    regAllocWarp       :: Int,          -- ^ Register allocation granularity for warps
    allocation         :: Allocation    -- ^ How multiprocessor resources are divided
  }

-- GPU Occupancy per multiprocessor
--
data Occupancy = Occupancy
  {
    activeThreads     :: Int,           -- ^ Active threads per multiprocessor
    activeTreadBlocks :: Int,           -- ^ Active thread blocks per multiprocessor
    activeWarps       :: Int,           -- ^ Active warps per multiprocessor
    occupancy100      :: Double         -- ^ Occupancy of each multiprocessor (percent)
  }
  deriving (Eq, Ord, Show)


-- Raw GPU Data
--
gpuData :: [(Compute, Resources)]
gpuData =
  [(1.0, Resources 32  768 8 24 16384 512  8192 256 2 Block)
  ,(1.1, Resources 32  768 8 24 16384 512  8192 256 2 Block)
  ,(1.2, Resources 32 1024 8 32 16384 512 16384 512 2 Block)
  ,(1.3, Resources 32 1024 8 32 16384 512 16384 512 2 Block)
  ,(2.0, Resources 32 1536 8 48 49152 128 32768  64 1 Warp)
  ]


-- Calculate occupancy data for a given GPU and kernel resource usage
--
calcOccupancy :: Compute        -- ^ Compute capability of card in question
              -> Int            -- ^ Threads per block
              -> Int            -- ^ Registers per thread
              -> Int            -- ^ Shared memory per block (bytes)
              -> Occupancy
calcOccupancy capability thds regs smem
  = Occupancy at ab aw oc
  where
    at = ab * thds
    aw = ab * warps
    ab = minimum [limitWarpBlock, limitRegMP, limitSMemMP]
    oc = 100 * fromIntegral aw / fromIntegral (warpsPerMP gpu)

    regs' = 1 `max` regs
    smem' = 1 `max` smem

    floor'        = floor   :: Double -> Int
    ceiling'      = ceiling :: Double -> Int
    ceilingBy x s = s * ceiling' (fromIntegral x / fromIntegral s)

    -- Physical resources (default to compute 1.0)
    --
    def = snd (head gpuData)
    gpu = fromMaybe def (lookup capability gpuData)

    -- Allocation per thread block
    --
    warps     = ceiling' (fromIntegral thds / fromIntegral (threadsPerWarp gpu))
    sharedMem = ceilingBy smem' (sharedMemAllocUnit gpu)
    registers = case allocation gpu of
      Block -> (warps `ceilingBy` regAllocWarp gpu * regs' * threadsPerWarp gpu) `ceilingBy` regAllocUnit gpu
      Warp  -> warps * ceilingBy (regs' * threadsPerWarp gpu) (regAllocUnit gpu)

    -- Maximum thread blocks per multiprocessor
    --
    limitWarpBlock = threadBlocksPerMP gpu `min` floor' (fromIntegral (warpsPerMP gpu)     / fromIntegral warps)
    limitRegMP     = threadBlocksPerMP gpu `min` floor' (fromIntegral (regFileSize gpu)    / fromIntegral registers)
    limitSMemMP    = threadBlocksPerMP gpu `min` floor' (fromIntegral (sharedMemPerMP gpu) / fromIntegral sharedMem)

