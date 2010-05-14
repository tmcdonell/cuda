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
--------------------------------------------------------------------------------


module Foreign.CUDA.Analysis.Occupancy
  (
    Occupancy(..),
    occupancy, optimalBlockSize
  )
  where

import Data.Ord
import Data.List

import Foreign.CUDA.Analysis.Device


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


-- |
-- Calculate occupancy data for a given GPU and kernel resource usage
--
occupancy
    :: DeviceProperties -- ^ Properties of the card in question
    -> Int              -- ^ Threads per block
    -> Int              -- ^ Registers per thread
    -> Int              -- ^ Shared memory per block (bytes)
    -> Occupancy
occupancy dev thds regs smem
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

    -- Physical resources
    --
    gpu = resources dev

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


-- |
-- Optimise multiprocessor occupancy as a function of thread block size and
-- resource usage. This returns the smallest satisfying block size.
--
optimalBlockSize
    :: DeviceProperties         -- ^ Architecture to optimise for
    -> (Int -> Int)             -- ^ Register count as a function of thread block size
    -> (Int -> Int)             -- ^ Shared memory usage (bytes) as a function of thread block size
    -> (Int, Occupancy)
optimalBlockSize dev freg fsmem
  = maximumBy (comparing (occupancy100 . snd)) $ zip threads residency
  where
    residency = map (\t -> occupancy dev t (freg t) (fsmem t)) threads
    threads   = let det = warpSize dev `div` 2
                    mts = maxThreadsPerBlock dev
                in  enumFromThenTo mts (mts - det) det

