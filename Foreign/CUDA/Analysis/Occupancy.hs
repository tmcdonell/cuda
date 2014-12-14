{-# LANGUAGE BangPatterns #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Analysis.Occupancy
-- Copyright : [2009..2014] Trevor L. McDonell
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


module Foreign.CUDA.Analysis.Occupancy (

    Occupancy(..),
    occupancy, optimalBlockSize, optimalBlockSizeBy, maxResidentBlocks,
    incPow2, incWarp, decPow2, decWarp

) where

import Data.Ord
import Data.List

import Foreign.CUDA.Analysis.Device


-- GPU Occupancy per multiprocessor
--
data Occupancy = Occupancy
  {
    activeThreads      :: !Int,         -- ^ Active threads per multiprocessor
    activeThreadBlocks :: !Int,         -- ^ Active thread blocks per multiprocessor
    activeWarps        :: !Int,         -- ^ Active warps per multiprocessor
    occupancy100       :: !Double       -- ^ Occupancy of each multiprocessor (percent)
  }
  deriving (Eq, Ord, Show)


-- |
-- Calculate occupancy data for a given GPU and kernel resource usage
--
{-# INLINEABLE occupancy #-}
occupancy
    :: DeviceProperties -- ^ Properties of the card in question
    -> Int              -- ^ Threads per block
    -> Int              -- ^ Registers per thread
    -> Int              -- ^ Shared memory per block (bytes)
    -> Occupancy
occupancy !dev !thds !regs !smem
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
    gpu = deviceResources dev

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
-- resource usage. This returns the smallest satisfying block size in increments
-- of a single warp.
--
{-# INLINEABLE optimalBlockSize #-}
optimalBlockSize
    :: DeviceProperties         -- ^ Architecture to optimise for
    -> (Int -> Int)             -- ^ Register count as a function of thread block size
    -> (Int -> Int)             -- ^ Shared memory usage (bytes) as a function of thread block size
    -> (Int, Occupancy)
optimalBlockSize = flip optimalBlockSizeBy decWarp


-- |
-- As 'optimalBlockSize', but with a generator that produces the specific thread
-- block sizes that should be tested. The generated list can produce values in
-- any order, but the last satisfying block size will be returned. Hence, values
-- should be monotonically decreasing to return the smallest block size yielding
-- maximum occupancy, and vice-versa.
--
{-# INLINEABLE optimalBlockSizeBy #-}
optimalBlockSizeBy
    :: DeviceProperties
    -> (DeviceProperties -> [Int])
    -> (Int -> Int)
    -> (Int -> Int)
    -> (Int, Occupancy)
optimalBlockSizeBy !dev !fblk !freg !fsmem
  = maximumBy (comparing (occupancy100 . snd)) $ zip threads residency
  where
    residency = map (\t -> occupancy dev t (freg t) (fsmem t)) threads
    threads   = fblk dev


-- | Increments in powers-of-two, over the range of supported thread block sizes
-- for the given device.
--
{-# INLINEABLE incPow2 #-}
incPow2 :: DeviceProperties -> [Int]
incPow2 !dev = map ((2::Int)^) [lb, lb+1 .. ub]
  where
    round' = round :: Double -> Int
    lb     = round' . logBase 2 . fromIntegral $ warpSize dev
    ub     = round' . logBase 2 . fromIntegral $ maxThreadsPerBlock dev

-- | Decrements in powers-of-two, over the range of supported thread block sizes
-- for the given device.
--
{-# INLINEABLE decPow2 #-}
decPow2 :: DeviceProperties -> [Int]
decPow2 !dev = map ((2::Int)^) [ub, ub-1 .. lb]
  where
    round' = round :: Double -> Int
    lb     = round' . logBase 2 . fromIntegral $ warpSize dev
    ub     = round' . logBase 2 . fromIntegral $ maxThreadsPerBlock dev

-- | Decrements in the warp size of the device, over the range of supported
-- thread block sizes.
--
{-# INLINEABLE decWarp #-}
decWarp :: DeviceProperties -> [Int]
decWarp !dev = [block, block-warp .. warp]
  where
    !warp  = warpSize dev
    !block = maxThreadsPerBlock dev

-- | Increments in the warp size of the device, over the range of supported
-- thread block sizes.
--
{-# INLINEABLE incWarp #-}
incWarp :: DeviceProperties -> [Int]
incWarp !dev = [warp, 2*warp .. block]
  where
    warp  = warpSize dev
    block = maxThreadsPerBlock dev


-- |
-- Determine the maximum number of CTAs that can be run simultaneously for a
-- given kernel / device combination.
--
{-# INLINEABLE maxResidentBlocks #-}
maxResidentBlocks
  :: DeviceProperties   -- ^ Properties of the card in question
  -> Int                -- ^ Threads per block
  -> Int                -- ^ Registers per thread
  -> Int                -- ^ Shared memory per block (bytes)
  -> Int                -- ^ Maximum number of resident blocks
maxResidentBlocks !dev !thds !regs !smem =
  multiProcessorCount dev * activeThreadBlocks (occupancy dev thds regs smem)

