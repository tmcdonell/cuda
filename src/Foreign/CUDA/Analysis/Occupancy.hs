{-# LANGUAGE BangPatterns #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Analysis.Occupancy
-- Copyright : [2009..2023] Trevor L. McDonell
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
-- __Warning__: Like the official Occupancy Calculator in NVidia Nsight
-- Compute, the calculator in this module does not support or consider Thread
-- Block Clusters
-- (<https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters>)
-- that have been introduced with compute capability 9.0 (Hopper). If you use
-- thread block clusters in your kernels, the results you get with the
-- functions in this module may not be accurate. Profile and measure.
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
    occupancy, optimalBlockSize, optimalBlockSizeOf, maxResidentBlocks,
    incPow2, incWarp, decPow2, decWarp

) where

import Data.Ord
import Data.List

import Foreign.CUDA.Analysis.Device                                 hiding ( maxSharedMemPerBlock )


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
occupancy !dev !threads !regs !smem
  = Occupancy at ab aw oc
  where
    at = ab * threads
    aw = ab * limitWarpsPerBlock
    ab = minimum [limitDueToWarps, limitDueToRegs, limitDueToSMem]
    oc = 100 * fromIntegral aw / fromIntegral (warpsPerMP gpu)

    ceiling'      = ceiling :: Double -> Int
    floor'        = floor   :: Double -> Int
    ceilingBy x s = s * ceiling' (fromIntegral x / fromIntegral s)
    floorBy x s   = s * floor' (fromIntegral x / fromIntegral s)

    -- Physical resources
    --
    gpu                  = deviceResources dev
    maxSharedMemPerBlock = fromIntegral (sharedMemPerBlock dev) -- 2080Ti reports 48KB, but should be 64KB !

    -- Allocated resource limits
    --
    limitWarpsPerBlock  = ceiling' (fromIntegral threads / fromIntegral (threadsPerWarp gpu))
    -- limitWarpsPerSM     = warpsPerMP gpu

    limitRegsPerBlock   = case regAllocationStyle gpu of
                            Block -> ((limitWarpsPerBlock `ceilingBy` warpAllocUnit gpu) * regs * threadsPerWarp gpu) `ceilingBy` regAllocUnit gpu
                            Warp  -> limitWarpsPerBlock
    limitRegsPerSM      = case regAllocationStyle gpu of
                            Block -> maxRegPerBlock gpu
                            Warp  -> (maxRegPerBlock gpu `div` ((regs * threadsPerWarp gpu) `ceilingBy` warpRegAllocUnit gpu)) `floorBy` warpAllocUnit gpu

    limitSMemPerBlock   = smem `ceilingBy` sharedMemAllocUnit gpu
    -- limitSMemPerSM      = maxSharedMemPerBlock gpu

    -- Maximum thread blocks per multiprocessor
    --
    limitDueToWarps                 = threadBlocksPerMP gpu `min` (warpsPerMP gpu `div` limitWarpsPerBlock)
    limitDueToRegs
      | regs > maxRegPerThread gpu  = 0
      | regs > 0                    = (limitRegsPerSM `div` limitRegsPerBlock) * (regFileSizePerMP gpu `div` maxRegPerBlock gpu)
      | otherwise                   = threadBlocksPerMP gpu

    limitDueToSMem
      | smem > maxSharedMemPerBlock = 0
      | smem > 0                    = sharedMemPerMP gpu `div` limitSMemPerBlock
      | otherwise                   = threadBlocksPerMP gpu


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
optimalBlockSize dev = optimalBlockSizeOf dev (decWarp dev)


-- |
-- As 'optimalBlockSize', but with a generator that produces the specific thread
-- block sizes that should be tested. The generated list can produce values in
-- any order, but the last satisfying block size will be returned. Hence, values
-- should be monotonically decreasing to return the smallest block size yielding
-- maximum occupancy, and vice-versa.
--
{-# INLINEABLE optimalBlockSizeOf #-}
optimalBlockSizeOf
    :: DeviceProperties         -- ^ Architecture to optimise for
    -> [Int]                    -- ^ Thread block sizes to consider
    -> (Int -> Int)             -- ^ Register count as a function of thread block size
    -> (Int -> Int)             -- ^ Shared memory usage (bytes) as a function of thread block size
    -> (Int, Occupancy)
optimalBlockSizeOf !dev !threads !freg !fsmem
  = maximumBy (comparing (occupancy100 . snd))
  $ [ (t, occupancy dev t (freg t) (fsmem t)) | t <- threads ]


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

