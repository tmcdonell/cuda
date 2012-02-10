{-# LANGUAGE ParallelListComp, CPP, BangPatterns, ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
--
-- Module    : MultiGPU
-- Copyright : (c) 2012 Kevin Ying
-- License   : BSD
--
-- An implemenations of Nvidia's simpleMultiGPU example
--
--------------------------------------------------------------------------------

module Main where

import Time

-- System
import System.Random.MWC
import Data.List
import Control.Monad
import Foreign
import Foreign.CUDA                     (HostPtr, DevicePtr, withDevicePtr, withHostPtr) 
import Foreign.CUDA.Runtime.Stream      as CR
import qualified Foreign.CUDA.Runtime   as CR

import qualified Data.Vector.Unboxed    as U

-- Settings
data_n   = 1048576 * 32
block_n  = 32
thread_n = 256 
accum_n  = block_n * thread_n

--
-- |A set of data prepared for execution on a GPU
--
data Plan = Plan
  {
    device            :: Int,              -- ^ device number
    stream            :: Stream,           -- ^ stream on the device
    dataN             :: Int,              -- ^ number of elements in data
    v_data            :: U.Vector Float,   -- ^ host data
    h_data            :: HostPtr Float,    -- ^ host data in pagelocked memory

    d_data            :: DevicePtr Float,  -- ^ device data
    d_sum             :: DevicePtr Float,  -- ^ device results

    h_sum_from_device :: HostPtr Float     -- ^ results from device
  }
  deriving Show


main :: IO ()
main = do
  gpu_n <- CR.count
  putStrLn $ "Number of GPUs found: " ++ show gpu_n

  -- Execute on all GPUs
  putStrLn $ "Executing " ++ (show gpu_n) ++ " sets of data over " ++ (show gpu_n) ++ " GPU(s)"
  withPlans gpu_n gpu_n executePlans

  -- Use only first GPU
  putStrLn $ "Executing " ++ (show gpu_n) ++ " sets of data over 1 GPU"
  withPlans gpu_n 1 executePlans

  putStrLn "Finished"
  

-- |Launches kernels on the specified gpu and compares with the CPU 
executePlans :: [Plan] -> IO ()
executePlans plans = do
    --Copy data to GPU, launch Kernel, copy data back all asynchronously
    (t,_) <- flip (benchmark 100) CR.sync $ forM_ plans $ \plan -> do 
      CR.set (device plan)
      --Copy input data from CPU
      CR.pokeArrayAsync (dataN plan) (h_data plan) (d_data plan) (Just $ stream plan)
      --Perform GPU computations
      reduceKernel (d_sum plan) (d_data plan) (dataN plan) block_n thread_n (stream plan)
      --Read back GPU results
      CR.peekArrayAsync accum_n (d_sum plan) (h_sum_from_device plan) (Just $ stream plan)
    
    --Process results
    h_sumGPU <- forM plans $ \plan -> do
      CR.set (device plan)
      CR.block (stream plan)
      hs_sum <- withHostPtr (h_sum_from_device plan) $ \ptr -> peekArray accum_n ptr
      return $ sum hs_sum
    let sumGPU = sum h_sumGPU

    putStrLn $ " GPU processing time: " ++ (show $ timeIn millisecond t) ++ " ms"


    let sumCPU = sum $ map (\p -> U.sum $ v_data p) plans

    let diff = (abs $ sumCPU - sumGPU) / (abs sumCPU);
    putStrLn $ "  GPU sum: " ++ show sumGPU 
    putStrLn $ "  CPU sum: " ++ show sumCPU
    putStrLn $ "  Relative difference: " ++ show diff

    if (diff < 1e-4) then 
      putStrLn $ "Passed!"
      else 
      putStrLn $ "Failed!"

-- |Makes n plans spread over ndevices
withPlans :: Int -> Int -> ([Plan] -> IO ()) ->  IO ()
withPlans n ndevices action = do
  devStrms' <- forM [0..(ndevices-1)] $ \dev -> do
    CR.set dev
    strm <- CR.create
    return (dev, strm)

  let devStrms = take n $ cycle devStrms'  

  plans <- forM devStrms $ \(i, stream) -> do
    let dataN = if (i < data_n `mod` n) then (data_n `div` n) + 1 else data_n `div` n
    CR.set i
    -- Allocate memory
    -- Host
    v_data <- randomList dataN 
    h_data <- CR.mallocHostArray [] dataN 
    withHostPtr h_data $ \ptr -> pokeArray ptr (U.toList v_data)

    h_sum_from_device <- CR.mallocHostArray [] accum_n 

    -- Device
    d_data <- CR.mallocArray (dataN   * sizeOf (undefined :: DevicePtr Float))
    d_sum  <- CR.mallocArray (accum_n * sizeOf (undefined :: DevicePtr Float))


    return $ Plan i stream dataN v_data h_data d_data d_sum h_sum_from_device

  action plans

  -- clean up
  forM_ plans $ \plan -> do
    CR.set (device plan)
    CR.freeHost (h_data plan)
    CR.freeHost (h_sum_from_device plan)

    CR.free (d_data plan)
    CR.free (d_sum plan)
    --CR.destroy (stream plan)
  forM_ devStrms' $ \(_,stream) -> CR.destroy stream


randomList :: Int -> IO (U.Vector Float)
randomList n = withSystemRandom $ \gen -> U.replicateM n (uniform gen :: IO Float)

-- |Binding for the reduce kernel
reduceKernel :: DevicePtr Float -> DevicePtr Float -> Int -> Int -> Int -> Stream -> IO ()
reduceKernel a1 a2 a3 a4 a5 a6 =
  withDevicePtr a1 $ \a1' ->
  withDevicePtr a2 $ \a2' ->
  reduceKernel'_ a1' a2' a3 a4 a5 a6

foreign import ccall unsafe "simpleMultiGPU.h launch_reduceKernel"
  reduceKernel'_ :: Ptr Float -> Ptr Float -> Int -> Int -> Int -> Stream -> IO ()
