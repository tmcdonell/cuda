{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- Implementation of a prefix-sum. The kernel code is taken from the NVIDIA SDK
-- sample project 'scan'.
--
-- This implementation is limited to arrays that can be processed by a single
-- block of threads, and size equal to a power of two.
--------------------------------------------------------------------------------

module Main where

import Foreign
import Foreign.C

import Foreign.CUDA (DevicePtr, withDevicePtr)
import qualified Foreign.CUDA as C

#include "src/stubs.h"


-- Kernel
-- ------
plusScan      :: [CFloat] -> IO [CFloat]
plusScan vals =
    let len   = length vals
        bytes = fromIntegral len * fromIntegral (sizeOf (undefined::CFloat))
    in
        C.allocaBytes bytes $ \out_ptr -> do
        C.withArray vals    $ \in_ptr  -> do
        cuda_plus_scan out_ptr in_ptr len >> do
        C.forceEither `fmap` C.peekArray len out_ptr

{# fun unsafe cuda_plus_scan
    { withDevicePtr* `DevicePtr CFloat' ,
      withDevicePtr* `DevicePtr CFloat' ,
      fromIntegral   `Int'              } -> `()' #}


-- Main
-- ----
main :: IO ()
main =
    let nums = take 512 [1..] in
    do
    gpu  <- plusScan nums

    if gpu == init (scanl (+) 0 nums)
      then putStrLn "Test PASSED"
      else error ("Test FAILED: " ++ show gpu)

