{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- Implementation of a prefix-sum. The kernel code is taken from the NVIDIA SDK
-- sample project 'scan'.
--------------------------------------------------------------------------------

module Main where

import Foreign
import Foreign.C

import Foreign.CUDA (DevicePtr, withDevicePtr)
import qualified Foreign.CUDA as C

#include "stubs.h"


-- Kernel
-- ------
plusScan      :: [CFloat] -> IO [CFloat]
plusScan vals =
    let len   = length vals
        bytes = fromIntegral len * fromIntegral (sizeOf (undefined::CFloat))
    in
        C.allocaBytes bytes $ \o_ptr -> do
        C.withArray vals    $ \i_ptr -> do
        plus_scan o_ptr i_ptr len    >> do
        C.forceEither `fmap` C.peekArray len o_ptr

{# fun plus_scan
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
      then putStrLn  "Test PASSED"
      else putStrLn ("Test FAILED: " ++ show gpu)

