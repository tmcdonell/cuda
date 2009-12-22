{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
--
-- Module    : Scan
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Apply a binary operator to an array similar to 'fold', but return a
-- successive list of values reduced from the left (or right).
--
--------------------------------------------------------------------------------

module Main where

#include "scan.h"

-- Friends
import C2HS
import RandomVector

-- System
import Control.Exception
import qualified Foreign.CUDA as CUDA

--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

--
-- Note that this requires two memory copies: once from a Haskell list to the C
-- heap, and from there into the graphics card memory.
--
scanl_plus :: [Float] -> IO [Float]
scanl_plus xs =
  let len = length xs in
  CUDA.withListArray xs $ \d_xs  ->
  CUDA.allocaArray len  $ \d_res -> do
    scanl1_plusf d_xs d_res len
    CUDA.sync
    CUDA.peekListArray len d_res

{# fun unsafe scanl1_plusf
  { withDP* `CUDA.DevicePtr Float'
  , withDP* `CUDA.DevicePtr Float'
  ,         `Int'
  } -> `()' #}
  where
    withDP p a = CUDA.withDevicePtr p $ \p' -> a (castPtr p')


--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  putStrLn "== Generating random numbers"
  xs   <- randomList 10000

  putStrLn "== Generating reference solution"
  ref  <- evaluate (scanl1 (+) xs)

  putStrLn "== Testing CUDA"
  cuda <- scanl_plus xs

  putStr   "== Validating: "
  putStrLn $ if verifyList ref cuda then "Ok!" else "INVALID!"

