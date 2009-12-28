{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
--
-- Module    : Fold
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Reduce a vector to a single value
--
--------------------------------------------------------------------------------

module Main where

#include "fold.h"

-- Friends
import C2HS
import Time
import RandomVector

-- System
import Control.Exception
import qualified Foreign.CUDA.Runtime as CUDA


--------------------------------------------------------------------------------
-- Reference
--------------------------------------------------------------------------------

foldRef :: Num e => [e] -> IO e
foldRef xs = do
  (t,r) <- benchmark 100 (evaluate (foldl (+) 0 xs)) (return ())
  putStrLn $ "== Reference: " ++ show (timeIn millisecond t) ++ "ms"
  return r

--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

--
-- Note that this requires two memory copies: once from a Haskell list to the C
-- heap, and from there into the graphics card memory. See the `bandwidthTest'
-- example for the atrocious performance of this operation.
--
-- For this test, cheat a little and just time the pure computation.
--
foldCUDA :: [Float] -> IO Float
foldCUDA xs = do
  let len = length xs
  CUDA.withListArray xs $ \d_xs -> do
    (t,r) <- benchmark 100 (fold_plusf d_xs len) CUDA.sync
    putStrLn $ "== CUDA: " ++ show (timeIn millisecond t) ++ "ms"
    return r

{# fun unsafe fold_plusf
  { withDP* `CUDA.DevicePtr Float'
  ,         `Int'
  }
  -> `Float' cFloatConv #}
  where
    withDP p a = CUDA.withDevicePtr p $ \p' -> a (castPtr p')


--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  dev   <- CUDA.get
  props <- CUDA.props dev
  putStrLn $ "Using device " ++ show dev ++ ": " ++ CUDA.deviceName props

  xs   <- randomList 30000
  ref  <- foldRef xs
  cuda <- foldCUDA xs

  putStrLn $ "== Validating: " ++ if ((ref-cuda)/ref) < 0.0001 then "Ok!" else "INVALID!"

