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
import RandomVector

-- System
import Control.Exception
import qualified Foreign.CUDA.Runtime as CUDA


--------------------------------------------------------------------------------
-- Reference
--------------------------------------------------------------------------------

foldRef :: (Num e, Storable e) => [e] -> IO e
foldRef xs = evaluate (foldl (+) 0 xs)

--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

--
-- Note that this requires two memory copies: once from a Haskell list to the C
-- heap, and from there into the graphics card memory.
--
foldCUDA :: [Float] -> IO Float
foldCUDA xs = CUDA.withArrayLen xs $ \len d_xs -> fold_plusf d_xs len

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
  putStrLn "== Generating random numbers"
  xs   <- randomList 30000

  putStrLn "== Generating reference solution"
  ref  <- foldRef xs

  putStrLn "== Testing CUDA"
  cuda <- foldCUDA xs

  putStr   "== Validating: "
  putStrLn $ if ((ref-cuda)/ref) < 0.0001 then "Ok!" else "INVALID!"

