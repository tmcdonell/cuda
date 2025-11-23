{-# LANGUAGE ScopedTypeVariables #-}
--------------------------------------------------------------------------------
--
-- Module    : Map
-- Copyright : (c) 2010 Trevor L. McDonell
-- License   : BSD
--
--------------------------------------------------------------------------------

module Main where

import qualified Foreign.CUDA                 as CUDA

import Foreign.Storable
import Foreign.Storable.Tuple ()
import qualified Foreign.Storable.Traversable as Store

instance Storable a => Storable [a] where
  sizeOf    = Store.sizeOf
  alignment = Store.alignment
  peek      = Store.peek (error "instance Traversable [a] is lazy, so we do not provide a real value here")
  poke      = Store.poke

main :: IO ()
main =
  let len = 16
      xs  = take len [0..] :: [Int]
      ys  = take len [0..] :: [Int]
  in
  CUDA.allocaArray len  $ \d_out ->
  CUDA.withListArray xs $ \d_in0 ->
  CUDA.withListArray ys $ \d_in1 -> do
    CUDA.setConfig (1,1) (len,1,1) 0 Nothing
    CUDA.setParams [CUDA.VArg d_out, CUDA.VArg (d_in0, d_in1), CUDA.IArg len]
    CUDA.launch "map"

    out :: [Int] <- CUDA.peekListArray len d_out
    print out

