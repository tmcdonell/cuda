{-# LANGUAGE CPP #-}
--------------------------------------------------------------------------------
--
-- Module    : MatrixMul
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Matrix multiplication using runtime interface and execution control instead
-- of calling C functions via the FFI.
--
--------------------------------------------------------------------------------

module Main where

#include "matrix_mul.h"

-- Friends
import Time
import RandomVector

-- System
import Data.Array
import System.IO
import Foreign
import qualified Foreign.CUDA as CUDA


--------------------------------------------------------------------------------
-- Reference
--------------------------------------------------------------------------------

matMult :: (Num e, Storable e) => Matrix e -> Matrix e -> IO (Matrix e)
matMult mx my = do
  x <- unsafeFreeze mx
  y <- unsafeFreeze my
  let ((li, lj), (ui, uj))  = bounds x
      ((li',lj'),(ui',uj')) = bounds y
      resBnds | (lj,uj) == (li',ui') = ((li,lj'),(ui,uj'))
              | otherwise            = error "matrix dimensions must agree"

  newListArray resBnds [sum [x!(i,k) * y!(k,j) | k <- range (lj,uj)]
                         | i <- range (li,ui)
                         , j <- range (lj',uj') ]


--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

matMultCUDA :: (Num e, Storable e) => Matrix e -> Matrix e -> IO (Matrix e)
matMultCUDA xs' ys' = doMult undefined xs' ys'
  where
    doMult :: (Num e', Storable e') => e' -> Matrix e' -> Matrix e' -> IO (Matrix e')
    doMult dummy xs ys = do

      -- Setup matrix parameters
      --
      ((li, lj), (ui, uj))  <- getBounds xs
      ((li',lj'),(ui',uj')) <- getBounds ys
      let wx = rangeSize (lj,uj)
          hx = rangeSize (li,ui)
          wy = rangeSize (lj',uj')
          hy = rangeSize (li',ui')
          resBnds | wx == hy  = ((li,lj'),(ui,uj'))
                  | otherwise = error "matrix dimensions must agree"

      -- Allocate memory and copy test data
      --
      CUDA.allocaArray (wx*hx) $ \d_xs -> do
      CUDA.allocaArray (wy*hy) $ \d_ys -> do
      CUDA.allocaArray (wy*hx) $ \d_zs -> do
      withMatrix xs $ \p -> CUDA.pokeArray (wx*hx) p d_xs
      withMatrix ys $ \p -> CUDA.pokeArray (wy*hy) p d_ys

      -- Launch the kernel
      --
      let gridDim   = (wy`div`BLOCK_SIZE, hx`div`BLOCK_SIZE)
          blockDim  = (BLOCK_SIZE,BLOCK_SIZE,1)
          sharedMem = 2 * BLOCK_SIZE * BLOCK_SIZE * fromIntegral (sizeOf dummy)

      CUDA.setConfig gridDim blockDim sharedMem Nothing
      CUDA.setParams [CUDA.VArg d_xs, CUDA.VArg d_ys, CUDA.VArg d_zs, CUDA.IArg wx, CUDA.IArg wy]
      CUDA.launch "matrixMul"

      -- Copy back result
      zs <- newArray_ resBnds
      withMatrix zs $ \p -> CUDA.peekArray (wy*hx) d_zs p
      return zs


--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  dev   <- CUDA.get
  props <- CUDA.props dev
  putStrLn $ "Using device " ++ show dev ++ ": " ++ CUDA.deviceName props

  xs <- randomArr ((1,1),(8*BLOCK_SIZE, 4*BLOCK_SIZE)) :: IO (Matrix Float)
  ys <- randomArr ((1,1),(4*BLOCK_SIZE,12*BLOCK_SIZE)) :: IO (Matrix Float)

  putStr   "== Reference: " >> hFlush stdout
  (tr,ref) <- benchmark 100 (matMult xs ys) (return ())
  putStrLn $  shows (fromInteger (timeIn millisecond tr) / 100::Float) " ms"

  putStr   "== CUDA: " >> hFlush stdout
  (tc,mat) <- benchmark 100 (matMultCUDA xs ys) (CUDA.sync)
  putStrLn $  shows (fromInteger (timeIn millisecond tc) / 100::Float) " ms"

  putStr "== Validating: "
  verify ref mat >>= \rv -> putStrLn $ if rv then "Ok!" else "INVALID!"

