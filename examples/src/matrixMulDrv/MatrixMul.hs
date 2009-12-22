{-# LANGUAGE CPP #-}
--------------------------------------------------------------------------------
--
-- Module    : MatrixMul
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Matrix multiplication
--
--------------------------------------------------------------------------------

module Main where

#include "matrix_mul.h"

-- Friends
import RandomVector

-- System
import Numeric
import Data.Array
import Control.Exception
import Data.Array.Storable
import Foreign.Storable
import qualified Data.ByteString.Char8 as B
import qualified Foreign.CUDA.Driver   as CUDA


-- Return the (width,height) of a matrix
--
getSize :: Storable e => Matrix e -> IO (Int,Int)
getSize mat = do
  ((li,lj),(ui,uj)) <- getBounds mat
  return (rangeSize (lj,uj), rangeSize (li,ui))

--------------------------------------------------------------------------------
-- Reference implementation
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

--
-- Initialise the device and context. Load the PTX source code, and return a
-- reference to the kernel function.
--
initCUDA :: IO (CUDA.Context, CUDA.Fun)
initCUDA = do
  CUDA.initialise []
  dev     <- CUDA.device 0
  ctx     <- CUDA.create dev []
  ptx     <- B.readFile "data/matrix_mul.ptx"
  (mdl,r) <- CUDA.loadDataEx ptx [CUDA.ThreadsPerBlock (BLOCK_SIZE*BLOCK_SIZE)]
  fun     <- CUDA.getFun mdl "matrixMul"

  putStrLn $ ">> PTX JIT compilation (" ++ showFFloat (Just 2) (CUDA.jitTime r) " ms)"
  B.putStrLn (CUDA.jitInfoLog r)
  return (ctx,fun)


--
-- Allocate some memory, and copy over the input data to the device. Should
-- probably catch allocation exceptions individually...
--
initData :: (Num e, Storable e)
         => Matrix e -> Matrix e -> IO (CUDA.DevicePtr e, CUDA.DevicePtr e, CUDA.DevicePtr e)
initData xs ys = do
  (wx,hx) <- getSize xs
  (wy,hy) <- getSize ys
  dxs     <- CUDA.mallocArray (wx*hx)
  dys     <- CUDA.mallocArray (wy*hy)
  res     <- CUDA.mallocArray (wy*hx)

  flip onException (mapM_ CUDA.free [dxs,dys,res]) $ do
  withMatrix xs $ \p -> CUDA.pokeArray (wx*hx) p dxs
  withMatrix ys $ \p -> CUDA.pokeArray (wy*hy) p dys
  return (dxs, dys, res)


--
-- Run the test
--
testCUDA :: (Num e, Storable e) => Matrix e -> Matrix e -> IO (Matrix e)
testCUDA xs' ys' = doTest undefined xs' ys'
  where
    doTest :: (Num e', Storable e') => e' -> Matrix e' -> Matrix e' -> IO (Matrix e')
    doTest dummy xs ys = do
      (widthX,heightX)      <- getSize xs
      (widthY,_)            <- getSize ys
      ((li, lj), (ui, uj))  <- getBounds xs
      ((li',lj'),(ui',uj')) <- getBounds ys
      let resBnds | (lj,uj) == (li',ui') = ((li,lj'),(ui,uj'))
                  | otherwise            = error "matrix dimensions must agree"

      -- Initialise environment and copy over test data
      --
      putStrLn ">> Initialising"
      bracket initCUDA (\(ctx,_) -> CUDA.destroy ctx) $ \(_,matMul) -> do

      -- Ensure we release the memory, even if there was an error
      --
      putStrLn ">> Executing"
      bracket
        (initData xs ys)
        (\(dx,dy,dz) -> mapM_ CUDA.free [dx,dy,dz]) $
         \(dx,dy,dz) -> do
          -- Repeat test many times...
          --
          CUDA.setParams     matMul [CUDA.VArg dx, CUDA.VArg dy, CUDA.VArg dz, CUDA.IArg widthX, CUDA.IArg widthY]
          CUDA.setBlockShape matMul (BLOCK_SIZE,BLOCK_SIZE,1)
          CUDA.setSharedSize matMul (fromIntegral (2 * BLOCK_SIZE * BLOCK_SIZE * sizeOf dummy))
          CUDA.launch        matMul (widthY `div` BLOCK_SIZE, heightX `div` BLOCK_SIZE) Nothing
          CUDA.sync

          -- Copy back result
          --
          zs <- newArray_ resBnds
          withMatrix zs $ \p -> CUDA.peekArray (widthY*heightX) dz p
          return zs


--------------------------------------------------------------------------------
-- Test & Verify
--------------------------------------------------------------------------------

main :: IO ()
main = do
  putStrLn "== Generating random matrices"
  xs <- randomArr ((1,1),(8*BLOCK_SIZE, 4*BLOCK_SIZE)) :: IO (Matrix Float)
  ys <- randomArr ((1,1),(4*BLOCK_SIZE,12*BLOCK_SIZE)) :: IO (Matrix Float)

  putStrLn "== Generating reference solution"
  ref <- matMult xs ys

  putStrLn "== Testing CUDA"
  mat <- testCUDA xs ys

  putStr "== Validating: "
  verify ref mat >>= \rv -> putStrLn $ if rv then "Ok!" else "INVALID!"

