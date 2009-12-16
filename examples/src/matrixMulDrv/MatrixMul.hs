{-# LANGUAGE CPP, FlexibleContexts, ParallelListComp #-}
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

#include "MatrixMul.h"

import Foreign
import qualified Foreign.CUDA.Driver as CUDA

import Numeric
import Data.Array
import Data.Array.Storable
import System.Random
import Control.Monad
import Control.Exception
import qualified Data.ByteString.Char8 as B


type Matrix e = StorableArray (Int,Int) e

withMatrix :: Matrix e -> (Ptr e -> IO a) -> IO a
withMatrix = withStorableArray


-- To ensure the array is fully evaluated, force one element
--
evaluateMat :: (Ix i, Storable e) => i -> StorableArray i e -> IO (StorableArray i e)
evaluateMat l mat = (join $! evaluate (mat `readArray` l)) >> return mat

-- Generate a new random Array
--
randomMat :: (Ix i, Num e, Storable e, Random e, MArray StorableArray e IO)
           => i -> i -> IO (StorableArray i e)
randomMat l u = do
  rg <- newStdGen
  let -- The standard random number generator is too slow to generate really
      -- large vectors. Instead, we generate a short vector and repeat that.
      k     = 1000
      rands = take k (randomRs (-100,100) rg)

  newListArray (l,u) [rands !! (index (l,u) i`mod`k) | i <- range (l,u)] >>= evaluateMat l


--------------------------------------------------------------------------------
-- Reference implementation
--------------------------------------------------------------------------------

matMult :: (Num e, Storable e) => Matrix e -> Matrix e -> IO (Matrix e)
matMult mx my = do
  x <- unsafeFreeze mx
  y <- unsafeFreeze my
  let ((li, lj), (ui, uj))   = bounds x
      ((li',lj'),(ui',uj'))  = bounds y
      resultBounds | (lj,uj) == (li',ui') = ((li,lj'),(ui,uj'))
                   | otherwise            = error "matMult: incompatible bounds"

      mat = accumArray (+) 0 resultBounds
              [((i,j), x!(i,k) * y!(k,j)) | i <- range (li, ui),
                                            j <- range (lj',uj'),
                                            k <- range (lj, uj) ]
  unsafeThaw mat


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
  ptx     <- B.readFile "data/MatrixMul.ptx"
  (mdl,r) <- CUDA.loadDataEx ptx [CUDA.MaxRegisters 32]
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
  (ix,jx) <- getBounds xs
  (iy,jy) <- getBounds ys
  let sx = rangeSize (ix,jx)
      sy = rangeSize (iy,jy)

  dxs  <- CUDA.malloc sx
  dys  <- CUDA.malloc sy
  res  <- CUDA.malloc (rangeSize (ix,jy))

  flip onException (mapM_ CUDA.free [dxs,dys,res]) $ do
  withMatrix xs $ \p -> CUDA.pokeArray sx p dxs
  withMatrix ys $ \p -> CUDA.pokeArray sy p dys
  return (dxs, dys, res)


--
-- Run the test
--
testCUDA :: (Num e, Storable e) => Matrix e -> Matrix e -> IO (Matrix e)
testCUDA xs' ys' = doTest undefined xs' ys'
  where
    doTest :: (Num e', Storable e') => e' -> Matrix e' -> Matrix e' -> IO (Matrix e')
    doTest dummy xs ys = do
      (wz,_) <- getBounds xs
      (_,hz) <- getBounds ys

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
          CUDA.setParams     matMul [CUDA.VArg dx, CUDA.VArg dy, CUDA.VArg dz, CUDA.IArg (rangeSize wz), CUDA.IArg (rangeSize hz)]
          CUDA.setBlockShape matMul (BLOCK_SIZE,BLOCK_SIZE,1)
          CUDA.setSharedSize matMul (fromIntegral (2 * BLOCK_SIZE * BLOCK_SIZE * sizeOf dummy))
          CUDA.launch        matMul ((rangeSize wz) `div` BLOCK_SIZE, (rangeSize hz) `div` BLOCK_SIZE) Nothing
          CUDA.sync

          -- Copy back result
          --
          zs <- newArray_ (wz,hz)
          withMatrix zs $ \p -> CUDA.peekArray (rangeSize (wz,hz)) dz p
          return zs


--------------------------------------------------------------------------------
-- Test & Verify
--------------------------------------------------------------------------------

verify :: (Ord e, Fractional e, Storable e) => Matrix e -> Matrix e -> IO ()
verify ref arr = do
  as <- getElems arr
  bs <- getElems ref
  if similar as bs
    then putStrLn "Valid"
    else putStrLn "INVALID!"
  where
    similar xs ys = all (< epsilon) [abs ((x-y)/x) | x <- xs | y <- ys]
    epsilon       = 0.0001


main :: IO ()
main = do
  putStrLn "== Generating random matrices"
  xs <- randomMat (0,0) (28*BLOCK_SIZE, 42*BLOCK_SIZE)  :: IO (Matrix Float)
  ys <- randomMat (0,0) (42*BLOCK_SIZE, 128*BLOCK_SIZE) :: IO (Matrix Float)

  putStrLn "== Generating reference solution"
  ref <- matMult xs ys

  putStrLn "== Testing CUDA"
  mat <- testCUDA xs ys

  putStrLn "== Validating: "
  verify ref mat

