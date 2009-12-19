--------------------------------------------------------------------------------
--
-- Module    : VectorAdd
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Element-wise addition of two vectors
--
--------------------------------------------------------------------------------

module Main where

-- Friends
import RandomVector

-- System
import Numeric
import Control.Monad
import Control.Exception
import Data.Array.Storable
import qualified Data.ByteString.Char8 as B
import qualified Foreign.CUDA.Driver   as CUDA


--------------------------------------------------------------------------------
-- Reference implementation
--------------------------------------------------------------------------------

testRef :: (Num e, Storable e) => Vector e -> Vector e -> IO (Vector e)
testRef xs ys = do
  (i,j) <- getBounds xs
  res   <- newArray_ (i,j)
  forM_ [i..j] (add res)
  return res
  where
    add res idx = do
      a <- readArray xs idx
      b <- readArray ys idx
      writeArray res idx (a+b)

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
  ptx     <- B.readFile "data/vector_add.ptx"
  (mdl,r) <- CUDA.loadDataEx ptx [CUDA.MaxRegisters 32]
  fun     <- CUDA.getFun mdl "VecAdd"

  putStrLn $ ">> PTX JIT compilation (" ++ showFFloat (Just 2) (CUDA.jitTime r) " ms)"
  B.putStrLn (CUDA.jitInfoLog r)
  return (ctx,fun)

--
-- Allocate some memory, and copy over the input data to the device. Should
-- probably catch allocation exceptions individually...
--
initData :: (Num e, Storable e)
         => Vector e -> Vector e -> IO (CUDA.DevicePtr e, CUDA.DevicePtr e, CUDA.DevicePtr e)
initData xs ys = do
  (m,n) <- getBounds xs
  let len = (n-m+1)
  dxs   <- CUDA.malloc len
  dys   <- CUDA.malloc len
  res   <- CUDA.malloc len

  flip onException (mapM_ CUDA.free [dxs,dys,res]) $ do
  withVector xs $ \p -> CUDA.pokeArray len p dxs
  withVector ys $ \p -> CUDA.pokeArray len p dys
  return (dxs, dys, res)


--
-- Run the test
--
testCUDA :: (Num e, Storable e) => Vector e -> Vector e -> IO (Vector e)
testCUDA xs ys = do
  (m,n)   <- getBounds xs
  let len = (n-m+1)

  -- Initialise environment and copy over test data
  --
  putStrLn ">> Initialising"
  bracket initCUDA (\(ctx,_) -> CUDA.destroy ctx) $ \(_,addVec) -> do

  -- Ensure we release the memory, even if there was an error
  --
  putStrLn ">> Executing"
  bracket
    (initData xs ys)
    (\(dx,dy,dz) -> mapM_ CUDA.free [dx,dy,dz]) $
     \(dx,dy,dz) -> do
      -- Repeat test many times...
      --
      CUDA.setParams     addVec [CUDA.VArg dx, CUDA.VArg dy, CUDA.VArg dz, CUDA.IArg len]
      CUDA.setBlockShape addVec (128,1,1)
      CUDA.launch        addVec ((len+128-1) `div` 128, 1) Nothing
      CUDA.sync

      -- Copy back result
      --
      zs <- newArray_ (m,n)
      withVector zs $ \p -> CUDA.peekArray len dz p
      return zs


--------------------------------------------------------------------------------
-- Test & Verify
--------------------------------------------------------------------------------

main :: IO ()
main = do
  putStrLn "== Generating random vectors"
  xs  <- randomArr (1,10000) :: IO (Vector Float)
  ys  <- randomArr (1,10000) :: IO (Vector Float)

  putStrLn "== Generating reference solution"
  ref <- testRef  xs ys

  putStrLn "== Testing CUDA"
  arr <- testCUDA xs ys

  putStr   "== Validating: "
  verify ref arr >>= \rv -> putStrLn $ if rv then "Ok!" else "INVALID!"

