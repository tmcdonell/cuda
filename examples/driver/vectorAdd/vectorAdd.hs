{-# LANGUAGE FlexibleContexts, ParallelListComp #-}
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

import Foreign
import Foreign.CUDA.Driver (forceEither)
import qualified Foreign.CUDA.Driver as CUDA

import Data.Array.Storable
import System.Random
import Control.Monad
import Control.Exception


type Vector e = StorableArray Int e

withVector :: Vector e -> (Ptr e -> IO a) -> IO a
withVector = withStorableArray


-- To ensure the array is fully evaluated, force one element
--
evaluateVec :: Storable e => Vector e -> IO (Vector e)
evaluateVec vec = (join $ evaluate (vec `readArray` 0)) >> return vec

-- Generate a new random vector
--
randomVec :: (Num e, Storable e, Random e, MArray StorableArray e IO)
          => Int -> IO (Vector e)
randomVec n = do
  rg <- newStdGen
  let -- The standard random number generator is too slow to generate really
      -- large vectors. Instead, we generate a short vector and repeat that.
      k     = 1000
      rands = take k (randomRs (-100,100) rg)

  newListArray (0,n-1) [rands !! (i`mod`k) | i <- [0..n-1]] >>= evaluateVec


--------------------------------------------------------------------------------
-- Reference implementation
--------------------------------------------------------------------------------

testRef :: (Num e, Storable e) => Vector e -> Vector e -> IO (Vector e)
testRef xs ys = do
  (i,j) <- getBounds xs
  res   <- newArray_ (i,j)
  forM_ [i..j] (add res)
  evaluateVec res
  where
    add res idx = do
      a <- readArray xs idx
      b <- readArray ys idx
      writeArray res idx (a+b)

--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

initCUDA :: IO (CUDA.Fun)
initCUDA = do
  res <- CUDA.initialise []
  case res of
    Just e  -> error e
    Nothing -> do
      dev <- forceEither `fmap` CUDA.get 0
      ctx <- forceEither `fmap` CUDA.create dev []
      mdl <- forceEither `fmap` CUDA.loadFile   "data/VectorAdd.ptx"
      fun <- forceEither `fmap` CUDA.getFun mdl "VecAdd"
      return fun

initData :: (Num e, Storable e)
         => Vector e -> Vector e -> IO (CUDA.DevicePtr e, CUDA.DevicePtr e, CUDA.DevicePtr e)
initData xs ys = do
  (m,n) <- getBounds xs
  let len = (n-m+1)
  dxs   <- forceEither `fmap` CUDA.malloc len
  dys   <- forceEither `fmap` CUDA.malloc len
  res   <- forceEither `fmap` CUDA.malloc len
  withVector xs $ \p -> CUDA.pokeArray len p dxs
  withVector ys $ \p -> CUDA.pokeArray len p dys
  return (dxs, dys, res)


testCUDA :: (Num e, Storable e) => Vector e -> Vector e -> IO (Vector e)
testCUDA xs ys = do
  -- Initialise environment and copy over test data
  --
  addVec  <- initCUDA
  (x,y,z) <- initData xs ys
  (m,n)   <- getBounds xs
  let len = (n-m+1)

  -- Repeat test many times...
  --
  CUDA.setParams     addVec [CUDA.VArg x, CUDA.VArg y, CUDA.VArg z, CUDA.IArg len]
  CUDA.setBlockShape addVec (128,1,1)
  CUDA.launch        addVec (len+128-1 `div` 128, 1) Nothing
--  CUDA.sync

  -- Copy back result
  --
  zs <- newArray_ (m,n)
  withVector zs $ \p -> CUDA.peekArray len z p

  -- Cleanup and exit
  --
  mapM_ CUDA.free [x,y,z]
  return zs


--------------------------------------------------------------------------------
-- Test & Verify
--------------------------------------------------------------------------------

verify :: (Ord e, Fractional e, Storable e) => Vector e -> Vector e -> IO ()
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
  xs  <- randomVec 10000 :: IO (Vector Float)
  ys  <- randomVec 10000 :: IO (Vector Float)

  ref <- testRef  xs ys
  arr <- testCUDA xs ys
  verify ref arr

