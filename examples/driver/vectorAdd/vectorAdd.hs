--------------------------------------------------------------------------------
--
-- Module    : vectorAdd
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Element-wise addition of two vectors
--
--------------------------------------------------------------------------------

module VectorAdd where

import Foreign
import qualified Foreign.CUDA.Driver as CUDA

import Data.Array.Storable
import System.Random
import Control.Monad
import Control.Exception


type Vector e = StorableArray Int e

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

testCUDA :: (Num e, Storable e) => Vector e -> Vector e -> IO (Vector e)
testCUDA xs ys = error "not implemented yet"


--------------------------------------------------------------------------------
-- Test & Verify
--------------------------------------------------------------------------------

main :: IO ()
main = error "not implemented yet"

