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

import Data.Array.Storable
import System.Random
import Foreign
import qualified Foreign.CUDA.Driver as CUDA


type Vector e = StorableArray Int e

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

  -- To ensure the array is fully evaluated, force one element
  -- (might not be necessary for StorableArray)
  vec <- newListArray (0,n-1) [rands !! (i`mod`k) | i <- [0..n-1]]
  vec `readArray` 0 >> return vec


--------------------------------------------------------------------------------
-- Reference implementation
--------------------------------------------------------------------------------

testRef :: Vector Float -> Vector Float -> IO (Vector Float)
testRef xs ys = error "not implemented yet"


--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

testCUDA :: Vector Float -> Vector Float -> IO (Vector Float)
testCUDA xs ys = error "not implemented yet"


--------------------------------------------------------------------------------
-- Test & Verify
--------------------------------------------------------------------------------

main :: IO ()
main = error "not implemented yet"

