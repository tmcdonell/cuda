{-# LANGUAGE FlexibleContexts, ParallelListComp #-}
--------------------------------------------------------------------------------
--
-- Module    : RandomVector
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Storable multi-dimensional arrays and lists of random numbers
--
--------------------------------------------------------------------------------

module RandomVector
  (
    Storable,
    module RandomVector,
    module Data.Array.Storable
  )
  where

import Foreign                                          (Ptr, Storable)
import Control.Monad                                    (join)
import Control.Exception                                (evaluate)
import Data.Array.Storable
import System.Random


--------------------------------------------------------------------------------
-- Arrays
--------------------------------------------------------------------------------

type Vector e = StorableArray Int e
type Matrix e = StorableArray (Int,Int) e

withVector :: Vector e -> (Ptr e -> IO a) -> IO a
withVector = withStorableArray

withMatrix :: Matrix e -> (Ptr e -> IO a) -> IO a
withMatrix = withStorableArray


--
-- To ensure the array is fully evaluated, force one element
--
evaluateArr :: (Ix i, MArray StorableArray e IO)
            => i -> StorableArray i e -> IO (StorableArray i e)
evaluateArr l arr = (join $ evaluate (arr `readArray` l)) >> return arr


--
-- Generate a new random array
--
randomArr :: (Ix i, Num e, Storable e, Random e, MArray StorableArray e IO)
          => (i,i) -> IO (StorableArray i e)
randomArr (l,u) = do
  rg <- newStdGen
  let -- The standard random number generator is too slow to generate really
      -- large vectors. Instead, we generate a short vector and repeat that.
      k     = 1000
      rands = take k (randomRs (-1,1) rg)

  newListArray (l,u) [rands !! (index (l,u) i`mod`k) | i <- range (l,u)] >>= evaluateArr l


--
-- Verify similarity of two arrays
--
verify :: (Ix i, Ord e, Fractional e, Storable e)
       => StorableArray i e -> StorableArray i e -> IO (Bool)
verify ref arr = do
  as <- getElems arr
  bs <- getElems ref
  return (similar as bs)
  where
    similar xs ys = all (< epsilon) [abs ((x-y)/x) | x <- xs | y <- ys]
    epsilon       = 0.001


--------------------------------------------------------------------------------
-- Lists
--------------------------------------------------------------------------------

randomList :: (Num e, Random e, Storable e) => Int -> IO [e]
randomList len = do
  rg <- newStdGen
  let -- The standard random number generator is too slow to generate really
      -- large vectors. Instead, we generate a short vector and repeat that.
      k     = 1000
      rands = take k (randomRs (-1,1) rg)

  evaluate [rands !! (i`mod`k) | i <- [0..len-1]]

