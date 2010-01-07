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
randomArrR :: (Ix i, Num e, Storable e, Random e, MArray StorableArray e IO)
           => (i,i) -> (e,e) -> IO (StorableArray i e)
randomArrR (l,u) bnds = do
  rg <- newStdGen
  let -- The standard random number generator is too slow to generate really
      -- large vectors. Instead, we generate a short vector and repeat that.
      k     = 1000
      rands = take k (randomRs bnds rg)

  newListArray (l,u) [rands !! (index (l,u) i`mod`k) | i <- range (l,u)] >>= evaluateArr l


randomArr :: (Ix i, Num e, Storable e, Random e, MArray StorableArray e IO)
          => (i,i) -> IO (StorableArray i e)
randomArr (l,u) = randomArrR (l,u) (-1,1)


--
-- Verify similarity of two arrays
--
verify :: (Ix i, Ord e, Fractional e, Storable e)
       => StorableArray i e -> StorableArray i e -> IO (Bool)
verify ref arr = do
  as <- getElems arr
  bs <- getElems ref
  return (verifyList as bs)


--------------------------------------------------------------------------------
-- Lists
--------------------------------------------------------------------------------

randomListR :: (Num e, Random e, Storable e) => Int -> (e,e) -> IO [e]
randomListR len bnds = do
  rg <- newStdGen
  let -- The standard random number generator is too slow to generate really
      -- large vectors. Instead, we generate a short vector and repeat that.
      k     = 1000
      rands = take k (randomRs bnds rg)

  evaluate [rands !! (i`mod`k) | i <- [0..len-1]]

randomList :: (Num e, Random e, Storable e) => Int -> IO [e]
randomList len = randomListR len (-1,1)


verifyList :: (Ord e, Fractional e) => [e] -> [e] -> Bool
verifyList xs ys = all (< epsilon) [abs ((x-y)/(x+y+epsilon)) | x <- xs | y <- ys]
  where epsilon = 0.0005

