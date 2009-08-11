{-# LANGUAGE GADTs, FlexibleInstances #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Array
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Multi-dimensional arrays for CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Array
  where

-- Friends
import Foreign.CUDA.Marshal (DevicePtr)


-- |
-- Mutable array representation
--
type ArrayData e = DevicePtr e

-- |
-- Multi-dimensional array for the GPU. If the device and host memory spaces are
-- discrete, the data will be transferred as necessary.
--
data Array dim e where
    Array :: (Ix dim)
          => dim
          -> ArrayData e
          -> Array dim e

-- |
-- Index representation (which are nested pairs)
--
class Ix ix where
    dim  :: ix -> Int           -- ^ Number of dimensions
    size :: ix -> Int           -- ^ Total number of elements

instance Ix () where
    dim ()  = 0
    size () = 1

instance Ix ix => Ix (ix, Int) where
    dim (sh,_)  = dim sh + 1
    size (sh,s) = size sh * s

-- |
-- Shorthand for shape representations
--
type DIM0 = ()
type DIM1 = ((),Int)

-- |
-- Special case for singleton and one-dimensional arrays
--
type Scalar e = Array DIM0 e
type Vector e = Array DIM1 e

