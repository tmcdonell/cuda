{-# LANGUAGE ForeignFunctionInterface, TypeFamilies #-}
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
  (
  )
  where

import Foreign.CUDA.Internal.C2HS


-- |
-- Multi-dimensional array for the GPU. If the device and host memory spaces are
-- discrete, the data will be transferred as necessary.
--
data DeviceArray d t = DeviceArray
  {
    aShape :: Shape d,                  -- ^ Dimensions of the data
    aSize  :: Int64,                    -- ^ Size of the data in bytes
    aPtr   :: ForeignPtr t              -- ^ Location of the data
  }


-- |
-- The shape of an array gives the number of elements in each dimension
--

-- data Shape = Nil ()
--            | D1 Int64
--            | D2 (Int64,Int64)
--            | D3 (Int64,Int64,Int64)
-- 
type family   Shape d
type instance Shape ()                  = ()
type instance Shape Int64               = Int64
type instance Shape (Int64,Int64)       = (Int64,Int64)
type instance Shape (Int64,Int64,Int64) = (Int64,Int64,Int64)

