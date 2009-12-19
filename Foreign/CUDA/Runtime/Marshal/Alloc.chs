{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Marshal.Alloc
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Memory allocation for CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Marshal.Alloc
  (
    -- ** Dynamic allocation
    --
    -- Basic methods to allocate a block of memory on the device. The memory
    -- should be deallocated using 'free' when no longer necessary. The
    -- advantage is that failed allocations will not raise an exception.
    --
    free,
    malloc,
    malloc2D,
    malloc3D,
    memset,
    memset2D,
    memset3D,

    -- ** Local allocation
    --
    -- Execute a computation, passing as argument a pointer to a newly allocated
    -- block of memory. The memory is freed when the computation terminates.
    --
    alloca,
    allocaBytes,
    allocaBytesMemset,

    -- ** Copying
    --
    -- Basic copying between the host and device
    --
    memcpy,
    CopyDirection(..)
  )
  where

import Data.Int
import Control.Exception

import Foreign.C
import Foreign.Ptr
import Foreign.Storable (Storable)
import qualified Foreign.Storable as F
import qualified Foreign.Marshal  as F

import Foreign.CUDA.Runtime.DevicePtr
import Foreign.CUDA.Runtime.Error
-- import Foreign.CUDA.Runtime.Stream
import Foreign.CUDA.Internal.C2HS

#include <cuda_runtime_api.h>
{# context lib="cudart" #}


--------------------------------------------------------------------------------
-- Dynamic allocation
--------------------------------------------------------------------------------

-- |
-- Allocate the specified number of bytes in linear memory on the device. The
-- memory is suitably aligned for any kind of variable, and is not cleared.
--
malloc :: Int64 -> IO (DevicePtr a)
malloc bytes =  do
  (rv,ptr) <- cudaMalloc bytes
  case rv of
      Success -> newDevicePtr (castPtr ptr)
      _       -> resultIfOk (rv,undefined)

{# fun unsafe cudaMalloc
  { alloca'- `Ptr ()' peek'*
  , cIntConv `Int64'         } -> `Status' cToEnum #}
  where
    -- C-> Haskell doesn't like qualified imports in marshaller specifications
    alloca' = F.alloca
    peek'   = F.peek


-- |
-- Allocate at least @width * height@ bytes of linear memory on the device. The
-- function may pad the allocation to ensure corresponding pointers in each row
-- meet coalescing requirements. The actual allocation width is returned.
--
malloc2D :: (Int64, Int64)              -- ^ allocation (width,height) in bytes
         -> IO (DevicePtr a, Int64)
malloc2D (width,height) =  do
  (rv,ptr,pitch) <- cudaMallocPitch width height
  case rv of
      Success -> (\p -> (p,pitch)) `fmap` newDevicePtr (castPtr ptr)
      _       -> resultIfOk (rv,undefined)

{# fun unsafe cudaMallocPitch
  { alloca'-  `Ptr ()' peek'*
  , alloca'-  `Int64'  peekIntConv*
  , cIntConv  `Int64'
  , cIntConv  `Int64'               } -> `Status' cToEnum #}
  where
    alloca' :: Storable a => (Ptr a -> IO b) -> IO b
    alloca' =  F.alloca
    peek'   =  F.peek


-- |
-- Allocate at least @width * height * depth@ bytes of linear memory on the
-- device. The function may pad the allocation to ensure hardware alignment
-- requirements are met. The actual allocation pitch is returned
--
malloc3D :: (Int64,Int64,Int64)         -- ^ allocation (width,height,depth) in bytes
         -> IO (DevicePtr a, Int64)
malloc3D = moduleErr "malloc3D" "not implemented yet"


-- |
-- Free previously allocated memory on the device
--
free :: DevicePtr a -> IO ()
free =  finalizeDevicePtr

--free   :: DevicePtr a -> IO (Maybe String)
--free p =  nothingIfOk `fmap` (withDevicePtr p cudaFree)


-- |
-- Initialise device memory to a given value
--
memset :: DevicePtr a                   -- ^ The device memory
       -> Int64                         -- ^ Number of bytes
       -> Int                           -- ^ Value to set for each byte
       -> IO ()
memset ptr bytes symbol =
  nothingIfOk =<< cudaMemset ptr symbol bytes

{# fun unsafe cudaMemset
  { withDevPtrCast* `DevicePtr a'
  ,                 `Int'
  , cIntConv        `Int64'       } -> `Status' cToEnum #}
  where
    withDevPtrCast = withDevicePtr . castDevicePtr


-- |
-- Initialise a matrix to a given value
--
memset2D :: DevicePtr a                 -- ^ The device memory
         -> (Int64, Int64)              -- ^ The (width,height) of the matrix in bytes
         -> Int64                       -- ^ The allocation pitch, as returned by 'malloc2D'
         -> Int                         -- ^ Value to set for each byte
         -> IO ()
memset2D ptr (width,height) pitch symbol =
  nothingIfOk =<< cudaMemset2D ptr pitch symbol width height

{# fun unsafe cudaMemset2D
  { withDevPtrCast* `DevicePtr a'
  , cIntConv        `Int64'
  ,                 `Int'
  , cIntConv        `Int64'
  , cIntConv        `Int64'       } -> `Status' cToEnum #}
  where
    withDevPtrCast = withDevicePtr . castDevicePtr


-- |
-- Initialise the elements of a 3D array to a given value
--
memset3D :: DevicePtr a                 -- ^ The device memory
         -> (Int64,Int64,Int64)         -- ^ The (width,height,depth) of the array in bytes
         -> Int64                       -- ^ The allocation pitch, as returned by 'malloc3D'
         -> Int                         -- ^ Value to set for each byte
         -> IO ()
memset3D = moduleErr "memset3D" "not implemented yet"


--------------------------------------------------------------------------------
-- Local allocation
--------------------------------------------------------------------------------

-- |
-- Execute a computation, passing a pointer to a temporarily allocated block of
-- memory
--
alloca :: Storable a => (DevicePtr a -> IO b) -> IO b
alloca =  doAlloca undefined
  where
    doAlloca   :: Storable a' => a' -> (DevicePtr a' -> IO b') -> IO b'
    doAlloca x =  allocaBytes (fromIntegral (F.sizeOf x))


-- |
-- Execute a computation, passing a pointer to a block of 'n' bytes of memory
--
allocaBytes :: Int64 -> (DevicePtr a -> IO b) -> IO b
allocaBytes bytes =
  bracket (malloc bytes) free


-- |
-- Execute a computation, passing a pointer to a  block of memory initialised to
-- a given value
--
allocaBytesMemset :: Int64 -> Int -> (DevicePtr a -> IO b) -> IO b
allocaBytesMemset bytes symbol action =
  allocaBytes bytes $ \dptr ->
  memset dptr bytes symbol  >>
  action dptr


--------------------------------------------------------------------------------
-- Copying
--------------------------------------------------------------------------------

--
-- Memory copy
--
{# enum cudaMemcpyKind as CopyDirection {}
    with prefix="cudaMemcpy" deriving (Eq, Show) #}

-- |
-- Copy data between host and device
--
memcpy :: Ptr a                 -- ^ destination
       -> Ptr a                 -- ^ source
       -> Int64                 -- ^ number of bytes
       -> CopyDirection
       -> IO ()
memcpy dst src bytes dir =
    nothingIfOk =<< cudaMemcpy dst src bytes dir

{# fun unsafe cudaMemcpy
  { castPtr   `Ptr a'
  , castPtr   `Ptr a'
  , cIntConv  `Int64'
  , cFromEnum `CopyDirection' } -> `Status' cToEnum #}


{-
-- |
-- Copy data between host and device asynchronously
--
memcpyAsync :: Stream           -- ^ asynchronous stream to operate over
            -> Ptr a            -- ^ destination
            -> Ptr a            -- ^ source
            -> Int64            -- ^ number of bytes
            -> CopyDirection
            -> IO (Maybe String)
memcpyAsync stream dst src bytes dir =
    nothingIfOk `fmap` cudaMemcpyAsync dst src bytes dir stream

{# fun unsafe cudaMemcpyAsync
    { castPtr    `Ptr a'         ,
      castPtr    `Ptr a'         ,
      cIntConv   `Int64'         ,
      cFromEnum  `CopyDirection' ,
      cIntConv   `Stream'        } -> `Status' cToEnum #}
-}

--------------------------------------------------------------------------------
-- Auxiliary utilities
--------------------------------------------------------------------------------

--
-- Errors
--
moduleErr :: String -> String -> a
moduleErr fun msg =  error ("Foreign.CUDA." ++ fun ++ ':':' ':msg)

