{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Marshal
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Memory allocation and marshalling support for CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Marshal
  (
    DevicePtr,

    -- ** Dynamic allocation
    malloc,
    malloc2D,
    malloc3D,
    memset,
    memset2D,
    memset3D,

    -- ** Marshalling

    -- ** Combined allocation and marshalling

    -- ** Copying
  )
  where

import Foreign.CUDA.Error
import Foreign.CUDA.Stream
import Foreign.CUDA.Internal.C2HS hiding (malloc)

#include <cuda_runtime_api.h>
{# context lib="cudart" #}


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A reference to data stored on the device. It is automatically freed.
--
type DevicePtr a = ForeignPtr a

withDevicePtr :: DevicePtr a -> (Ptr a -> IO b) -> IO b
withDevicePtr =  withForeignPtr

newDevicePtr   :: Ptr () -> IO (DevicePtr ())
newDevicePtr p =  (doAutoRelease free_) >>= \f -> newForeignPtr f p

--
-- Memory copy
--
{# enum cudaMemcpyKind as CopyDirection {}
    with prefix="cudaMemcpy" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Dynamic allocation
--------------------------------------------------------------------------------

-- |
-- Allocate the specified number of bytes in linear memory on the device. The
-- memory is suitably aligned for any kind of variable, and is not cleared.
--
malloc       :: Int64 -> IO (Either String (DevicePtr ()))
malloc bytes = do
    (rv,ptr) <- cudaMalloc bytes
    case rv of
        Success -> Right `fmap` newDevicePtr ptr
        _       -> return . Left . describe $ rv

{# fun unsafe cudaMalloc
    { alloca-  `Ptr ()' peek* ,
      cIntConv `Int64'        } -> `Status' cToEnum #}

-- |
-- Allocate at least @width * height@ bytes of linear memory on the device. The
-- function may pad the allocation to ensure corresponding pointers in each row
-- meet coalescing requirements. The actual allocation width is returned.
--
malloc2D :: (Int64, Int64)              -- ^ allocation (width,height) in bytes
         -> IO (Either String (DevicePtr (),Int64))
malloc2D (width,height) =  do
    (rv,ptr,pitch) <- cudaMallocPitch width height
    case rv of
        Success -> (\p -> Right (p,pitch)) `fmap` newDevicePtr ptr
        _       -> return . Left . describe $ rv

{# fun unsafe cudaMallocPitch
    { alloca-  `Ptr ()' peek*        ,
      alloca-  `Int64'  peekIntConv* ,
      cIntConv `Int64'               ,
      cIntConv `Int64'               } -> `Status' cToEnum #}


-- |
-- Allocate at least @width * height * depth@ bytes of linear memory on the
-- device. The function may pad the allocation to ensure hardware alignment
-- requirements are met. The actual allocation pitch is returned
--
malloc3D :: (Int64,Int64,Int64)         -- ^ allocation (width,height,depth) in bytes
         -> IO (Either String (DevicePtr (),Int64))
malloc3D = error "not implemented yet"


-- |
-- Free previously allocated memory on the device
--
-- free   :: Ptr () -> IO (Maybe String)
-- free p =  nothingIfOk `fmap` cudaFree p

free_   :: Ptr () -> IO ()
free_ p =  throwIf_ (/= Success) (describe) (cudaFree p)

{# fun unsafe cudaFree
    { id `Ptr ()' } -> `Status' cToEnum #}

foreign import ccall "wrapper"
    doAutoRelease   :: (Ptr () -> IO ()) -> IO (FunPtr (Ptr () -> IO ()))


-- |
-- Initialise device memory to a given value
--
memset                  :: DevicePtr () -> Int64 -> Int -> IO (Maybe String)
memset ptr bytes symbol =  nothingIfOk `fmap` cudaMemset ptr symbol bytes

{# fun unsafe cudaMemset
    { withDevicePtr* `DevicePtr ()' ,
                     `Int'          ,
      cIntConv       `Int64'        } -> `Status' cToEnum #}


-- |
-- Initialise a matrix to a given value
--
memset2D :: DevicePtr ()                -- ^ The device memory
         -> (Int64,Int64)               -- ^ The (width,height) of the matrix in bytes
         -> Int64                       -- ^ The allocation pitch, as returned by 'malloc2D'
         -> Int                         -- ^ Value to set for each byte
         -> IO (Maybe String)
memset2D ptr (width,height) pitch symbol = nothingIfOk `fmap` cudaMemset2D ptr pitch symbol width height

{# fun unsafe cudaMemset2D
    { withDevicePtr* `DevicePtr ()' ,
      cIntConv       `Int64'        ,
                     `Int'          ,
      cIntConv       `Int64'        ,
      cIntConv       `Int64'        } -> `Status' cToEnum #}


-- |
-- Initialise the elements of a 3D array to a given value
--
memset3D :: DevicePtr ()                -- ^ The device memory
         -> (Int64,Int64,Int64)         -- ^ The (width,height,depth) of the array in bytes
         -> Int64                       -- ^ The allocation pitch, as returned by 'malloc3D'
         -> Int                         -- ^ Value to set for each byte
         -> IO (Maybe String)
memset3D = error "not implemented yet"


--------------------------------------------------------------------------------
-- Marshalling
--------------------------------------------------------------------------------

-- peek
-- poke


--------------------------------------------------------------------------------
-- Combined allocation and marshalling
--------------------------------------------------------------------------------

-- newArray
-- withArray


--------------------------------------------------------------------------------
-- Copying
--------------------------------------------------------------------------------

-- |
-- Copy data between host and device
--
memcpy :: Ptr a                 -- ^ destination
       -> Ptr a                 -- ^ source
       -> Int64                 -- ^ number of bytes
       -> CopyDirection
       -> IO (Maybe String)
memcpy dst src bytes dir =  nothingIfOk `fmap` cudaMemcpy dst src bytes dir

{# fun unsafe cudaMemcpy
    { castPtr   `Ptr a'         ,
      castPtr   `Ptr a'         ,
      cIntConv  `Int64'         ,
      cFromEnum `CopyDirection' } -> `Status' cToEnum #}

-- |
-- Copy data between host and device asynchronously
--
memcpyAsync :: Ptr a            -- ^ destination
            -> Ptr a            -- ^ source
            -> Int64            -- ^ number of bytes
            -> CopyDirection
            -> Stream
            -> IO (Maybe String)
memcpyAsync dst src bytes dir stream =  nothingIfOk `fmap` cudaMemcpyAsync dst src bytes dir stream

{# fun unsafe cudaMemcpyAsync
    { castPtr    `Ptr a'         ,
      castPtr    `Ptr a'         ,
      cIntConv   `Int64'         ,
      cFromEnum  `CopyDirection' ,
      cIntConv   `Stream'        } -> `Status' cToEnum #}

