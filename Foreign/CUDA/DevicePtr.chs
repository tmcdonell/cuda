{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.DevicePtr
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- References to objects stored on the CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.DevicePtr
  (
    DevicePtr,
    withDevicePtr,
    withDevicePtrByteOff,

    castDevicePtr,
    newDevicePtr,
    finalizeDevicePtr
  )
  where

import Foreign.C
import Foreign.Ptr
import Foreign.Concurrent
import Foreign.ForeignPtr hiding (newForeignPtr, addForeignPtrFinalizer)

import Foreign.CUDA.Error
import Foreign.CUDA.Internal.C2HS

#include <cuda_runtime_api.h>
{# context lib="cudart" #}


--------------------------------------------------------------------------------
-- Device Data Pointer
--------------------------------------------------------------------------------

-- |
-- A reference to data stored on the device
--
newtype DevicePtr a = DevicePtr (ForeignPtr a)
  deriving (Eq,Show,Ord)

-- |
-- Unwrap the device pointer, yielding a device memory address that can be
-- passed to a kernel function callable from the host code. This allows the type
-- system to strictly separate host and device memory references, a non-obvious
-- distinction in pure CUDA.
--
withDevicePtr :: DevicePtr a -> (Ptr a -> IO b) -> IO b
withDevicePtr (DevicePtr p) = withForeignPtr p

-- |
-- Similar to `withDevicePtr', but with the address advanced by the given offset
-- in bytes.
--
withDevicePtrByteOff :: DevicePtr a -> Int -> (Ptr a -> IO b) -> IO b
withDevicePtrByteOff (DevicePtr dptr) offset action =
    withForeignPtr dptr $ \p -> action (p `plusPtr` offset)

-- |
-- Explicitly cast a DevicePtr parameterised by one type into another type
--
castDevicePtr :: DevicePtr a -> DevicePtr b
castDevicePtr (DevicePtr p) = DevicePtr $ castForeignPtr p

-- |
-- Wrap a device memory pointer. The memory will be freed once the last
-- reference to the data is dropped, although there is no guarantee as to
-- exactly when this will occur.
--

-- XXX: This has not been extensively tested. Kernel calls may be asynchronous,
-- under which this might try to release memory while it is still being used?
--
newDevicePtr   :: Ptr a -> IO (DevicePtr a)
newDevicePtr p =  DevicePtr `fmap` newForeignPtr p (free_ p)

-- |
-- Release the memory associated with a device pointer immediately
--
finalizeDevicePtr :: DevicePtr a -> IO ()
finalizeDevicePtr (DevicePtr p) = finalizeForeignPtr p


--
-- Release previously allocated memory on the device
--
-- TODO: This should probably live in Foreign.CUDA.Marshal.Alloc
--
free_   :: Ptr a -> IO ()
free_ p =  cudaFree p >>= \rv ->
    case rv of
        Success -> return ()
        _       -> error (describe rv)

{# fun unsafe cudaFree
    { castPtr `Ptr a' } -> `Status' cToEnum #}

-- foreign import ccall "wrapper"
--     doAutoRelease   :: (Ptr a -> IO ()) -> IO (FunPtr (Ptr a -> IO ()))


