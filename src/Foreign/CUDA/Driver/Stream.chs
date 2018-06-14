{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Stream
-- Copyright : [2009..2017] Trevor L. McDonell
-- License   : BSD
--
-- Stream management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Stream (

  -- * Stream Management
  Stream(..), StreamFlag(..), StreamWriteFlag(..), StreamWaitFlag(..),
  create, createWithPriority, destroy, finished, block, getPriority, getContext,
  write, wait,

  defaultStream,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Types
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Context.Base                 ( Context(..) )
import Foreign.CUDA.Internal.C2HS

-- System
import Control.Exception                                ( throwIO )
import Control.Monad                                    ( liftM )
import Foreign
import Foreign.C
import Unsafe.Coerce


--------------------------------------------------------------------------------
-- Stream management
--------------------------------------------------------------------------------

-- |
-- Create a new stream.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4>
--
{-# INLINEABLE create #-}
create :: [StreamFlag] -> IO Stream
create !flags = resultIfOk =<< cuStreamCreate flags

{-# INLINE cuStreamCreate #-}
{# fun unsafe cuStreamCreate
  { alloca-         `Stream'       peekStream*
  , combineBitMasks `[StreamFlag]'             } -> `Status' cToEnum #}
  where
    peekStream = liftM Stream . peek


-- |
-- Create a stream with the given priority. Work submitted to
-- a higher-priority stream may preempt work already executing in a lower
-- priority stream.
--
-- The convention is that lower numbers represent higher priorities. The
-- default priority is zero. The range of meaningful numeric priorities can
-- be queried using 'Foreign.CUDA.Driver.Context.Config.getStreamPriorityRange'.
-- If the specified priority is outside the supported numerical range, it
-- will automatically be clamped to the highest or lowest number in the
-- range
--
-- Requires CUDA-5.5.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471>
--
{-# INLINEABLE createWithPriority #-}
createWithPriority :: StreamPriority -> [StreamFlag] -> IO Stream
#if CUDA_VERSION < 5050
createWithPriority _ _              = requireSDK 'createWithPriority 5.5
#else
createWithPriority !priority !flags = resultIfOk =<< cuStreamCreateWithPriority flags priority

{-# INLINE cuStreamCreateWithPriority #-}
{# fun unsafe cuStreamCreateWithPriority
  { alloca-         `Stream'         peekStream*
  , combineBitMasks `[StreamFlag]'
  , cIntConv        `StreamPriority'
  }
  -> `Status' cToEnum #}
  where
    peekStream = liftM Stream . peek
#endif


-- |
-- Destroy a stream. If the device is still doing work in the stream when
-- 'destroy' is called, the function returns immediately and the resources
-- associated with the stream will be released automatically once the
-- device has completed all work.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758>
--
{-# INLINEABLE destroy #-}
destroy :: Stream -> IO ()
destroy !st = nothingIfOk =<< cuStreamDestroy st

{-# INLINE cuStreamDestroy #-}
{# fun unsafe cuStreamDestroy
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Check if all operations in the stream have completed.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b>
--
{-# INLINEABLE finished #-}
finished :: Stream -> IO Bool
finished !st =
  cuStreamQuery st >>= \rv ->
  case rv of
    Success  -> return True
    NotReady -> return False
    _        -> throwIO (ExitCode rv)

{-# INLINE cuStreamQuery #-}
{# fun unsafe cuStreamQuery
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Wait until the device has completed all operations in the Stream.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad>
--
{-# INLINEABLE block #-}
block :: Stream -> IO ()
block !st = nothingIfOk =<< cuStreamSynchronize st

{-# INLINE cuStreamSynchronize #-}
{# fun cuStreamSynchronize
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Query the priority of a stream.
--
-- Requires CUDA-5.5.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484>
--
{-# INLINEABLE getPriority #-}
getPriority :: Stream -> IO StreamPriority
#if CUDA_VERSION < 5050
getPriority _   = requireSDK 'getPriority 5.5
#else
getPriority !st = resultIfOk =<< cuStreamGetPriority st

{-# INLINE cuStreamGetPriority #-}
{# fun unsafe cuStreamGetPriority
  { useStream `Stream'
  , alloca-   `StreamPriority' peekIntConv*
  }
  -> `Status' cToEnum #}
#endif


-- |
-- Query the context associated with a stream
--
-- Requires CUDA-9.2.
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484>
--
{-# INLINEABLE getContext #-}
getContext :: Stream -> IO Context
#if CUDA_VERSION < 9020
getContext _   = requireSDK 'getContext 9.2
#else
getContext !st = resultIfOk =<< cuStreamGetCtx st

{-# INLINE cuStreamGetCtx #-}
{# fun unsafe cuStreamGetCtx
  { useStream `Stream'
  , alloca-   `Context' peekCtx* } -> `Status' cToEnum #}
  where
    peekCtx = liftM Context . peek
#endif


#if CUDA_VERSION < 8000
data StreamWriteFlag
data StreamWaitFlag
#else
{# enum CUstreamWriteValue_flags as StreamWriteFlag
  { underscoreToCase }
  with prefix="CU_STREAM" deriving (Eq, Show, Bounded) #}

{# enum CUstreamWaitValue_flags as StreamWaitFlag
  { underscoreToCase }
  with prefix="CU_STREAM" deriving (Eq, Show, Bounded) #}
#endif


-- | Write a value to memory, (presumably) after all preceding work in the
-- stream has completed. Unless the option 'WriteValueNoMemoryBarrier' is
-- supplied, the write is preceded by a system-wide memory fence.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g091455366d56dc2f1f69726aafa369b0>
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gc8af1e8b96d7561840affd5217dd6830>
--
-- Requires CUDA-8.0 for 32-bit values.
--
-- Requires CUDA-9.0 for 64-bit values.
--
{-# INLINEABLE write #-}
write :: Storable a => DevicePtr a -> a -> Stream -> [StreamWriteFlag] -> IO ()
write ptr val stream flags =
  case sizeOf val of
    4 -> write32 (castDevPtr ptr) (unsafeCoerce val) stream flags
    8 -> write64 (castDevPtr ptr) (unsafeCoerce val) stream flags
    _ -> cudaError "Stream.write: can only write 32- and 64-bit values"

{-# INLINE write32 #-}
write32 :: DevicePtr Word32 -> Word32 -> Stream -> [StreamWriteFlag] -> IO ()
#if CUDA_VERSION < 8000
write32 _   _   _      _     = requireSDK 'write32 8.0
#else
write32 ptr val stream flags = nothingIfOk =<< cuStreamWriteValue32 stream ptr val flags

{-# INLINE cuStreamWriteValue32 #-}
{# fun unsafe cuStreamWriteValue32
  { useStream       `Stream'
  , useDeviceHandle `DevicePtr Word32'
  ,                 `Word32'
  , combineBitMasks `[StreamWriteFlag]'
  }
  -> `Status' cToEnum #}
  where
    useDeviceHandle = fromIntegral . ptrToIntPtr . useDevicePtr
#endif

{-# INLINE write64 #-}
write64 :: DevicePtr Word64 -> Word64 -> Stream -> [StreamWriteFlag] -> IO ()
#if CUDA_VERSION < 9000
write64 _   _   _      _     = requireSDK 'write64 9.0
#else
write64 ptr val stream flags = nothingIfOk =<< cuStreamWriteValue64 stream ptr val flags

{-# INLINE cuStreamWriteValue64 #-}
{# fun unsafe cuStreamWriteValue64
  { useStream       `Stream'
  , useDeviceHandle `DevicePtr Word64'
  ,                 `Word64'
  , combineBitMasks `[StreamWriteFlag]'
  }
  -> `Status' cToEnum #}
  where
    useDeviceHandle = fromIntegral . ptrToIntPtr . useDevicePtr
#endif


-- | Wait on a memory location. Work ordered after the operation will block
-- until the given condition on the memory is satisfied.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g629856339de7bc6606047385addbb398>
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6910c1258c5f15aa5d699f0fd60d6933>
--
-- Requires CUDA-8.0 for 32-bit values.
--
-- Requires CUDA-9.0 for 64-bit values.
--
{-# INLINEABLE wait #-}
wait :: Storable a => DevicePtr a -> a -> Stream -> [StreamWaitFlag] -> IO ()
wait ptr val stream flags =
  case sizeOf val of
    4 -> wait32 (castDevPtr ptr) (unsafeCoerce val) stream flags
    8 -> wait64 (castDevPtr ptr) (unsafeCoerce val) stream flags
    _ -> cudaError "Stream.wait: can only wait on 32- and 64-bit values"

{-# INLINE wait32 #-}
wait32 :: DevicePtr Word32 -> Word32 -> Stream -> [StreamWaitFlag] -> IO ()
#if CUDA_VERSION < 8000
wait32 _   _   _      _     = requireSDK 'wait32 8.0
#else
wait32 ptr val stream flags = nothingIfOk =<< cuStreamWaitValue32 stream ptr val flags

{-# INLINE cuStreamWaitValue32 #-}
{# fun unsafe cuStreamWaitValue32
  { useStream       `Stream'
  , useDeviceHandle `DevicePtr Word32'
  ,                 `Word32'
  , combineBitMasks `[StreamWaitFlag]'
  } -> `Status' cToEnum #}
  where
    useDeviceHandle = fromIntegral . ptrToIntPtr . useDevicePtr
#endif

{-# INLINE wait64 #-}
wait64 :: DevicePtr Word64 -> Word64 -> Stream -> [StreamWaitFlag] -> IO ()
#if CUDA_VERSION < 9000
wait64 _   _   _      _     = requireSDK 'wait64 9.0
#else
wait64 ptr val stream flags = nothingIfOk =<< cuStreamWaitValue64 stream ptr val flags

{-# INLINE cuStreamWaitValue64 #-}
{# fun unsafe cuStreamWaitValue64
  { useStream       `Stream'
  , useDeviceHandle `DevicePtr Word64'
  ,                 `Word64'
  , combineBitMasks `[StreamWaitFlag]'
  } -> `Status' cToEnum #}
  where
    useDeviceHandle = fromIntegral . ptrToIntPtr . useDevicePtr
#endif

