{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash                #-}
{-# LANGUAGE TemplateHaskell          #-}
#ifdef USE_EMPTY_CASE
{-# LANGUAGE EmptyCase                #-}
#endif
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Stream
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- Stream management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Stream (

  -- * Stream Management
  Stream(..), StreamPriority, StreamCallback,
  StreamFlag(..), StreamWriteFlag(..), StreamWaitFlag(..), StreamCallbackFlag,

  create, createWithPriority, destroy, finished, block, callback,
  getFlags, getPriority, getContext,
  write, wait,

  defaultStream,
  defaultStreamLegacy,
  defaultStreamPerThread,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Context.Base                   ( Context(..) )
import Foreign.CUDA.Internal.C2HS

-- System
import Control.Exception                                  ( throwIO )
import Control.Monad                                      ( liftM )
import Foreign
import Foreign.C
import Unsafe.Coerce

import GHC.Base
import GHC.Ptr
import GHC.Word


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A processing stream. All operations in a stream are synchronous and executed
-- in sequence, but operations in different non-default streams may happen
-- out-of-order or concurrently with one another.
--
-- Use 'Event's to synchronise operations between streams.
--
newtype Stream = Stream { useStream :: {# type CUstream #}}
  deriving (Eq, Show)

-- |
-- Priority of an execution stream. Work submitted to a higher priority
-- stream may preempt execution of work already executing in a lower
-- priority stream. Lower numbers represent higher priorities.
--
type StreamPriority = Int

-- |
-- Execution stream creation flags
--
#if CUDA_VERSION < 8000
data StreamFlag
data StreamWriteFlag
data StreamWaitFlag

instance Enum StreamFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("StreamFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif

instance Enum StreamWriteFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("StreamWriteFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif

instance Enum StreamWaitFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("StreamWaitFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif

#else
{# enum CUstream_flags as StreamFlag
  { underscoreToCase }
  with prefix="CU_STREAM" deriving (Eq, Show, Bounded) #}

{# enum CUstreamWriteValue_flags as StreamWriteFlag
  { underscoreToCase }
  with prefix="CU_STREAM" deriving (Eq, Show, Bounded) #}

{# enum CUstreamWaitValue_flags as StreamWaitFlag
  { underscoreToCase }
  with prefix="CU_STREAM" deriving (Eq, Show, Bounded) #}
#endif


-- | A 'Stream' callback function
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge5743a8c48527f1040107a68205c5ba9>
--
-- @since 0.10.0.0
--
type StreamCallback = {# type CUstreamCallback #}

data StreamCallbackFlag
instance Enum StreamCallbackFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("StreamCallbackFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif


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
  , fromIntegral    `StreamPriority'
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


-- | Query the flags of a given stream
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7>
--
-- @since 0.10.0.0
--
{-# INLINEABLE getFlags #-}
{# fun unsafe cuStreamGetFlags as getFlags
  { useStream `Stream'
  , alloca-   `[StreamFlag]' extract*
  }
  -> `()' checkStatus*- #}
  where
#if CUDA_VERSION < 8000
    extract _ = return []
#else
    extract p = extractBitMasks `fmap` peek p
#endif


-- |
-- Query the context associated with a stream
--
-- Requires CUDA-9.2.
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484>
--
-- @since 0.10.0.0
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
    _ -> cudaErrorIO "Stream.write: can only write 32- and 64-bit values"

{-# INLINE write32 #-}
write32 :: DevicePtr Word32 -> Word32 -> Stream -> [StreamWriteFlag] -> IO ()
#if CUDA_VERSION < 8000
write32 _   _   _      _     = requireSDK 'write32 8.0
#else
write32 ptr val stream flags = nothingIfOk =<< cuStreamWriteValue32_v2 stream ptr val flags

{-# INLINE cuStreamWriteValue32_v2 #-}
{# fun unsafe cuStreamWriteValue32_v2
  { useStream       `Stream'
  , useDeviceHandle `DevicePtr Word32'
  ,                 `Word32'
  , combineBitMasks `[StreamWriteFlag]'
  }
  -> `Status' cToEnum #}
#endif

{-# INLINE write64 #-}
write64 :: DevicePtr Word64 -> Word64 -> Stream -> [StreamWriteFlag] -> IO ()
#if CUDA_VERSION < 9000
write64 _   _   _      _     = requireSDK 'write64 9.0
#else
write64 ptr val stream flags = nothingIfOk =<< cuStreamWriteValue64_v2 stream ptr val flags

{-# INLINE cuStreamWriteValue64_v2 #-}
{# fun unsafe cuStreamWriteValue64_v2
  { useStream       `Stream'
  , useDeviceHandle `DevicePtr Word64'
  ,                 `Word64'
  , combineBitMasks `[StreamWriteFlag]'
  }
  -> `Status' cToEnum #}
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
    _ -> cudaErrorIO "Stream.wait: can only wait on 32- and 64-bit values"

{-# INLINE wait32 #-}
wait32 :: DevicePtr Word32 -> Word32 -> Stream -> [StreamWaitFlag] -> IO ()
#if CUDA_VERSION < 8000
wait32 _   _   _      _     = requireSDK 'wait32 8.0
#else
wait32 ptr val stream flags = nothingIfOk =<< cuStreamWaitValue32_v2 stream ptr val flags

{-# INLINE cuStreamWaitValue32_v2 #-}
{# fun unsafe cuStreamWaitValue32_v2
  { useStream       `Stream'
  , useDeviceHandle `DevicePtr Word32'
  ,                 `Word32'
  , combineBitMasks `[StreamWaitFlag]'
  } -> `Status' cToEnum #}
#endif

{-# INLINE wait64 #-}
wait64 :: DevicePtr Word64 -> Word64 -> Stream -> [StreamWaitFlag] -> IO ()
#if CUDA_VERSION < 9000
wait64 _   _   _      _     = requireSDK 'wait64 9.0
#else
wait64 ptr val stream flags = nothingIfOk =<< cuStreamWaitValue64_v2 stream ptr val flags

{-# INLINE cuStreamWaitValue64_v2 #-}
{# fun unsafe cuStreamWaitValue64_v2
  { useStream       `Stream'
  , useDeviceHandle `DevicePtr Word64'
  ,                 `Word64'
  , combineBitMasks `[StreamWaitFlag]'
  } -> `Status' cToEnum #}
#endif


-- | Add a callback to a compute stream. This function will be executed on the
-- host after all currently queued items in the stream have completed.
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483>
--
-- @since 0.10.0.0
--
{-# INLINEABLE callback #-}
{# fun unsafe cuStreamAddCallback as callback
  { useStream       `Stream'
  , id              `StreamCallback'
  , id              `Ptr ()'
  , combineBitMasks `[StreamCallbackFlag]'
  } -> `()' checkStatus*- #}


-- | The default execution stream. This can be configured to have either
-- 'defaultStreamLegacy' or 'defaultStreamPerThread' synchronisation behaviour.
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream>
--
{-# INLINE defaultStream #-}
defaultStream :: Stream
defaultStream = Stream (Ptr (int2Addr# 0#))


-- | The legacy default stream is an implicit stream which synchronises with all
-- other streams in the same 'Context', except for non-blocking streams.
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream>
--
-- @since 0.10.0.0
--
{-# INLINE defaultStreamLegacy #-}
defaultStreamLegacy :: Stream
defaultStreamLegacy = Stream (Ptr (int2Addr# 0x1#))


-- | The per-thread default stream is an implicit stream local to both the
-- thread and the calling 'Context', and which does not synchronise with other
-- streams (just like explicitly created streams). The per-thread default stream
-- is not a non-blocking stream and will synchronise with the legacy default
-- stream if both are used in the same program.
--
-- <file:///Developer/NVIDIA/CUDA-9.2/doc/html/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream>
--
-- @since 0.10.0.0
--
{-# INLINE defaultStreamPerThread #-}
defaultStreamPerThread :: Stream
defaultStreamPerThread = Stream (Ptr (int2Addr# 0x2#))


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

-- Use a device pointer as an opaque handle type
--
{-# INLINE useDeviceHandle #-}
useDeviceHandle :: DevicePtr a -> {# type CUdeviceptr #}
useDeviceHandle (DevicePtr (Ptr addr#)) =
  CULLong (W64# (wordToWord64# (int2Word# (addr2Int# addr#))))

#if __GLASGOW_HASKELL__ < 904
{-# INLINE wordToWord64# #-}
wordToWord64# :: Word# -> Word#
wordToWord64# x = x
#endif

