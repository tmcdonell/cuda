{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
#ifdef USE_EMPTY_CASE
{-# LANGUAGE EmptyCase                #-}
#endif
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Graph.Capture
-- Copyright : [2018] Trevor L. McDonell
-- License   : BSD
--
-- Requires CUDA-10
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Graph.Capture (

  Status(..),
  start, stop, status,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

import Foreign.CUDA.Driver.Error                          hiding ( Status )
import Foreign.CUDA.Driver.Graph.Base
import Foreign.CUDA.Driver.Stream
import Foreign.CUDA.Internal.C2HS

import Control.Monad                                      ( liftM )

import Foreign
import Foreign.C


#if CUDA_VERSION < 10000
data Status

instance Enum Status where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("Status.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif

#else
{# enum CUstreamCaptureStatus as Status
  { underscoreToCase }
  with prefix="CU_STREAM_CAPTURE_STATUS" deriving (Eq, Show, Bounded) #}
#endif


--------------------------------------------------------------------------------
-- Graph capture
--------------------------------------------------------------------------------

-- | Begin graph capture on a stream
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1gea22d4496b1c8d02d0607bb05743532f>
--
-- @since 0.10.0.0
--
#if CUDA_VERSION < 10000
start :: Stream -> IO ()
start = requireSDK 'start 10.0
#else
{# fun unsafe cuStreamBeginCapture as start
  { useStream `Stream'
  }
  -> `()' checkStatus*- #}
#endif


-- | End graph capture on a stream
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c>
--
-- @since 0.10.0.0
--
#if CUDA_VERSION < 10000
stop :: Stream -> IO Graph
stop = requireSDK 'stop 10.0
#else
{# fun unsafe cuStreamEndCapture as stop
  { useStream `Stream'
  , alloca-   `Graph'  peekGraph*
  }
  -> `()' checkStatus*- #}
#endif


-- | Return a stream's capture status
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca>
--
-- @since 0.10.0.0
--
#if CUDA_VERSION < 10000
status :: Stream -> IO Status
status = requireSDK 'status 10.0
#else
{# fun unsafe cuStreamIsCapturing as status
  { useStream `Stream'
  , alloca-   `Status' peekEnum*
  }
  -> `()' checkStatus*- #}
#endif


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

#if CUDA_VERSION >= 10000
{-# INLINE peekGraph #-}
peekGraph :: Ptr {# type CUgraph #} -> IO Graph
peekGraph = liftM Graph . peek
#endif

