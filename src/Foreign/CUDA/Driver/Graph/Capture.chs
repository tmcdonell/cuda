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
-- Copyright : [2018..2023] Trevor L. McDonell
-- License   : BSD
--
-- Requires CUDA-10
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Graph.Capture (

  Status(..), Mode(..),
  start, stop, status, info, mode,

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

#if CUDA_VERSION < 10010
data Mode

instance Enum Mode where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("Mode.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif

#else
{# enum CUstreamCaptureMode as Mode
  { underscoreToCase }
  with prefix="CU_STREAM_CAPTURE_MODE" deriving (Eq, Show, Bounded) #}
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
start :: Stream -> Mode -> IO ()
start = requireSDK 'start 10.0
#elif CUDA_VERSION < 10010
start :: Stream -> Mode -> IO ()
start s _ = cuStreamBeginCapture s
  where
    {# fun unsafe cuStreamBeginCapture
      { useStream `Stream'
      }
      -> `()' checkStatus*- #}
#else
{# fun unsafe cuStreamBeginCapture as start
  { useStream `Stream'
  ,           `Mode'
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


-- | Query the capture status of a stream and get an id for the capture
-- sequence, which is unique over the lifetime of the process.
--
-- Since CUDA-13, "edge data" can be associated with an edge in the graph. This
-- function assumes no such data is present (if there is, a CUDA error
-- (@CUDA_ERROR_LOSSY_QUERY@) will be raised).
--
-- Requires CUDA-10.1
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g13145ece1d79a1d79a1d22abb9663216>
--
-- @since 0.10.1.0
--
#if CUDA_VERSION < 10010
info :: Stream -> IO (Status, Int64)
info = requireSDK 'info 10.1
-- Not another elif because c2hs seems to be buggy
#else
#if CUDA_VERSION < 12000
{# fun unsafe cuStreamGetCaptureInfo as info
  { useStream `Stream'
  , alloca-   `Status' peekEnum*
  , alloca-   `Int64'  peekIntConv*
  }
  -> `()' checkStatus*- #}
#elif CUDA_VERSION < 13000
{# fun unsafe cuStreamGetCaptureInfo_v2 as info
  { useStream `Stream'
  , alloca-   `Status' peekEnum*
  , alloca-   `Int64'  peekIntConv*
  , withNullPtr- `Graph'
  , withNullPtr- `Node'
  , withNullPtr- `CSize'
  }
  -> `()' checkStatus*- #}
#else
{# fun unsafe cuStreamGetCaptureInfo_v3 as info
  { useStream `Stream'
  , alloca-   `Status' peekEnum*
  , alloca-   `Int64'  peekIntConv*
  , withNullPtr- `Graph'
  , withNullPtr- `()'
  , withNullPtr- `Node'
  , withNullPtr- `CSize'
  }
  -> `()' checkStatus*- #}
#endif
#endif


-- | Set the stream capture interaction mode for this thread. Return the previous value.
--
-- Requires CUDA-10.1
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d>
--
-- @since 0.10.1.0
--
#if CUDA_VERSION < 10010
mode :: Mode -> IO Mode
mode = requireSDK 'mode 10.1
#else
{# fun unsafe cuThreadExchangeStreamCaptureMode as mode
  { withEnum* `Mode' peekEnum*
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

withNullPtr :: (Ptr a -> r) -> r
withNullPtr f = f nullPtr
