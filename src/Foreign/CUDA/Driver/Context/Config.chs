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
-- Module    : Foreign.CUDA.Driver.Context.Config
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- Context configuration for the low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Context.Config (

  -- * Context configuration
  getFlags,

  -- ** Resource limits
  Limit(..),
  getLimit, setLimit,

  -- ** Cache
  Cache(..),
  getCache, setCache,

  -- ** Shared memory
  SharedMem(..),
  getSharedMem, setSharedMem,

  -- ** Streams
  StreamPriority,
  getStreamPriorityRange,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Context.Base
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Stream                         ( Stream, StreamPriority )

-- System
import Control.Monad
import Foreign
import Foreign.C


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- Device limits flags
--
#if CUDA_VERSION < 3010
data Limit
#else
{# enum CUlimit_enum as Limit
    { underscoreToCase }
    with prefix="CU_LIMIT" deriving (Eq, Show) #}
#endif


-- |
-- Device cache configuration preference
--
#if CUDA_VERSION < 3020
data Cache
#else
{# enum CUfunc_cache_enum as Cache
    { underscoreToCase }
    with prefix="CU_FUNC_CACHE" deriving (Eq, Show) #}
#endif


-- |
-- Device shared memory configuration preference
--
#if CUDA_VERSION < 4020
data SharedMem
#else
{# enum CUsharedconfig as SharedMem
    { underscoreToCase }
    with prefix="CU_SHARED_MEM_CONFIG" deriving (Eq, Show) #}
#endif


--------------------------------------------------------------------------------
-- Context configuration
--------------------------------------------------------------------------------

-- |
-- Return the flags that were used to create the current context.
--
-- Requires CUDA-7.0
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d>
--
{-# INLINEABLE getFlags #-}
getFlags :: IO [ContextFlag]
#if CUDA_VERSION < 7000
getFlags = requireSDK 'getFlags 7.0
#else
getFlags = resultIfOk =<< cuCtxGetFlags

{-# INLINE cuCtxGetFlags #-}
{# fun unsafe cuCtxGetFlags
  { alloca- `[ContextFlag]' peekFlags* } -> `Status' cToEnum #}
  where
    peekFlags = liftM extractBitMasks . peek
#endif


-- |
-- Query compute 2.0 call stack limits.
--
-- Requires CUDA-3.1.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8>
--
{-# INLINEABLE getLimit #-}
getLimit :: Limit -> IO Int
#if CUDA_VERSION < 3010
getLimit _  = requireSDK 'getLimit 3.1
#else
getLimit !l = resultIfOk =<< cuCtxGetLimit l

{-# INLINE cuCtxGetLimit #-}
{# fun unsafe cuCtxGetLimit
  { alloca-   `Int' peekIntConv*
  , cFromEnum `Limit'            } -> `Status' cToEnum #}
#endif


-- |
-- Specify the size of the call stack, for compute 2.0 devices.
--
-- Requires CUDA-3.1.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a>
--
{-# INLINEABLE setLimit #-}
setLimit :: Limit -> Int -> IO ()
#if CUDA_VERSION < 3010
setLimit _ _   = requireSDK 'setLimit 3.1
#else
setLimit !l !n = nothingIfOk =<< cuCtxSetLimit l n

{-# INLINE cuCtxSetLimit #-}
{# fun unsafe cuCtxSetLimit
  { cFromEnum `Limit'
  ,           `Int'   } -> `Status' cToEnum #}
#endif


-- |
-- On devices where the L1 cache and shared memory use the same hardware
-- resources, this function returns the preferred cache configuration for
-- the current context.
--
-- Requires CUDA-3.2.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360>
--
{-# INLINEABLE getCache #-}
getCache :: IO Cache
#if CUDA_VERSION < 3020
getCache = requireSDK 'getCache 3.2
#else
getCache = resultIfOk =<< cuCtxGetCacheConfig

{-# INLINE cuCtxGetCacheConfig #-}
{# fun unsafe cuCtxGetCacheConfig
  { alloca- `Cache' peekEnum* } -> `Status' cToEnum #}
#endif


-- |
-- On devices where the L1 cache and shared memory use the same hardware
-- resources, this sets the preferred cache configuration for the current
-- context. This is only a preference.
--
-- Any function configuration set via
-- 'Foreign.CUDA.Driver.Exec.setCacheConfigFun' will be preferred over this
-- context-wide setting.
--
-- Requires CUDA-3.2.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3>
--
{-# INLINEABLE setCache #-}
setCache :: Cache -> IO ()
#if CUDA_VERSION < 3020
setCache _  = requireSDK 'setCache 3.2
#else
setCache !c = nothingIfOk =<< cuCtxSetCacheConfig c

{-# INLINE cuCtxSetCacheConfig #-}
{# fun unsafe cuCtxSetCacheConfig
  { cFromEnum `Cache' } -> `Status' cToEnum #}
#endif


-- |
-- Return the current size of the shared memory banks in the current
-- context. On devices with configurable shared memory banks,
-- 'setSharedMem' can be used to change the configuration, so that
-- subsequent kernel launches will by default us the new bank size. On
-- devices without configurable shared memory, this function returns the
-- fixed bank size of the hardware.
--
-- Requires CUDA-4.2
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g17153a1b8b8c756f7ab8505686a4ad74>
--
{-# INLINEABLE getSharedMem #-}
getSharedMem :: IO SharedMem
#if CUDA_VERSION < 4020
getSharedMem = requireSDK 'getSharedMem 4.2
#else
getSharedMem = resultIfOk =<< cuCtxGetSharedMemConfig

{-# INLINE cuCtxGetSharedMemConfig #-}
{# fun unsafe cuCtxGetSharedMemConfig
  { alloca- `SharedMem' peekEnum*
  } -> `Status' cToEnum #}
#endif


-- |
-- On devices with configurable shared memory banks, this function will set
-- the context's shared memory bank size that will be used by default for
-- subsequent kernel launches.
--
-- Changing the shared memory configuration between launches may insert
-- a device synchronisation.
--
-- Shared memory bank size does not affect shared memory usage or kernel
-- occupancy, but may have major effects on performance. Larger bank sizes
-- allow for greater potential bandwidth to shared memory, but change the
-- kinds of accesses which result in bank conflicts.
--
-- Requires CUDA-4.2
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2574235fa643f8f251bf7bc28fac3692>
--
{-# INLINEABLE setSharedMem #-}
setSharedMem :: SharedMem -> IO ()
#if CUDA_VERSION < 4020
setSharedMem _  = requireSDK 'setSharedMem 4.2
#else
setSharedMem !c = nothingIfOk =<< cuCtxSetSharedMemConfig c

{-# INLINE cuCtxSetSharedMemConfig #-}
{# fun unsafe cuCtxSetSharedMemConfig
  { cFromEnum `SharedMem' } -> `Status' cToEnum #}
#endif



-- |
-- Returns the numerical values that correspond to the greatest and least
-- priority execution streams in the current context respectively. Stream
-- priorities follow the convention that lower numerical numbers correspond
-- to higher priorities. The range of meaningful stream priorities is given
-- by the inclusive range [greatestPriority,leastPriority].
--
-- Requires CUDA-5.5.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091>
--
{-# INLINEABLE getStreamPriorityRange #-}
getStreamPriorityRange :: IO (StreamPriority, StreamPriority)
#if CUDA_VERSION < 5050
getStreamPriorityRange = requireSDK 'getStreamPriorityRange 5.5
#else
getStreamPriorityRange = do
  (r,l,h) <- cuCtxGetStreamPriorityRange
  resultIfOk (r, (h,l))

{-# INLINE cuCtxGetStreamPriorityRange #-}
{# fun unsafe cuCtxGetStreamPriorityRange
  { alloca- `Int' peekIntConv*
  , alloca- `Int' peekIntConv*
  }
  -> `Status' cToEnum #}
#endif

