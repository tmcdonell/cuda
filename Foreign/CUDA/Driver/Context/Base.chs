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
-- Module    : Foreign.CUDA.Driver.Context.Base
-- Copyright : [2009..2015] Trevor L. McDonell
-- License   : BSD
--
-- Context management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Context.Base (

  -- * Context Management
  Context(..), ContextFlag(..),
  create, destroy, device, pop, push, sync, get, set,

  -- Deprecated
  attach, detach,

  -- * Cache Configuration
  Cache(..), Limit(..),
  getLimit, setLimit, setCacheConfig

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Device                       ( Device(..) )
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad                                    ( liftM )


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A device context
--
newtype Context = Context { useContext :: {# type CUcontext #}}
  deriving (Eq, Show)


-- |
-- Context creation flags
--
{# enum CUctx_flags as ContextFlag
    { underscoreToCase }
    with prefix="CU_CTX" deriving (Eq, Show, Bounded) #}

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
#if CUDA_VERSION < 3000
data Cache
#else
{# enum CUfunc_cache_enum as Cache
    { underscoreToCase }
    with prefix="CU_FUNC_CACHE" deriving (Eq, Show) #}
#endif


#if CUDA_VERSION >= 4000
{-# DEPRECATED attach, detach "as of CUDA-4.0" #-}
{-# DEPRECATED BlockingSync "use SchedBlockingSync instead" #-}
#endif


--------------------------------------------------------------------------------
-- Context management
--------------------------------------------------------------------------------

-- |
-- Create a new CUDA context and associate it with the calling thread. The
-- context is created with a usage count of one, and the caller of 'create'
-- must call 'destroy' when done using the context. If a context is already
-- current to the thread, it is supplanted by the newly created context and
-- must be restored by a subsequent call to 'pop'.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf>
--
{-# INLINEABLE create #-}
create :: Device -> [ContextFlag] -> IO Context
create !dev !flags = resultIfOk =<< cuCtxCreate flags dev

{-# INLINE cuCtxCreate #-}
{# fun unsafe cuCtxCreate
  { alloca-         `Context'       peekCtx*
  , combineBitMasks `[ContextFlag]'
  , useDevice       `Device'                 } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek


-- |
-- Increments the usage count of the context. API: no context flags are
-- currently supported, so this parameter must be empty.
--
{-# INLINEABLE attach #-}
attach :: Context -> [ContextFlag] -> IO ()
attach !ctx !flags = nothingIfOk =<< cuCtxAttach ctx flags

{-# INLINE cuCtxAttach #-}
{# fun unsafe cuCtxAttach
  { withCtx*        `Context'
  , combineBitMasks `[ContextFlag]' } -> `Status' cToEnum #}
  where withCtx = with . useContext


-- |
-- Detach the context, and destroy if no longer used
--
{-# INLINEABLE detach #-}
detach :: Context -> IO ()
detach !ctx = nothingIfOk =<< cuCtxDetach ctx

{-# INLINE cuCtxDetach #-}
{# fun unsafe cuCtxDetach
  { useContext `Context' } -> `Status' cToEnum #}


-- |
-- Destroy the specified context, regardless of how many threads it is
-- current to. The context will be 'pop'ed from the current thread's
-- context stack, but if it is current on any other threads it will remain
-- current to those threads, and attempts to access it will result in an
-- error.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e>
--
{-# INLINEABLE destroy #-}
destroy :: Context -> IO ()
destroy !ctx = nothingIfOk =<< cuCtxDestroy ctx

{-# INLINE cuCtxDestroy #-}
{# fun unsafe cuCtxDestroy
  { useContext `Context' } -> `Status' cToEnum #}


-- |
-- Return the context bound to the calling CPU thread.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0>
--
{-# INLINEABLE get #-}
get :: IO (Maybe Context)
#if CUDA_VERSION < 4000
get = requireSDK 'get 4.0
#else
get = resultIfOk =<< cuCtxGetCurrent

{-# INLINE cuCtxGetCurrent #-}
{# fun unsafe cuCtxGetCurrent
  { alloca- `Maybe Context' peekCtx* } -> `Status' cToEnum #}
  where peekCtx = liftM (nothingIfNull Context) . peek
#endif


-- |
-- Bind the specified context to the calling thread.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7>
--
{-# INLINEABLE set #-}
set :: Context -> IO ()
#if CUDA_VERSION < 4000
set _    = requireSDK 'set 4.0
#else
set !ctx = nothingIfOk =<< cuCtxSetCurrent ctx

{-# INLINE cuCtxSetCurrent #-}
{# fun unsafe cuCtxSetCurrent
  { useContext `Context' } -> `Status' cToEnum #}
#endif

-- |
-- Return the device of the currently active context
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e>
--
{-# INLINEABLE device #-}
device :: IO Device
device = resultIfOk =<< cuCtxGetDevice

{-# INLINE cuCtxGetDevice #-}
{# fun unsafe cuCtxGetDevice
  { alloca- `Device' dev* } -> `Status' cToEnum #}
  where dev = liftM Device . peekIntConv


-- |
-- Pop the current CUDA context from the CPU thread. The context may then
-- be attached to a different CPU thread by calling 'push'.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902>
--
{-# INLINEABLE pop #-}
pop :: IO Context
pop = resultIfOk =<< cuCtxPopCurrent

{-# INLINE cuCtxPopCurrent #-}
{# fun unsafe cuCtxPopCurrent
  { alloca- `Context' peekCtx* } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek


-- |
-- Push the given context onto the CPU's thread stack of current contexts.
-- The specified context becomes the CPU thread's current context, so all
-- operations that operate on the current context are affected.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba>
--
{-# INLINEABLE push #-}
push :: Context -> IO ()
push !ctx = nothingIfOk =<< cuCtxPushCurrent ctx

{-# INLINE cuCtxPushCurrent #-}
{# fun unsafe cuCtxPushCurrent
  { useContext `Context' } -> `Status' cToEnum #}


-- |
-- Block until the device has completed all preceding requests. If the
-- context was created with the 'SchedBlockingSync' flag, the CPU thread
-- will block until the GPU has finished its work.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616>
--
{-# INLINEABLE sync #-}
sync :: IO ()
sync = nothingIfOk =<< cuCtxSynchronize

{-# INLINE cuCtxSynchronize #-}
{# fun cuCtxSynchronize
  { } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Cache configuration
--------------------------------------------------------------------------------

-- |
-- Query compute 2.0 call stack limits. Requires cuda-3.1.
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
-- Specify the size of the call stack, for compute 2.0 devices. Requires
-- cuda-3.1.
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
  , cIntConv  `Int'   } -> `Status' cToEnum #}
#endif

-- |
-- On devices where the L1 cache and shared memory use the same hardware
-- resources, this sets the preferred cache configuration for the current
-- context. This is only a preference. Requires cuda-3.2.
--
{-# INLINEABLE setCacheConfig #-}
setCacheConfig :: Cache -> IO ()
#if CUDA_VERSION < 3020
setCacheConfig _  = requireSDK 'setCacheConfig 3.2
#else
setCacheConfig !c = nothingIfOk =<< cuCtxSetCacheConfig c

{-# INLINE cuCtxSetCacheConfig #-}
{# fun unsafe cuCtxSetCacheConfig
  { cFromEnum `Cache' } -> `Status' cToEnum #}
#endif

