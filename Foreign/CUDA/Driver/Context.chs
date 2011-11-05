{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Context
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Context management for low-level driver interface
--
--------------------------------------------------------------------------------

#include <cuda.h>
#include "cbits/stubs.h"
{# context lib="cuda" #}

module Foreign.CUDA.Driver.Context
  (
    Context(..), ContextFlag(..),
#if CUDA_VERSION >= 3010
    Limit(..),
#endif
#if CUDA_VERSION >= 3020
    Cache(..),
#endif
    create, attach, detach, destroy, device, pop, push, sync,
#if CUDA_VERSION >= 4000
    get, set,
#endif
#if CUDA_VERSION >= 3010
    getLimit, setLimit
#endif
#if CUDA_VERSION >= 3020
    , setCacheConfig
#endif
  )
  where

-- Friends
import Foreign.CUDA.Driver.Device       (Device(..))
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad                    (liftM)


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A device context
--
newtype Context = Context { useContext :: {# type CUcontext #}}


-- |
-- Context creation flags
--
{# enum CUctx_flags as ContextFlag
    { underscoreToCase }
    with prefix="CU_CTX" deriving (Eq, Show) #}


#if CUDA_VERSION >= 3010
-- |
-- Device limits flags
--
{# enum CUlimit_enum as Limit
    { underscoreToCase }
    with prefix="CU_LIMIT" deriving (Eq, Show) #}
#endif
#if CUDA_VERSION >= 3020
-- |
-- Device cache configuration flags
--
{# enum CUfunc_cache_enum as Cache
    { underscoreToCase }
    with prefix="CU_FUNC_CACHE" deriving (Eq, Show) #}
#endif

#if CUDA_VERSION >= 4000
{-# DEPRECATED attach, detach "deprecated as of CUDA-4.0" #-}
{-# DEPRECATED BlockingSync "use SchedBlockingSync instead" #-}
#endif


--------------------------------------------------------------------------------
-- Context management
--------------------------------------------------------------------------------

-- |
-- Create a new CUDA context and associate it with the calling thread
--
create :: Device -> [ContextFlag] -> IO Context
create dev flags = resultIfOk =<< cuCtxCreate flags dev

{# fun unsafe cuCtxCreate
  { alloca-         `Context'       peekCtx*
  , combineBitMasks `[ContextFlag]'
  , useDevice       `Device'                 } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek


-- |
-- Increments the usage count of the context. API: no context flags are
-- currently supported, so this parameter must be empty.
--
attach :: Context -> [ContextFlag] -> IO ()
attach ctx flags = nothingIfOk =<< cuCtxAttach ctx flags

{# fun unsafe cuCtxAttach
  { withCtx*        `Context'
  , combineBitMasks `[ContextFlag]' } -> `Status' cToEnum #}
  where withCtx = with . useContext


-- |
-- Detach the context, and destroy if no longer used
--
detach :: Context -> IO ()
detach ctx = nothingIfOk =<< cuCtxDetach ctx

{# fun unsafe cuCtxDetach
  { useContext `Context' } -> `Status' cToEnum #}


-- |
-- Destroy the specified context. This fails if the context is more than a
-- single attachment (including that from initial creation).
--
destroy :: Context -> IO ()
destroy ctx = nothingIfOk =<< cuCtxDestroy ctx

{# fun unsafe cuCtxDestroy
  { useContext `Context' } -> `Status' cToEnum #}

#if CUDA_VERSION >= 4000
-- |
-- Return the context bound to the calling CPU thread
--
get :: IO Context
get = resultIfOk =<< cuCtxGetCurrent

{# fun unsafe cuCtxGetCurrent
  { alloca- `Context' peekCtx* } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek


-- |
-- Bind the specified context to the calling thread
--
set :: Context -> IO ()
set ctx = nothingIfOk =<< cuCtxSetCurrent ctx

{# fun unsafe cuCtxSetCurrent
  { useContext `Context' } -> `Status' cToEnum #}
#endif

-- |
-- Return the device of the currently active context
--
device :: IO Device
device = resultIfOk =<< cuCtxGetDevice

{# fun unsafe cuCtxGetDevice
  { alloca- `Device' dev* } -> `Status' cToEnum #}
  where dev = liftM Device . peekIntConv


-- |
-- Pop the current CUDA context from the CPU thread. The context must have a
-- single usage count (matching calls to attach/detach). If successful, the new
-- context is returned, and the old may be attached to a different CPU.
--
pop :: IO Context
pop = resultIfOk =<< cuCtxPopCurrent

{# fun unsafe cuCtxPopCurrent
  { alloca- `Context' peekCtx* } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek


-- |
-- Push the given context onto the CPU's thread stack of current contexts. The
-- context must be floating (via `pop'), i.e. not attached to any thread.
--
push :: Context -> IO ()
push ctx = nothingIfOk =<< cuCtxPushCurrent ctx

{# fun unsafe cuCtxPushCurrent
  { useContext `Context' } -> `Status' cToEnum #}


-- |
-- Block until the device has completed all preceding requests
--
sync :: IO ()
sync = nothingIfOk =<< cuCtxSynchronize

{# fun unsafe cuCtxSynchronize
  { } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Cache configuration
--------------------------------------------------------------------------------

#if CUDA_VERSION >= 3010
-- |
-- Query compute 2.0 call stack limits
--
getLimit :: Limit -> IO Int
getLimit l = resultIfOk =<< cuCtxGetLimit l

-- |
-- Specify the size of the call stack, for compute 2.0 devices
--
setLimit :: Limit -> Int -> IO ()
setLimit l n = nothingIfOk =<< cuCtxSetLimit l n

{# fun unsafe cuCtxGetLimit
  { alloca-   `Int' peekIntConv*
  , cFromEnum `Limit'            } -> `Status' cToEnum #}

{# fun unsafe cuCtxSetLimit
  { cFromEnum `Limit'
  , cIntConv  `Int'   } -> `Status' cToEnum #}
#endif
#if CUDA_VERSION >= 3020
-- |
-- On devices where the L1 cache and shared memory use the same hardware
-- resources, this sets the preferred cache configuration for the current
-- context. This is only a preference.
--
setCacheConfig :: Cache -> IO ()
setCacheConfig c = nothingIfOk =<< cuCtxSetCacheConfig c

{# fun unsafe cuCtxSetCacheConfig
  { cFromEnum `Cache' } -> `Status' cToEnum #}
#endif

