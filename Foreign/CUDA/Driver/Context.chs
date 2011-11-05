{-# LANGUAGE CPP, ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Context
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Context management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Context (

  -- * Context Management
  Context(..), ContextFlag(..),
  create, attach, detach, destroy, device, pop, push, sync, get, set,

  -- * Peer Access
  PeerFlag,
  accessible, add, remove,

  -- * Cache Configuration
  Cache(..), Limit(..),
  getLimit, setLimit, setCacheConfig

) where

#include <cuda.h>
#include "cbits/stubs.h"
{# context lib="cuda" #}

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
-- Device cache configuration flags
--
#if CUDA_VERSION < 3020
data Cache
#else
{# enum CUfunc_cache_enum as Cache
    { underscoreToCase }
    with prefix="CU_FUNC_CACHE" deriving (Eq, Show) #}
#endif

-- |
-- Possible option values for direct peer memory access
--
data PeerFlag
instance Enum PeerFlag where


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


-- |
-- Return the context bound to the calling CPU thread. Requires cuda-4.0.
--
get :: IO Context
#if CUDA_VERSION < 4000
get = requireSDK 4.0 "get"
#else
get = resultIfOk =<< cuCtxGetCurrent

{# fun unsafe cuCtxGetCurrent
  { alloca- `Context' peekCtx* } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek
#endif


-- |
-- Bind the specified context to the calling thread. Requires cuda-4.0.
--
set :: Context -> IO ()
#if CUDA_VERSION < 4000
set _   = requireSDK 4.0 "set"
#else
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
-- single usage count (matching calls to 'attach' and 'detach'). If successful,
-- the new context is returned, and the old may be attached to a different CPU.
--
pop :: IO Context
pop = resultIfOk =<< cuCtxPopCurrent

{# fun unsafe cuCtxPopCurrent
  { alloca- `Context' peekCtx* } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek


-- |
-- Push the given context onto the CPU's thread stack of current contexts. The
-- context must be floating (via 'pop'), i.e. not attached to any thread.
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
-- Peer access
--------------------------------------------------------------------------------

-- |
-- Queries if the first device can directly access the memory of the second. If
-- direct access is possible, it can then be enabled with 'add'. Requires
-- cuda-4.0.
--
accessible :: Device -> Device -> IO Bool
#if CUDA_VERSION < 4000
accessible _   _    = requireSDK 4.0 "accessible"
#else
accessible dev peer = resultIfOk =<< cuDeviceCanAccessPeer dev peer

{# fun unsafe cuDeviceCanAccessPeer
  { alloca-   `Bool'   peekBool*
  , useDevice `Device'
  , useDevice `Device'           } -> `Status' cToEnum #}
#endif


-- |
-- If the devices of both the current and supplied contexts support unified
-- addressing, then enable allocations in the supplied context to be accessible
-- by the current context. Requires cuda-4.0.
--
add :: Context -> [PeerFlag] -> IO ()
#if CUDA_VERSION < 4000
add _   _     = requireSDK 4.0 "add"
#else
add ctx flags = nothingIfOk =<< cuCtxEnablePeerAccess ctx flags

{# fun unsafe cuCtxEnablePeerAccess
  { useContext      `Context'
  , combineBitMasks `[PeerFlag]' } -> `Status' cToEnum #}
#endif


-- |
-- Disable direct memory access from the current context to the supplied
-- context. Requires cuda-4.0.
--
remove :: Context -> IO ()
#if CUDA_VERSION < 4000
remove _   = requireSDK 4.0 "remove"
#else
remove ctx = nothingIfOk =<< cuCtxDisablePeerAccess ctx

{# fun unsafe cuCtxDisablePeerAccess
  { useContext `Context' } -> `Status' cToEnum #}
#endif


--------------------------------------------------------------------------------
-- Cache configuration
--------------------------------------------------------------------------------

-- |
-- Query compute 2.0 call stack limits. Requires cuda-3.1.
--
getLimit :: Limit -> IO Int
#if CUDA_VERSION < 3010
getLimit _ = requireSDK 3.1 "getLimit"
#else
getLimit l = resultIfOk =<< cuCtxGetLimit l

{# fun unsafe cuCtxGetLimit
  { alloca-   `Int' peekIntConv*
  , cFromEnum `Limit'            } -> `Status' cToEnum #}
#endif

-- |
-- Specify the size of the call stack, for compute 2.0 devices. Requires
-- cuda-3.1.
--
setLimit :: Limit -> Int -> IO ()
#if CUDA_VERSION < 3010
setLimit _ _ = requireSDK 3.1 "setLimit"
#else
setLimit l n = nothingIfOk =<< cuCtxSetLimit l n

{# fun unsafe cuCtxSetLimit
  { cFromEnum `Limit'
  , cIntConv  `Int'   } -> `Status' cToEnum #}
#endif

-- |
-- On devices where the L1 cache and shared memory use the same hardware
-- resources, this sets the preferred cache configuration for the current
-- context. This is only a preference. Requires cuda-3.2.
--
setCacheConfig :: Cache -> IO ()
#if CUDA_VERSION < 3020
setCacheConfig _ = requireSDK 3.2 "setCacheConfig"
#else
setCacheConfig c = nothingIfOk =<< cuCtxSetCacheConfig c

{# fun unsafe cuCtxSetCacheConfig
  { cFromEnum `Cache' } -> `Status' cToEnum #}
#endif

