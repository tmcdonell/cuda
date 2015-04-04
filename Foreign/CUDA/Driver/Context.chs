{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
#ifdef USE_EMPTY_CASE
{-# LANGUAGE EmptyCase                #-}
#endif
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Context
-- Copyright : [2009..2014] Trevor L. McDonell
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
-- Device cache configuration preference
--
#if CUDA_VERSION < 3000
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
#ifdef USE_EMPTY_CASE
  toEnum   x = case x of {}
  fromEnum x = case x of {}
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
-- Destroy the specified context. This fails if the context is more than a
-- single attachment (including that from initial creation).
--
{-# INLINEABLE destroy #-}
destroy :: Context -> IO ()
destroy !ctx = nothingIfOk =<< cuCtxDestroy ctx

{-# INLINE cuCtxDestroy #-}
{# fun unsafe cuCtxDestroy
  { useContext `Context' } -> `Status' cToEnum #}


-- |
-- Return the context bound to the calling CPU thread. Requires cuda-4.0.
--
{-# INLINEABLE get #-}
get :: IO Context
#if CUDA_VERSION < 4000
get = requireSDK 4.0 "get"
#else
get = resultIfOk =<< cuCtxGetCurrent

{-# INLINE cuCtxGetCurrent #-}
{# fun unsafe cuCtxGetCurrent
  { alloca- `Context' peekCtx* } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek
#endif


-- |
-- Bind the specified context to the calling thread. Requires cuda-4.0.
--
{-# INLINEABLE set #-}
set :: Context -> IO ()
#if CUDA_VERSION < 4000
set _    = requireSDK 4.0 "set"
#else
set !ctx = nothingIfOk =<< cuCtxSetCurrent ctx

{-# INLINE cuCtxSetCurrent #-}
{# fun unsafe cuCtxSetCurrent
  { useContext `Context' } -> `Status' cToEnum #}
#endif

-- |
-- Return the device of the currently active context
--
{-# INLINEABLE device #-}
device :: IO Device
device = resultIfOk =<< cuCtxGetDevice

{-# INLINE cuCtxGetDevice #-}
{# fun unsafe cuCtxGetDevice
  { alloca- `Device' dev* } -> `Status' cToEnum #}
  where dev = liftM Device . peekIntConv


-- |
-- Pop the current CUDA context from the CPU thread. The context must have a
-- single usage count (matching calls to 'attach' and 'detach'). If successful,
-- the new context is returned, and the old may be attached to a different CPU.
--
{-# INLINEABLE pop #-}
pop :: IO Context
pop = resultIfOk =<< cuCtxPopCurrent

{-# INLINE cuCtxPopCurrent #-}
{# fun unsafe cuCtxPopCurrent
  { alloca- `Context' peekCtx* } -> `Status' cToEnum #}
  where peekCtx = liftM Context . peek


-- |
-- Push the given context onto the CPU's thread stack of current contexts. The
-- context must be floating (via 'pop'), i.e. not attached to any thread.
--
{-# INLINEABLE push #-}
push :: Context -> IO ()
push !ctx = nothingIfOk =<< cuCtxPushCurrent ctx

{-# INLINE cuCtxPushCurrent #-}
{# fun unsafe cuCtxPushCurrent
  { useContext `Context' } -> `Status' cToEnum #}


-- |
-- Block until the device has completed all preceding requests
--
{-# INLINEABLE sync #-}
sync :: IO ()
sync = nothingIfOk =<< cuCtxSynchronize

{-# INLINE cuCtxSynchronize #-}
{# fun cuCtxSynchronize
  { } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Peer access
--------------------------------------------------------------------------------

-- |
-- Queries if the first device can directly access the memory of the second. If
-- direct access is possible, it can then be enabled with 'add'. Requires
-- cuda-4.0.
--
{-# INLINEABLE accessible #-}
accessible :: Device -> Device -> IO Bool
#if CUDA_VERSION < 4000
accessible _ _        = requireSDK 4.0 "accessible"
#else
accessible !dev !peer = resultIfOk =<< cuDeviceCanAccessPeer dev peer

{-# INLINE cuDeviceCanAccessPeer #-}
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
{-# INLINEABLE add #-}
add :: Context -> [PeerFlag] -> IO ()
#if CUDA_VERSION < 4000
add _ _         = requireSDK 4.0 "add"
#else
add !ctx !flags = nothingIfOk =<< cuCtxEnablePeerAccess ctx flags

{-# INLINE cuCtxEnablePeerAccess #-}
{# fun unsafe cuCtxEnablePeerAccess
  { useContext      `Context'
  , combineBitMasks `[PeerFlag]' } -> `Status' cToEnum #}
#endif


-- |
-- Disable direct memory access from the current context to the supplied
-- context. Requires cuda-4.0.
--
{-# INLINEABLE remove #-}
remove :: Context -> IO ()
#if CUDA_VERSION < 4000
remove _    = requireSDK 4.0 "remove"
#else
remove !ctx = nothingIfOk =<< cuCtxDisablePeerAccess ctx

{-# INLINE cuCtxDisablePeerAccess #-}
{# fun unsafe cuCtxDisablePeerAccess
  { useContext `Context' } -> `Status' cToEnum #}
#endif


--------------------------------------------------------------------------------
-- Cache configuration
--------------------------------------------------------------------------------

-- |
-- Query compute 2.0 call stack limits. Requires cuda-3.1.
--
{-# INLINEABLE getLimit #-}
getLimit :: Limit -> IO Int
#if CUDA_VERSION < 3010
getLimit _  = requireSDK 3.1 "getLimit"
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
setLimit _ _   = requireSDK 3.1 "setLimit"
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
setCacheConfig _  = requireSDK 3.2 "setCacheConfig"
#else
setCacheConfig !c = nothingIfOk =<< cuCtxSetCacheConfig c

{-# INLINE cuCtxSetCacheConfig #-}
{# fun unsafe cuCtxSetCacheConfig
  { cFromEnum `Cache' } -> `Status' cToEnum #}
#endif

