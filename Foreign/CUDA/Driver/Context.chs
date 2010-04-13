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

module Foreign.CUDA.Driver.Context
  (
    Context, ContextFlag(..),
    create, attach, detach, destroy, current, pop, push, sync
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Device
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
-- Return the device of the currently active context
--
current :: IO Device
current = resultIfOk =<< cuCtxGetDevice

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

