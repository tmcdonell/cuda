{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Context.Primary
-- Copyright : [2009..2018] Trevor L. McDonell
-- License   : BSD
--
-- Primary context management for low-level driver interface. The primary
-- context is unique per device and shared with the Runtime API. This
-- allows integration with other libraries using CUDA.
--
-- Since: CUDA-7.0
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Context.Primary (

  status, setup, reset, retain, release,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Context.Base
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Control.Exception
import Control.Monad
import Foreign
import Foreign.C


--------------------------------------------------------------------------------
-- Primary context management
--------------------------------------------------------------------------------


-- |
-- Get the status of the primary context. Returns whether the current
-- context is active, and the flags it was (or will be) created with.
--
-- Requires CUDA-7.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g65f3e018721b6d90aa05cfb56250f469>
--
{-# INLINEABLE status #-}
status :: Device -> IO (Bool, [ContextFlag])
#if CUDA_VERSION < 7000
status _    = requireSDK 'status 7.0
#else
status !dev =
  cuDevicePrimaryCtxGetState dev >>= \(rv, !flags, !active) ->
  case rv of
    Success -> return (active, flags)
    _       -> throwIO (ExitCode rv)

{# fun unsafe cuDevicePrimaryCtxGetState
  { useDevice `Device'
  , alloca-   `[ContextFlag]' peekFlags*
  , alloca-   `Bool'          peekBool*
  } -> `Status' cToEnum #}
  where
    peekFlags = liftM extractBitMasks . peek
#endif


-- |
-- Specify the flags that the primary context should be created with. Note
-- that this is an error if the primary context is already active.
--
-- Requires CUDA-7.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf>
--
{-# INLINEABLE setup #-}
setup :: Device -> [ContextFlag] -> IO ()
#if CUDA_VERSION < 7000
setup _    _      = requireSDK 'setup 7.0
#else
setup !dev !flags = nothingIfOk =<< cuDevicePrimaryCtxSetFlags dev flags

{-# INLINE cuDevicePrimaryCtxSetFlags #-}
{# fun unsafe cuDevicePrimaryCtxSetFlags
  { useDevice       `Device'
  , combineBitMasks `[ContextFlag]'
  } -> `Status' cToEnum #}
#endif


-- |
-- Destroy all allocations and reset all state on the primary context of
-- the given device in the current process. Requires cuda-7.0
--
-- Requires CUDA-7.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g5d38802e8600340283958a117466ce12>
--
{-# INLINEABLE reset #-}
reset :: Device -> IO ()
#if CUDA_VERSION < 7000
reset _    = requireSDK 'reset 7.0
#else
reset !dev = nothingIfOk =<< cuDevicePrimaryCtxReset dev

{-# INLINE cuDevicePrimaryCtxReset #-}
{# fun unsafe cuDevicePrimaryCtxReset
  { useDevice `Device' } -> `Status' cToEnum #}
#endif


-- |
-- Release the primary context on the given device. If there are no more
-- references to the primary context it will be destroyed, regardless of
-- how many threads it is current to.
--
-- Unlike 'Foreign.CUDA.Driver.Context.Base.pop' this does not pop the
-- context from the stack in any circumstances.
--
-- Requires CUDA-7.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad>
--
{-# INLINEABLE release #-}
release :: Device -> IO ()
#if CUDA_VERSION < 7000
release _    = requireSDK 'release 7.0
#else
release !dev = nothingIfOk =<< cuDevicePrimaryCtxRelease dev

{-# INLINE cuDevicePrimaryCtxRelease #-}
{# fun unsafe cuDevicePrimaryCtxRelease
  { useDevice `Device' } -> `Status' cToEnum #}
#endif


-- |
-- Retain the primary context for the given device, creating it if
-- necessary, and increasing its usage count. The caller must call
-- 'release' when done using the context. Unlike
-- 'Foreign.CUDA.Driver.Context.Base.create' the newly retained context is
-- not pushed onto the stack.
--
-- Requires CUDA-7.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300>
--
{-# INLINEABLE retain #-}
retain :: Device -> IO Context
#if CUDA_VERSION < 7000
retain _    = requireSDK 'retain 7.0
#else
retain !dev = resultIfOk =<< cuDevicePrimaryCtxRetain dev

{-# INLINE cuDevicePrimaryCtxRetain #-}
{# fun unsafe cuDevicePrimaryCtxRetain
  { alloca-   `Context' peekCtx*
  , useDevice `Device'
  } -> `Status' cToEnum #}
  where
    peekCtx = liftM Context . peek
#endif

