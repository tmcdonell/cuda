{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.IPC.Marshal
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- IPC memory management for low-level driver interface.
--
-- Restricted to devices which support unified addressing on Linux
-- operating systems.
--
-- Since CUDA-4.0.
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.IPC.Marshal (

  -- ** IPC memory management
  IPCDevicePtr, IPCFlag(..),
  export, open, close,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Marshal

-- System
import Control.Monad
import Prelude

import Foreign.C
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.Marshal


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A CUDA memory handle used for inter-process communication.
--
newtype IPCDevicePtr a = IPCDevicePtr { useIPCDevicePtr :: IPCMemHandle }
  deriving (Eq, Show)


-- |
-- Flags for controlling IPC memory access
--
#if CUDA_VERSION < 4010
data IPCFlag
#else
{# enum CUipcMem_flags as IPCFlag
  { underscoreToCase }
  with prefix="CU_IPC_MEM" deriving (Eq, Show, Bounded) #}
#endif


--------------------------------------------------------------------------------
-- IPC memory management
--------------------------------------------------------------------------------

-- |
-- Create an inter-process memory handle for an existing device memory
-- allocation. The handle can then be sent to another process and made
-- available to that process via 'open'.
--
-- Requires CUDA-4.1.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1>
--
{-# INLINEABLE export #-}
export :: DevicePtr a -> IO (IPCDevicePtr a)
#if CUDA_VERSION < 4010
export _     = requireSDK 'export 4.1
#else
export !dptr = do
  h <- newIPCMemHandle
  r <- cuIpcGetMemHandle h dptr
  resultIfOk (r, IPCDevicePtr h)

{-# INLINE cuIpcGetMemHandle #-}
{# fun unsafe cuIpcGetMemHandle
  { withForeignPtr* `IPCMemHandle'
  , useDeviceHandle `DevicePtr a'
  }
  -> `Status' cToEnum #}
#endif


-- |
-- Open an inter-process memory handle exported from another process,
-- returning a device pointer usable in the current process.
--
-- Maps memory exported by another process with 'export into the current
-- device address space. For contexts on different devices, 'open' can
-- attempt to enable peer access if the user called
-- 'Foreign.CUDA.Driver.Context.Peer.add', and is controlled by the
-- 'LazyEnablePeerAccess' flag.
--
-- Each handle from a given device and context may only be 'open'ed by one
-- context per device per other process. Memory returned by 'open' must be
-- freed via 'close'.
--
-- Requires CUDA-4.1.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79>
--
{-# INLINEABLE open #-}
open :: IPCDevicePtr a -> [IPCFlag]-> IO (DevicePtr a)
#if CUDA_VERSION < 4010
open _    _      = requireSDK 'open 4.1
#else
open !hdl !flags = resultIfOk =<< cuIpcOpenMemHandle (useIPCDevicePtr hdl) flags

{-# INLINE cuIpcOpenMemHandle #-}
{# fun unsafe cuIpcOpenMemHandle
  { alloca-         `DevicePtr a'  peekDeviceHandle*
  , withForeignPtr* `IPCMemHandle'
  , combineBitMasks `[IPCFlag]'
  }
  -> `Status' cToEnum #}
#endif


-- |
-- Close and unmap memory returned by 'open'. The original allocation in
-- the exporting process as well as imported mappings in other processes
-- are unaffected.
--
-- Any resources used to enable peer access will be freed if this is the
-- last mapping using them.
--
-- Requires CUDA-4.1.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e>
--
{-# INLINEABLE close #-}
close :: DevicePtr a -> IO ()
#if CUDA_VERSION < 4010
close _     = requireSDK 'close 4.1
#else
close !dptr = nothingIfOk =<< cuIpcCloseMemHandle dptr

{-# INLINE cuIpcCloseMemHandle #-}
{# fun unsafe cuIpcCloseMemHandle
  { useDeviceHandle `DevicePtr a'
  }
  -> `Status' cToEnum #}
#endif


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

type IPCMemHandle = ForeignPtr ()

newIPCMemHandle :: IO IPCMemHandle
#if CUDA_VERSION < 4010
newIPCMemHandle = requireSDK 'newIPCMemHandle 4.1
#else
newIPCMemHandle = mallocForeignPtrBytes {#sizeof CUipcMemHandle#}
#endif

