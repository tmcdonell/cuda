{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.IPC.Event
-- Copyright : [2009..2020] Trevor L. McDonell
-- License   : BSD
--
-- IPC event management for low-level driver interface.
--
-- Restricted to devices which support unified addressing on Linux
-- operating systems.
--
-- Since CUDA-4.1.
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.IPC.Event (

  IPCEvent,
  export, open,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Event
import Foreign.CUDA.Internal.C2HS

-- System
import Control.Monad
import Prelude

import Foreign.C
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.Marshal
import Foreign.Storable


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A CUDA inter-process event handle.
--
newtype IPCEvent = IPCEvent { useIPCEvent :: IPCEventHandle }
  deriving (Eq, Show)


--------------------------------------------------------------------------------
-- IPC event management
--------------------------------------------------------------------------------

-- |
-- Create an inter-process event handle for a previously allocated event.
-- The event must be created with the 'Interprocess' and 'DisableTiming'
-- event flags. The returned handle may then be sent to another process and
-- 'open'ed to allow efficient hardware synchronisation between GPU work in
-- other processes.
--
-- After the event has been opened in the importing process, 'record',
-- 'block', 'wait', 'query' may be used in either process.
--
-- Performing operations on the imported event after the event has been
-- 'destroy'ed in the exporting process is undefined.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c>
--
{-# INLINEABLE export #-}
export :: Event -> IO IPCEvent
#if CUDA_VERSION < 4010
export _   = requireSDK 'create 4.1
#else
export !ev = do
  h <- newIPCEventHandle
  r <- cuIpcGetEventHandle h ev
  resultIfOk (r, IPCEvent h)

{-# INLINE cuIpcGetEventHandle #-}
{# fun unsafe cuIpcGetEventHandle
  { withForeignPtr* `IPCEventHandle'
  , useEvent        `Event'
  }
  -> `Status' cToEnum #}
#endif


-- |
-- Open an inter-process event handle for use in the current process,
-- returning an event that can be used in the current process and behaving
-- as a locally created event with the 'DisableTiming' flag specified.
--
-- The event must be freed with 'destroy'. Performing operations on the
-- imported event after the exported event has been 'destroy'ed in the
-- exporting process is undefined.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf1d525918b6c643b99ca8c8e42e36c2e>
--
{-# INLINEABLE open #-}
open :: IPCEvent -> IO Event
#if CUDA_VERSION < 4010
open _    = requireSDK 'open 4.1
#else
open !ev = resultIfOk =<< cuIpcOpenEventHandle (useIPCEvent ev)

{-# INLINE cuIpcOpenEventHandle #-}
{# fun unsafe cuIpcOpenEventHandle
  { alloca-         `Event'          peekEvent*
  , withForeignPtr* `IPCEventHandle'
  }
  -> `Status' cToEnum #}
  where
    peekEvent = liftM Event . peek
#endif


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

type IPCEventHandle = ForeignPtr ()

newIPCEventHandle :: IO IPCEventHandle
#if CUDA_VERSION < 4010
newIPCEventHandle = requireSDK 'newIPCEventHandle 4.1
#else
newIPCEventHandle = mallocForeignPtrBytes {#sizeof CUipcEventHandle#}
#endif

