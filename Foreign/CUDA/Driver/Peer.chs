{-# LANGUAGE ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Peer
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Peer context memory access for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Peer
  (
    PeerFlag,
    addPeer, removePeer, accessible
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Driver.Context
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C


-- |
-- Possible option values for direct peer memory access
--
data PeerFlag

instance Enum PeerFlag where


--------------------------------------------------------------------------------
-- Peer access
--------------------------------------------------------------------------------

-- |
-- Queries if the first device can directly access the memory of the second
--
accessible :: Device -> Device -> IO Bool
#if CUDA_VERSION < 4000
accessible _   _    = error "accessible: requires at least CUDA-4.0"
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
-- by the current context.
--
addPeer :: Context -> [PeerFlag] -> IO ()
#if CUDA_VERSION < 4000
addPeer _   _     = error "addPeer: requires at least CUDA-4.0"
#else
addPeer ctx flags = nothingIfOk =<< cuCtxEnablePeerAccess ctx flags

{# fun unsafe cuCtxEnablePeerAccess
  { useContext      `Context'
  , combineBitMasks `[PeerFlag]' } -> `Status' cToEnum #}
#endif


-- |
-- Disable direct memory access from the current context to the supplied context
--
removePeer :: Context -> IO ()
#if CUDA_VERSION < 4000
removePeer _   = error "removePeer: requires at least CUDA-4.0"
#else
removePeer ctx = nothingIfOk =<< cuCtxDisablePeerAccess ctx

{# fun unsafe cuCtxDisablePeerAccess
  { useContext `Context' } -> `Status' cToEnum #}
#endif


