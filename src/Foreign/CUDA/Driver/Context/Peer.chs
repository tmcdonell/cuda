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
-- Module    : Foreign.CUDA.Driver.Context.Peer
-- Copyright : [2009..2020] Trevor L. McDonell
-- License   : BSD
--
-- Direct peer context access functions for the low-level driver interface.
--
-- Since: CUDA-4.0
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Context.Peer (

  -- * Peer Access
  PeerFlag, PeerAttribute(..),
  accessible, add, remove, getAttribute,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Context.Base                 ( Context(..) )
import Foreign.CUDA.Driver.Device                       ( Device(..) )
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- Possible option values for direct peer memory access
--
data PeerFlag
instance Enum PeerFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("PeerFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif


-- | Peer-to-peer attributes
--
#if CUDA_VERSION < 8000
data PeerAttribute
instance Enum PeerAttribute where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("PeerAttribute.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif
#else
{# enum CUdevice_P2PAttribute as PeerAttribute
  { underscoreToCase }
  with prefix="CU_DEVICE_P2P_ATTRIBUTE" deriving (Eq, Show) #}
#endif


--------------------------------------------------------------------------------
-- Peer access
--------------------------------------------------------------------------------

-- |
-- Queries if the first device can directly access the memory of the second. If
-- direct access is possible, it can then be enabled with 'add'.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e>
--
{-# INLINEABLE accessible #-}
accessible :: Device -> Device -> IO Bool
#if CUDA_VERSION < 4000
accessible _ _        = requireSDK 'accessible 4.0
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
-- by the current context.
--
-- Note that access is unidirectional, and in order to access memory in the
-- current context from the peer context, a separate symmetric call to
-- 'add' is required.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a>
--
{-# INLINEABLE add #-}
add :: Context -> [PeerFlag] -> IO ()
#if CUDA_VERSION < 4000
add _ _         = requireSDK 'add 4.0
#else
add !ctx !flags = nothingIfOk =<< cuCtxEnablePeerAccess ctx flags

{-# INLINE cuCtxEnablePeerAccess #-}
{# fun unsafe cuCtxEnablePeerAccess
  { useContext      `Context'
  , combineBitMasks `[PeerFlag]' } -> `Status' cToEnum #}
#endif


-- |
-- Disable direct memory access from the current context to the supplied
-- peer context, and unregisters any registered allocations.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g5b4b6936ea868d4954ce4d841a3b4810>
--
{-# INLINEABLE remove #-}
remove :: Context -> IO ()
#if CUDA_VERSION < 4000
remove _    = requireSDK 'remave 4.0
#else
remove !ctx = nothingIfOk =<< cuCtxDisablePeerAccess ctx

{-# INLINE cuCtxDisablePeerAccess #-}
{# fun unsafe cuCtxDisablePeerAccess
  { useContext `Context' } -> `Status' cToEnum #}
#endif


-- |
-- Queries attributes of the link between two devices
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g4c55c60508f8eba4546b51f2ee545393>
--
-- Requires CUDA-8.0
--
-- @since 0.9.0.0@
--
{-# INLINEABLE getAttribute #-}
getAttribute :: PeerAttribute -> Device -> Device -> IO Int
#if CUDA_VERSION < 8000
getAttribute _      _   _   = requireSDK 'getAttribute 8.0
#else
getAttribute attrib src dst = resultIfOk =<< cuDeviceGetP2PAttribute attrib src dst

{-# INLINE cuDeviceGetP2PAttribute #-}
{# fun unsafe cuDeviceGetP2PAttribute
  { alloca-   `Int' peekIntConv*
  , cFromEnum `PeerAttribute'
  , useDevice `Device'
  , useDevice `Device'
  } -> `Status' cToEnum #}
#endif

