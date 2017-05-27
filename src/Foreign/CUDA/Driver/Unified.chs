{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE ScopedTypeVariables      #-}
{-# LANGUAGE TemplateHaskell          #-}
{-# OPTIONS_HADDOCK prune #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Unified
-- Copyright : [2017] Trevor L. McDonell
-- License   : BSD
--
-- Unified addressing functions for the low-level driver interface
--
-- [/Overview/]
--
-- CUDA devices can share a unified address space with the host. For these
-- devices, there is no distinction between a device pointer and a host
-- pointer---the same pointer value may be used to access memory from the host
-- program and from a kernel running on the device (with exceptions enumerated
-- below).
--
-- [/Support/]
--
-- Whether or not a device supports unified addressing may be queried by calling
-- 'Foreign.CUDA.Driver.Device.attribute' with the
-- 'Foreign.CUDA.Driver.Device.UnifiedAddressing' attribute.
--
-- Unified addressing is automatically enabled in 64-bit processes on devices
-- with compute capability at leas 2.0.
--
-- [/Looking up information about pointers/]
--
-- It is possible to look up information about the memory which backs a pointer;
-- that is, whether the memory resides on the host or the device (and in
-- particular, which device).
--
-- [/Automatic mapping of host memory/]
--
-- All host memory allocated in all contexts using
-- 'Foreign.CUDA.Driver.Marshal.mallocHostArray' or
-- 'Foreign.CUDA.Driver.Marshal.mallocHostForeignPtr' is always directly
-- accessible from all contexts on all devices which support unified addressing.
-- This is the case whether or not the flags
-- 'Foreign.CUDA.Driver.Marshal.Portable' or
-- 'Foreign.CUDA.Driver.Marshal.DeviceMapped' are specified.
--
-- The pointer value through which allocated host memory may be accessed in
-- kernels on all devices which support unified addressing is the same as the
-- pointer value as on the host; that is, it is not necessary to call
-- 'Foreign.CUDA.Driver.Marshal.getDevicePtr' for these allocations.
--
-- Note that this is not the case for memory allocated using the
-- 'Foreign.CUDA.Driver.Marshal.WriteCombined' option; see below.
--
-- [/Automatic registration of peer memory/]
--
-- Upon enabling direct access from a context which supports unified addressing
-- to another peer context which supports unified addressing using
-- 'Foreign.CUDA.Driver.Context.Peer.add', all memory allocated in the peer
-- context will immediately be accessible by the current context. The device
-- pointer values are the same on both contexts.
--
-- [/Exceptions (disjoint addressing/]
--
-- Not all memory may be accessed on devices through the same pointer value
-- as they are accessed with on the host. These exceptions are host arrays
-- registered with 'Foreign.CUDA.Driver.Marshal.registerArray', and those
-- allocated with the flag 'Foreign.CUDA.Driver.Marshal.WriteCombined'. In these
-- cases, the host and device arrays have distinct addresses (pointer values).
-- However, the device address is guaranteed to not overlap with any valid host
-- pointer range and is guaranteed to have the same value across all contexts
-- which support unified addressing.
--
-- The value of the device pointer may be queried with
-- 'Foreign.CUDA.Driver.Marshal.getDevicePtr' from any context supporting
-- unified addressing.
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Unified (

  -- ** Querying pointer attributes
  PointerAttributes(..), MemoryType(..),
  getAttributes,

  -- ** Setting pointer attributes
  Advice(..),
  setSyncMemops,
  advise,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Context
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Marshal
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Types

-- System
import Control.Applicative
import Control.Monad
import Data.Maybe
import Foreign
import Foreign.C
import Foreign.Storable
import Prelude



#if CUDA_VERSION < 7000
data PointerAttributes
data MemoryType
#else
-- | Information about a pointer
--
data PointerAttributes a = PointerAttributes
  { ptrContext    :: {-# UNPACK #-} !Context
  , ptrDevice     :: {-# UNPACK #-} !(DevicePtr a)
  , ptrHost       :: {-# UNPACK #-} !(HostPtr a)
  , ptrBufferID   :: {-# UNPACK #-} !CULLong
  , ptrMemoryType :: !MemoryType
  , ptrSyncMemops :: !Bool
  , ptrIsManaged  :: !Bool
  }
  deriving Show

{# enum CUmemorytype as MemoryType
  { underscoreToCase
  , DEVICE  as DeviceMemory
  , HOST    as HostMemory
  , ARRAY   as ArrayMemory
  , UNIFIED as UnifiedMemory }
  with prefix="CU_MEMORYTYPE" deriving (Eq, Show, Bounded) #}

{# enum CUpointer_attribute as PointerAttribute
  { underscoreToCase }
  with prefix="CU_POINTER" deriving (Eq, Show, Bounded) #}
#endif

#if CUDA_VERSION < 8000
data Advice
#else
{# enum CUmem_advise as Advice
  { underscoreToCase }
  with prefix="CU_MEM_ADVISE" deriving (Eq, Show, Bounded) #}
#endif


-- Return information about a pointer.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g0c28ed0aff848042bc0533110e45820c>
--
-- Requires CUDA-7.0.
--
{-# INLINEABLE getAttributes #-}
getAttributes :: Ptr a -> IO (PointerAttributes a)
#if CUDA_VERSION < 7000
getAttributes _   = requireSDK 'getAttributes 7.0
#else
getAttributes ptr =
  alloca $ \p_ctx  ->
  alloca $ \p_dptr ->
  alloca $ \p_hptr ->
  alloca $ \(p_bid :: Ptr CULLong) ->
  alloca $ \(p_mt  :: Ptr CUInt)   ->
  alloca $ \(p_sm  :: Ptr CInt)    ->
  alloca $ \(p_im  :: Ptr CInt)    -> do
    let n       = length as
        (as,ps) = unzip [ (AttributeContext,       castPtr p_ctx)
                        , (AttributeDevicePointer, castPtr p_dptr)
                        , (AttributeHostPointer,   castPtr p_hptr)
                        , (AttributeBufferId,      castPtr p_bid)
                        , (AttributeMemoryType,    castPtr p_mt)
                        , (AttributeSyncMemops,    castPtr p_sm)
                        , (AttributeIsManaged,     castPtr p_im)
                        ]
    --
    nothingIfOk =<< cuPointerGetAttributes n as ps ptr
    PointerAttributes
      <$> liftM Context (peek p_ctx)
      <*> liftM DevicePtr (peek p_dptr)
      <*> liftM HostPtr   (peek p_hptr)
      <*> peek p_bid
      <*> liftM cToEnum (peek p_mt)
      <*> liftM cToBool (peek p_sm)
      <*> liftM cToBool (peek p_im)

{-# INLINE cuPointerGetAttributes #-}
{# fun unsafe cuPointerGetAttributes
  {            `Int'
  , withAttrs* `[PointerAttribute]'
  , withArray* `[Ptr ()]'
  , useHandle  `Ptr a'
  }
  -> `Status' cToEnum #}
  where
    withAttrs as = withArray (map cFromEnum as)
    useHandle    = fromIntegral . ptrToIntPtr
#endif


-- Set whether or not the given memory region is guaranteed to always
-- synchronise memory operations that are synchronous. If there are some
-- previously initiated synchronous memory operations that are pending when this
-- attribute is set, the function does not return until those memory operations
-- are complete. See
-- <http://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html API
-- synchronisation behaviour> for more information on cases where synchronous
-- memory operations can exhibit asynchronous behaviour.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g89f7ad29a657e574fdea2624b74d138e>
--
-- Requires CUDA-7.0.
--
{-# INLINE setSyncMemops #-}
setSyncMemops :: Ptr a -> Bool -> IO ()
#if CUDA_VERSION < 7000
setSyncMemops _   _   = requireSDK 'setSyncMemops 7.0
#else
setSyncMemops ptr val = nothingIfOk =<< cuPointerSetAttribute val AttributeSyncMemops ptr

{-# INLINE cuPointerSetAttribute #-}
{# fun unsafe cuPointerSetAttribute
  { withBool'* `Bool'
  , cFromEnum  `PointerAttribute'
  , useHandle  `Ptr a'
  }
  -> `Status' cToEnum #}
  where
    withBool' :: Bool -> (Ptr () -> IO b) -> IO b
    withBool' v k = with (fromBool v :: CUInt) (k . castPtr)

    useHandle = fromIntegral . ptrToIntPtr
#endif


-- | Advise about the usage of a given range of memory. If the supplied device
-- is Nothing, then the preferred location is taken to mean the CPU.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g27608c857a9254789c13f3e3b72029e2>
--
-- Requires CUDA-8.0.
--
{-# INLINEABLE advise #-}
advise :: Storable a => Ptr a -> Int -> Advice -> Maybe Device -> IO ()
#if CUDA_VERSION < 8000
advise _   _ _ _    = requireSDK 'advise 8.0
#else
advise ptr n a mdev = go undefined ptr
  where
    go :: Storable a' => a' -> Ptr a' -> IO ()
    go x _ = nothingIfOk =<< cuMemAdvise ptr (n * sizeOf x) a (maybe (-1) useDevice mdev)

{-# INLINE cuMemAdvise #-}
{# fun unsafe cuMemAdvise
  { useHandle `Ptr a'
  ,           `Int'
  , cFromEnum `Advice'
  ,           `CInt'
  }
  -> `Status' cToEnum #}
  where
    useHandle = fromIntegral . ptrToIntPtr
#endif

