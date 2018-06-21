{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash                #-}
{-# LANGUAGE TemplateHaskell          #-}
{-# OPTIONS_HADDOCK prune #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Marshal
-- Copyright : [2009..2017] Trevor L. McDonell
-- License   : BSD
--
-- Memory management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Marshal (

  -- * Host Allocation
  AllocFlag(..),
  mallocHostArray, mallocHostForeignPtr, freeHost,
  registerArray, unregisterArray,

  -- * Device Allocation
  mallocArray, allocaArray, free,

  -- * Unified Memory Allocation
  AttachFlag(..),
  mallocManagedArray,
  prefetchArrayAsync,

  -- * Marshalling
  peekArray, peekArrayAsync, peekArray2D, peekArray2DAsync, peekListArray,
  pokeArray, pokeArrayAsync, pokeArray2D, pokeArray2DAsync, pokeListArray,
  copyArray, copyArrayAsync, copyArray2D, copyArray2DAsync,
  copyArrayPeer, copyArrayPeerAsync,

  -- * Combined Allocation and Marshalling
  newListArray,  newListArrayLen,
  withListArray, withListArrayLen,

  -- * Utility
  memset, memsetAsync,
  getDevicePtr, getBasePtr, getMemInfo,

  -- Internal
  useDeviceHandle, peekDeviceHandle

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream                       ( Stream(..), defaultStream )
import Foreign.CUDA.Driver.Context.Base                 ( Context(..) )
import Foreign.CUDA.Internal.C2HS

-- System
import Data.Int
import Data.Maybe
import Data.Word
import Unsafe.Coerce
import Control.Applicative
import Control.Exception
import Prelude

import Foreign.C
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.Storable
import qualified Foreign.Marshal                        as F

import GHC.Ptr
import GHC.Word
import GHC.Base

#c
typedef enum CUmemhostalloc_option_enum {
    CU_MEMHOSTALLOC_OPTION_PORTABLE       = CU_MEMHOSTALLOC_PORTABLE,
    CU_MEMHOSTALLOC_OPTION_DEVICE_MAPPED  = CU_MEMHOSTALLOC_DEVICEMAP,
    CU_MEMHOSTALLOC_OPTION_WRITE_COMBINED = CU_MEMHOSTALLOC_WRITECOMBINED
} CUmemhostalloc_option;
#endc


--------------------------------------------------------------------------------
-- Host Allocation
--------------------------------------------------------------------------------

-- |
-- Options for host allocation
--
{# enum CUmemhostalloc_option as AllocFlag
    { underscoreToCase }
    with prefix="CU_MEMHOSTALLOC_OPTION" deriving (Eq, Show, Bounded) #}

-- |
-- Allocate a section of linear memory on the host which is page-locked and
-- directly accessible from the device. The storage is sufficient to hold the
-- given number of elements of a storable type.
--
-- Note that since the amount of pageable memory is thusly reduced, overall
-- system performance may suffer. This is best used sparingly to allocate
-- staging areas for data exchange.
--
-- Host memory allocated in this way is automatically and immediately
-- accessible to all contexts on all devices which support unified
-- addressing.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0>
--
{-# INLINEABLE mallocHostArray #-}
mallocHostArray :: Storable a => [AllocFlag] -> Int -> IO (HostPtr a)
mallocHostArray !flags = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (HostPtr a')
    doMalloc x !n = resultIfOk =<< cuMemHostAlloc (n * sizeOf x) flags

-- |
-- As 'mallocHostArray', but return a 'ForeignPtr' instead. The array will be
-- deallocated automatically once the last reference to the 'ForeignPtr' is
-- dropped.
--
{-# INLINEABLE mallocHostForeignPtr #-}
{-# SPECIALISE mallocHostForeignPtr :: [AllocFlag] -> Int -> IO (ForeignPtr Word8) #-}
mallocHostForeignPtr :: Storable a => [AllocFlag] -> Int -> IO (ForeignPtr a)
mallocHostForeignPtr !flags !size = do
  HostPtr ptr <- mallocHostArray flags size
  newForeignPtr finalizerMemFreeHost ptr

{-# INLINE cuMemHostAlloc #-}
{# fun unsafe cuMemHostAlloc
  { alloca'-        `HostPtr a'   peekHP*
  ,                 `Int'
  , combineBitMasks `[AllocFlag]'         } -> `Status' cToEnum #}
  where
    alloca' !f = F.alloca $ \ !p -> poke p nullPtr >> f (castPtr p)
    peekHP !p  = HostPtr . castPtr <$> peek p

-- Pointer to the foreign function to release host arrays, which may be used as
-- a finalizer. Technically this function has a non-void return type, but I am
-- hoping that that doesn't matter...
--
foreign import ccall "&cuMemFreeHost" finalizerMemFreeHost :: FinalizerPtr a

-- |
-- Free a section of page-locked host memory.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c>
--
{-# INLINEABLE freeHost #-}
freeHost :: HostPtr a -> IO ()
freeHost !p = nothingIfOk =<< cuMemFreeHost p

{-# INLINE cuMemFreeHost #-}
{# fun unsafe cuMemFreeHost
  { useHP `HostPtr a' } -> `Status' cToEnum #}
  where
    useHP = castPtr . useHostPtr


-- |
-- Page-locks the specified array (on the host) and maps it for the device(s) as
-- specified by the given allocation flags. Subsequently, the memory is accessed
-- directly by the device so can be read and written with much higher bandwidth
-- than pageable memory that has not been registered. The memory range is added
-- to the same tracking mechanism as 'mallocHostArray' to automatically
-- accelerate calls to functions such as 'pokeArray'.
--
-- Note that page-locking excessive amounts of memory may degrade system
-- performance, since it reduces the amount of pageable memory available. This
-- is best used sparingly to allocate staging areas for data exchange.
--
-- This function has limited support on Mac OS X. OS 10.7 or later is
-- required.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223>
--
{-# INLINEABLE registerArray #-}
registerArray :: Storable a => [AllocFlag] -> Int -> Ptr a -> IO (HostPtr a)
#if CUDA_VERSION < 4000
registerArray _ _ _     = requireSDK 'registerArray 4.0
#else
registerArray !flags !n = go undefined
  where
    go :: Storable b => b -> Ptr b -> IO (HostPtr b)
    go x !p = do
      status <- cuMemHostRegister p (n * sizeOf x) flags
      resultIfOk (status,HostPtr p)

{-# INLINE cuMemHostRegister #-}
{# fun unsafe cuMemHostRegister
  { castPtr         `Ptr a'
  ,                 `Int'
  , combineBitMasks `[AllocFlag]' } -> `Status' cToEnum #}
#endif


-- |
-- Unmaps the memory from the given pointer, and makes it pageable again.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g63f450c8125359be87b7623b1c0b2a14>
--
{-# INLINEABLE unregisterArray #-}
unregisterArray :: HostPtr a -> IO (Ptr a)
#if CUDA_VERSION < 4000
unregisterArray _           = requireSDK 'unregisterArray 4.0
#else
unregisterArray (HostPtr !p) = do
  status <- cuMemHostUnregister p
  resultIfOk (status,p)

{-# INLINE cuMemHostUnregister #-}
{# fun unsafe cuMemHostUnregister
  { castPtr `Ptr a' } -> `Status' cToEnum #}
#endif


--------------------------------------------------------------------------------
-- Device Allocation
--------------------------------------------------------------------------------

-- |
-- Allocate a section of linear memory on the device, and return a reference to
-- it. The memory is sufficient to hold the given number of elements of storable
-- type. It is suitably aligned for any type, and is not cleared.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467>
--
{-# INLINEABLE mallocArray #-}
mallocArray :: Storable a => Int -> IO (DevicePtr a)
mallocArray = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (DevicePtr a')
    doMalloc x !n = resultIfOk =<< cuMemAlloc (n * sizeOf x)

{-# INLINE cuMemAlloc #-}
{# fun unsafe cuMemAlloc
  { alloca'- `DevicePtr a' peekDeviceHandle*
  ,          `Int'                           } -> `Status' cToEnum #}
  where
    alloca' !f = F.alloca $ \ !p -> poke p nullPtr >> f (castPtr p)


-- |
-- Execute a computation on the device, passing a pointer to a temporarily
-- allocated block of memory sufficient to hold the given number of elements of
-- storable type. The memory is freed when the computation terminates (normally
-- or via an exception), so the pointer must not be used after this.
--
-- Note that kernel launches can be asynchronous, so you may want to add a
-- synchronisation point using 'Foreign.CUDA.Driver.Context.sync' as part
-- of the continuation.
--
{-# INLINEABLE allocaArray #-}
allocaArray :: Storable a => Int -> (DevicePtr a -> IO b) -> IO b
allocaArray !n = bracket (mallocArray n) free


-- |
-- Release a section of device memory.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a>
--
{-# INLINEABLE free #-}
free :: DevicePtr a -> IO ()
free !dp = nothingIfOk =<< cuMemFree dp

{-# INLINE cuMemFree #-}
{# fun unsafe cuMemFree
  { useDeviceHandle `DevicePtr a' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Unified memory allocations
--------------------------------------------------------------------------------

-- |
-- Options for unified memory allocations
--
#if CUDA_VERSION < 6000
data AttachFlag
#else
{# enum CUmemAttach_flags as AttachFlag
    { underscoreToCase }
    with prefix="CU_MEM_ATTACH_OPTION" deriving (Eq, Show, Bounded) #}
#endif

-- |
-- Allocates memory that will be automatically managed by the Unified Memory
-- system. The returned pointer is valid on the CPU and on all GPUs which
-- supported managed memory. All accesses to this pointer must obey the
-- Unified Memory programming model.
--
-- On a multi-GPU system with peer-to-peer support, where multiple GPUs
-- support managed memory, the physical storage is created on the GPU which
-- is active at the time 'mallocManagedArray' is called. All other GPUs
-- will access the array at reduced bandwidth via peer mapping over the
-- PCIe bus. The Unified Memory system does not migrate memory between
-- GPUs.
--
-- On a multi-GPU system where multiple GPUs support managed memory, but
-- not all pairs of such GPUs have peer-to-peer support between them, the
-- physical storage is allocated in system memory (zero-copy memory) and
-- all GPUs will access the data at reduced bandwidth over the PCIe bus.
--
-- Requires CUDA-6.0
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32>
--
{-# INLINEABLE mallocManagedArray #-}
mallocManagedArray :: Storable a => [AttachFlag] -> Int -> IO (DevicePtr a)
#if CUDA_VERSION < 6000
mallocManagedArray _ _    = requireSDK 'mallocManagedArray 6.0
#else
mallocManagedArray !flags = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (DevicePtr a')
    doMalloc x !n = resultIfOk =<< cuMemAllocManaged (n * sizeOf x) flags

{-# INLINE cuMemAllocManaged #-}
{# fun unsafe cuMemAllocManaged
  { alloca'-        `DevicePtr a' peekDeviceHandle*
  ,                 `Int'
  , combineBitMasks `[AttachFlag]'                  } -> `Status' cToEnum #}
  where
    alloca' !f = F.alloca $ \ !p -> poke p nullPtr >> f (castPtr p)
#endif


-- | Pre-fetches the given number of elements to the specified destination
-- device. If the specified device is Nothing, the data is pre-fetched to host
-- memory. The pointer must refer to a memory range allocated with
-- 'mallocManagedArray'.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gfe94f8b7fb56291ebcea44261aa4cb84>
--
-- Requires CUDA-8.0.
--
prefetchArrayAsync :: Storable a => DevicePtr a -> Int -> Maybe Device -> Maybe Stream -> IO ()
#if CUDA_VERSION < 8000
prefetchArrayAsync _   _ _    _   = requireSDK 'prefetchArrayAsync 8.0
#else
prefetchArrayAsync ptr n mdev mst = go undefined ptr
  where
    go :: Storable a' => a' -> DevicePtr a' -> IO ()
    go x _ = nothingIfOk =<< cuMemPrefetchAsync ptr (n * sizeOf x) (maybe (-1) useDevice mdev) (fromMaybe defaultStream mst)

{-# INLINE cuMemPrefetchAsync #-}
{# fun unsafe cuMemPrefetchAsync
  { useDeviceHandle `DevicePtr a'
  ,                 `Int'
  , id              `CInt'
  , useStream       `Stream'
  }
  -> `Status' cToEnum #}
#endif


--------------------------------------------------------------------------------
-- Marshalling
--------------------------------------------------------------------------------

-- Device -> Host
-- --------------

-- |
-- Copy a number of elements from the device to host memory. This is a
-- synchronous operation.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893>
--
{-# INLINEABLE peekArray #-}
peekArray :: Storable a => Int -> DevicePtr a -> Ptr a -> IO ()
peekArray !n !dptr !hptr = doPeek undefined dptr
  where
    doPeek :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPeek x _ = nothingIfOk =<< cuMemcpyDtoH hptr dptr (n * sizeOf x)

{-# INLINE cuMemcpyDtoH #-}
{# fun cuMemcpyDtoH
  { castPtr         `Ptr a'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'         } -> `Status' cToEnum #}


-- |
-- Copy memory from the device asynchronously, possibly associated with a
-- particular stream. The destination host memory must be page-locked.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362>
--
{-# INLINEABLE peekArrayAsync #-}
peekArrayAsync :: Storable a => Int -> DevicePtr a -> HostPtr a -> Maybe Stream -> IO ()
peekArrayAsync !n !dptr !hptr !mst = doPeek undefined dptr
  where
    doPeek :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPeek x _ = nothingIfOk =<< cuMemcpyDtoHAsync hptr dptr (n * sizeOf x) (fromMaybe defaultStream mst)

{-# INLINE cuMemcpyDtoHAsync #-}
{# fun cuMemcpyDtoHAsync
  { useHP           `HostPtr a'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}
  where
    useHP = castPtr . useHostPtr


-- |
-- Copy a 2D array from the device to the host.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27>
--
{-# INLINEABLE peekArray2D #-}
peekArray2D
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> DevicePtr a              -- ^ source array
    -> Int                      -- ^ source array width
    -> Int                      -- ^ source x-coordinate
    -> Int                      -- ^ source y-coordinate
    -> Ptr a                    -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Int                      -- ^ destination x-coordinate
    -> Int                      -- ^ destination y-coordinate
    -> IO ()
peekArray2D !w !h !dptr !dw !dx !dy !hptr !hw !hx !hy = doPeek undefined dptr
  where
    doPeek :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPeek x _ =
      let bytes = sizeOf x
          w'    = w  * bytes
          hw'   = hw * bytes
          hx'   = hx * bytes
          dw'   = dw * bytes
          dx'   = dx * bytes
      in
      nothingIfOk =<< cuMemcpy2DDtoH hptr hw' hx' hy dptr dw' dx' dy w' h

{-# INLINE cuMemcpy2DDtoH #-}
{# fun cuMemcpy2DDtoH
  { castPtr         `Ptr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  }
  -> `Status' cToEnum #}


-- |
-- Copy a 2D array from the device to the host asynchronously, possibly
-- associated with a particular execution stream. The destination host memory
-- must be page-locked.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274>
--
{-# INLINEABLE peekArray2DAsync #-}
peekArray2DAsync
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> DevicePtr a              -- ^ source array
    -> Int                      -- ^ source array width
    -> Int                      -- ^ source x-coordinate
    -> Int                      -- ^ source y-coordinate
    -> HostPtr a                -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Int                      -- ^ destination x-coordinate
    -> Int                      -- ^ destination y-coordinate
    -> Maybe Stream             -- ^ stream to associate to
    -> IO ()
peekArray2DAsync !w !h !dptr !dw !dx !dy !hptr !hw !hx !hy !mst = doPeek undefined dptr
  where
    doPeek :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPeek x _ =
      let bytes = sizeOf x
          w'    = w  * bytes
          hw'   = hw * bytes
          hx'   = hx * bytes
          dw'   = dw * bytes
          dx'   = dx * bytes
          st    = fromMaybe defaultStream mst
      in
      nothingIfOk =<< cuMemcpy2DDtoHAsync hptr hw' hx' hy dptr dw' dx' dy w' h st

{-# INLINE cuMemcpy2DDtoHAsync #-}
{# fun cuMemcpy2DDtoHAsync
  { useHP           `HostPtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , useStream       `Stream'
  }
  -> `Status' cToEnum #}
  where
    useHP = castPtr . useHostPtr


-- |
-- Copy a number of elements from the device into a new Haskell list. Note that
-- this requires two memory copies: firstly from the device into a heap
-- allocated array, and from there marshalled into a list.
--
{-# INLINEABLE peekListArray #-}
peekListArray :: Storable a => Int -> DevicePtr a -> IO [a]
peekListArray !n !dptr =
  F.allocaArray n $ \p -> do
    peekArray   n dptr p
    F.peekArray n p


-- Host -> Device
-- --------------

-- |
-- Copy a number of elements onto the device. This is a synchronous operation.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169>
--
{-# INLINEABLE pokeArray #-}
pokeArray :: Storable a => Int -> Ptr a -> DevicePtr a -> IO ()
pokeArray !n !hptr !dptr = doPoke undefined dptr
  where
    doPoke :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPoke x _ = nothingIfOk =<< cuMemcpyHtoD dptr hptr (n * sizeOf x)

{-# INLINE cuMemcpyHtoD #-}
{# fun cuMemcpyHtoD
  { useDeviceHandle `DevicePtr a'
  , castPtr         `Ptr a'
  ,                 `Int'         } -> `Status' cToEnum #}


-- |
-- Copy memory onto the device asynchronously, possibly associated with a
-- particular stream. The source host memory must be page-locked.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3>
--
{-# INLINEABLE pokeArrayAsync #-}
pokeArrayAsync :: Storable a => Int -> HostPtr a -> DevicePtr a -> Maybe Stream -> IO ()
pokeArrayAsync !n !hptr !dptr !mst = dopoke undefined dptr
  where
    dopoke :: Storable a' => a' -> DevicePtr a' -> IO ()
    dopoke x _ = nothingIfOk =<< cuMemcpyHtoDAsync dptr hptr (n * sizeOf x) (fromMaybe defaultStream mst)

{-# INLINE cuMemcpyHtoDAsync #-}
{# fun cuMemcpyHtoDAsync
  { useDeviceHandle `DevicePtr a'
  , useHP           `HostPtr a'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}
  where
    useHP = castPtr . useHostPtr


-- |
-- Copy a 2D array from the host to the device.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27>
--
{-# INLINEABLE pokeArray2D #-}
pokeArray2D
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> Ptr a                    -- ^ source array
    -> Int                      -- ^ source array width
    -> Int                      -- ^ source x-coordinate
    -> Int                      -- ^ source y-coordinate
    -> DevicePtr a              -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Int                      -- ^ destination x-coordinate
    -> Int                      -- ^ destination y-coordinate
    -> IO ()
pokeArray2D !w !h !hptr !hw !hx !hy !dptr !dw !dx !dy = doPoke undefined dptr
  where
    doPoke :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPoke x _ =
      let bytes = sizeOf x
          w'    = w  * bytes
          hw'   = hw * bytes
          hx'   = hx * bytes
          dw'   = dw * bytes
          dx'   = dx * bytes
      in
      nothingIfOk =<< cuMemcpy2DHtoD dptr dw' dx' dy hptr hw' hx' hy w' h

{-# INLINE cuMemcpy2DHtoD #-}
{# fun cuMemcpy2DHtoD
  { useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , castPtr         `Ptr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  }
  -> `Status' cToEnum #}


-- |
-- Copy a 2D array from the host to the device asynchronously, possibly
-- associated with a particular execution stream. The source host memory must be
-- page-locked.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274>
--
{-# INLINEABLE pokeArray2DAsync #-}
pokeArray2DAsync
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> HostPtr a                -- ^ source array
    -> Int                      -- ^ source array width
    -> Int                      -- ^ source x-coordinate
    -> Int                      -- ^ source y-coordinate
    -> DevicePtr a              -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Int                      -- ^ destination x-coordinate
    -> Int                      -- ^ destination y-coordinate
    -> Maybe Stream             -- ^ stream to associate to
    -> IO ()
pokeArray2DAsync !w !h !hptr !hw !hx !hy !dptr !dw !dx !dy !mst = doPoke undefined dptr
  where
    doPoke :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPoke x _ =
      let bytes = sizeOf x
          w'    = w  * bytes
          hw'   = hw * bytes
          hx'   = hx * bytes
          dw'   = dw * bytes
          dx'   = dx * bytes
          st    = fromMaybe defaultStream mst
      in
      nothingIfOk =<< cuMemcpy2DHtoDAsync dptr dw' dx' dy hptr hw' hx' hy w' h st

{-# INLINE cuMemcpy2DHtoDAsync #-}
{# fun cuMemcpy2DHtoDAsync
  { useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , useHP           `HostPtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , useStream       `Stream'
  }
  -> `Status' cToEnum #}
  where
    useHP = castPtr . useHostPtr


-- |
-- Write a list of storable elements into a device array. The device array must
-- be sufficiently large to hold the entire list. This requires two marshalling
-- operations.
--
{-# INLINEABLE pokeListArray #-}
pokeListArray :: Storable a => [a] -> DevicePtr a -> IO ()
pokeListArray !xs !dptr = F.withArrayLen xs $ \ !len !p -> pokeArray len p dptr


-- Device -> Device
-- ----------------

-- |
-- Copy the given number of elements from the first device array (source) to the
-- second device (destination). The copied areas may not overlap. This operation
-- is asynchronous with respect to the host, but will never overlap with kernel
-- execution.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b>
--
{-# INLINEABLE copyArray #-}
copyArray :: Storable a => Int -> DevicePtr a -> DevicePtr a -> IO ()
copyArray !n = docopy undefined
  where
    docopy :: Storable a' => a' -> DevicePtr a' -> DevicePtr a' -> IO ()
    docopy x src dst = nothingIfOk =<< cuMemcpyDtoD dst src (n * sizeOf x)

{-# INLINE cuMemcpyDtoD #-}
{# fun unsafe cuMemcpyDtoD
  { useDeviceHandle `DevicePtr a'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'         } -> `Status' cToEnum #}


-- |
-- Copy the given number of elements from the first device array (source) to the
-- second device array (destination). The copied areas may not overlap. The
-- operation is asynchronous with respect to the host, and can be asynchronous
-- to other device operations by associating it with a particular stream.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8>
--
{-# INLINEABLE copyArrayAsync #-}
copyArrayAsync :: Storable a => Int -> DevicePtr a -> DevicePtr a -> Maybe Stream -> IO ()
copyArrayAsync !n !src !dst !mst = docopy undefined src
  where
    docopy :: Storable a' => a' -> DevicePtr a' -> IO ()
    docopy x _ = nothingIfOk =<< cuMemcpyDtoDAsync dst src (n * sizeOf x) (fromMaybe defaultStream mst)

{-# INLINE cuMemcpyDtoDAsync #-}
{# fun unsafe cuMemcpyDtoDAsync
  { useDeviceHandle `DevicePtr a'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}


-- |
-- Copy a 2D array from the first device array (source) to the second device
-- array (destination). The copied areas must not overlap. This operation is
-- asynchronous with respect to the host, but will never overlap with kernel
-- execution.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27>
--
{-# INLINEABLE copyArray2D #-}
copyArray2D
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> DevicePtr a              -- ^ source array
    -> Int                      -- ^ source array width
    -> Int                      -- ^ source x-coordinate
    -> Int                      -- ^ source y-coordinate
    -> DevicePtr a              -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Int                      -- ^ destination x-coordinate
    -> Int                      -- ^ destination y-coordinate
    -> IO ()
copyArray2D !w !h !src !hw !hx !hy !dst !dw !dx !dy = doCopy undefined dst
  where
    doCopy :: Storable a' => a' -> DevicePtr a' -> IO ()
    doCopy x _ =
      let bytes = sizeOf x
          w'    = w  * bytes
          hw'   = hw * bytes
          hx'   = hx * bytes
          dw'   = dw * bytes
          dx'   = dx * bytes
      in
      nothingIfOk =<< cuMemcpy2DDtoD dst dw' dx' dy src hw' hx' hy w' h

{-# INLINE cuMemcpy2DDtoD #-}
{# fun unsafe cuMemcpy2DDtoD
  { useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  }
  -> `Status' cToEnum #}


-- |
-- Copy a 2D array from the first device array (source) to the second device
-- array (destination). The copied areas may not overlap. The operation is
-- asynchronous with respect to the host, and can be asynchronous to other
-- device operations by associating it with a particular execution stream.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274>
--
{-# INLINEABLE copyArray2DAsync #-}
copyArray2DAsync
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> DevicePtr a              -- ^ source array
    -> Int                      -- ^ source array width
    -> Int                      -- ^ source x-coordinate
    -> Int                      -- ^ source y-coordinate
    -> DevicePtr a              -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Int                      -- ^ destination x-coordinate
    -> Int                      -- ^ destination y-coordinate
    -> Maybe Stream             -- ^ stream to associate to
    -> IO ()
copyArray2DAsync !w !h !src !hw !hx !hy !dst !dw !dx !dy !mst = doCopy undefined dst
  where
    doCopy :: Storable a' => a' -> DevicePtr a' -> IO ()
    doCopy x _ =
      let bytes = sizeOf x
          w'    = w  * bytes
          hw'   = hw * bytes
          hx'   = hx * bytes
          dw'   = dw * bytes
          dx'   = dx * bytes
          st    = fromMaybe defaultStream mst
      in
      nothingIfOk =<< cuMemcpy2DDtoDAsync dst dw' dx' dy src hw' hx' hy w' h st

{-# INLINE cuMemcpy2DDtoDAsync #-}
{# fun unsafe cuMemcpy2DDtoDAsync
  { useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int'
  , useStream       `Stream'
  }
  -> `Status' cToEnum #}



-- Context -> Context
-- ------------------

-- |
-- Copies an array from device memory in one context to device memory in another
-- context. Note that this function is asynchronous with respect to the host,
-- but serialised with respect to all pending and future asynchronous work in
-- the source and destination contexts. To avoid this synchronisation, use
-- 'copyArrayPeerAsync' instead.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1ge1f5c7771544fee150ada8853c7cbf4a>
--
{-# INLINEABLE copyArrayPeer #-}
copyArrayPeer :: Storable a
              => Int                            -- ^ number of array elements
              -> DevicePtr a -> Context         -- ^ source array and context
              -> DevicePtr a -> Context         -- ^ destination array and context
              -> IO ()
#if CUDA_VERSION < 4000
copyArrayPeer _ _ _ _ _                    = requireSDK 'copyArrayPeer 4.0
#else
copyArrayPeer !n !src !srcCtx !dst !dstCtx = go undefined src dst
  where
    go :: Storable b => b -> DevicePtr b -> DevicePtr b -> IO ()
    go x _ _ = nothingIfOk =<< cuMemcpyPeer dst dstCtx src srcCtx (n * sizeOf x)

{-# INLINE cuMemcpyPeer #-}
{# fun unsafe cuMemcpyPeer
  { useDeviceHandle `DevicePtr a'
  , useContext      `Context'
  , useDeviceHandle `DevicePtr a'
  , useContext      `Context'
  ,                 `Int'         } -> `Status' cToEnum #}
#endif


-- |
-- Copies from device memory in one context to device memory in another context.
-- Note that this function is asynchronous with respect to the host and all work
-- in other streams and devices.
--
-- Requires CUDA-4.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2>
--
{-# INLINEABLE copyArrayPeerAsync #-}
copyArrayPeerAsync :: Storable a
                   => Int                       -- ^ number of array elements
                   -> DevicePtr a -> Context    -- ^ source array and context
                   -> DevicePtr a -> Context    -- ^ destination array and device context
                   -> Maybe Stream              -- ^ stream to associate with
                   -> IO ()
#if CUDA_VERSION < 4000
copyArrayPeerAsync _ _ _ _ _ _                      = requireSDK 'copyArrayPeerAsync 4.0
#else
copyArrayPeerAsync !n !src !srcCtx !dst !dstCtx !st = go undefined src dst
  where
    go :: Storable b => b -> DevicePtr b -> DevicePtr b -> IO ()
    go x _ _ = nothingIfOk =<< cuMemcpyPeerAsync dst dstCtx src srcCtx (n * sizeOf x) stream
    stream   = fromMaybe defaultStream st

{-# INLINE cuMemcpyPeerAsync #-}
{# fun unsafe cuMemcpyPeerAsync
  { useDeviceHandle `DevicePtr a'
  , useContext      `Context'
  , useDeviceHandle `DevicePtr a'
  , useContext      `Context'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}
#endif


--------------------------------------------------------------------------------
-- Combined Allocation and Marshalling
--------------------------------------------------------------------------------

-- |
-- Write a list of storable elements into a newly allocated device array,
-- returning the device pointer together with the number of elements that were
-- written. Note that this requires two memory copies: firstly from a Haskell
-- list to a heap allocated array, and from there onto the graphics device. The
-- memory should be 'free'd when no longer required.
--
{-# INLINEABLE newListArrayLen #-}
newListArrayLen :: Storable a => [a] -> IO (DevicePtr a, Int)
newListArrayLen xs =
  F.withArrayLen xs                     $ \len p ->
  bracketOnError (mallocArray len) free $ \d_xs  -> do
    pokeArray len p d_xs
    return (d_xs, len)


-- |
-- Write a list of storable elements into a newly allocated device array. This
-- is 'newListArrayLen' composed with 'fst'.
--
{-# INLINEABLE newListArray #-}
newListArray :: Storable a => [a] -> IO (DevicePtr a)
newListArray xs = fst `fmap` newListArrayLen xs


-- |
-- Temporarily store a list of elements into a newly allocated device array. An
-- IO action is applied to to the array, the result of which is returned.
-- Similar to 'newListArray', this requires copying the data twice.
--
-- As with 'allocaArray', the memory is freed once the action completes, so you
-- should not return the pointer from the action, and be wary of asynchronous
-- kernel execution.
--
{-# INLINEABLE withListArray #-}
withListArray :: Storable a => [a] -> (DevicePtr a -> IO b) -> IO b
withListArray xs = withListArrayLen xs . const


-- |
-- A variant of 'withListArray' which also supplies the number of elements in
-- the array to the applied function
--
{-# INLINEABLE withListArrayLen #-}
withListArrayLen :: Storable a => [a] -> (Int -> DevicePtr a -> IO b) -> IO b
withListArrayLen xs f =
  bracket (newListArrayLen xs) (free . fst) (uncurry . flip $ f)
--
-- XXX: Will this attempt to double-free the device array on error (together
-- with newListArrayLen)?
--


--------------------------------------------------------------------------------
-- Utility
--------------------------------------------------------------------------------

-- |
-- Set a number of data elements to the specified value, which may be either 8-,
-- 16-, or 32-bits wide.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b>
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c>
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132>
--
{-# INLINEABLE memset #-}
memset :: Storable a => DevicePtr a -> Int -> a -> IO ()
memset !dptr !n !val = case sizeOf val of
    1 -> nothingIfOk =<< cuMemsetD8  dptr val n
    2 -> nothingIfOk =<< cuMemsetD16 dptr val n
    4 -> nothingIfOk =<< cuMemsetD32 dptr val n
    _ -> cudaError "can only memset 8-, 16-, and 32-bit values"

--
-- We use unsafe coerce below to reinterpret the bits of the value to memset as,
-- into the integer type required by the setting functions.
--
{-# INLINE cuMemsetD8 #-}
{# fun unsafe cuMemsetD8
  { useDeviceHandle `DevicePtr a'
  , unsafeCoerce    `a'
  ,                 `Int'         } -> `Status' cToEnum #}

{-# INLINE cuMemsetD16 #-}
{# fun unsafe cuMemsetD16
  { useDeviceHandle `DevicePtr a'
  , unsafeCoerce    `a'
  ,                 `Int'         } -> `Status' cToEnum #}

{-# INLINE cuMemsetD32 #-}
{# fun unsafe cuMemsetD32
  { useDeviceHandle `DevicePtr a'
  , unsafeCoerce    `a'
  ,                 `Int'         } -> `Status' cToEnum #}


-- |
-- Set the number of data elements to the specified value, which may be either
-- 8-, 16-, or 32-bits wide. The operation is asynchronous and may optionally be
-- associated with a stream.
--
-- Requires CUDA-3.2.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627>
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c>
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5>
--
{-# INLINEABLE memsetAsync #-}
memsetAsync :: Storable a => DevicePtr a -> Int -> a -> Maybe Stream -> IO ()
#if CUDA_VERSION < 3020
memsetAsync _ _ _ _            = requireSDK 'memsetAsync 3.2
#else
memsetAsync !dptr !n !val !mst = case sizeOf val of
    1 -> nothingIfOk =<< cuMemsetD8Async  dptr val n stream
    2 -> nothingIfOk =<< cuMemsetD16Async dptr val n stream
    4 -> nothingIfOk =<< cuMemsetD32Async dptr val n stream
    _ -> cudaError "can only memset 8-, 16-, and 32-bit values"
    where
      stream = fromMaybe defaultStream mst

{-# INLINE cuMemsetD8Async #-}
{# fun unsafe cuMemsetD8Async
  { useDeviceHandle `DevicePtr a'
  , unsafeCoerce    `a'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}

{-# INLINE cuMemsetD16Async #-}
{# fun unsafe cuMemsetD16Async
  { useDeviceHandle `DevicePtr a'
  , unsafeCoerce    `a'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}

{-# INLINE cuMemsetD32Async #-}
{# fun unsafe cuMemsetD32Async
  { useDeviceHandle `DevicePtr a'
  , unsafeCoerce    `a'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}
#endif


-- |
-- Return the device pointer associated with a mapped, pinned host buffer, which
-- was allocated with the 'DeviceMapped' option by 'mallocHostArray'.
--
-- Currently, no options are supported and this must be empty.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0>
--
{-# INLINEABLE getDevicePtr #-}
getDevicePtr :: [AllocFlag] -> HostPtr a -> IO (DevicePtr a)
getDevicePtr !flags !hp = resultIfOk =<< cuMemHostGetDevicePointer hp flags

{-# INLINE cuMemHostGetDevicePointer #-}
{# fun unsafe cuMemHostGetDevicePointer
  { alloca'-        `DevicePtr a' peekDeviceHandle*
  , useHP           `HostPtr a'
  , combineBitMasks `[AllocFlag]'                   } -> `Status' cToEnum #}
  where
    alloca'  = F.alloca
    useHP    = castPtr . useHostPtr


-- |
-- Return the base address and allocation size of the given device pointer.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b>
--
{-# INLINEABLE getBasePtr #-}
getBasePtr :: DevicePtr a -> IO (DevicePtr a, Int64)
getBasePtr !dptr = do
  (status,base,size) <- cuMemGetAddressRange dptr
  resultIfOk (status, (base,size))

{-# INLINE cuMemGetAddressRange #-}
{# fun unsafe cuMemGetAddressRange
  { alloca'-        `DevicePtr a' peekDeviceHandle*
  , alloca'-        `Int64'       peekIntConv*
  , useDeviceHandle `DevicePtr a'                   } -> `Status' cToEnum #}
  where
    alloca' :: Storable a => (Ptr a -> IO b) -> IO b
    alloca' = F.alloca

-- |
-- Return the amount of free and total memory respectively available to the
-- current context (bytes).
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0>
--
{-# INLINEABLE getMemInfo #-}
getMemInfo :: IO (Int64, Int64)
getMemInfo = do
  (!status,!f,!t) <- cuMemGetInfo
  resultIfOk (status,(f,t))

{-# INLINE cuMemGetInfo #-}
{# fun unsafe cuMemGetInfo
  { alloca'- `Int64' peekIntConv*
  , alloca'- `Int64' peekIntConv* } -> `Status' cToEnum #}
  where
    alloca' = F.alloca


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

type DeviceHandle = {# type CUdeviceptr #}

-- Lift an opaque handle to a typed DevicePtr representation. This occasions
-- arcane distinctions for the different driver versions and Tesla (compute 1.x)
-- and Fermi (compute 2.x) class architectures on 32- and 64-bit hosts.
--
{-# INLINE peekDeviceHandle #-}
peekDeviceHandle :: Ptr DeviceHandle -> IO (DevicePtr a)
peekDeviceHandle !p = do
  CULLong (W64# w#) <- peek p
  return $! DevicePtr (Ptr (int2Addr# (word2Int# w#)))

-- Use a device pointer as an opaque handle type
--
{-# INLINE useDeviceHandle #-}
useDeviceHandle :: DevicePtr a -> DeviceHandle
useDeviceHandle (DevicePtr (Ptr addr#)) =
  CULLong (W64# (int2Word# (addr2Int# addr#)))

