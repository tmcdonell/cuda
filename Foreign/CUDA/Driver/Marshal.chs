{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# OPTIONS_HADDOCK prune #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Marshal
-- Copyright : (c) [2009..2012] Trevor L. McDonell
-- License   : BSD
--
-- Memory management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Marshal (

  -- * Host Allocation
  AllocFlag(..),
  mallocHostArray, freeHost, registerArray, unregisterArray,

  -- * Device Allocation
  mallocArray, allocaArray, free,

  -- * Marshalling
  peekArray, peekArrayAsync, peekListArray,
  pokeArray, pokeArrayAsync, pokeListArray,
  copyArrayAsync,
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
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream               (Stream(..))
import Foreign.CUDA.Driver.Context              (Context(..))
import Foreign.CUDA.Internal.C2HS

-- System
import Data.Int
import Data.Maybe
import Unsafe.Coerce
import Control.Applicative
import Control.Exception

import Foreign.C
import Foreign.Ptr
import Foreign.Storable
import qualified Foreign.Marshal                as F

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
    with prefix="CU_MEMHOSTALLOC_OPTION" deriving (Eq, Show) #}

-- |
-- Allocate a section of linear memory on the host which is page-locked and
-- directly accessible from the device. The storage is sufficient to hold the
-- given number of elements of a storable type.
--
-- Note that since the amount of pageable memory is thusly reduced, overall
-- system performance may suffer. This is best used sparingly to allocate
-- staging areas for data exchange.
--
{-# INLINEABLE mallocHostArray #-}
mallocHostArray :: Storable a => [AllocFlag] -> Int -> IO (HostPtr a)
mallocHostArray !flags = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (HostPtr a')
    doMalloc x !n = resultIfOk =<< cuMemHostAlloc (n * sizeOf x) flags

{-# INLINE cuMemHostAlloc #-}
{# fun unsafe cuMemHostAlloc
  { alloca'-        `HostPtr a'   peekHP*
  ,                 `Int'
  , combineBitMasks `[AllocFlag]'         } -> `Status' cToEnum #}
  where
    alloca' !f = F.alloca $ \ !p -> poke p nullPtr >> f (castPtr p)
    peekHP !p  = HostPtr . castPtr <$> peek p


-- |
-- Free a section of page-locked host memory
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
-- This function is not yet implemented on Mac OS X. Requires cuda-4.0.
--
{-# INLINEABLE registerArray #-}
registerArray :: Storable a => [AllocFlag] -> Int -> Ptr a -> IO (HostPtr a)
#if CUDA_VERSION < 4000
registerArray _ _ _     = requireSDK 4.0 "registerArray"
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
-- This function is not yet implemented on Mac OS X. Requires cuda-4.0.
--
{-# INLINEABLE unregisterArray #-}
unregisterArray :: HostPtr a -> IO (Ptr a)
#if CUDA_VERSION < 4000
unregisterArray _           = requireSDK 4.0 "unregisterArray"
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
-- synchronisation point using 'sync' as part of the computation.
--
{-# INLINEABLE allocaArray #-}
allocaArray :: Storable a => Int -> (DevicePtr a -> IO b) -> IO b
allocaArray !n = bracket (mallocArray n) free


-- |
-- Release a section of device memory
--
{-# INLINEABLE free #-}
free :: DevicePtr a -> IO ()
free !dp = nothingIfOk =<< cuMemFree dp

{-# INLINE cuMemFree #-}
{# fun unsafe cuMemFree
  { useDeviceHandle `DevicePtr a' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Marshalling
--------------------------------------------------------------------------------

-- |
-- Copy a number of elements from the device to host memory. This is a
-- synchronous operation
--
{-# INLINEABLE peekArray #-}
peekArray :: Storable a => Int -> DevicePtr a -> Ptr a -> IO ()
peekArray !n !dptr !hptr = doPeek undefined dptr
  where
    doPeek :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPeek x _ = nothingIfOk =<< cuMemcpyDtoH hptr dptr (n * sizeOf x)

{-# INLINE cuMemcpyDtoH #-}
{# fun unsafe cuMemcpyDtoH
  { castPtr         `Ptr a'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'         } -> `Status' cToEnum #}


-- |
-- Copy memory from the device asynchronously, possibly associated with a
-- particular stream. The destination host memory must be page-locked.
--
{-# INLINEABLE peekArrayAsync #-}
peekArrayAsync :: Storable a => Int -> DevicePtr a -> HostPtr a -> Maybe Stream -> IO ()
peekArrayAsync !n !dptr !hptr !mst = doPeek undefined dptr
  where
    doPeek :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPeek x _ = nothingIfOk =<< cuMemcpyDtoHAsync hptr dptr (n * sizeOf x) (fromMaybe (Stream nullPtr) mst)

{-# INLINE cuMemcpyDtoHAsync #-}
{# fun unsafe cuMemcpyDtoHAsync
  { useHP           `HostPtr a'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}
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


-- |
-- Copy a number of elements onto the device. This is a synchronous operation
--
{-# INLINEABLE pokeArray #-}
pokeArray :: Storable a => Int -> Ptr a -> DevicePtr a -> IO ()
pokeArray !n !hptr !dptr = doPoke undefined dptr
  where
    doPoke :: Storable a' => a' -> DevicePtr a' -> IO ()
    doPoke x _ = nothingIfOk =<< cuMemcpyHtoD dptr hptr (n * sizeOf x)

{-# INLINE cuMemcpyHtoD #-}
{# fun unsafe cuMemcpyHtoD
  { useDeviceHandle `DevicePtr a'
  , castPtr         `Ptr a'
  ,                 `Int'         } -> `Status' cToEnum #}


-- |
-- Copy memory onto the device asynchronously, possibly associated with a
-- particular stream. The source host memory must be page-locked.
--
{-# INLINEABLE pokeArrayAsync #-}
pokeArrayAsync :: Storable a => Int -> HostPtr a -> DevicePtr a -> Maybe Stream -> IO ()
pokeArrayAsync !n !hptr !dptr !mst = dopoke undefined dptr
  where
    dopoke :: Storable a' => a' -> DevicePtr a' -> IO ()
    dopoke x _ = nothingIfOk =<< cuMemcpyHtoDAsync dptr hptr (n * sizeOf x) (fromMaybe (Stream nullPtr) mst)

{-# INLINE cuMemcpyHtoDAsync #-}
{# fun unsafe cuMemcpyHtoDAsync
  { useDeviceHandle `DevicePtr a'
  , useHP           `HostPtr a'
  ,                 `Int'
  , useStream       `Stream'      } -> `Status' cToEnum #}
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


-- |
-- Copy the given number of elements from the first device array (source) to the
-- second (destination). The copied areas may not overlap. This operation is
-- asynchronous with respect to the host, but will never overlap with kernel
-- execution.
--
{-# INLINEABLE copyArrayAsync #-}
copyArrayAsync :: Storable a => Int -> DevicePtr a -> DevicePtr a -> IO ()
copyArrayAsync !n = docopy undefined
  where
    docopy :: Storable a' => a' -> DevicePtr a' -> DevicePtr a' -> IO ()
    docopy x src dst = nothingIfOk =<< cuMemcpyDtoD dst src (n * sizeOf x)

{-# INLINE cuMemcpyDtoD #-}
{# fun unsafe cuMemcpyDtoD
  { useDeviceHandle `DevicePtr a'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'         } -> `Status' cToEnum #}


-- |
-- Copies an array from device memory in one context to device memory in another
-- context. Note that this function is asynchronous with respect to the host,
-- but serialised with respect to all pending and future asynchronous work in
-- the source and destination contexts. To avoid this synchronisation, use
-- 'copyArrayPeerAsync' instead.
--
{-# INLINEABLE copyArrayPeer #-}
copyArrayPeer :: Storable a
              => Int                            -- ^ number of array elements
              -> DevicePtr a -> Context         -- ^ source array and context
              -> DevicePtr a -> Context         -- ^ destination array and context
              -> IO ()
#if CUDA_VERSION < 4000
copyArrayPeer _ _ _ _ _                    = requireSDK 4.0 "copyArrayPeer"
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
{-# INLINEABLE copyArrayPeerAsync #-}
copyArrayPeerAsync :: Storable a
                   => Int                       -- ^ number of array elements
                   -> DevicePtr a -> Context    -- ^ source array and context
                   -> DevicePtr a -> Context    -- ^ destination array and device context
                   -> Maybe Stream              -- ^ stream to associate with
                   -> IO ()
#if CUDA_VERSION < 4000
copyArrayPeerAsync _ _ _ _ _ _                      = requireSDK 4.0 "copyArrayPeerAsync"
#else
copyArrayPeerAsync !n !src !srcCtx !dst !dstCtx !st = go undefined src dst
  where
    go :: Storable b => b -> DevicePtr b -> DevicePtr b -> IO ()
    go x _ _ = nothingIfOk =<< cuMemcpyPeerAsync dst dstCtx src srcCtx (n * sizeOf x) stream
    stream   = fromMaybe (Stream nullPtr) st

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
-- associated with a stream. Requires cuda-3.2.
--
{-# INLINEABLE memsetAsync #-}
memsetAsync :: Storable a => DevicePtr a -> Int -> a -> Maybe Stream -> IO ()
#if CUDA_VERSION < 3020
memsetAsync _ _ _ _            = requireSDK 3.2 "memsetAsync"
#else
memsetAsync !dptr !n !val !mst = case sizeOf val of
    1 -> nothingIfOk =<< cuMemsetD8Async  dptr val n stream
    2 -> nothingIfOk =<< cuMemsetD16Async dptr val n stream
    4 -> nothingIfOk =<< cuMemsetD32Async dptr val n stream
    _ -> cudaError "can only memset 8-, 16-, and 32-bit values"
    where
      stream = fromMaybe (Stream nullPtr) mst

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
-- Return the base address and allocation size of the given device pointer
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
-- current context (bytes)
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
peekDeviceHandle !p = DevicePtr . intPtrToPtr . fromIntegral <$> peek p

-- Use a device pointer as an opaque handle type
--
{-# INLINE useDeviceHandle #-}
useDeviceHandle :: DevicePtr a -> DeviceHandle
useDeviceHandle = fromIntegral . ptrToIntPtr . useDevicePtr

