{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Marshal
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Memory management for CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Marshal
  (
    -- * Host Allocation
    AllocFlag(..),
    mallocHostArray, freeHost,

    -- * Device Allocation
    mallocArray, allocaArray, free,

    -- * Marshalling
    peekArray, peekArrayAsync, peekListArray,
    pokeArray, pokeArrayAsync, pokeListArray,
    copyArray, copyArrayAsync,

    -- * Combined Allocation and Marshalling
    newListArray, withListArray, withListArrayLen,

    -- * Utility
    memset
  )
  where

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Ptr
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Runtime.Stream
import Foreign.CUDA.Internal.C2HS

-- System
import Data.Int
import Control.Exception.Extensible

import Foreign.C
import Foreign.Ptr
import Foreign.Storable
import qualified Foreign.Marshal as F

#c
typedef enum cudaMemHostAlloc_option_enum {
//  CUDA_MEMHOSTALLOC_OPTION_DEFAULT        = cudaHostAllocDefault,
    CUDA_MEMHOSTALLOC_OPTION_DEVICE_MAPPED  = cudaHostAllocMapped,
    CUDA_MEMHOSTALLOC_OPTION_PORTABLE       = cudaHostAllocPortable,
    CUDA_MEMHOSTALLOC_OPTION_WRITE_COMBINED = cudaHostAllocWriteCombined
} cudaMemHostAlloc_option;
#endc


--------------------------------------------------------------------------------
-- Host Allocation
--------------------------------------------------------------------------------

-- |
-- Options for host allocation
--
{# enum cudaMemHostAlloc_option as AllocFlag
    { underscoreToCase }
    with prefix="CUDA_MEMHOSTALLOC_OPTION" deriving (Eq, Show) #}


-- |
-- Allocate a section of linear memory on the host which is page-locked and
-- directly accessible from the device. The storage is sufficient to hold the
-- given number of elements of a storable type. The runtime system automatically
-- accelerates calls to functions such as 'memcpy' to page-locked memory.
--
-- Note that since the amount of pageable memory is thusly reduced, overall
-- system performance may suffer. This is best used sparingly to allocate
-- staging areas for data exchange
--
mallocHostArray :: Storable a => [AllocFlag] -> Int -> IO (HostPtr a)
mallocHostArray flags = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (HostPtr a')
    doMalloc x n = resultIfOk =<< cudaHostAlloc (fromIntegral n * fromIntegral (sizeOf x)) flags

{# fun unsafe cudaHostAlloc
  { alloca'-        `HostPtr a' hptr*
  , cIntConv        `Int64'
  , combineBitMasks `[AllocFlag]'     } -> `Status' cToEnum #}
  where
    alloca' = F.alloca
    hptr p  = (HostPtr . castPtr) `fmap` peek p


-- |
-- Free page-locked host memory previously allocated with 'mallecHost'
--
freeHost :: HostPtr a -> IO ()
freeHost p = nothingIfOk =<< cudaFreeHost p

{# fun unsafe cudaFreeHost
  { hptr `HostPtr a' } -> `Status' cToEnum #}
  where hptr = castPtr . useHostPtr


--------------------------------------------------------------------------------
-- Device Allocation
--------------------------------------------------------------------------------

-- |
-- Allocate a section of linear memory on the device, and return a reference to
-- it. The memory is sufficient to hold the given number of elements of storable
-- type. It is suitable aligned, and not cleared.
--
mallocArray :: Storable a => Int -> IO (DevicePtr a)
mallocArray = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (DevicePtr a')
    doMalloc x n = resultIfOk =<< cudaMalloc (fromIntegral n * fromIntegral (sizeOf x))

{# fun unsafe cudaMalloc
  { alloca'- `DevicePtr a' dptr*
  , cIntConv `Int64'             } -> `Status' cToEnum #}
  where
    -- C-> Haskell doesn't like qualified imports in marshaller specifications
    alloca' = F.alloca
    dptr p  = (castDevPtr . DevicePtr) `fmap` peek p


-- |
-- Execute a computation, passing a pointer to a temporarily allocated block of
-- memory sufficient to hold the given number of elements of storable type. The
-- memory is freed when the computation terminates (normally or via an
-- exception), so the pointer must not be used after this.
--
-- Note that kernel launches can be asynchronous, so you may need to add a
-- synchronisation point at the end of the computation.
--
allocaArray :: Storable a => Int -> (DevicePtr a -> IO b) -> IO b
allocaArray n = bracket (mallocArray n) free


-- |
-- Free previously allocated memory on the device
--
free :: DevicePtr a -> IO ()
free p = nothingIfOk =<< cudaFree p

{# fun unsafe cudaFree
  { dptr `DevicePtr a' } -> `Status' cToEnum #}
  where
    dptr = useDevicePtr . castDevPtr


--------------------------------------------------------------------------------
-- Marshalling
--------------------------------------------------------------------------------

-- |
-- Copy a number of elements from the device to host memory. This is a
-- synchronous operation.
--
peekArray :: Storable a => Int -> DevicePtr a -> Ptr a -> IO ()
peekArray n dptr hptr = memcpy hptr (useDevicePtr dptr) n DeviceToHost


-- |
-- Copy memory from the device asynchronously, possibly associated with a
-- particular stream. The destination memory must be page locked.
--
peekArrayAsync :: Storable a => Int -> DevicePtr a -> HostPtr a -> Maybe Stream -> IO ()
peekArrayAsync n dptr hptr mst =
  memcpyAsync (useHostPtr hptr) (useDevicePtr dptr) n DeviceToHost mst


-- |
-- Copy a number of elements from the device into a new Haskell list. Note that
-- this requires two memory copies: firstly from the device into a heap
-- allocated array, and from there marshalled into a list
--
peekListArray :: Storable a => Int -> DevicePtr a -> IO [a]
peekListArray n dptr =
  F.allocaArray n $ \p -> do
    peekArray   n dptr p
    F.peekArray n p


-- |
-- Copy a number of elements onto the device. This is a synchronous operation.
--
pokeArray :: Storable a => Int -> Ptr a -> DevicePtr a -> IO ()
pokeArray n hptr dptr = memcpy (useDevicePtr dptr) hptr n HostToDevice


-- |
-- Copy memory onto the device asynchronously, possibly associated with a
-- particular stream. The source memory must be page-locked.
--
pokeArrayAsync :: Storable a => Int -> HostPtr a -> DevicePtr a -> Maybe Stream -> IO ()
pokeArrayAsync n hptr dptr mst =
  memcpyAsync (useDevicePtr dptr) (useHostPtr hptr) n HostToDevice mst


-- |
-- Write a list of storable elements into a device array. The array must be
-- sufficiently large to hold the entire list. This requires two marshalling
-- operations
--
pokeListArray :: Storable a => [a] -> DevicePtr a -> IO ()
pokeListArray xs dptr =
  bracket (mallocHostArray [WriteCombined] len) freeHost $ \(HostPtr h_xs) -> do
    F.pokeArray h_xs xs
    pokeArray len h_xs dptr
  where
    len = length xs


-- |
-- Copy the given number of elements from the first device array (source) to the
-- second (destination). The copied areas may not overlap. This is a synchronous
-- operation.
--
copyArray :: Storable a => Int -> DevicePtr a -> DevicePtr a -> IO ()
copyArray n src dst = memcpy (useDevicePtr dst) (useDevicePtr src) n DeviceToDevice


-- |
-- Copy the given number of elements from the first device array (source) to the
-- second (destination). The copied areas may not overlap. This operation is
-- asynchronous with respect to host, but will never overlap with kernel
-- execution.
--
copyArrayAsync :: Storable a => Int -> DevicePtr a -> DevicePtr a -> Maybe Stream -> IO ()
copyArrayAsync n src dst mst =
  memcpyAsync (useDevicePtr dst) (useDevicePtr src) n DeviceToDevice mst


--
-- Memory copy kind
--
{# enum cudaMemcpyKind as CopyDirection {}
    with prefix="cudaMemcpy" deriving (Eq, Show) #}

-- |
-- Copy data between host and device. This is a synchronous operation.
--
memcpy :: Storable a
       => Ptr a                 -- ^ destination
       -> Ptr a                 -- ^ source
       -> Int                   -- ^ number of elements
       -> CopyDirection
       -> IO ()
memcpy dst src n dir = doMemcpy undefined dst
  where
    doMemcpy :: Storable a' => a' -> Ptr a' -> IO ()
    doMemcpy x _ =
      nothingIfOk =<< cudaMemcpy dst src (fromIntegral n * fromIntegral (sizeOf x)) dir

{# fun unsafe cudaMemcpy
  { castPtr   `Ptr a'
  , castPtr   `Ptr a'
  , cIntConv  `Int64'
  , cFromEnum `CopyDirection' } -> `Status' cToEnum #}


-- |
-- Copy data between the host and device asynchronously, possibly associated
-- with a particular stream. The host-side memory must be page-locked (allocated
-- with 'mallocHostArray').
--
memcpyAsync :: Storable a
            => Ptr a            -- ^ destination
            -> Ptr a            -- ^ source
            -> Int              -- ^ number of elements
            -> CopyDirection
            -> Maybe Stream
            -> IO ()
memcpyAsync dst src n kind mst = doMemcpy undefined dst
  where
    doMemcpy :: Storable a' => a' -> Ptr a' -> IO ()
    doMemcpy x _ =
      let bytes = fromIntegral n * fromIntegral (sizeOf x) in
      nothingIfOk =<< case mst of
        Nothing -> cudaMemcpyAsync dst src bytes kind (Stream 0)
        Just st -> cudaMemcpyAsync dst src bytes kind st

{# fun unsafe cudaMemcpyAsync
  { castPtr   `Ptr a'
  , castPtr   `Ptr a'
  , cIntConv  `Int64'
  , cFromEnum `CopyDirection'
  , useStream `Stream'        } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Combined Allocation and Marshalling
--------------------------------------------------------------------------------

-- |
-- Write a list of storable elements into a newly allocated device array. Note
-- that this requires two copy operations: firstly from a Haskell list into a
-- heap-allocated array, and from there into device memory. The array should be
-- 'free'd when no longer required.
--
newListArray :: Storable a => [a] -> IO (DevicePtr a)
newListArray xs =
  bracketOnError (mallocArray len) free $ \d_xs -> do
    pokeListArray xs d_xs
    return d_xs
  where
    len = length xs


-- |
-- Temporarily store a list of elements into a newly allocated device array. An
-- IO action is applied to the array, the result of which is returned. Similar
-- to 'newListArray', this requires two marshalling operations of the data.
--
-- As with 'allocaArray', the memory is freed once the action completes, so you
-- should not return the pointer from the action, and be sure that any
-- asynchronous operations (such as kernel execution) have completed.
--
withListArray :: Storable a => [a] -> (DevicePtr a -> IO b) -> IO b
withListArray xs = withListArrayLen xs . const


-- |
-- A variant of 'withListArray' which also supplies the number of elements in
-- the array to the applied function
--
withListArrayLen :: Storable a => [a] -> (Int -> DevicePtr a -> IO b) -> IO b
withListArrayLen xs f =
  allocaArray len $ \d_xs -> do
    pokeListArray xs d_xs
    f len d_xs
  where
    len = length xs


--------------------------------------------------------------------------------
-- Utility
--------------------------------------------------------------------------------

-- |
-- Initialise device memory to a given 8-bit value
--
memset :: DevicePtr a                   -- ^ The device memory
       -> Int64                         -- ^ Number of bytes
       -> Int8                          -- ^ Value to set for each byte
       -> IO ()
memset dptr bytes symbol = nothingIfOk =<< cudaMemset dptr symbol bytes

{# fun unsafe cudaMemset
  { dptr     `DevicePtr a'
  , cIntConv `Int8'
  , cIntConv `Int64'       } -> `Status' cToEnum #}
  where
    dptr = useDevicePtr . castDevPtr

