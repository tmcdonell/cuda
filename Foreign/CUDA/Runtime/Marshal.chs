{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Marshal
-- Copyright : [2009..2014] Trevor L. McDonell
-- License   : BSD
--
-- Memory management for CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Marshal (

  -- * Host Allocation
  AllocFlag(..),
  mallocHostArray, freeHost,

  -- * Device Allocation
  mallocArray, allocaArray, free,

  -- * Unified Memory Allocation
  AttachFlag(..),
  mallocManagedArray,

  -- * Marshalling
  peekArray, peekArrayAsync, peekArray2D, peekArray2DAsync, peekListArray,
  pokeArray, pokeArrayAsync, pokeArray2D, pokeArray2DAsync, pokeListArray,
  copyArray, copyArrayAsync, copyArray2D, copyArray2DAsync,

  -- * Combined Allocation and Marshalling
  newListArray,  newListArrayLen,
  withListArray, withListArrayLen,

  -- * Utility
  memset

) where

#include "cbits/stubs.h"
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Runtime.Stream
import Foreign.CUDA.Internal.C2HS

-- System
import Data.Int
import Data.Maybe
import Control.Exception

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

#if CUDART_VERSION >= 6000
#c
typedef enum cudaMemAttachFlags_option_enum {
    CUDA_MEM_ATTACH_OPTION_GLOBAL = cudaMemAttachGlobal,
    CUDA_MEM_ATTACH_OPTION_HOST   = cudaMemAttachHost,
    CUDA_MEM_ATTACH_OPTION_SINGLE = cudaMemAttachSingle
} cudaMemAttachFlags_option;
#endc
#endif


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
-- accelerates calls to functions such as 'peekArrayAsync' and 'pokeArrayAsync'
-- that refer to page-locked memory.
--
-- Note that since the amount of pageable memory is thusly reduced, overall
-- system performance may suffer. This is best used sparingly to allocate
-- staging areas for data exchange
--
{-# INLINEABLE mallocHostArray #-}
mallocHostArray :: Storable a => [AllocFlag] -> Int -> IO (HostPtr a)
mallocHostArray !flags = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (HostPtr a')
    doMalloc x !n = resultIfOk =<< cudaHostAlloc (fromIntegral n * fromIntegral (sizeOf x)) flags

{-# INLINE cudaHostAlloc #-}
{# fun unsafe cudaHostAlloc
  { alloca'-        `HostPtr a' hptr*
  , cIntConv        `Int64'
  , combineBitMasks `[AllocFlag]'     } -> `Status' cToEnum #}
  where
    alloca' !f = F.alloca $ \ !p -> poke p nullPtr >> f (castPtr p)
    hptr !p    = (HostPtr . castPtr) `fmap` peek p


-- |
-- Free page-locked host memory previously allocated with 'mallecHost'
--
{-# INLINEABLE freeHost #-}
freeHost :: HostPtr a -> IO ()
freeHost !p = nothingIfOk =<< cudaFreeHost p

{-# INLINE cudaFreeHost #-}
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
{-# INLINEABLE mallocArray #-}
mallocArray :: Storable a => Int -> IO (DevicePtr a)
mallocArray = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (DevicePtr a')
    doMalloc x !n = resultIfOk =<< cudaMalloc (fromIntegral n * fromIntegral (sizeOf x))

{-# INLINE cudaMalloc #-}
{# fun unsafe cudaMalloc
  { alloca'- `DevicePtr a' dptr*
  , cIntConv `Int64'             } -> `Status' cToEnum #}
  where
    -- C-> Haskell doesn't like qualified imports in marshaller specifications
    alloca' !f = F.alloca $ \ !p -> poke p nullPtr >> f (castPtr p)
    dptr !p    = (castDevPtr . DevicePtr) `fmap` peek p


-- |
-- Execute a computation, passing a pointer to a temporarily allocated block of
-- memory sufficient to hold the given number of elements of storable type. The
-- memory is freed when the computation terminates (normally or via an
-- exception), so the pointer must not be used after this.
--
-- Note that kernel launches can be asynchronous, so you may need to add a
-- synchronisation point at the end of the computation.
--
{-# INLINEABLE allocaArray #-}
allocaArray :: Storable a => Int -> (DevicePtr a -> IO b) -> IO b
allocaArray n = bracket (mallocArray n) free


-- |
-- Free previously allocated memory on the device
--
{-# INLINEABLE free #-}
free :: DevicePtr a -> IO ()
free !p = nothingIfOk =<< cudaFree p

{-# INLINE cudaFree #-}
{# fun unsafe cudaFree
  { dptr `DevicePtr a' } -> `Status' cToEnum #}
  where
    dptr = useDevicePtr . castDevPtr


--------------------------------------------------------------------------------
-- Unified memory allocation
--------------------------------------------------------------------------------

-- |
-- Options for unified memory allocations
--
#if CUDART_VERSION >= 6000
{# enum cudaMemAttachFlags_option as AttachFlag
    { underscoreToCase }
    with prefix="CUDA_MEM_ATTACH_OPTION" deriving (Eq, Show) #}
#else
data AttachFlag
#endif

-- |
-- Allocates memory that will be automatically managed by the Unified Memory
-- system
--
{-# INLINEABLE mallocManagedArray #-}
mallocManagedArray :: Storable a => [AttachFlag] -> Int -> IO (DevicePtr a)
#if CUDART_VERSION < 6000
mallocManagedArray _ _    = requireSDK 6.0 "mallocManagedArray"
#else
mallocManagedArray !flags = doMalloc undefined
  where
    doMalloc :: Storable a' => a' -> Int -> IO (DevicePtr a')
    doMalloc x !n = resultIfOk =<< cudaMallocManaged (fromIntegral n * fromIntegral (sizeOf x)) flags

{-# INLINE cudaMallocManaged #-}
{# fun unsafe cudaMallocManaged
  { alloca'-        `DevicePtr a' dptr*
  , cIntConv        `Int64'
  , combineBitMasks `[AttachFlag]'      } -> `Status' cToEnum #}
  where
    alloca' !f = F.alloca $ \ !p -> poke p nullPtr >> f (castPtr p)
    dptr !p    = (castDevPtr . DevicePtr) `fmap` peek p
#endif


--------------------------------------------------------------------------------
-- Marshalling
--------------------------------------------------------------------------------

-- |
-- Copy a number of elements from the device to host memory. This is a
-- synchronous operation.
--
{-# INLINEABLE peekArray #-}
peekArray :: Storable a => Int -> DevicePtr a -> Ptr a -> IO ()
peekArray !n !dptr !hptr = memcpy hptr (useDevicePtr dptr) n DeviceToHost


-- |
-- Copy memory from the device asynchronously, possibly associated with a
-- particular stream. The destination memory must be page locked.
--
{-# INLINEABLE peekArrayAsync #-}
peekArrayAsync :: Storable a => Int -> DevicePtr a -> HostPtr a -> Maybe Stream -> IO ()
peekArrayAsync !n !dptr !hptr !mst =
  memcpyAsync (useHostPtr hptr) (useDevicePtr dptr) n DeviceToHost mst


-- |
-- Copy a 2D memory area from the device to the host. This is a synchronous
-- operation.
--
{-# INLINEABLE peekArray2D #-}
peekArray2D
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> DevicePtr a              -- ^ source array
    -> Int                      -- ^ source array width
    -> Ptr a                    -- ^ destination array
    -> Int                      -- ^ destination array width
    -> IO ()
peekArray2D !w !h !dptr !dw !hptr !hw =
  memcpy2D hptr hw (useDevicePtr dptr) dw w h DeviceToHost


-- |
-- Copy a 2D memory area from the device to the host asynchronously, possibly
-- associated with a particular stream. The destination array must be page
-- locked.
--
{-# INLINEABLE peekArray2DAsync #-}
peekArray2DAsync
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> DevicePtr a              -- ^ source array
    -> Int                      -- ^ source array width
    -> HostPtr a                -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Maybe Stream
    -> IO ()
peekArray2DAsync !w !h !dptr !dw !hptr !hw !mst =
  memcpy2DAsync (useHostPtr hptr) hw (useDevicePtr dptr) dw w h DeviceToHost mst


-- |
-- Copy a number of elements from the device into a new Haskell list. Note that
-- this requires two memory copies: firstly from the device into a heap
-- allocated array, and from there marshalled into a list
--
{-# INLINEABLE peekListArray #-}
peekListArray :: Storable a => Int -> DevicePtr a -> IO [a]
peekListArray !n !dptr =
  F.allocaArray n $ \p -> do
    peekArray   n dptr p
    F.peekArray n p


-- |
-- Copy a number of elements onto the device. This is a synchronous operation.
--
{-# INLINEABLE pokeArray #-}
pokeArray :: Storable a => Int -> Ptr a -> DevicePtr a -> IO ()
pokeArray !n !hptr !dptr = memcpy (useDevicePtr dptr) hptr n HostToDevice


-- |
-- Copy memory onto the device asynchronously, possibly associated with a
-- particular stream. The source memory must be page-locked.
--
{-# INLINEABLE pokeArrayAsync #-}
pokeArrayAsync :: Storable a => Int -> HostPtr a -> DevicePtr a -> Maybe Stream -> IO ()
pokeArrayAsync !n !hptr !dptr !mst =
  memcpyAsync (useDevicePtr dptr) (useHostPtr hptr) n HostToDevice mst


-- |
-- Copy a 2D memory area onto the device. This is a synchronous operation.
--
{-# INLINEABLE pokeArray2D #-}
pokeArray2D
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> Ptr a                    -- ^ source array
    -> Int                      -- ^ source array width
    -> DevicePtr a              -- ^ destination array
    -> Int                      -- ^ destination array width
    -> IO ()
pokeArray2D !w !h !hptr !dw !dptr !hw =
  memcpy2D (useDevicePtr dptr) dw hptr hw w h HostToDevice


-- |
-- Copy a 2D memory area onto the device asynchronously, possibly associated
-- with a particular stream. The source array must be page locked.
--
{-# INLINEABLE pokeArray2DAsync #-}
pokeArray2DAsync
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> HostPtr a                -- ^ source array
    -> Int                      -- ^ source array width
    -> DevicePtr a              -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Maybe Stream
    -> IO ()
pokeArray2DAsync !w !h !hptr !hw !dptr !dw !mst =
  memcpy2DAsync (useDevicePtr dptr) dw (useHostPtr hptr) hw w h HostToDevice mst

-- |
-- Write a list of storable elements into a device array. The array must be
-- sufficiently large to hold the entire list. This requires two marshalling
-- operations
--
{-# INLINEABLE pokeListArray #-}
pokeListArray :: Storable a => [a] -> DevicePtr a -> IO ()
pokeListArray !xs !dptr = F.withArrayLen xs $ \len p -> pokeArray len p dptr


-- |
-- Copy the given number of elements from the first device array (source) to the
-- second (destination). The copied areas may not overlap. This operation is
-- asynchronous with respect to host, but will not overlap other device
-- operations.
--
{-# INLINEABLE copyArray #-}
copyArray :: Storable a => Int -> DevicePtr a -> DevicePtr a -> IO ()
copyArray !n !src !dst = memcpy (useDevicePtr dst) (useDevicePtr src) n DeviceToDevice


-- |
-- Copy the given number of elements from the first device array (source) to the
-- second (destination). The copied areas may not overlap. This operation is
-- asynchronous with respect to the host, and may be associated with a
-- particular stream.
--
{-# INLINEABLE copyArrayAsync #-}
copyArrayAsync :: Storable a => Int -> DevicePtr a -> DevicePtr a -> Maybe Stream -> IO ()
copyArrayAsync !n !src !dst !mst =
  memcpyAsync (useDevicePtr dst) (useDevicePtr src) n DeviceToDevice mst


-- |
-- Copy a 2D memory area from the first device array (source) to the second
-- (destination). The copied areas may not overlap. This operation is
-- asynchronous with respect to the host, but will not overlap other device
-- operations.
--
{-# INLINEABLE copyArray2D #-}
copyArray2D
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> DevicePtr a              -- ^ source array
    -> Int                      -- ^ source array width
    -> DevicePtr a              -- ^ destination array
    -> Int                      -- ^ destination array width
    -> IO ()
copyArray2D !w !h !src !sw !dst !dw =
  memcpy2D (useDevicePtr dst) dw (useDevicePtr src) sw w h DeviceToDevice


-- |
-- Copy a 2D memory area from the first device array (source) to the second
-- device array (destination). The copied areas may not overlay. This operation
-- is asynchronous with respect to the host, and may be associated with a
-- particular stream.
--
{-# INLINEABLE copyArray2DAsync #-}
copyArray2DAsync
    :: Storable a
    => Int                      -- ^ width to copy (elements)
    -> Int                      -- ^ height to copy (elements)
    -> DevicePtr a              -- ^ source array
    -> Int                      -- ^ source array width
    -> DevicePtr a              -- ^ destination array
    -> Int                      -- ^ destination array width
    -> Maybe Stream
    -> IO ()
copyArray2DAsync !w !h !src !sw !dst !dw !mst =
  memcpy2DAsync (useDevicePtr dst) dw (useDevicePtr src) sw w h DeviceToDevice mst


--
-- Memory copy kind
--
{# enum cudaMemcpyKind as CopyDirection {}
    with prefix="cudaMemcpy" deriving (Eq, Show) #}

-- |
-- Copy data between host and device. This is a synchronous operation.
--
{-# INLINEABLE memcpy #-}
memcpy :: Storable a
       => Ptr a                 -- ^ destination
       -> Ptr a                 -- ^ source
       -> Int                   -- ^ number of elements
       -> CopyDirection
       -> IO ()
memcpy !dst !src !n !dir = doMemcpy undefined dst
  where
    doMemcpy :: Storable a' => a' -> Ptr a' -> IO ()
    doMemcpy x _ =
      nothingIfOk =<< cudaMemcpy dst src (fromIntegral n * fromIntegral (sizeOf x)) dir

{-# INLINE cudaMemcpy #-}
{# fun cudaMemcpy
  { castPtr   `Ptr a'
  , castPtr   `Ptr a'
  , cIntConv  `Int64'
  , cFromEnum `CopyDirection' } -> `Status' cToEnum #}


-- |
-- Copy data between the host and device asynchronously, possibly associated
-- with a particular stream. The host-side memory must be page-locked (allocated
-- with 'mallocHostArray').
--
{-# INLINEABLE memcpyAsync #-}
memcpyAsync :: Storable a
            => Ptr a            -- ^ destination
            -> Ptr a            -- ^ source
            -> Int              -- ^ number of elements
            -> CopyDirection
            -> Maybe Stream
            -> IO ()
memcpyAsync !dst !src !n !kind !mst = doMemcpy undefined dst
  where
    doMemcpy :: Storable a' => a' -> Ptr a' -> IO ()
    doMemcpy x _ =
      let bytes = fromIntegral n * fromIntegral (sizeOf x) in
      nothingIfOk =<< cudaMemcpyAsync dst src bytes kind (fromMaybe defaultStream mst)

{-# INLINE cudaMemcpyAsync #-}
{# fun cudaMemcpyAsync
  { castPtr   `Ptr a'
  , castPtr   `Ptr a'
  , cIntConv  `Int64'
  , cFromEnum `CopyDirection'
  , useStream `Stream'        } -> `Status' cToEnum #}


-- |
-- Copy a 2D memory area between the host and device. This is a synchronous
-- operation.
--
{-# INLINEABLE memcpy2D #-}
memcpy2D :: Storable a
         => Ptr a               -- ^ destination
         -> Int                 -- ^ width of destination array
         -> Ptr a               -- ^ source
         -> Int                 -- ^ width of source array
         -> Int                 -- ^ width to copy
         -> Int                 -- ^ height to copy
         -> CopyDirection
         -> IO ()
memcpy2D !dst !dw !src !sw !w !h !kind = doCopy undefined dst
  where
    doCopy :: Storable a' => a' -> Ptr a' -> IO ()
    doCopy x _ =
      let bytes = fromIntegral (sizeOf x)
          dw'   = fromIntegral dw * bytes
          sw'   = fromIntegral sw * bytes
          w'    = fromIntegral w  * bytes
          h'    = fromIntegral h
      in
      nothingIfOk =<< cudaMemcpy2D dst dw' src sw' w' h' kind

{-# INLINE cudaMemcpy2D #-}
{# fun cudaMemcpy2D
  { castPtr   `Ptr a'
  ,           `Int64'
  , castPtr   `Ptr a'
  ,           `Int64'
  ,           `Int64'
  ,           `Int64'
  , cFromEnum `CopyDirection'
  }
  -> `Status' cToEnum #}


-- |
-- Copy a 2D memory area between the host and device asynchronously, possibly
-- associated with a particular stream. The host-side memory must be
-- page-locked.
--
{-# INLINEABLE memcpy2DAsync #-}
memcpy2DAsync :: Storable a
              => Ptr a          -- ^ destination
              -> Int            -- ^ width of destination array
              -> Ptr a          -- ^ source
              -> Int            -- ^ width of source array
              -> Int            -- ^ width to copy
              -> Int            -- ^ height to copy
              -> CopyDirection
              -> Maybe Stream
              -> IO ()
memcpy2DAsync !dst !dw !src !sw !w !h !kind !mst = doCopy undefined dst
  where
    doCopy :: Storable a' => a' -> Ptr a' -> IO ()
    doCopy x _ =
      let bytes = fromIntegral (sizeOf x)
          dw'   = fromIntegral dw * bytes
          sw'   = fromIntegral sw * bytes
          w'    = fromIntegral w  * bytes
          h'    = fromIntegral h
          st    = fromMaybe defaultStream mst
      in
      nothingIfOk =<< cudaMemcpy2DAsync dst dw' src sw' w' h' kind st

{-# INLINE cudaMemcpy2DAsync #-}
{# fun cudaMemcpy2DAsync
  { castPtr   `Ptr a'
  ,           `Int64'
  , castPtr   `Ptr a'
  ,           `Int64'
  ,           `Int64'
  ,           `Int64'
  , cFromEnum `CopyDirection'
  , useStream `Stream'
  }
  -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Combined Allocation and Marshalling
--------------------------------------------------------------------------------

-- |
-- Write a list of storable elements into a newly allocated device array,
-- returning the device pointer together with the number of elements that were
-- written. Note that this requires two copy operations: firstly from a Haskell
-- list into a heap-allocated array, and from there into device memory. The
-- array should be 'free'd when no longer required.
--
{-# INLINEABLE newListArrayLen #-}
newListArrayLen :: Storable a => [a] -> IO (DevicePtr a, Int)
newListArrayLen !xs =
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
newListArray !xs = fst `fmap` newListArrayLen xs


-- |
-- Temporarily store a list of elements into a newly allocated device array. An
-- IO action is applied to the array, the result of which is returned. Similar
-- to 'newListArray', this requires two marshalling operations of the data.
--
-- As with 'allocaArray', the memory is freed once the action completes, so you
-- should not return the pointer from the action, and be sure that any
-- asynchronous operations (such as kernel execution) have completed.
--
{-# INLINEABLE withListArray #-}
withListArray :: Storable a => [a] -> (DevicePtr a -> IO b) -> IO b
withListArray !xs = withListArrayLen xs . const


-- |
-- A variant of 'withListArray' which also supplies the number of elements in
-- the array to the applied function
--
{-# INLINEABLE withListArrayLen #-}
withListArrayLen :: Storable a => [a] -> (Int -> DevicePtr a -> IO b) -> IO b
withListArrayLen !xs !f =
  bracket (newListArrayLen xs) (free . fst) (uncurry . flip $ f)
--
-- XXX: Will this attempt to double-free the device array on error (together
-- with newListArrayLen)?
--


--------------------------------------------------------------------------------
-- Utility
--------------------------------------------------------------------------------

-- |
-- Initialise device memory to a given 8-bit value
--
{-# INLINEABLE memset #-}
memset :: DevicePtr a                   -- ^ The device memory
       -> Int64                         -- ^ Number of bytes
       -> Int8                          -- ^ Value to set for each byte
       -> IO ()
memset !dptr !bytes !symbol = nothingIfOk =<< cudaMemset dptr symbol bytes

{-# INLINE cudaMemset #-}
{# fun unsafe cudaMemset
  { dptr     `DevicePtr a'
  , cIntConv `Int8'
  , cIntConv `Int64'       } -> `Status' cToEnum #}
  where
    dptr = useDevicePtr . castDevPtr

