{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Marshal
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Memory management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Marshal
  (
    DevicePtr, HostPtr, AllocFlag(..), withHostPtr,
    malloc, free, mallocHost, freeHost,
    peekArray, peekArrayAsync, pokeArray, pokeArrayAsync, memset,
    getDevicePtr
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream               (Stream, withStream)

-- System
import Foreign                           hiding (peekArray, pokeArray, malloc, free)
import Foreign.C
import Control.Monad                            (liftM)
import Unsafe.Coerce

#c
typedef enum CUmemhostalloc_option_enum {
    CU_MEMHOSTALLOC_OPTION_PORTABLE       = CU_MEMHOSTALLOC_PORTABLE,
    CU_MEMHOSTALLOC_OPTION_DEVICE_MAPPED  = CU_MEMHOSTALLOC_DEVICEMAP,
    CU_MEMHOSTALLOC_OPTION_WRITE_COMBINED = CU_MEMHOSTALLOC_WRITECOMBINED
} CUmemhostalloc_option;
#endc

--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A reference to memory allocated on the device
--
newtype DevicePtr a = DevicePtr { useDevicePtr :: {# type CUdeviceptr #}}
  deriving (Show)

instance Storable (DevicePtr a) where
  sizeOf _      = sizeOf    (undefined :: {# type CUdeviceptr #})
  alignment _   = alignment (undefined :: {# type CUdeviceptr #})
  peek p        = DevicePtr `fmap` peek (castPtr p)
  poke p v      = poke (castPtr p) (useDevicePtr v)


-- |
-- A reference to memory on the host that is page-locked and directly-accessible
-- from the device. Since the memory can be accessed directly, it can be read or
-- written at a much higher bandwidth than pageable memory from the traditional
-- malloc.
--
-- The driver automatically accelerates calls to functions such as `memcpy'
-- which reference page-locked memory.
--
newtype HostPtr a = HostPtr (ForeignPtr a)

withHostPtr :: HostPtr a -> (Ptr a -> IO b) -> IO b
withHostPtr (HostPtr hptr) = withForeignPtr hptr

newHostPtr :: IO (HostPtr a)
newHostPtr = HostPtr `fmap` mallocForeignPtrBytes (sizeOf (undefined :: Ptr ()))

-- |
-- Options for host allocation
--
{# enum CUmemhostalloc_option as AllocFlag
    { underscoreToCase }
    with prefix="CU_MEMHOSTALLOC_OPTION" deriving (Eq, Show) #}

--------------------------------------------------------------------------------
-- Host Allocation
--------------------------------------------------------------------------------

-- |
-- Allocate a section of linear memory on the host which is page-locked and
-- directly accessible from the device. Note that since the amount of pageable
-- memory is thusly reduced, overall system performance may suffer. This is best
-- used sparingly to allocate staging areas for data exchange.
--
mallocHost :: [AllocFlag] -> Int -> IO (Either String (HostPtr a))
mallocHost flags bytes =
  newHostPtr >>= \hp -> withHostPtr hp $ \p ->
  cuMemHostAlloc p bytes flags >>= \rv ->
  return $ case nothingIfOk rv of
    Nothing  -> Right hp
    Just err -> Left err

{# fun unsafe cuMemHostAlloc
  { with'* `Ptr a'
  ,        `Int'
  , combineBitMasks `[AllocFlag]' } -> `Status' cToEnum #}
  where with' = with . castPtr


-- |
-- Free a section of page-locked host memory
--
freeHost :: HostPtr a -> IO (Maybe String)
freeHost hp = withHostPtr hp $ \p -> (nothingIfOk `fmap` cuMemFreeHost p)

{# fun unsafe cuMemFreeHost
  { castPtr `Ptr a' } -> `Status' cToEnum #}

--------------------------------------------------------------------------------
-- Device Allocation
--------------------------------------------------------------------------------

-- |
-- Allocate a section of linear memory on the device, and return a reference to
-- it. The memory is suitably aligned for any type, and is not cleared.
--
malloc :: Int -> IO (Either String (DevicePtr a))
malloc bytes = resultIfOk `fmap` cuMemAlloc bytes

{# fun unsafe cuMemAlloc
  { alloca-  `DevicePtr a' dptr*
  ,          `Int'               } -> `Status' cToEnum #}
  where dptr = liftM DevicePtr . peekIntConv

-- |
-- Release a section of device memory
--
free :: DevicePtr a -> IO (Maybe String)
free dp = nothingIfOk `fmap` cuMemFree dp

{# fun unsafe cuMemFree
  { useDevicePtr `DevicePtr a' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Transfer
--------------------------------------------------------------------------------

-- |
-- Copy a number of elements from the device to host memory. This is a
-- synchronous operation
--
peekArray :: Storable a => Int -> DevicePtr a -> Ptr a -> IO (Maybe String)
peekArray n dptr hptr = doPeek undefined dptr
  where
    doPeek :: Storable a' => a' -> DevicePtr a' -> IO (Maybe String)
    doPeek x _ = nothingIfOk `fmap` cuMemcpyDtoH hptr dptr (n * sizeOf x)

{# fun unsafe cuMemcpyDtoH
  { castPtr      `Ptr a'
  , useDevicePtr `DevicePtr a'
  ,              `Int'         } -> `Status' cToEnum #}


-- |
-- Copy memory from the device asynchronously, possibly associated with a
-- particular stream. The destination host memory must be page-locked (allocated
-- by mallocHost)
--
peekArrayAsync :: Storable a => Int -> DevicePtr a -> Ptr a -> Maybe Stream -> IO (Maybe String)
peekArrayAsync n dptr hptr mst = doPeek undefined dptr
  where
    doPeek :: Storable a' => a' -> DevicePtr a' -> IO (Maybe String)
    doPeek x _ =
      nothingIfOk `fmap` case mst of
        Nothing -> cuMemcpyDtoHAsync hptr dptr (n * sizeOf x) nullPtr
        Just st -> (withStream st $ \s -> cuMemcpyDtoHAsync hptr dptr (n * sizeOf x) s)

{# fun unsafe cuMemcpyDtoHAsync
  { castPtr      `Ptr a'
  , useDevicePtr `DevicePtr a'
  ,              `Int'
  , castPtr      `Ptr Stream'  } -> `Status' cToEnum #}


-- |
-- Copy a number of elements onto the device. This is a synchronous operation
--
pokeArray :: Storable a => Int -> Ptr a -> DevicePtr a -> IO (Maybe String)
pokeArray n hptr dptr = doPoke undefined dptr
  where
    doPoke :: Storable a' => a' -> DevicePtr a' -> IO (Maybe String)
    doPoke x _ = nothingIfOk `fmap` cuMemcpyHtoD dptr hptr (n * sizeOf x)

{# fun unsafe cuMemcpyHtoD
  { useDevicePtr `DevicePtr a'
  , castPtr      `Ptr a'
  ,              `Int'         } -> `Status' cToEnum #}


-- |
-- Copy memory onto the device asynchronously, possibly associated with a
-- particular stream. The source host memory must be page-locked (allocated by
-- mallocHost)
--
pokeArrayAsync :: Storable a => Int -> Ptr a -> DevicePtr a -> Maybe Stream -> IO (Maybe String)
pokeArrayAsync n hptr dptr mst = dopoke undefined dptr
  where
    dopoke :: Storable a' => a' -> DevicePtr a' -> IO (Maybe String)
    dopoke x _ =
      nothingIfOk `fmap` case mst of
        Nothing -> cuMemcpyHtoDAsync dptr hptr (n * sizeOf x) nullPtr
        Just st -> (withStream st $ \s -> cuMemcpyHtoDAsync dptr hptr (n * sizeOf x) s)

{# fun unsafe cuMemcpyHtoDAsync
  { useDevicePtr `DevicePtr a'
  , castPtr      `Ptr a'
  ,              `Int'
  , castPtr      `Ptr Stream'  } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Utility
--------------------------------------------------------------------------------

-- |
-- Set a number of data elements to the specified value, which may be either 8-,
-- 16-, or 32-bits wide.
--
memset :: Storable a => DevicePtr a -> Int -> a -> IO (Maybe String)
memset dptr n val = case sizeOf val of
    1 -> nothingIfOk `fmap` cuMemsetD8  dptr val n
    2 -> nothingIfOk `fmap` cuMemsetD16 dptr val n
    4 -> nothingIfOk `fmap` cuMemsetD32 dptr val n
    _ -> return $ Just "can only memset 8-, 16-, and 32-bit values"

--
-- We use unsafe coerce below to reinterpret the bits of the value to memset as,
-- into the integer type required by the setting functions.
--
{# fun unsafe cuMemsetD8
  { useDevicePtr `DevicePtr a'
  , unsafeCoerce `a'
  ,              `Int'         } -> `Status' cToEnum #}

{# fun unsafe cuMemsetD16
  { useDevicePtr `DevicePtr a'
  , unsafeCoerce `a'
  ,              `Int'         } -> `Status' cToEnum #}

{# fun unsafe cuMemsetD32
  { useDevicePtr `DevicePtr a'
  , unsafeCoerce `a'
  ,              `Int'         } -> `Status' cToEnum #}


-- |
-- Return the device pointer associated with a mapped, pinned host buffer, which
-- was allocated with the DEVICE_MAPPED option.
--
-- Currently, no options are supported and this must be empty.
--
getDevicePtr :: [AllocFlag] -> HostPtr a -> IO (Either String (DevicePtr a))
getDevicePtr flags hptr = withHostPtr hptr $ \hp -> (resultIfOk `fmap` cuMemHostGetDevicePointer hp flags)

{# fun unsafe cuMemHostGetDevicePointer
  { alloca-         `DevicePtr a' dptr*
  , castPtr         `Ptr a'
  , combineBitMasks `[AllocFlag]'       } -> `Status' cToEnum #}
  where dptr = liftM DevicePtr . peekIntConv

