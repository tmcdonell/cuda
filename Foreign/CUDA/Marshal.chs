{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Marshal
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Memory allocation and marshalling support for CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Marshal
  (
    DevicePtr,
    withDevicePtr,

    -- ** Dynamic allocation
    --
    -- Basic methods to allocate a block of memory on the device. The memory
    -- should be deallocated using 'free' when no longer necessary. The
    -- advantage is that failed allocations will not raise an exception.
    --
    free,
    malloc,
    malloc2D,
    malloc3D,
    memset,
    memset2D,
    memset3D,

    -- ** Local allocation
    --
    -- Execute a computation, passing as argument a pointer to a newly allocated
    -- block of memory. The memory is freed when the computation terminates.
    --
    alloca,
    allocaBytes,
    allocaBytesMemset,

    -- ** Marshalling
    --
    -- Store and retrieve Haskell lists as arrays on the device
    --
    peek,
    poke,
    peekArray,
    peekArrayAt,
    pokeArray,
    pokeArrayAt,

    -- ** Combined allocation and marshalling
    --
    -- Write Haskell values (or lists) to newly allocated device memory. This
    -- may be a temporary store.
    --
    new,
    with,
    newArray,
    withArray,
    withArrayLen,

    -- ** Copying
    --
    -- Copy data between the host and device
    --
  )
  where

import Data.Int
import Control.Exception

import Foreign.C
import Foreign.Ptr
import Foreign.Concurrent
import Foreign.ForeignPtr hiding (newForeignPtr, addForeignPtrFinalizer)
import Foreign.Storable (Storable)

import qualified Foreign.Storable as F
import qualified Foreign.Marshal  as F

import Foreign.CUDA.Utils
import Foreign.CUDA.Error
import Foreign.CUDA.Stream
import Foreign.CUDA.Internal.C2HS

#include <cuda_runtime_api.h>
{# context lib="cudart" #}


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A reference to data stored on the device
--
type DevicePtr a = ForeignPtr a

-- |
-- Unwrap the device pointer, yielding a device memory address that can be
-- passed to a kernel function callable from the host code. This allows the type
-- system to strictly separate host and device memory references, a non-obvious
-- distinction in pure CUDA.
--
withDevicePtr :: DevicePtr a -> (Ptr a -> IO b) -> IO b
withDevicePtr =  withForeignPtr

--
-- Internal, unwrap and cast. Required as some functions such as memset work
-- with untyped `void' data.
--
withDevicePtr' :: DevicePtr a -> (Ptr b -> IO c) -> IO c
withDevicePtr' =  withForeignPtr . castForeignPtr

--
-- Wrap a device memory pointer. The memory will be freed once the last
-- reference to the data is dropped, although there is no guarantee as to
-- exactly when this will occur.
--
newDevicePtr   :: Ptr a -> IO (DevicePtr a)
newDevicePtr p =  newForeignPtr p (free_ p)


--
-- Memory copy
--
{# enum cudaMemcpyKind as CopyDirection {}
    with prefix="cudaMemcpy" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Dynamic allocation
--------------------------------------------------------------------------------

-- |
-- Allocate the specified number of bytes in linear memory on the device. The
-- memory is suitably aligned for any kind of variable, and is not cleared.
--
malloc       :: Int64 -> IO (Either String (DevicePtr a))
malloc bytes =  do
    (rv,ptr) <- cudaMalloc bytes
    case rv of
        Success -> Right `fmap` newDevicePtr (castPtr ptr)
        _       -> return . Left . describe $ rv

{# fun unsafe cudaMalloc
    { alloca'-  `Ptr ()' peek'* ,
      cIntConv  `Int64'         } -> `Status' cToEnum #}
    where alloca' = F.alloca
          peek'   = F.peek


-- |
-- Allocate at least @width * height@ bytes of linear memory on the device. The
-- function may pad the allocation to ensure corresponding pointers in each row
-- meet coalescing requirements. The actual allocation width is returned.
--
malloc2D :: (Int64, Int64)              -- ^ allocation (width,height) in bytes
         -> IO (Either String (DevicePtr a, Int64))
malloc2D (width,height) =  do
    (rv,ptr,pitch) <- cudaMallocPitch width height
    case rv of
        Success -> (\p -> Right (p,pitch)) `fmap` newDevicePtr (castPtr ptr)
        _       -> return . Left . describe $ rv

{# fun unsafe cudaMallocPitch
    { alloca'-  `Ptr ()' peek'*       ,
      alloca'-  `Int64'  peekIntConv* ,
      cIntConv  `Int64'               ,
      cIntConv  `Int64'               } -> `Status' cToEnum #}
    where
        -- C->Haskell doesn't like qualified imports
        --
        alloca' :: Storable a => (Ptr a -> IO b) -> IO b
        alloca' =  F.alloca
        peek'   =  F.peek


-- |
-- Allocate at least @width * height * depth@ bytes of linear memory on the
-- device. The function may pad the allocation to ensure hardware alignment
-- requirements are met. The actual allocation pitch is returned
--
malloc3D :: (Int64,Int64,Int64)         -- ^ allocation (width,height,depth) in bytes
         -> IO (Either String (DevicePtr a, Int64))
malloc3D = moduleErr "malloc3D" "not implemented yet"


-- |
-- Free previously allocated memory on the device
--

--free   :: DevicePtr a -> IO (Maybe String)
--free p =  nothingIfOk `fmap` (withDevicePtr p cudaFree)

free :: DevicePtr a -> IO ()
free =  finalizeForeignPtr

free_   :: Ptr a -> IO ()
free_ p =  cudaFree p >>= \rv ->
    case rv of
        Success -> return ()
        _       -> moduleErr "free" (describe rv)

{# fun unsafe cudaFree
    { castPtr `Ptr a' } -> `Status' cToEnum #}

foreign import ccall "wrapper"
    doAutoRelease   :: (Ptr a -> IO ()) -> IO (FunPtr (Ptr a -> IO ()))


-- |
-- Initialise device memory to a given value
--
memset :: DevicePtr a                   -- ^ The device memory
       -> Int64                         -- ^ Number of bytes
       -> Int                           -- ^ Value to set for each byte
       -> IO (Maybe String)
memset ptr bytes symbol =
    nothingIfOk `fmap` cudaMemset ptr symbol bytes

{# fun unsafe cudaMemset
    { withDevicePtr'* `DevicePtr a' ,
                      `Int'         ,
      cIntConv        `Int64'       } -> `Status' cToEnum #}


-- |
-- Initialise a matrix to a given value
--
memset2D :: DevicePtr a                 -- ^ The device memory
         -> (Int64, Int64)              -- ^ The (width,height) of the matrix in bytes
         -> Int64                       -- ^ The allocation pitch, as returned by 'malloc2D'
         -> Int                         -- ^ Value to set for each byte
         -> IO (Maybe String)
memset2D ptr (width,height) pitch symbol =
    nothingIfOk `fmap` cudaMemset2D ptr pitch symbol width height

{# fun unsafe cudaMemset2D
    { withDevicePtr'* `DevicePtr a' ,
      cIntConv        `Int64'       ,
                      `Int'         ,
      cIntConv        `Int64'       ,
      cIntConv        `Int64'       } -> `Status' cToEnum #}


-- |
-- Initialise the elements of a 3D array to a given value
--
memset3D :: DevicePtr a                 -- ^ The device memory
         -> (Int64,Int64,Int64)         -- ^ The (width,height,depth) of the array in bytes
         -> Int64                       -- ^ The allocation pitch, as returned by 'malloc3D'
         -> Int                         -- ^ Value to set for each byte
         -> IO (Maybe String)
memset3D = moduleErr "memset3D" "not implemented yet"


--------------------------------------------------------------------------------
-- Local allocation
--------------------------------------------------------------------------------

-- |
-- Execute a computation, passing a pointer to a temporarily allocated block of
-- memory
--
alloca :: Storable a => (DevicePtr a -> IO b) -> IO b
alloca =  doAlloca undefined
  where
    doAlloca   :: Storable a' => a' -> (DevicePtr a' -> IO b') -> IO b'
    doAlloca x =  allocaBytes (fromIntegral (F.sizeOf x))


-- |
-- Execute a computation, passing a pointer to a block of 'n' bytes of memory
--
allocaBytes :: Int64 -> (DevicePtr a -> IO b) -> IO b
allocaBytes bytes =
    bracket (forceEither `fmap` malloc bytes) free


-- |
-- Execute a computation, passing a pointer to a  block of memory initialised to
-- a given value
--
allocaBytesMemset :: Int64 -> Int -> (DevicePtr a -> IO b) -> IO b
allocaBytesMemset bytes symbol f =
    allocaBytes bytes $ \dptr ->
    memset dptr bytes symbol >>= \rv ->
      case rv of
        Nothing -> f dptr
        Just e  -> error e


--------------------------------------------------------------------------------
-- Marshalling
--------------------------------------------------------------------------------

-- |
-- Retrieve the specified value from device memory
--
peek :: Storable a => DevicePtr a -> IO (Either String a)
peek d = doPeek undefined
  where
    doPeek   :: Storable a' => a' -> IO (Either String a')
    doPeek x =  F.alloca         $ \hptr ->
                withDevicePtr' d $ \dptr ->
                memcpy hptr dptr (fromIntegral (F.sizeOf x)) DeviceToHost >>= \rv ->
                case rv of
                    Nothing -> Right `fmap` F.peek hptr
                    Just s  -> return (Left s)


-- |
-- Retrieve the specified number of elements from device memory and return as a
-- Haskell list
--
peekArray :: Storable a => Int -> DevicePtr a -> IO (Either String [a])
peekArray =  peekArrayAt 0


-- |
-- Retrieve the specified number of elements from device memory, beginning at
-- the specified index (from zero), and return as a Haskell list.
--
peekArrayAt :: Storable a => Int -> Int -> DevicePtr a -> IO (Either String [a])
peekArrayAt o n d = doPeek undefined
  where
    doPeek   :: Storable a' => a' -> IO (Either String [a'])
    doPeek x =  let bytes = fromIntegral n * (fromIntegral (F.sizeOf x)) in
                F.allocaArray  n $ \hptr ->
                withDevicePtr' d $ \dptr ->
                memcpy hptr (dptr `plusPtr` (o * F.sizeOf x)) bytes DeviceToHost >>= \rv ->
                case rv of
                   Nothing -> Right `fmap` F.peekArray n hptr
                   Just s  -> return (Left s)


-- |
-- Store the given value to device memory
--
poke :: Storable a => DevicePtr a -> a -> IO (Maybe String)
poke d v =
    F.with v        $ \hptr ->
    withDevicePtr d $ \dptr ->
    memcpy dptr hptr (fromIntegral (F.sizeOf v)) HostToDevice


-- |
-- Store the list elements consecutively in device memory
--
pokeArray :: Storable a => DevicePtr a -> [a] -> IO (Maybe String)
pokeArray = pokeArrayAt 0


-- |
-- Store the list elements consecutively in device memory, beginning at the
-- specified index (from zero)
--
pokeArrayAt :: Storable a => Int -> DevicePtr a -> [a] -> IO (Maybe String)
pokeArrayAt o d = doPoke undefined
  where
    doPoke     :: Storable a' => a' -> [a'] -> IO (Maybe String)
    doPoke x v =  F.withArrayLen v $ \n hptr ->
                  withDevicePtr' d $ \dptr ->
                  let bytes = (fromIntegral n) * (fromIntegral (F.sizeOf x)) in
                  memcpy (dptr `plusPtr` (o * F.sizeOf x)) hptr bytes HostToDevice


--------------------------------------------------------------------------------
-- Combined allocation and marshalling
--------------------------------------------------------------------------------

-- |
-- Allocate a block of memory on the device and transfer a value into it. The
-- memory may be deallocated using 'free' when no longer required.
--
new :: Storable a => a -> IO (DevicePtr a)
new v =
    let bytes = fromIntegral (F.sizeOf v) in
    forceEither `fmap` malloc bytes >>= \dptr ->
    poke dptr v >>= \rv ->
    case rv of
        Nothing -> return dptr
        Just s  -> moduleErr "new" s


-- |
-- Write the list of storable elements into a newly allocated linear array on
-- the device. The memory may be deallocated when no longer required.
--
newArray   :: Storable a => [a] -> IO (DevicePtr a)
newArray v =
    let bytes = fromIntegral (length v) * fromIntegral (F.sizeOf (head v)) in
    forceEither  `fmap` malloc bytes >>= \dptr ->
    pokeArray dptr v >>= \rv ->
    case rv of
        Nothing -> return dptr
        Just s  -> moduleErr "newArray" s


-- |
-- Execute a computation passing a pointer to a temporarily allocated block of
-- device memory containing the value
--
with :: Storable a => a -> (DevicePtr a -> IO b) -> IO b
with v f =
    alloca $ \dptr ->
    poke dptr v >>= \rv ->
    case rv of
        Nothing -> f dptr
        Just s  -> moduleErr "with" s


-- |
-- Temporarily store a list of variable in device memory and operate on them
--
withArray :: Storable a => [a] -> (DevicePtr a -> IO b) -> IO b
withArray vs f = withArrayLen vs (\_ -> f)


-- |
-- Like 'withArray', but the action gets the number of values as an additional
-- parameter
--
withArrayLen :: Storable a => [a] -> (Int -> DevicePtr a -> IO b) -> IO b
withArrayLen vs f =
    let l = length vs
        b = fromIntegral l * fromIntegral (F.sizeOf (head vs))
    in
        allocaBytes b $ \dptr ->
        pokeArray dptr vs >>= \rv ->
        case rv of
            Nothing -> f l dptr
            Just s  -> moduleErr "withArray" s


--------------------------------------------------------------------------------
-- Copying
--------------------------------------------------------------------------------

-- |
-- Copy data between host and device
--
memcpy :: Ptr a                 -- ^ destination
       -> Ptr a                 -- ^ source
       -> Int64                 -- ^ number of bytes
       -> CopyDirection
       -> IO (Maybe String)
memcpy dst src bytes dir =
    nothingIfOk `fmap` cudaMemcpy dst src bytes dir

{# fun unsafe cudaMemcpy
    { castPtr   `Ptr a'         ,
      castPtr   `Ptr a'         ,
      cIntConv  `Int64'         ,
      cFromEnum `CopyDirection' } -> `Status' cToEnum #}


-- |
-- Copy data between host and device asynchronously
--
memcpyAsync :: Stream           -- ^ asynchronous stream to operate over
            -> Ptr a            -- ^ destination
            -> Ptr a            -- ^ source
            -> Int64            -- ^ number of bytes
            -> CopyDirection
            -> IO (Maybe String)
memcpyAsync stream dst src bytes dir =
    nothingIfOk `fmap` cudaMemcpyAsync dst src bytes dir stream

{# fun unsafe cudaMemcpyAsync
    { castPtr    `Ptr a'         ,
      castPtr    `Ptr a'         ,
      cIntConv   `Int64'         ,
      cFromEnum  `CopyDirection' ,
      cIntConv   `Stream'        } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Internal utilities
--------------------------------------------------------------------------------

moduleErr :: String -> String -> a
moduleErr fun msg =  error ("Foreign.CUDA." ++ fun ++ ':':' ':msg)

