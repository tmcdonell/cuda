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

    -- ** Dynamic allocation
    free,
    malloc,
--    malloc2D,
--    malloc3D,
    memset,
--    memset2D,
--    memset3D,

    -- ** Marshalling
    peek,
    poke,
    peekArray,
    pokeArray,

    -- ** Combined allocation and marshalling
    new,
    with,
    newArray,
    withArray,

    -- ** Copying
  )
  where

import Data.Int

import Foreign.C
import Foreign.Ptr
import Foreign.ForeignPtr
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
-- A reference to data stored on the device. It is automatically freed.
--
type DevicePtr a = ForeignPtr a

withDevicePtr :: DevicePtr a -> (Ptr b -> IO c) -> IO c
withDevicePtr =  withForeignPtr . castForeignPtr

newDevicePtr   :: Ptr a -> IO (DevicePtr a)
newDevicePtr p =  (doAutoRelease free_) >>= \f -> newForeignPtr f p

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
malloc bytes = do
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
free   :: Ptr a -> IO (Maybe String)
free p =  nothingIfOk `fmap` cudaFree p

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
memset                  :: DevicePtr a -> Int64 -> Int -> IO (Maybe String)
memset ptr bytes symbol =  nothingIfOk `fmap` cudaMemset ptr symbol bytes

{# fun unsafe cudaMemset
    { withDevicePtr* `DevicePtr a' ,
                     `Int'         ,
      cIntConv       `Int64'       } -> `Status' cToEnum #}


-- |
-- Initialise a matrix to a given value
--
memset2D :: DevicePtr a                 -- ^ The device memory
         -> (Int64, Int64)              -- ^ The (width,height) of the matrix in bytes
         -> Int64                       -- ^ The allocation pitch, as returned by 'malloc2D'
         -> Int                         -- ^ Value to set for each byte
         -> IO (Maybe String)
memset2D ptr (width,height) pitch symbol = nothingIfOk `fmap` cudaMemset2D ptr pitch symbol width height

{# fun unsafe cudaMemset2D
    { withDevicePtr* `DevicePtr a' ,
      cIntConv       `Int64'       ,
                     `Int'         ,
      cIntConv       `Int64'       ,
      cIntConv       `Int64'       } -> `Status' cToEnum #}


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
-- Marshalling
--------------------------------------------------------------------------------

-- |
-- Retrieve the specified value from device memory
--
peek :: Storable a => DevicePtr a -> IO a
peek d = doPeek undefined
  where
    doPeek   :: Storable a' => a' -> IO a'
    doPeek x =  F.alloca        $ \hptr ->
                withDevicePtr d $ \dptr ->
                memcpy hptr dptr (fromIntegral (F.sizeOf x)) DeviceToHost >>= \rv ->
                case rv of
                    Nothing -> F.peek hptr
                    Just s  -> moduleErr "peek" s


-- |
-- Retrieve the specified number of elements from device memory and return as a
-- Haskell list
--
peekArray :: Storable a => Int -> DevicePtr a -> IO [a]
peekArray n d = doPeek undefined
  where
    doPeek   :: Storable a' => a' -> IO [a']
    doPeek x =  let bytes = fromIntegral n * (fromIntegral (F.sizeOf x)) in
                F.allocaArray n $ \hptr ->
                withDevicePtr d $ \dptr ->
                memcpy hptr dptr bytes DeviceToHost >>= \rv ->
                case rv of
                   Nothing -> F.peekArray n hptr
                   Just s  -> moduleErr "peekArray" s


-- |
-- Store the given value to device memory
--
poke :: Storable a => DevicePtr a -> a -> IO ()
poke d v =
    F.with v        $ \hptr ->
    withDevicePtr d $ \dptr ->
    memcpy dptr hptr (fromIntegral (F.sizeOf v)) HostToDevice >>= \rv ->
    case rv of
        Nothing -> return ()
        Just s  -> moduleErr "poke" s


-- |
-- Store the list elements consecutively in device memory
--
pokeArray :: Storable a => DevicePtr a -> [a] -> IO ()
pokeArray d = doPoke undefined
  where
    doPoke     :: Storable a' => a' -> [a'] -> IO ()
    doPoke x v =  F.withArrayLen v $ \n hptr ->
                  withDevicePtr  d $ \dptr ->
                  let bytes = (fromIntegral n) * (fromIntegral (F.sizeOf x)) in
                  memcpy dptr hptr bytes HostToDevice >>= \rv ->
                  case rv of
                      Nothing -> return ()
                      Just s  -> moduleErr "pokeArray" s


--------------------------------------------------------------------------------
-- Combined allocation and marshalling
--------------------------------------------------------------------------------

-- |
-- Allocate a block of memory on the device and transfer a value into it
--
new :: Storable a => a -> IO (DevicePtr a)
new v =
    let bytes = fromIntegral (F.sizeOf v) in
    forceEither `fmap` malloc bytes >>= \dptr ->
    poke dptr v >> return dptr


-- |
-- Write the list of storable elements into a newly allocated linear array on
-- the device
--
newArray   :: Storable a => [a] -> IO (DevicePtr a)
newArray v =
    let bytes = fromIntegral (length v) * fromIntegral (F.sizeOf (head v)) in
    forceEither `fmap` malloc bytes >>= \dptr ->
    pokeArray dptr v >> return dptr


-- |
-- Execute a computation passing a pointer to a temporarily allocated block of
-- device memory containing the value
--
with :: Storable a => a -> (DevicePtr a -> IO b) -> IO b
with v f =
    new v  >>= \dptr ->
    f dptr >>= return


-- |
-- Temporarily store a list of variable in device memory and operate on them
--
withArray :: Storable a => [a] -> (DevicePtr a -> IO b) -> IO b
withArray v f =
    newArray v >>= \dptr ->
    f dptr     >>= return


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

