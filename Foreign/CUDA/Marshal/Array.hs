--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Marshal.Array
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Multi-dimensional arrays for CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Marshal.Array
  (
    -- ** Marshalling
    --
    -- Store and retrieve Haskell lists as arrays on the device
    --
    peek,
    poke,
    peekArray,
    peekArrayElemOff,
    pokeArray,
    pokeArrayElemOff,

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
  )
  where

import Foreign.Ptr
import Foreign.Storable (Storable)
import qualified Foreign.Storable as F
import qualified Foreign.Marshal  as F

-- Friends
import Foreign.CUDA.DevicePtr
import Foreign.CUDA.Marshal.Alloc


--------------------------------------------------------------------------------
-- Primitive Types
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
-- Store the given value to device memory
--
poke :: Storable a => DevicePtr a -> a -> IO (Maybe String)
poke d v =
    F.with v        $ \hptr ->
    withDevicePtr d $ \dptr ->
    memcpy dptr hptr (fromIntegral (F.sizeOf v)) HostToDevice


--------------------------------------------------------------------------------
-- Arrays
--------------------------------------------------------------------------------

-- |
-- Retrieve the specified number of elements from device memory and return as a
-- Haskell list
--
peekArray :: Storable a => Int -> DevicePtr a -> IO (Either String [a])
peekArray =  peekArrayElemOff 0


-- |
-- Retrieve the specified number of elements from device memory, beginning at
-- the specified index (from zero), and return as a Haskell list.
--
peekArrayElemOff :: Storable a => Int -> Int -> DevicePtr a -> IO (Either String [a])
peekArrayElemOff o n d = doPeek undefined
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
-- Store the list elements consecutively in device memory
--
pokeArray :: Storable a => DevicePtr a -> [a] -> IO (Maybe String)
pokeArray = pokeArrayElemOff 0


-- |
-- Store the list elements consecutively in device memory, beginning at the
-- specified index (from zero)
--
pokeArrayElemOff :: Storable a => Int -> DevicePtr a -> [a] -> IO (Maybe String)
pokeArrayElemOff o d = doPoke undefined
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
    moduleForceEither `fmap` malloc bytes >>= \dptr ->
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
    moduleForceEither `fmap` malloc bytes >>= \dptr ->
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
-- Auxiliary utilities
--------------------------------------------------------------------------------

--
-- Internal, unwrap and cast. Required as some functions which work with untyped
-- `void' data.
--
withDevicePtr' :: DevicePtr a -> (Ptr b -> IO c) -> IO c
withDevicePtr' =  withDevicePtr . castDevicePtr

--
-- Errors
--
moduleErr :: String -> String -> a
moduleErr fun msg =  error ("Foreign.CUDA." ++ fun ++ ':':' ':msg)

moduleForceEither :: Either String a -> a
moduleForceEither (Left  s) = moduleErr "Marshal" s
moduleForceEither (Right r) = r

