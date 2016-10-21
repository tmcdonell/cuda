{-# LANGUAGE BangPatterns #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Ptr
-- Copyright : [2009..2014] Trevor L. McDonell
-- License   : BSD
--
-- Data pointers on the host and device. These can be shared freely between the
-- CUDA runtime and Driver APIs.
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Ptr (

  -- * Device pointers
  DevicePtr(..),
  withDevicePtr,
  devPtrToWordPtr,
  wordPtrToDevPtr,
  nullDevPtr,
  castDevPtr,
  plusDevPtr,
  alignDevPtr,
  minusDevPtr,
  advanceDevPtr,

  -- * Host pointers
  HostPtr(..),
  withHostPtr,
  nullHostPtr,
  castHostPtr,
  plusHostPtr,
  alignHostPtr,
  minusHostPtr,
  advanceHostPtr,

) where

-- friends
import Foreign.CUDA.Types

-- system
import Foreign.Ptr
import Foreign.Storable


--------------------------------------------------------------------------------
-- Device Pointer
--------------------------------------------------------------------------------

-- |
-- Look at the contents of device memory. This takes an IO action that will be
-- applied to that pointer, the result of which is returned. It would be silly
-- to return the pointer from the action.
--
{-# INLINEABLE withDevicePtr #-}
withDevicePtr :: DevicePtr a -> (Ptr a -> IO b) -> IO b
withDevicePtr !p !f = f (useDevicePtr p)

-- |
-- Return a unique handle associated with the given device pointer
--
{-# INLINEABLE devPtrToWordPtr #-}
devPtrToWordPtr :: DevicePtr a -> WordPtr
devPtrToWordPtr = ptrToWordPtr . useDevicePtr

-- |
-- Return a device pointer from the given handle
--
{-# INLINEABLE wordPtrToDevPtr #-}
wordPtrToDevPtr :: WordPtr -> DevicePtr a
wordPtrToDevPtr = DevicePtr . wordPtrToPtr

-- |
-- The constant 'nullDevPtr' contains the distinguished memory location that is
-- not associated with a valid memory location
--
{-# INLINEABLE nullDevPtr #-}
nullDevPtr :: DevicePtr a
nullDevPtr =  DevicePtr nullPtr

-- |
-- Cast a device pointer from one type to another
--
{-# INLINEABLE castDevPtr #-}
castDevPtr :: DevicePtr a -> DevicePtr b
castDevPtr (DevicePtr !p) = DevicePtr (castPtr p)

-- |
-- Advance the pointer address by the given offset in bytes.
--
{-# INLINEABLE plusDevPtr #-}
plusDevPtr :: DevicePtr a -> Int -> DevicePtr a
plusDevPtr (DevicePtr !p) !d = DevicePtr (p `plusPtr` d)

-- |
-- Given an alignment constraint, align the device pointer to the next highest
-- address satisfying the constraint
--
{-# INLINEABLE alignDevPtr #-}
alignDevPtr :: DevicePtr a -> Int -> DevicePtr a
alignDevPtr (DevicePtr !p) !i = DevicePtr (p `alignPtr` i)

-- |
-- Compute the difference between the second and first argument. This fulfils
-- the relation
--
-- > p2 == p1 `plusDevPtr` (p2 `minusDevPtr` p1)
--
{-# INLINEABLE minusDevPtr #-}
minusDevPtr :: DevicePtr a -> DevicePtr a -> Int
minusDevPtr (DevicePtr !a) (DevicePtr !b) = a `minusPtr` b

-- |
-- Advance a pointer into a device array by the given number of elements
--
{-# INLINEABLE advanceDevPtr #-}
advanceDevPtr :: Storable a => DevicePtr a -> Int -> DevicePtr a
advanceDevPtr = doAdvance undefined
  where
    doAdvance :: Storable a' => a' -> DevicePtr a' -> Int -> DevicePtr a'
    doAdvance x !p !i = p `plusDevPtr` (i * sizeOf x)


--------------------------------------------------------------------------------
-- Host Pointer
--------------------------------------------------------------------------------

-- |
-- Apply an IO action to the memory reference living inside the host pointer
-- object. All uses of the pointer should be inside the 'withHostPtr' bracket.
--
{-# INLINEABLE withHostPtr #-}
withHostPtr :: HostPtr a -> (Ptr a -> IO b) -> IO b
withHostPtr !p !f = f (useHostPtr p)


-- |
-- The constant 'nullHostPtr' contains the distinguished memory location that is
-- not associated with a valid memory location
--
{-# INLINEABLE nullHostPtr #-}
nullHostPtr :: HostPtr a
nullHostPtr =  HostPtr nullPtr

-- |
-- Cast a host pointer from one type to another
--
{-# INLINEABLE castHostPtr #-}
castHostPtr :: HostPtr a -> HostPtr b
castHostPtr (HostPtr !p) = HostPtr (castPtr p)

-- |
-- Advance the pointer address by the given offset in bytes
--
{-# INLINEABLE plusHostPtr #-}
plusHostPtr :: HostPtr a -> Int -> HostPtr a
plusHostPtr (HostPtr !p) !d = HostPtr (p `plusPtr` d)

-- |
-- Given an alignment constraint, align the host pointer to the next highest
-- address satisfying the constraint
--
{-# INLINEABLE alignHostPtr #-}
alignHostPtr :: HostPtr a -> Int -> HostPtr a
alignHostPtr (HostPtr !p) !i = HostPtr (p `alignPtr` i)

-- |
-- Compute the difference between the second and first argument
--
{-# INLINEABLE minusHostPtr #-}
minusHostPtr :: HostPtr a -> HostPtr a -> Int
minusHostPtr (HostPtr !a) (HostPtr !b) = a `minusPtr` b

-- |
-- Advance a pointer into a host array by a given number of elements
--
{-# INLINEABLE advanceHostPtr #-}
advanceHostPtr :: Storable a => HostPtr a -> Int -> HostPtr a
advanceHostPtr = doAdvance undefined
  where
    doAdvance :: Storable a' => a' -> HostPtr a' -> Int -> HostPtr a'
    doAdvance x !p !i = p `plusHostPtr` (i * sizeOf x)

