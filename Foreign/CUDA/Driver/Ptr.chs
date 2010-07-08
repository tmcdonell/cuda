--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Ptr
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Ptr
  where

-- System
import Foreign.Ptr
import Foreign.Storable
import Foreign.C.Types

#include <cuda.h>

--------------------------------------------------------------------------------
-- Device Pointer
--------------------------------------------------------------------------------

-- |
-- A reference to data stored on the device
--
data DevicePtr a = DevicePtr { useDevicePtr :: {#type CUdeviceptr#} }
  deriving (Eq,Ord)
--
-- The driver interface requires this special device pointer type. Specifically,
-- 'mallocArray' and the like operate on objects of type CUdeviceptr, which is a
-- 32-bit value on all platforms.
--
-- Tesla architecture products (compute 1.x) support only a 32-bit address space
-- and expect pointers passed to kernels to be of this width, while the Fermi
-- architecture supports a 64-bit address space and associated pointer width.
--
-- When interfacing with the driver functions, we require this 32-bit opaque
-- type. When passing pointers to kernel functions, we require the native host
-- pointer type. On 32-bit platforms, this distinction is irrelevant. For Tesla
-- devices executing on 64-bit host platforms, the runtime will automatically
-- squash pointers to 32-bits. Fermi architectures however may use the entire
-- bitwidth, so the 32-bit CUdeviceptr must be converted to a 64-bit pointer by
-- the application before passing to the kernel.
--

instance Show (DevicePtr a) where
  showsPrec n (DevicePtr p) = showsPrec n p

instance Storable (DevicePtr a) where
  sizeOf _    = sizeOf    (undefined :: Ptr a)
  alignment _ = alignment (undefined :: Ptr a)
  peek p      = ptrToDevPtr `fmap` peek (castPtr p)
  poke p v    = poke (castPtr p) (devPtrToPtr v)


-- |
-- Look at the contents of device memory. This takes an IO action that will be
-- applied to that pointer, the result of which is returned. It would be silly
-- to return the pointer from the action.
--
withDevicePtr :: DevicePtr a -> (Ptr a -> IO b) -> IO b
withDevicePtr p f = f (devPtrToPtr p)

-- |
-- Return a type pointer representation of the opaque device pointer
--
devPtrToPtr :: DevicePtr a -> Ptr a
devPtrToPtr = wordPtrToPtr . devPtrToWordPtr

-- |
-- Convert a type pointer to device pointer
--
ptrToDevPtr :: Ptr a -> DevicePtr a
ptrToDevPtr = wordPtrToDevPtr . ptrToWordPtr

-- |
-- Return a unique handle associated with the given device pointer
--
devPtrToWordPtr :: DevicePtr a -> WordPtr
devPtrToWordPtr = fromIntegral . useDevicePtr

-- |
-- Return a device pointer from the given handle
--
wordPtrToDevPtr :: WordPtr -> DevicePtr a
wordPtrToDevPtr = DevicePtr . fromIntegral

-- |
-- The constant 'nullDevPtr' contains the distinguished memory location that is
-- not associated with a valid memory location
--
nullDevPtr :: DevicePtr a
nullDevPtr =  DevicePtr 0

-- |
-- Cast a device pointer from one type to another
--
castDevPtr :: DevicePtr a -> DevicePtr b
castDevPtr = DevicePtr . useDevicePtr

-- |
-- Advance the pointer address by the given offset in bytes.
--
plusDevPtr :: DevicePtr a -> Int -> DevicePtr a
plusDevPtr (DevicePtr p) d = DevicePtr (p + fromIntegral d)

-- |
-- Given an alignment constraint, align the device pointer to the next highest
-- address satisfying the constraint
--
alignDevPtr :: DevicePtr a -> Int -> DevicePtr a
alignDevPtr addr@(DevicePtr a) i =
  let x = fromIntegral i
  in case a `rem` x of
       0 -> addr
       n -> DevicePtr (a + (x-n))

-- |
-- Compute the difference between the second and first argument. This fulfils
-- the relation
--
-- > p2 == p1 `plusDevPtr` (p2 `minusDevPtr` p1)
--
minusDevPtr :: DevicePtr a -> DevicePtr a -> Int
minusDevPtr (DevicePtr a) (DevicePtr b) = fromIntegral (a - b)

-- |
-- Advance a pointer into a device array by the given number of elements
--
advanceDevPtr :: Storable a => DevicePtr a -> Int -> DevicePtr a
advanceDevPtr  = doAdvance undefined
  where
    doAdvance :: Storable a' => a' -> DevicePtr a' -> Int -> DevicePtr a'
    doAdvance x p i = p `plusDevPtr` (i * sizeOf x)


--------------------------------------------------------------------------------
-- Host Pointer
--------------------------------------------------------------------------------

-- |
-- A reference to page-locked host memory
--
data HostPtr a = HostPtr { useHostPtr :: Ptr a }
  deriving (Eq,Ord)

instance Show (HostPtr a) where
  showsPrec n (HostPtr p) = showsPrec n p

instance Storable (HostPtr a) where
  sizeOf _    = sizeOf    (undefined :: Ptr a)
  alignment _ = alignment (undefined :: Ptr a)
  peek p      = HostPtr `fmap` peek (castPtr p)
  poke p v    = poke (castPtr p) (useHostPtr v)


-- |
-- Apply an IO action to the memory reference living inside the host pointer
-- object. All uses of the pointer should be inside the 'withHostPtr' bracket.
--
withHostPtr :: HostPtr a -> (Ptr a -> IO b) -> IO b
withHostPtr p f = f (useHostPtr p)


-- |
-- The constant 'nullHostPtr' contains the distinguished memory location that is
-- not associated with a valid memory location
--
nullHostPtr :: HostPtr a
nullHostPtr =  HostPtr nullPtr

-- |
-- Cast a host pointer from one type to another
--
castHostPtr :: HostPtr a -> HostPtr b
castHostPtr (HostPtr p) = HostPtr (castPtr p)

-- |
-- Advance the pointer address by the given offset in bytes
--
plusHostPtr :: HostPtr a -> Int -> HostPtr a
plusHostPtr (HostPtr p) d = HostPtr (p `plusPtr` d)

-- |
-- Given an alignment constraint, align the host pointer to the next highest
-- address satisfying the constraint
--
alignHostPtr :: HostPtr a -> Int -> HostPtr a
alignHostPtr (HostPtr p) i = HostPtr (p `alignPtr` i)

-- |
-- Compute the difference between the second and first argument
--
minusHostPtr :: HostPtr a -> HostPtr a -> Int
minusHostPtr (HostPtr a) (HostPtr b) = a `minusPtr` b

-- |
-- Advance a pointer into a host array by a given number of elements
--
advanceHostPtr :: Storable a => HostPtr a -> Int -> HostPtr a
advanceHostPtr  = doAdvance undefined
  where
    doAdvance :: Storable a' => a' -> HostPtr a' -> Int -> HostPtr a'
    doAdvance x p i = p `plusHostPtr` (i * sizeOf x)

