--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.DevicePtr
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- References to objects stored on the CUDA devices
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.DevicePtr
  where

-- System
import Foreign.Ptr


--------------------------------------------------------------------------------
-- Device Pointer
--------------------------------------------------------------------------------

-- |
-- A reference to data stored on the device
--
data DevicePtr a = DevicePtr { useDevicePtr :: Ptr a }
  deriving (Eq,Ord)

instance Show (DevicePtr a) where
  showsPrec n (DevicePtr p) rs = showsPrec n p rs


-- |
-- The constant 'nullDevPtr' contains the distinguished memory location that is
-- not associated with a valid memory location
--
nullDevPtr :: DevicePtr a
nullDevPtr =  DevicePtr nullPtr


-- |
-- Cast a device pointer from one type to another
--
castDevPtr :: DevicePtr a -> DevicePtr b
castDevPtr (DevicePtr p) = DevicePtr (castPtr p)


-- |
-- Advance the pointer address by the given offset in bytes.
--
plusDevPtr :: DevicePtr a -> Int -> DevicePtr a
plusDevPtr (DevicePtr p) d = DevicePtr (p `plusPtr` d)


-- |
-- Given an alignment constraint, align the device pointer to the next highest
-- address satisfying the constraint
--
alignDevPtr :: DevicePtr a -> Int -> DevicePtr a
alignDevPtr (DevicePtr p) i = DevicePtr (p `alignPtr` i)


-- |
-- Compute the difference between the second and first argument
--
minusDevPtr :: DevicePtr a -> DevicePtr a -> Int
minusDevPtr (DevicePtr a) (DevicePtr b) = a `minusPtr` b

