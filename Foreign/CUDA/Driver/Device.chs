{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Device
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Error handling
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Device
  (
    Device, DeviceProperties(..), Attribute(..),

    initialise,
    capability,
    get,
    attribute,
    count,
    name,
    props,
    totalMem
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Internal.Offsets

-- System
import Foreign
import Foreign.C
import Control.Monad            (liftM)


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

newtype Device = Device {# type CUdevice #}
  deriving (Show)

use :: Device -> {#type CUdevice#}
use (Device d) = d


-- |
-- Device attributes
--
{# enum CUdevice_attribute as Attribute
    { underscoreToCase }
    with prefix="CU_DEVICE_ATTRIBUTE" deriving (Eq, Show) #}

{# pointer *CUdevprop as ^ foreign -> DeviceProperties nocode #}


-- |
-- Properties of the compute device
--
data DeviceProperties = DeviceProperties
  {
    maxThreadsPerBlock  :: Int,           -- ^ Maximum number of threads per block
    maxThreadsDim       :: (Int,Int,Int), -- ^ Maximum size of each dimension of a block
    maxGridSize         :: (Int,Int,Int), -- ^ Maximum size of each dimension of a grid
    sharedMemPerBlock   :: Int,           -- ^ Shared memory available per block in bytes
    totalConstantMemory :: Int,           -- ^ Constant memory available on device in bytes
    warpSize            :: Int,           -- ^ Warp size in threads (SIMD width)
    memPitch            :: Int,           -- ^ Maximum pitch in bytes allowed by memory copies
    regsPerBlock        :: Int,           -- ^ 32-bit registers available per block
    clockRate           :: Int,           -- ^ Clock frequency in kilohertz
    textureAlign        :: Int            -- ^ Alignment requirement for textures
  }
  deriving (Show)

instance Storable DeviceProperties where
  sizeOf _    = {#sizeof CUdevprop#}
  alignment _ = alignment (undefined :: Ptr ())

  peek p      = do
    tb <- cIntConv `fmap` {#get CUdevprop.maxThreadsPerBlock#} p
    sm <- cIntConv `fmap` {#get CUdevprop.sharedMemPerBlock#} p
    cm <- cIntConv `fmap` {#get CUdevprop.totalConstantMemory#} p
    ws <- cIntConv `fmap` {#get CUdevprop.SIMDWidth#} p
    mp <- cIntConv `fmap` {#get CUdevprop.memPitch#} p
    rb <- cIntConv `fmap` {#get CUdevprop.regsPerBlock#} p
    cl <- cIntConv `fmap` {#get CUdevprop.clockRate#} p
    ta <- cIntConv `fmap` {#get CUdevprop.textureAlign#} p

    (t1:t2:t3:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxThreadDimOffset' :: Ptr CInt)
    (g1:g2:g3:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxGridSizeOffset'  :: Ptr CInt)

    return DeviceProperties
      {
        maxThreadsPerBlock  = tb,
        maxThreadsDim       = (t1,t2,t3),
        maxGridSize         = (g1,g2,g3),
        sharedMemPerBlock   = sm,
        totalConstantMemory = cm,
        warpSize            = ws,
        memPitch            = mp,
        regsPerBlock        = rb,
        clockRate           = cl,
        textureAlign        = ta
      }

--------------------------------------------------------------------------------
-- Initialisation
--------------------------------------------------------------------------------

-- |
-- Initialise the CUDA driver API. Fills in the `flags' parameter with the
-- required zero value. Must be called before any other driver function.
--
initialise :: IO (Maybe String)
initialise =  nothingIfOk `fmap` cuInit 0

{# fun unsafe cuInit
  { cIntConv `Int' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Device Management
--------------------------------------------------------------------------------

-- |
-- Return the compute compatibility revision supported by the device
--
capability :: Device -> IO (Either String Double)
capability dev =
  (\(s,a,b) -> resultIfOk (s,cap a b)) `fmap` cuDeviceComputeCapability dev
  where
    cap a b = let a' = fromIntegral a in
              let b' = fromIntegral b in
              a' + b' / max 10 (10^ ((ceiling . logBase 10) b' :: Int))

{# fun unsafe cuDeviceComputeCapability
  { alloca-  `Int'    peekIntConv*
  , alloca-  `Int'    peekIntConv*
  , use      `Device'              } -> `Status' cToEnum #}


-- |
-- Return a device handle
--
get :: Int -> IO (Either String Device)
get d = resultIfOk `fmap` cuDeviceGet d

{# fun unsafe cuDeviceGet
  { alloca-  `Device' dev*
  , cIntConv `Int'           } -> `Status' cToEnum #}
  where dev = liftM Device . peekIntConv


-- |
-- Return the selected attribute for the given device
--
attribute :: Device -> Attribute -> IO (Either String Int)
attribute d a = resultIfOk `fmap` cuDeviceGetAttribute a d

{# fun unsafe cuDeviceGetAttribute
  { alloca-   `Int'       peekIntConv*
  , cFromEnum `Attribute'
  , use       `Device'                 } -> `Status' cToEnum #}


-- |
-- Return the number of device with compute capability > 1.0
--
count :: IO (Either String Int)
count = resultIfOk `fmap` cuDeviceGetCount

{# fun unsafe cuDeviceGetCount
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Name of the device
--
name :: Device -> IO (Either String String)
name d = resultIfOk `fmap` cuDeviceGetName d

{# fun unsafe cuDeviceGetName
  { allocaS- `String'& peekS*
  , use      `Device'         } -> `Status' cToEnum #}
  where
    len            = 512
    allocaS a      = allocaBytes len $ \p -> a (p, cIntConv len)
    peekS s _      = peekCString s


-- |
-- Return the properties of the selected device
--
props :: Device -> IO (Either String DeviceProperties)
props d = resultIfOk `fmap` cuDeviceGetProperties d

{# fun unsafe cuDeviceGetProperties
  { alloca-  `DeviceProperties' peek*
  , use      `Device'                 } -> `Status' cToEnum #}


-- |
-- Total memory available on the device (bytes)
--
totalMem :: Device -> IO (Either String Int)
totalMem d = resultIfOk `fmap` cuDeviceTotalMem d

{# fun unsafe cuDeviceTotalMem
  { alloca-  `Int' peekIntConv*
  , use      `Device'                 } -> `Status' cToEnum #}

