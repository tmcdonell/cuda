{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
#ifdef USE_EMPTY_CASE
{-# LANGUAGE EmptyCase                #-}
#endif
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Device
-- Copyright : [2009..2017] Trevor L. McDonell
-- License   : BSD
--
-- Device management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Device (

  -- * Device Management
  Device(..),
  DeviceProperties(..), DeviceAttribute(..), Compute(..), ComputeMode(..), InitFlag,
  initialise, capability, device, attribute, count, name, props, uuid, totalMem,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Analysis.Device
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Control.Applicative
import Control.Monad                                    ( liftM )
import Data.Bits
import Data.UUID.Types
import Foreign
import Foreign.C
import Prelude


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A CUDA device
--
newtype Device = Device { useDevice :: {# type CUdevice #}}
  deriving (Eq, Show)


-- |
-- Device attributes
--
{# enum CUdevice_attribute as DeviceAttribute
    { underscoreToCase
    , MAX as CU_DEVICE_ATTRIBUTE_MAX }          -- ignore
    with prefix="CU_DEVICE_ATTRIBUTE" deriving (Eq, Show) #}


#if CUDA_VERSION < 5000
{# pointer *CUdevprop as ^ foreign -> CUDevProp nocode #}

--
-- Properties of the compute device (internal helper).
-- Replaced by cuDeviceGetAttribute in CUDA-5.0 and later.
--
data CUDevProp = CUDevProp
  {
    cuMaxThreadsPerBlock :: !Int,               -- Maximum number of threads per block
    cuMaxBlockSize       :: !(Int,Int,Int),     -- Maximum size of each dimension of a block
    cuMaxGridSize        :: !(Int,Int,Int),     -- Maximum size of each dimension of a grid
    cuSharedMemPerBlock  :: !Int64,             -- Shared memory available per block in bytes
    cuTotalConstMem      :: !Int64,             -- Constant memory available on device in bytes
    cuWarpSize           :: !Int,               -- Warp size in threads (SIMD width)
    cuMemPitch           :: !Int64,             -- Maximum pitch in bytes allowed by memory copies
    cuRegsPerBlock       :: !Int,               -- 32-bit registers available per block
    cuClockRate          :: !Int,               -- Clock frequency in kilohertz
    cuTextureAlignment   :: !Int64              -- Alignment requirement for textures
  }
  deriving (Show)


instance Storable CUDevProp where
  sizeOf _    = {#sizeof CUdevprop#}
  alignment _ = alignment (undefined :: Ptr ())

  poke _ _    = error "no instance for Foreign.Storable.poke DeviceProperties"
  peek p      = do
    tb <- cIntConv `fmap` {#get CUdevprop.maxThreadsPerBlock#} p
    sm <- cIntConv `fmap` {#get CUdevprop.sharedMemPerBlock#} p
    cm <- cIntConv `fmap` {#get CUdevprop.totalConstantMemory#} p
    ws <- cIntConv `fmap` {#get CUdevprop.SIMDWidth#} p
    mp <- cIntConv `fmap` {#get CUdevprop.memPitch#} p
    rb <- cIntConv `fmap` {#get CUdevprop.regsPerBlock#} p
    cl <- cIntConv `fmap` {#get CUdevprop.clockRate#} p
    ta <- cIntConv `fmap` {#get CUdevprop.textureAlign#} p

    [t1,t2,t3] <- peekArrayWith cIntConv 3 =<< {#get CUdevprop.maxThreadsDim#} p
    [g1,g2,g3] <- peekArrayWith cIntConv 3 =<< {#get CUdevprop.maxGridSize#} p

    return CUDevProp
      {
        cuMaxThreadsPerBlock = tb,
        cuMaxBlockSize       = (t1,t2,t3),
        cuMaxGridSize        = (g1,g2,g3),
        cuSharedMemPerBlock  = sm,
        cuTotalConstMem      = cm,
        cuWarpSize           = ws,
        cuMemPitch           = mp,
        cuRegsPerBlock       = rb,
        cuClockRate          = cl,
        cuTextureAlignment   = ta
      }
#endif


-- |
-- Possible option flags for CUDA initialisation. Dummy instance until the API
-- exports actual option values.
--
data InitFlag
instance Enum InitFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = case x of {}
  fromEnum x = case x of {}
#endif


--------------------------------------------------------------------------------
-- Initialisation
--------------------------------------------------------------------------------

-- |
-- Initialise the CUDA driver API. This must be called before any other
-- driver function.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3>
--
{-# INLINEABLE initialise #-}
initialise :: [InitFlag] -> IO ()
initialise !flags = nothingIfOk =<< cuInit flags <* enable_constructors

{-# INLINE enable_constructors #-}
{# fun unsafe enable_constructors { } -> `()' #}

{-# INLINE cuInit #-}
{# fun unsafe cuInit
  { combineBitMasks `[InitFlag]' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Device Management
--------------------------------------------------------------------------------

-- |
-- Return the compute compatibility revision supported by the device
--
{-# INLINEABLE capability #-}
capability :: Device -> IO Compute
#if CUDA_VERSION >= 5000
capability !dev =
  Compute <$> attribute dev ComputeCapabilityMajor
          <*> attribute dev ComputeCapabilityMinor
#else
-- Deprecated as of CUDA-5.0
--
capability !dev =
  (\(!s,!a,!b) -> resultIfOk (s,Compute a b)) =<< cuDeviceComputeCapability dev

{-# INLINE cuDeviceComputeCapability #-}
{# fun unsafe cuDeviceComputeCapability
  { alloca-   `Int'    peekIntConv*
  , alloca-   `Int'    peekIntConv*
  , useDevice `Device'              } -> `Status' cToEnum #}
#endif


-- |
-- Return a handle to the compute device at the given ordinal.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb>
--
{-# INLINEABLE device #-}
device :: Int -> IO Device
device !d = resultIfOk =<< cuDeviceGet d

{-# INLINE cuDeviceGet #-}
{# fun unsafe cuDeviceGet
  { alloca-  `Device' dev*
  , cIntConv `Int'           } -> `Status' cToEnum #}
  where dev = liftM Device . peek


-- |
-- Return the selected attribute for the given device.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266>
--
{-# INLINEABLE attribute #-}
attribute :: Device -> DeviceAttribute -> IO Int
attribute !d !a = resultIfOk =<< cuDeviceGetAttribute a d

{-# INLINE cuDeviceGetAttribute #-}
{# fun unsafe cuDeviceGetAttribute
  { alloca-   `Int'             peekIntConv*
  , cFromEnum `DeviceAttribute'
  , useDevice `Device'                       } -> `Status' cToEnum #}


-- |
-- Return the number of device with compute capability > 1.0.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74>
--
{-# INLINEABLE count #-}
count :: IO Int
count = resultIfOk =<< cuDeviceGetCount

{-# INLINE cuDeviceGetCount #-}
{# fun unsafe cuDeviceGetCount
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- The identifying name of the device.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f>
--
{-# INLINEABLE name #-}
name :: Device -> IO String
name !d = resultIfOk =<< cuDeviceGetName d

{-# INLINE cuDeviceGetName #-}
{# fun unsafe cuDeviceGetName
  { allocaS-  `String'& peekS*
  , useDevice `Device'         } -> `Status' cToEnum #}
  where
    len       = 512
    allocaS a = allocaBytes len $ \p -> a (p, cIntConv len)
    peekS s _ = peekCString s


-- | Returns a UUID for the device
--
-- Requires CUDA-9.2
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4>
--
{-# INLINE uuid #-}
uuid :: Device -> IO UUID
#if CUDA_VERSION < 9020
uuid _ = requireSDK 'uuid 9.2
#else
uuid !dev =
  allocaBytes 16 $ \ptr -> do
    cuDeviceGetUuid ptr dev
    unpack ptr
  where
    {-# INLINE cuDeviceGetUuid #-}
    {# fun unsafe cuDeviceGetUuid
      {           `Ptr ()'
      , useDevice `Device'
      } -> `()' checkStatus*- #}

    {-# INLINE unpack #-}
    unpack :: Ptr () -> IO UUID
    unpack !p = do
      let !q = castPtr p
      a <- word <$> peekElemOff q  0 <*> peekElemOff q  1 <*> peekElemOff q  2 <*> peekElemOff q  3
      b <- word <$> peekElemOff q  4 <*> peekElemOff q  5 <*> peekElemOff q  6 <*> peekElemOff q  7
      c <- word <$> peekElemOff q  8 <*> peekElemOff q  9 <*> peekElemOff q 10 <*> peekElemOff q 11
      d <- word <$> peekElemOff q 12 <*> peekElemOff q 13 <*> peekElemOff q 14 <*> peekElemOff q 15
      return $! fromWords a b c d

    {-# INLINE word #-}
    word :: Word8 -> Word8 -> Word8 -> Word8 -> Word32
    word !a !b !c !d
      =  (fromIntegral a `shiftL` 24)
     .|. (fromIntegral b `shiftL` 16)
     .|. (fromIntegral c `shiftL`  8)
     .|. (fromIntegral d            )
#endif


-- |
-- Return the properties of the selected device
--
{-# INLINEABLE props #-}
props :: Device -> IO DeviceProperties
props !d = do

#if CUDA_VERSION < 5000
  -- Old versions of the CUDA API used the separate cuDeviceGetProperties
  -- function to probe some properties, and cuDeviceGetAttribute for
  -- others. As of CUDA-5.0, the former was deprecated and its
  -- functionality subsumed by the latter, which we use below.
  --
  p   <- resultIfOk =<< cuDeviceGetProperties d
  let cm = cuTotalConstMem p
      sm = cuSharedMemPerBlock p
      rb = cuRegsPerBlock p
      ws = cuWarpSize p
      tb = cuMaxThreadsPerBlock p
      bs = cuMaxBlockSize p
      gs = cuMaxGridSize p
      cl = cuClockRate p
      mp = cuMemPitch p
      ta = cuTextureAlignment p
#else
  cm  <- fromIntegral <$> attribute d TotalConstantMemory
  sm  <- fromIntegral <$> attribute d SharedMemoryPerBlock
  mp  <- fromIntegral <$> attribute d MaxPitch
  ta  <- fromIntegral <$> attribute d TextureAlignment
  cl  <- attribute d ClockRate
  ws  <- attribute d WarpSize
  rb  <- attribute d RegistersPerBlock
  tb  <- attribute d MaxThreadsPerBlock
  bs  <- (,,) <$> attribute d MaxBlockDimX
              <*> attribute d MaxBlockDimY
              <*> attribute d MaxBlockDimZ
  gs  <- (,,) <$> attribute d MaxGridDimX
              <*> attribute d MaxGridDimY
              <*> attribute d MaxGridDimZ
#endif

  -- The rest of the properties.
  --
  n   <- name d
  cc  <- capability d
  gm  <- totalMem d
  pc  <- attribute d MultiprocessorCount
  md  <- toEnum `fmap` attribute d ComputeMode
  ov  <- toBool `fmap` attribute d GpuOverlap
  ke  <- toBool `fmap` attribute d KernelExecTimeout
  tg  <- toBool `fmap` attribute d Integrated
  hm  <- toBool `fmap` attribute d CanMapHostMemory
#if CUDA_VERSION >= 3000
  ck  <- toBool `fmap` attribute d ConcurrentKernels
  ee  <- toBool `fmap` attribute d EccEnabled
  u1  <- attribute d MaximumTexture1dWidth
  u21 <- attribute d MaximumTexture2dWidth
  u22 <- attribute d MaximumTexture2dHeight
  u31 <- attribute d MaximumTexture3dWidth
  u32 <- attribute d MaximumTexture3dHeight
  u33 <- attribute d MaximumTexture3dDepth
#endif
#if CUDA_VERSION >= 4000
  ae  <- attribute d AsyncEngineCount
  l2  <- attribute d L2CacheSize
  tm  <- attribute d MaxThreadsPerMultiprocessor
  mw  <- attribute d GlobalMemoryBusWidth
  mc  <- attribute d MemoryClockRate
  pb  <- attribute d PciBusId
  pd  <- attribute d PciDeviceId
  pm  <- attribute d PciDomainId
  ua  <- toBool `fmap` attribute d UnifiedAddressing
  tcc <- toBool `fmap` attribute d TccDriver
#endif
#if CUDA_VERSION >= 5050
  sp  <- toBool `fmap` attribute d StreamPrioritiesSupported
#endif
#if CUDA_VERSION >= 6000
  gl1 <- toBool `fmap` attribute d GlobalL1CacheSupported
  ll1 <- toBool `fmap` attribute d LocalL1CacheSupported
  mm  <- toBool `fmap` attribute d ManagedMemory
  mg  <- toBool `fmap` attribute d MultiGpuBoard
  mid <- attribute d MultiGpuBoardGroupId
#endif

  return DeviceProperties
    {
      deviceName                        = n
    , computeCapability                 = cc
    , totalGlobalMem                    = gm
    , totalConstMem                     = cm
    , sharedMemPerBlock                 = sm
    , regsPerBlock                      = rb
    , warpSize                          = ws
    , maxThreadsPerBlock                = tb
    , maxBlockSize                      = bs
    , maxGridSize                       = gs
    , clockRate                         = cl
    , multiProcessorCount               = pc
    , memPitch                          = mp
    , textureAlignment                  = ta
    , computeMode                       = md
    , deviceOverlap                     = ov
    , kernelExecTimeoutEnabled          = ke
    , integrated                        = tg
    , canMapHostMemory                  = hm
#if CUDA_VERSION >= 3000
    , concurrentKernels                 = ck
    , eccEnabled                        = ee
    , maxTextureDim1D                   = u1
    , maxTextureDim2D                   = (u21,u22)
    , maxTextureDim3D                   = (u31,u32,u33)
#endif
#if CUDA_VERSION >= 4000
    , asyncEngineCount                  = ae
    , cacheMemL2                        = l2
    , maxThreadsPerMultiProcessor       = tm
    , memBusWidth                       = mw
    , memClockRate                      = mc
    , pciInfo                           = PCI pb pd pm
    , tccDriverEnabled                  = tcc
    , unifiedAddressing                 = ua
#endif
#if CUDA_VERSION >= 5050
    , streamPriorities                  = sp
#endif
#if CUDA_VERSION >= 6000
    , globalL1Cache                     = gl1
    , localL1Cache                      = ll1
    , managedMemory                     = mm
    , multiGPUBoard                     = mg
    , multiGPUBoardGroupID              = mid
#endif
    }

#if CUDA_VERSION < 5000
-- Deprecated as of CUDA-5.0
{-# INLINE cuDeviceGetProperties #-}
{# fun unsafe cuDeviceGetProperties
  { alloca-   `CUDevProp' peek*
  , useDevice `Device'          } -> `Status' cToEnum #}
#endif


-- |
-- The total memory available on the device (bytes).
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d>
--
{-# INLINEABLE totalMem #-}
totalMem :: Device -> IO Int64
totalMem !d = resultIfOk =<< cuDeviceTotalMem d

{-# INLINE cuDeviceTotalMem #-}
{# fun unsafe cuDeviceTotalMem
  { alloca-   `Int64'  peekIntConv*
  , useDevice `Device'              } -> `Status' cToEnum #}

