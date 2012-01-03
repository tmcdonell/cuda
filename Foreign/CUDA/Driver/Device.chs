{-# LANGUAGE CPP, ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Device
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Device management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Device (

  -- * Device Management
  Device(..), -- should be exported abstractly
  DeviceProperties(..), DeviceAttribute(..), ComputeMode(..), InitFlag,
  initialise, capability, device, attribute, count, name, props, totalMem

) where

#include <cuda.h>
#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Analysis.Device
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

newtype Device = Device { useDevice :: {# type CUdevice #}}


-- |
-- Device attributes
--
{# enum CUdevice_attribute as DeviceAttribute
    { underscoreToCase }
    with prefix="CU_DEVICE_ATTRIBUTE" deriving (Eq, Show) #}

{# pointer *CUdevprop as ^ foreign -> CUDevProp nocode #}


--
-- Properties of the compute device (internal helper)
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

    (t1:t2:t3:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxThreadDimOffset' :: Ptr CInt)
    (g1:g2:g3:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxGridSizeOffset'  :: Ptr CInt)

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


-- |
-- Possible option flags for CUDA initialisation. Dummy instance until the API
-- exports actual option values.
--
data InitFlag
instance Enum InitFlag where


--------------------------------------------------------------------------------
-- Initialisation
--------------------------------------------------------------------------------

-- |
-- Initialise the CUDA driver API. Must be called before any other driver
-- function.
--
initialise :: [InitFlag] -> IO ()
initialise flags = nothingIfOk =<< cuInit flags

{# fun unsafe cuInit
  { combineBitMasks `[InitFlag]' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Device Management
--------------------------------------------------------------------------------

-- |
-- Return the compute compatibility revision supported by the device
--
capability :: Device -> IO Double
capability dev =
  (\(s,a,b) -> resultIfOk (s,cap a b)) =<< cuDeviceComputeCapability dev
  where
    cap a 0 = fromIntegral a
    cap a b = let a' = fromIntegral a in
              let b' = fromIntegral b in
              a' + b' / max 10 (10^ ((ceiling . logBase 10) b' :: Int))

{# fun unsafe cuDeviceComputeCapability
  { alloca-   `Int'    peekIntConv*
  , alloca-   `Int'    peekIntConv*
  , useDevice `Device'              } -> `Status' cToEnum #}


-- |
-- Return a device handle
--
device :: Int -> IO Device
device d = resultIfOk =<< cuDeviceGet d

{# fun unsafe cuDeviceGet
  { alloca-  `Device' dev*
  , cIntConv `Int'           } -> `Status' cToEnum #}
  where dev = liftM Device . peek


-- |
-- Return the selected attribute for the given device
--
attribute :: Device -> DeviceAttribute -> IO Int
attribute d a = resultIfOk =<< cuDeviceGetAttribute a d

{# fun unsafe cuDeviceGetAttribute
  { alloca-   `Int'             peekIntConv*
  , cFromEnum `DeviceAttribute'
  , useDevice `Device'                       } -> `Status' cToEnum #}


-- |
-- Return the number of device with compute capability > 1.0
--
count :: IO Int
count = resultIfOk =<< cuDeviceGetCount

{# fun unsafe cuDeviceGetCount
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Name of the device
--
name :: Device -> IO String
name d = resultIfOk =<< cuDeviceGetName d

{# fun unsafe cuDeviceGetName
  { allocaS-  `String'& peekS*
  , useDevice `Device'         } -> `Status' cToEnum #}
  where
    len       = 512
    allocaS a = allocaBytes len $ \p -> a (p, cIntConv len)
    peekS s _ = peekCString s


-- |
-- Return the properties of the selected device
--

-- Annoyingly, the driver API requires several different functions to extract
-- all device properties that are part of a single structure in the runtime API
--
props :: Device -> IO DeviceProperties
props d = do
  p   <- resultIfOk =<< cuDeviceGetProperties d

  -- And the remaining properties
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

  return DeviceProperties
    {
      deviceName                        = n,
      computeCapability                 = cc,
      totalGlobalMem                    = gm,
      totalConstMem                     = cuTotalConstMem p,
      sharedMemPerBlock                 = cuSharedMemPerBlock p,
      regsPerBlock                      = cuRegsPerBlock p,
      warpSize                          = cuWarpSize p,
      maxThreadsPerBlock                = cuMaxThreadsPerBlock p,
      maxBlockSize                      = cuMaxBlockSize p,
      maxGridSize                       = cuMaxGridSize p,
      clockRate                         = cuClockRate p,
      multiProcessorCount               = pc,
      memPitch                          = cuMemPitch p,
      textureAlignment                  = cuTextureAlignment p,
      computeMode                       = md,
      deviceOverlap                     = ov,
#if CUDA_VERSION >= 3000
      concurrentKernels                 = ck,
      eccEnabled                        = ee,
      maxTextureDim1D                   = u1,
      maxTextureDim2D                   = (u21,u22),
      maxTextureDim3D                   = (u31,u32,u33),
#endif
#if CUDA_VERSION >= 4000
      asyncEngineCount                  = ae,
      cacheMemL2                        = l2,
      maxThreadsPerMultiProcessor       = tm,
      memBusWidth                       = mw,
      memClockRate                      = mc,
      pciInfo                           = PCI pb pd pm,
      tccDriverEnabled                  = tcc,
      unifiedAddressing                 = ua,
#endif
      kernelExecTimeoutEnabled          = ke,
      integrated                        = tg,
      canMapHostMemory                  = hm
    }


{# fun unsafe cuDeviceGetProperties
  { alloca-   `CUDevProp' peek*
  , useDevice `Device'          } -> `Status' cToEnum #}


-- |
-- Total memory available on the device (bytes)
--
totalMem :: Device -> IO Int64
totalMem d = resultIfOk =<< cuDeviceTotalMem d

{# fun unsafe cuDeviceTotalMem
  { alloca-   `Int64'  peekIntConv*
  , useDevice `Device'              } -> `Status' cToEnum #}

