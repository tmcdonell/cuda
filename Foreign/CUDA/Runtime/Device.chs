{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Device
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Device management routines
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Device
  (
    ComputeMode(..), DeviceFlag(..), DeviceProperties(..),

    -- ** Device management
    choose, get, count, props, set, setFlags, setOrder
  )
  where

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Internal.Offsets

-- System
import Foreign
import Foreign.C

#c
typedef struct cudaDeviceProp   cudaDeviceProp;

typedef enum
{
    cudaDeviceFlagScheduleAuto  = cudaDeviceScheduleAuto,
    cudaDeviceFlagScheduleSpin  = cudaDeviceScheduleSpin,
    cudaDeviceFlagScheduleYield = cudaDeviceScheduleYield,
    cudaDeviceFlagBlockingSync  = cudaDeviceBlockingSync,
    cudaDeviceFlagMapHost       = cudaDeviceMapHost
} cudaDeviceFlags;
#endc


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

{# pointer *cudaDeviceProp as ^ foreign -> DeviceProperties nocode #}

-- |
-- The compute mode the device is currently in
--
{# enum cudaComputeMode as ComputeMode { }
    with prefix="cudaComputeMode" deriving (Eq, Show) #}

-- |
-- Device execution flags
--
{# enum cudaDeviceFlags as DeviceFlag { }
    with prefix="cudaDeviceFlag" deriving (Eq, Show) #}

-- |
-- The properties of a compute device
--
data DeviceProperties = DeviceProperties
  {
    deviceName               :: String,         -- ^ Identifier
    computeCapability        :: Double,         -- ^ Supported compute capability
    totalGlobalMem           :: Int64,          -- ^ Available global memory on the device in bytes
    totalConstMem            :: Int64,          -- ^ Available constant memory on the device in bytes
    sharedMemPerBlock        :: Int64,          -- ^ Available shared memory per block in bytes
    regsPerBlock             :: Int,            -- ^ 32-bit registers per block
    warpSize                 :: Int,            -- ^ Warp size in threads
    maxThreadsPerBlock       :: Int,            -- ^ Max number of threads per block
    maxThreadsDim            :: (Int,Int,Int),  -- ^ Max size of each dimension of a block
    maxGridSize              :: (Int,Int,Int),  -- ^ Max size of each dimension of a grid
    clockRate                :: Int,            -- ^ Clock frequency in kilohertz
    multiProcessorCount      :: Int,            -- ^ Number of multiprocessors on the device
    memPitch                 :: Int64,          -- ^ Max pitch in bytes allowed by memory copies
    textureAlignment         :: Int64,          -- ^ Alignment requirement for textures
    computeMode              :: ComputeMode,
    deviceOverlap            :: Bool,           -- ^ Device can concurrently copy memory and execute a kernel
    kernelExecTimeoutEnabled :: Bool,           -- ^ Whether there is a runtime limit on kernels
    integrated               :: Bool,           -- ^ As opposed to discrete
    canMapHostMemory         :: Bool            -- ^ Device can use pinned memory
  }
  deriving (Show)


instance Storable DeviceProperties where
  sizeOf _    = {#sizeof cudaDeviceProp#}
  alignment _ = alignment (undefined :: Ptr ())

  peek p      = do
    gm <- cIntConv     `fmap` {#get cudaDeviceProp.totalGlobalMem#} p
    sm <- cIntConv     `fmap` {#get cudaDeviceProp.sharedMemPerBlock#} p
    rb <- cIntConv     `fmap` {#get cudaDeviceProp.regsPerBlock#} p
    ws <- cIntConv     `fmap` {#get cudaDeviceProp.warpSize#} p
    mp <- cIntConv     `fmap` {#get cudaDeviceProp.memPitch#} p
    tb <- cIntConv     `fmap` {#get cudaDeviceProp.maxThreadsPerBlock#} p
    cl <- cIntConv     `fmap` {#get cudaDeviceProp.clockRate#} p
    cm <- cIntConv     `fmap` {#get cudaDeviceProp.totalConstMem#} p
    v1 <- fromIntegral `fmap` {#get cudaDeviceProp.major#} p
    v2 <- fromIntegral `fmap` {#get cudaDeviceProp.minor#} p
    ta <- cIntConv     `fmap` {#get cudaDeviceProp.textureAlignment#} p
    ov <- cToBool      `fmap` {#get cudaDeviceProp.deviceOverlap#} p
    pc <- cIntConv     `fmap` {#get cudaDeviceProp.multiProcessorCount#} p
    ke <- cToBool      `fmap` {#get cudaDeviceProp.kernelExecTimeoutEnabled#} p
    tg <- cToBool      `fmap` {#get cudaDeviceProp.integrated#} p
    hm <- cToBool      `fmap` {#get cudaDeviceProp.canMapHostMemory#} p
    md <- cToEnum      `fmap` {#get cudaDeviceProp.computeMode#} p

    --
    -- C->Haskell returns the wrong type when accessing static arrays in
    -- structs, returning the dereferenced element but with a Ptr type. Work
    -- around this with manual pointer arithmetic...
    --
    n            <- peekCString (p `plusPtr` devNameOffset)
    (t1:t2:t3:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxThreadDimOffset :: Ptr CInt)
    (g1:g2:g3:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxGridSizeOffset  :: Ptr CInt)

    return DeviceProperties
      {
        deviceName               = n,
        computeCapability        = v1 + v2 / max 10 (10^ ((ceiling . logBase 10) v2 :: Int)),
        totalGlobalMem           = gm,
        totalConstMem            = cm,
        sharedMemPerBlock        = sm,
        regsPerBlock             = rb,
        warpSize                 = ws,
        maxThreadsPerBlock       = tb,
        maxThreadsDim            = (t1,t2,t3),
        maxGridSize              = (g1,g2,g3),
        clockRate                = cl,
        multiProcessorCount      = pc,
        memPitch                 = mp,
        textureAlignment         = ta,
        computeMode              = md,
        deviceOverlap            = ov,
        kernelExecTimeoutEnabled = ke,
        integrated               = tg,
        canMapHostMemory         = hm
      }


--------------------------------------------------------------------------------
-- Functions
--------------------------------------------------------------------------------

-- |
-- Select the compute device which best matches the given criteria
--
choose     :: DeviceProperties -> IO Int
choose dev =  resultIfOk =<< cudaChooseDevice dev

{# fun unsafe cudaChooseDevice
  { alloca-      `Int'              peekIntConv*
  , withDevProp* `DeviceProperties'              } -> `Status' cToEnum #}
  where
      withDevProp = with


-- |
-- Returns which device is currently being used
--
get :: IO Int
get =  resultIfOk =<< cudaGetDevice

{# fun unsafe cudaGetDevice
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Returns the number of devices available for execution, with compute
-- capability >= 1.0
--
count :: IO Int
count =  resultIfOk =<< cudaGetDeviceCount

{# fun unsafe cudaGetDeviceCount
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Return information about the selected compute device
--
props   :: Int -> IO DeviceProperties
props n =  resultIfOk =<< cudaGetDeviceProperties n

{# fun unsafe cudaGetDeviceProperties
  { alloca- `DeviceProperties' peek*
  ,         `Int'                    } -> `Status' cToEnum #}


-- |
-- Set device to be used for GPU execution
--
set   :: Int -> IO ()
set n =  nothingIfOk =<< cudaSetDevice n

{# fun unsafe cudaSetDevice
  { `Int' } -> `Status' cToEnum #}


-- |
-- Set flags to be used for device executions
--
setFlags   :: [DeviceFlag] -> IO ()
setFlags f =  nothingIfOk =<< cudaSetDeviceFlags (combineBitMasks f)

{# fun unsafe cudaSetDeviceFlags
  { `Int' } -> `Status' cToEnum #}


-- |
-- Set list of devices for CUDA execution in priority order
--
setOrder   :: [Int] -> IO ()
setOrder l =  nothingIfOk =<< cudaSetValidDevices l (length l)

{# fun unsafe cudaSetValidDevices
  { withArrayIntConv* `[Int]'
  ,                   `Int'   } -> `Status' cToEnum #}
  where
      withArrayIntConv = withArray . map cIntConv

