{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE RecordWildCards          #-}
{-# LANGUAGE TemplateHaskell          #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
#ifdef USE_EMPTY_CASE
{-# LANGUAGE EmptyCase                #-}
#endif
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Device
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- Device management routines
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Device (

  -- * Device Management
  Device, DeviceFlag(..), DeviceProperties(..), Compute(..), ComputeMode(..),
  choose, get, count, props, set, setFlags, setOrder, reset, sync,

  -- * Peer Access
  PeerFlag,
  accessible, add, remove,

  -- * Cache Configuration
  Limit(..),
  getLimit, setLimit

) where

#include "cbits/stubs.h"
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Analysis.Device
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Control.Applicative
import Foreign
import Foreign.C
import Prelude

#c
typedef struct cudaDeviceProp   cudaDeviceProp;

typedef enum
{
    cudaDeviceFlagScheduleAuto    = cudaDeviceScheduleAuto,
    cudaDeviceFlagScheduleSpin    = cudaDeviceScheduleSpin,
    cudaDeviceFlagScheduleYield   = cudaDeviceScheduleYield,
    cudaDeviceFlagBlockingSync    = cudaDeviceBlockingSync,
    cudaDeviceFlagMapHost         = cudaDeviceMapHost,
#if CUDART_VERSION >= 3000
    cudaDeviceFlagLMemResizeToMax = cudaDeviceLmemResizeToMax
#endif
} cudaDeviceFlags;
#endc


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A device identifier
--
type Device = Int

{# pointer *cudaDeviceProp as ^ foreign -> DeviceProperties nocode #}

-- |
-- Device execution flags
--
{# enum cudaDeviceFlags as DeviceFlag { }
    with prefix="cudaDeviceFlag" deriving (Eq, Show, Bounded) #}


instance Storable DeviceProperties where
  sizeOf _    = {#sizeof cudaDeviceProp#}
  alignment _ = alignment (undefined :: Ptr ())

  poke _ _    = error "no instance for Foreign.Storable.poke DeviceProperties"
  peek p      = do
    deviceName                    <- peekCString =<< {#get cudaDeviceProp.name#} p
    computeCapability             <- Compute <$> (fromIntegral <$> {#get cudaDeviceProp.major#} p)
                                             <*> (fromIntegral <$> {#get cudaDeviceProp.minor#} p)
    totalGlobalMem                <- cIntConv <$> {#get cudaDeviceProp.totalGlobalMem#} p
    sharedMemPerBlock             <- cIntConv <$> {#get cudaDeviceProp.sharedMemPerBlock#} p
    regsPerBlock                  <- cIntConv <$> {#get cudaDeviceProp.regsPerBlock#} p
    warpSize                      <- cIntConv <$> {#get cudaDeviceProp.warpSize#} p
    memPitch                      <- cIntConv <$> {#get cudaDeviceProp.memPitch#} p
    maxThreadsPerBlock            <- cIntConv <$> {#get cudaDeviceProp.maxThreadsPerBlock#} p
    clockRate                     <- cIntConv <$> {#get cudaDeviceProp.clockRate#} p
    totalConstMem                 <- cIntConv <$> {#get cudaDeviceProp.totalConstMem#} p
    textureAlignment              <- cIntConv <$> {#get cudaDeviceProp.textureAlignment#} p
    deviceOverlap                 <- cToBool  <$> {#get cudaDeviceProp.deviceOverlap#} p
    multiProcessorCount           <- cIntConv <$> {#get cudaDeviceProp.multiProcessorCount#} p
    kernelExecTimeoutEnabled      <- cToBool  <$> {#get cudaDeviceProp.kernelExecTimeoutEnabled#} p
    integrated                    <- cToBool  <$> {#get cudaDeviceProp.integrated#} p
    canMapHostMemory              <- cToBool  <$> {#get cudaDeviceProp.canMapHostMemory#} p
    computeMode                   <- cToEnum  <$> {#get cudaDeviceProp.computeMode#} p
#if CUDART_VERSION >= 3000
    concurrentKernels             <- cToBool  <$> {#get cudaDeviceProp.concurrentKernels#} p
    maxTextureDim1D               <- cIntConv <$> {#get cudaDeviceProp.maxTexture1D#} p
#endif
#if CUDART_VERSION >= 3000 && CUDART_VERSION < 3010
    -- XXX: not visible from CUDA runtime API 3.0 (only accessible from the driver API)
    let eccEnabled = False
#endif
#if CUDART_VERSION >= 3010
    eccEnabled                    <- cToBool  <$> {#get cudaDeviceProp.ECCEnabled#} p
#endif
#if CUDART_VERSION >= 4000
    asyncEngineCount              <- cIntConv <$> {#get cudaDeviceProp.asyncEngineCount#} p
    cacheMemL2                    <- cIntConv <$> {#get cudaDeviceProp.l2CacheSize#} p
    maxThreadsPerMultiProcessor   <- cIntConv <$> {#get cudaDeviceProp.maxThreadsPerMultiProcessor#} p
    memBusWidth                   <- cIntConv <$> {#get cudaDeviceProp.memoryBusWidth#} p
    memClockRate                  <- cIntConv <$> {#get cudaDeviceProp.memoryClockRate#} p
    pciInfo                       <- PCI <$> (cIntConv <$> {#get cudaDeviceProp.pciBusID#} p)
                                         <*> (cIntConv <$> {#get cudaDeviceProp.pciDeviceID#} p)
                                         <*> (cIntConv <$> {#get cudaDeviceProp.pciDomainID#} p)
    tccDriverEnabled              <- cToBool <$> {#get cudaDeviceProp.tccDriver#} p
    unifiedAddressing             <- cToBool <$> {#get cudaDeviceProp.unifiedAddressing#} p
#endif
    [t1,t2,t3]                    <- peekArrayWith cIntConv 3 =<< {#get cudaDeviceProp.maxThreadsDim#} p
    [g1,g2,g3]                    <- peekArrayWith cIntConv 3 =<< {#get cudaDeviceProp.maxGridSize#} p
    let maxBlockSize = (t1,t2,t3)
        maxGridSize  = (g1,g2,g3)
#if CUDART_VERSION >= 3000
    [u21,u22]                     <- peekArrayWith cIntConv 2 =<< {#get cudaDeviceProp.maxTexture2D#} p
    [u31,u32,u33]                 <- peekArrayWith cIntConv 3 =<< {#get cudaDeviceProp.maxTexture3D#} p
    let maxTextureDim2D = (u21,u22)
        maxTextureDim3D = (u31,u32,u33)
#endif
#if CUDART_VERSION >= 5050
    streamPriorities              <- cToBool  <$> {#get cudaDeviceProp.streamPrioritiesSupported#} p
#endif
#if CUDART_VERSION >= 6000
    globalL1Cache                 <- cToBool  <$> {#get cudaDeviceProp.globalL1CacheSupported#} p
    localL1Cache                  <- cToBool  <$> {#get cudaDeviceProp.localL1CacheSupported#} p
    managedMemory                 <- cToBool  <$> {#get cudaDeviceProp.managedMemory#} p
    multiGPUBoard                 <- cToBool  <$> {#get cudaDeviceProp.isMultiGpuBoard#} p
    multiGPUBoardGroupID          <- cIntConv <$> {#get cudaDeviceProp.multiGpuBoardGroupID#} p
#endif
#if CUDART_VERSION >= 8000
#if CUDART_VERSION == 8000
    -- XXX: Not visible from the CUDA runtime API 8.0 (only accessible from the driver API)
    let preemption = False
#else
    preemption                    <- cToBool  <$> {#get cudaDeviceProp.computePreemptionSupported#} p
#endif
    singleToDoublePerfRatio       <- cIntConv <$> {#get cudaDeviceProp.singleToDoublePrecisionPerfRatio#} p
#endif
#if CUDART_VERSION >= 9000
    cooperativeLaunch             <- cToBool <$> {#get cudaDeviceProp.cooperativeLaunch#} p
    cooperativeLaunchMultiDevice  <- cToBool <$> {#get cudaDeviceProp.cooperativeMultiDeviceLaunch#} p
#endif

    return DeviceProperties{..}


--------------------------------------------------------------------------------
-- Device Management
--------------------------------------------------------------------------------

-- |
-- Select the compute device which best matches the given criteria
--
{-# INLINEABLE choose #-}
choose :: DeviceProperties -> IO Device
choose !dev = resultIfOk =<< cudaChooseDevice dev

{-# INLINE cudaChooseDevice #-}
{# fun unsafe cudaChooseDevice
  { alloca-      `Int'              peekIntConv*
  , withDevProp* `DeviceProperties'              } -> `Status' cToEnum #}
  where
      withDevProp = with


-- |
-- Returns which device is currently being used
--
{-# INLINEABLE get #-}
get :: IO Device
get = resultIfOk =<< cudaGetDevice

{-# INLINE cudaGetDevice #-}
{# fun unsafe cudaGetDevice
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Returns the number of devices available for execution, with compute
-- capability >= 1.0
--
{-# INLINEABLE count #-}
count :: IO Int
count = resultIfOk =<< cudaGetDeviceCount

{-# INLINE cudaGetDeviceCount #-}
{# fun unsafe cudaGetDeviceCount
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Return information about the selected compute device
--
{-# INLINEABLE props #-}
props :: Device -> IO DeviceProperties
props !n = resultIfOk =<< cudaGetDeviceProperties_v2 n

{-# INLINE cudaGetDeviceProperties_v2 #-}
{# fun unsafe cudaGetDeviceProperties_v2
  { alloca- `DeviceProperties' peek*
  ,         `Int'                    } -> `Status' cToEnum #}


-- |
-- Set device to be used for GPU execution
--
{-# INLINEABLE set #-}
set :: Device -> IO ()
set !n = nothingIfOk =<< cudaSetDevice n

{-# INLINE cudaSetDevice #-}
{# fun unsafe cudaSetDevice
  { `Int' } -> `Status' cToEnum #}


-- |
-- Set flags to be used for device executions
--
{-# INLINEABLE setFlags #-}
setFlags :: [DeviceFlag] -> IO ()
setFlags !f = nothingIfOk =<< cudaSetDeviceFlags (combineBitMasks f)

{-# INLINE cudaSetDeviceFlags #-}
{# fun unsafe cudaSetDeviceFlags
  { `Int' } -> `Status' cToEnum #}


-- |
-- Set list of devices for CUDA execution in priority order
--
{-# INLINEABLE setOrder #-}
setOrder :: [Device] -> IO ()
setOrder !l = nothingIfOk =<< cudaSetValidDevices l (length l)

{-# INLINE cudaSetValidDevices #-}
{# fun unsafe cudaSetValidDevices
  { withArrayIntConv* `[Int]'
  ,                   `Int'   } -> `Status' cToEnum #}
  where
      withArrayIntConv = withArray . map cIntConv

-- |
-- Block until the device has completed all preceding requested tasks. Returns
-- an error if one of the tasks fails.
--
{-# INLINEABLE sync #-}
sync :: IO ()
#if CUDART_VERSION < 4000
{-# INLINE cudaThreadSynchronize #-}
sync = nothingIfOk =<< cudaThreadSynchronize
{# fun cudaThreadSynchronize { } -> `Status' cToEnum #}
#else
{-# INLINE cudaDeviceSynchronize #-}
sync = nothingIfOk =<< cudaDeviceSynchronize
{# fun cudaDeviceSynchronize { } -> `Status' cToEnum #}
#endif

-- |
-- Explicitly destroys and cleans up all runtime resources associated with the
-- current device in the current process. Any subsequent API call will
-- reinitialise the device.
--
-- Note that this function will reset the device immediately. It is the caller’s
-- responsibility to ensure that the device is not being accessed by any other
-- host threads from the process when this function is called.
--
{-# INLINEABLE reset #-}
reset :: IO ()
#if CUDART_VERSION >= 4000
{-# INLINE cudaDeviceReset #-}
reset = nothingIfOk =<< cudaDeviceReset
{# fun unsafe cudaDeviceReset { } -> `Status' cToEnum #}
#else
{-# INLINE cudaThreadExit #-}
reset = nothingIfOk =<< cudaThreadExit
{# fun unsafe cudaThreadExit  { } -> `Status' cToEnum #}
#endif


--------------------------------------------------------------------------------
-- Peer Access
--------------------------------------------------------------------------------

-- |
-- Possible option values for direct peer memory access
--
data PeerFlag
instance Enum PeerFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("PeerFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif

-- |
-- Queries if the first device can directly access the memory of the second. If
-- direct access is possible, it can then be enabled with 'add'. Requires
-- cuda-4.0.
--
{-# INLINEABLE accessible #-}
accessible :: Device -> Device -> IO Bool
#if CUDART_VERSION < 4000
accessible _ _        = requireSDK 'accessible 4.0
#else
accessible !dev !peer = resultIfOk =<< cudaDeviceCanAccessPeer dev peer

{-# INLINE cudaDeviceCanAccessPeer #-}
{# fun unsafe cudaDeviceCanAccessPeer
  { alloca-  `Bool'   peekBool*
  , cIntConv `Device'
  , cIntConv `Device'           } -> `Status' cToEnum #}
#endif

-- |
-- If the devices of both the current and supplied contexts support unified
-- addressing, then enable allocations in the supplied context to be accessible
-- by the current context. Requires cuda-4.0.
--
{-# INLINEABLE add #-}
add :: Device -> [PeerFlag] -> IO ()
#if CUDART_VERSION < 4000
add _ _         = requireSDK 'add 4.0
#else
add !dev !flags = nothingIfOk =<< cudaDeviceEnablePeerAccess dev flags

{-# INLINE cudaDeviceEnablePeerAccess #-}
{# fun unsafe cudaDeviceEnablePeerAccess
  { cIntConv        `Device'
  , combineBitMasks `[PeerFlag]' } -> `Status' cToEnum #}
#endif


-- |
-- Disable direct memory access from the current context to the supplied
-- context. Requires cuda-4.0.
--
{-# INLINEABLE remove #-}
remove :: Device -> IO ()
#if CUDART_VERSION < 4000
remove _    = requireSDK 'remove 4.0
#else
remove !dev = nothingIfOk =<< cudaDeviceDisablePeerAccess dev

{-# INLINE cudaDeviceDisablePeerAccess #-}
{# fun unsafe cudaDeviceDisablePeerAccess
  { cIntConv `Device' } -> `Status' cToEnum #}
#endif


--------------------------------------------------------------------------------
-- Cache Configuration
--------------------------------------------------------------------------------

-- |
-- Device limit flags
--
#if CUDART_VERSION < 3010
data Limit
#else
{# enum cudaLimit as Limit
    { underscoreToCase }
    with prefix="cudaLimit" deriving (Eq, Show) #}
#endif


-- |
-- Query compute 2.0 call stack limits. Requires cuda-3.1.
--
{-# INLINEABLE getLimit #-}
getLimit :: Limit -> IO Int
#if   CUDART_VERSION < 3010
getLimit _  = requireSDK 'getLimit 3.1
#elif CUDART_VERSION < 4000
getLimit !l = resultIfOk =<< cudaThreadGetLimit l

{-# INLINE cudaThreadGetLimit #-}
{# fun unsafe cudaThreadGetLimit
  { alloca-   `Int' peekIntConv*
  , cFromEnum `Limit'            } -> `Status' cToEnum #}
#else
getLimit !l = resultIfOk =<< cudaDeviceGetLimit l

{-# INLINE cudaDeviceGetLimit #-}
{# fun unsafe cudaDeviceGetLimit
  { alloca-   `Int' peekIntConv*
  , cFromEnum `Limit'            } -> `Status' cToEnum #}
#endif


-- |
-- Set compute 2.0 call stack limits. Requires cuda-3.1.
--
{-# INLINEABLE setLimit #-}
setLimit :: Limit -> Int -> IO ()
#if   CUDART_VERSION < 3010
setLimit _ _   = requireSDK 'setLimit 3.1
#elif CUDART_VERSION < 4000
setLimit !l !n = nothingIfOk =<< cudaThreadSetLimit l n

{-# INLINE cudaThreadSetLimit #-}
{# fun unsafe cudaThreadSetLimit
  { cFromEnum `Limit'
  , cIntConv  `Int'   } -> `Status' cToEnum #}
#else
setLimit !l !n = nothingIfOk =<< cudaDeviceSetLimit l n

{-# INLINE cudaDeviceSetLimit #-}
{# fun unsafe cudaDeviceSetLimit
  { cFromEnum `Limit'
  , cIntConv  `Int'   } -> `Status' cToEnum #}
#endif

