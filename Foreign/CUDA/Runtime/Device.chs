{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Device
-- Copyright : (c) [2009..2012] Trevor L. McDonell
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
import Foreign.CUDA.Internal.Offsets

-- System
import Foreign
import Foreign.C

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
    with prefix="cudaDeviceFlag" deriving (Eq, Show) #}


instance Storable DeviceProperties where
  sizeOf _    = {#sizeof cudaDeviceProp#}
  alignment _ = alignment (undefined :: Ptr ())

  poke _ _    = error "no instance for Foreign.Storable.poke DeviceProperties"
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
#if CUDART_VERSION >= 3000
    ck <- cToBool      `fmap` {#get cudaDeviceProp.concurrentKernels#} p
    u1 <- cIntConv     `fmap` {#get cudaDeviceProp.maxTexture1D#} p
#endif
#if CUDART_VERSION >= 3010
    ee <- cToBool      `fmap` {#get cudaDeviceProp.ECCEnabled#} p
#endif
#if CUDART_VERSION >= 4000
    ae <- cIntConv     `fmap` {#get cudaDeviceProp.asyncEngineCount#} p
    l2 <- cIntConv     `fmap` {#get cudaDeviceProp.l2CacheSize#} p
    tm <- cIntConv     `fmap` {#get cudaDeviceProp.maxThreadsPerMultiProcessor#} p
    mw <- cIntConv     `fmap` {#get cudaDeviceProp.memoryBusWidth#} p
    mc <- cIntConv     `fmap` {#get cudaDeviceProp.memoryClockRate#} p
    pb <- cIntConv     `fmap` {#get cudaDeviceProp.pciBusID#} p
    pd <- cIntConv     `fmap` {#get cudaDeviceProp.pciDeviceID#} p
    pm <- cIntConv     `fmap` {#get cudaDeviceProp.pciDomainID#} p
    tc <- cToBool      `fmap` {#get cudaDeviceProp.tccDriver#} p
    ua <- cToBool      `fmap` {#get cudaDeviceProp.unifiedAddressing#} p
#endif

    --
    -- C->Haskell returns the wrong type when accessing static arrays in
    -- structs, returning the dereferenced element but with a Ptr type. Work
    -- around this with manual pointer arithmetic...
    --
    n            <- peekCString (p `plusPtr` devNameOffset)
    (t1:t2:t3:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxThreadDimOffset :: Ptr CInt)
    (g1:g2:g3:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxGridSizeOffset  :: Ptr CInt)
#if CUDART_VERSION >= 3000
    (u21:u22:_)     <- map cIntConv `fmap` peekArray 2 (p `plusPtr` devMaxTexture2DOffset :: Ptr CInt)
    (u31:u32:u33:_) <- map cIntConv `fmap` peekArray 3 (p `plusPtr` devMaxTexture3DOffset :: Ptr CInt)
#endif

    return DeviceProperties
      {
        deviceName                      = n,
        computeCapability               = Compute v1 v2,
        totalGlobalMem                  = gm,
        totalConstMem                   = cm,
        sharedMemPerBlock               = sm,
        regsPerBlock                    = rb,
        warpSize                        = ws,
        maxThreadsPerBlock              = tb,
        maxBlockSize                    = (t1,t2,t3),
        maxGridSize                     = (g1,g2,g3),
        clockRate                       = cl,
        multiProcessorCount             = pc,
        memPitch                        = mp,
        textureAlignment                = ta,
        computeMode                     = md,
        deviceOverlap                   = ov,
#if CUDART_VERSION >= 3000
        concurrentKernels               = ck,
        maxTextureDim1D                 = u1,
        maxTextureDim2D                 = (u21,u22),
        maxTextureDim3D                 = (u31,u32,u33),
#endif
#if CUDART_VERSION >= 3010
        eccEnabled                      = ee,
#endif
#if CUDART_VERSION >= 3000 && CUDART_VERSION < 3010
        -- not visible from runtime API < 3.1
        eccEnabled                      = False,
#endif
#if CUDART_VERSION >= 4000
        asyncEngineCount                = ae,
        cacheMemL2                      = l2,
        maxThreadsPerMultiProcessor     = tm,
        memBusWidth                     = mw,
        memClockRate                    = mc,
        tccDriverEnabled                = tc,
        unifiedAddressing               = ua,
        pciInfo                         = PCI pb pd pm,
#endif
        kernelExecTimeoutEnabled        = ke,
        integrated                      = tg,
        canMapHostMemory                = hm
      }


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
props !n = resultIfOk =<< cudaGetDeviceProperties n

{-# INLINE cudaGetDeviceProperties #-}
{# fun unsafe cudaGetDeviceProperties
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
{# fun unsafe cudaThreadSynchronize { } -> `Status' cToEnum #}
#else
{-# INLINE cudaDeviceSynchronize #-}
sync = nothingIfOk =<< cudaDeviceSynchronize
{# fun unsafe cudaDeviceSynchronize { } -> `Status' cToEnum #}
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

-- |
-- Queries if the first device can directly access the memory of the second. If
-- direct access is possible, it can then be enabled with 'add'. Requires
-- cuda-4.0.
--
{-# INLINEABLE accessible #-}
accessible :: Device -> Device -> IO Bool
#if CUDART_VERSION < 4000
accessible _ _        = requireSDK 4.0 "accessible"
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
add _ _         = requireSDK 4.0 "add"
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
remove _    = requireSDK 4.0 "remove"
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
getLimit _  = requireSDK 3.1 "getLimit"
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
setLimit _ _   = requireSDK 3.1 "setLimit"
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

