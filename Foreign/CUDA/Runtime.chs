{-
 - Haskell bindings to the CUDA library
 -
 - This uses the higher-level "C for CUDA" interface and runtime API, which
 - itself is implemented with the lower-level driver API.
 -}

{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.Runtime where

import C2HS
import Control.Monad

#include "cbits/stubs.h"
#include <cuda_runtime_api.h>
{#context lib="cudart" prefix="cuda"#}


--------------------------------------------------------------------------------
-- Enumerations
--------------------------------------------------------------------------------

--
-- Error Codes
--
{# enum cudaError as Result
    { cudaSuccess as Success }
    with prefix="cudaError" deriving (Eq, Show) #}

--
-- Memory copy
--
{# enum cudaMemcpyKind as CopyDirection {}
    with prefix="cudaMemcpy" deriving (Eq, Show) #}

--
-- Compute mode
--
{# enum ComputeMode {}
    with prefix="cudaComputeMode" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Error Handling
--------------------------------------------------------------------------------

{# fun unsafe GetErrorString as ^
    { cFromEnum `Result' } -> `String' #}

--------------------------------------------------------------------------------
-- Device Management
--------------------------------------------------------------------------------

--
-- Store all device properties
--
data DeviceProperties = DeviceProperties
  {
    deviceName               :: String,
    computeCapability        :: (Int,Int),
    totalGlobalMem           :: Integer,
    totalConstMem            :: Integer,
    sharedMemPerBlock        :: Integer,
    regsPerBlock             :: Int,
    warpSize                 :: Int,
    maxThreadsPerBlock       :: Int,
    maxThreadsDim            :: (Int,Int,Int),
    maxGridSize              :: (Int,Int,Int),
    clockRate                :: Int,
    multiProcessorCount      :: Int,
    memPitch                 :: Integer,
    textureAlignment         :: Integer,
    computeMode              :: ComputeMode,
    deviceOverlap            :: Bool,
    kernelExecTimeoutEnabled :: Bool,
    integrated               :: Bool,
    canMapHostMemory         :: Bool
  }
  deriving (Show)

instance Storable DeviceProperties where
    sizeOf _    = {#sizeof cudaDeviceProp#}
    alignment _ = alignment (undefined::Int)

    peek p      = do
        gm <- liftM cIntConv $ {#get cudaDeviceProp.totalGlobalMem#} p
        sm <- liftM cIntConv $ {#get cudaDeviceProp.sharedMemPerBlock#} p
        rb <- liftM cIntConv $ {#get cudaDeviceProp.regsPerBlock#} p
        ws <- liftM cIntConv $ {#get cudaDeviceProp.warpSize#} p
        mp <- liftM cIntConv $ {#get cudaDeviceProp.memPitch#} p
        tb <- liftM cIntConv $ {#get cudaDeviceProp.maxThreadsPerBlock#} p
        cl <- liftM cIntConv $ {#get cudaDeviceProp.clockRate#} p
        cm <- liftM cIntConv $ {#get cudaDeviceProp.totalConstMem#} p
        v1 <- liftM cIntConv $ {#get cudaDeviceProp.major#} p
        v2 <- liftM cIntConv $ {#get cudaDeviceProp.minor#} p
        ta <- liftM cIntConv $ {#get cudaDeviceProp.textureAlignment#} p
        ov <- liftM cToBool  $ {#get cudaDeviceProp.deviceOverlap#} p
        pc <- liftM cIntConv $ {#get cudaDeviceProp.multiProcessorCount#} p
        ke <- liftM cToBool  $ {#get cudaDeviceProp.kernelExecTimeoutEnabled#} p
        tg <- liftM cToBool  $ {#get cudaDeviceProp.integrated#} p
        hm <- liftM cToBool  $ {#get cudaDeviceProp.canMapHostMemory#} p
        md <- liftM cToEnum  $ {#get cudaDeviceProp.computeMode#} p

        with p $ \p' -> do
          n            <- {#get cudaDeviceProp.name#} p' >>= peekCString
          (t1:t2:t3:_) <- liftM (map cIntConv) $ {#get cudaDeviceProp.maxThreadsDim#} p' >>= (peekArray 3)
          (g1:g2:g3:_) <- liftM (map cIntConv) $ {#get cudaDeviceProp.maxGridSize#} p'   >>= (peekArray 3)

          return DeviceProperties
            {
              deviceName               = n,
              computeCapability        = (v1,v2),
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


{# pointer *cudaDeviceProp as ^ foreign -> DeviceProperties nocode #}

--
-- Returns the number of compute-capable devices
--
{# fun unsafe GetDeviceCount as ^
    { alloca- `Int' peekIntConv* } -> `Result' cToEnum #}

--
-- Return information about the selected compute device
--
{# fun unsafe GetDeviceProperties as ^
    { alloca- `DeviceProperties' peek*,
              `Int'                   } -> `Result' cToEnum #}

