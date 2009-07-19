{-
 - CUDA data types
 -}

{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.Types
  (
    Result(..),
    CopyDirection(..),
    ComputeMode(..),
    DeviceFlags(..),
    DeviceProperties(..),
    Stream,

    DevicePtr, newDevicePtr, withDevicePtr
  ) where


import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Internal.Offsets

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

--
-- Device execution flags
--
{# enum DeviceFlags {}
    with prefix="cudaDeviceFlag" deriving (Eq, Show) #}

--------------------------------------------------------------------------------
-- Device Properties
--------------------------------------------------------------------------------

{# pointer *cudaDeviceProp as ^ foreign -> DeviceProperties nocode #}

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
    alignment _ = alignment (undefined :: Ptr ())

    peek p      = do
        gm <- cIntConv `fmap` {#get cudaDeviceProp.totalGlobalMem#} p
        sm <- cIntConv `fmap` {#get cudaDeviceProp.sharedMemPerBlock#} p
        rb <- cIntConv `fmap` {#get cudaDeviceProp.regsPerBlock#} p
        ws <- cIntConv `fmap` {#get cudaDeviceProp.warpSize#} p
        mp <- cIntConv `fmap` {#get cudaDeviceProp.memPitch#} p
        tb <- cIntConv `fmap` {#get cudaDeviceProp.maxThreadsPerBlock#} p
        cl <- cIntConv `fmap` {#get cudaDeviceProp.clockRate#} p
        cm <- cIntConv `fmap` {#get cudaDeviceProp.totalConstMem#} p
        v1 <- cIntConv `fmap` {#get cudaDeviceProp.major#} p
        v2 <- cIntConv `fmap` {#get cudaDeviceProp.minor#} p
        ta <- cIntConv `fmap` {#get cudaDeviceProp.textureAlignment#} p
        ov <- cToBool  `fmap` {#get cudaDeviceProp.deviceOverlap#} p
        pc <- cIntConv `fmap` {#get cudaDeviceProp.multiProcessorCount#} p
        ke <- cToBool  `fmap` {#get cudaDeviceProp.kernelExecTimeoutEnabled#} p
        tg <- cToBool  `fmap` {#get cudaDeviceProp.integrated#} p
        hm <- cToBool  `fmap` {#get cudaDeviceProp.canMapHostMemory#} p
        md <- cToEnum  `fmap` {#get cudaDeviceProp.computeMode#} p

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


--------------------------------------------------------------------------------
-- Memory Management
--------------------------------------------------------------------------------

newtype DevicePtr = DevicePtr (ForeignPtr ())

withDevicePtr :: DevicePtr -> (Ptr () -> IO b) -> IO b
withDevicePtr (DevicePtr fptr) = withForeignPtr fptr

newDevicePtr      :: FinalizerPtr () -> Ptr () -> IO DevicePtr
newDevicePtr fp p =  newForeignPtr fp p >>= \dp -> return (DevicePtr dp)


--------------------------------------------------------------------------------
-- Stream Management
--------------------------------------------------------------------------------

type Stream = {# type cudaStream_t #}

