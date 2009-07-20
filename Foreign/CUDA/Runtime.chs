{-
 - Haskell bindings to the CUDA library
 -
 - This uses the higher-level "C for CUDA" interface and runtime API, which
 - itself is implemented with the lower-level driver API.
 -}

{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.Runtime
  (
    --
    -- Thread management
    --
    threadSynchronize,

    --
    -- Device management
    --
    chooseDevice,
    getDevice,
    getDeviceCount,
    getDeviceProperties,
    setDevice,
    setDeviceFlags,
    setValidDevices,

    --
    -- Memory management
    --
    malloc, mallocPitch,
    memcpy, memcpyAsync,
    memset,


    --
    -- Stream management
    --
    streamCreate,
    streamDestroy,
    streamQuery,
    streamSynchronize
  ) where

import Foreign.CUDA.Types
import Foreign.CUDA.Utils

import Foreign.CUDA.Internal.C2HS hiding (malloc)


#include <cuda_runtime_api.h>
{#context lib="cudart" prefix="cuda"#}

{# pointer *cudaDeviceProp as ^ foreign -> DeviceProperties nocode #}


--------------------------------------------------------------------------------
-- Thread Management
--------------------------------------------------------------------------------

--
-- Block until the device has completed all preceding requests
--
threadSynchronize :: IO (Maybe String)
threadSynchronize =  nothingIfOk `fmap` cudaThreadSynchronize

{# fun unsafe cudaThreadSynchronize
    { } -> `Status' cToEnum #}

--------------------------------------------------------------------------------
-- Device Management
--------------------------------------------------------------------------------

--
-- Select compute device which best matches criteria
--
chooseDevice     :: DeviceProperties -> IO (Either String Int)
chooseDevice dev =  resultIfOk `fmap` cudaChooseDevice dev

{# fun unsafe cudaChooseDevice
    { alloca-      `Int'              peekIntConv* ,
      withDevProp* `DeviceProperties'              } -> `Status' cToEnum #}
    where
        withDevProp = with

--
-- Returns which device is currently being used
--
getDevice :: IO (Either String Int)
getDevice =  resultIfOk `fmap` cudaGetDevice

{# fun unsafe cudaGetDevice
    { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}

--
-- Returns the number of compute-capable devices
--
getDeviceCount :: IO (Either String Int)
getDeviceCount =  resultIfOk `fmap` cudaGetDeviceCount

{# fun unsafe cudaGetDeviceCount
    { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}

--
-- Return information about the selected compute device
--
getDeviceProperties   :: Int -> IO (Either String DeviceProperties)
getDeviceProperties n =  resultIfOk `fmap` cudaGetDeviceProperties n

{# fun unsafe cudaGetDeviceProperties
    { alloca- `DeviceProperties' peek* ,
              `Int'                    } -> `Status' cToEnum #}

--
-- Set device to be used for GPU execution
--
setDevice   :: Int -> IO (Maybe String)
setDevice n =  nothingIfOk `fmap` cudaSetDevice n

{# fun unsafe cudaSetDevice
    { `Int' } -> `Status' cToEnum #}

--
-- Set flags to be used for device executions
--
setDeviceFlags   :: [DeviceFlags] -> IO (Maybe String)
setDeviceFlags f =  nothingIfOk `fmap` cudaSetDeviceFlags (combineBitMasks f)

{# fun unsafe cudaSetDeviceFlags
    { `Int' } -> `Status' cToEnum #}

--
-- Set list of devices for CUDA execution in priority order
--
setValidDevices   :: [Int] -> IO (Maybe String)
setValidDevices l =  nothingIfOk `fmap` cudaSetValidDevices l (length l)

{# fun unsafe cudaSetValidDevices
    { withArrayIntConv* `[Int]' ,
                        `Int'   } -> `Status' cToEnum #}
    where
        withArrayIntConv = withArray . map cIntConv


--------------------------------------------------------------------------------
-- Memory Management
--------------------------------------------------------------------------------

--
-- Allocate the specified number of bytes in linear memory on the device. The
-- memory is suitably aligned for any kind of variable, and is not cleared.
--
-- If the allocation is successful, it is marked to be automatically released
-- when no longer needed.
--
malloc       :: Integer -> IO (Either String DevicePtr)
malloc bytes = do
    (rv,ptr) <- cudaMalloc bytes
    case rv of
        Success -> doAutoRelease cudaFree_ >>= \fp ->
                   newDevicePtr fp ptr >>= (return.Right)
        _       -> return.Left $ getErrorString rv

{# fun unsafe cudaMalloc
    { alloca-  `Ptr ()'  peek* ,
      cIntConv `Integer'       } -> `Status' cToEnum #}

--
-- Allocate pitched memory on the device
--
mallocPitch              :: Integer -> Integer -> IO (Either String (DevicePtr,Integer))
mallocPitch width height =  do
    (rv,ptr,pitch) <- cudaMallocPitch width height
    case rv of
        Success -> doAutoRelease cudaFree_ >>= \fp ->
                   newDevicePtr fp ptr >>= \dp -> return.Right $ (dp,pitch)
        _       -> return.Left $ getErrorString rv

{# fun unsafe cudaMallocPitch
    { alloca-  `Ptr ()' peek*         ,
      alloca-  `Integer' peekIntConv* ,
      cIntConv `Integer'              ,
      cIntConv `Integer'              } -> `Status' cToEnum #}

--
-- Free previously allocated memory on the device
--
cudaFree_   :: Ptr () -> IO ()
cudaFree_ p =  throwIf_ (/= Success) (getErrorString) (cudaFree p)

{# fun unsafe cudaFree
    { id `Ptr ()' } -> `Status' cToEnum #}

foreign import ccall "wrapper"
    doAutoRelease   :: (Ptr () -> IO ()) -> IO (FunPtr (Ptr () -> IO ()))

--
-- Copy data between host and device
--
memcpy :: DevicePtr -> DevicePtr -> Integer -> CopyDirection -> IO (Maybe String)
memcpy dst src bytes dir =  nothingIfOk `fmap` cudaMemcpy dst src bytes dir

{# fun unsafe cudaMemcpy
    { withDevicePtr* `DevicePtr'     ,
      withDevicePtr* `DevicePtr'     ,
      cIntConv       `Integer'       ,
      cFromEnum      `CopyDirection' } -> `Status' cToEnum #}

--
-- Copy data between host and device asynchronously
--
memcpyAsync :: DevicePtr -> DevicePtr -> Integer -> CopyDirection -> Stream -> IO (Maybe String)
memcpyAsync dst src bytes dir stream =  nothingIfOk `fmap` cudaMemcpyAsync dst src bytes dir stream

{# fun unsafe cudaMemcpyAsync
    { withDevicePtr* `DevicePtr'     ,
      withDevicePtr* `DevicePtr'     ,
      cIntConv       `Integer'       ,
      cFromEnum      `CopyDirection' ,
      cIntConv       `Stream'        } -> `Status' cToEnum #}

--
-- Initialize device memory to a given value
--
memset                  :: DevicePtr -> Integer -> Int -> IO (Maybe String)
memset ptr symbol bytes =  nothingIfOk `fmap` cudaMemset ptr bytes symbol

{# fun unsafe cudaMemset
    { withDevicePtr* `DevicePtr' ,
                     `Int'       ,
      cIntConv       `Integer'   } -> `Status' cToEnum #}

--------------------------------------------------------------------------------
-- Stream Management
--------------------------------------------------------------------------------

--
-- Create an asynchronous stream
--
streamCreate :: IO (Either String Stream)
streamCreate =  resultIfOk `fmap` cudaStreamCreate

{# fun unsafe cudaStreamCreate
    { alloca- `Stream' peek* } -> `Status' cToEnum #}

--
-- Destroy and clean up an asynchronous stream
--
streamDestroy   :: Stream -> IO (Maybe String)
streamDestroy s =  nothingIfOk `fmap` cudaStreamDestroy s

{# fun unsafe cudaStreamDestroy
    { cIntConv `Stream' } -> `Status' cToEnum #}

--
-- Determine if all operations in a stream have completed
--
streamQuery   :: Stream -> IO (Either String Bool)
streamQuery s = cudaStreamQuery s >>= \rv -> do
    return $ case rv of
        Success  -> Right True
        NotReady -> Right False
        _        -> Left (getErrorString rv)

{# fun unsafe cudaStreamQuery
    { cIntConv `Stream' } -> `Status' cToEnum #}

--
-- Block until all operations have been completed
--
streamSynchronize   :: Stream -> IO (Maybe String)
streamSynchronize s =  nothingIfOk `fmap` cudaStreamSynchronize s

{# fun unsafe cudaStreamSynchronize
    { cIntConv `Stream' } -> `Status' cToEnum #}

