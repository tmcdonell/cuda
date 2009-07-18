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
    -- Device management
    --
    getDeviceCount, getDeviceProperties,
    setDevice,

    --
    -- Memory management
    --
    malloc,

    --
    -- Stream management
    --
    streamCreate, streamDestroy, streamQuery, streamSynchronize
  ) where

import Foreign.CUDA.Types
import Foreign.CUDA.Utils

import Foreign.CUDA.Internal.C2HS hiding (malloc)


#include <cuda_runtime_api.h>
{#context lib="cudart" prefix="cuda"#}

{# pointer *cudaDeviceProp as ^ foreign -> DeviceProperties nocode #}

--------------------------------------------------------------------------------
-- Device Management
--------------------------------------------------------------------------------

--
-- Returns the number of compute-capable devices
--
getDeviceCount :: IO (Either String Int)
getDeviceCount =  resultIfOk `fmap` cudaGetDeviceCount

{# fun unsafe cudaGetDeviceCount
    { alloca- `Int' peekIntConv* } -> `Result' cToEnum #}

--
-- Return information about the selected compute device
--
getDeviceProperties   :: Int -> IO (Either String DeviceProperties)
getDeviceProperties n =  resultIfOk `fmap` cudaGetDeviceProperties n

{# fun unsafe cudaGetDeviceProperties
    { alloca- `DeviceProperties' peek*,
              `Int'                   } -> `Result' cToEnum #}

--
-- Set device to be used for GPU execution
--
setDevice   :: Int -> IO (Maybe String)
setDevice n =  nothingIfOk `fmap` cudaSetDevice n

{# fun unsafe cudaSetDevice
    { `Int' } -> `Result' cToEnum #}

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
    { alloca-  `Ptr ()'  peek*,
      cIntConv `Integer'      } -> `Result' cToEnum #}


--
-- Free previously allocated memory on the device
--
cudaFree_   :: Ptr () -> IO ()
cudaFree_ p =  throwIf_ (/= Success) (getErrorString) (cudaFree p)

{# fun unsafe cudaFree
    { id `Ptr ()' } -> `Result' cToEnum #}

foreign import ccall "wrapper"
    doAutoRelease   :: (Ptr () -> IO ()) -> IO (FunPtr (Ptr () -> IO ()))


--------------------------------------------------------------------------------
-- Stream Management
--------------------------------------------------------------------------------

--
-- Create an asynchronous stream
--
streamCreate :: IO (Either String Stream)
streamCreate =  resultIfOk `fmap` cudaStreamCreate

{# fun unsafe cudaStreamCreate
    { alloca- `Stream' peek* } -> `Result' cToEnum #}

--
-- Destroy and clean up an asynchronous stream
--
streamDestroy   :: Stream -> IO (Maybe String)
streamDestroy s =  nothingIfOk `fmap` cudaStreamDestroy s

{# fun unsafe cudaStreamDestroy
    { cIntConv `Stream' } -> `Result' cToEnum #}

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
    { cIntConv `Stream' } -> `Result' cToEnum #}

--
-- Block until all operations have been completed
--
streamSynchronize   :: Stream -> IO (Maybe String)
streamSynchronize s =  nothingIfOk `fmap` cudaStreamSynchronize s

{# fun unsafe cudaStreamSynchronize
    { cIntConv `Stream' } -> `Result' cToEnum #}

