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

    --
    -- Memory management
    --
    malloc
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
getDeviceCount = resultIfOk `fmap` cudaGetDeviceCount

{# fun unsafe cudaGetDeviceCount
    { alloca- `Int' peekIntConv* } -> `Result' cToEnum #}

--
-- Return information about the selected compute device
--
getDeviceProperties   :: Int -> IO (Either String DeviceProperties)
getDeviceProperties n  = resultIfOk `fmap` cudaGetDeviceProperties n

{# fun unsafe cudaGetDeviceProperties
    { alloca- `DeviceProperties' peek*,
              `Int'                   } -> `Result' cToEnum #}

--------------------------------------------------------------------------------
-- Memory Management
--------------------------------------------------------------------------------

--
-- Allocate the specified number of bytes in linear memory on the device. The
-- memory is suitably aligned for any kind of variable, and is not cleared.
--
malloc       :: Integer -> IO (Either String DevicePtr)
malloc bytes = do
    (rv,ptr) <- cudaMalloc bytes
    case rv of
        Success -> doAutoRelease cudaFree_ >>= \fp ->
                   newDevicePtr fp ptr >>= (return.Right)
        _       -> return (Left (getErrorString rv))

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
