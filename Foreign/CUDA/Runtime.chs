{-
 - Haskell bindings to the CUDA library
 -
 - This uses the higher-level "C for CUDA" interface and runtime API, which
 - itself is implemented with the lower-level driver API.
 -}

{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.Runtime
  (
    getDeviceCount, getDeviceProperties
  ) where

import Foreign.CUDA.C2HS
import Foreign.CUDA.Types
import Foreign.CUDA.Utils


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
getDeviceCount = checkError =<< cudaGetDeviceCount

{# fun unsafe cudaGetDeviceCount
    { alloca- `Int' peekIntConv* } -> `Result' cToEnum #}

--
-- Return information about the selected compute device
--
getDeviceProperties   :: Int -> IO (Either String DeviceProperties)
getDeviceProperties n  = checkError =<< cudaGetDeviceProperties n

{# fun unsafe cudaGetDeviceProperties
    { alloca- `DeviceProperties' peek*,
              `Int'                   } -> `Result' cToEnum #}

