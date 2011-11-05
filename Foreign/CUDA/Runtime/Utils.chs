{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Utils
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Utility functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Utils (runtimeVersion, driverVersion)
  where

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C


-- |
-- Return the version number of the installed CUDA driver
--
runtimeVersion :: IO Int
runtimeVersion =  resultIfOk =<< cudaRuntimeGetVersion

{# fun unsafe cudaRuntimeGetVersion
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Return the version number of the installed CUDA runtime
--
driverVersion :: IO Int
driverVersion =  resultIfOk =<< cudaDriverGetVersion

{# fun unsafe cudaDriverGetVersion
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}

