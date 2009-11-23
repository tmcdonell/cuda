{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Utils
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Utility functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Utils
  (
    forceEither,
    runtimeVersion,
    driverVersion
  )
  where

import Foreign
import Foreign.C

import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

#include <cuda_runtime_api.h>
{# context lib="cudart" #}


-- |
-- Unwrap an Either construct, throwing an error for non-right values
--
forceEither :: Either String a -> a
forceEither (Left  s) = error s
forceEither (Right r) = r


-- |
-- Return the version number of the installed CUDA driver
--
runtimeVersion :: IO (Either String Int)
runtimeVersion =  resultIfOk `fmap` cudaRuntimeGetVersion

{# fun unsafe cudaRuntimeGetVersion
    { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Return the version number of the installed CUDA runtime
--
driverVersion :: IO (Either String Int)
driverVersion =  resultIfOk `fmap` cudaDriverGetVersion

{# fun unsafe cudaDriverGetVersion
    { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}

