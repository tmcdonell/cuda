{-# LANGUAGE ForeignFunctionInterface, FlexibleContexts #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Utils
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Utility functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Utils
  (
    driverVersion
  )
  where

#include <cuda.h>
{# context lib="cuda" #}


-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad.Exception.MTL


-- |
-- Unwrap an Either construct, throwing an error for non-right values
--
forceEither :: Either String a -> a
forceEither (Left  s) = error s
forceEither (Right r) = r

-- |
-- Return the version number of the installed CUDA driver
--
--driverVersion :: IO Int
driverVersion :: Throws CUDAException l => EMT l IO Int
driverVersion = resultIfOk =<< lift cuDriverGetVersion

{# fun unsafe cuDriverGetVersion
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}

