{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Utils
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Utility functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Utils (driverVersion)
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C


-- |
-- Return the version number of the installed CUDA driver
--
driverVersion :: IO Int
driverVersion =  resultIfOk =<< cuDriverGetVersion

{# fun unsafe cuDriverGetVersion
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}

