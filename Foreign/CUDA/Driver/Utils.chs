{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Utils
-- Copyright : (c) [2009..2012] Trevor L. McDonell
-- License   : BSD
--
-- Utility functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Utils (driverVersion)
  where

#include "cbits/stubs.h"
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
{-# INLINEABLE driverVersion #-}
driverVersion :: IO Int
driverVersion =  resultIfOk =<< cuDriverGetVersion

{-# INLINE cuDriverGetVersion #-}
{# fun unsafe cuDriverGetVersion
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}

