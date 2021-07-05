{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Utils
-- Copyright : [2009..2020] Trevor L. McDonell
-- License   : BSD
--
-- Utility functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Utils (

  driverVersion,
  libraryVersion,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C


-- |
-- Return the version number of the installed CUDA driver.
--
{-# INLINEABLE driverVersion #-}
driverVersion :: IO Int
driverVersion =  resultIfOk =<< cuDriverGetVersion

{-# INLINE cuDriverGetVersion #-}
{# fun unsafe cuDriverGetVersion
  { alloca- `Int' peekIntConv* } -> `Status' cToEnum #}


-- |
-- Return the version number of the CUDA library (API) that this package was
-- compiled against.
--
{-# INLINEABLE libraryVersion #-}
libraryVersion :: Int
libraryVersion = {#const CUDA_VERSION #}

