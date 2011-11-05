--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Top level bindings to CUDA driver API
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver (module Driver)
  where

import Foreign.CUDA.Ptr                 as Driver
import Foreign.CUDA.Driver.Context      as Driver hiding (device, useContext)
import Foreign.CUDA.Driver.Device       as Driver
import Foreign.CUDA.Driver.Error        as Driver
import Foreign.CUDA.Driver.Exec         as Driver
import Foreign.CUDA.Driver.Marshal      as Driver hiding (useDeviceHandle, peekDeviceHandle)
import Foreign.CUDA.Driver.Module       as Driver
import Foreign.CUDA.Driver.Peer         as Driver
import Foreign.CUDA.Driver.Utils        as Driver

