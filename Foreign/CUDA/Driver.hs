--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Top level bindings to CUDA driver API
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver (

  module Foreign.CUDA.Ptr,
  module Foreign.CUDA.Driver.Context,
  module Foreign.CUDA.Driver.Device,
  module Foreign.CUDA.Driver.Error,
  module Foreign.CUDA.Driver.Exec,
  module Foreign.CUDA.Driver.Marshal,
  module Foreign.CUDA.Driver.Module,
  module Foreign.CUDA.Driver.Utils

) where

import Foreign.CUDA.Ptr
import Foreign.CUDA.Driver.Context      hiding ( device, useContext )
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Exec
import Foreign.CUDA.Driver.Marshal      hiding ( useDeviceHandle, peekDeviceHandle )
import Foreign.CUDA.Driver.Module
import Foreign.CUDA.Driver.Utils

