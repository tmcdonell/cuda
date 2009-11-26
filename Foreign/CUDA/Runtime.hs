--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Top level bindings to the C-for-CUDA runtime API
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime
  (
    module Foreign.CUDA.Runtime.Device,
    module Foreign.CUDA.Runtime.DevicePtr,
    module Foreign.CUDA.Runtime.Marshal,
    module Foreign.CUDA.Runtime.Stream,
    module Foreign.CUDA.Runtime.Utils
  )
  where

import Foreign.CUDA.Runtime.Device
import Foreign.CUDA.Runtime.DevicePtr
import Foreign.CUDA.Runtime.Marshal
import Foreign.CUDA.Runtime.Stream
import Foreign.CUDA.Runtime.Utils

