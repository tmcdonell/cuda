--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Top level bindings to the C-for-CUDA runtime API
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime
  (
    module Foreign.CUDA.Ptr,
    module Foreign.CUDA.Runtime.Device,
    module Foreign.CUDA.Runtime.Error,
    module Foreign.CUDA.Runtime.Exec,
    module Foreign.CUDA.Runtime.Marshal,
    module Foreign.CUDA.Runtime.Thread,
    module Foreign.CUDA.Runtime.Utils
  )
  where

import Foreign.CUDA.Ptr
import Foreign.CUDA.Runtime.Device
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Runtime.Exec
import Foreign.CUDA.Runtime.Marshal
import Foreign.CUDA.Runtime.Thread
import Foreign.CUDA.Runtime.Utils

