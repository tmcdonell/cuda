--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Module
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- Module management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Module (

  module Foreign.CUDA.Driver.Module.Base,
  module Foreign.CUDA.Driver.Module.Query,

) where

import Foreign.CUDA.Driver.Module.Base  hiding ( JITOptionInternal(..), useModule, jitOptionUnpack, jitTargetOfCompute )
import Foreign.CUDA.Driver.Module.Query

