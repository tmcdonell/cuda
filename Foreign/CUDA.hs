--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Top level bindings. By default, expose the C-for-CUDA runtime API bindings,
-- as they are slightly more user friendly.
--
--------------------------------------------------------------------------------

module Foreign.CUDA
  (
    module Foreign.CUDA.Runtime
  )
  where

import Foreign.CUDA.Runtime

