{-
 - Utilities
 -}

{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.Utils
  (
    getErrorString, resultIfOk, nothingIfOk
  )
  where

import Foreign.CUDA.Types

import Foreign.CUDA.Internal.C2HS


#include <cuda_runtime_api.h>
{#context lib="cudart" prefix="cuda"#}


--------------------------------------------------------------------------------
-- Error Handling
--------------------------------------------------------------------------------

--
-- Return the descriptive string associated with a particular error code.
-- Logically, this must a pure function, returning a pointer to a statically
-- defined string constant.
--
{# fun pure unsafe GetErrorString as ^
    { cFromEnum `Result' } -> `String' #}


resultIfOk :: (Result, a) -> Either String a
resultIfOk (status,result) =
    case status of
        Success -> Right result
        _       -> Left (getErrorString status)

nothingIfOk :: Result -> Maybe String
nothingIfOk = nothingIf (/= Success) getErrorString

