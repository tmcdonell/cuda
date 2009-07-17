{-
 - Utilities
 -}

{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.Utils
  (
    resultIfSuccess, nothingIfSuccess
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
-- Return the descriptive string associated with a particular error code
--
{# fun unsafe GetErrorString as ^
    { cFromEnum `Result' } -> `String' #}


resultIfSuccess :: (Result, a) -> IO (Either String a)
resultIfSuccess (status,result) =
    case status of
        Success -> (return.Right) result
        _       -> getErrorString status >>= (return.Left)


nothingIfSuccess :: Result -> IO (Maybe String)
nothingIfSuccess status =
    case status of
        Success -> return Nothing
        _       -> getErrorString status >>= (return.Just)

