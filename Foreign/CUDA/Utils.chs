{-
 - Utilities
 -}

module Foreign.CUDA.Utils
  (
    checkError
  )
  where

import Foreign.CUDA.C2HS
import Foreign.CUDA.Types


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


checkError :: (Result, a) -> IO (Either String a)
checkError (status,result) =
    case status of
        Success -> (return.Right) result
        _       -> getErrorString status >>= (return.Left)

