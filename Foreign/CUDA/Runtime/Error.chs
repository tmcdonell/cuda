{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Error
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Error handling functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Error
  (
    Status(..),
    describe,
    resultIfOk,
    nothingIfOk
  )
  where

import Foreign
import Foreign.C

import Foreign.CUDA.Internal.C2HS

#include <cuda_runtime_api.h>
{# context lib="cudart" #}


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

--
-- Error Codes
--
{# enum cudaError as Status
    { cudaSuccess as Success }
    with prefix="cudaError" deriving (Eq, Show) #}

--------------------------------------------------------------------------------
-- Functions
--------------------------------------------------------------------------------

-- |
-- Return the descriptive string associated with a particular error code
--
{# fun pure unsafe cudaGetErrorString as describe
    { cFromEnum `Status' } -> `String' #}
--
-- Logically, this must be a pure function, returning a pointer to a statically
-- defined string constant.
--


-- |
-- Return the results of a function on successful execution, otherwise return
-- the error string associated with the return code
--
resultIfOk :: (Status, a) -> Either String a
resultIfOk (status,result) =
    case status of
        Success -> Right result
        _       -> Left (describe status)


-- |
-- Return the error string associated with an unsuccessful return code,
-- otherwise Nothing
--
nothingIfOk :: Status -> Maybe String
nothingIfOk = nothingIf (== Success) describe

