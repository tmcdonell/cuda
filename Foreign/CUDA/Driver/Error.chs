{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Error
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Error handling
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Error
  where


import Foreign.CUDA.Internal.C2HS

#include <cuda.h>
{# context lib="cuda" #}


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

--
-- Error Codes
--
{# enum CUresult as Status
    { underscoreToCase
    , CUDA_SUCCESS as Success }
    with prefix="CUDA_ERROR" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Functions
--------------------------------------------------------------------------------

-- |
-- Return a descriptive error string associated with a particular error code
-- XXX: yes, not very descriptive...
--
describe :: Status -> String
describe = show

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

