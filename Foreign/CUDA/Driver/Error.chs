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
describe Success                     = "no error"
describe InvalidValue                = "invalid argument"
describe OutOfMemory                 = "out of memory"
describe NotInitialized              = "driver not initialised"
describe Deinitialized               = "driver deinitialised"
describe NoDevice                    = "no CUDA-capable device is available"
describe InvalidDevice               = "invalid device ordinal"
describe InvalidImage                = "invalid kernel image"
describe InvalidContext              = "invalid context handle"
describe ContextAlreadyCurrent       = "context already current"
describe MapFailed                   = "map failed"
describe UnmapFailed                 = "unmap failed"
describe ArrayIsMapped               = "array is mapped"
describe AlreadyMapped               = "already mapped"
describe NoBinaryForGpu              = "no binary available for this GPU"
describe AlreadyAcquired             = "resource already acquired"
describe NotMapped                   = "not mapped"
describe InvalidSource               = "invalid source"
describe FileNotFound                = "file not found"
describe InvalidHandle               = "invalid handle"
describe NotFound                    = "not found"
describe NotReady                    = "device not ready"
describe LaunchFailed                = "unspecified launch failure"
describe LaunchOutOfResources        = "too many resources requested for launch"
describe LaunchTimeout               = "the launch timed out and was terminated"
describe LaunchIncompatibleTexturing = "launch with incompatible texturing"
describe Unknown                     = "unknown error"


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

