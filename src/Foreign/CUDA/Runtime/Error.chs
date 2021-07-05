{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE DeriveDataTypeable       #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Error
-- Copyright : [2009..2020] Trevor L. McDonell
-- License   : BSD
--
-- Error handling functions
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Error (

  Status(..), CUDAException(..),

  cudaError, describe, requireSDK,
  resultIfOk, nothingIfOk, checkStatus,

) where

-- Friends
import Foreign.CUDA.Internal.C2HS
import Text.Show.Describe

-- System
import Control.Exception
import Data.Typeable
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import System.IO.Unsafe
import Text.Printf

#include "cbits/stubs.h"
{# context lib="cudart" #}


--------------------------------------------------------------------------------
-- Return Status
--------------------------------------------------------------------------------

-- |
-- Return codes from API functions
--
{# enum cudaError as Status
    { cudaSuccess as Success }
    with prefix="cudaError" deriving (Eq, Show) #}

--------------------------------------------------------------------------------
-- Exceptions
--------------------------------------------------------------------------------

data CUDAException
  = ExitCode  Status
  | UserError String
  deriving Typeable

instance Exception CUDAException

instance Show CUDAException where
  showsPrec _ (ExitCode  s) = showString ("CUDA Exception: " ++ describe s)
  showsPrec _ (UserError s) = showString ("CUDA Exception: " ++ s)


-- |
-- Raise a 'CUDAException' in the IO Monad
--
cudaError :: String -> IO a
cudaError s = throwIO (UserError s)

-- |
-- A specially formatted error message
--
requireSDK :: Name -> Double -> IO a
requireSDK n v = cudaError $ printf "'%s' requires at least cuda-%3.1f\n" (show n) v


--------------------------------------------------------------------------------
-- Helper Functions
--------------------------------------------------------------------------------

-- |
-- Return the descriptive string associated with a particular error code
--
instance Describe Status where
    describe = cudaGetErrorString

-- Logically, this must be a pure function, returning a pointer to a statically
-- defined string constant.
--
{# fun pure unsafe cudaGetErrorString
    { cFromEnum `Status' } -> `String' #}


-- |
-- Return the results of a function on successful execution, otherwise return
-- the error string associated with the return code
--
{-# INLINE resultIfOk #-}
resultIfOk :: (Status, a) -> IO a
resultIfOk (status, !result) =
    case status of
        Success -> return  result
        _       -> throwIO (ExitCode status)


-- |
-- Return the error string associated with an unsuccessful return code,
-- otherwise Nothing
--
{-# INLINE nothingIfOk #-}
nothingIfOk :: Status -> IO ()
nothingIfOk status =
    case status of
        Success -> return  ()
        _       -> throwIO (ExitCode status)

{-# INLINE checkStatus #-}
checkStatus :: CInt -> IO ()
checkStatus = nothingIfOk . cToEnum

