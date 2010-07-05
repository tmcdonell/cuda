{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Thread
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Thread management routines
--
--------------------------------------------------------------------------------

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

module Foreign.CUDA.Runtime.Thread
  (
#if CUDART_VERSION >= 3010
    Limit(..), getLimit, setLimit,
#endif
    sync, exit
  )
  where

-- Friends
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C


#if CUDART_VERSION >= 3010
{# enum cudaLimit as Limit
    { underscoreToCase }
    with prefix="cudaLimit" deriving (Eq, Show) #}
#endif

--------------------------------------------------------------------------------
-- Thread management
--------------------------------------------------------------------------------

-- |
-- Block until the device has completed all preceding requested tasks. Returns
-- an error if one of the tasks fails.
--
sync :: IO ()
sync =  nothingIfOk =<< cudaThreadSynchronize

{# fun unsafe cudaThreadSynchronize
  { } -> `Status' cToEnum #}


-- |
-- Explicitly clean up all runtime related resources associated with the calling
-- host thread. Any subsequent API call re-initialised the environment.
-- Implicitly called on thread exit.
--
exit :: IO ()
exit =  nothingIfOk =<< cudaThreadExit

{# fun unsafe cudaThreadExit
  { } -> `Status' cToEnum #}


#if CUDART_VERSION >= 3010
-- |
-- Query compute 2.0 call stack limits
--
getLimit :: Limit -> IO Int
getLimit l = resultIfOk =<< cudaThreadGetLimit l

-- |
-- Set compute 2.0 call stack limits
--
setLimit :: Limit -> Int -> IO ()
setLimit l n = nothingIfOk =<< cudaThreadSetLimit l n

{# fun unsafe cudaThreadGetLimit
  { alloca-   `Int' peekIntConv*
  , cFromEnum `Limit'            } -> `Status' cToEnum #}

{# fun unsafe cudaThreadSetLimit
  { cFromEnum `Limit'
  , cIntConv  `Int'   } -> `Status' cToEnum #}
#endif

