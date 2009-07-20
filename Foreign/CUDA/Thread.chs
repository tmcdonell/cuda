{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Thread
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Thread management routines
--
--------------------------------------------------------------------------------


module Foreign.CUDA.Thread
  (
    exit,
    sync
  )
  where

import Foreign.CUDA.Error
import Foreign.CUDA.Internal.C2HS

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

-- |
-- Block until the device has completed all preceding requested tasks. Returns
-- an error if one of the tasks fails.
--
sync :: IO (Maybe String)
sync =  nothingIfOk `fmap` cudaThreadSynchronize

{# fun unsafe cudaThreadSynchronize
    { } -> `Status' cToEnum #}


-- |
-- Explicitly clean up all runtime related resources associated with the calling
-- host thread. Any subsequent API call re-initialised the environment.
-- Implicitly called on thread exit.
--
exit :: IO (Maybe String)
exit =  nothingIfOk `fmap` cudaThreadExit

{# fun unsafe cudaThreadExit
    { } -> `Status' cToEnum #}

