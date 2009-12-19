{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Thread
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Thread management routines
--
--------------------------------------------------------------------------------


module Foreign.CUDA.Runtime.Thread
  (
    sync, exit
  )
  where

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign.C


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

