{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Stream
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Stream management routines
--
--------------------------------------------------------------------------------


module Foreign.CUDA.Stream
  (
    Stream,

    -- ** Stream management
    create,
    destroy,
    finished,
    sync
  )
  where

import Foreign.CUDA.Error
import Foreign.CUDA.Internal.C2HS hiding (malloc)

#include <cuda_runtime_api.h>
{# context lib="cudart" #}


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

type Stream = {# type cudaStream_t #}


--------------------------------------------------------------------------------
-- Functions
--------------------------------------------------------------------------------

-- |
-- Create a new asynchronous stream
--
create :: IO (Either String Stream)
create =  resultIfOk `fmap` cudaStreamCreate

{# fun unsafe cudaStreamCreate
    { alloca- `Stream' peek* } -> `Status' cToEnum #}


-- |
-- Destroy and clean up an asynchronous stream
--
destroy   :: Stream -> IO (Maybe String)
destroy s =  nothingIfOk `fmap` cudaStreamDestroy s

{# fun unsafe cudaStreamDestroy
    { cIntConv `Stream' } -> `Status' cToEnum #}


-- |
-- Determine if all operations in a stream have completed
--
finished   :: Stream -> IO (Either String Bool)
finished s = cudaStreamQuery s >>= \rv -> do
    return $ case rv of
        Success  -> Right True
        NotReady -> Right False
        _        -> Left (describe rv)

{# fun unsafe cudaStreamQuery
    { cIntConv `Stream' } -> `Status' cToEnum #}


-- |
-- Block until all operations in a stream have been completed
--
sync   :: Stream -> IO (Maybe String)
sync s =  nothingIfOk `fmap` cudaStreamSynchronize s

{# fun unsafe cudaStreamSynchronize
    { cIntConv `Stream' } -> `Status' cToEnum #}

