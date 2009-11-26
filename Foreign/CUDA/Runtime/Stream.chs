{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Stream
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Stream management routines
--
--------------------------------------------------------------------------------


module Foreign.CUDA.Runtime.Stream
  (
    Stream,

    -- ** Stream management
    create,
    destroy,
    finished,
    block
  )
  where

import Foreign
import Foreign.C
import Control.Monad			(liftM)

import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

#include <cuda_runtime_api.h>
{# context lib="cudart" #}


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

newtype Stream = Stream { useStream :: {# type cudaStream_t #}}
  deriving (Show)


--------------------------------------------------------------------------------
-- Functions
--------------------------------------------------------------------------------

-- |
-- Create a new asynchronous stream
--
create :: IO (Either String Stream)
create =  resultIfOk `fmap` cudaStreamCreate

{# fun unsafe cudaStreamCreate
    { alloca- `Stream' stream* } -> `Status' cToEnum #}
    where stream = liftM Stream . peekIntConv


-- |
-- Destroy and clean up an asynchronous stream
--
destroy   :: Stream -> IO (Maybe String)
destroy s =  nothingIfOk `fmap` cudaStreamDestroy s

{# fun unsafe cudaStreamDestroy
    { useStream `Stream' } -> `Status' cToEnum #}


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
    { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Block until all operations in a stream have been completed
--
block   :: Stream -> IO (Maybe String)
block s =  nothingIfOk `fmap` cudaStreamSynchronize s

{# fun unsafe cudaStreamSynchronize
    { useStream `Stream' } -> `Status' cToEnum #}

