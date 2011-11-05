{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Stream
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Stream management routines
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Stream (

  -- * Stream Management
  Stream(..),
  create, destroy, finished, block, defaultStream

) where

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad                                    (liftM)


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A processing stream
--
newtype Stream = Stream { useStream :: {# type cudaStream_t #}}
  deriving (Show)


--------------------------------------------------------------------------------
-- Functions
--------------------------------------------------------------------------------

-- |
-- Create a new asynchronous stream
--
create :: IO Stream
create = resultIfOk =<< cudaStreamCreate

{# fun unsafe cudaStreamCreate
  { alloca- `Stream' peekStream* } -> `Status' cToEnum #}


-- |
-- Destroy and clean up an asynchronous stream
--
destroy :: Stream -> IO ()
destroy s = nothingIfOk =<< cudaStreamDestroy s

{# fun unsafe cudaStreamDestroy
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Determine if all operations in a stream have completed
--
finished   :: Stream -> IO Bool
finished s =
  cudaStreamQuery s >>= \rv -> do
  case rv of
      Success  -> return True
      NotReady -> return False
      _        -> resultIfOk (rv,undefined)

{# fun unsafe cudaStreamQuery
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Block until all operations in a Stream have been completed
--
block :: Stream -> IO ()
block s = nothingIfOk =<< cudaStreamSynchronize s

{# fun unsafe cudaStreamSynchronize
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- The main execution stream (0)
--
defaultStream :: Stream
#if CUDART_VERSION < 3010
defaultStream = Stream 0
#else
defaultStream = Stream nullPtr
#endif

--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

peekStream :: Ptr {#type cudaStream_t#} -> IO Stream
#if CUDART_VERSION < 3010
peekStream = liftM Stream . peekIntConv
#else
peekStream = liftM Stream . peek
#endif


