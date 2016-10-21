{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Stream
-- Copyright : [2009..2014] Trevor L. McDonell
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

#include "cbits/stubs.h"
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Types
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad                                    ( liftM )
import Control.Exception                                ( throwIO )


--------------------------------------------------------------------------------
-- Functions
--------------------------------------------------------------------------------

-- |
-- Create a new asynchronous stream
--
{-# INLINEABLE create #-}
create :: IO Stream
create = resultIfOk =<< cudaStreamCreate

{-# INLINE cudaStreamCreate #-}
{# fun unsafe cudaStreamCreate
  { alloca- `Stream' peekStream* } -> `Status' cToEnum #}


-- |
-- Destroy and clean up an asynchronous stream
--
{-# INLINEABLE destroy #-}
destroy :: Stream -> IO ()
destroy !s = nothingIfOk =<< cudaStreamDestroy s

{-# INLINE cudaStreamDestroy #-}
{# fun unsafe cudaStreamDestroy
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Determine if all operations in a stream have completed
--
{-# INLINEABLE finished #-}
finished :: Stream -> IO Bool
finished !s =
  cudaStreamQuery s >>= \rv -> do
  case rv of
      Success  -> return True
      NotReady -> return False
      _        -> throwIO (ExitCode rv)

{-# INLINE cudaStreamQuery #-}
{# fun unsafe cudaStreamQuery
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Block until all operations in a Stream have been completed
--
{-# INLINEABLE block #-}
block :: Stream -> IO ()
block !s = nothingIfOk =<< cudaStreamSynchronize s

{-# INLINE cudaStreamSynchronize #-}
{# fun cudaStreamSynchronize
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- The main execution stream (0)
--
-- {-# INLINE defaultStream #-}
-- defaultStream :: Stream
-- #if CUDART_VERSION < 3010
-- defaultStream = Stream 0
-- #else
-- defaultStream = Stream nullPtr
-- #endif

--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

{-# INLINE peekStream #-}
peekStream :: Ptr {#type cudaStream_t#} -> IO Stream
#if CUDART_VERSION < 3010
peekStream = liftM Stream . peekIntConv
#else
peekStream = liftM Stream . peek
#endif

