{-# LANGUAGE ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Stream
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Stream management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Stream (

  -- * Stream Management
  Stream(..), StreamFlag,
  create, destroy, finished, block

) where

#include <cuda.h>
#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad                            (liftM)


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A processing stream
--
newtype Stream = Stream { useStream :: {# type CUstream #}}


-- |
-- Possible option flags for stream initialisation. Dummy instance until the API
-- exports actual option values.
--
data StreamFlag

instance Enum StreamFlag where

--------------------------------------------------------------------------------
-- Stream management
--------------------------------------------------------------------------------

-- |
-- Create a new stream
--
create :: [StreamFlag] -> IO Stream
create flags = resultIfOk =<< cuStreamCreate flags

{# fun unsafe cuStreamCreate
  { alloca-         `Stream'       peekStream*
  , combineBitMasks `[StreamFlag]'             } -> `Status' cToEnum #}
  where peekStream = liftM Stream . peek

-- |
-- Destroy a stream
--
destroy :: Stream -> IO ()
destroy st = nothingIfOk =<< cuStreamDestroy st

{# fun unsafe cuStreamDestroy
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Check if all operations in the stream have completed
--
finished :: Stream -> IO Bool
finished st =
  cuStreamQuery st >>= \rv ->
  case rv of
    Success  -> return True
    NotReady -> return False
    _        -> resultIfOk (rv,undefined)

{# fun unsafe cuStreamQuery
  { useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Wait until the device has completed all operations in the Stream
--
block :: Stream -> IO ()
block st = nothingIfOk =<< cuStreamSynchronize st

{# fun unsafe cuStreamSynchronize
  { useStream `Stream' } -> `Status' cToEnum #}

