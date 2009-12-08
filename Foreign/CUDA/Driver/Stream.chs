{-# LANGUAGE ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Stream
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Stream management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Stream
  (
    Stream, StreamFlag,
    withStream,
    create, destroy, finished, block
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A processing stream
--
{# pointer *CUstream as Stream foreign newtype #}
withStream :: Stream -> (Ptr Stream -> IO a) -> IO a

newStream :: IO Stream
newStream = Stream `fmap` mallocForeignPtrBytes (sizeOf (undefined :: Ptr ()))


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
create :: [StreamFlag] -> IO (Either String Stream)
create flags =
  newStream              >>= \st -> withStream st $ \s ->
  cuStreamCreate s flags >>= \rv ->
    return $ case nothingIfOk rv of
      Nothing -> Right st
      Just e  -> Left e

{# fun unsafe cuStreamCreate
  { id              `Ptr Stream'
  , combineBitMasks `[StreamFlag]' } -> `Status' cToEnum #}

-- |
-- Destroy a stream
--
destroy :: Stream -> IO (Maybe String)
destroy st = withStream st $ \s -> (nothingIfOk `fmap` cuStreamDestroy s)

{# fun unsafe cuStreamDestroy
  { castPtr `Ptr Stream' } -> `Status' cToEnum #}


-- |
-- Check if all operations in the stream have completed
--
finished :: Stream -> IO (Either String Bool)
finished st =
  withStream st $ \s -> cuStreamQuery s >>= \rv ->
  return $ case rv of
    Success  -> Right True
    NotReady -> Right False
    _        -> Left (describe rv)

{# fun unsafe cuStreamQuery
  { castPtr `Ptr Stream' } -> `Status' cToEnum #}


-- |
-- Wait until the device has completed all operations in the Stream
--
block :: Stream -> IO (Maybe String)
block st = withStream st $ \s -> (nothingIfOk `fmap` cuStreamSynchronize s)

{# fun unsafe cuStreamSynchronize
  { castPtr `Ptr Stream' } -> `Status' cToEnum #}

