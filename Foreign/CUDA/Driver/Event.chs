{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Event
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Event management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Event
  (
    Event, EventFlag(..),
    create, destroy, elapsedTime, query, record, block
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream               (Stream, withStream)

-- System
import Foreign
import Foreign.C


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

{# pointer *CUevent as Event foreign newtype #}
withEvent :: Event -> (Ptr Event -> IO a) -> IO a

newEvent :: IO Event
newEvent = Event `fmap` mallocForeignPtrBytes (sizeOf (undefined :: Ptr ()))


-- |
-- Event creation flags
--
{# enum CUevent_flags as EventFlag
    { underscoreToCase }
    with prefix="CU_EVENT" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Event management
--------------------------------------------------------------------------------

-- |
-- Create a new event
--
create :: [EventFlag] -> IO (Either String Event)
create flags =
  newEvent              >>= \ev -> withEvent ev $ \e ->
  cuEventCreate e flags >>= \rv ->
    return $ case nothingIfOk rv of
      Nothing  -> Right ev
      Just err -> Left err

{# fun unsafe cuEventCreate
  { id              `Ptr Event'
  , combineBitMasks `[EventFlag]' } -> `Status' cToEnum #}


-- |
-- Destroy an event
--
destroy :: Event -> IO (Maybe String)
destroy ev = withEvent ev $ \e -> (nothingIfOk `fmap` cuEventDestroy e)

{# fun unsafe cuEventDestroy
  { castPtr `Ptr Event' } -> `Status' cToEnum #}


-- |
-- Determine the elapsed time (in milliseconds) between two events
--
elapsedTime :: Event -> Event -> IO (Either String Float)
elapsedTime ev1 ev2 =
  withEvent ev1 $ \e1 ->
  withEvent ev2 $ \e2 ->
  cuEventElapsedTime e1 e2 >>= \(rv,tm) ->
    return $ case nothingIfOk rv of
      Nothing  -> Right tm
      Just err -> Left err

{# fun unsafe cuEventElapsedTime
  { alloca- `Float'     peekFloatConv*
  , castPtr `Ptr Event'
  , castPtr `Ptr Event'                } -> `Status' cToEnum #}


-- |
-- Determines if a event has actually been recorded
--
query :: Event -> IO (Either String Bool)
query ev =
  withEvent ev $ \e -> cuEventQuery e >>= \rv ->
  return $ case rv of
    Success  -> Right True
    NotReady -> Right False
    _        -> Left (describe rv)

{# fun unsafe cuEventQuery
  { castPtr `Ptr Event' } -> `Status' cToEnum #}


-- |
-- Record an event once all operations in the current context (or optionally
-- specified stream) have completed. This operation is asynchronous.
--
record :: Event -> Maybe Stream -> IO (Maybe String)
record ev mst =
  withEvent ev $ \e ->
  nothingIfOk `fmap` case mst of
    Just st -> (withStream st $ \s -> cuEventRecord e s)
    Nothing -> cuEventRecord e nullPtr

{# fun unsafe cuEventRecord
  { castPtr `Ptr Event'
  , castPtr `Ptr Stream' } -> `Status' cToEnum #}


-- |
-- Wait until the event has been recorded
--
block :: Event -> IO (Maybe String)
block ev = withEvent ev $ \e -> (nothingIfOk `fmap` cuEventSynchronize e)

{# fun unsafe cuEventSynchronize
  { castPtr `Ptr Event' } -> `Status' cToEnum #}

