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
import Control.Monad				(liftM)


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- Events
--
newtype Event = Event { useEvent :: {# type CUevent #}}


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
create flags = resultIfOk `fmap` cuEventCreate flags

{# fun unsafe cuEventCreate
  { alloca-         `Event'       peekEvt*
  , combineBitMasks `[EventFlag]'          } -> `Status' cToEnum #}
  where peekEvt = liftM Event . peek


-- |
-- Destroy an event
--
destroy :: Event -> IO (Maybe String)
destroy ev = nothingIfOk `fmap` cuEventDestroy ev

{# fun unsafe cuEventDestroy
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Determine the elapsed time (in milliseconds) between two events
--
elapsedTime :: Event -> Event -> IO (Either String Float)
elapsedTime ev1 ev2 = resultIfOk `fmap` cuEventElapsedTime ev1 ev2

{# fun unsafe cuEventElapsedTime
  { alloca-  `Float' peekFloatConv*
  , useEvent `Event'
  , useEvent `Event'                } -> `Status' cToEnum #}


-- |
-- Determines if a event has actually been recorded
--
query :: Event -> IO (Either String Bool)
query ev =
  cuEventQuery ev >>= \rv ->
  return $ case rv of
    Success  -> Right True
    NotReady -> Right False
    _        -> Left (describe rv)

{# fun unsafe cuEventQuery
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Record an event once all operations in the current context (or optionally
-- specified stream) have completed. This operation is asynchronous.
--
record :: Event -> Maybe Stream -> IO (Maybe String)
record ev mst =
  nothingIfOk `fmap` case mst of
    Just st -> (withStream st $ \s -> cuEventRecord ev s)
    Nothing -> cuEventRecord ev nullPtr

{# fun unsafe cuEventRecord
  { useEvent `Event'
  , castPtr  `Ptr Stream' } -> `Status' cToEnum #}


-- |
-- Wait until the event has been recorded
--
block :: Event -> IO (Maybe String)
block ev = nothingIfOk `fmap` cuEventSynchronize ev

{# fun unsafe cuEventSynchronize
  { useEvent `Event' } -> `Status' cToEnum #}

