{-# LANGUAGE CPP, ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Event
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Event management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Event (

  -- * Event Management
  Event, EventFlag(..), WaitFlag,
  create, destroy, elapsedTime, query, record, wait, block

) where

#include <cuda.h>
#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream               (Stream(..))

-- System
import Foreign
import Foreign.C
import Control.Monad                            (liftM)


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

-- |
-- Possible option flags for waiting for events
--
data WaitFlag
instance Enum WaitFlag where


--------------------------------------------------------------------------------
-- Event management
--------------------------------------------------------------------------------

-- |
-- Create a new event
--
create :: [EventFlag] -> IO Event
create flags = resultIfOk =<< cuEventCreate flags

{# fun unsafe cuEventCreate
  { alloca-         `Event'       peekEvt*
  , combineBitMasks `[EventFlag]'          } -> `Status' cToEnum #}
  where peekEvt = liftM Event . peek


-- |
-- Destroy an event
--
destroy :: Event -> IO ()
destroy ev = nothingIfOk =<< cuEventDestroy ev

{# fun unsafe cuEventDestroy
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Determine the elapsed time (in milliseconds) between two events
--
elapsedTime :: Event -> Event -> IO Float
elapsedTime ev1 ev2 = resultIfOk =<< cuEventElapsedTime ev1 ev2

{# fun unsafe cuEventElapsedTime
  { alloca-  `Float' peekFloatConv*
  , useEvent `Event'
  , useEvent `Event'                } -> `Status' cToEnum #}


-- |
-- Determines if a event has actually been recorded
--
query :: Event -> IO Bool
query ev =
  cuEventQuery ev >>= \rv ->
  case rv of
    Success  -> return True
    NotReady -> return False
    _        -> resultIfOk (rv,undefined)

{# fun unsafe cuEventQuery
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Record an event once all operations in the current context (or optionally
-- specified stream) have completed. This operation is asynchronous.
--
record :: Event -> Maybe Stream -> IO ()
record ev mst =
  nothingIfOk =<< case mst of
    Just st -> cuEventRecord ev st
    Nothing -> cuEventRecord ev (Stream nullPtr)

{# fun unsafe cuEventRecord
  { useEvent  `Event'
  , useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Makes all future work submitted to the (optional) stream wait until the given
-- event reports completion before beginning execution. Requires cuda-3.2.
--
wait :: Event -> Maybe Stream -> [WaitFlag] -> IO ()
#if CUDA_VERSION < 3020
wait _  _   _     = requireSDK 3.2 "wait"
#else
wait ev mst flags =
  nothingIfOk =<< case mst of
    Just st -> cuStreamWaitEvent st ev flags
    Nothing -> cuStreamWaitEvent (Stream nullPtr) ev flags

{# fun unsafe cuStreamWaitEvent
  { useStream       `Stream'
  , useEvent        `Event'
  , combineBitMasks `[WaitFlag]' } -> `Status' cToEnum #}
#endif

-- |
-- Wait until the event has been recorded
--
block :: Event -> IO ()
block ev = nothingIfOk =<< cuEventSynchronize ev

{# fun unsafe cuEventSynchronize
  { useEvent `Event' } -> `Status' cToEnum #}

