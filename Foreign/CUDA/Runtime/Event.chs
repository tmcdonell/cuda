{-# LANGUAGE CPP, ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Event
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Event management for C-for-CUDA runtime environment
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Event (

  -- * Event Management
  Event, EventFlag(..), WaitFlag,
  create, destroy, elapsedTime, query, record, wait, block

) where

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Runtime.Stream                      ( Stream(..), defaultStream )
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad                                    ( liftM )
import Data.Maybe                                       ( fromMaybe )


#c
typedef enum cudaEvent_option_enum {
//  CUDA_EVENT_OPTION_DEFAULT       = cuduEventDefault,
    CUDA_EVENT_OPTION_BLOCKING_SYNC = cudaEventBlockingSync
} cudaEvent_option;
#endc

--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- Events
--
newtype Event = Event { useEvent :: {# type cudaEvent_t #}}

-- |
-- Event creation flags
--
{# enum cudaEvent_option_enum as EventFlag
    { underscoreToCase }
    with prefix="CUDA_EVENT_OPTION" deriving (Eq,Show) #}

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
create flags = resultIfOk =<< cudaEventCreateWithFlags flags

{# fun unsafe cudaEventCreateWithFlags
  { alloca-         `Event'       peekEvt*
  , combineBitMasks `[EventFlag]'          } -> `Status' cToEnum #}
  where peekEvt = liftM Event . peek


-- |
-- Destroy an event
--
destroy :: Event -> IO ()
destroy ev = nothingIfOk =<< cudaEventDestroy ev

{# fun unsafe cudaEventDestroy
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Determine the elapsed time (in milliseconds) between two events
--
elapsedTime :: Event -> Event -> IO Float
elapsedTime ev1 ev2 = resultIfOk =<< cudaEventElapsedTime ev1 ev2

{# fun unsafe cudaEventElapsedTime
  { alloca-  `Float' peekFloatConv*
  , useEvent `Event'
  , useEvent `Event'                } -> `Status' cToEnum #}


-- |
-- Determines if a event has actually been recorded
--
query :: Event -> IO Bool
query ev =
  cudaEventQuery ev >>= \rv ->
  case rv of
    Success  -> return True
    NotReady -> return False
    _        -> resultIfOk (rv,undefined)

{# fun unsafe cudaEventQuery
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Record an event once all operations in the current context (or optionally
-- specified stream) have completed. This operation is asynchronous.
--
record :: Event -> Maybe Stream -> IO ()
record ev mst =
  nothingIfOk =<< cudaEventRecord ev (maybe defaultStream id mst)

{# fun unsafe cudaEventRecord
  { useEvent  `Event'
  , useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Makes all future work submitted to the (optional) stream wait until the given
-- event reports completion before beginning execution. Requires cuda-3.2.
--
wait :: Event -> Maybe Stream -> [WaitFlag] -> IO ()
#if CUDART_VERSION < 3020
wait _  _   _     = requireSDK 3.2 "wait"
#else
wait ev mst flags =
  let st = fromMaybe defaultStream mst
  in  nothingIfOk =<< cudaStreamWaitEvent st ev flags

{# fun unsafe cudaStreamWaitEvent
  { useStream       `Stream'
  , useEvent        `Event'
  , combineBitMasks `[WaitFlag]' } -> `Status' cToEnum #}
#endif

-- |
-- Wait until the event has been recorded
--
block :: Event -> IO ()
block ev = nothingIfOk =<< cudaEventSynchronize ev

{# fun unsafe cudaEventSynchronize
  { useEvent `Event' } -> `Status' cToEnum #}

