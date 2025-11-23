--------------------------------------------------------------------------------
--
-- Module    : Time
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Simple timing benchmarks
--
--------------------------------------------------------------------------------

module Time where

import System.CPUTime
import Control.Monad


-- Timing
--
data Time = Time { cpu_time :: Integer }

type TimeUnit = Integer -> Integer

picosecond, millisecond, second :: TimeUnit
picosecond  n = n
millisecond n = n `div` 1000000000
second      n = n `div` 1000000000000

getTime :: IO Time
getTime = Time `fmap` getCPUTime

timeIn :: TimeUnit -> Time -> Integer
timeIn u (Time t) = u t

elapsedTime :: Time -> Time -> Time
elapsedTime (Time t1) (Time t2) = Time (t2 - t1)


-- Simple benchmarking
--
{-# NOINLINE benchmark #-}
benchmark
  :: Int                -- Number of times to repeat test
  -> IO a               -- Test to run
  -> IO b               -- Finaliser to before measuring elapsed time
  -> IO (Time,a)
benchmark n testee finaliser = do
  t1    <- getTime
  (r:_) <- replicateM n testee
  _     <- finaliser
  t2    <- getTime
  return (elapsedTime t1 t2, r)

