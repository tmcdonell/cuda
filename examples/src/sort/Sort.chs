{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
--
-- Module    : Sort
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Reduce a vector of (key,value) pairs
--
--------------------------------------------------------------------------------

module Main where

#include "sort.h"

import C2HS
import RandomVector

import Data.Ord
import Data.List
import Control.Monad
import qualified Foreign.CUDA as C



--------------------------------------------------------------------------------
-- CUDA

test_f :: (Storable a, Eq a) => [(Float,a)] -> IO Bool
test_f kv =
  let l     = length kv
      (k,v) = unzip kv
  in
  C.withListArray k $ \d_k ->
  C.withListArray v $ \d_v -> do

    sort_f d_k d_v (length kv)
    res <- liftM2 zip (C.peekListArray l d_k) (C.peekListArray l d_v)
    return (res == sortBy (comparing fst) kv)


test_i :: (Storable a, Eq a) => [(Int,a)] -> IO Bool
test_i kv =
  let l     = length kv
      (k,v) = unzip kv
  in
  C.withListArray k $ \d_k ->
  C.withListArray v $ \d_v -> do

    sort_ui d_k d_v (length kv)
    res <- liftM2 zip (C.peekListArray l d_k) (C.peekListArray l d_v)
    return (res == sortBy (comparing fst) kv)


{# fun unsafe sort_f
  { withDP* `C.DevicePtr Float'
  , withDP* `C.DevicePtr a'
  ,         `Int'
  } -> `()' #}
  where
    withDP p a = C.withDevicePtr p $ \p' -> a (castPtr p')

{# fun unsafe sort_ui
  { withDP* `C.DevicePtr Int'
  , withDP* `C.DevicePtr a'
  ,         `Int'
  } -> `()' #}
  where
    withDP p a = C.withDevicePtr p $ \p' -> a (castPtr p')

--
-- I don't need to learn template haskell or quick check... nah, not at all...
--
main :: IO ()
main = do
  f <- randomList  10000
  i <- randomListR 10000 (0,1000)

  putStr "Test (float,int): "
  test_f (zip f i) >>= \r -> case r of
    True -> putStrLn "Ok!"
    _    -> putStrLn "INVALID!"

  putStr "Test (float,float): "
  test_f (zip f f) >>= \r -> case r of
    True -> putStrLn "Ok!"
    _    -> putStrLn "INVALID!"

  putStr "Test (int,int): "
  test_i (zip i i) >>= \r -> case r of
    True -> putStrLn "Ok!"
    _    -> putStrLn "INVALID!"

  putStr "Test (int,float): "
  test_i (zip i f) >>= \r -> case r of
    True -> putStrLn "Ok!"
    _    -> putStrLn "INVALID!"

