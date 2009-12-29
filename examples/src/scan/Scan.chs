{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
--
-- Module    : Scan
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Apply a binary operator to an array similar to 'fold', but return a
-- successive list of values reduced from the left (or right).
--
--------------------------------------------------------------------------------

module Main where

#include "scan.h"

-- Friends
import C2HS                                     hiding (newArray)
import Time
import RandomVector

-- System
import Control.Monad
import Control.Exception
import qualified Foreign.CUDA as CUDA


--------------------------------------------------------------------------------
-- Reference
--------------------------------------------------------------------------------

scanList :: (Num e, Storable e) => Vector e -> IO (Vector e)
scanList xs = do
  bnds    <- getBounds xs
  xs'     <- getElems  xs
  (t,zs') <- benchmark 100 (return (scanl1 (+) xs')) (return ())
  putStrLn $ "List: " ++ shows (fromInteger (timeIn millisecond t`div`100)::Float) " ms"
  newListArray bnds zs'


scanArr :: (Num e, Storable e) => Vector e -> IO (Vector e)
scanArr xs = do
  bnds  <- getBounds xs
  zs    <- newArray_ bnds
  let idx = range bnds
  (t,_) <- benchmark 100 (foldM_ (k zs) 0 idx) (return ())
  putStrLn $ "Array: " ++ shows (fromInteger (timeIn millisecond t)/100::Float) " ms"
  return zs
  where
    k zs a i = do
      x <- readArray xs i
      let z = x+a
      writeArray zs i z
      return z


--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

--
-- Include the time to copy the data to/from the storable array (significantly
-- faster than from a Haskell list)
--
scanCUDA :: Vector Float -> IO (Vector Float)
scanCUDA xs = do
  bnds  <- getBounds xs
  zs    <- newArray_ bnds
  let len = rangeSize bnds
  CUDA.allocaArray len $ \d_xs -> do
  CUDA.allocaArray len $ \d_zs -> do
  (t,_) <- flip (benchmark 100) (return ()) $ do
    withVector xs $ \p -> CUDA.pokeArray len p d_xs
    scanl1_plusf d_xs d_zs len
    withVector zs $ \p -> CUDA.peekArray len d_zs p
  putStrLn $ "CUDA: " ++ shows (fromInteger (timeIn millisecond t)/100::Float) " ms (with copy)"

  (t',_) <- benchmark 100 (scanl1_plusf d_xs d_zs len) (return ())
  putStrLn $ "CUDA: " ++ shows (fromInteger (timeIn millisecond t')/100::Float) " ms (compute only)"

  return zs

{# fun unsafe scanl1_plusf
  { withDP* `CUDA.DevicePtr Float'
  , withDP* `CUDA.DevicePtr Float'
  ,         `Int'
  } -> `()' #}
  where
    withDP p a = CUDA.withDevicePtr p $ \p' -> a (castPtr p')


--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  dev   <- CUDA.get
  props <- CUDA.props dev
  putStrLn $ "Using device " ++ show dev ++ ": " ++ CUDA.deviceName props

  arr  <- randomArr (1,100000) :: IO (Vector Float)
  ref  <- scanList arr
  ref' <- scanArr  arr
  cuda <- scanCUDA arr

  return ()

  putStr   "== Validating: "
  verify ref ref' >>= \rv -> assert rv (return ())
  verify ref cuda >>= \rv -> putStrLn $ if rv then "Ok!" else "INVALID!"

