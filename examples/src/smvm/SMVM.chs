{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
--
-- Module    : SMVM
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Sparse-matrix dense-vector multiplication
--
--------------------------------------------------------------------------------

module Main where

#include "smvm.h"

-- Friends
import Time
import C2HS
import RandomVector                             (randomList,randomListR,verifyList)

-- System
import Numeric
import Data.List
import Control.Monad
import Control.Applicative
import System.Random
import Foreign.CUDA                             (withDevicePtr)
import qualified Foreign.CUDA as CUDA

--
-- A very simple sparse-matrix / vector representation
-- (confusingly, different from that in RandomVector and used elsewhere)
--
type Vector e       = [e]
type SparseVector e = [(Int,e)]
type SparseMatrix e = [SparseVector e]

--------------------------------------------------------------------------------
-- Reference
--------------------------------------------------------------------------------

smvm :: (Num e, Storable e) => SparseMatrix e -> Vector e -> Vector e
smvm sm v = [ sum [ x * (v!!col) | (col,x) <- sv ]  | sv <- sm ]

--------------------------------------------------------------------------------
-- CUDA
--------------------------------------------------------------------------------

--
-- Sparse-matrix vector multiplication, using compressed-sparse row format.
--
-- Lots of boilerplate to copy data to the device. Our simple list
-- representation has atrocious copy performance (see the `bandwidthTest'
-- example), so don't include that in the benchmarking
--
smvm_csr :: SparseMatrix Float -> Vector Float -> IO (Vector Float)
smvm_csr sm v =
  let matData  = concatMap (map cFloatConv . snd . unzip) sm
      colIdx   = concatMap (map cIntConv   . fst . unzip) sm
      rowPtr   = scanl (+) 0 (map (cIntConv . length) sm)
      v'       = map cFloatConv v
#ifdef __DEVICE_EMULATION__
      iters    = 1
#else
      iters    = 100
#endif
  in
  CUDA.withListArray    matData  $ \d_data       ->
  CUDA.withListArray    rowPtr   $ \d_ptr        ->
  CUDA.withListArray    colIdx   $ \d_indices    ->
  CUDA.withListArrayLen v'       $ \num_rows d_x ->
  CUDA.allocaArray      num_rows $ \d_y          -> do
    (t,_) <- benchmark iters (smvm_csr_f d_y d_x d_data d_ptr d_indices num_rows) CUDA.sync
    putStrLn $ "Elapsed time: " ++ shows (fromInteger (timeIn millisecond t)/100::Float) " ms"
    map cFloatConv <$> CUDA.peekListArray num_rows d_y

{# fun unsafe smvm_csr_f
  { withDevicePtr* `CUDA.DevicePtr CFloat'
  , withDevicePtr* `CUDA.DevicePtr CFloat'
  , withDevicePtr* `CUDA.DevicePtr CFloat'
  , withDevicePtr* `CUDA.DevicePtr CUInt'
  , withDevicePtr* `CUDA.DevicePtr CUInt'
  ,                `Int'                   } -> `()' #}

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

randomSM :: (Num e, Random e, Storable e) => (Int,Int) -> IO (SparseMatrix e)
randomSM (h,w) = replicateM h sparseVec
  where
    sparseVec = do
      nz  <- randomRIO (0,w`div`50)                 -- number of non-zero elements (max 2% occupancy)
      idx <- nub . sort <$> randomListR nz (0,w-1)  -- remove duplicate column indices
      zip idx <$> randomList (length idx)           -- (column indices don't actually need to be sorted)


main :: IO ()
main = do
  dev   <- CUDA.get
  props <- CUDA.props dev
  putStrLn $ "Using device " ++ show dev ++ ": \"" ++ CUDA.deviceName props ++ "\""
  putStrLn $ "  Compute capability:  " ++ show (CUDA.computeCapability props)
  putStrLn $ "  Total global memory: " ++
    showFFloat (Just 2) (fromIntegral (CUDA.totalGlobalMem props) / (1024*1024) :: Double) " GB\n"

  let k = 2048                  -- size of the matrix/vector
  vec <- randomList k
  mat <- randomSM (k,k)
  putStrLn $ "Sparse-matrix size: " ++ show k ++ " x " ++ show k
  putStrLn $ "Non-zero elements: "  ++ show (sum (map length mat))

  let ref = smvm mat vec
  cuda   <- smvm_csr mat vec

  putStrLn $ "Validating: " ++ if verifyList ref cuda then "Ok!" else "INVALID!"

