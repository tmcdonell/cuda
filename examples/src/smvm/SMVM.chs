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

smvm :: Num e => SparseMatrix e -> Vector e -> Vector e
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
smvm_csr :: SparseMatrix Float -> Vector Float -> IO (Float, Vector Float)
smvm_csr sm v =
  let matData = concatMap (map cFloatConv . snd . unzip) sm
      colIdx  = concatMap (map cIntConv   . fst . unzip) sm
      rowPtr  = scanl (+) 0 (map (cIntConv . length) sm)
      v'      = map cFloatConv v
#ifdef __DEVICE_EMULATION__
      iters   = 1
#else
      iters   = 100
#endif
  in
  CUDA.withListArray    matData  $ \d_data       ->
  CUDA.withListArray    rowPtr   $ \d_ptr        ->
  CUDA.withListArray    colIdx   $ \d_indices    ->
  CUDA.withListArrayLen v'       $ \num_rows d_x ->
  CUDA.allocaArray      num_rows $ \d_y          -> do
    (t,_) <- benchmark iters (smvm_csr_f d_y d_x d_data d_ptr d_indices num_rows) CUDA.sync
    y     <- map cFloatConv <$> CUDA.peekListArray num_rows d_y
    return (fromInteger (timeIn millisecond t) / fromIntegral iters, y)


{# fun unsafe smvm_csr_f
  { withDevicePtr* `CUDA.DevicePtr CFloat'
  , withDevicePtr* `CUDA.DevicePtr CFloat'
  , withDevicePtr* `CUDA.DevicePtr CFloat'
  , withDevicePtr* `CUDA.DevicePtr CUInt'
  , withDevicePtr* `CUDA.DevicePtr CUInt'
  ,                `Int'                   } -> `()' #}


--
-- Sparse-matrix vector multiplication from CUDPP
--
smvm_cudpp :: SparseMatrix Float -> Vector Float -> IO (Float, Vector Float)
smvm_cudpp sm v =
  let matData = concatMap (map cFloatConv . snd . unzip) sm
      colIdx  = concatMap (map cIntConv   . fst . unzip) sm
      rowPtr  = scanl (+) 0 (map (cIntConv . length) sm)
      v'      = map cFloatConv v
#ifdef __DEVICE_EMULATION__
      iters   = 1
#else
      iters   = 100
#endif
  in
  CUDA.withListArrayLen v'       $ \num_rows     d_x    ->
  CUDA.allocaArray      num_rows $ \d_y                 ->
  withArrayLen          matData  $ \num_nonzeros h_data ->
  withArray             rowPtr   $ \h_rowPtr            ->
  withArray             colIdx   $ \h_colIdx            -> do
    (t,_) <- benchmark iters (smvm_cudpp_f d_y d_x h_data h_rowPtr h_colIdx num_rows num_nonzeros) CUDA.sync
    y     <- map cFloatConv <$> CUDA.peekListArray num_rows d_y
    return (fromInteger (timeIn millisecond t) / fromIntegral iters, y)

{# fun unsafe smvm_cudpp_f
  { withDevicePtr* `CUDA.DevicePtr CFloat'
  , withDevicePtr* `CUDA.DevicePtr CFloat'
  , id             `Ptr CFloat'
  , id             `Ptr CUInt'
  , id             `Ptr CUInt'
  ,                `Int'
  ,                `Int'                   } -> `()' #}


--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

--
-- Generate random matrices
--
sparseMat :: (Num e, Random e, Storable e) => (Int,Int) -> (Int,Int) -> IO (SparseMatrix e)
sparseMat bnds (h,w) = replicateM h sparseVec
  where
    sparseVec = do
      nz  <- randomRIO bnds                         -- number of non-zero elements
      idx <- nub . sort <$> randomListR nz (0,w-1)  -- remove duplicate column indices
      zip idx <$> randomList (length idx)           -- (column indices don't actually need to be sorted)

denseMat :: (Num e, Random e, Storable e) => (Int,Int) -> IO (SparseMatrix e)
denseMat (h,w) = replicateM h (zip [0..] <$> randomList w)

--
-- Some test-harness utilities
--
stats :: (Floating a, Ord a) => [a] -> (a,a,a,a,a,a)
stats (x:xs) = finish . foldl' stats' (x,x,x,x*x,1) $ xs
  where
    stats' (mn,mx,s,ss,n) v = (min v mn, max v mx, s+v, ss+v*v, n+1)
    finish (mn,mx,s,ss,n)   = (mn, mx, av, var, stdev, n)
      where av    = s/n
            var   = (1/(n-1))*ss - (n/(n-1))*av*av
            stdev = sqrt var


testAlgorithm :: (Num e, Ord e, Floating e)
             => String                                                  -- name of the algorithm
             -> (SparseMatrix e -> Vector e -> IO (Float, Vector e))    -- return time (ms), and result
             -> SparseMatrix e                                          -- input matrix
             -> Vector e                                                -- input vector
             -> Vector e                                                -- reference solution
             -> IO ()
testAlgorithm name f m v ref = do
    putStr name
    (t,y) <- f m v
    putStr   $ if verifyList ref y then "Ok!      " else "INVALID! "
    putStr   $ "( " ++ showFFloat (Just 2) t " ms, "
    putStrLn $ showFFloat (Just 2) (fromIntegral (2 * 1000 * sum (map length m)) / (t * 1E9)) " GFLOPS )"


testMatrix :: String -> SparseMatrix Float -> Vector Float -> IO ()
testMatrix name sm v = do
  let w                  = length v
      (_,_,av,_,stdev,h) = stats (map (fromIntegral . length) sm)
      ref                = smvm sm v

  putStr   $ name ++ ": " ++ show w ++ "x" ++ show (round h)
  putStr   $ ", " ++ shows (round (av*h)) " non-zero elements "
  putStrLn $ "( " ++ showFFloat (Just 2) av " +/- " ++ showFFloat (Just 2) stdev " )"

  testAlgorithm "  smvm-csr:     " smvm_csr   sm v ref
  testAlgorithm "  smvm-cudpp:   " smvm_cudpp sm v ref
  putStrLn ""


--
-- Finally, the main function
--
main :: IO ()
main = do
  dev   <- CUDA.get
  props <- CUDA.props dev
  putStrLn $ "Using device " ++ show dev ++ ": \"" ++ CUDA.deviceName props ++ "\""
  putStrLn $ "  Compute capability:  " ++ show (CUDA.computeCapability props)
  putStrLn $ "  Total global memory: " ++
    showFFloat (Just 2) (fromIntegral (CUDA.totalGlobalMem props) / (1024*1024) :: Double) " GB\n"

  v1 <- randomList 512
  v2 <- randomList 2048
  m1 <- denseMat  (512,512)
  m2 <- sparseMat (20,200) (20 * 2048,2048)

  testMatrix "Dense Matrix"  m1 v1
  testMatrix "Sparse Matrix" m2 v2

