{-# LANGUAGE ParallelListComp #-}
--------------------------------------------------------------------------------
--
-- Module    : SimpleTexture
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Demonstrates texture references. Takes an input PGM file and generates an
-- output that has been rotated. The kernel performs a simple 2D transformation
-- on the texture coordinates (u,v).
--
--------------------------------------------------------------------------------

module Main where

import Data.Array.Unboxed
import Graphics.Pgm
import qualified Foreign.CUDA                 as CUDA
import qualified Foreign.CUDA.Runtime.Texture as CUDA


imageFile, outputFile, referenceFile :: FilePath
imageFile     = "data/lena_bw.pgm"
outputFile    = "data/lena_bw_out.pgm"
referenceFile = "data/ref_rotated.pgm"

rotation :: Float
rotation = 0.5          -- radians


main :: IO ()
main = do
  img <- either (error . show) head `fmap` pgmsFromFile imageFile
  ref <- either (error . show) head `fmap` pgmsFromFile referenceFile
  runTest img ref


runTest :: UArray (Int,Int) Int -> UArray (Int,Int) Int -> IO ()
runTest img ref' = do
  rotated <- transform (amap (\x-> fromIntegral x/255) img)
  arrayToFile outputFile (amap (\x -> round (x*255)) rotated)
  if verify (elems ref) (elems rotated)
     then putStrLn "PASSED"
     else putStrLn "FAILED!"

  where
    ref          = amap (\x -> fromIntegral x / 255) ref'
    verify xs ys = all (< epsilon) [abs ((x-y)/(x+y+epsilon)) | x <- xs | y <- ys]
    epsilon      = 0.05


transform :: UArray (Int,Int) Float -> IO (UArray (Int,Int) Float)
transform pgm =
  let bnds@((x0,y0),(x1,y1)) = bounds pgm
      width   = x1-x0+1
      height  = y1-y0+1
      desc    = CUDA.FormatDesc (32,0,0,0) CUDA.Float
      texture = CUDA.Texture True CUDA.Linear (CUDA.Wrap,CUDA.Wrap,CUDA.Wrap) desc
  in
  CUDA.allocaArray (width * height) $ \d_out ->
  CUDA.withListArray (elems pgm)    $ \d_tex -> do
    CUDA.bind2D "tex" texture d_tex (width,height) (fromIntegral $ height * 4)
    CUDA.setConfig (width `div` 8,height `div` 8) (8,8,1) 0 Nothing
    CUDA.setParams [CUDA.VArg d_out, CUDA.IArg width, CUDA.IArg height, CUDA.FArg rotation]
    CUDA.launch "transformKernel"

    listArray bnds `fmap` CUDA.peekListArray (width*height) d_out

