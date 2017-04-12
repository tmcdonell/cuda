{-# LANGUAGE CPP #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Path
-- Copyright : [2017] Trevor L. McDonell
-- License   : BSD
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Path (

  cudaInstallPath,
  cudaBinPath, cudaLibraryPath, cudaIncludePath,

) where

import System.FilePath

-- | The base path to the CUDA toolkit installation that this package was
-- compiled against.
--
cudaInstallPath :: FilePath
cudaInstallPath = {#const CUDA_INSTALL_PATH#}

cudaBinPath :: FilePath
cudaBinPath = cudaInstallPath </> "bin"

cudaLibraryPath :: FilePath
cudaLibraryPath = {#const CUDA_LIBRARY_PATH#}

cudaIncludePath :: FilePath
cudaIncludePath = cudaInstallPath </> "include"

