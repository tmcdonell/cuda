{-# LANGUAGE CPP #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Path
-- Copyright : [2017..2020] Trevor L. McDonell
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

-- | The path where the CUDA toolkit executables, such as @nvcc@ and @ptxas@,
-- can be found.
--
cudaBinPath :: FilePath
cudaBinPath = cudaInstallPath </> "bin"

-- | The path where the CUDA libraries this package was linked against are
-- located
--
cudaLibraryPath :: FilePath
cudaLibraryPath = {#const CUDA_LIBRARY_PATH#}

-- | The path where the CUDA headers this package was built against are located
--
cudaIncludePath :: FilePath
cudaIncludePath = cudaInstallPath </> "include"

