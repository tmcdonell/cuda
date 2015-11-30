Haskell FFI Bindings to CUDA
============================

[![Build status](https://travis-ci.org/tmcdonell/cuda.svg?branch=master)](https://travis-ci.org/tmcdonell/cuda)

The CUDA library provides a direct, general purpose C-like SPMD programming
model for NVIDIA graphics cards (G8x series onwards). This is a collection of
bindings to allow you to call and control, although not write, such functions
from Haskell-land. You will need to install the CUDA driver and developer
toolkit.

  <http://developer.nvidia.com/object/cuda.html>

The configure step will look for your CUDA installation in the standard places,
and if the `nvcc` compiler is found in your `PATH`, relative to that.

For important information on installing on Windows, see:

  <https://github.com/tmcdonell/cuda/blob/master/WINDOWS.markdown>

