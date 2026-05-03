# CUDA bindings for Haskell

This repository contains two packages:
1. `cuda`: Haskell bindings for the CUDA driver interface.
   See the `cuda/` subdirectory and its README.
2. `cuda-runtime`: Haskell bindings for the CUDA runtime interface, currently not published on Hackage.
   This code was previously part of the `cuda` package but to avoid API breakage in the CUDA 13 update, it was separated out into a prototype separate package.
   If you have need of this code and wish to maintain it, let us know.

The code in this repository is in **maintenance mode**.
For details, see the [`cuda` README](https://github.com/tmcdonell/cuda/tree/master/cuda#readme).
