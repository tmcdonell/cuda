{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    buildInputs = [ pkgs.git pkgs.ghc pkgs.cabal-install ];
  }
