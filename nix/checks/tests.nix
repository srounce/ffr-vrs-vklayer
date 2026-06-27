# `nix flake check` gate: workspace unit tests.
{ pkgs, system, flake, ... }:
let
  inherit (flake.lib.craneSetup { inherit pkgs system; }) craneLib commonArgs cargoArtifacts;
in
craneLib.cargoTest (commonArgs // {
  inherit cargoArtifacts;
})
