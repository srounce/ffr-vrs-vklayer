# `nix flake check` gate: clippy with warnings denied.
{ pkgs, system, flake, ... }:
let
  inherit (flake.lib.craneSetup { inherit pkgs system; }) craneLib commonArgs cargoArtifacts;
in
craneLib.cargoClippy (commonArgs // {
  inherit cargoArtifacts;
  cargoClippyExtraArgs = "--all-targets -- -D warnings";
})
