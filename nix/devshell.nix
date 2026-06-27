{ pkgs, system, flake, ... }:
let
  packages = [
    (flake.lib.mkRustToolchain system)
    pkgs.just

    # Build deps
    pkgs.pkg-config
    pkgs.vulkan-headers
    pkgs.vulkan-loader
    pkgs.openxr-loader

    # Dev / inspection tooling
    pkgs.vulkan-tools # vulkaninfo
    pkgs.glslang # glslangValidator for shader compilation
  ];
in
pkgs.mkShell {
  inherit packages;

  # Heavier runtime/debug tooling (monado, renderdoc, validation layers) is
  # added when we reach M1 (running under Monado) to keep the dev shell light.

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;
}
