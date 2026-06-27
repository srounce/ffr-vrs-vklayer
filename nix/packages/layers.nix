# Reproducible build of the two layers + the shared registry cdylib, laid out
# with their loader manifests. Built with crane on the pinned fenix toolchain.
{ pkgs, system, flake, ... }:
let
  inherit (flake.lib.craneSetup { inherit pkgs system; }) craneLib commonArgs cargoArtifacts;
in
craneLib.buildPackage (commonArgs // {
  inherit cargoArtifacts;
  pname = "ffr-vrs-layers";
  version = "0.1.0";
  # Tests run via `nix flake check`, not here.
  doCheck = false;

  # Install the layer manifests into the *implicit* loader directories so that,
  # when this package is in environment.systemPackages (which links share/ into
  # /run/current-system/sw/share, on XDG_DATA_DIRS), both loaders auto-load the
  # layers with no per-app configuration. The shared registry is found next to
  # the layers via dladdr, so no LD_LIBRARY_PATH is required.
  postInstall = ''
    mkdir -p \
      $out/lib \
      $out/share/vulkan/implicit_layer.d \
      $out/share/openxr/1/api_layers/implicit.d

    for so in libffr_shared.so libVkLayer_FFR_VRS.so libXrApiLayer_FFR_VRS.so; do
      cp target/release/$so $out/lib/
    done

    substitute ${../../manifests/VkLayer_FFR_VRS.json.in} \
      $out/share/vulkan/implicit_layer.d/VkLayer_FFR_VRS.json \
      --replace-fail '@library_path@' "$out/lib/libVkLayer_FFR_VRS.so"

    substitute ${../../manifests/XrApiLayer_FFR_VRS.json.in} \
      $out/share/openxr/1/api_layers/implicit.d/XrApiLayer_FFR_VRS.json \
      --replace-fail '@library_path@' "$out/lib/libXrApiLayer_FFR_VRS.so"
  '';
})
