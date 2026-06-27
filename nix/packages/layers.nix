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

  postInstall = ''
    mkdir -p \
      $out/lib \
      $out/share/vulkan/explicit_layer.d \
      $out/share/openxr/1/api_layers/explicit.d

    for so in libffr_shared.so libVkLayer_FFR_VRS.so libXrApiLayer_FFR_VRS.so; do
      cp target/release/$so $out/lib/
    done

    substitute ${../../manifests/VkLayer_FFR_VRS.json.in} \
      $out/share/vulkan/explicit_layer.d/VkLayer_FFR_VRS.json \
      --replace-fail '@library_path@' "$out/lib/libVkLayer_FFR_VRS.so"

    substitute ${../../manifests/XrApiLayer_FFR_VRS.json.in} \
      $out/share/openxr/1/api_layers/explicit.d/XrApiLayer_FFR_VRS.json \
      --replace-fail '@library_path@' "$out/lib/libXrApiLayer_FFR_VRS.so"
  '';
})
