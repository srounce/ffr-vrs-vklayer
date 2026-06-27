{ inputs, ... }:
let
  mkRustToolchain = system:
    inputs.fenix.packages.${system}.fromToolchainFile {
      file = ../../rust-toolchain.toml;
      sha256 = "sha256-gh/xTkxKHL4eiRXzWv8KP7vfjSk61Iq48x47BEDFgfk=";
    };

  # Shared crane setup so the package build and the checks reuse one set of
  # cargo dependency artifacts (same derivation → built once, deduped).
  craneSetup = { pkgs, system }:
    let
      craneLib = (inputs.crane.mkLib pkgs).overrideToolchain (mkRustToolchain system);
      # Keep cargo sources plus GLSL shaders (the test app's build.rs compiles
      # them with glslangValidator).
      src = pkgs.lib.cleanSourceWith {
        src = ../../.;
        filter = path: type:
          (craneLib.filterCargoSources path type)
          || (builtins.match ".*\\.(vert|frag|glsl|comp)$" (baseNameOf path) != null);
      };
      commonArgs = {
        inherit src;
        strictDeps = true;
        # glslang provides glslangValidator for the test app's build.rs shader
        # compilation.
        nativeBuildInputs = [ pkgs.pkg-config pkgs.glslang ];
        buildInputs = [ ];
      };
      cargoArtifacts = craneLib.buildDepsOnly commonArgs;
    in
    { inherit craneLib commonArgs cargoArtifacts; };
in
{
  inherit mkRustToolchain craneSetup;
}
