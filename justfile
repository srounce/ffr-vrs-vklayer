# FFR-VRS layer dev loop. Run inside `nix develop` (direnv loads it automatically).

# Default: list recipes
default:
    @just --list

# Build the whole workspace (debug)
build:
    cargo build --workspace

# Build optimized layers
build-release:
    cargo build --workspace --release

# Run all unit tests
test:
    cargo test --workspace

# Lint (matches `nix flake check`)
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Format
fmt:
    cargo fmt

# Reproducible build of the packaged layers + manifests via Nix
build-nix:
    nix build .#layers

# Print env exports that point both loaders at the Nix build output.
# Usage:  eval "$(just env)"   or   just env > .env.local && source .env.local
env:
    #!/usr/bin/env bash
    out="$(nix build .#layers --no-link --print-out-paths)"
    echo "export VK_ADD_LAYER_PATH=$out/share/vulkan/explicit_layer.d"
    echo "export VK_LOADER_LAYERS_ENABLE='VK_LAYER_FFR_VRS*'"
    echo "export XR_API_LAYER_PATH=$out/share/openxr/1/api_layers/explicit.d"
    echo "export XR_ENABLE_API_LAYERS=XR_APILAYER_FFRVRS_foveation"
    echo "export LD_LIBRARY_PATH=$out/lib:\${LD_LIBRARY_PATH:-}"
    echo "export FFR_VRS_LOG=debug"

# Build the layers, enable BOTH via env only (no system install), and run the
# test app. The app creates an OpenXR instance (loading the OpenXR layer); it
# needs a running runtime such as `monado-service`. Extra args are forwarded.
run-app *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    out="$(nix build .#layers --no-link --print-out-paths)"
    # --- enable both layers from the build output, env only ---
    export VK_ADD_LAYER_PATH="$out/share/vulkan/explicit_layer.d"
    export VK_LOADER_LAYERS_ENABLE='VK_LAYER_FFR_VRS*'
    export XR_API_LAYER_PATH="$out/share/openxr/1/api_layers/explicit.d"
    export XR_ENABLE_API_LAYERS=XR_APILAYER_FFRVRS_foveation
    export LD_LIBRARY_PATH="$out/lib:${LD_LIBRARY_PATH:-}"
    export FFR_VRS_LOG="${FFR_VRS_LOG:-debug}"
    echo "FFR layers enabled from: $out"
    cargo run -p ffr-test-app -- {{ ARGS }}

# Same as run-app, but with loader debug to watch the layers join the chains.
debug-loaders *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    out="$(nix build .#layers --no-link --print-out-paths)"
    export VK_ADD_LAYER_PATH="$out/share/vulkan/explicit_layer.d"
    export VK_LOADER_LAYERS_ENABLE='VK_LAYER_FFR_VRS*'
    export XR_API_LAYER_PATH="$out/share/openxr/1/api_layers/explicit.d"
    export XR_ENABLE_API_LAYERS=XR_APILAYER_FFRVRS_foveation
    export LD_LIBRARY_PATH="$out/lib:${LD_LIBRARY_PATH:-}"
    export FFR_VRS_LOG="${FFR_VRS_LOG:-debug}"
    export VK_LOADER_DEBUG=layer
    export XRT_LOG=debug
    cargo run -p ffr-test-app -- {{ ARGS }}

# M0-meaningful smoke test: enable the Vulkan layer via env only and run
# vulkaninfo (which actually creates a Vulkan instance). At M0 you'll see the
# loader discover the layer and then "Skipping layer" (the negotiate entry
# points arrive in M1); after M1 this shows the layer active with its banner.
smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    out="$(nix build .#layers --no-link --print-out-paths)"
    export VK_ADD_LAYER_PATH="$out/share/vulkan/explicit_layer.d"
    export VK_LOADER_LAYERS_ENABLE='VK_LAYER_FFR_VRS*'
    export VK_LOADER_DEBUG=layer
    echo "Enabled VK_LAYER_FFR_VRS_foveation from $out (env only). Loader says:"
    vulkaninfo 2>&1 \
      | grep -iE 'VK_LAYER_FFR_VRS_foveation|Negotiate|Skipping layer|forced enabled' \
      | sort -u
