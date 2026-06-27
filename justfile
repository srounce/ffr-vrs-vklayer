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

# Print env exports that make the implicit layers discoverable, exactly as
# `environment.systemPackages` does (it links share/ onto XDG_DATA_DIRS).
# Usage:  eval "$(just env)"
env:
    #!/usr/bin/env bash
    out="$(nix build .#layers --no-link --print-out-paths)"
    echo "export XDG_DATA_DIRS=$out/share:\${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
    echo "export FFR_VRS_LOG=debug"

# Build the layers and run the test app with the layers auto-loaded via
# XDG_DATA_DIRS (no per-app enable needed — they are implicit). Needs a running
# OpenXR runtime such as `monado-service`.
run-app *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    out="$(nix build .#layers --no-link --print-out-paths)"
    export XDG_DATA_DIRS="$out/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
    export FFR_VRS_LOG="${FFR_VRS_LOG:-debug}"
    echo "FFR layers auto-loaded (implicit) from: $out"
    cargo run -p ffr-test-app -- {{ ARGS }}

# Same as run-app, with loader debug to watch the layers join the chains.
debug-loaders *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    out="$(nix build .#layers --no-link --print-out-paths)"
    export XDG_DATA_DIRS="$out/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
    export FFR_VRS_LOG="${FFR_VRS_LOG:-debug}"
    export VK_LOADER_DEBUG=layer
    export XRT_LOG=debug
    cargo run -p ffr-test-app -- {{ ARGS }}

# Smoke test: with the implicit Vulkan layer discoverable via XDG_DATA_DIRS,
# vulkaninfo should auto-load and list it (no force-enable needed).
smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    out="$(nix build .#layers --no-link --print-out-paths)"
    export XDG_DATA_DIRS="$out/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
    echo "Implicit VK_LAYER_FFR_VRS_foveation discovered via XDG_DATA_DIRS; vulkaninfo says:"
    vulkaninfo 2>&1 | grep -iE 'VK_LAYER_FFR_VRS_foveation' | sort -u
