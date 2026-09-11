#!/usr/bin/env bash
# Build every trainer bot backend from the reference clones in tmp/.
#
#   tmp/ (gitignored, the cloned upstream projects)
#     -> copied/portable sources under bots/<name>/
#     -> compiled shared libraries to bots/build/lib<name>.so
#
# Each backend is optional at runtime: the trainer lists whichever
# libraries exist; a missing one simply drops out of the model picker.
# Nothing here modifies the upstream clones — portable sources are copied
# out, bridges live beside the copies.
set -euo pipefail

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/.." && pwd)
build=$root/bots/build
mkdir -p "$build"

CARGO=${CARGO:-$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo}

# --- MisaMino: portable core + Linux bridge --------------------------------
mm=$root/bots/misamino
src=$root/tmp/MisaMino/dllai
if [ -d "$src" ]; then
    for f in ai.cpp genmove.cpp tetris_gem.cpp ai.h gamefield.h tetris_gem.h; do
        if ! cmp -s "$src/$f" "$mm/$f"; then
            cp "$src/$f" "$mm/$f"
            echo "refreshed $f from tmp/MisaMino"
        fi
    done
    # MSVC source fixes, applied to the COPY (the clone in tmp/ is never
    # touched): (1) MSVC accepts ## in invalid paste positions where the
    # intended meaning is plain token separation — replace with spaces;
    # (2) abs() needs <cstdlib> under libstdc++. The pristine copy is
    # re-copied from tmp/ first, so the patches re-apply cleanly each build.
    # MSVC source fixes, applied to the COPY (the clone in tmp/ is never
    # touched). genmove.cpp's macros paste argument values into identifiers
    # (n##y -> ny) — GNU cpp supports that fine; what MSVC silently ignores
    # and GNU rejects is a ## adjacent to '[', '::' or ')'. Remove only
    # those (with a space after '::'), keep the identifier pastes. abs()
    # needs <cstdlib> under libstdc++.
    sed -i 's/\[##/[/g; s/##)/)/g; s/##;/;/g; s/##\]/]/g; s/Moving::##/Moving:: /g' "$mm/genmove.cpp"
    sed -i 's|#include "ai.h"|#include <cstdlib>\n#include "ai.h"|' "$mm/ai.cpp"
    g++ -O2 -fPIC -shared -std=c++14 \
        -Wall -Wno-unused-value -Wno-sign-compare -Wno-parentheses \
        -Wno-maybe-uninitialized -Wno-unknown-pragmas \
        "$mm/misamino_bridge.cpp" "$mm/ai.cpp" "$mm/genmove.cpp" "$mm/tetris_gem.cpp" \
        -o "$build/libmisamino.so"
    echo "built libmisamino.so"
fi

# --- cold-clear: upstream C API crate ---------------------------------------
cc=$root/tmp/cold-clear
if [ -d "$cc" ] && [ -x "$CARGO" ]; then
    (cd "$cc" && "$CARGO" build --release -p c-api)
    cp "$cc/target/release/libcold_clear.so" "$build/"
    echo "built libcold_clear.so"
fi

# --- fusion (MochBot): shim crate over the engine ---------------------------
fu=$root/tmp/fusion
if [ -d "$fu" ] && [ -x "$CARGO" ]; then
    (cd "$root/bots/fusion-shim" && "$CARGO" build --release)
    cp "$fu/target/release/libfusion_shim.so" "$build/" 2>/dev/null \
        || cp "$root/bots/fusion-shim/target/release/libfusion_shim.so" "$build/"
    echo "built libfusion_shim.so"
fi

# --- blockfish (iitalics): shim crate over the engine ------------------------
bf=$root/tmp/blockfish
if [ -d "$bf/blockfish-engine" ] && [ -x "$CARGO" ]; then
    (cd "$root/bots/blockfish-shim" && "$CARGO" build --release)
    cp "$root/bots/blockfish-shim/target/release/libblockfish_shim.so" "$build/"
    echo "built libblockfish_shim.so"
fi

ls -la "$build"
