#!/bin/bash
#
# Build + sweep the three paper variants on macOS (M3 Air, M4 mini, …):
#
#   v0          — 2026-initial transaction.{h,_impl.h} (commit 1d49b07c)
#                 layered on top of the current allocator / tests / TLS
#                 pool fixes. Represents the legacy 16-years-in-production
#                 KAME STM before the AdaptiveNegotiation refactor.
#
#   no_backoff  — current transaction.* but with
#                 -DKAME_STM_DISABLE_BACKOFF=1. Pure CAS-retry, no
#                 sleep/negotiate/yield layer. Ablation row showing
#                 why the backoff layer earns its place.
#
#   full        — current transaction.* with default flags. The
#                 shipping AdaptiveNegotiation build.
#
# Each variant produces bench/<HOSTSHORT>_<variant>.tsv (same 3×3 s
# median format as M3air.tsv / M4mini.tsv) and appends dyn/neg wall
# times at the bottom for Table 11.
#
# Usage:
#   cd bench
#   ./run_mac_variants.sh                  # all three
#   ./run_mac_variants.sh v0 full          # specific subset
#
# Env vars:
#   KAME_ROOT      : default ~/KAME
#   KAME_V0_COMMIT : default 1d49b07c (pre-AdaptiveNegotiation base)
#   HOST_LABEL     : default auto from hostname
#   BENCH          : default script's own directory
#
# The script leaves the tree on the AdaptiveNegotiation tip with the
# head transaction.{h,_impl.h} restored, so it is safe to re-run.

set -u

KAME_ROOT=${KAME_ROOT:-$HOME/KAME}
KAME_V0_COMMIT=${KAME_V0_COMMIT:-1d49b07c}
BENCH=${BENCH:-$(cd "$(dirname "$0")" && pwd)}
HOST_LABEL=${HOST_LABEL:-$(hostname -s | tr -c '[:alnum:]' '_')}

VARIANTS=${*:-v0 no_backoff full}

# Resolve base branch once so we can restore transaction.* after v0.
cd "$KAME_ROOT"
BASE_REF=$(git rev-parse --abbrev-ref HEAD)
if [ "$BASE_REF" = "HEAD" ]; then
    BASE_REF=$(git rev-parse HEAD)        # detached → pin sha
fi

restore_head_tx() {
    cd "$KAME_ROOT"
    # Post-library-split: kamestm/; pre-split commits: kame/
    if [ -f kamestm/transaction.h ]; then
        git checkout "$BASE_REF" -- kamestm/transaction.h kamestm/transaction_impl.h
    else
        git checkout "$BASE_REF" -- kame/transaction.h kame/transaction_impl.h
    fi
    # The v0 overlay now writes into kamestm/ (restored above). Older,
    # buggy runs of this script dropped v0 headers into kame/; remove any
    # such leftovers so a stale kame/transaction.* can never be picked up
    # by include_directories instead of the canonical kamestm/ copy.
    if [ -f kamestm/transaction.h ] && [ -f kame/transaction.h ]; then
        rm -f kame/transaction.h kame/transaction_impl.h
        echo "[restore_head_tx] removed stale kame/transaction.{h,_impl.h}"
    fi
}

build_variant() {
    local variant=$1
    local bld="build-$variant"
    cd "$KAME_ROOT"
    # Fresh build dir per variant — sharing one build/ across variants
    # has historically produced ambiguous "no_backoff looks like full"
    # TSVs because cmake's dependency tracking couldn't always see that
    # CMAKE_CXX_FLAGS changed (especially after a transaction.h revert
    # for v0 left stale .o files behind). Isolating build trees makes
    # the variant provenance unambiguous.
    # Clean build every time — stale .o files from a previous variant
    # (especially v0's kame/transaction.h) can poison the link.
    rm -rf "$KAME_ROOT/$bld"

    case "$variant" in
        v0)
            # v0 commit (1d49b07c) predates the kamestm/ library split, so
            # its transaction headers live under kame/. The CURRENT tree
            # compiles kamestm/transaction.*, therefore the v0 overlay MUST
            # be written THERE. The previous code did `git checkout
            # $V0 -- kame/transaction.h`, which dropped the v0 headers into
            # kame/ — a path the post-split build no longer reads — so the
            # build silently used master's kamestm/transaction.* and the
            # "v0" sweep was actually benchmarking master (telltale: nonzero
            # LL_med, which pre-AdaptiveNegotiation v0 cannot produce).
            if git cat-file -e "$KAME_V0_COMMIT":kamestm/transaction.h 2>/dev/null; then
                v0dir=kamestm
            else
                v0dir=kame
            fi
            git show "$KAME_V0_COMMIT:$v0dir/transaction.h"      > kamestm/transaction.h
            git show "$KAME_V0_COMMIT:$v0dir/transaction_impl.h" > kamestm/transaction_impl.h
            echo "[v0] overlaid kamestm/transaction.{h,_impl.h} from ${KAME_V0_COMMIT}:${v0dir}/"
            cmake -S tests -B "$bld" -DCMAKE_BUILD_TYPE=Release \
                  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
                  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > /dev/null
            ;;
        no_backoff)
            restore_head_tx
            cmake -S tests -B "$bld" -DCMAKE_BUILD_TYPE=Release \
                  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -DKAME_STM_DISABLE_BACKOFF=1" \
                  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > /dev/null
            ;;
        full)
            restore_head_tx
            cmake -S tests -B "$bld" -DCMAKE_BUILD_TYPE=Release \
                  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
                  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > /dev/null
            ;;
        *)
            echo "unknown variant: $variant" >&2
            return 1
            ;;
    esac
    cmake --build "$bld" -j

    # Locate the headline binary. `cmake -S tests -B build-X` puts
    # executables at the build-dir root (build-X/), but some historic
    # kame trees built via `cmake -S . -B build` land them under
    # build/tests/. Probe both so we don't silently test the wrong tree.
    resolved_bin_dir=""
    if [ -x "$KAME_ROOT/$bld/kamestm-tests/transaction_payload_integrity_mixed_test" ]; then
        resolved_bin_dir="$KAME_ROOT/$bld/kamestm-tests"
    elif [ -x "$KAME_ROOT/$bld/transaction_payload_integrity_mixed_test" ]; then
        resolved_bin_dir="$KAME_ROOT/$bld"
    fi

    # Confirm the -D reached compile_commands.json. Non-fatal — prints
    # so the sweep log has documentary evidence that each binary really
    # was built with the claimed flags.
    echo "--- $variant: compile-flag audit ---"
    local cc="$KAME_ROOT/$bld/compile_commands.json"
    if [ -f "$cc" ]; then
        local hits
        hits=$(grep -o "KAME_STM_DISABLE_BACKOFF[^ \",]*" "$cc" | sort -u | tr '\n' ' ')
        echo "  KAME_STM_DISABLE_BACKOFF in compile_commands: ${hits:-<none>}"
    else
        echo "  (no compile_commands.json; skipping audit)"
    fi
    if [ -n "$resolved_bin_dir" ]; then
        local bin="$resolved_bin_dir/transaction_payload_integrity_mixed_test"
        echo "  binary: $(ls -l "$bin" | awk '{print $5, $NF}')"
    else
        echo "  WARNING: transaction_payload_integrity_mixed_test not found under $KAME_ROOT/$bld"
    fi
    echo "--- end audit ---"
}

run_sweep() {
    local variant=$1
    local bld="build-$variant"
    local out="$BENCH/${HOST_LABEL}_${variant}.tsv"

    # Reuse the probe from build_variant (resolved_bin_dir is set there).
    # If we ended up here without it (e.g. build failed), re-probe now
    # so the error surfaces in the TSV rather than as a mysterious hang.
    if [ -z "${resolved_bin_dir:-}" ]; then
        if [ -x "$KAME_ROOT/$bld/kamestm-tests/transaction_payload_integrity_mixed_test" ]; then
            resolved_bin_dir="$KAME_ROOT/$bld/kamestm-tests"
        elif [ -x "$KAME_ROOT/$bld/transaction_payload_integrity_mixed_test" ]; then
            resolved_bin_dir="$KAME_ROOT/$bld"
        else
            echo "ERROR: no binary under $KAME_ROOT/$bld — build may have failed." >&2
            return 1
        fi
    fi

    # Label the sweep header with the variant + transaction provenance so
    # the TSV self-documents which build it really came from.
    case "$variant" in
        v0)         export KAME_REV_NOTE="variant=v0 (transaction overlay ${KAME_V0_COMMIT})" ;;
        no_backoff) export KAME_REV_NOTE="variant=no_backoff (KAME_STM_DISABLE_BACKOFF=1)" ;;
        full)       export KAME_REV_NOTE="variant=full (shipping AdaptiveNegotiation)" ;;
        *)          export KAME_REV_NOTE="variant=$variant" ;;
    esac

    cd "$BENCH"
    # tee so the operator sees progress in real time — the TSV alone
    # was a 15-minute silence that looked indistinguishable from a hang.
    # stderr from run_kame_mixed_mac.sh (binary stderr on crash/timeout)
    # is intentionally NOT routed here; it goes to terminal stderr so the
    # TSV column layout stays clean.
    KAME_ROOT="$KAME_ROOT" BIN_DIR="$resolved_bin_dir" \
        ./run_kame_mixed_mac.sh "${HOST_LABEL}_${variant}" | tee "$out"
    echo "" | tee -a "$out"
    echo "--- dyn / neg wall times ---" | tee -a "$out"
    # Timeout guard: no_backoff / pathological builds can livelock on
    # these single-binary stress tests forever. 120 s is ~20× the
    # happy-path 5 s so a real slow run still completes, but a hung
    # one gives up with a clear '*** hit 120 s timeout ***' line.
    local tcmd=""
    if command -v gtimeout >/dev/null 2>&1; then tcmd="gtimeout --kill-after=5 120"
    elif command -v timeout  >/dev/null 2>&1; then tcmd="timeout  --kill-after=5 120"
    fi
    ( cd "$resolved_bin_dir" && \
        $tcmd /usr/bin/time -p ./transaction_negotiation_test 2>&1 | tail -5
        [ "${PIPESTATUS[0]:-0}" = 124 ] && echo "*** transaction_negotiation_test hit 120 s timeout ***"
        echo ""
        $tcmd /usr/bin/time -p ./transaction_dynamic_node_test 2>&1 | tail -5
        [ "${PIPESTATUS[0]:-0}" = 124 ] && echo "*** transaction_dynamic_node_test hit 120 s timeout ***" ) \
        2>&1 | tee -a "$out"
    echo ">>> wrote $out (BIN_DIR=$resolved_bin_dir)"
}

for v in $VARIANTS; do
    echo "=========================================="
    echo "== building + sweeping: $v"
    echo "=========================================="
    resolved_bin_dir=""                    # reset so run_sweep re-probes
    build_variant "$v"
    run_sweep "$v"
done

# Leave the tree clean.
restore_head_tx
echo "All variants done. Build trees kept at $KAME_ROOT/build-{v0,no_backoff,full} for rerun or inspection."
echo "Tree restored to $BASE_REF head transaction.*."
