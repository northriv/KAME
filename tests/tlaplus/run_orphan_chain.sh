#!/bin/bash
# Regression guard for the orphan-chain TLA+ models.
#
# Runs every OrphanChain_* model with each of its cfgs and ASSERTS the EXPECTED
# outcome — CLEAN ("No error has been found") or a specific invariant VIOLATION.
# This is a true regression gate, not just a "does it pass" check: a cfg that is
# SUPPOSED to violate (e.g. the shipped raw-DLL design under OrphanChain_adopt,
# or any gate/self-ref knob turned off) must STILL violate, or the model has
# silently weakened.
#
# Why keep this after the KAME_ORPHAN_CHAIN flip (Stage 7): the owner-free vs
# residual-scrub-pin race (Inv_NoBadOwnerFree) is a narrow timing-dependent UAF
# that runtime stress CANNOT reproduce — 21.32M-op ASan/TSan on the pre-fix code
# did NOT trip it.  Only this model catches that class.  Run it on every change
# to allocator.cpp's orphan-chain / adopt / owner-free paths.
#
# Usage:   ./run_orphan_chain.sh           # run all checks
#          ./run_orphan_chain.sh adopt     # only cfgs whose name matches 'adopt'
#
# Requires tla2tools.jar in this directory (gitignored — copy it in, e.g.
#   cp ../../../kamestm/tests/tlaplus/tla2tools.jar .
# ).  If absent, the script SKIPS with a warning (exit 0) so a TLA-less CI is
# not broken; pass STRICT=1 to make a missing jar a hard failure.

set -uo pipefail
cd "$(dirname "$0")"

JAR=tla2tools.jar
FILTER="${1:-}"
STRICT="${STRICT:-0}"

if [[ ! -f "$JAR" ]]; then
    echo "WARN: $JAR not found — skipping TLA regression checks."
    echo "      cp ../../../kamestm/tests/tlaplus/$JAR ."
    [[ "$STRICT" == "1" ]] && { echo "STRICT=1 → treating as failure."; exit 1; }
    exit 0
fi
if ! command -v java >/dev/null 2>&1; then
    echo "WARN: java not found — skipping TLA regression checks."
    [[ "$STRICT" == "1" ]] && exit 1
    exit 0
fi

# spec | cfg | expectation
#   expectation = CLEAN                         (no error found)
#               | VIOLATION:<InvariantName>     (that invariant must be reported)
CHECKS=(
  "OrphanChain_atomicshared|OrphanChain_atomicshared_selfref_mc.cfg|CLEAN"
  "OrphanChain_atomicshared|OrphanChain_atomicshared_noselfref_mc.cfg|VIOLATION:Inv_NoBadRelease"
  "OrphanChain_atomicshared|OrphanChain_atomicshared_live_mc.cfg|CLEAN"
  "OrphanChain_atomicshared|OrphanChain_atomicshared_push_mc.cfg|CLEAN"
  "OrphanChain_pathB|OrphanChain_pathB_mc.cfg|CLEAN"
  "OrphanChain_pathB|OrphanChain_pathB_liveremoval_mc.cfg|VIOLATION:Inv_NoBadRelease"
  "OrphanChain_pathB|OrphanChain_pathB_live_mc.cfg|CLEAN"
  "OrphanChain_adopt|OrphanChain_adopt_mc.cfg|VIOLATION:Inv_NoBadOwnerFree"
  "OrphanChain_adopt|OrphanChain_adopt_nogate_mc.cfg|VIOLATION:Inv_NoBadRelease"
  "OrphanChain_adopt|OrphanChain_adopt_ownerref_mc.cfg|CLEAN"
)

pass=0 fail=0 skip=0
for entry in "${CHECKS[@]}"; do
    IFS='|' read -r spec cfg expect <<< "$entry"
    if [[ -n "$FILTER" && "$cfg" != *"$FILTER"* ]]; then continue; fi
    if [[ ! -f "$spec.tla" || ! -f "$cfg" ]]; then
        printf "  SKIP  %-42s (missing %s)\n" "$cfg" "$( [[ -f "$spec.tla" ]] && echo "$cfg" || echo "$spec.tla")"
        skip=$((skip+1)); continue
    fi
    out=$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -config "$cfg" "$spec.tla" 2>&1)
    clean=$(grep -c "No error has been found" <<< "$out")
    if [[ "$expect" == CLEAN ]]; then
        if [[ "$clean" -ge 1 ]]; then
            printf "  PASS  %-42s CLEAN\n" "$cfg"; pass=$((pass+1))
        else
            printf "  FAIL  %-42s expected CLEAN but got:\n" "$cfg"
            grep -E "is violated|Error:" <<< "$out" | head -2 | sed 's/^/        /'
            fail=$((fail+1))
        fi
    else
        inv="${expect#VIOLATION:}"
        if grep -q "Invariant $inv is violated" <<< "$out"; then
            printf "  PASS  %-42s VIOLATION:%s (as expected)\n" "$cfg" "$inv"; pass=$((pass+1))
        else
            printf "  FAIL  %-42s expected VIOLATION:%s but:\n" "$cfg" "$inv"
            { grep -E "No error has been found|is violated" <<< "$out" | head -2; } | sed 's/^/        /'
            fail=$((fail+1))
        fi
    fi
done

echo "----------------------------------------------------------------"
echo "orphan-chain TLA regression: pass=$pass fail=$fail skip=$skip"
[[ "$fail" -eq 0 ]] && exit 0 || exit 1
