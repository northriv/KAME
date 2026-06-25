#!/bin/bash
#SBATCH -p F1cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --time=24:00:00
#SBATCH -J l3sigmaT3
#SBATCH -o %x.%j.log
# ===========================================================================
# 3-level σ-closure at T=3 (confC all-root, superfine).
# Checks the 6 safety invariants + 4 candidate structural conjuncts
# (SubNeverMissing, BundledHasCopy, StaleParentExcluded, SubPresenceUniform)
# hold across the full ~640M-state T=3 exhaustion.
#
# This is the VERIFIED confC superfine T=3 run with exactly 3 changes:
#   - cfg = *_Is_confC_T3_mc.cfg (adds the 4 conjuncts as INVARIANTs)
#   - NO -dump (we only need the yes/no closure result, not the state set)
#   - NO liveness PROPERTY (safety/invariant checking only -> no SCC pass)
#
# Env is aligned to the liveness PHASE=1 run by the Ohtaka session:
# $SLURM_SUBMIT_DIR-based cwd, tla_preflight.sh (java/module), and a /work
# scratch probe. tla2tools.jar resolved relative to the submit dir.
# ===========================================================================
set -e
cd "$SLURM_SUBMIT_DIR"
. "$SLURM_SUBMIT_DIR/tla_preflight.sh"

SPEC=BundleUnbundle_3level_LLfree
CFG=${SPEC}_Is_confC_T3_mc.cfg

HEAP="${HEAP:--Xmx200g -Xms100g}"

# Probe for a writable large scratch fs for the TLC metadir (fingerprint/queue
# files run to tens of GB at ~640M states; $HOME quota is likely too small).
GROUP=$(id -gn 2>/dev/null)
SCRATCH=""
for candidate in \
        "${GROUP:+/work/$GROUP/$USER}" \
        "/work/$USER" \
        "$HOME" ; do
    [ -z "$candidate" ] && continue
    if mkdir -p "$candidate/tlc-sigma" 2>/dev/null; then
        SCRATCH="$candidate/tlc-sigma/l3_sigma_${SLURM_JOB_ID:-local}"
        break
    fi
done

WORKERS=$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} - 2 ))

if [ -n "$SCRATCH" ]; then
    METADIR="$SCRATCH"
    mkdir -p "$METADIR"
    METADIR_ARG=(-metadir "$METADIR")
else
    METADIR_ARG=()              # fall back to TLC default (states/ under CWD)
    echo "WARN: no writable scratch found -> TLC metadir defaults under CWD"
fi

echo "node=$(hostname)  workers=$WORKERS  cfg=$CFG  metadir=${METADIR:-<default>}"
lscpu --extended | head -3 || true

java -XX:+UseParallelGC $HEAP \
     -cp tla2tools.jar tlc2.TLC \
     -workers "$WORKERS" \
     "${METADIR_ARG[@]}" \
     -config "$CFG" \
     "$SPEC.tla"

echo "=== exit $? ==="
