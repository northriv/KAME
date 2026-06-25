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
# >>> OHTAKA SESSION: please validate the 5 env knobs below against the
# >>> WORKING confC-T3 *liveness* sbatch script (the one that produced
# >>> ~640M states / ~15h25m). Mirror its paths exactly; nothing else here
# >>> should differ from that run. Checklist:
# >>>   1. TLA_TOOLS  -- path to tla2tools.jar
# >>>   2. java       -- on PATH? need a `module load` first?
# >>>   3. SCRATCH    -- large writable fs for TLC metadir (the liveness run's
# >>>                    /work path; $HOME likely has a quota too small for the
# >>>                    ~tens-of-GB fingerprint/queue files). /work/$USER is
# >>>                    NOT writable -- use the group/project path that run used.
# >>>   4. partition + --time  (F1cpu/24h default; adjust to what's free)
# >>>   5. -Xmx       -- match the liveness run's heap vs node RAM
# ===========================================================================
set -e
WORK=$HOME/kameSTMpaper
cd "$WORK"

SPEC=BundleUnbundle_3level_LLfree
CFG=${SPEC}_Is_confC_T3_mc.cfg

# --- env knobs (Ohtaka session: confirm/adjust to match the liveness run) ---
TLA_TOOLS="$WORK/tla2tools.jar"
HEAP="-Xmx200g"
SCRATCH="${SCRATCH:-}"          # <-- set to the liveness run's /work scratch path
# ----------------------------------------------------------------------------

WORKERS=$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} - 2 ))

if [ -n "$SCRATCH" ]; then
    METADIR="$SCRATCH/tlc_${SLURM_JOB_ID:-local}"
    mkdir -p "$METADIR"
    METADIR_ARG=(-metadir "$METADIR")
else
    METADIR_ARG=()              # fall back to TLC default (states/ under CWD)
    echo "WARN: SCRATCH unset -> TLC metadir defaults under $WORK (watch \$HOME quota)"
fi

echo "node=$(hostname)  workers=$WORKERS  cfg=$CFG  metadir=${METADIR:-<default>}"
lscpu --extended | head -3 || true

java -XX:+UseParallelGC $HEAP \
     -cp "$TLA_TOOLS" tlc2.TLC \
     -workers "$WORKERS" \
     "${METADIR_ARG[@]}" \
     -config "$CFG" \
     "$SPEC.tla"

echo "=== exit $? ==="
