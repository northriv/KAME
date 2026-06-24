#!/bin/bash
#SBATCH -p F1cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --time=24:00:00
#SBATCH -J l3sigmaT3
#SBATCH -o %x.%j.log
# ---------------------------------------------------------------------------
# 3-level σ-closure at T=3 (confC all-root, superfine).
# Checks the 6 safety invariants + 4 candidate structural conjuncts
# (SubNeverMissing, BundledHasCopy, StaleParentExcluded, SubPresenceUniform)
# hold across the full ~640M-state T=3 exhaustion.
#
# Differs from the liveness run in 3 ways ONLY:
#   - cfg = *_Is_confC_T3_mc.cfg (adds the 4 conjuncts as INVARIANTs)
#   - NO -dump (we only need the yes/no closure result, not the state set)
#   - NO liveness PROPERTY (safety/invariant checking only -> no SCC pass)
# Everything else mirrors the verified confC superfine T=3 run.
# ---------------------------------------------------------------------------
set -e
WORK=$HOME/kameSTMpaper
cd "$WORK"

SPEC=BundleUnbundle_3level_LLfree
CFG=${SPEC}_Is_confC_T3_mc.cfg
WORKERS=$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} - 2 ))
# No -dump => fingerprint/queue metadir only; put it on /work (large, fast).
METADIR=/work/$USER/tlc_${SLURM_JOB_ID:-local}
mkdir -p "$METADIR"

echo "node=$(hostname)  workers=$WORKERS  cfg=$CFG  metadir=$METADIR"
lscpu --extended | head -3 || true

java -XX:+UseParallelGC -Xmx200g \
     -cp "$WORK/tla2tools.jar" tlc2.TLC \
     -workers "$WORKERS" \
     -metadir "$METADIR" \
     -config "$CFG" \
     "$SPEC.tla"

echo "=== exit $? ==="
