#!/bin/bash
#SBATCH --job-name=l1_asp
#SBATCH --partition=i8cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --mem=350G
#SBATCH --output=l1_asp_%j.log
#SBATCH --error=l1_asp_%j.err

# Layer 1 (atomic_shared_ptr.tla) TLC verification on ohtaka.
#
# Usage:
#   CFG=atomic_shared_ptr_3thr_cas_mc.cfg sbatch run_atomic_shared_ptr.sh
#   CFG=atomic_shared_ptr_3thr_full_mc.cfg sbatch run_atomic_shared_ptr.sh
#
# Optional:
#   TAG=l1_3thr_cas  (output filename suffix; default = cfg basename)
#   MEM=180          (Xmx in GB; default 180 for i8cpu node)

set -euo pipefail

: "${CFG:?must set CFG (cfg filename)}"
: "${TAG:=${CFG%.cfg}}"
: "${MEM:=180}"

cd "$(dirname "$0")"

OUT="${TAG}_${SLURM_JOB_ID:-local}.log"

echo "=== Layer 1 TLC ==="
echo "CFG  = $CFG"
echo "TAG  = $TAG"
echo "MEM  = ${MEM}g"
echo "OUT  = $OUT"
echo "Spec = atomic_shared_ptr.tla"
date

java -XX:+UseParallelGC -Xmx${MEM}g -cp tla2tools.jar tlc2.TLC \
    -workers auto \
    -config "$CFG" \
    atomic_shared_ptr.tla > "$OUT" 2>&1

echo "=== Done ==="
date
grep -E "Model checking completed|Error|Invariant|violated" "$OUT" | head -5
grep "states generated" "$OUT" | tail -1
grep "depth of" "$OUT"
grep "^Finished" "$OUT"
