#!/usr/bin/env bash
# Reproducible head-to-head: kame vs system / mimalloc / jemalloc on the
# tight alloc/free loop in `bench_loop` (one slot at a time:
# `malloc(N)` → write `p[0]` → `free(p)`).  Prints the two markdown
# tables that live in kamepoolalloc/README.md (1T and 4-process aggregate).
#
# Usage:
#   ./bench_compare.sh [--build-dir DIR] [--runs N] [--threads N] [--no-mt]
#                      [--mimalloc /path/to/libmimalloc.so]
#                      [--jemalloc /path/to/libjemalloc.so.2]
#
# Defaults:
#   build dir : ./build (relative to this script's parent — `tests/`)
#   runs      : 5 (median reported)
#   threads   : 4 (parallel processes for the MT table)
#   mimalloc  : auto-detect from common system paths
#   jemalloc  : auto-detect from common system paths
#
# Platforms: Linux (LD_PRELOAD, .so) and macOS (DYLD_INSERT_LIBRARIES, .dylib).
# Requires bash 4+ (for (( )) and arrays).  macOS ships bash 3.2; install a
# newer bash via Homebrew and invoke with `bash bench_compare.sh` if needed.
#
# Output format mirrors the README tables so you can paste directly.
# Exit codes:
#   0 success, 1 missing build, 2 missing comparator (still prints kame)

set -euo pipefail

# --- platform detection -------------------------------------------------------
OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
    LIB_EXT="dylib"
    PRELOAD_VAR="DYLD_INSERT_LIBRARIES"
else
    LIB_EXT="so"
    PRELOAD_VAR="LD_PRELOAD"
fi

# --- defaults ---------------------------------------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TESTS_DIR=$(dirname "$SCRIPT_DIR")
BUILD_DIR="$TESTS_DIR/build"
RUNS=5
NTHREADS=4
DO_MT=1
MI=""
JE=""

# Auto-detect comparators (first hit wins).
for cand in \
    /usr/lib/x86_64-linux-gnu/libmimalloc.so \
    /usr/lib/aarch64-linux-gnu/libmimalloc.so \
    /usr/local/lib/libmimalloc.so \
    /opt/local/lib/libmimalloc.dylib \
    /opt/homebrew/lib/libmimalloc.dylib ; do
    [ -f "$cand" ] && MI="$cand" && break
done
for cand in \
    /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
    /usr/local/lib/libjemalloc.so.2 \
    /opt/local/lib/libjemalloc.2.dylib \
    /opt/homebrew/lib/libjemalloc.2.dylib ; do
    [ -f "$cand" ] && JE="$cand" && break
done

# --- args -------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --runs)      RUNS="$2";      shift 2 ;;
        --threads)   NTHREADS="$2";  shift 2 ;;
        --no-mt)     DO_MT=0;        shift   ;;
        --mimalloc)  MI="$2";        shift 2 ;;
        --jemalloc)  JE="$2";        shift 2 ;;
        -h|--help)   sed -n '2,/^set -/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

BENCH="$BUILD_DIR/bench_loop"
KAME="$BUILD_DIR/libkamepoolalloc.$LIB_EXT"

if [ ! -x "$BENCH" ] || [ ! -f "$KAME" ]; then
    echo "ERROR: missing build artifacts under $BUILD_DIR" >&2
    echo "       expected $BENCH and $KAME" >&2
    echo "       cd $TESTS_DIR/build && cmake .. && make -j" >&2
    exit 1
fi

# --- helpers ----------------------------------------------------------------

# Extract M ops/s value from bench_loop output line.
# Uses awk instead of grep -oP (no Perl regex on macOS).
extract_rate() {
    awk -F'rate=' '{gsub(/M.*/, "", $2); print $2}'
}

# `python3 -c "import statistics; ..."` — used for median; fall back to a
# pure-bash sort + middle pick if python3 is missing.
median() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "import sys,statistics; d=[float(x) for x in sys.stdin.read().split()]; print(f'{statistics.median(d):.0f}')"
    else
        # bash fallback (works for odd-N samples; rounds toward zero)
        local sorted; sorted=$(tr ' ' '\n' | sort -n | grep -v '^$')
        local n; n=$(echo "$sorted" | wc -l)
        echo "$sorted" | sed -n "$(( (n+1)/2 ))p" | awk '{printf "%.0f\n", $1}'
    fi
}

# Run `bench_loop SIZE ITERS` once with optional preload; emit M ops/s.
# Uses the platform-appropriate preload variable (LD_PRELOAD / DYLD_INSERT_LIBRARIES).
one_run() {
    local preload="$1" size="$2" iters="$3"
    if [ -z "$preload" ]; then
        "$BENCH" "$size" "$iters" 2>/dev/null | extract_rate
    else
        env "$PRELOAD_VAR=$preload" "$BENCH" "$size" "$iters" 2>/dev/null \
            | extract_rate
    fi
}

# Median of $RUNS single-thread runs.
med_st() {
    local preload="$1" size="$2" iters="$3"
    local vals=""
    for ((i=0; i<RUNS; i++)); do
        vals="$vals $(one_run "$preload" "$size" "$iters")"
    done
    echo "$vals" | median
}

# One aggregate $NTHREADS-process run — sum of per-process rates.
mt_run_sum() {
    local preload="$1" size="$2" iters="$3"
    local tfs=()
    for ((i=0; i<NTHREADS; i++)); do
        local tf; tf=$(mktemp); tfs+=("$tf")
        if [ -z "$preload" ]; then
            "$BENCH" "$size" "$iters" >"$tf" 2>/dev/null &
        else
            env "$PRELOAD_VAR=$preload" "$BENCH" "$size" "$iters" >"$tf" 2>/dev/null &
        fi
    done
    wait
    local total=0
    for tf in "${tfs[@]}"; do
        local r; r=$(extract_rate <"$tf"); rm "$tf"
        total=$(awk -v a="$total" -v b="$r" 'BEGIN{printf "%.2f", a+b}')
    done
    echo "$total"
}

# Median of $RUNS multi-process aggregate runs.
med_mt() {
    local preload="$1" size="$2" iters="$3"
    local vals=""
    for ((i=0; i<RUNS; i++)); do
        vals="$vals $(mt_run_sum "$preload" "$size" "$iters")"
    done
    echo "$vals" | median
}

# Human-readable size label (e.g. 65536 → "64 KiB").
# numfmt is GNU coreutils (not available on macOS); fall back to awk.
size_label() {
    local sz=$1
    if command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec --suffix=B "$sz" 2>/dev/null || echo "${sz}B"
    else
        awk -v n="$sz" 'BEGIN{
            if (n >= 1048576) printf "%.0f MiB\n", n/1048576
            else if (n >= 1024) printf "%.0f KiB\n", n/1024
            else printf "%dB\n", n
        }'
    fi
}

# --- header -----------------------------------------------------------------
HEAD_SHA=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
HOST=$(uname -mn 2>/dev/null | tr -s ' ' '/')
echo "kamepoolalloc bench_compare: HEAD=$HEAD_SHA  host=$HOST  runs=$RUNS  os=$OS"
[ -z "$MI" ] && echo "  (no mimalloc found — column will be '-')"
[ -z "$JE" ] && echo "  (no jemalloc found — column will be '-')"
echo

# --- iteration count lookup (replaces declare -A for bash 3 compat) ---------
iters_1t() {
    case "$1" in
        64|1024)   echo 10000000 ;;
        16384)     echo 5000000  ;;
        4194304)   echo 500000   ;;
        *)         echo 2000000  ;;
    esac
}
iters_mt() {
    case "$1" in
        64)        echo 50000000 ;;
        16384)     echo 10000000 ;;
        65536)     echo 5000000  ;;
        1048576)   echo 2000000  ;;
        *)         echo 2000000  ;;
    esac
}

# --- 1T table ---------------------------------------------------------------
printf "## 1T (median of %d, M ops/s)\n\n" "$RUNS"
printf "| %-9s | %7s | %8s | %8s | %8s |\n" size system mimalloc jemalloc kame
printf "|-%-9s-|-%7s-|-%8s-|-%8s-|-%8s-|\n" \
       "---------" "-------" "--------" "--------" "--------"
for sz in 64 1024 16384 65536 262144 1048576 4194304; do
    iters=$(iters_1t "$sz")
    sys=$(med_st "" "$sz" "$iters")
    mi=$([ -n "$MI" ] && med_st "$MI" "$sz" "$iters" || echo "-")
    je=$([ -n "$JE" ] && med_st "$JE" "$sz" "$iters" || echo "-")
    km=$(med_st "$KAME" "$sz" "$iters")
    label=$(size_label "$sz")
    printf "| %-9s | %7s | %8s | %8s | %8s |\n" "$label" "$sys" "$mi" "$je" "$km"
done
echo

# --- MT table ---------------------------------------------------------------
if [ "$DO_MT" -eq 1 ]; then
    printf "## %d processes (aggregate, M ops/s, median of %d)\n\n" \
           "$NTHREADS" "$RUNS"
    printf "| %-9s | %7s | %8s | %8s | %8s |\n" size system mimalloc jemalloc kame
    printf "|-%-9s-|-%7s-|-%8s-|-%8s-|-%8s-|\n" \
           "---------" "-------" "--------" "--------" "--------"
    for sz in 64 16384 65536 1048576; do
        iters=$(iters_mt "$sz")
        sys=$(med_mt "" "$sz" "$iters")
        mi=$([ -n "$MI" ] && med_mt "$MI" "$sz" "$iters" || echo "-")
        je=$([ -n "$JE" ] && med_mt "$JE" "$sz" "$iters" || echo "-")
        km=$(med_mt "$KAME" "$sz" "$iters")
        label=$(size_label "$sz")
        printf "| %-9s | %7s | %8s | %8s | %8s |\n" "$label" "$sys" "$mi" "$je" "$km"
    done
    echo
fi
