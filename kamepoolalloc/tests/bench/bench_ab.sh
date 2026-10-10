#!/usr/bin/env bash
# bench_ab.sh — interleaved A/B of allocator builds: one binary per
# workload, the arm switched by preload (LD_PRELOAD / DYLD_INSERT_LIBRARIES).
# Made for release comparisons — by default kamepoolalloc v1.1.0 against
# v1.2.0, each built from its tag with the recipe mimalloc-bench uses.
#
# Every repetition runs each workload under every arm back to back, and the
# arm order alternates between repetitions, so drift in machine state
# cancels instead of landing on one arm.  Reported: median (min–max) per
# arm and the ratio to the first arm.
#
#   loop     bench_loop, one thread, 64 B / 1 KiB / 16 KiB          M ops/s
#            (the single-thread hot path)
#   xthread  bench_xthread -w N -s 1024                             M frees/s
#   xmalloc  mimalloc-bench xmalloc-test -w N -s 64                 M frees/s
#   larson   mimalloc-bench larson ... N                            M ops/s
#   rptest   mimalloc-bench rptest N 0 1 2 500 1000 100 8 16000     M ops/CPU-s
#   xlat     bench_xlatency: producer malloc latency while a consumer frees
#            at random across 1 KiB x 200 K / 1 KiB x 1 M / 3000 B x 200 K
#            live blocks — median / p99.9 / max ns, and the rate
#
# Usage:
#   bench_ab.sh [--tags v1.1.0,v1.2.0] [--arm NAME=LIB]... [--reps N]
#               [--threads "2 4"] [--only "loop xthread ..."] [--quick]
#               [--work DIR] [--mbench DIR] [--xlat "1024x200000 ..."]
#               [--xlat-cpu P,C] [--jobs N] [--build-only | --run-only]
#
#   --tags     kamepoolalloc tags to build as arms ("" for none)
#   --arm      an extra arm; LIB empty means no preload (the system malloc),
#              e.g. --arm sys=  --arm mi=/path/libmimalloc.so
#   --reps     repetitions (default 5; 1 with --quick)
#   --threads  thread counts for xthread / xmalloc / larson / rptest
#              (xthread and xmalloc run N producers + N consumers)
#   --quick    short runs and small live sets: a smoke test of the setup
#   --work     work directory (default ./bench-ab): clones, builds, results
#   --mbench   an existing mimalloc-bench checkout; its out/bench binaries
#              are used if built, else its sources (default: a fresh clone)
#   --xlat     bench_xlatency configurations, SIZExLIVE
#   --xlat-cpu pin bench_xlatency's producer and consumer (Linux only)
#   --jobs     build parallelism (default: min(cpus, 8))
#
# Results land in WORK/results-HOST-TIME/: raw.tsv (every run),
# summary.md (the table), provenance.txt (machine, compilers, arms), logs/.
#
# macOS (M5 Ultra and the like), from any directory:
#   kamepoolalloc/tests/bench/bench_ab.sh --threads "4 8"
# Keep thread counts within the performance cores; macOS cannot pin.
#
# Ohtaka: build on the login node, measure on a compute node — never the
# reverse (the measurement refuses to run outside a SLURM job there):
#   CC=~/llvm-install/bin/clang CXX=~/llvm-install/bin/clang++ \
#     ~/kame/kamepoolalloc/tests/bench/bench_ab.sh --build-only \
#     --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench
#   srun -p i8cpu --time=02:00:00 --exclusive \
#     ~/kame/kamepoolalloc/tests/bench/bench_ab.sh --run-only \
#     --work ~/kame-claude/bench-ab --mbench ~/mimalloc-bench \
#     --threads "4 16 64" --xlat-cpu 0,1
#
# Licensed under Apache-2.0 OR GPL-2.0-or-later, as the rest of the tree.

set -o pipefail

case "$(uname -s)" in
    Darwin) OS=Darwin; EXT=dylib; PRELOAD_VAR=DYLD_INSERT_LIBRARIES ;;
    Linux)  OS=Linux;  EXT=so;    PRELOAD_VAR=LD_PRELOAD ;;
    *) echo "bench_ab.sh: Linux and macOS only (the arms are preloaded)" >&2
       exit 1 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KPA_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

TAGS="v1.1.0,v1.2.0"
EXTRA_ARMS=()
REPS=""
THREADS="2 4"
ONLY="loop xthread xmalloc larson rptest xlat"
QUICK=0
WORK="$PWD/bench-ab"
MBENCH=""
XLAT_CONFIGS=""
XLAT_CPU=""
JOBS=""
MODE=all
KP_URL=https://github.com/northriv/kamepoolalloc
MB_URL=https://github.com/daanx/mimalloc-bench

log() { printf '%s\n' "$*" >&2; }
die() { log "bench_ab.sh: $*"; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --tags)       TAGS="$2"; shift 2 ;;
        --arm)        EXTRA_ARMS+=("$2"); shift 2 ;;
        --reps)       REPS="$2"; shift 2 ;;
        --threads)    THREADS="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        --quick)      QUICK=1; shift ;;
        --work)       WORK="$2"; shift 2 ;;
        --mbench)     MBENCH="$2"; shift 2 ;;
        --xlat)       XLAT_CONFIGS="$2"; shift 2 ;;
        --xlat-cpu)   XLAT_CPU="$2"; shift 2 ;;
        --jobs)       JOBS="$2"; shift 2 ;;
        --build-only) MODE=build; shift ;;
        --run-only)   MODE=run; shift ;;
        -h|--help)    sed -n '2,/^# Licensed/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

if [ "$QUICK" = 1 ]; then
    : "${REPS:=1}"
    LOOP_ITERS=2000000; XT_SECS=1; LARSON_SECS=1; RPT_LOOPS=50
    XLAT_N=200000; XLAT_T=2; : "${XLAT_CONFIGS:=1024x50000 1024x200000}"
else
    : "${REPS:=5}"
    LOOP_ITERS=20000000; XT_SECS=5; LARSON_SECS=5; RPT_LOOPS=500
    XLAT_N=2000000; XLAT_T=10
    : "${XLAT_CONFIGS:=1024x200000 1024x1000000 3000x200000}"
fi
if [ -z "$JOBS" ]; then
    if [ "$OS" = Darwin ]; then JOBS=$(sysctl -n hw.ncpu); else JOBS=$(nproc); fi
    [ "$JOBS" -gt 8 ] && JOBS=8
fi

mkdir -p "$WORK" || die "cannot create $WORK"
WORK=$(cd "$WORK" && pwd)
if [ -n "$MBENCH" ]; then
    MBENCH=$(cd "$MBENCH" 2>/dev/null && pwd) || die "--mbench: no such directory"
fi
BENCH_BUILD="$WORK/bench-build"
BENCH_BIN="$BENCH_BUILD/tests"
MB_BUILD="$WORK/mbench-build"

# ---- arms --------------------------------------------------------------
ARM_NAMES=(); ARM_LIBS=(); ARM_TAGS=()
IFS=, read -r -a TAG_LIST <<< "$TAGS"
for t in "${TAG_LIST[@]}"; do
    [ -n "$t" ] || continue
    ARM_NAMES+=("kp-$t"); ARM_LIBS+=("$WORK/kp-$t/out/libkamepoolalloc.$EXT")
    ARM_TAGS+=("$t")
done
for a in "${EXTRA_ARMS[@]}"; do
    case "$a" in *=*) ;; *) die "--arm wants NAME=LIB, got: $a" ;; esac
    ARM_NAMES+=("${a%%=*}"); ARM_LIBS+=("${a#*=}"); ARM_TAGS+=("")
done
NARMS=${#ARM_NAMES[@]}
[ "$NARMS" -ge 1 ] || die "no arms (give --tags or --arm)"

# ---- build -------------------------------------------------------------
build_kp_tag() {
    local tag=$1 dir="$WORK/kp-$1" blog="$WORK/kp-$1.log"
    if [ -f "$dir/out/libkamepoolalloc.$EXT" ]; then
        log "kp $tag: already built"
        return 0
    fi
    log "kp $tag: clone and build"
    rm -rf "$dir"
    git clone --quiet --depth 1 --branch "$tag" "$KP_URL" "$dir" >"$blog" 2>&1 \
        || die "cloning $tag failed, see $blog"
    # mimalloc-bench's recipe, minus the test scaffold: the option only adds
    # tests/ after the library target is defined, so the library is the same.
    cmake -S "$dir" -B "$dir/out" -DCMAKE_BUILD_TYPE=Release \
          -DKAMEPOOLALLOC_BUILD_TESTS=OFF >>"$blog" 2>&1 \
        && cmake --build "$dir/out" --parallel "$JOBS" >>"$blog" 2>&1 \
        || die "building $tag failed, see $blog"
}

build_benches() {
    local blog="$WORK/bench-build.log" t
    log "benches: build from $KPA_DIR"
    cmake -S "$KPA_DIR" -B "$BENCH_BUILD" -DCMAKE_BUILD_TYPE=Release \
        >"$blog" 2>&1 || die "configuring the benches failed, see $blog"
    for t in bench_loop bench_xthread bench_xlatency; do
        cmake --build "$BENCH_BUILD" --target "$t" --parallel "$JOBS" \
            >>"$blog" 2>&1 || die "building $t failed, see $blog"
    done
}

# Sets MB_BIN (and MB_SRC for provenance).  With $1 = build, builds what is
# missing; otherwise only locates.
mbench() {
    local blog="$WORK/mbench-build.log" t
    if [ -n "$MBENCH" ]; then
        MB_SRC="$MBENCH"
        if [ -x "$MBENCH/out/bench/xmalloc-test" ] && [ -x "$MBENCH/out/bench/larson" ] \
           && [ -x "$MBENCH/out/bench/rptest" ]; then
            MB_BIN="$MBENCH/out/bench"
            return 0
        fi
    else
        MB_SRC="$WORK/mimalloc-bench"
    fi
    MB_BIN="$MB_BUILD"
    if [ -x "$MB_BIN/xmalloc-test" ] && [ -x "$MB_BIN/larson" ] && [ -x "$MB_BIN/rptest" ]; then
        return 0
    fi
    [ "$1" = build ] || return 1
    if [ ! -d "$MB_SRC/bench" ]; then
        log "mimalloc-bench: clone"
        git clone --quiet --depth 1 "$MB_URL" "$MB_SRC" >"$blog" 2>&1 \
            || die "cloning mimalloc-bench failed, see $blog"
    fi
    log "mimalloc-bench: build xmalloc-test, larson, rptest"
    cmake -S "$MB_SRC/bench" -B "$MB_BUILD" >>"$blog" 2>&1 \
        || die "configuring mimalloc-bench failed, see $blog"
    for t in xmalloc-test larson rptest; do
        cmake --build "$MB_BUILD" --target "$t" --parallel "$JOBS" >>"$blog" 2>&1 \
            || die "building $t failed, see $blog"
    done
}

wants() { case " $ONLY " in *" $1 "*) return 0 ;; esac; return 1; }
need_mbench() { wants xmalloc || wants larson || wants rptest; }

if [ "$MODE" != run ]; then
    for i in $(seq 0 $((NARMS - 1))); do
        [ -n "${ARM_TAGS[$i]}" ] && build_kp_tag "${ARM_TAGS[$i]}"
    done
    build_benches
    need_mbench && mbench build
    if [ "$MODE" = build ]; then
        log "built under $WORK"
        exit 0
    fi
fi

# ---- run ---------------------------------------------------------------
if [ -e /home/system/bin/check_usage ] && [ -z "$SLURM_JOB_ID" ]; then
    die "this looks like an Ohtaka login node: measure under srun (see --help)"
fi
for t in bench_loop bench_xthread bench_xlatency; do
    [ -x "$BENCH_BIN/$t" ] || die "$BENCH_BIN/$t missing — build first (drop --run-only)"
done
if need_mbench; then
    mbench locate || die "mimalloc-bench binaries missing — build first (drop --run-only)"
fi

# The preload must take, or every arm silently measures the system malloc.
for i in $(seq 0 $((NARMS - 1))); do
    lib=${ARM_LIBS[$i]}
    [ -n "$lib" ] || continue
    [ -f "$lib" ] || die "arm ${ARM_NAMES[$i]}: $lib not found — build first"
    case "$lib" in
        *kamepoolalloc*)
            # Captured first: grep -q quitting early would SIGPIPE the bench,
            # and pipefail would report that as a failed preload.
            probe=$(env "$PRELOAD_VAR=$lib" KAME_POOL_VERBOSE=1 \
                        "$BENCH_BIN/bench_loop" 64 1000 2>&1)
            case "$probe" in
                *'Reserve swap space'*) ;;
                *) die "arm ${ARM_NAMES[$i]}: the preload did not take (no activation banner)" ;;
            esac ;;
    esac
done

TIMEOUT=""
if command -v timeout >/dev/null 2>&1; then TIMEOUT="timeout 1800"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT="gtimeout 1800"; fi

RES="$WORK/results-$(hostname -s 2>/dev/null || hostname)-$(date '+%Y%m%d-%H%M%S')"
mkdir -p "$RES/logs" "$RES/cwd" || die "cannot create $RES"
RAW="$RES/raw.tsv"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-16
    else shasum -a 256 "$1" | cut -c1-16; fi
}
cmake_compiler() {
    sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$1/CMakeCache.txt" 2>/dev/null | head -1
}

provenance() {
    local i c
    echo "date      $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "host      $(hostname)"
    echo "kernel    $(uname -srm)"
    if [ "$OS" = Darwin ]; then
        echo "cpu       $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)P + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null)E, $(( $(sysctl -n hw.memsize) >> 30 )) GiB, $(sysctl -n hw.model)"
        echo "os        $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
        echo "power     $(pmset -g batt 2>/dev/null | head -1)"
    else
        echo "cpu       $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1), $(nproc) cpus, $(lscpu 2>/dev/null | sed -n 's/^NUMA node(s): *//p') NUMA nodes"
        echo "thp       $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
        echo "governor  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
        echo "libc      $(ldd --version 2>&1 | head -1)"
        [ -n "$SLURM_JOB_ID" ] && echo "slurm     job $SLURM_JOB_ID, partition ${SLURM_JOB_PARTITION:-?}, node ${SLURMD_NODENAME:-?}"
    fi
    c=$(cmake_compiler "$BENCH_BUILD")
    echo "benches   $KPA_DIR @ $(git -C "$KPA_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'), $c ($("$c" --version 2>/dev/null | head -1))"
    for i in $(seq 0 $((NARMS - 1))); do
        local lib=${ARM_LIBS[$i]}
        if [ -z "$lib" ]; then
            echo "arm       ${ARM_NAMES[$i]}: no preload (system malloc)"
        else
            local extra=""
            if [ -n "${ARM_TAGS[$i]}" ]; then
                extra=", commit $(git -C "$WORK/kp-${ARM_TAGS[$i]}" rev-parse --short HEAD 2>/dev/null), built by $(cmake_compiler "$WORK/kp-${ARM_TAGS[$i]}/out")"
            fi
            echo "arm       ${ARM_NAMES[$i]}: $lib ($(wc -c <"$lib" | tr -d ' ') B, sha256 $(sha256 "$lib")$extra)"
        fi
    done
    need_mbench && echo "mbench    $MB_SRC @ $(git -C "$MB_SRC" rev-parse --short HEAD 2>/dev/null || echo '?'), binaries in $MB_BIN"
    echo "params    reps=$REPS threads=\"$THREADS\" only=\"$ONLY\" quick=$QUICK xlat=\"$XLAT_CONFIGS\" xlat_cpu=${XLAT_CPU:-none} loop_iters=$LOOP_ITERS"
}

# ---- jobs: (bench, config, command) ---------------------------------------
JOB_BENCH=(); JOB_CFG=(); JOB_CMD=()
add_job() {
    JOB_BENCH+=("$1"); JOB_CFG+=("$2"); shift 2
    JOB_CMD+=("$(printf '%q ' "$@")")
}
if wants loop; then
    for s in 64 1024 16384; do
        add_job loop "${s}B" "$BENCH_BIN/bench_loop" "$s" "$LOOP_ITERS"
    done
fi
for n in $THREADS; do
    wants xthread && add_job xthread "w$n,1KiB" "$BENCH_BIN/bench_xthread" -w "$n" -s 1024 -t "$XT_SECS"
    wants xmalloc && add_job xmalloc "w$n" "$MB_BIN/xmalloc-test" -w "$n" -t "$XT_SECS" -s 64
    wants larson  && add_job larson "t$n" "$MB_BIN/larson" "$LARSON_SECS" 8 1000 5000 100 4141 "$n"
    wants rptest  && add_job rptest "t$n" "$MB_BIN/rptest" "$n" 0 1 2 "$RPT_LOOPS" 1000 100 8 16000
done
if wants xlat; then
    for c in $XLAT_CONFIGS; do
        sz=${c%%x*}; lv=${c#*x}
        if [ -n "$XLAT_CPU" ]; then
            add_job xlat "${sz}Bx${lv}" "$BENCH_BIN/bench_xlatency" -s "$sz" -l "$lv" \
                -n "$XLAT_N" -t "$XLAT_T" -f 900 --cpu "$XLAT_CPU"
        else
            add_job xlat "${sz}Bx${lv}" "$BENCH_BIN/bench_xlatency" -s "$sz" -l "$lv" \
                -n "$XLAT_N" -t "$XLAT_T" -f 900
        fi
    done
fi
NJOBS=${#JOB_BENCH[@]}
[ "$NJOBS" -ge 1 ] || die "nothing to run (check --only)"

# Prints "metric value" lines parsed from a run's log.
parse() {
    case "$1" in
        loop)    sed -n 's/.*rate=\([0-9.]*\)M ops\/s.*/Mops \1/p' "$2" | head -1 ;;
        xthread|xmalloc)
                 sed -n 's/.*free\/sec: \([0-9.]*\) M.*/Mfree \1/p' "$2" | head -1 ;;
        larson)  sed -n 's/.*Throughput = *\([0-9.]*\) operations per second.*/\1/p' "$2" \
                     | head -1 | awk '{ printf "Mops %.3f\n", $1 / 1e6 }' ;;
        rptest)  sed -n 's/.*[^0-9]\([0-9][0-9]*\) memory ops\/CPU second.*/\1/p' "$2" \
                     | head -1 | awk '{ printf "Mops_cpu %.3f\n", $1 / 1e6 }' ;;
        xlat)    grep '^\[bench_xlatency\]' "$2" | head -1 | tr ' ' '\n' | awk -F= '
                     $1 == "med_ns" || $1 == "p999_ns" || $1 == "max_ns" || $1 == "rate_M" { print $1, $2 }' ;;
    esac
}

run_one() {    # arm-index logfile command...
    local i=$1 logf=$2; shift 2
    if [ -n "${ARM_LIBS[$i]}" ]; then
        $TIMEOUT env "$PRELOAD_VAR=${ARM_LIBS[$i]}" "$@" >"$logf" 2>&1
    else
        $TIMEOUT "$@" >"$logf" 2>&1
    fi
}

provenance | tee "$RES/provenance.txt" >&2
printf '# rep\tbench\tconfig\tarm\tmetric\tvalue\tstatus\n' >"$RAW"
cd "$RES/cwd" || die "cannot enter $RES/cwd"    # rptest writes files into its cwd

for rep in $(seq 1 "$REPS"); do
    if [ $((rep % 2)) = 1 ]; then ORDER=$(seq 0 $((NARMS - 1)))
    else ORDER=$(seq $((NARMS - 1)) -1 0); fi
    for j in $(seq 0 $((NJOBS - 1))); do
        b=${JOB_BENCH[$j]}; c=${JOB_CFG[$j]}
        for a in $ORDER; do
            arm=${ARM_NAMES[$a]}
            logf="$RES/logs/r${rep}_${b}_${c//,/_}_${arm}.log"
            eval "run_one $a \"\$logf\" ${JOB_CMD[$j]}"
            rc=$?
            status=ok; [ "$rc" = 0 ] || status="exit=$rc"
            out=$(parse "$b" "$logf")
            if [ -z "$out" ]; then
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rep" "$b" "$c" "$arm" "-" NA "$status" >>"$RAW"
                log "[rep $rep/$REPS] $b $c $arm: no result ($status), see $logf"
                continue
            fi
            echo "$out" | while read -r m v; do
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rep" "$b" "$c" "$arm" "$m" "$v" "$status" >>"$RAW"
            done
            log "[rep $rep/$REPS] $b $c $arm: $(echo $out)"
        done
    done
done

# ---- summary -------------------------------------------------------------
ARMS_CSV=$(IFS=,; echo "${ARM_NAMES[*]}")
if [ "$(python3 -c 'print(7)' 2>/dev/null)" = 7 ]; then
    # UTF-8 forced: a C locale (common under srun) makes python3.6 refuse
    # to print the table's dashes and arrows.
    PYTHONIOENCODING=utf-8 python3 - "$RAW" "$ARMS_CSV" <<'PY' | tee "$RES/summary.md"
import sys, statistics
from collections import OrderedDict
raw, arms = sys.argv[1], sys.argv[2].split(',')
table = OrderedDict()
with open(raw) as f:
    for line in f:
        if line.startswith('#'):
            continue
        rep, bench, cfg, arm, metric, value, status = line.rstrip('\n').split('\t')
        if metric == '-':
            continue
        try:
            v = float(value)
        except ValueError:
            v = None
        table.setdefault((bench, cfg, metric), {}).setdefault(arm, []).append(v)

def fmt(x):
    if x >= 100:
        return '%.0f' % x
    if x >= 10:
        return '%.1f' % x
    return '%.3g' % x

head = ['bench', 'config', 'metric'] + arms + ['%s / %s' % (a, arms[0]) for a in arms[1:]]
print('| ' + ' | '.join(head) + ' |')
print('|' + '---|' * len(head))
for (bench, cfg, metric), per in table.items():
    cells, meds = [], []
    for a in arms:
        vals = per.get(a, [])
        good = [v for v in vals if v is not None]
        if not good:
            cells.append('n/a')
            meds.append(None)
            continue
        m = statistics.median(good)
        s = fmt(m)
        if len(good) > 1:
            s += ' (%s–%s)' % (fmt(min(good)), fmt(max(good)))
        if len(good) < len(vals):
            s += ' [%d n/a]' % (len(vals) - len(good))
        cells.append(s)
        meds.append(m)
    ratios = []
    for m in meds[1:]:
        ratios.append('%.2f×' % (m / meds[0]) if m is not None and meds[0] else 'n/a')
    name = metric + (' ↓' if metric.endswith('_ns') else ' ↑')
    print('| ' + ' | '.join([bench, cfg, name] + cells + ratios) + ' |')
PY
else
    log "python3 not found: the raw results are in $RAW"
fi
log "results: $RES"
