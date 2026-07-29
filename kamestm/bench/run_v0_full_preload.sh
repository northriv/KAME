#!/bin/bash
#
# run_v0_full_preload.sh — measure the legacy v0 STM with a runtime-injected
# modern allocator, the SIMPLE way (no header overlay, no composability hell).
#
# IDEA: instead of grafting v0's transaction.{h,_impl.h} onto the
# current split tree (which fails: force_intrusive_ref / ProcessCounter), check
# out v0 in FULL so everything is self-consistent, add ONLY the mixed-test
# source, build with -DUSE_KAME_ALLOCATOR=OFF (STM allocates via operator new ->
# libc malloc), and inject the modern allocator at runtime via
# DYLD_INSERT_LIBRARIES. Run the `full` variant the same way -> both variants
# use the IDENTICAL injected allocator, so the v0-vs-full comparison has no
# allocator confound.
#
#   VARIANT=v0         : git worktree @ KAME_V0_FULL (1d49b07c), full tree + test graft
#   VARIANT=no_backoff : same ref as full, but compiled with -DKAME_STM_DISABLE_BACKOFF=1
#   VARIANT=full       : current KAME_ROOT tree (test already present on master)
#   all three          : -DUSE_KAME_ALLOCATOR=OFF + DYLD_INSERT_LIBRARIES / LD_PRELOAD
#
# USAGE
#   cd bench
#   VARIANT=v0         LABEL=M4macmini ./run_v0_full_preload.sh
#   VARIANT=no_backoff LABEL=M4macmini ./run_v0_full_preload.sh
#   VARIANT=full       LABEL=M4macmini ./run_v0_full_preload.sh
#
# KNOBS (env)
#   KAME_ROOT     ~/KAME    (current tree)
#   VARIANT       v0 | full                        (default v0)
#   KAME_V0_FULL  1d49b07c                          (full v0 commit to check out)
#   TEST_PATCH    ../cpp_test_drafts/kame_payload_integrity_mixed_test.patch
#   KAME_DYLIB    auto = first libkamepoolalloc.dylib under $KAME_ROOT/build*
#   FLAT_NS       0  (set 1 to add DYLD_FORCE_FLAT_NAMESPACE=1 if interception
#                     is not observed — needed when the dylib overrides operator
#                     new by symbol rather than malloc-zone/__interpose)
#   WORKTREE_DIR  $KAME_ROOT/../kame-v0full
#   CMAKE_SRC     tests        (cmake -S <dir>; falls back to ".")
#   TESTS         "payload_integrity_mixed payload_integrity_3level_mixed"
#   HOST_LABEL / LABEL / $1 -> label; output <HOST_LABEL>_<VARIANT>_preload.tsv
#   + run_kame_mixed_mac.sh knobs (THREADS_LIST, K_LIST, RUNS, PAYLOAD, …)
#
set -u

strip_ws() { printf '%s' "${1:-}" | sed 's/[[:space:]]//g; s/　//g'; }

KAME_ROOT=${KAME_ROOT:-$HOME/KAME}
VARIANT=$(strip_ws "${VARIANT:-v0}")
KAME_V0_FULL=$(strip_ws "${KAME_V0_FULL:-1d49b07c}")
FULL_REF=$(strip_ws "${FULL_REF:-HEAD}")   # ref to use for VARIANT=full
BENCH=$(cd "$(dirname "$0")" && pwd)
# Patch supplying the mixed-test sources for pre-split (v0) checkouts.
# Looked up next to this script first (the published kamestm/bench layout),
# then in ../cpp_test_drafts (the paper-repository layout).
if [ -z "${TEST_PATCH:-}" ]; then
    TEST_PATCH=$BENCH/kame_payload_integrity_mixed_test.patch
    [ -f "$TEST_PATCH" ] || TEST_PATCH=$BENCH/../cpp_test_drafts/kame_payload_integrity_mixed_test.patch
fi
# no_backoff uses the same source tree as full (same ref, different compile flags),
# so they share a worktree to avoid git's "commit already checked out" error.
_wt_variant=$VARIANT; [ "$VARIANT" = no_backoff ] && _wt_variant=full
WORKTREE_DIR=${WORKTREE_DIR:-$KAME_ROOT/../kame-${_wt_variant}-preload}
CMAKE_SRC=${CMAKE_SRC:-tests}
TESTS=${TESTS:-"payload_integrity_mixed payload_integrity_3level_mixed"}
FLAT_NS=${FLAT_NS:-0}
HOST_LABEL=$(strip_ws "${1:-${HOST_LABEL:-${LABEL:-$(hostname -s)}}}" | tr -c '[:alnum:]_' '_')

die() { echo "ERROR: $*" >&2; exit 1; }

[ -d "$KAME_ROOT/.git" ] || die "KAME_ROOT=$KAME_ROOT is not a git repo"
case "$VARIANT" in v0|no_backoff|full) ;; *) die "VARIANT must be v0, no_backoff, or full (got $VARIANT)";; esac

# ----------------------------------------------------------------------------
# 1) Detached worktree at this variant's ref (keeps KAME_ROOT untouched), then
#    graft the mixed test if absent. v0/no_backoff/full all use the same flow;
#    they differ only in the commit checked out and the compile flags.
# ----------------------------------------------------------------------------
case "$VARIANT" in
    v0)          REF=$(git -C "$KAME_ROOT" rev-parse "$KAME_V0_FULL") || die "bad KAME_V0_FULL=$KAME_V0_FULL" ;;
    no_backoff)  REF=$(git -C "$KAME_ROOT" rev-parse "$FULL_REF")     || die "bad FULL_REF=$FULL_REF" ;;
    full)        REF=$(git -C "$KAME_ROOT" rev-parse "$FULL_REF")     || die "bad FULL_REF=$FULL_REF" ;;
esac
echo "[v0full] VARIANT=$VARIANT  ref=$REF"
if [ -e "$WORKTREE_DIR" ]; then
    git -C "$WORKTREE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || die "$WORKTREE_DIR exists but is not a git worktree"
    echo "[v0full] reusing worktree $WORKTREE_DIR, resetting to $REF"
    git -C "$WORKTREE_DIR" checkout -f --detach "$REF" || die "checkout failed"
else
    git -C "$KAME_ROOT" worktree add --detach "$WORKTREE_DIR" "$REF" || die "worktree add failed"
fi
SRCTREE="$WORKTREE_DIR"

# ---- graft the mixed-test sources if not native to this checkout ----
# Detect by git TRACKED-ness, not a content grep: on master the test is a
# committed (tracked) file in kamestm/tests/ and CMake builds it natively;
# on pre-split branches it is absent. CRUCIAL for worktree reuse — `git
# checkout -f` resets tracked files (so a prior graft's CMakeLists targets
# are GONE) but leaves the untracked grafted .cpp behind; a content grep
# would see that stray .cpp and wrongly skip the re-graft, producing a tree
# with the .cpp but no build target (the "binary not found" failure).
has_mixed_test=0
mt=transaction_payload_integrity_mixed_test.cpp
if git -C "$SRCTREE" ls-files --error-unmatch "kamestm/tests/$mt" >/dev/null 2>&1 \
   || git -C "$SRCTREE" ls-files --error-unmatch "tests/$mt" >/dev/null 2>&1; then
    has_mixed_test=1
    echo "[v0full] mixed test is native (tracked) in this checkout — no graft needed"
fi
if [ "$has_mixed_test" = 0 ]; then
    tcmk="$SRCTREE/tests/CMakeLists.txt"
    [ -f "$tcmk" ] || die "no tests/CMakeLists.txt in the worktree (set CMAKE_SRC?)"
    # (a) materialize the two NEW test .cpp from the patch only — skip the
    #     CMakeLists hunk (its context need not match the target branch).
    #     On worktree reuse the .cpp may already exist from a prior run;
    #     remove them first so git-apply doesn't choke on "already exists".
    for f in transaction_payload_integrity_mixed_test.cpp \
             transaction_payload_integrity_3level_mixed_test.cpp; do
        rm -f "$SRCTREE/tests/$f"
    done
    if ! git -C "$SRCTREE" apply --include='*mixed_test.cpp' \
                                 --whitespace=nowarn "$TEST_PATCH" 2>/tmp/v0full_apply.$$; then
        echo "[v0full] could not extract test sources from $TEST_PATCH:" >&2
        sed 's/^/    /' /tmp/v0full_apply.$$ >&2; rm -f /tmp/v0full_apply.$$
        die "graft failed — apply $TEST_PATCH inside $SRCTREE by hand, then re-run."
    fi
    rm -f /tmp/v0full_apply.$$
    echo "[v0full] created tests/transaction_payload_integrity_{mixed,3level_mixed}_test.cpp"

    # (b) append CMake targets by MIRRORING an existing transaction test.
    tmpl=$(grep -m1 -E 'add_executable\(transaction_payload_integrity_test ' "$tcmk" \
           || grep -m1 -E 'add_executable\(transaction_[A-Za-z0-9_]*test ' "$tcmk" || true)
    [ -n "$tmpl" ] || die "found no 'add_executable(transaction_*_test ...)' in $tcmk to \
mirror; add the two mixed-test targets by hand."
    tmpl_tgt=$(printf '%s' "$tmpl" | sed -E 's/^add_executable\(([^ )]+).*/\1/')
    extra=$(printf '%s' "$tmpl" | sed -E 's/^add_executable\([^ ]+[[:space:]]+[^ )]+[[:space:]]*//; s/\)[[:space:]]*$//')
    libline=$(grep -m1 -E "target_link_libraries\(${tmpl_tgt}[[:space:]]" "$tcmk" || true)
    {
        echo ""
        echo "# --- appended by run_v0_full_preload.sh (mixed tests, mirroring ${tmpl_tgt}) ---"
        for t in transaction_payload_integrity_mixed_test transaction_payload_integrity_3level_mixed_test; do
            echo "add_executable($t $t.cpp $extra)"
            [ -n "$libline" ] && printf '%s\n' "$libline" | sed "s/${tmpl_tgt}/$t/"
        done
    } >> "$tcmk"
    echo "[v0full] appended mixed-test targets (mirrored '$tmpl_tgt'; extra sources: '${extra:-none}')"
fi

# ----------------------------------------------------------------------------
# 2) Configure + build with USE_KAME_ALLOCATOR=OFF (STM -> operator new -> malloc)
# ----------------------------------------------------------------------------
bld="$SRCTREE/build-${VARIANT}preload"
rm -rf "$bld"
src="$CMAKE_SRC"; [ -e "$SRCTREE/$src/CMakeLists.txt" ] || src="."
extra_cflags="-O3 -DNDEBUG${OHTAKA_CXX_EXTRA:+ $OHTAKA_CXX_EXTRA}"
if [ "$VARIANT" = no_backoff ]; then
    extra_cflags="$extra_cflags -DKAME_STM_DISABLE_BACKOFF=1"
fi
cmake_extra=""
[ -n "${CC:-}" ]  && cmake_extra="$cmake_extra -DCMAKE_C_COMPILER=$CC"
[ -n "${CXX:-}" ] && cmake_extra="$cmake_extra -DCMAKE_CXX_COMPILER=$CXX"
echo "[v0full] cmake -S $src -B $(basename "$bld") -DUSE_KAME_ALLOCATOR=OFF CXX_FLAGS='$extra_cflags'${cmake_extra:+ $cmake_extra}"
# shellcheck disable=SC2086
( cd "$SRCTREE" && cmake -S "$src" -B "$(basename "$bld")" \
        -DCMAKE_BUILD_TYPE=Release -DUSE_KAME_ALLOCATOR=OFF \
        -DCMAKE_CXX_FLAGS="$extra_cflags" \
        $cmake_extra \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null ) \
    || die "cmake configure failed.
  If VARIANT=v0 and it complains about support_SRCS/Threads or the test build,
  the v0-era tests/CMakeLists.txt differs — apply $TEST_PATCH manually, or set
  CMAKE_SRC=. / TESTS=payload_integrity_mixed."
( cd "$SRCTREE" && cmake --build "$(basename "$bld")" -j -- VERBOSE=1 ) || die "build failed — inspect $bld"

# ---- build audit: show the actual compile + link commands for the mixed test ----
echo "[v0full] --- build audit ---"
cc="$bld/compile_commands.json"
if [ -f "$cc" ]; then
    echo "[v0full] compile flags (from compile_commands.json):"
    grep -A2 "payload_integrity_mixed_test.cpp" "$cc" | grep '"command"' | head -2 | sed 's/^/  /'
fi
probe_bin="$bld/transaction_payload_integrity_mixed_test"
[ -f "$probe_bin" ] || probe_bin=$(find "$bld" -name transaction_payload_integrity_mixed_test -type f 2>/dev/null | head -1)
if [ -n "$probe_bin" ] && [ -f "$probe_bin" ]; then
    echo "[v0full] linked libs (otool -L / ldd):"
    if command -v otool >/dev/null 2>&1; then
        otool -L "$probe_bin" 2>/dev/null | sed 's/^/  /'
    elif command -v ldd >/dev/null 2>&1; then
        ldd "$probe_bin" 2>/dev/null | sed 's/^/  /'
    fi
    echo "[v0full] binary size: $(ls -la "$probe_bin" | awk '{print $5}') bytes"
fi
echo "[v0full] --- end audit ---"

# ----------------------------------------------------------------------------
# 3) Locate the mixed-test binary.
# ----------------------------------------------------------------------------
bindir=""
for c in "$bld" "$bld/tests" "$bld/kamestm-tests" "$bld/kame-tests"; do
    [ -x "$c/transaction_payload_integrity_mixed_test" ] && { bindir="$c"; break; }
done
[ -n "$bindir" ] || die "mixed-test binary not found under $bld (built: \
$(find "$bld" -name 'transaction_*_test' -maxdepth 3 2>/dev/null | tr '\n' ' '))"
echo "[v0full] BIN_DIR: $bindir"

have_tests=""
for T in $TESTS; do
    [ -x "$bindir/transaction_${T}_test" ] && have_tests="$have_tests $T" \
        || echo "[v0full] note: transaction_${T}_test absent — skipping"
done
have_tests=$(echo "$have_tests" | sed 's/^ *//')
[ -n "$have_tests" ] || die "no mixed-test binaries to run"

# ----------------------------------------------------------------------------
# 4) Resolve the modern allocator dylib + verify injection actually happens.
# ----------------------------------------------------------------------------
# Find the allocator shared library: .dylib (macOS) or .so (Linux/ohtaka).
if [ -z "${KAME_DYLIB:-}" ]; then
    KAME_DYLIB=$(find "$KAME_ROOT" \( -name libkamepoolalloc.dylib -o -name libkamepoolalloc.so \) 2>/dev/null | head -1)
fi
[ -n "${KAME_DYLIB:-}" ] && [ -f "$KAME_DYLIB" ] \
    || die "libkamepoolalloc.{dylib,so} not found. Build it in $KAME_ROOT (e.g.
  cmake -S tests -B build && cmake --build build -j) or pass KAME_DYLIB=<path>."
echo "[v0full] injecting allocator: $KAME_DYLIB"

# Platform-appropriate injection env var.
case "$(uname -s)" in
    Darwin) inject="DYLD_INSERT_LIBRARIES=$KAME_DYLIB"
            [ "$FLAT_NS" = 1 ] && inject="$inject DYLD_FORCE_FLAT_NAMESPACE=1"
            print_env="DYLD_PRINT_LIBRARIES=1" ;;
    *)      inject="LD_PRELOAD=$KAME_DYLIB"
            print_env="LD_DEBUG=libs" ;;
esac

# Sanity: confirm the library is actually loaded into the test process.
probe_bin="$bindir/transaction_payload_integrity_mixed_test"
if env $inject $print_env "$probe_bin" 1 1 64 0 >/dev/null 2>/tmp/v0full_dyld.$$; then :; fi
if grep -q "libkamepoolalloc\|kamepoolalloc" /tmp/v0full_dyld.$$ 2>/dev/null; then
    echo "[v0full] OK: injected allocator observed in loaded images"
else
    echo "[v0full] WARNING: injected allocator NOT observed loaded."
    echo "[v0full]          try FLAT_NS=1 (macOS) or check LD_PRELOAD path (Linux)."
fi
rm -f /tmp/v0full_dyld.$$

# ----------------------------------------------------------------------------
# 5) Sweep (modern allocator injected), back up any existing TSV.
# ----------------------------------------------------------------------------
out="$BENCH/${HOST_LABEL}_${VARIANT}_preload.tsv"
if [ -f "$out" ]; then
    cp "$out" "$out.bak-$(date +%Y%m%d_%H%M%S)"
    echo "[v0full] backed up existing $out"
fi
case "$VARIANT" in
    no_backoff) _note="variant=no_backoff FULL + DISABLE_BACKOFF + USE_KAME_ALLOCATOR=OFF + PRELOAD $(basename "$KAME_DYLIB")" ;;
    *)          _note="variant=${VARIANT} FULL + USE_KAME_ALLOCATOR=OFF + PRELOAD $(basename "$KAME_DYLIB")" ;;
esac
export KAME_REV_NOTE="$_note"
echo "[v0full] sweeping -> $out"
env $inject KAME_ROOT="$SRCTREE" BIN_DIR="$bindir" TESTS="$have_tests" \
    "$BENCH/run_kame_mixed_mac.sh" "${HOST_LABEL}_${VARIANT}_preload" | tee "$out"

echo ""
echo "[v0full] DONE. Wrote $out"
echo "[v0full] For a fair 3-way comparison run all variants this way:"
echo "[v0full]   VARIANT=v0         LABEL=$HOST_LABEL ./run_v0_full_preload.sh"
echo "[v0full]   VARIANT=no_backoff LABEL=$HOST_LABEL ./run_v0_full_preload.sh"
echo "[v0full]   VARIANT=full       LABEL=$HOST_LABEL ./run_v0_full_preload.sh"
echo "[v0full] worktree kept at $SRCTREE (git -C \"$KAME_ROOT\" worktree remove --force \"$SRCTREE\")"
