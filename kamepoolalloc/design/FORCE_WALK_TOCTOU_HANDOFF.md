# Handoff: the force-walk hint is written through a pointer into a thread that may have exited

Written 2026-10-08, x86-64 Linux, g++ 13.3, 4 cores.  Found while testing
`b23a57ec1` (`claude/asp-serial-orphan-chain`), but **the defect is on
`master` and predates that branch** — see §3.

**Status:** fixed on `claude/force-walk-hint-table` (`ea8d70766`, the global
hint table from §4).  Validated on Linux, LP64 and ILP32 (§8).  **That commit
does not compile on any 32-bit target** without the one-line patch in §8.

---

## 1. Summary

Three cross-thread free paths in `allocator.cpp` read a pointer into the
**owner thread's TLS**, call `batch_return_to_bitmap`, and only then write
through the pointer.  Nothing keeps the owner thread alive between the read
and the write.  If the owner exits in that window and glibc unmaps its stack,
the write faults:

```
mov 0xa8(%rdi),%r12     ; cached = chunk->m_owner_dll_force_walk_ptr.load(acquire)
call *0x18(%rax)        ; chunk->batch_return_to_bitmap(...)
movb $0x1,(%r12)        ; cached->store(true)            <-- SIGSEGV here
```

Measured: **5 / 1000** runs of `alloc_tsd_exclusivity_test` crash on
`master` under 4-way concurrency, every one at that instruction.

## 2. The defect

`m_owner_dll_force_walk_ptr` is set to `&s_tls.dll_force_walk_from_head`
(`allocator.cpp:2998`, also `:1298`, `:2300`), and `s_tls` is
`static ALLOC_TLS ThreadLocalState s_tls` (`allocator_prv.h:2031`).  The
pointer therefore targets the owner thread's static TLS, which on glibc lives
in that thread's stack mapping.

The three sites, line numbers identical on `master` `872d89030` and on
`b23a57ec1`:

| site | load | store through it |
|---|---|---|
| `CrossDeallocBatch::push_direct` (direct arm) | `:803` | `:815` |
| `CrossDeallocBatch::flush` | `:874` | `:878` |
| `deallocate_pooled` direct return | `:1985` | `:2010` |

Each caches the pointer *before* `batch_return_to_bitmap`, deliberately,
because that call may release the chunk.  That fixes the chunk side.  It does
not address the thread side.

At owner exit (`release_dll_chunks_for_thread`, `:3311`) the field is nulled
with a release store, and the comment above it (`:3298–3310`) argues:

> a freer that observes nullptr ... skips the deref.  A freer that observes
> the old non-null pointer must have loaded BEFORE our release, in which case
> our TLS is still live.

That holds **at the moment of the load**.  The freer writes later, after
`batch_return_to_bitmap`.  Nothing extends the owner's lifetime across that
gap.  The owner can finish `release_dll_chunks_for_thread`, run the rest of
its teardown, be joined, and have glibc release its stack while the freer is
still inside `batch_return_to_bitmap`.  This is a time-of-check / time-of-use
gap: a safe load does not imply a safe store.

## 3. Evidence

Crashes captured with an `LD_PRELOAD` SIGSEGV backtrace handler (§6):

- **13 / 13** stacks are identical:
  `std::thread` routine → test body → `PoolAllocatorBase::deallocate_cold`
  → `PoolAllocator<32,1,1>::deallocate_pooled_static+0xd5` →
  `CrossDeallocBatch::flush(bool) [clone .constprop.0]` at the store above
  (`+0x27794` on master, `+0x27724` on the branch — the same instruction).
- The crashing thread is in its **normal body**, not in teardown — no
  `__call_tls_dtors` / `__nptl_deallocate_tsd` frame.  The *other* thread,
  the chunk's owner, is the one exiting.

Rates, `alloc_tsd_exclusivity_test`, same build config, scored by exit status:

| condition | `master` 872d89030 | `b23a57ec1` |
|---|---|---|
| 4 concurrent instances, interleaved, 1000 runs each | **5 / 1000** SIGSEGV | **8 / 1000** SIGSEGV |
| standalone, 30 runs | 0 / 30 | 0 / 30 |
| 4-way under gdb, 60 runs | — | 0 / 60 |

8 vs 5 is not a significant difference (Fisher p ≈ 0.58).  `b23a57ec1` does
not touch these lines and neither causes nor fixes this.  It surfaced as an
`alloc_tsd_exclusivity_test` SEGFAULT in one `ctest -j2` run on that branch.
**Do not attribute that failure to the serial change.**

### Established, and not

- **Established:** the fault is the post-call store through the cached
  force-walk pointer in `flush`.  It reproduces on `master`.  It is
  independent of `b23a57ec1`.
- **Measured since (§3a):** the window itself.  Stores through a pointer whose
  owner had already finished its allocator teardown happen in about half of
  all runs, and never in a control where no owner exits during the frees.
- **Still argued, not measured:** that each individual SIGSEGV is one of those
  stores, rather than a read of the field from a dead chunk.  The argument:
  the entries being flushed are freed-but-not-yet-returned slots, so their
  bits are still set and the chunk cannot be released before
  `batch_return_to_bitmap` runs.  The chunk is therefore valid at the load,
  and `r12` is the real field value.  To close it, capture `si_addr` / `r12`
  at the fault (SA_SIGINFO + ucontext) and match them against a hit logged
  at the same address.
- **Not observed:** crashes at the `push_direct` and `deallocate_pooled`
  sites.  The 32 B class takes `push` → `flush` (`ALIGN <= 48`), so this test
  mostly exercises `flush`.  The other two sites have the same shape and
  should be treated as affected.

### 3a. Measured: the window is entered (`KAME_ALLOC_FORCE_WALK_TRACE`)

A debug-only counter in `allocator.cpp` (`namespace fw_trace`).  Without the
macro it compiles to the shipped code: the linked `.text` of
`libkamepoolalloc.so` is byte-identical with and without the change
(299,244 bytes), and every object section has the same name and size.

**What a HIT is.**  A store through the cached force-walk pointer that loaded
the pointer before the owner nulled it (it read non-null) and runs after that
owner FINISHED `release_dll_chunks_for_thread` for that size class.  From
that point nothing keeps the target mapped, so this is exactly the store the
code's safety argument does not cover.  A hit is not a fault.  Whether it
faults depends on what the dead thread's TLS has become (§3, macOS).

**How.**  The owner, after its walk, appends (its TLS address, a fresh epoch)
to a 4096-entry ring.  A freer reads the epoch before loading the pointer.
Just before storing, it scans only the ring entries newer than that epoch —
usually none.  The epoch bound is what stops a reused address from matching.
Every miss is an under-count, never a false hit: an entry not yet written, or
more than 4096 exits in one window (`overruns`).  All atomics are pointer
width, so the trace build runs on i486.  A premise self-check counts any
chunk whose pointer, at the null-out, was not the address the walk then
records (`premise mismatches`).

Use:

```bash
cmake ... -DCMAKE_CXX_FLAGS="-DKAME_ALLOC_FORCE_WALK_TRACE"
./alloc_tsd_exclusivity_test            # prints at exit:
# [fw-trace] 2 hit(s) in ~11314 cross-thread force-walk stores; 388 owner
#            exits recorded; overruns 0; premise mismatches 0
# [fw-trace]   flush   hits 2 / stores ~11314   (and the other two sites)
KAME_FW_TRACE_SKIP=1 ./...              # skip the store on a hit: a run that
                                        # would fault survives and reports
```

The store count is approximate: it is folded per thread at owner exit.  The
hit count is exact apart from the misses above.

**Results**, `alloc_tsd_exclusivity_test`, x86-64 Linux, g++ 13.3:

| condition | LP64 | `-m32 -march=i486` |
|---|---|---|
| one run, no other load | 2 hits / ~11,314 stores | 4 / ~8,093 |
| 4 concurrent, 100 runs, `SKIP=1` | **52 / 100 runs** hit; 81 / ~253,122 | **44 / 100**; 53 / ~160,839 |
| **negative control**: all cross-thread frees done, then a barrier, then any exit (24 rounds × 8 threads) | **0** / ~4,032 | **0** / ~4,032 |
| premise mismatches, every run above | 0 | 0 |

Two readings:

- The window is entered routinely: in about half of all runs on both widths,
  and in an unloaded single run.  The negative control has owner exits and
  stores in every round, just never in that order, and reads 0.  So the
  counter is not firing on owner exits alone.
- Hits outnumber faults about 100 to 1.  The unmodified build faults in about
  0.5 % of runs (§3), while about 0.8 hits occur per run.  Most hits land in
  memory that is still mapped — consistent with glibc's stack cache.  That
  ratio is why a crash-rate A/B needs thousands of runs, and the hit counter
  needs about a hundred.

`SKIP=1` runs had 0 non-zero exits, but at a 0.5 % per-run fault rate 100
runs say nothing about whether skipping prevents the fault.  This is not a
fix claim.

**Earlier diagnosis this supersedes.**  The comment above
`CrossDeallocBatch::flush` records a SIGSEGV at this same store, attributed
it to musl's TSD-destructor ordering ("observed only on Alpine/musl; glibc
CI is clean"), and dropped the poke only when the *freer* is at teardown.
The measurement here is on glibc, with the freer in its normal body; it is
the owner that is exiting.  That fix does not cover this case.

**On macOS** the counter answers the question §3's macOS section leaves
open: if `hits > 0` there with no crash, the window is hit and the writes
are silent.

### Worse than a crash, by implication (not observed)

The observed failure is the benign one: the stack was unmapped, so the write
faulted.  glibc also caches and reuses thread stacks:

- If a new thread with the **same stack size** reuses the mapping, the
  address lands on the same TLS variable in the new thread.  The result is a
  spurious "walk from head" hint, which is harmless.
- If the range is reused by **anything else** (a thread with a different
  stack size, or a later `mmap`, including the pool's own regions), the
  result is a stray one-byte write of `0x01` into live memory, with no fault.

### macOS: not reproduced, and why that does not clear it

The user reported (2026-10-08) that the crash **does not reproduce on
macOS**.  The run conditions (count, concurrency, build) are not recorded
here yet.

This is the expected outcome even if the defect is present, because the
code is the same on both platforms and only the consequence differs.
`ALLOC_TLS` is `__thread` on GCC and clang on both (`allocator_prv.h`), so
`s_tls` is a `__thread` variable in a shared library:

- **Linux (glibc):** a startup-linked library's `__thread` data is static
  TLS, which glibc places inside each thread's **stack mapping**.  A joined
  thread's stack goes into glibc's stack cache (~40 MiB by default), and the
  overflow is `munmap`ed.  At 8 MiB stacks, roughly 3 of this test's 8 threads
  per generation get unmapped.  Writing there faults — the SIGSEGV in §3.
- **macOS (Mach-O):** a dylib's `__thread` data is a TLV reached through
  `_tlv_get_addr`.  To my understanding (**not verified on a Mac here**), dyld
  allocates each thread's TLV block with `malloc` and frees it with `free`
  from a pthread-key destructor at thread exit.  A freed small heap block stays
  mapped, so the same window yields a **silent** one-byte write of `0x01`
  into freed heap.  If the pool interposes `malloc` in that process, the block
  came from the pool, and the write can land in a slot already reissued to a
  live object.

So "no crash on macOS" cannot distinguish "the window is never hit there"
from "it is hit and corrupts silently".  To separate them on any platform:

1. **Count the window directly** — now implemented as
   `KAME_ALLOC_FORCE_WALK_TRACE` (§3a).  It counts hits whether or not the
   write faults, and whether or not the address has been reused since.  On
   macOS, `hits > 0` with no crash means "hit, and corrupting silently".
2. **ASan on macOS** (cheaper, less certain).  If dyld's TLV block comes from
   an ASan-intercepted `malloc`, the stray store reports as
   heap-use-after-free.  The pool's own `malloc` / `operator new`
   interposition may conflict with ASan.

## 4. Fix directions (none tested)

The hint is only a hint: a lost or spurious `true` costs at most one extra DLL
walk.  So the target only needs to be **memory that is always mapped and
type-correct**.  It does not have to be the owner's own TLS.

1. **Global hint table indexed by owner id** (recommended to try first).
   `m_owner_dll_force_walk_ptr` becomes an index, or points into a
   process-lifetime `std::atomic<bool>[]`.  The store always hits valid memory.
   If an owner id is reused, the worst case is a spurious hint to the next
   thread with that id, which is benign.  Check how `kame_owner_id()` allocates
   and recycles ids, and size the table from that.  `atomic<bool>` keeps the
   no-DCAS audit clean.
2. **Heap cell refcounted by the chunks that point to it.**  The owner drops
   its reference at exit, and the last chunk frees the cell.  This is correct,
   but it adds RMW traffic to chunk lifecycle paths.
3. **Owner exit waits for in-flight freers** (hazard pointer or epoch).
   This is correct, but complex, and it adds latency to every thread exit.

These are not fixes:

- Moving the store before `batch_return_to_bitmap` shrinks the window but
  does not close it.
- Reloading the pointer after the call is still a load-then-store.
- Checking the owner's `BIT_OWNED` / `m_owner_id` before storing has the same
  gap.

Caution for any fix: `flush` and its `.constprop` clone are where earlier
work found the fault rate codegen-sensitive enough that instrumentation
deleted it (see `BATCH_AFTER_DESTRUCTOR.md`).  Validate by interleaved A/B,
not by a clean run.

## 5. Validating a fix

At a ~0.5 % per-run baseline, 1000 runs per arm gives about 5 expected
events.  A result of 0 vs 5 is **not significant** (two-sided Fisher
p ≈ 0.06).  Run **≥ 2000 per arm**, where 0 vs 10 gives p ≈ 0.002.  Run the
arms interleaved and score them by exit status (a crash and a pass look the
same to a marker grep).  Then:

- confirm the baseline arm actually fails in the same session (null-baseline
  trap);
- run ctest on LP64 and `-m32` (i586 and i486);
- run `tools/audit/check_no_dcas.sh`.

**The hit counter (§3a) is the sharper metric, but only for some fixes.**
Baseline is about 0.8 hits per run, against about 0.005 faults, so ~100
runs per arm decide what a crash A/B needs thousands for.

- A fix that **keeps** the owner-TLS target and closes the window (for
  example, owner exit waiting for in-flight freers): hits should go to 0.
  Use the counter directly.
- A fix that **changes** the target (the global owner-id table): the store
  can no longer reach dead TLS, so the counter measures nothing.  Validate by
  the crash A/B.  Keep the counter's sites until then and confirm they record
  0 stores.

## 6. Reproduction

Build the test tree (any recent g++):

```bash
cmake -S tests -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG" -DUSE_KAME_ALLOCATOR=ON
cmake --build build -j4 --target alloc_tsd_exclusivity_test kamepoolalloc
```

Backtrace-on-crash preload.  It warms `backtrace()` in its constructor,
because the first call dlopens libgcc_s and allocates:

```c
// segvbt.c — gcc -O2 -fPIC -shared segvbt.c -o segvbt.so
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
static void h(int sig) {
    void *bt[48]; int n = backtrace(bt, 48);
    const char m[] = "\n=== SEGVBT signal ===\n"; (void)!write(2, m, sizeof m - 1);
    backtrace_symbols_fd(bt, n, 2);
    signal(sig, SIG_DFL); raise(sig);
}
__attribute__((constructor)) static void init(void) {
    void *w[2]; backtrace(w, 2);
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = h;
    sigaction(SIGSEGV, &sa, 0); sigaction(SIGBUS, &sa, 0); sigaction(SIGABRT, &sa, 0);
}
```

Four concurrent instances per round, scored by exit status:

```bash
T=build/kamepoolalloc-tests/alloc_tsd_exclusivity_test
for r in $(seq 1 250); do
  for k in 1 2 3 4; do
    ( LD_PRELOAD=./segvbt.so timeout 60 $T > run_${r}_$k.log 2>&1; echo "$?" >> exits ) &
  done; wait
done
sort exits | uniq -c        # 139 = SIGSEGV; expect ~5 in 1000 on master
grep -l SEGVBT run_*.log    # the crashing runs, with stacks
```

Resolve the faulting frame with
`addr2line -f -C -e build/kamepoolalloc-tests/libkamepoolalloc.so <offset>`.

## 7. Context: the rest of the `b23a57ec1` test

Recorded here so it is not re-run:

| check | result |
|---|---|
| ctest, LP64, g++ 13.3 | 41 / 42 — the one failure is this defect |
| ctest, `-m32 -march=i586`, release | 42 / 42 |
| ctest, `-m32 -march=i486`, release | 35 / 35 (7 excluded as no-DCAS, by design) |
| ctest, `-m32` i586 / i486, **asserts on** | 42 / 42, 35 / 35 |
| `orphan_chain_push`'s `0x3F040` low-bits assert on `-m32`, asserts on | held on 771 + 6363 pushes (two tests, gdb hit count) |
| `tools/audit/check_no_dcas.sh` | 3 / 3 ok |
| hang A/B, `transaction_payload_integrity_mixed_test 1 64 256 0`, 15 s cut-off, interleaved | **0 / 200 vs 0 / 200**.  The master arm never hung, so this A/B is **null** on this box (x86-64, 4 cores).  It neither confirms nor refutes the branch's M5 Ultra result (5 / 300 → 0 / 300).  Confirming it needs a machine where master's arm actually hangs. |
| TLA+ `OrphanChain_aba` 2-thread cfgs | not run (stopped) |

The branch's own "Not yet built with -m32" is answered by the `-m32` rows.

## 8. The fix, tested: `ea8d70766` (`claude/force-walk-hint-table`)

Tested 2026-10-08, x86-64 Linux, g++ 13.3, 4 cores, against its parent
`master` `872d89030` with the same build config.  The commit replaces the TLS
pointer with a static table `g_force_walk` (owner id mod 4096, one bit per
DLL-owning template).  All three free sites now cache `m_owner_id` and
`m_force_walk_bit` before `batch_return_to_bitmap`, and nothing points into
TLS any more.  The §3a counter therefore no longer applies; the fix is
validated by crash A/B, as §5 says.

| check | result |
|---|---|
| crash A/B, `alloc_tsd_exclusivity_test`, 4 concurrent, interleaved, by exit status | **master 17 / 2000 SIGSEGV, fix 0 / 2000** (two-sided Fisher p = 1.5 × 10⁻⁵).  All 17 at the same store: `flush` via `deallocate_pooled_static<32>`. |
| ctest, LP64 | 41 / 41 |
| hint still functional: `bench_xthread_pool -w 2 -t 3`, mmap regions added, 3 reps × 64 / 256 / 1024 B | fix +2 / +1 / +1, master +2 / +1 / +1–2.  A dead hint shows +15–17 (the commit's own measurement), so this rules that out.  Throughput not compared — a 4-core shared box is not a benchmark host. |
| **32-bit build** | **fails** on `-m32 -march=i486`, `-m32 -march=i586` and plain `-m32`: `force_walk_bit`'s `static_assert` |
| with the patch below: `check_no_dcas.sh` | 3 / 3 ok — phases 2–3 are the ones the commit had to skip on macOS |
| with the patch: ctest `-m32 -march=i586` / `-m32 -march=i486` | 41 / 41, 34 / 34 |
| with the patch: LP64 linked `.text` | byte-identical to `ea8d70766` (301,916 bytes), so every LP64 row above holds for the patched build too |

**The 32-bit break.**  `force_walk_bit` maps the variable-size templates by
an explicit ALIGN table, {32, 64, 256, 1024, 4096} → bits 24..28.  That set is
the LP64 one.  The variable-size ALIGNs follow `ALLOC_ALIGN2`, which is 256 on
LP64 and **128 on ILP32**, so ILP32 has a sixth template, `<128, true, false>`.
It falls through to 32 and fails the assert.  Enumerated by instantiation:
LP64 {32, 64, 256, 1024, 4096}; ILP32 {32, 64, 128, 256, 1024, 4096}.  The
fix gives 128 the free bit 29:

```diff
-	    : ALIGN == 1024u ? 27u : ALIGN == 4096u ? 28u : 32u;
+	    : ALIGN == 1024u ? 27u : ALIGN == 4096u ? 28u
+	    : ALIGN == 128u ? 29u : 32u;
```

(plus the comment: "24..29 by ALIGN", and why ILP32 has six).  Same failure
class as `06d046d6e`'s `KameTlsPage` asserts: written where the i486 audit
phases skip.  A width-independent alternative is `24 + log2(ALIGN / 32)`, which
covers 32..4096 in bits 24..31.  It would make the table impossible to fall
behind `ALLOC_ALIGN*`, at the cost of renumbering the LP64 bits.
