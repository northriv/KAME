# Handoff: the force-walk hint is written through a pointer into a thread that may have exited

Written 2026-10-08, x86-64 Linux, g++ 13.3, 4 cores.  Found while testing
`b23a57ec1` (`claude/asp-serial-orphan-chain`), but **the defect is on
`master` and predates that branch** — see §3.  Not fixed.

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
- **Argued, not measured:** that the pointer is stale because the *owner
  thread* died, rather than because the field was read from a dead chunk.  The
  argument: the entries being flushed are freed-but-not-yet-returned slots, so
  their bits are still set and the chunk cannot be released before
  `batch_return_to_bitmap` runs.  The chunk is therefore valid at the load,
  and `r12` is the real field value.  **First next step:** capture `si_addr`
  and `r12` at the fault (SA_SIGINFO + ucontext) and check them against the
  stack ranges of the threads that had exited.  Alternatively, A/B a fix.
- **Not observed:** crashes at the `push_direct` and `deallocate_pooled`
  sites.  The 32 B class takes `push` → `flush` (`ALIGN <= 48`), so this test
  mostly exercises `flush`.  The other two sites have the same shape and
  should be treated as affected.

### Worse than a crash, by implication (not observed)

The observed failure is the benign one: the stack was unmapped, so the write
faulted.  glibc also caches and reuses thread stacks:

- If a new thread with the **same stack size** reuses the mapping, the
  address lands on the same TLS variable in the new thread.  The result is a
  spurious "walk from head" hint, which is harmless.
- If the range is reused by **anything else** (a thread with a different
  stack size, or a later `mmap`, including the pool's own regions), the
  result is a stray one-byte write of `0x01` into live memory, with no fault.

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
