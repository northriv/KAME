# The cross-dealloc batch is used after its own destructor

**The `transaction_dynamic_node_test` failure is an allocator exclusivity
failure, not the STM `Packet` use-after-free that
`kamestm/tests/DYNNODE_UAF_HANDOFF.md` §1–2 concluded.**  That handoff is
corrected in place; §3–§6 of it (reproducer, ablation, retracted claims,
experimental record) stand.

## 1. The defect

glibc's `start_thread` runs, in this order:

```
__call_tls_dtors()        // C++ thread_local destructors — ~CrossDeallocBatch
__nptl_deallocate_tsd()   // pthread_key destructors
```

KAME registers one of the latter: `Transactional::detail::pthread_destroy`
(`kamestm/threadlocal.cpp:81`), which **frees STM objects**.  Those frees
run after `~CrossDeallocBatch` has already destroyed the per-thread
cross-dealloc batch, and they were pushing into it anyway — resurrecting a
destroyed object, refilling it to the full 1024 entries and flushing it.
Caught in both teardown phases:

```
[batch used after its own destructor: count=1024, at_teardown=0,
 count left by dtor flush=0]
__call_tls_dtors -> deallocate_cold ->
    PoolAllocator<32,1,1>::deallocate_pooled_static -> flush
```

Independently, a **live** slot was caught being handed back to the bitmap
from the same teardown path:

```
*** A LIVE SLOT IS BEING RETURNED TO THE BITMAP ***
  0x7f98309d8800 is live as a Packet (32 bytes) allocated on tid 198 by
  tier 6: slow_allocate: allocate_chunk_path
  ...and tid 205 is handing it back to the allocator's bitmap now.
  __nptl_deallocate_tsd -> pthread_destroy -> deallocate_cold ->
      deallocate_pooled_static<16> -> CrossDeallocBatch::flush ->
      PoolAllocator<32,1,1>::batch_return_to_bitmap
```

Once a live slot's bitmap bit reads zero, the whole-word grab or the
bitmap scan re-issues it.  The tier trace shows exactly that — one block
claimed twice, by two different paths:

```
*** ALLOCATOR RETURNED A LIVE BLOCK ***
  0x7f1ac28c1460 handed out for a Packet (32 bytes) on tid 594,
  but our records still show it live as a Packet (32 bytes)
  from tid 594 -- no operator delete for it ever ran.
  served the FIRST time by  tier 1: new_redirected: TLS cell freelist pop
  served THIS time by       tier 3: new_redirected_cold: word-cache ctz serve
```

Two live objects then share one block, which is why the damage surfaces as
garbage inside a still-referenced `Packet` in `bundle()`.

Note the 32 B class takes the `push` (buffering) arm of
`deallocate_pooled`, not `push_direct` — `ALIGN <= 48` — so this is
precisely the path in the reports.

## 2. The fix (upstream, on `master`)

`b1d127a14` / `06d046d6e`, merged to `master` as of `796d9fe1e`.

A thread's frees are unsafe from the moment the FIRST of its two exit
objects is destroyed, so both arm one flag —
`kame_mark_thread_torn_down()`, read by the allocator's existing
`kame_thread_torn_down()` predicate:

- `~AllocThreadExitCleanup` — after it, `s_tls.my_chunk` / `&s_tls.dll_head`
  are torn down and the thread's chunks disowned.
- `~CrossDeallocBatch` — after it, `tls_cross_dealloc_batch` is a destroyed
  object that `push` / `push_direct` would resurrect.

Their order is not fixed: thread_locals are destroyed in reverse order of
construction, so which goes first depends on whether the thread's first
pool operation was an allocation or a cross-thread free.  Both occur, which
is why arming from only one of them (as `06d046d6e` and an earlier version
of this branch each did, from opposite ends) still leaves a window.
`kame_pool_set_realtime_thread` is guarded too — the one `flush` caller
outside the alloc/free paths.

The predicate lives in `KameTlsPage`, in what used to be alignment padding,
so the free path reads it off a pointer it already holds.  Two
`static_assert`s pin that layout — stated in **widths**, not LP64 byte
counts (§3, ILP32).

Superseded, and removed from this branch: a second namespace-scope flag
(`tls_batch_dead`) guarding `push` / `push_direct` / `flush` directly.  It
worked, but it is a parallel mechanism for one invariant, it armed from
only `~CrossDeallocBatch`, and guarding `flush` is the worse trade —
`flush` runs per-free at `cap = 1` and sits in the region a measurement had
already shown codegen-sensitive enough to delete the fault.

## 3. Result

> **RETRACTED (2026-08-28): the headline "11/14 → 0/14" was invalid.**
> Its fixed arm's binary **segfaulted at startup**, at the first
> allocation, and the A/B classified an arm by grepping its log — so 14
> startup crashes were scored as 14 clean runs.  Cause: `pf_alloc_tier()`
> null-tested a **weak `__thread` symbol** as `(&g_kame_alloc_tier) ? … : 0`.
> A weak TLS symbol cannot be null-tested that way: the TLS access sequence
> itself faults when the symbol is undefined, which it is whenever the
> linked pool was built without `KAME_ALLOC_TIER_TRACE`.  The two arms had
> also been built from different revisions of the instrumentation header —
> the baseline predated `pf_alloc_tier`, the fixed arm did not — so they
> were never comparable.  See §4.
>
> **Two lessons, both now enforced:** never classify an A/B arm by the
> absence of a marker (score the exit status too — a crash and a pass look
> identical to `grep -L`), and never rebuild one arm of an A/B without the
> other.

Valid measurements, arms verified to actually execute:

| measurement | baseline | fixed |
|---|---|---|
| STM reproducer + exclusivity detector, `origin/master` vs `06d046d6e` | **11 / 12** | **0 / 12** |
| `alloc_tsd_exclusivity_test`, `origin/master` vs `06d046d6e` | **10 / 24** | **0 / 24** |
| `alloc_tsd_exclusivity_test`, this branch's fix | **5 / 20** | **0 / 20** |
| crashes, original `tmin_dynnode` reproducer | 2 / 16 | 0 / 16 |

The first row is the re-run of the retracted A/B with the harness bug fixed
and both arms smoke-tested before use: `origin/master` gives 10 double
allocations plus one `SIGABRT` in 12, `06d046d6e` gives 12 clean runs.
That is the upstream fix (`fix/post-teardown-free-bypass`), which supersedes
this branch's `tls_batch_dead`: it reaches the same invariant through the
allocator's existing `kame_thread_torn_down()` predicate, with the flag in
the TLS page the free path already loads, rather than adding a second
mechanism.

### ILP32 — the width the fault originally reproduced on

`alloc_tsd_exclusivity_test`, real `-m32` builds, one pool per arm,
interleaved, 20 runs each:

| target | `origin/master` | `b1d127a14` |
|---|---|---|
| `-m32 -march=i486` (no CMPXCHG8B, `KAME_STM_COMPACT_STATE`) | **11 / 20** | **0 / 20** |
| `-m32 -march=i586` (has CMPXCHG8B, compact off) | **12 / 20** | **0 / 20** |

And the original reproducer itself, `tmin_dynnode` built `-m32 -march=i586`,
at `100 16 1250` (the parameters at which it reproduces — see §4 on the
null-baseline trap), 14 interleaved runs each:

| | `origin/master` | `b1d127a14` |
|---|---|---|
| `tmin_dynnode`, ILP32 | **14 / 14** | **0 / 14** |

Master's 14 split 8 `SIGABRT` / 6 `SIGSEGV`.  The `SIGABRT`s are the
uncaught `NodeNotFoundError` the handoff lists as a separate loose end
(§7), and one carries `tr_serial=0` — a zeroed serial, i.e. the transaction
was reading a `Packet` whose storage had been handed to someone else.  That
is the ILP32 face of the same defect, and it is exactly the signature the
original 2026-08-22 soak recorded.

This is the check `b1d127a14` itself asked for — its own run had the i486
audit phases *skip*, and a skip is not a pass.  Two things it settles: the
width-based `static_assert`s do build on ILP32 (all three audit phases
green), and the fix holds on the platform where the original fault ran at
10–25 %, not only on LP64.

It also confirms the ILP32 hash correction in the test was load-bearing:
with the old `uintptr_t` mix every address collapsed onto slot 0, so the
32-bit arms above would have reported 0/20 on BOTH sides — a clean sweep
that meant nothing.

The rows above are the evidence; all are independent of the broken
instrumentation, since `alloc_tsd_exclusivity_test` includes none of it.
The crash row is consistent but has almost no power on its own at a 12 %
baseline (p ≈ 0.48).

### Verified against shipped `master` (`796d9fe1e`)

Built from the merged tree, not from this branch:

| | result |
|---|---|
| `alloc_tsd_exclusivity_test`, LP64 | **0 / 20** |
| `alloc_tsd_exclusivity_test`, `-m32 -march=i486` | **0 / 20** |
| `tmin_dynnode`, `-m32 -march=i586`, `100 16 1250` | **0 / 10** |
| `tools/audit/check_no_dcas.sh` | all 3 phases ok |

(One near-miss worth recording: the first ILP32 attempt reported 10/10
failures because its compile had failed — a `cd` left the source path
relative — and the run used a STALE binary from the previous arm.  Always
confirm the binary you are about to measure was actually produced by the
build you just ran.)

`transaction_dynamic_node_test` passes, and so does the rest of the suite
bar four pre-existing failures — `starvation`, `priv_strip`,
`highest_older_wins`, `priv_expiry` — verified to fail identically with the
fix stashed.  Those are an artifact of a scratch tree configured with
`-DKAME_STM_COMPACT_STATE=1` by hand: compact mode seals the privilege
bits, but the `kamestm_tests_need_prio` exclusion in `tests/CMakeLists.txt`
only fires on genuine no-DCAS detection.

`alloc_tsd_exclusivity_test` is the one to keep: it reproduces in **0.39
seconds** against the STM reproducer's two minutes, it fails on the unfixed
allocator, and it uses only the `kame_pool_*` C API — no forensics header,
so it cannot be undermined the way the retracted row was.

## 4. Instrumentation traps

Both produced a confident wrong answer before being caught.  Do not
re-walk them.

- **`backtrace()` allocates on its first call.**  glibc lazily dlopens
  libgcc_s from `__libc_unwind_link_get`, and that dlopen allocates.  A
  backtrace on a free/flush *entry* path therefore manufactures the very
  re-entrancy it is looking for: the first report was "flush RE-ENTERED"
  whose inner stack was `backtrace -> __libc_unwind_link_get -> ld.so ->
  operator new -> flush`.  `kame_backtrace_warmed` forces the unwinder to
  load once, at load time.
- **A guard flag kept as a member of the object being destroyed.**
  `-flifetime-dse` legally drops stores into an object whose lifetime is
  ending, so an in-object `destroyed` flag read back false on a thread
  whose destructor had demonstrably already run.
- **An A/B whose BASELINE never fails.**  The ILP32 run of the original
  `tmin_dynnode` reproducer first came back 0/24 vs 0/24 and reads like a
  pass; it is a *null result*.  The parameters (60 rounds, 8 threads) were
  simply too weak to reproduce at all — at `100 16 1250` the same unfixed
  binary fails 4/4.  An arm can only be cleared by a comparison whose
  control actually fired, so always confirm the baseline reproduces at the
  chosen parameters BEFORE reading anything into the fixed arm.
- **Null-testing a weak `__thread` symbol.**  `(&weak_tls) ? weak_tls : 0`
  does not degrade gracefully — the TLS access sequence faults when the
  symbol is undefined.  Linking the exclusivity detector against a pool
  built WITHOUT `KAME_ALLOC_TIER_TRACE` therefore segfaulted at the first
  allocation, silently voiding a whole A/B arm (§3).  The tier is a
  compile-time opt-in now (`KAME_PF_TIER`), so a mismatch is a link error
  naming the symbol rather than a runtime crash.

## 5. Refuted, with the numbers

- **`count = 0` at the end of `flush` is dead-store-eliminated.**  No —
  measured `count left by dtor flush = 0`, not 1024.  (Mine.)
- **The word cache is the mechanism.**  `KAME_FS_WORDCACHE=0` still fails
  **6/6**.  It is a consumer of the corrupt bitmap state: with the grab
  gone, the same zero bit is found by the bitmap scan instead.
- **§29's freelist pre-fill is the mechanism.**  ON **5/10** vs OFF
  **4/10**.  An earlier 2/3-vs-0/3 reading of it was noise.
- **The one confirmed UB in `allocator.cpp` (the dedicated-chunk type-pun)
  is involved.**  It is on the large-block path; the double-allocated
  block is 32 B, which is the bucket path.
- **`new_redirected_cold`'s word-cache address arithmetic
  (`(b * bucket) << 4`) is wrong for the LUT range.**  It is guarded on
  `m_fs_flag`, and FS=true is exactly the range where `bucket * 16 ==
  ALIGN`.
- **`scan_dll_freelist` leaves a stale list head.**  It pops correctly and
  re-pins the chunk.

## 6. `-fipa-cp-clone`

The flag governs the double allocation itself, not merely whether the
corruption is lethal:

| pool build | double allocations |
|---|---|
| `-O3` | 5 / 8 |
| `-O3 -fno-ipa-cp-clone` | 0 / 8 |

mirroring its effect on the crash (43/100 vs 0/167).  What it is NOT: this
does not establish a miscompile.  An exposed latent race fits these
numbers exactly as well as wrong code, and the fault above is a genuine
lifetime bug that exists in the source regardless of the flag.  The
handoff's §5 retraction of the miscompile claim stands.

`LD_LIBRARY_PATH` cannot be used to A/B the pool here: the test binaries
carry RPATH, which is searched first and silently defeats it.  Link one
binary per arm.

## 7. Debug instrumentation

- `KAME_ALLOC_TIER_TRACE` (this library) — publishes the serving path in
  one TLS word, plus the post-destructor warning, the duplicate-entry scan
  and the `kame_pool_debug_slot_returned` hook.  Compiled out otherwise;
  the shipped build references no `backtrace`.
- `KAME_STM_ALLOC_EXCLUSIVITY` (`kamestm/packet_forensics.h`) — the
  layout-preserving detector.  Defines the hook and answers "is this slot
  live?".  Requires the client linked `-rdynamic`.
- `KAME_STM_PACKET_FORENSICS` (same header) — per-object headers and
  quarantine.  **Adds 424 bytes per object**, which moves a 40-byte
  `Packet` into a different size class; a clean result from it is not
  evidence about an allocator-reuse fault.
