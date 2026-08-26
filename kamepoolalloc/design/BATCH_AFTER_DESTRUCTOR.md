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

## 2. The fix

`push` settles a slot directly when the batch is dead instead of buffering
into a destroyed object.  That is also the only way those slots get
returned at all: `~CrossDeallocBatch` will not run a second time, so
anything buffered afterwards was never going to be flushed.

The flag **must** be namespace-scope TLS (`tls_batch_dead`), not a member.
A member is written inside `~CrossDeallocBatch`, i.e. into an object whose
lifetime is ending, where `-flifetime-dse` may legally delete the store —
see §4.

This is the same class of problem the allocator already solves elsewhere
for the TLS page (`kame_thread_torn_down()` / `g_teardown_page`); the batch
was the gap.

## 3. Result

Interleaved, one binary per arm, pool rpath-pinned:

| measurement | baseline | fixed |
|---|---|---|
| double allocations, exclusivity detector on the STM reproducer | **11 / 14** | **0 / 14** |
| `alloc_tsd_exclusivity_test` (this commit) | **5 / 20** | **0 / 20** |
| crashes, original `tmin_dynnode` reproducer | 2 / 16 | 0 / 16 |

Read the third row as consistent, not as proof: at a 12 % baseline it has
almost no power on its own (p ≈ 0.48).  The first two rows are the
evidence.  `transaction_dynamic_node_test` itself passes, and so does the
rest of the suite bar four pre-existing failures — `starvation`,
`priv_strip`, `highest_older_wins`, `priv_expiry` — which fail identically
without the fix.  Those are an artifact of a scratch tree configured with
`-DKAME_STM_COMPACT_STATE=1` by hand: compact mode seals the privilege
bits, but the `kamestm_tests_need_prio` exclusion in `tests/CMakeLists.txt`
only fires on genuine no-DCAS detection.

`alloc_tsd_exclusivity_test` is the one to keep: it reproduces in **0.39
seconds** against the STM reproducer's two minutes, and it fails on the
unfixed allocator, which is what makes it a regression test rather than a
demonstration.

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
