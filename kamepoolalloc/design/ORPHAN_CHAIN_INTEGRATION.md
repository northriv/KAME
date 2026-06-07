# Orphan-chain integration plan (atomic_shared_ptr refcount chunk-release)

Staged plan to replace the §36 orphan Treiber stack + `BIT_OWNED`
arbitration with the **TLA+-verified atomic_shared_ptr-refcounted intrusive
singly-linked orphan chain** (`tests/tlaplus/OrphanChain_atomicshared.tla`).

Branch: continuation of `orphan-atomicshared` (which already provides the
`force_intrusive_ref` / `atomic_intrusive_dispose` primitives in
`atomic_smart_ptr.h` and the `atomic_intrusive_dll_test.cpp` PoC).

## Refcount model — SEPARATE counts (Path B, decided)

MASK_CNT (live-slot count, in `m_flags_packed`) and the atomic_shared_ptr
intrusive `refcnt` are **two independent counters** — NOT unified.  Unifying
them (Stage 2's manual self-ref, reverted in 7db53986) raced
atomic_smart_ptr's split (local-tag) counting: a manual `fetch_sub` to 0
while a `load_shared` reader's ref sits in a cell's local tag → premature
dispose / UAF.

So the intrusive `refcnt` is managed **EXCLUSIVELY by atomic_smart_ptr — no
manual ops**:

- **owner-ref** = a `local_shared_ptr<PoolAllocator>` the owner's per-thread
  DLL holds for each owned chunk (your original "相互参照DLLは local_shared_ptr
  ベース").  Keeps owned chunks alive (empty or not).  Dropped at owner-exit.
- **chain-ref** = head / a predecessor's `m_orphan_next` (`atomic_shared_ptr`).
  Keeps an orphan alive while on the chain.
- **pins** = transient `load_shared` handles (local-tag).

`refcnt -> 0` fires `atomic_intrusive_dispose`.  "Don't release a non-empty
chunk" is enforced **structurally**, not by a self-ref: an owned non-empty
chunk has an owner-ref; a non-empty orphan keeps its chain-ref (the sweeper
removes only DEAD nodes and relink preserves successors, so a non-empty node
never loses its incoming link).  Therefore `refcnt -> 0  ⟹  MASK_CNT == 0`;
the disposer **asserts** `MASK_CNT == 0` as a safety net.  No `self-reset`
either — an unlinked dead node's dtor drops its `m_orphan_next`, releasing
the forward ref.

**Intrusive-refcnt location = accessor hook (not a `refcnt` member, not a
template offset).**  Add to atomic_smart_ptr (mirroring `has_intrusive_dispose`):
if `T` provides `static atomic<Refcnt>& T::atomic_intrusive_refcnt(T*)`, the
smart_ptr uses it; else falls back to `p->refcnt`.  Rationale: the chunk type
is self-referential/incomplete at the trait point (a member-pointer template
param can't be taken there; an integer-offset param is a fragile magic
constant with no compiler check), whereas an accessor body is instantiated
late (T complete) like the dispose hook, is compiler-checked, zero-cost
inlined, and lets `refcnt` live in the chunk HEADER (free bytes [40..63],
chunk-relative — uniform across ALIGN, reachable from `chunk_base` in
cross-thread paths) cleanly separate from MASK_CNT.  Small, consistent
addition for the atomic_smart_ptr owner (other session); does not touch their
GenMC model.

Lifecycle (separate counts; refcnt = owner-ref + chain-ref + pins):

| event | refcnt | MASK_CNT | note |
|---|---|---|---|
| create (empty, owned) | owner=1 → **1** | 0 | |
| first alloc | **1** (unchanged) | 0→1 | refcnt untouched by alloc |
| owner frees all (still owned) | **1** (unchanged) | 1→0 | owner-ref keeps it; reusable |
| owner-exit, non-empty | −owner, +chain → **1** | >0 | push transfers owner-ref→chain-ref |
| owner-exit, empty | −owner → **0** ⇒ release | 0 | assert MASK_CNT==0 ✓ |
| orphan drains to empty (cross-free) | **1** (unchanged) | →0 | stays on chain until swept |
| orphan swept off chain (empty) | −chain → **0** ⇒ release | 0 | assert MASK_CNT==0 ✓ |
| orphan adopted (non-empty) | −chain, +owner → **1** | >0 | back to owned |

`refcnt` moves ONLY on owner-ref (DLL local_shared_ptr) / chain-ref
(head/m_orphan_next) / pin transitions — never on alloc/free.  The structural
invariant (sweeper removes only DEAD nodes; relink preserves successors; no
self-reset) guarantees a non-empty chunk never loses its last ref, so
`refcnt→0 ⟹ MASK_CNT==0`.

**Re-verify in TLA+:** the committed `OrphanChain_atomicshared.tla` modelled
the self-ref + self-reset variant.  Path B is a *different* mechanism
(separate counts, chain-ref+owner-ref only, dead-only removal, no
self-reset) and needs its own small model proving: a non-empty node never
loses its last incoming ref, `released ⟹ MASK_CNT==0`, no double-release, no
leak.

## Link fields — SEPARATE (decided)

The per-thread DLL link and the orphan-chain link are **separate fields**,
NOT unified into one `next` (rejected):

- **chain-next** = `m_orphan_next` (`atomic_shared_ptr`), touched ONLY in the
  ORPHAN role.  Concurrent sweepers/`load_shared` readers walk it.
- **DLL-next** = its own field; **DLL-prev** stays raw (back-link for O(1)
  middle removal; chain is singly-linked).
- **owner-ref** lives on the DLL **forward** link as a `local_shared_ptr`
  (predecessor / `dll_head` holds the successor's owner-ref — a chain, no
  self-cycle).  Single-thread `local_shared_ptr` (local-tag, not atomic):
  the owner's DLL walk pays a refcount op per step, acceptable on the
  cold-ish chunk-claim walk / AllocPinCleanup.

Why not unify (even though a chunk is OWNED xor ORPHAN, so §36 reused
`m_dll_next`): Path B's chain has **concurrent middle-traversal** (sweepers),
unlike §36's head-only stack.  A unified field would (1) force `atomic`
(not just local) on the single-thread DLL → traversal atomic tax, and (2)
create a **transition race**: at owner-exit / adopt the field's role flips
single-thread↔shared, and a stale chain-walker could read it as chain-next
while the owner reinterprets it as DLL-next → corruption.  Separate fields
make a stale chain-walker read a valid (merely stale) chain-next = safe-side.
Cost of separating = 8 B/chunk, negligible (chunk ≥ 256 KiB).

## Design recap (what the model proved)

- **Self-ref on `m_filled`** (= `MASK_CNT`): a chunk holds ONE refcount to
  itself while `MASK_CNT > 0`, bumped at the `0->1` edge, dropped at the
  `1->0` edge (`atomicDecAndTest`-returns-true).  Decouples chunk lifetime
  from chain membership.  **One bump per non-empty/empty edge, NOT per slot**
  — the hot path (`atomicInc`/`atomicDecAndTest`) gains only a boundary
  branch, not an extra atomic.
- **Release = refcount -> 0** (the atomic_shared_ptr deleter), gated:
  `released  <=>  StructRefs = 0  <=>  no chain ref AND no pin AND MASK_CNT==0`.
- **Multiple sweepers, safe-side only**; **revival vs release** mutually
  exclusive via try_promote (`StructRefs>0`) vs release (`StructRefs=0`).
- **All chain ops are single-node / in-place** — push 1 at head, **pop 1 at
  head** (reuse adopts ONE chunk, never the whole list), `scrub` relinks
  in place.  No detach-and-reprocess / whole-list steal.  Reclaim
  completeness of mid-chain empties comes from `scrub`'s full walk, not from
  pop.  (The model's `Revive` is unconstrained — adopt any node, keep or
  unlink — so head-only pop is a verified-clean subset, a fortiori.)
- TLC: `selfref` CLEAN (269), `noselfref` VIOLATION depth 3 (self-ref is
  load-bearing), `live` no-leak, `push` (head insert) CLEAN (541).

## Lifetime mapping: TLA+ action -> code site (from the integration map)

| TLA+ action | Current code site | Becomes |
|---|---|---|
| self-ref bump (`filled 0->1`) | `atomicInc(&m_flags_packed)` allocate_pooled FS=false (alloc.cpp:2070); FS=true alloc inc | bump chunk refcnt iff `MASK_CNT` was 0 |
| `Free` / self-ref drop (`1->0`) | `atomicDecAndTest(&c->m_flags_packed)` deallocate_pooled (alloc.cpp:1606,1639) | on true: empty-eager madvise (RSS) + drop self-ref |
| `Push` (owner-exit non-empty) | `orphan_push(c)` (alloc.cpp:3048) in `release_dll_chunks_for_thread` | atomic_shared_ptr Treiber push onto `g_orphan_head` |
| `Revive` (reuse adopt) | `orphan_pop()` + `BIT_OWNED` claim loop (alloc.cpp:2700-2756) | `try_promote` adopt from chain (acquire-if-nonzero) |
| `SweepRelink`/`SelfResetNext`/`HeadAdvance` | (none — stack has no compaction) | new `orphan_scrub()` GC pass (CAS relink past empties) |
| `Release` (refcount 0) | `deallocate_chunk` (alloc.cpp:3054) via `bucket_release_chunk` (3733) | the intrusive `atomic_intrusive_dispose` -> `deallocate_chunk` |
| owner-exit empty release | `release_dll_chunks_for_thread` empty branch (3010-3031) | drop self-ref -> refcount path |

## Chunk-header budget

64-byte header: `[0..39]` used (size_info / palloc / fn / sizeof_fn /
dedicated_size), **`[40..63]` = 24 free bytes**.  The embedded
`PoolAllocator` (chunk_base+64) already holds `m_flags_packed`,
`m_owner_id`, `m_dll_prev/next`.  The intrusive node needs: `atomic<Refcnt>
refcnt` (8B) + `atomic_shared_ptr<...> m_orphan_next` (8B).  Plan: place
`refcnt` + `m_orphan_next` in the embedded object (NOT the 24 header bytes —
those stay for `lookup_chunk`'s plain reads).  `m_orphan_next` REPLACES the
stack's reuse of `m_dll_next` (the two must not alias; orphan chain link is
distinct from per-thread DLL link).

## Stages (each gated by: build + `alloc_stress` residual=0 + `bench_loop` 64B non-regression)

All work behind `#if KAME_ORPHAN_CHAIN` (default **0**), so every stage
leaves the shipping path (orphan stack) intact and the build green until the
final flip.

- **Stage 1 — intrusive-node embed (scaffolding, flag OFF, unused).**
  `force_intrusive_ref<PoolAllocator<...>>`; add `Refcnt`/`refcnt` +
  `m_orphan_next` (atomic_shared_ptr) + `atomic_intrusive_dispose` ->
  `deallocate_chunk`.  Static-asserts mirror the PoC.  *Gate:* build
  identical, flag-OFF object code unchanged.

- **Stage 2 — self-ref accounting.**  Wire the `MASK_CNT` `0->1` / `1->0`
  edges to `refcnt` bump/drop (boundary-only).  *Gate:* flag-ON unit test
  asserts `refcnt == (MASK_CNT>0) + chain-refs`; hot-path disasm shows only
  a boundary branch added.

- **Stage 3 — owner-exit publish.**  Non-empty branch of
  `release_dll_chunks_for_thread`: push to `g_orphan_head` chain (Treiber,
  atomic_shared_ptr) instead of `orphan_push`.  Empty branch: drop self-ref
  (refcount path) instead of direct `deallocate_chunk`.  *Gate:*
  alloc_thread_churn scenario 2.

- **Stage 4 — empty-eager release.**  Cross-free `atomicDecAndTest`-true:
  `madvise` slot pages immediately (RSS, chain-independent) + drop self-ref;
  chunk released when refcount hits 0.  *Gate:* reserved plateau (macOS
  too — re-check the churn gate `tests/CMakeLists.txt`).

- **Stage 5 — reuse adopt (single head node).**  Replace `orphan_pop` +
  `BIT_OWNED` claim with a single Treiber **head pop** + `try_promote`
  (acquire-if-nonzero = the revival gate).  Pops exactly ONE node — the
  whole chunk with its surviving slots — **NOT the whole list**; the rest
  stay on the chain for other adopters / `scrub`.  The adopted node leaves
  the chain and re-splices into the per-thread DLL as today.  Reclaim
  completeness of mid-chain empties is `scrub`'s job (Stage 6), not pop's.
  (Matches the PoC `pop_head()`; detach-and-reprocess / whole-list steal is
  NOT used.)  *Gate:* alloc_stress + bench.

- **Stage 6 — sweep compaction.**  `orphan_scrub()`: walk the chain holding
  `local_shared_ptr` pins, CAS `pred->m_orphan_next` past empty nodes,
  idempotent-null a dead node's own next.  Multi-sweeper (no
  serialization).  Trigger at owner-exit (garbage-aligned) + opportunistic
  at adopt.  *Gate:* alloc_thread_churn + a churn stress with K survivors.

- **Stage 7 — retire + flip.**  Remove `s_orphan_head`/`orphan_push`/
  `orphan_pop`, the `BIT_OWNED`-arbitration release paths subsumed by
  refcount, and any now-dead seqlock.  Default `KAME_ORPHAN_CHAIN = 1`.
  *Gate:* full `ctest`, `alloc_stress` residual=0, `bench_compare.sh`
  non-regression (esp. 64B 1T/MT), `alloc_thread_churn` on macOS+Linux,
  STM `3level_mixed`.

## Invariants carried from the model (assert in debug builds)

1. `released(c) => MASK_CNT(c)==0` — the deleter's DEBUG_GUARD (the
   `Inv_NoBadRelease` the model enforces).
2. `MASK_CNT(c)>0 => refcnt(c)>=1` — self-ref (lifetime decoupling).
3. a live chunk's `m_orphan_next` never points at a released chunk
   (`Inv_NoDanglingNext`).
4. adopt (`Revive`) only via successful `try_promote` (refcnt was >0).
5. chain stays acyclic; relinks reachability-preserving + dead-only.

## Implementation note — cross-free already defers (no conflict)

Confirmed in `deallocate_pooled`'s OnClearFn (allocator.cpp ~1787): the
current §36 code **does NOT release a chunk on the cross-free dec-to-0 when
BIT_OWNED is clear** — such a chunk is an orphan whose memory must stay valid
for `orphan_pop`; cross-free only decrements MASK_CNT (result ignored).  This
means **replacing `orphan_push` (the §36 stack) with the atomic_shared_ptr
chain push at owner-exit (allocator.cpp:3048) needs NO change to the
cross-free path** — a drained orphan simply stays on the chain (chain-ref
held) until swept/adopted.  No release-vs-chain conflict.  (Distinguished
from owner-exit-empty, a separate path where BIT_OWNED is still set during
the drain.)

Step-1 wiring (chain head + push), then, is:
- `static atomic_shared_ptr<PoolAllocator> s_orphan_chain_head;` in the FS=true
  primary (next to `s_orphan_head`), same injected-class-name type as
  `m_orphan_next` (NOT the `<ALIGN,DUMMY,DUMMY>` erasure — that re-triggers the
  Stage-1 circular incomplete-type).
- `orphan_chain_push(c)`: `c->refcnt.store(1, relaxed)` (owner-private,
  pre-publish — no race) → adopt `local_shared_ptr` → Treiber push via
  `m_orphan_next` + `s_orphan_chain_head.compareAndSwap` (mirror the PoC
  `push_head`).  At the call site, `#if KAME_ORPHAN_CHAIN` selects
  `orphan_chain_push(c)` else `orphan_push(c)` (upcast c to the FS=true-base
  type for the chain).
- flag-ON is INCOMPLETE until step 4 (no pop/scrub ⇒ orphans accumulate =
  leak, but no corruption); full alloc_stress gate lands at step 4–5.

## Step 3 plan — owner-ref via DLL forward = local_shared_ptr (MOST INVASIVE)

DLL access surface (allocator.cpp), to convert under `#if KAME_ORPHAN_CHAIN`:

| site | role | flag-ON handling |
|---|---|---|
| create_allocator / allocate_chunk append (1990–96, 2800–05) | tail append | `refcnt.store(1)` (owner-private) then `tail->m_dll_next = local_shared_ptr(adopt, chunk)` — establishes owner-ref |
| owner_release neighbour unlink (2532–47) | remove a chunk | move the link: `pred->m_dll_next = move(nx->m_dll_next)` (transfers nx's successor's owner-ref); dropping nx's incoming link drops nx's owner-ref → dispose if 0 |
| owner-exit walk (2951–64) | drop the whole DLL | dropping each `m_dll_next` drops owner-refs; empties dispose; non-empties were `orphan_chain_push`'d FIRST (transfer owner→chain) |
| §36 adopt re-splice (2750–54) | orphan_pop adoption | flag-OFF only — replaced by step 4's chain adopt under flag-ON |
| traversal reads (2680, 2877, 2651) | walk | `cur->m_dll_next.get()` (raw read, zero-cost) |

**§36 `orphan_push`/`orphan_pop` (6748, 6774) reuse `m_dll_next` as the stack
link** with raw `=`.  Type-changing `m_dll_next` breaks them, so they must be
flag-gated too (flag-ON never calls them — the chain replaces the stack).

**DllLink abstraction** to localize the flag-conditional:
- `m_dll_next` type: `#if KAME_ORPHAN_CHAIN local_shared_ptr<PoolAllocator> #else PoolAllocator* #endif`.
- `m_dll_prev`, `dll_tail`: stay RAW (back-hint / cursor, non-owning) in both.
- `dll_head` (in ThreadLocalState): becomes `local_shared_ptr<PoolAllocator>`
  under the flag (holds the first chunk's owner-ref).
- accessors: `dll_next_raw(c)` → raw read (`.get()` / direct); link-move /
  link-set helpers for the mutation sites so the per-site bodies stay uniform.
- `m_owner_dll_head_addr` compares (`&s_tls.dll_head`) — still valid (address
  of the TLS handle); the handle's TYPE changed but its address is stable.

Owner-ref lifecycle: established on append (refcnt 0→1) / transferred on
adopt (chain-ref→owner-ref, step 4) and owner-exit-nonempty (owner→chain);
dropped on owner_release / owner-exit-empty (→ dispose if refcnt 0).  NOT
functionally testable until step 4 (no adopt/scrub) — gate is build + flag-OFF
identical; full alloc_stress at step 4.  **Risk: highest of all steps**
(core field type change across the DLL subsystem) — execute as a focused unit
with build iteration, not rushed.

## DECISION (resolves the step-3 blocker) — owner DLL stays RAW

Because the chain link (`m_orphan_next`) and the DLL link (`m_dll_next`) are
SEPARATE fields, **owned chunks need NOT enter the smart_ptr world at all**:

- **Owned chunks: RAW DLL, owner-managed** (existing release path), exactly as
  today.  NOT refcounted, NO owner-ref.  `m_dll_next`/`m_dll_prev`/`dll_head`
  stay raw → the trivial-`__thread` TLS is preserved → **step 3 (DLL →
  local_shared_ptr) is CANCELLED, the TLS blocker is gone, and the 3-ref
  "owner-ref" model is dropped** (it was unnecessary).
- **refcnt / the chain manage ORPHANS ONLY**: `refcnt = chain-ref + pins`.
  A chunk enters the smart_ptr world at owner-exit-push (refcnt 0→1, step 1 —
  already consistent: it stores 1 on a previously-raw/refcnt-0 chunk) and
  leaves it at dispose (refcnt 0).
- Owner-exit-empty: release directly (deallocate_chunk), as today.

Revised plan:
- **step 3 — CANCELLED** (DLL raw).
- **step 4 — scrub-reclaim** (dispose empty orphans off the chain): needs NO
  owner-ref (orphans only) ⇒ makes flag-ON FUNCTIONAL (push + reclaim → no
  leak, reserved bounded).  This is the verified Path-B chain-reclaim core.
- **adopt (reuse of NON-empty orphans into the raw DLL) — DEFERRED**: the one
  spot with a residual-pin hazard (a concurrent sweeper's load_shared pin
  could dispose the chunk after it is re-owned raw — no owner-ref to hold it).
  push + scrub-reclaim already bounds `reserved` (the primary goal); in-place
  reuse is a later optimization needing a defined pin-safe re-own protocol
  (transient hold / quiescence) — NOT the TLS-DLL change.

## (historical) Step 3 BLOCKER (found by build) — owner-ref TLS home vs trivial-TLS

Attempted step 3 (m_dll_next / dll_head → local_shared_ptr).  Most errors were
mechanical (`.get()` for reads, local_shared_ptr `=` for the ~8 mutation
sites), BUT one is a genuine design blocker:

```
allocator.cpp:6693: error: type of thread-local variable has non-trivial destruction
```

Making `dll_head` a `local_shared_ptr` gives `ThreadLocalState s_tls`
(`ALLOC_TLS` = `__thread`) a non-trivial destructor — illegal for `__thread`
(and entangled with KAME_FAST_TSD + the teardown-crash machinery already fixed
in 8dd6365d/§ teardown).  The head chunk's owner-ref needs a per-thread
`local_shared_ptr` home, which hits the trivial-TLS constraint.

Why the obvious escapes don't work:
- **Spin-drain pins at adopt** (pop, wait until refcnt==1, then re-own raw):
  UNRELIABLE — a manual read of the global `refcnt` can't see pins parked in
  atomic_smart_ptr cell local-tags (the same split-refcount issue that killed
  the manual self-ref).  Only `release_tag_ref_` knows the true count.
- **Keep dll_head raw + owner-ref elsewhere**: the owner-ref is fundamentally
  a per-thread persistent `local_shared_ptr` — its natural home IS the TLS.
- **Fresh chunks don't need it** (no residual pins, owner-managed) — true, but
  ADOPTED chunks (from the chain, may carry residual pins) DO, and that ref
  must persist while owned ⇒ per-thread home ⇒ TLS.

Leading resolution (needs a decision + careful verification):
- Change `ALLOC_TLS s_tls` to C++ `thread_local` (non-trivial type allowed),
  and **explicitly drop the owner-refs (clear dll_head / walk-drop m_dll_next)
  in the registered `AllocThreadExitCleanup` callback** — which runs BEFORE
  the dangerous teardown phase — so the TLS dtor is a runtime no-op (dll_head
  already null).  MUST verify the KAME_FAST_TSD fast path (which reads s_tls
  via a TSD-slot offset assuming the current layout) still works, and that no
  owner-ref drop disposes a chunk during teardown (bucket_release_chunk at
  teardown = the crash territory).
- Alternative: a separate `thread_local` just for the owner-ref handles,
  drained explicitly at owner-exit, keeping the hot `s_tls` `__thread` + raw.

Until decided, step 3 is BLOCKED; steps 1–2 (chain push) stand.  The flag-OFF
shipping path is unaffected throughout (all behind KAME_ORPHAN_CHAIN).

## Step-4 measurements (flag-ON, MacBook Air M3) — and what they reveal

- **bench_loop 64B (hot path): NON-REGRESSED.**  flag-OFF median ~106 M/s vs
  flag-ON median ~106 M/s (high run-to-run variance, no systematic gap) —
  scrub is on the cold claim path, the hot freelist path is untouched.
- **alloc_stress flag-ON: PASS**, diff=0, chunks final 16087 ≈ flag-OFF 16086
  — empty-orphan reclaim works, no leak.
- **alloc_thread_churn (survivor churn, scenario 2): flag-ON FAILs ≈ §36.**
  flag-OFF(§36) reserved 256→384 MiB; flag-ON(chain) 288→416 MiB.  FAIL reason
  is "orphaned partially-used chunks STRANDED, not reused": the growth is from
  NON-EMPTY orphans (survivors=3907 held by main), which `scrub` does NOT
  reclaim (it only frees EMPTY orphans).  Reusing them needs **adopt**
  (deferred) — and **even §36's adopt (orphan_pop) does not bound it on
  macOS** (why the churn test is Linux-only-gated).

**Implication:** scrub-reclaim alone does NOT deliver the macOS reserved
improvement — it matches §36.  The survivor-churn growth needs (a) effective
**adopt** (reuse of non-empty orphans' free slots) AND likely (b)
**region-level munmap** (bucket regions are never munmap'd → `reserved` is a
monotonic high-water; even perfect reuse can't shrink it — the "real fix"
flagged early).  §36's adopt failing on macOS points at (b).

**So the flip (step 5) is PREMATURE**: flag-ON is verified, hot-path-clean,
and reclaims empties, but does not beat §36 on macOS `reserved`.  Flipping now
would leave the macOS churn Linux-only-gated as today.  Next: implement adopt
(needs the pin-safe re-own protocol) and investigate region munmap.

## Risks / watch-items

- **Hot path**: Stage 2's boundary branch must NOT add an atomic to the
  per-slot inc/dec.  Verify by disasm (objdump) + 64B bench.
- **`m_orphan_next` vs `m_dll_next`**: distinct fields — the orphan chain
  and the per-thread DLL must not alias links (the §36 stack reused
  `m_dll_next`; the chain cannot, since an orphan may be both chain-linked
  and transiently pin-walked).
- **Disposer re-entrancy**: `atomic_intrusive_dispose -> deallocate_chunk`
  must not allocate (it doesn't) and must run the madvise/claim-clear once.
- **LOCAL_REF_CAPACITY contention** on the hot `g_orphan_head` under many
  concurrent adopters — measure.
