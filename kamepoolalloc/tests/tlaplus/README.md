# kamepoolalloc — chunk-claim / recycle formal models (TLA+)

Formal models of the allocator's region → unit → chunk claim/recycle
protocol, used to hunt the `alloc_stress_test` double-allocation flake
(sentinel mismatch, ~10–47 % of runs at config `10000 16 1000 20`).
See `../CHUNK_CLAIM_TLA_NOTES.md` for the protocol/state/invariant
brief. These specs are the allocator layer; the STM specs live in
`kamestm/tests/tlaplus/`.

Two investigation sessions converged on the same conclusion via three
complementary specs.

## Specs

| Spec | Layer | Result |
|---|---|---|
| `ChunkClaim.tla` | the `s_back_offset[]` publication race (cross-stride `CHUNK_UNITS`) | pre-fix → **INV2 violation**; post-fix → clean |
| `ChunkAlloc_microscopic.tla` | Phase-5j bit-level reclaim + `m_flags_packed` (MASK_CNT) inc/dec accounting (FS=true, 1 bit/slot) | **clean** |
| `ChunkRecycle_microscopic.tla` | address-level `lookup_chunk`/`deallocate` + `deallocate_chunk→allocate_chunk` recycle handover | **VIOLATION** iff double-payout allowed |
| `ChunkRecycle_threadepoch.tla` | **candidate root-cure**: per-thread epoch + seqlock `lookup_chunk` | **verified SAFE** (3.69M states); ABA only on deliberate counter wrap |

## Running

`tla2tools.jar` is gitignored here (an identical copy is committed in
`kamestm/tests/tlaplus/`). To run:

```sh
cp ../../../kamestm/tests/tlaplus/tla2tools.jar .
java -cp tla2tools.jar tlc2.TLC -config <cfg> <Spec>.tla
```

## The three results

### 1. `ChunkClaim` — back_offset speculative-write clobber (FIXED)
`Speculative=TRUE` (pre-fix: `s_back_offset[]` written before the claim
CAS) → **INV2 `BackoffConsistent` violation**: a CAS loser of a
different `CHUNK_UNITS` stride clobbers the per-unit back_offset entries
the CAS winner owns. `Speculative=FALSE` (back_offset published only
after the CAS wins) → clean. This is the fix in commit *"fix
back_offset speculative-write data race"* (move the write inside the
CAS-success branch).

### 2. `ChunkAlloc_microscopic` — MASK_CNT accounting (CLEAN)
Splits allocate into `A_CAS → A_BumpCount` (the separate `atomicInc`)
and the cross-free `atomicDecAndTest` at op granularity, multi-word
`m_flags_packed.count`. No error: `Inv_NoDoubleClaim`,
`Inv_AtMostOneReclaim`, `Inv_NoUseAfterReclaim`,
`Inv_FlagsPackedConsistency` all hold. **Rules out** the MASK_CNT
inc/dec non-atomicity that was the leading residual hypothesis — it is
NOT the bug.

### 3. `ChunkRecycle_microscopic` — double-payout → bit-state -3 (ROOT)
Models the two unsynchronised `lookup_chunk` loads (`back_offset`, then
`palloc`) across a reclaim+recycle. 3-way controlled experiment
(`Threads={1,2}, NumUnits=2, MaxGen=3`):

| `FreeBeforeLookup` | `SinglePayout` | result |
|---|---|---|
| TRUE  | FALSE | VIOLATION (389 distinct) |
| FALSE (faithful) | FALSE | **VIOLATION (706 distinct)** |
| FALSE (faithful) | TRUE  | **NO ERROR (400 distinct)** |

Decisive: under the *faithful* lookup ordering (bit cleared only after
the lookup resolves), the corruption appears **iff** a slot is
double-paid-out to two threads. Forbidding double-payout makes the
address-level lookup+reclaim+recycle provably safe.

## Converged conclusion

- **First-order cause = double-payout** (two threads handed the same
  slot). FS=true single-bit (spec 2) is clean, so the remaining
  source is in the **FS=false / buddy claim-recycle** path.
- The `ChunkClaim` back_offset clobber (spec 1, fixed) was the dominant
  double-payout source: empirically ~40 % → ~0.24 % residual.
- **Second-order amplifier = `lookup_chunk` trusts `palloc != 0`
  alone** — no generation/identity check, two unsynchronised loads. It
  silently returns a recycled chunk and computes an out-of-range slot
  index (the `bit-state = -3` signature) once any upstream early-free
  occurs.

## Recommended fix — VERIFIED root-cure (`ChunkRecycle_threadepoch`)

The chunk-recycle race (a `lookup_chunk` resolving a recycled chunk's
layout for an in-flight old-generation slot) is closed by a
**per-thread epoch + seqlock `lookup_chunk`**, proven sound by TLC
(3,693,839 states, all invariants hold; ABA appears only under a
deliberately too-narrow counter that wraps).

Three components — **all three necessary** (TLC found a violation when
any was dropped):

1. **Per-thread epoch counter in TLS** (no global atomic): the epoch
   is the pair `<<owner_tid, counter>>`, globally unique without
   contention.
2. **`unitMeta[u]` = packed `{back_off, epoch}` in ONE atomic word**
   (e.g. `uint64_t` = 8-bit back_off + 16-bit tid + 40-bit counter).
   Separate atomic loads of back_off and palloc (the current code)
   are what produce bit-state -3.
3. **Seqlock re-read in `lookup_chunk`**: load `meta1`, load `palloc`
   (acquire fence between), re-load `meta2`, and require
   `meta1 == meta2` AND `meta.epoch == expected`. A single pair-atomic
   load is insufficient — a reclaim slipping in between the two loads
   goes undetected without the re-read.

Implementation hint (from the spec): `std::atomic<uint64_t>` per unit
holding `<<8-bit back_off, 16-bit tid, 40-bit counter>>`; lookup does
two relaxed loads of `chunk_base.unitMeta` (acquire fence before the
palloc load) and rejects on inequality / epoch mismatch, falling
through to libsystem free. The chunk-header epoch becomes a secondary
defense, not the primary identity.

Counter width: 40-bit at ≥1 ns/alloc wraps in centuries — safe in
practice; the `wrap` cfg only violates because it forces wrap at
`MaxLocalEpoch=1` to demonstrate the bound is real.

### Relation to the shipped fix
Two committed changes close the slot-resolution race in shipped code,
*without* the epoch+seqlock:

1. The `ChunkClaim` back_offset-after-CAS publish (commit `d2e2c32b`)
   removed the dominant double-payout source (the ~40 % → ~0.24 % step).
2. The current live-slot resolver (`resolve_chunk_from_slot` / `deallocate`,
   "NO seqlock, NO epoch") makes the reclaim+recycle race **structurally
   impossible** on the lookup/free path: the pointer being resolved is a
   LIVE slot, which keeps `m_flags_packed != 0`, which is the precondition
   for EVERY chunk-release path (owner_release, cross-flush dec-to-zero,
   thread-exit).  The chunk therefore cannot be released — let alone
   recycled into a *different* chunk — while the resolution runs, so the
   two-unsynchronised-loads window the model exploited cannot open.  The
   seqlock validation is unnecessary here.

So the residual the ad-hoc fix left is closed in shipped code by (2), not
by amplification-prevention.  The epoch+seqlock (`ChunkRecycle_threadepoch`)
remains the verified root-cure for the *other* caller class — a DLL-walk
holding a chunk POINTER without holding any slot in it (those paths gate on
BIT_OWNED instead) — and as the complete-coverage alternative; it is not
needed for slot resolution.  See also the GenMC `cds_seqlock_recycle.c`
fence check above and `cds/cds_radix_install.c`.

### GenMC C11 memory-ordering complement (`../cds/`)

TLC verifies the **logical** protocol but assumes SC and collapses the
writer (claim + meta-write + data-write) into ONE atomic step. The
weak-memory (RC11) realizability of the C11 orderings is checked
separately by the GenMC suite under [`../cds/`](../cds) (`make run`):

| Test | Checks | Result (GenMC v0.17, `--rc11 --unroll=5`) |
|---|---|---|
| `cds_seqlock_recycle.c` | the candidate seqlock `lookup_chunk` re-read with a **multi-store writer** — the reader's acquire fence the atomic-writer TLC model cannot exercise | **clean, 81 exec**; negative control `make run-seqlock-nofence` (fence removed) → **torn-read Safety violation** (fence is load-bearing) |
| `cds_radix_install.c` | §13 radix L2 lazy-install single-winner CAS (INV-9) | **clean, 42 exec** |

Two findings from this pass worth recording: (1) GenMC/RC11 does **not**
treat a failed `compare_exchange_strong`'s `failure=acquire` as
synchronising with a release-CAS writer (it keys the RMW read order off
the *success* order) — harmless for `radix_insert` (its loser only
`entries[].store()`s into an all-atomic, mmap-zeroed leaf and reads no
winner-initialised non-atomic state), but a sharp edge for any future
CAS whose loser dereferences winner-initialised data. (2) The shipped
`lookup_chunk` is **seqlock-free** (live-slot invariant; covered by the
`ChunkClaim` GenMC test) — `cds_seqlock_recycle.c` verifies the
*candidate* epoch+seqlock root-cure's fence, not shipped code.

---

## §36b orphan-reuse array — TLA+ verification (June 2026)

`OrphanReuse_36b*.tla` models the candidate §36b design from
`kamepoolalloc/ORPHAN_REUSE_HANDOFF.md` (versioned per-template slot
array + `m_orphan_disp` arbiter), commissioned to fix the §36
orphan-chunk LEAK without re-introducing corruption.  The C++ prototype
on branch `claude/youthful-mayer-FUhq2` plateaus the leak but
stress-tests show **5/300 (1.67 %) corruption** = 4× ABA-style
`ORPHAN_RELEASE_BAD` + 1× sentinel mismatch.  This TLA+ work pins down
both modes.

| Spec | Configuration | Result |
|---|---|---|
| `OrphanReuse_36b.tla` | as-documented design | **VIOLATION** (`Inv_NoBadRelease`) at depth 6 |
| `OrphanReuse_36b_REOWNED.tla` | + PoP2 stores REOWNED (not OWNED) | **VIOLATION** at depth 9 — deeper race |
| `OrphanReuse_optC.tla` | drop the array, restore release-on-empty | **clean** (6 states) |

### Race 1 — ABA at PoP2 (`OrphanReuse_36b.tla`, depth 6)

Reproduces the 1.3 % `ORPHAN_RELEASE_BAD` mode.

```
1. Init             chunk on slot 0, mask_cnt=1, bit_owned=F
2. PopP0(t1, 0)     slot 0 emptied, reown_pc[t1]=got_c
                    (disp still 1 — STALE)
3. FreeDecToZero    bit_owned=F at this instant → pending_claim[t1]=TRUE
4. PopP1(t1)        bit_owned=TRUE  ← popper re-acquires ownership
5. PopP2(t1)        disp = OWNED
6. ClaimOWNED(t1)   disp==OWNED → CAS OWNED→RELEASED succeeds
                    bad_release because bit_owned=TRUE  💥
```

**Root cause:** `disp = OWNED` is overloaded — it means BOTH
"never-pushed, ready for orphan_push" AND "popped + re-armed".  The
freer's claim-for-release CAS can't tell them apart.

### Race 2 — life-cycle-spanning ABA (`OrphanReuse_36b_REOWNED.tla`, depth 9)

The naïve fix (PoP2 stores REOWNED instead of OWNED) closes Race 1
but exposes the underlying *temporal* defect:

```
1-4. (as before, with disp = 1 stale, pending_claim[t1]=TRUE,
                       bit_owned=TRUE)
5.  ReownAllocSlot(t1)    mask_cnt 0→1 (allocator hands out a fresh slot)
6.  PopP2(t1)             disp = REOWNED
7.  OwnerExit_NonEmpty    bit_owned=F, disp=PUSHING
8.  PushP1(t1, 0)         slot 0 = (C, ver=3), disp = 1
9.  ClaimSLOT(t1, 0)      reads (C, v=3), CAS succeeds → RELEASE
                          bad_release because mask_cnt=1 (live!)  💥
```

The slot version went 1 → 2 (PopP0) → 3 (PushP1).  The freer's
`pending_claim[t1]` was set at step 3 but never carried a captured
slot version — at step 9 it just reads `(C, *)` and matches whatever
is there.  This matches the stress-test's sentinel-mismatch mode.

**Fundamental defect:** the per-slot version protects against a
same-slot ABA between *adjacent* push/pop ops, but cannot detect a
*full life-cycle* (`pop → re-own → exit → re-push`) since the freer's
`atomicDecAndTest()==true` return value carries no chunk-generation /
captured-version with it.  The §36b design needs an additional epoch
or generation captured at dec instant, matched at claim CAS — non-
trivial.

### Verdict

`OrphanReuse_optC.tla` confirms the handoff's §8 recommendation: drop
the array entirely; restore pre-§36 release-on-empty in the cross-
thread dec-to-0 path.  This is **provably free of `bad_release`** in
the model, matches the on-branch `1e65bdd` ("option B") prototype's
empirical leak-plateau on Linux churn, and gives up only the in-place
reuse of non-empty orphans (their free slots drain + recycle warm
instead of being re-handed-out by a new owner).

If §36b reuse is later judged essential, the model points at the only
sound completion: capture `<chunk_gen, slot_ver>` at the dec-and-test
instant (snapshot returned by `atomicDecAndTest`'s extended variant)
and require the claim CAS to match BOTH the disp slot AND the captured
generation.  Without that, the freer cannot distinguish "release the
chunk I just decremented to 0" from "release whatever chunk happens to
sit at this disp now".

### Running
```
java -cp tla2tools.jar tlc2.TLC -config OrphanReuse_36b_2thr_mc.cfg OrphanReuse_36b.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanReuse_36b_REOWNED_2thr_mc.cfg OrphanReuse_36b_REOWNED.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanReuse_optC_2thr_mc.cfg OrphanReuse_optC.tla
```

All three terminate in <1s on a single core.  No model-checker tuning
required — the state space is small (≤ a few hundred distinct states).

---

## OrphanChain_atomicshared — refcount-as-generation, the sound completion (June 2026)

`OrphanChain_atomicshared.tla` is the successor to §36b.  The §36b verdict
above said in-place orphan reuse is only sound if a `<chunk_gen, slot_ver>`
generation is *captured* at the dec-and-test instant and matched at the
claim CAS.  This spec realises exactly that — but with the **verified
atomic_shared_ptr REFCOUNT as the captured generation**, instead of a
hand-rolled version/disp word.  The intrusive singly-linked orphan/reuse
chain is `atomic_shared_ptr` (head) / `atomic_shared_ptr` next; the
local_shared_ptr pin held across the dec/claim *is* the generation capture,
and `try_promote` (acquire-if-nonzero) is the generation-matched claim.

Layer 0 (`../../../kamestm/tests/tlaplus/atomic_shared_ptr.tla`) already
proves the refcount/CAS primitive.  This spec **abstracts** it and verifies
the chain protocol + the lifetime gate that sit on top:

- **Self-ref on `m_filled`** — a chunk holds one structural ref to itself
  while it has live slots, dropped at the `m_filled -> 0` edge (one bump per
  non-empty/empty edge, *not* per slot — the hot path never refcounts).
  This **decouples chunk lifetime from chain membership**.
- **Multiple sweepers, safe-side only** — any thread may relink past a dead
  successor (CAS, reachability-preserving) or idempotently null a dead
  node's own `next`.  Because lifetime is decoupled, every structural race
  (lost update, tail-loss, a live node falling off the chain, a dead node
  surviving a round) costs only a *reuse hint* — never a leak or a UAF.  No
  single-sweeper serialization.
- **Release vs revival** — the one non-safe-side edge.  Revive (reuse) is
  gated by `StructRefs > 0` (try_promote success); Release fires at
  `StructRefs = 0`.  Mutually exclusive in every state ⇒ a revived chunk is
  never released, a released chunk never revived.  The atomicity of the
  try_promote-vs-final-unref boundary is Layer 0's, cited not re-proved.
  This single atomic gate is what §36b's 3-step (PoP0/1/2) re-own could not
  give — and is why this spec is **CLEAN where `OrphanReuse_36b` is not**.

Microscopic model: one chain `head -> N1 -> N2(dead) -> N3 -> NIL`,
`m_filled ∈ {0,1}` (only the zero boundary drives the protocol), revivals
bounded by `MaxGen` (precondition counter — no `StateConstraint`).  Threads
are not modelled: every op is a single atomic memory access with no
thread-local pc, so interleaving node-indexed steps already covers N
concurrent sweepers (the multi-step §36b re-own collapses to one atomic
`try_promote`).

| Cfg | `SelfRef` | Result |
|---|---|---|
| `OrphanChain_atomicshared_selfref_mc.cfg`   | TRUE  | **CLEAN** — 269 distinct; `Inv_NoBadRelease`, `Inv_LiveNeverReleased`, `Inv_NoDanglingNext`, `Inv_HeadAlive`, `Inv_ReleasedNoIncoming`, `Inv_Acyclic` all hold |
| `OrphanChain_atomicshared_noselfref_mc.cfg` | FALSE | **VIOLATION** (`Inv_NoBadRelease`, depth 3) — proves the self-ref is load-bearing |
| `OrphanChain_atomicshared_live_mc.cfg`      | TRUE  | **Liveness HOLDS** (no leak) under WF on the reclaim actions |
| `OrphanChain_atomicshared_push_mc.cfg`      | TRUE  | + head insert (`MaxPush=1`): **CLEAN**, 541 distinct — the Push × HeadAdvance race preserves every invariant |

### Head insert (Push) — covered, and why it is mostly Layer 0
Owner-exit re-publishes a non-empty orphan by CASing it at the head
(Treiber push).  Its races decompose as: **Push-vs-Push and the head-cell
ABA are Layer 0** (`atomic_shared_ptr.tla` — the refcount/pin kills the
ABA); **Push vs interior sweep is disjoint** (Push touches only
`{head, new->next}`); and **Push only ADDS refs / forward links
(monotone)**, so it cannot of itself produce a bad release, a dangling
link, or a cycle.  The one genuinely new interleaving — **Push vs
HeadAdvance, both CAS `head`** — is modelled in `*_push_mc.cfg`
(`MaxPush=1`) and is CLEAN (541 distinct), confirming the protocol
invariants survive that combination.  Re-pushing a fallen-off LIVE orphan
also shows fall-off is recoverable.

### Why the `SelfRef=FALSE` knob violates (the load-bearing proof)
```
1. Init                head -> N1(live) -> N2(dead) -> N3(live) -> NIL
2. SelfResetNext(N2)    N2 nulls its own next  (N2 -> NIL)
                        => N3's only incoming ref (from N2) is gone
3. Release(N3)          StructRefs(N3)=0 (no chain-in, no self-ref since
                        SelfRef=FALSE) -> deleter fires while m_filled=1  💥
```
A live successor that falls off the chain is released with live slots.
With `SelfRef=TRUE` the self-ref keeps `StructRefs(N3) ≥ 1`, so the same
trace is *safe-side* (N3 lives off-chain until it drains, then releases).

### Verdict
The §36b array (`OrphanReuse_36b`) **violates**; `OrphanReuse_optC` is
**clean but gives up reuse**; `OrphanChain_atomicshared` **recovers reuse
and is clean** — the refcount is the generation capture the §36b verdict
demanded, and the self-ref on `m_filled` is what makes the whole multi-
sweeper chain safe-side.  This is the formal backing for the
"orphan-atomicshared" chunk-release redesign.

### Running
```
cp ../../../kamestm/tests/tlaplus/tla2tools.jar .
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_atomicshared_selfref_mc.cfg   OrphanChain_atomicshared.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_atomicshared_noselfref_mc.cfg OrphanChain_atomicshared.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_atomicshared_live_mc.cfg      OrphanChain_atomicshared.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_atomicshared_push_mc.cfg      OrphanChain_atomicshared.tla
```
All terminate in <1s on a single core.

---

## OrphanChain_pathB — separate-counts variant (the decided design)

`OrphanChain_pathB.tla` is the SEPARATE-COUNTS successor to
`OrphanChain_atomicshared.tla`.  Designing the C++ integration revealed that
a count-based self-ref (manual `refcnt.fetch_add/sub` at the MASK_CNT edge)
races atomic_smart_ptr's split (local-tag) reference counting — a manual
`fetch_sub` to 0 while a `load_shared` reader's ref sits in a cell's local
tag → premature dispose / UAF.  So MASK_CNT and the intrusive refcnt are kept
SEPARATE: the refcnt is managed EXCLUSIVELY by atomic_smart_ptr
(`refcnt = owner-ref + chain-ref [+ pins]`, **no manual ops**, never touched
by alloc/free), and "don't release a non-empty chunk" is **structural** —
the sweeper removes ONLY dead nodes and relink preserves successors, so a
non-empty node never loses its incoming chain-ref; an owned chunk has an
owner-ref (a `local_shared_ptr` in the per-thread DLL).  Hence
`refcnt→0 ⟹ MASK_CNT==0`; the disposer asserts it.  No self-reset (an
unlinked dead node's dtor drops its `m_orphan_next`).

| Cfg | knob | Result |
|---|---|---|
| `OrphanChain_pathB_mc.cfg`          | `AllowLiveRemoval=FALSE` (design) | **CLEAN** — 1944 distinct; `Inv_NoBadRelease`, `Inv_NonEmptyHasRef`, `Inv_NoDanglingNext`, `Inv_HeadAlive`, `Inv_ReleasedNoRefs`, `Inv_Acyclic` all hold |
| `OrphanChain_pathB_liveremoval_mc.cfg` | `AllowLiveRemoval=TRUE` | **VIOLATION** (`Inv_NoBadRelease`, depth 4) — proves dead-only removal is load-bearing |
| `OrphanChain_pathB_live_mc.cfg`     | design + fairness | **Liveness HOLDS** (no leak) |

### Why the `AllowLiveRemoval=TRUE` knob violates
```
1. Init                      N1 owned, non-empty
2. OwnerExitNonEmpty(N1)     owner-ref → chain-ref (pushed at head)
3. HeadAdvance (live!)       drops the non-empty head N1 off the chain
                             (only allowed under the knob)
4. Release(N1)               N1 now unowned + off-chain → refcnt 0 → dispose
                             while filled=1  💥
```
With dead-only removal (`FALSE`) step 3 cannot fire on a non-empty node, so a
non-empty orphan keeps its chain-ref until it drains — the structural
replacement for the self-ref.  This is the Path-B analogue of
`OrphanChain_atomicshared`'s `SelfRef=FALSE` violation; both pin down the one
load-bearing rule of their respective design.

### Running
```
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_pathB_mc.cfg             OrphanChain_pathB.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_pathB_liveremoval_mc.cfg OrphanChain_pathB.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_pathB_live_mc.cfg        OrphanChain_pathB.tla
```
All terminate in <1s on a single core.

## OrphanChain_adopt — the implemented adopt mechanism (the Linux-review watch-item)

`OrphanChain_adopt.tla` models the ACTUAL adopt code shipped behind
`KAME_ORPHAN_CHAIN` (commit `bb6d691d`), closing the watch-item raised in the
Linux review: *"adopt is a non-atomic 2-step `pop (remove from chain) →
BIT_OWNED CAS`; confirm the TLA model covers it together with the residual scrub
pin drain."*  Earlier chain models (`pathB`, `atomicshared`) modelled adopt as a
single atomic step and used the since-dropped owner-ref; this one is faithful to
the raw-DLL design and splits adopt into its three real atomic steps
(`orphan_chain_pop` → claim `BIT_OWNED` → drop `oc_hold`), with disposal folded
into each reference-dropping action so `refcnt=0 ⟺ released` (matching
`atomic_shared_ptr`'s synchronous deleter — no node lingers at refcnt 0 still
pointing at its successor).

It verifies **two independent safety properties**, because the code has **two
distinct free mechanisms**:

| Free mechanism | Code | Gated by |
|---|---|---|
| (1) smart_ptr disposal | `atomic_intrusive_dispose` (`allocator_prv.h:1917`) | **`BIT_OWNED`** (`if(BIT_OWNED) return;`) |
| (2) owner direct free | `release_dll_chunks_for_thread` empty branch (`allocator.cpp:3026`, `newv==0`) | **`MASK_CNT==0` only — does NOT read `refcnt`** |

| Cfg | knob | Result |
|---|---|---|
| `OrphanChain_adopt_mc.cfg`          | `GateOnOwned=TRUE, OwnerRef=FALSE` (shipped) | `Inv_NoBadRelease` **CLEAN** (1978 distinct) — gate (1) is correct; but `Inv_NoBadOwnerFree` **VIOLATION** (depth 7) — see below |
| `OrphanChain_adopt_nogate_mc.cfg`   | `GateOnOwned=FALSE`         | `Inv_NoBadRelease` **VIOLATION** (depth 5) — proves the `BIT_OWNED` disposal gate is load-bearing |
| `OrphanChain_adopt_ownerref_mc.cfg` | `OwnerRef=TRUE` (the fix)   | **CLEAN** (1715 distinct) — `Inv_NoBadRelease`, `Inv_NoBadOwnerFree`, `Inv_ReleasedNoRefs`, `Inv_NoDanglingNext`, `Inv_Acyclic` all hold |

### Result (1): the disposal gate is correct and load-bearing
`Inv_NoBadRelease` (smart_ptr disposal never frees an owned/non-empty chunk) is
**CLEAN** under `GateOnOwned=TRUE` and **VIOLATED** under `FALSE`.  The 2-step
`pop → claim` window is safe: `oc_hold` keeps the chunk alive through the claim,
and after `BIT_OWNED` is set a residual pin draining to refcnt 0 hits the gated
disposer, which no-ops.  This is exactly the watch-item's named concern, and the
gate handles it.

### Result (2): the OWNER's direct free does NOT honour the refcnt — `Inv_NoBadOwnerFree` VIOLATION
The disposer comment (`allocator_prv.h:1915`) states the design's **precondition**
for free mechanism (2): *"the owner releases it via deallocate_chunk on empty
(refcnt then stays 0, off every smart_ptr)."*  `Inv_NoBadOwnerFree` tests whether
that precondition is **enforced** or merely **timing-based**.  It is **VIOLATED**:
```
1. Free(n1)         n1 drains (MASK_CNT→0) while still an orphan on the chain
2. ScrubPin(n1)     a concurrent scrubber load_shared-pins n1   (refcnt += 1)
                    — the real `local_shared_ptr cur(s_orphan_chain_head)`
3. AdoptPop(n1)     another thread pops n1 (head→n2, oc_hold=n1)
4. AdoptClaim(n1)   sets BIT_OWNED  (re-owned, raw)
5. AdoptDropRef(n1) drops oc_hold; refcnt = scrub pin only = 1; disposal GATED
                    (no-op) — Result (1) holds here
6. OwnerExitEmpty(n1)  owner deallocate_chunk's n1 (empty) — but the scrub pin
                       is STILL outstanding (refcnt=1)  💥  bad_ownerfree
```
The minimal scenario does not depend on scrub-walk subtleties: a scrubber that is
preempted **immediately after** `local_shared_ptr cur(s_orphan_chain_head)` pins
the head holds a strong ref while another thread runs the entire
adopt → drain → owner-exit; the owner's `deallocate_chunk` ignores that ref, so
the pin dangles and the scrubber's later `cur` deref / `release_tag_ref_`
corrupts the freed-and-recycled chunk's control block.  The `BIT_OWNED` gate
covers *pin-drains-before-owner-free* (disposal no-ops); it does **not** cover
*owner-free-before-pin-drains*, and no happens-before forces the former ordering.

**Status:** Path B is flag-OFF by default (`KAME_ORPHAN_CHAIN=0`), so shipping
code (§36, which leaks rather than scrubs and has no smart_ptr pins) is
unaffected.  This gap is specific to the new scrub+adopt path and must be closed
before flipping the flag (Stage 7).  Candidate fixes: (a) the owner empty-free
branch consults `refcnt.load(acquire)` after clearing `BIT_OWNED` and, when
`>0`, defers to the (now-ungated) disposer instead of calling `deallocate_chunk`
— with a single-writer claim to avoid an owner-vs-disposer double free; or
(b) document it as a timing-based precondition (the pin window is a few
instructions; an owner-free requires a full adopt+drain+exit) — weaker, and
formally a UAF under arbitrary preemption.

**Note on the abstraction:** `ScrubPin`/`ScrubUnpin` model the pin lifetime
liberally (held until an independent unpin), over-approximating the real
walk-coupled lifetime.  The general trace therefore over-approximates, but the
**minimal preempt-after-head-pin scenario above is exact** — the pin is a real
`load_shared` strong ref and the owner-free path provably never reads `refcnt`
(verified by grep: `refcnt` appears only in `orphan_chain_push`'s `store(1)` and
the adopt's `local_shared_ptr` adoption).

### Result (3): the owner-ref fix closes the gap — `OwnerRef=TRUE` CLEAN
The `OwnerRef=TRUE` knob models the fix: a re-owned chunk carries an OWNER-REF —
a `local_shared_ptr<PoolAllocator>` the CHUNK HOLDS TO ITSELF
(`m_owner_self_ref`), **separate from the raw DLL (which is untouched)**.  At
adopt the popped `oc_hold` (which already points at the chunk) is moved into
`m_owner_self_ref` instead of dropped; at owner-free / thread-exit the owner
`m_owner_self_ref.reset()`s it.  Then `refcnt = chain + pins + owner-ref`, and the
owner does NOT call `deallocate_chunk` directly — the reset drops the self-ref and
the chunk is freed by whoever takes refcnt to 0 (the owner if no pins remain, else
the last scrub pin) via the disposer.  **Every free is refcnt-mediated.**  Under
`OwnerRef=TRUE` all safety invariants are **CLEAN** (1715 distinct):
`Inv_NoBadOwnerFree` is never set (no direct free), and `Inv_ReleasedNoRefs` /
`Inv_NoDanglingNext` hold (a chunk a pin still references is kept alive until the
pin drops).  With the owner-ref an owned chunk always has `refcnt ≥ 1`, so
disposal never fires while owned — the `BIT_OWNED` gate becomes **redundant** and
the disposer can instead `assert(MASK_CNT == 0)`.

The model is **agnostic to where the +1 owner-ref lives** — it accounts only
"owned ⇒ +1 refcnt, dropped at owner-free" — so this CLEAN result verifies the
self-ref realization (and equally the DLL-held owner-ref of `OrphanChain_pathB`).
The self-ref is a chunk→self cycle, broken explicitly by the `reset`, modelled as
the owner-free drop.

`Inv_OwnedNotChained` ("owned ⇒ not chained") is intentionally NOT checked: a
pin'd predecessor's stale `m_orphan_next` can transiently point at an owned node
in BOTH designs, so it is simply not an invariant.  Its *consequence* is harmful
only without the owner-ref (a free-while-referenced, caught by
`Inv_NoBadOwnerFree`); with the owner-ref the extra incoming ref merely keeps the
chunk alive — harmless.

**Implementation note (separate from the DLL; no `__thread` blocker).** The
owner-ref is a self-held member `local_shared_ptr<PoolAllocator> m_owner_self_ref`
INSIDE the heap chunk object — so it is unaffected by the `s_tls` `__thread`
triviality rule (`allocator_prv.h:224,1816`), needs no heap anchor, and does NOT
touch the raw DLL (`m_dll_next`/`dll_head` stay raw pointers).  It is set at the
adopt claim (move `oc_hold` in) and `reset()` at owner-free and in
`AllocThreadExitCleanup` (cache the next DLL pointer BEFORE the reset, since a
`reset` that takes refcnt to 0 disposes the chunk synchronously; `~PoolAllocator`
then sees `m_owner_self_ref` already null — no double decrement).  Crucially this
is a PROPER `local_shared_ptr` self-ref, NOT the reverted MANUAL
`refcnt.fetch_add/sub` self-ref (commit `7db53986`) that raced a `load_shared`
reader's parked local tag: set/`reset` go through atomic_smart_ptr's split-tag
release protocol, whose safety is verified at Layer 0
(`OrphanChain_atomicshared.tla` / the GenMC `cds_atomic_shared_ptr` tests).

### Running
```
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_adopt_mc.cfg          OrphanChain_adopt.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_adopt_nogate_mc.cfg   OrphanChain_adopt.tla
java -cp tla2tools.jar tlc2.TLC -config OrphanChain_adopt_ownerref_mc.cfg OrphanChain_adopt.tla
```
All terminate in <1s on a single core.

## Regression guard — `run_orphan_chain.sh`

`./run_orphan_chain.sh` runs every `OrphanChain_*` model with each cfg and
ASSERTS the expected outcome (CLEAN, or a specific invariant VIOLATION).  It is
a true gate, not a smoke test: a cfg that is *supposed* to violate (a gate or
self-ref knob turned off, or the shipped raw-DLL design under
`OrphanChain_adopt`) must STILL violate, else the model has silently weakened.

```
$ ./run_orphan_chain.sh            # all 10 checks (<10s total)
$ ./run_orphan_chain.sh adopt      # only the adopt cfgs
$ STRICT=1 ./run_orphan_chain.sh   # missing tla2tools.jar → hard fail (CI)
```

**Keep this as a standing regression guard — including after the
`KAME_ORPHAN_CHAIN` Stage-7 flip.**  The owner-free vs residual-scrub-pin race
(`Inv_NoBadOwnerFree`, closed by the chunk self-ref owner-ref) is a narrow
timing-dependent UAF that runtime stress **cannot** reproduce: 21.32M-op
ASan/TSan on the pre-fix code did not trip it.  This model is the *only* guard
for that class. Re-run it on any change to `allocator.cpp`'s orphan-chain /
adopt / owner-free paths.  Requires `tla2tools.jar` in this directory (gitignored
— `cp ../../../kamestm/tests/tlaplus/tla2tools.jar .`).
