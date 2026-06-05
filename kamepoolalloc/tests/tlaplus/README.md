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

### Relation to the shipped ad-hoc fix
The `ChunkClaim` back_offset-after-CAS fix (committed) removed the
*dominant* double-payout source (~40 % → ~0.24 %). The epoch+seqlock
design is the complete root-cure: it makes `lookup_chunk` *reject* a
stale/recycled resolution, so any residual double-payout source can no
longer be amplified into out-of-range corruption — it degrades to a
safe fallthrough.

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
