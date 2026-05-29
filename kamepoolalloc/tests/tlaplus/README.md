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
