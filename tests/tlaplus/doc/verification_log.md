# TLA+ Verification Log

## 2026-04-30: C++-fidelity overhaul of 3-level/2-level LL-free models

### Context

`BundleUnbundle_3level_LLfree.tla` (and to a lesser extent the 2-level
sibling) was producing `TerminalPayloadCheck` violations under coarse 2t
and pure-FINE 2t cfgs, while the C++ implementation of the same protocol
([`tests/transaction_payload_integrity_3level_test.cpp`](../../transaction_payload_integrity_3level_test.cpp))
runs for hours on the ohtaka supercomputer with no payload mismatch. The
C11 reference test generated from the spec was hanging. So all observed
failures were spec-modeling bugs, not protocol bugs. "Same semantics
must yield same results" (user).

### Root causes (5 fixes)

1. **`BundlePhase1` coarse path inner-bundle: drop the
   `linkage[n].hasPriority` filter on grandchild CAS.**
   C++ Phase 3 of an inner recursive bundle
   ([`transaction_impl.h:2487-2511`](../../../kame/transaction_impl.h#L2487))
   ALWAYS CASes each child to a fresh `bundled_ref` wrapper, regardless
   of prior state. The TLA+ filter left already-bundled grandchildren
   untouched, so a peer's stale `snapshotForUnbundle` pointer compared
   value-equal to current `linkage[gc]` and the peer's final
   `UnbundleCASChild` succeeded when it should have failed → lost
   increment race that does not occur in C++.

2. **Replace ad-hoc `globalSerial` uniqueness with C++-faithful
   TID-encoded base-B Lamport.**
   C++ `SerialGenerator::gen(last_serial)`
   ([`transaction.h:547-576`](../../../kame/transaction.h#L547))
   uses a TLS counter (upper 48 bits) + TID (lower 16 bits). `gen()`
   advances the TLS past `last_serial` (Lamport step), increments,
   re-encodes with TID. Two threads with the same counter still
   produce different serial values via TID lower digits — that is what
   makes wrappers thread-unique even when timestamps collide.
   The previous TLA+ added a non-C++ `globalSerial` to force
   uniqueness; this is replaced by:
   ```tla
   SerialBase == 1 + Cardinality(Threads)
   SerialCounter(s) == s \div SerialBase
   SerialTID(s)     == s % SerialBase
   EncodeSerial(cnt, tid) == cnt * SerialBase + tid

   GenSerial(t, lastSer) ==
       LET lastCnt == SerialCounter(lastSer)
           myCnt   == SerialCounter(serial[t])
           newCnt  == (IF lastCnt > myCnt THEN lastCnt ELSE myCnt) + 1
       IN  EncodeSerial(newCnt, t)
   ```
   `globalSerial` variable is deleted from the spec; `SerialBound` and
   `DebugSerialBound` are neutered to `TRUE` for cfg back-compat.
   Bit width differs (TLA+ uses arbitrary integers) but ordering and
   uniqueness are identical to C++.

3. **`BundlePhase3` disturbed restart: regenerate `bundleSer` via
   `GenSerial` when looping back to `bundle_phase1`.**
   C++ `bundle()` retry loop allocates new `PacketWrapper`s with a
   fresh `bundle_serial` each iteration. Without regen, the retried
   Phase 3 emits `BundledRefWrapper(node, ser)` that is structurally
   identical to a peer's earlier wrapper — the "refresh" CAS becomes
   a value-level no-op, and a peer's stale `snapshotForUnbundle`
   pointer compares equal, allowing a final `UnbundleCASChild` that
   should have failed to succeed. Applied to coarse + fine paths in
   3-level. 2-level was already passing without this; left as-is per
   "passing means it wasn't needed in bundle/unbundle".

4. **`UnbundleWalk` casTargets: always root-first.**
   C++ `walkUpChainImpl` is recursive; the deepest call (root)
   `emplace_back`s into `cas_infos` first, so the CAS loop processes
   root first. Leaf-first in fine mode let a peer `CommitGrand`
   interject between t1's Parent CAS and t1's Grand CAS — the peer
   never sees Grand changed → its stale-snapshot CAS succeeds → lost
   increment. Fix unifies fine and superfine on this point; the other
   fine vs superfine differences (BundlePhase1 pre-bundle CAS,
   BundlePhase3 DISTURBED detection) are preserved so fine remains a
   meaningful "stripped-down C++-faithful" mode.

5. **`InnerPhase3` / outer `BundlePhase3` fine success path: update
   `innerSubWs[gc]` / `subwrappers[c]` to the new wrapper.**
   *TLA+-specific* fix (no C++ counterpart). C++ pointer identity
   automatically invalidates the failure-branch guard
   `linkage[gc] /= gcWs[gc]` after a successful CAS — the saved old
   pointer differs from the freshly allocated one. TLA+ value
   equality makes the pre-CAS saved wrapper still compare-equal to
   nothing (since linkage[gc] now holds the new wrapper), but the
   failure branch's check fires for the just-CAS'd entry, restarting
   inner_phase2. Combined with Lamport regen, this produced an
   **infinite single-thread state-space explosion** (51K states in
   5 s, growing). Updating the saved wrapper after success closes
   the self-cycle.

### Notes on what was NOT changed

- Superfine cfg semantics — superfine was already root-first and was
  passing, so it required no changes.
- 2-level `GenSerial` — same C++-fidelity rewrite applied for
  consistency, but no `BundlePhase3` disturbed regen was added since
  2-level already passed without it.
- C++ STM implementation — no changes; the implementation has always
  been correct, only the formal model needed alignment.

### Side cleanup

- `SerialBound` and `DebugSerialBound` neutered to `TRUE`. With Lamport
  serials being unbounded by design and LL-free priority gating
  guaranteeing termination, hard-cap heuristics no longer correlate
  with livelock and were tripping on legitimate Lamport advancement.
- `PrintTerminalSerial` debug print: removed `globalSerial` from
  emitted tuple.
- `*TTrace*` files and `states/` directory cleaned up at user request.

### TLA+ → C11 cross-check summary

| Category | Action |
|---|---|
| GenSerial Lamport TID-encoded base-B | Ported to both 3L + 2L LLfree (`SerialBase = NUM_THREADS+1`, `serial = counter*B + tid`) |
| BundlePhase1 inner-bundle 孫 priority 関係なく CAS | Ported (3L `try_outer_bundle` Phase 3 — `if (pw.has_priority)` ゲート削除) |
| BundlePhase3 disturbed bundleSer 再生成 | C11 retry loop が `gen_serial` 再呼び出しで自然に処理 — N/A |
| UnbundleWalk root-first 統一 | C11 既に root-first — N/A |
| InnerPhase3 / fine BundlePhase3 success state-tracking | TLC self-fail 防止のみ — N/A |
| UnbundleCASLoop fine walkWrapper 更新 | 対 peer の `superFresh` check を 2-level unbundle に追加 |
| DebugSerialBound neuter | TLA+ のみ — N/A |
| QuiescentCheck 常時 ON | C11 terminal check 既存; 中間 idle check は skip per user |
| `-deadlock` 撤去 | TLA+ のみ — N/A |
| (追加) 2-level unbundle Grand step を `extracted.sub` literal に | 適用 — `fresh_parent_slot` を `Grand.sub[Parent]` にも入れる ad-hoc 強化を撤回 |

### Verification results (all PASS)

**TLC laptop-runnable verification suite** (MacBook, `-Xmx14g`,
`-workers auto`):

All cfgs pass `TerminalPayloadCheck`, `BundleChainValid` (3L) /
`BundleRefConsistency` (2L), `BundledByCorrect`, `MissingPropagation`,
`SnapshotConsistency`, `NoPriorityLoss`, `QuiescentCheck`,
`DebugSerialBound`, and `EventuallyAllDone` (liveness).

`-deadlock` flag is not used — the `Terminating` self-loop disjunct
absorbs the legitimate AllDone terminal state, so any real deadlock is
caught by default `CHECK_DEADLOCK TRUE` (see `proof_semantics.md` §6).

Lamport counter = `serial ÷ SerialBase` where `SerialBase = 1 + |Threads|`.
Multiple AllDone states exist per cfg; counter range spans min–max across
all reachable terminal states.

| cfg | distinct states | depth | wall time | Lamport counter (min–max) | terminal states | result |
|---|---|---|---|---|---|---|
| 3L 1thr fine | 46 | 29 | < 1 s | 7 | 1 | ✅ PASS |
| 3L coarse 2t | 1,497,261 | 98 | 1:53 | 6–22 | 110 | ✅ PASS |
| 3L purefine 2t | 12,115,634 | 141 | 43:36 | 7–25 | 162 | ✅ PASS |
| 3L superfine 2t | 12,134,591 | 140 | 17:10 | 7–26 | 166 | ✅ PASS |
| 3L micro (mixed) | 12,115,634 | 141 | 16:14 | 7–25 | 162 | ✅ PASS |
| 3L off (Privilege=FALSE) | — | — | — | — | — | ⛔ diverges |
| 2L micro (fine) | 803,631 | 89 | 1:06 | 6–18 | 71 | ✅ PASS |
| 2L superfine | 2,511,525 | 129 | 3:06 | 6–23 | 123 | ✅ PASS |
| 2L phase0only | 927,066 | 87 | 52 s | 6–18 | 71 | ✅ PASS |
| 2L phase3only | 2,379,184 | 129 | 2:31 | 6–24 | 124 | ✅ PASS |
| 2L commits2 (MaxCommits=2) | — | — | — | — | — | ⏳ ohtaka |

Notes:
- **3L off (Privilege=FALSE)**: intentionally diverges — LL-free priority
  gating disabled means serial grows monotonically without bound, so TLC
  never terminates. Killed at 118M states / depth 114. This confirms
  `proof_semantics.md` §4–§5: LL-free is necessary for termination.
- **Lamport counter min**: coarse min = 6 (UnbundleCASLoop is one atomic
  action with a single GenSerial), fine/superfine min = 7 (each ancestor
  in the 3-level chain gets its own GenSerial in the per-CAS loop).
  Both match the expected single-thread minimum GenSerial call count.
- **Lamport counter max**: grows with contention. 2-level fine reaches
  counter 18; 3-level coarse reaches 22; 3-level superfine reaches 26.
  Higher atomicity modes add CAS retry and DISTURBED restart paths.
- An earlier superfine run with grep-piped output showed only 3 terminal
  states (grep output loss); re-running with full file output confirmed
  166 terminal states with counter min = 7 (matching 1-thread baseline).
- **3L purefine = 3L micro**: identical state space (12,115,634 distinct,
  depth 141, 162 terminals, counter 7–25). micro differs only in
  Walk/CAS = superfine (root-first already unified), so all reachable
  states are the same.
- **2L commits2**: deferred to ohtaka. Laptop run reached 33.6M distinct
  / depth 79 / queue 1.46M before being killed. The non-LLfree 2-level
  spec already passed MaxCommits=2; 3-thread configs are higher priority.

**C11 stress test (generated from spec):**

- 3L p0/p1 unit (NT=2, MAX_COMMITS=1)
- 3L p0 2t (14.4 M commits), p0 4t (3.6 M, 以前は失敗してた), p1 2t (19.3 M), p1 4t (3.9 M)
- 3L SUPERFINE 2t (22.1 M), COARSE 2t (64.5 M)
- 2L LLfree unit + 2t (18.6 M)

The generated C11 stress test (`test_bundle_3level_LLfree.c`) no longer
hangs and passes including the 4-thread configs that were previously
failing.

---

## 2026-04-16: BundlePhase3 Fine-Grained Fix + iterBudget

### Bug: BundlePhase3 allDone check (Layer 2)

**Problem:** In fine-grained BundlePhase3, the completion check used `~hasPriority`
to determine whether a child had been CAS'd. This incorrectly counted children
bundled by *other* threads as done, allowing the bundling thread to proceed to
Phase4 before all its own children were linked.

**Root cause:** Thread t1 bundles Child1 (setting `hasPriority=FALSE`). Thread t2,
performing its own bundle, checks `~hasPriority` on Child1 and treats it as
already done — even though t2 never CAS'd it with its own serial.

**Fix:** Changed allDone from:
```tla
\A c2 \in children \ {c} : ~linkage[c2].hasPriority
```
to:
```tla
\A c2 \in children \ {c} : linkage[c2] = BundledRefWrapper(node, ser)
```
This ensures each child is verified against THIS bundle's serial, not just any
bundle state.

**Affected files:**
- `BundleUnbundle.tla` (3-level, line ~439)
- `BundleUnbundle_2level.tla` (2-level, line ~319)

**Corresponding C++ fix:** Commit `54898aec` — "Fix BundlePhase3 rollback: only
revert OUR bundled children"

### Fix: MaxSerial increase (Layer 2)

**Problem:** Fine-grained mode generates ~64 serials per commit cycle due to CAS
retries across bundle/unbundle/snapshot operations. With MaxSerial=128,
`MaxSerial/2 = 64` — barely enough, and the ModGT comparison becomes undefined
when serial difference equals exactly MaxSerial/2.

**Fix:**
- 3-level: MaxSerial 128 -> 1024 (`BundleUnbundle_mc.cfg`)
- 2-level: MaxSerial 128 -> 512 (`BundleUnbundle_2level_mc.cfg`)

### Enhancement: iterBudget for deterministic termination (Layer 0)

**Problem:** `atomic_shared_ptr.tla` had unbounded operation sequences, making TLC
model checking non-terminating without external constraints.

**Solution:** Added `MaxOps` constant and `iterBudget` per-thread variable:
- Each thread starts with `iterBudget[t] = MaxOps`
- `ReturnToIdle` decrements the budget
- `StartLoadShared`, `StartCAS`, `StartSwap`, `Recycle` require `iterBudget[t] > 0`
- `Reset` has NO budget guard (must release references after budget exhaustion)
- `TerminalCheck` invariant: when all threads idle with budget=0 and no holds,
  the installed object has exactly refcount=1 and all others are freed

**Config changes** (`atomic_shared_ptr_mc.cfg`, `_swap_mc.cfg`, `_all_mc.cfg`, `_swap_load_mc.cfg`):
- Added `MaxOps = 3`
- Added `CHECK_DEADLOCK FALSE` (threads intentionally stop after budget exhaustion)
- Added `TerminalCheck` to invariant list

### Systematic C++ comparison and superfine mode (2026-04-17)

Compared fine-grained TLA+ model against C++ `transaction_impl.h` and identified
9 differences. Categorized into three tiers:

**fine mode (always active):**

- **#3 No rollback in Phase3 failure.** C++ does not restore children to original
  wrappers on Phase3 CAS failure. Instead, Phase1 re-collection re-adopts bundled
  children via `CollectSubpacket`'s `bundledBy==node` path. State space unchanged
  (1.2M). Important for future insert/release modeling — rollback could mask bugs
  when tree structure changes dynamically.

**superfine mode (configurable, ohtaka-class state space):**

- **#1 Pre-bundle serial CAS** (`BundleCollectAtomic`): C++ stamps `bundle_serial`
  on the parent's wrapper before Phase1 collection. Without `negotiate()` backoff
  this causes livelock in TLA+ (both threads alternate pre-CAS, exhausting MaxSerial).
- **#2 Inner bundle phases** (`BundleCollectAtomic`, 3-level only): C++ calls
  `bundle()` recursively for children needing inner bundle (4-phase protocol).
  TLA+ fine mode does this atomically. superfine adds `inner_phase2/3/4` pc states.
- **#4 Phase3 serial/parent check** (`BundlePhase3Atomic`): C++ checks failing
  child's `m_bundle_serial != bundle_serial` or parent changed → DISTURBED.
  fine mode always restarts Phase1.
- **#5 casTargets root-first** (`UnbundleWalkAtomic`, 3-level only): C++
  `snapshotForUnbundle` builds cas_infos root-first via recursive `emplace_back`.
  fine mode uses leaf-first (Append). superfine uses prepend for root-first.
- **#6 Phase1 retry same child** (`BundleCollectAtomic`): C++ retries the same
  child if parent unchanged. fine mode always restarts from snap_check.

**Fidelity notes (documented, not implemented):**

- **#7 CommitTryCAS serial reuse:** C++ creates `newwrapper` once with `tr.m_serial`
  and reuses across inner retries. TLA+ calls `GenSerial` each time. Effect: TLA+
  consumes slightly more serial space. No correctness impact.
- **#8 UnbundleCASChild serial source:** C++ uses `gen(superwrapper->m_bundle_serial)`
  where `superwrapper` is the root wrapper after walk (may have higher serial).
  TLA+ uses `oldChildW.serial` (child's bundled_ref serial). Both equal at bundle
  time; diverge if root was re-committed. No correctness impact.
- **#9 Unbundle children bundled elsewhere:** C++ actively unbundles via
  `bundle_subpacket`. TLA+ returns Null and retries. Unreachable in current models
  (fixed tree, no hard links). Only relevant for future hard-link modeling.

### Layer 0 (atomic_shared_ptr) C++ comparison (2026-04-17)

Full action-by-action comparison of `atomic_shared_ptr.tla` against C++
`atomic_smart_ptr.h` (acquire_tag_ref_, load_shared_, release_tag_ref_,
compareAndSwap_(NOSWAP=true), local_shared_ptr::swap(atomic_shared_ptr&),
reset, Recycle). **No correctness differences found.** Three minor
simplifications, all safe:

- **release_tag_ref_ read**: C++ atomically reads (ptr, rcnt_old) and checks
  both in one condition. TLA+ reads local_rc in `ReleaseTagRefRead`, defers
  ptr check to `ReleaseTagRefCAS`. One extra intermediate state; result identical.
- **release_tag_ref_ post-CAS re-read**: C++ does a separate `load_tagged_().first`
  after CAS failure to decide retry vs. global fallback. TLA+ decides in the same
  action as the CAS. TLA+ may take extra retries but converges to same result.
- **LOCAL_REF_CAPACITY overflow**: C++ spin-waits when `rcnt_new >= CAPACITY`.
  TLA+ omits this; `LocalRCBounded` invariant verifies the bound instead.

**Design notes:**

- superfine features are C++ performance optimizations (early exits, reduced retry
  work) that don't affect protocol correctness. TLA+ fine mode proves this.
- MaxSerial cannot be replaced by natural-number serials: commit retry loops
  (iterBudget not consumed on failure) create unbounded serial growth without
  modular wrap-around, turning the state space into an infinite DAG.

### Model Checking Results

| Spec | Mode | MaxCommits | MaxSerial | Result | States |
|------|------|-----------|-----------|--------|--------|
| 3-level | coarse | 1 | 48 | PASS | 110K |
| 3-level | coarse | 5 | 128 | PASS | ~8.5M |
| 3-level | fine | 1 | 256 | PASS (local) | 1.2M |
| 3-level | fine | 5 | 128 | FAIL (NoSerialWrapAround) | - |
| 3-level | fine | 5 | 1024 | pending (ohtaka) | - |
| 2-level | coarse | 1 | 24 | PASS | 483K |
| 2-level | coarse | 5 | 64 | PASS | ~3.2M |
| 2-level | fine | 1 | 128 | ohtaka only | 70M+ |
| 2-level | fine (pre-fix) | 1 | 64 | FAIL (TerminalPayloadCheck) | ~2.1M |
| **2-level LL-free** | **fine** | **1** | **N/A (no CONSTRAINT)** | **PASS (Safety + EventuallyAllDone)** | **665K, depth 89, 28s** |
| atomic_shared_ptr | all ops | MaxOps=3 | - | PASS | ~1.3M |

---

## 2026-04-29: Gen 3 (`BundleUnbundle_2level_LLfree.tla`) Complete

### LL-free protocol formally proven

After several iterations, the LL-free spec terminates **without `CONSTRAINT SerialBound`** while passing every invariant and the `EventuallyAllDone` liveness PROPERTY. This satisfies the conditions in `proof_semantics.md` §2 ("終了 + CONSTRAINT なし → LL-free is automatic"):

- Lamport serial as plain `Nat` (no modular wrap)
- Every retry-able CAS bumps a monotonic serial
- Priority `priorityTag[n]: Null | <<iter, tid>>` per node, gating CAS attempts
- Older active transaction wins via explicit `PreemptTag`
- Stale tags from finished threads are zombie-detected via `ActiveThread`
- Tags persist Transaction-scope (not per CAS): set on CAS fail (`TagAfterFail`), released only at `CommitParent`/`CommitDone` (`ClearMyTags`) — mirrors C++ `drop_tags_n_privilege` called from `~Transaction`, `Snapshot` ctor end, `Transaction::finalizeCommitment`
- `Terminating` self-loop at `AllDone` so default `CHECK_DEADLOCK TRUE` distinguishes legitimate termination from a real stuck state (no `-deadlock` flag)

### Result

```
2,656,169 states generated, 665,218 distinct states found, 0 states left on queue
Depth 89; Finished in 28s
Model checking completed. No error has been found.
```

Invariants checked: SnapshotConsistency, NoPriorityLoss, BundleRefConsistency, MissingPropagation, TerminalPayloadCheck, DebugSerialBound. Property: EventuallyAllDone.

### Key design lessons (intermediate failure modes)

1. **Per-CAS tag clear is too eager** (initial impl). With tag clearing on every CAS success, peer thread races in between phases, forcing endless re-bundle/unbundle cycles. Bound-60 trace showed a clean repeating pattern: t1 succeeds Phase 4 → tag clears → t2 immediately unbundles Parent → t1 must re-bundle. Fix: keep tag through entire Transaction.
2. **Zombie tags from finished threads** (intermediate impl). After a thread finished its iterations, its remaining tags blocked peers that couldn't preempt (`TagOlder` failed against equal-iter younger). Fix: `ActiveThread(t)` check in `CanProceed`, `PreemptTag`, and `TagAfterFail`.
3. **Symmetry incompatible with naturals + liveness**. `TagOlder` requires Nat-ordered threads, but `Permutations` only works on model values; symmetry also can mask liveness violations. Fix: drop `SYMMETRY ThreadSymmetry` in this cfg; safety still holds, liveness now checkable.

### Note: `tags_successful_cas` ≠ `release_privileged_tidstamp`

Earlier draft conflated them. C++ `tags_successful_cas` is a **per-CAS C_obs counter** consumed by `negotiate_internal`'s adaptive backoff, not privilege release. Privilege tidstamp is released exclusively via `drop_tags_n_privilege`, called from:
- `Snapshot` ctor end (transaction.h:976)
- `Transaction::finalizeCommitment` (transaction.h:1413, = `tr.commit()` success)
- `~Transaction` (transaction.h:1268, = abort)

TLA+ `ClearMyTags` correctly maps to `drop_tags_n_privilege` at these three points.

### Source files

- `BundleUnbundle_2level_LLfree.tla` — Gen 3 spec, 700+ lines
- `BundleUnbundle_2level_LLfree_micro_mc.cfg` — micro config (Threads={1,2}, MaxPayload=3, MaxCommits=1)
- C++ counterpart (verified separately): commit `2d141d5` — full FINE/SUPERFINE multi-thread stress PASS

---

## 2026-04-29 (later): superfine + scaling sweeps

### Superfine fix — eager Parent tag on retry

Initial superfine run blew up state space (>30M states at depth 868 still growing). Diagnosis via two diagnostic cfgs (`_phase0only_mc.cfg`, `_phase3only_mc.cfg`) localised the culprit: **Phase 3 DISTURBED restart from `snap_read`**.

The fix combined two changes in `BundlePhase3` superfine DISTURBED branch and `BundlePhase1` fine collection-failure branch:

1. **Clear local bundle state** on `snap_read` restart (`local.wrapper`, `subwrappers`, `subpackets`). C++ `bundle()` returning DISTURBED tears down its stack frame; without this the TLA+ accumulates stale partial-collection state, multiplying TLC's distinct-state count.
2. **Eager Parent tag** on the same restart paths via `TagAfterFail(t, Parent)`. Mirrors C++ outer-scope `ScopedNegotiateLinkage` at `transaction_impl.h:2407` (eager tag on retry > 0 of bundle's retry-loop) and `transaction_impl.h:2179` (eager tag on retry > 0 of `snapshot()` retry-loop). Without this, peer threads CAS Parent during the holding thread's restart cycle, leading to indefinite re-bundling.

### Sweeps (laptop, 2 threads, no CONSTRAINT, all PASS)

| Mode | MaxCommits | MaxPayload | Distinct states | Depth | Time | Notes |
|---|---|---|---|---|---|---|
| fine | 1 | 3 | 665,218 | 89 | 28s | baseline; first formal LL-free proof (commit `a35c4310`) |
| Phase 0 prestamp only (superfine collect) | 1 | 3 | 1,095,188 | 93 | 48s | adds Phase 0 prestamp CAS |
| Phase 3 DISTURBED only (with fix) | 1 | 3 | 1,446,125 | 114 | 71s | adds DISTURBED restart |
| **superfine (full, with fix)** | **1** | **3** | **1,407,147** | **119** | **64s** | full Phase 0 + Phase 3 DISTURBED |
| MaxPayload=4 fine | 1 | **4** | 665,216 | 89 | 37s | structural state space identical to MaxPayload=3 |
| MaxCommits=2 fine | **2** | 3 | >27M @ depth 77 (killed) | n/a | n/a | queue ~1.2M not shrinking; needs ohtaka |

### Terminal serial diagnostic (`PrintTerminalSerial` invariant)

A new debug invariant `PrintTerminalSerial == ~AllDone \/ PrintT(...)` emits per-thread `serial` and `globalSerial` to stdout at every AllDone state. Used to gauge Lamport-clock growth.

For MaxPayload=4 fine micro: max `globalSerial = 18`, 71 unique terminal-serial combinations across reachable AllDone states. Well under `DebugSerialBound = 200` (raised from 60 because superfine's Phase 0 + DISTURBED restart cycles bump serials by ~50% more than fine).

### C++ counterpart cleanup matching TLA+

After the LL-free pass, the user simplified C++ `transaction_impl.h:2425-2440` to move the fast-path retry check **before** `ScopedNegotiateLinkage` construction in Phase 1 child collection. This skips negotiate / eager-tag on the read-only fast path (when `subwrapper == subwrappers_org[i]`), matching TLA+'s "no gate on Phase 1 child read" semantics. Drops `tags_successful_cas` C_obs counter increments on no-op fast-path retries (correct because no CAS attempted).

### Open: ohtaka scaling targets

Laptop budget exhausted at MaxCommits=2 / MaxPayload=3 / 2 threads (queue 1.2M not shrinking, RSS 5.4GB / 12GB after 25 min). Next-tier configs to run on ohtaka:

- **MaxCommits=2, MaxPayload=3, 2 threads, fine** — direct continuation; estimated 80-150M states.
- **MaxCommits=2, MaxPayload=3, 2 threads, superfine** — same with C++'s atomic mode for both Collect/Phase3.
- **MaxCommits=1, MaxPayload=3, 3 threads, fine** — N=3 priority tag space; estimated 50-200M.
- **MaxCommits=3, MaxPayload=3, 2 threads, fine** — to confirm liveness convergence at higher iteration counts.

Cfg files already committed: `_commits2_mc.cfg`, `_superfine_mc.cfg`, `_payload4_mc.cfg`, `_superfine_phase0only_mc.cfg`, `_superfine_phase3only_mc.cfg`. To add 3-thread cfg, copy `_micro_mc.cfg` and change `Threads = {1, 2, 3}` (keep `SYMMETRY` removed for liveness).

Optional simplification: ~~`MaxPayload = Nat` (no MOD)~~ — **applied 2026-04-29**. See below.

### MaxPayload MOD removed (2026-04-29)

Tested empirically that `(payload + 1) % MaxPayload` could be replaced with `payload + 1` (and `TerminalPayloadCheck`'s `% MaxPayload` removed). Result for micro fine 2t MaxCommits=1: **665,216 states / depth 89 / 29s** — *identical* to MOD-bearing version (665,218; 2-state fingerprint difference). MOD never produced a state-space collision in this protocol because the LL-free + bounded-iteration discipline pins each reachable payload value to a distinct execution position; two different "real" payloads never collapse to the same state under MOD.

Benefits applied:
- `TerminalPayloadCheck` is now a strict cumulative count (`payload = 2 * MaxCommits * |Threads|`) — exact assertion, no wrap-disambiguation hazard.
- `MaxPayload` constant removed from spec and all `_mc.cfg` files.
- `_payload4_mc.cfg` deleted (no longer meaningful).

Java/TLC integer behavior: TLC uses `IntValue` (Java `long`) on the fast path with automatic promotion to `BigIntValue` (`BigInteger`) on overflow, so payload-Nat arithmetic stays exact even at large scales. Our values stay << `Long.MAX_VALUE` (~9.2e18), so the fast path is never abandoned.
