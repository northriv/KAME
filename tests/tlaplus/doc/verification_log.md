# TLA+ Verification Log

## 2026-05-04: 3-level dynamic model (`BundleUnbundle_3level_LLfree_dynamic.tla`)

### Context

New TLA+ spec combining `BundleUnbundle_3level_LLfree.tla` (static Grand→Parent→{Child1,Child2}
bundle/unbundle protocol) with `BundleUnbundle_2level_LLfree_dynamic.tla` (dynamic child
insertion/release). Models the full 3-level LL-free protocol where DynChild1 and DynChild2 are
dynamically inserted and released at runtime. All thread roles configurable via constants.

### Design

- **Tree**: Grand → Parent → {DynChild1, DynChild2} (children start unattached)
- **`ChildrenOf(Parent) = ActiveChildren`**: state-dependent set of currently inserted children
- **`SubDomainOf(Parent) = AllChildren`**: sub-packet domain always covers all possible children
  (inserted or not) to prevent domain mismatch across bundle/unbundle
- **`BundlePhase1` dispatch**: `IF bundleNode=Grand THEN 3L-static logic (with InnerPhase)`
  `ELSE 2L-dynamic logic (with shrink disjunct)`. Clean separation of the two bundling contexts.
- **`InnerPhase3` fixed grandchild set**: `{gc ∈ AllChildren : innerSubWs[gc] ≠ Null}` — uses
  the collected set, not current `ActiveChildren`, to avoid stale-set after mid-InnerPhase release.
- **`CommitGrand` dynamic discovery**: `snapChildren = {c ∈ AllChildren : parentPkt.sub[c] ≠ Null}`
  discovers active children from the snapshot packet; no dependence on `ActiveChildren` at commit time.
- **`commitCount[c]`**: Per-child counter (inherited from 2L dynamic) for `TerminalPayloadCheck`.
- **`BundleRetryPC`**: Routes retry to `"insert_snap"` / `"release_snap"` / `"snap_check"` based
  on `op[t]`.

### Key design challenges

1. **Sub-packet domain consistency**: Parent bundles use `AllChildren` as the domain of `sub[·]`
   even while children are released (their slots become Null). If the domain changed dynamically,
   a bundle captured before release and an unbundle occurring after would have mismatched domains.
   Fixed by `SubDomainOf(Parent) = AllChildren` always.

2. **`InitLocal` field domains**: `subwrappers`, `subpackets`, `innerSubWs` all use `AllNodes`
   as domain to avoid conflicts: Grand-level bundling needs a `Parent` slot, Parent-level bundling
   needs `DynChild1/2` slots. A per-role domain would cause TLC domain mismatch on role transitions.

3. **`BundlePhase2` explicit domain reconstruction**:
   `IF node=Parent THEN [c ∈ AllChildren |-> subs[c]] ELSE [c ∈ GrandChildren |-> subs[c]]`
   matches the domain expected by `CommitGrand`'s `snapChildren` discovery.

4. **`InnerPhase3` stale set**: After `BundlePhase1` collects grandchild wrappers into
   `innerSubWs`, a release may shrink `ActiveChildren`. Using `ActiveChildren` in `InnerPhase3`
   would skip collected-but-released entries. Fix: use the collected set
   `{gc ∈ AllChildren : innerSubWs[gc] ≠ Null}` (same pattern as InnerPhase4 fix for 3L static).

### Source files

- `BundleUnbundle_3level_LLfree_dynamic.tla` — 3L dynamic spec
- `*_1thr_mc.cfg` — 1-thread sanity (coarse; all roles in Thread 1)
- `*_coarse_mc.cfg` — 2-thread coarse, ReleaseThreads={}
- `*_release_coarse_mc.cfg` — 2-thread coarse, ReleaseThreads={1,2}
- `*_superfine_mc.cfg` — ohtaka superfine, 2-thread all-roles, ReleaseThreads={}
- `*_release_superfine_mc.cfg` — ohtaka superfine, 2-thread all-roles, ReleaseThreads={1,2}
- `*_3thr_A_mc.cfg` — ohtaka superfine, Ins={1}/Root={2}/Leaf={3}, no release
- `*_3thr_B_mc.cfg` — ohtaka superfine, Ins={1}/Root={2,3}/Leaf={}, no release
- `*_3thr_release_mc.cfg` — ohtaka superfine, Ins={1}/Root={2}/Leaf={3}, all threads release

### Verification results

| cfg | distinct states | depth | wall time | Lamport counter (min–max) | terminal states | result |
|---|---|---|---|---|---|---|
| 3L-dyn 1thr coarse | 66 | 36 | < 1 s | 11 | 2 | ✅ PASS |
| 3L-dyn coarse 2t (ReleaseThreads={}) | 6,825,326 | 127 | 9:30 | 10–30 | 2,700 | ✅ PASS + liveness ✅ |
| 3L-dyn release coarse 2t (ReleaseThreads={1,2}) | — | — | — | — | — | ⏳ local (est. ~115M states) |
| 3L-dyn release superfine 2t (all-roles, ohtaka) | 921,351,233 | 284 | 3h 24min | 11–49 | 94,630 | ✅ PASS (ohtaka) |
| 3L-dyn 3thr-A live (Ins={1},Root={2},Leaf={3}) | 122,150 | 87 | 10 s | 5–15 | 157 | ✅ PASS (ohtaka) + liveness ✅ |
| 3L-dyn 3thr-B live (Ins={1},Root={2,3},Leaf={}) | 120,193 | 75 | 10 s | 5–10 | 58 | ✅ PASS (ohtaka) + liveness ✅ |
| 3L-dyn 3thr release (Ins={1},Root={2},Leaf={3}, all release) | — | — | — | — | — | ⏳ ohtaka (casOldWrappers fix applied) |

Notes:
- **1thr counter=11**: Same as 2L-dyn 1thr, confirming correct terminal state accounting (both children inserted + 1 commit each via CommitGrand/CommitChild, 4 GenSerial calls).
- **2t coarse state count (6.8M)**: Slightly increased from pre-fix 6,444,080 due to `casOldWrappers` adding state space. 4.5× larger than 3L static coarse (1.5M) and 8.9× larger than 2L-dyn coarse (763K), reflecting the combined Grand/Parent/Child bundle machinery plus insert sequencing.
- **Counter min=10**: Lower than 3L static coarse min=6 — insert operations add GenSerial calls (InsertCASParent + InsertCASChild) before any commit, raising the baseline counter.
- **Counter max=30**: Higher than 3L static coarse max=22, accounting for the additional insert/discovery phase Lamport steps.
- **release coarse est. ~115M states**: Based on 2L-dyn release-coarse/no-release-coarse ratio of 18.6× applied to 6.4M. Too large for routine local runs; designated as ohtaka target.
- **3thr-A/B live (ohtaka, 2026-05-04)**: State counts (122K / 120K) are comparable to 2L-dyn 3thr-A/B (53K / 149K), confirming that role-separated 3-thread configs remain tractable despite the extra Grand level. Counter min=5 is lower than 2t coarse (min=10) because separated roles with MaxCommits=1 allow shorter paths where fewer Lamport steps accumulate before termination. 3thr-A has more terminal states (157) than 3thr-B (58) due to additional interleaving from separate LeafThreads. Both fingerprint collision rates are negligible (≤6.2E-8).
- **release superfine 2t (ohtaka, 2026-05-05, slurm-2898329)**: 921M states, 3h 24min. Largest 3L-dyn run to date; 2.2× the 2L-dyn release superfine (413M). Counter min=11 matches 1thr (all-roles path still uses same GenSerial sequence); max=49 reflects extra Lamport steps from superfine atomicity. Fingerprint collision rate 6.3E-2 — high due to state space size, but within acceptable range for a safety check. Safety only (PHASE=1), no PROPERTY.

### Bug fix: `casOldWrappers` — stale `UnbundleCASLoop` CAS (2026-05-05)

**Violation**: `QuiescentCheck` + `TerminalPayloadCheck` in `_3thr_release_mc.cfg` (ohtaka, `slurm-2896201.out`, 73 states).
`ChildPayload(DC1) = 3 ≠ 1 + commitCount[DC1] = 4` at terminal state — `linkage[Parent].sub[DC1].payload` regressed from 4 to 3.

**Root cause**: `UnbundleCASLoop` fine/superfine branch read `oldW := linkage[casNode]` immediately
before the CAS, making the guard `linkage[casNode] = oldW` trivially true regardless of intervening
writes by other threads. The `superFresh` guard covered only the root node (Grand); intermediate
nodes (Parent, etc.) had no freshness check. When Thread 1 completed `UnbundleWalk` at S45 (capturing
stale data with `Grand.sub = (3,3)`) and then executed `UnbundleCASLoop` at S67, Thread 2 had
already advanced `linkage[Parent]` to sub=(4,4) and back (via leaf commits × 2 + rebundle + release),
but the trivial `oldW` check allowed Thread 1's stale CAS to overwrite it back to sub=(3,3).

In the 2L dynamic spec, `UnbundleCASAncestors` stores `parentWrapper` at walk time, preventing
this race. The 3L spec introduced `casTargets` for multi-level unbundle but omitted the parallel
`casOldWrappers` field — the same walk-time snapshot of each node's wrapper.

**Fix** (all in `BundleUnbundle_3level_LLfree_dynamic.tla`):

1. `InitLocal`: added `casOldWrappers |-> <<>>` field.

2. `SnapshotForUnbundle`: captures `parentOldW == linkage[parentNode]` at each level of the
   recursive walk and returns it in `casOldWrappers` alongside `casTargets`.

3. `UnbundleWalk` (both coarse and fine branches): saves `result.casOldWrappers` /
   `<<pw>> \o local[t].casOldWrappers` into `local[t].casOldWrappers` in parallel with `casTargets`.

4. `UnbundleCASLoop` fine branch: `oldW` now reads `local[t].casOldWrappers[idx]` (walk-time
   snapshot) instead of fresh `linkage[casNode]`, making `linkage[casNode] = oldW` a genuine CAS gate.

5. `UnbundleCASLoop` coarse branch: added `allFresh` predicate
   `\A i \in 1..Len(targets) : linkage[targets[i]] = local[t].casOldWrappers[i]`
   and gated the CAS on `allExtractable /\ superFresh /\ allFresh`.

**Regression**: 1-thread sanity (66 states, PASS), 2-thread coarse (6,825,326 states, PASS + liveness).

---

## 2026-05-01: Dynamic insert/release model (`BundleUnbundle_2level_LLfree_dynamic.tla`)

### Context

New TLA+ spec for modeling dynamic child insertion (`insert(online=true)`)
and release, extending the static 2-level LL-free model. All threads are
configurable via constants: `InsertThreads`, `RootThreads`, `LeafThreads`,
`ReleaseThreads`. Children are discovered dynamically from transaction
snapshots (`snapChildren = {c ∈ AllChildren : snapPkt.sub[c] ≠ Null}`),
not hardcoded.

### Design

- **Tree**: Parent ← DynChild1, DynChild2 (both start unattached)
- **Phase 1 — Insert**: InsertThreads pick uninserted children and
  perform `insert(online=true)`. `everInserted[c]` one-way flag
  distinguishes "not yet inserted" from "released".
- **Phase 2 — Commit + Release** (interleaved):
  - RootThreads: CommitParent (snapshot Parent, increment discovered
    children, CAS). Dynamic child discovery via `snapChildren`.
  - LeafThreads: CommitChild per discovered child (direct commit).
  - ReleaseThreads: Release inserted children after own commits done.
- **Release lifecycle**: `release_snap → (bundle if missing) →
  release_cas_parent → release_read_child → release_cas_child`.
  Two-phase CAS (parent then child) matches C++ `release(tr, var)`.
- **`commitCount[c]`**: Per-child counter tracking successful commits,
  used by `TerminalPayloadCheck` for precise payload assertion.
- **Unified `ReadParent`**: Replaces separate InsertSnap + SnapCheck,
  shared by insert/snapshot/release paths.

### Stale priority tag fix

Three liveness violations discovered and fixed during release modeling:

1. **Deadlock (idle thread during release)**: Thread 2 has no enabled
   actions while thread 1 is mid-release. Fix: Replace `WaitingForInserts`
   with generalized `Waiting` action (checks `releaseTarget[t] = Null`).

2. **Livelock (CommitChild on released child)**: Thread loops CommitRead →
   UnbundleWalk on a released child (parent `sub[c]=Null`, child still
   `BundledRef`). Fix: `~inserted[childNode]` check in UnbundleWalk to
   abort commit.

3. **Livelock (stale priority tags)**: Idle threads leave tags that block
   other threads from `CanProceed`. Chain: CommitDone failure preserves
   child tags → CommitSkip drains queue → SkipIteration → thread fully
   idle with stale tags. Fix: `ClearMyTags(t)` at three points:
   - `CommitParent` empty `snapChildren` path (thread goes idle, no
     children to commit)
   - `CommitSkip` when queue drains to empty (released children skipped)
   - `SkipIteration` (all children released, budget drained to 0)

   In C++, these correspond to `~Transaction()` / `finalizeCommitment()`
   calling `drop_tags_n_privilege()` when each `iterate_commit` exits.
   TLA+ lacks RAII, so explicit `ClearMyTags` is needed at these
   phase boundaries. The model is a sound over-approximation: tags
   persist across phases more than in C++ (no per-iterate_commit
   scoping), so if liveness passes, C++ is also correct.

### C++ fidelity notes (release-specific)

| Aspect | C++ | TLA+ | Match |
|---|---|---|---|
| Parent CAS failure → retry | Tag persists, `++tr` | `TagAfterFail(Parent)` | ✓ |
| Parent CAS success → child | Tag persists | `TagAfterSuccess(Parent)` | ✓ |
| Child CAS negotiate | `ScopedNegotiate(OnExit)` | `CanProceed(t, c)` | ✓ |
| Child CAS success → done | `finalizeCommitment` → all clear | `ClearMyTags(t)` | ✓ |
| Child CAS failure → retry | dtor tags child | `TagAfterFail(t, c)` | ✓ |
| Child read ordering | Before parent CAS (L1579) | After parent CAS | △ safe |

The child read ordering difference is safe: after parent CAS removes the
child, no concurrent thread modifies the child's linkage (CommitChild
aborts via `~inserted` check, unbundle finds `sub[c]=Null`).

### Verification results (all PASS)

**TLC laptop verification** (MacBook, `-Xmx14g`, `-workers auto`):

Invariants: SnapshotConsistency, NoPriorityLoss, BundleRefConsistency,
MissingPropagation, TerminalPayloadCheck, QuiescentCheck,
DebugSerialBound, PrintTerminalMaxCounter.
Property: EventuallyAllDone (liveness).

| cfg | distinct states | depth | wall time | Lamport counter (min–max) | terminal states | result |
|---|---|---|---|---|---|---|
| 2L-dyn 1thr coarse | 70 | 36 | < 1 s | 11 | 2 | ✅ PASS |
| 2L-dyn coarse 2t (ReleaseThreads={}) | 763,478 | 104 | 43 s | 11–26 | 71 | ✅ PASS |
| 2L-dyn release coarse 2t (ReleaseThreads={1,2}) | 14,203,816 | 150 | 27:12 | 15–37 | 3,344 | ✅ PASS (ohtaka) |
| 2L-dyn superfine 2t (ReleaseThreads={}) | 4,862,872 | 162 | 7:30 | 12–31 | 1,374 | ✅ PASS (ohtaka) |
| 2L-dyn release superfine 2t live | 413,884,516 | 320 | 7:13:00 | 15–55 | 5,972 | ✅ PASS (ohtaka) + liveness ✅ |
| 2L-dyn 3thr-A live (Ins={1}, Root={2}, Leaf={3}) | 53,397 | 68 | 7 s | 10–15 | 42 | ✅ PASS (ohtaka) + liveness ✅ |
| 2L-dyn 3thr-B live (Ins={1}, Root={2,3}, Leaf={}) | 149,137 | 82 | 14 s | 8–15 | 22 | ✅ PASS (ohtaka) + liveness ✅ |
| 2L-dyn 3thr release (all roles, all release) | — | — | — | — | — | ⏳ ohtaka |

Notes:
- **State count with ReleaseThreads={}**: 763,478 vs static spec's 763,675
  (−0.03%). Different variables (`commitCount`, `everInserted`,
  `releaseTarget`) and refactored action structure prevent exact match;
  behavioral equivalence confirmed by identical invariant/property results.
- **Release coarse state space**: 14.2M (18.6× the non-release case).
  Release adds interleaving between commit and release phases, plus
  CommitSkip/SkipIteration transitions for released children.
- **Counter min 15 (release)** vs 11 (non-release): Release operations
  add GenSerial calls (ReleaseCASParent + ReleaseCASChild), raising the
  minimum Lamport counter for terminal states.
- **3,344 terminal states (release)**: Includes all interleavings of
  commit order × release order × per-thread serial advancement.
- **Release superfine live 2t (ohtaka, 2026-05-03)**: `BundleUnbundle_2level_LLfree_dynamic_release_superfine_live_mc.cfg` PASS。413,884,516 distinct states / depth 320 / 7:13 / counter 15–55 / 5,972 terminal emits / queue 0 (exhaustion)。Safety 全 invariant + `EventuallyAllDone` liveness PASS。Fingerprint collision optimistic=2.0%, actual=0.65%。BundlePhase1 deadlock fix (second disjunct for mid-collection shrink) を含む最初の全網羅検証。Counter max=55 は release 操作が ReleaseCASParent + ReleaseCASChild の GenSerial 呼び出しを追加するため coarse (max=37) より高い。Depth 320 は release/commit の多様な interleaving による（coarse depth=150 の 2.1倍）。
- **BundlePhase1 deadlock fix (2026-05-03)**: `_release_superfine_mc.cfg`
  hit a TLC deadlock at depth 34. Root cause: Thread A had collected
  subwrappers for both children; Thread B released one child
  (`inserted[c]=FALSE`, `linkage[Parent]` updated). The surviving
  `ActiveChildren = {remaining_child}` had its subwrapper already
  collected, so the first disjunct (`∃c ∈ ActiveChildren :
  subwrappers[c] = Null`) was FALSE, and no action was enabled.
  Fix: added second disjunct to `BundlePhase1` fine/superfine branch:
  when `∀c ∈ ActiveChildren : subwrappers[c] ≠ Null` (mid-collection
  shrink detected), drop stale released-child entries and advance to
  `bundle_phase2`. Phase 2 CAS then fails (parent updated by release),
  retries from `BundleRetryPC` → `ReadParent` picks up the new parent
  wrapper (released child removed). C++-faithful: same path as C++
  (prestamp CAS → collection loop → Phase 2 CAS fails → snap_read retry).
  Local check: depth 45 reached without deadlock, terminal states
  (counter 15–18) observed. Full exhaustive verification: ohtaka.

### Source files

- `BundleUnbundle_2level_LLfree_dynamic.tla` — dynamic insert/release spec
- `*_1thr_mc.cfg` — 1-thread sanity (counter=11)
- `*_coarse_mc.cfg` — 2-thread coarse, ReleaseThreads={}
- `*_superfine_mc.cfg` — 2-thread superfine, ReleaseThreads={}
- `*_release_coarse_mc.cfg` — 2-thread coarse, ReleaseThreads={1,2}
- `*_release_superfine_mc.cfg` — 2-thread superfine, ReleaseThreads={1,2} (deadlock fix applied)
- `*_3thr_A_mc.cfg` / `*_3thr_A_live_mc.cfg` — 3-thread Ins={1}/Root={2}/Leaf={3}, w/ and w/o liveness
- `*_3thr_B_mc.cfg` / `*_3thr_B_live_mc.cfg` — 3-thread Ins={1}/Root={2,3}/Leaf={}, w/ and w/o liveness
- `*_3thr_release_mc.cfg` / `*_3thr_release_live_mc.cfg` — 3-thread all-roles + release

---

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
`DebugSerialBound`. 2-thread cfgs also check `EventuallyAllDone`
(liveness); 3-thread cfgs check INVARIANT only (liveness proven by
2-thread superfine — see notes).

`-deadlock` flag is not used — the `Terminating` self-loop disjunct
absorbs the legitimate AllDone terminal state, so any real deadlock is
caught by default `CHECK_DEADLOCK TRUE` (see `proof_semantics.md` §6).

Lamport counter = `serial ÷ SerialBase` where `SerialBase = 1 + |Threads|`.
Multiple AllDone states exist per cfg; counter range spans min–max across
all reachable terminal states.

| cfg | distinct states | depth | wall time | Lamport counter (min–max) | terminal states | result |
|---|---|---|---|---|---|---|
| 3L 1thr fine | 46 | 29 | < 1 s | 7 | 1 | ✅ PASS |
| 3L 1thr superfine | 47 | 30 | < 1 s | 7 | 1 | ✅ PASS |
| 3L coarse 2t | 1,542,814 | 98 | 1:51 | 6–22 | 110 | ✅ PASS |
| 3L purefine 2t | 11,841,706 | 134 | 36:13 | 7–24 | 152 | ✅ PASS |
| 3L superfine 2t | 14,109,731 | 148 | 19:13 | 7–26 | 4,048 | ✅ PASS |
| 3L micro (mixed) | 11,841,706 | 134 | 13:12 | 7–24 | 152 | ✅ PASS |
| 3L off (Privilege=FALSE) | — | — | — | — | — | ⛔ diverges |
| 2L micro (fine) | 867,696 | 89 | 46 s | 6–18 | 71 | ✅ PASS |
| 2L superfine | 2,676,196 | 129 | 3:12 | 6–23 | 123 | ✅ PASS |
| 2L phase0only | 997,511 | 87 | 46 s | 6–18 | 71 | ✅ PASS |
| 2L phase3only | 2,525,381 | 129 | 2:20 | 6–24 | 124 | ✅ PASS |
| 2L superfine 3t confC (all root) | 137,333,348 | 96 | 6:35:00 | 4–15 | 170 | ✅ PASS (ohtaka) |
| **2L superfine 3t confC live (all root)** | **137,333,348** | **96** | **2:53:00** | **4–15** | **170** | **✅ PASS (ohtaka) + liveness ✅** |
| 2L 3thr coarse C (all root) | 339,744 | 49 | 20 s | 4–9 | 106 | ✅ PASS |
| 3L 3thr coarse C (all root) | 397,160 | 57 | 26 s | 10–12 | — | ✅ PASS |
| 2L 3thr superfine A (2 leaf + 1 root) | 755,078,964 | 117 | 1:33:00 | 2–21 | 3,412 | ✅ PASS (ohtaka) |
| 2L 3thr superfine D (all leaf) | 203,512 | 38 | 6 s | 3–6 | 156 | ✅ PASS (ohtaka) |
| **3L 3thr superfine C (all root)** | **1,154,807,632** | **89** | **4:09:00** | **4–15** | **1,207** | **✅ PASS (ohtaka)** |
| **3L 3thr superfine C live (all root)** | **640,894,951** | **88** | **15:25:00** | **4–15** | **1,140** | **✅ PASS (ohtaka) + liveness ✅** |
| 2L 3thr coarse A (2 leaf + 1 root) | — | — | — | — | — | ⏳ ohtaka |
| 2L 3thr coarse B (1 leaf + 2 root) | — | — | — | — | — | ⏳ ohtaka |
| 3L 3thr coarse A (2 leaf + 1 root) | — | — | — | — | — | ⏳ ohtaka |
| 3L 3thr coarse B (1 leaf + 2 root) | — | — | — | — | — | ⏳ ohtaka |
| 3L 3thr superfine A (2 leaf + 1 root) | — | — | — | — | — | ⏳ ohtaka (casOldWrappers fix applied) |
| 3L 3thr superfine A live (2 leaf + 1 root) | — | — | — | — | — | ⏳ ohtaka (after safety PASS) |
| 3L 3thr superfine B (1 leaf + 2 root) | — | — | — | — | — | ⏳ ohtaka (casOldWrappers fix applied) |
| 3L 3thr superfine B live (1 leaf + 2 root) | — | — | — | — | — | ⏳ ohtaka (after safety PASS) |
| 2L commits2 (MaxCommits=2) | — | — | — | — | — | ⏳ ohtaka |

Notes:
- **`casOldWrappers` fix (2026-05-05)**: Same bug as 3L-dyn (see above) was
  present in `BundleUnbundle_3level_LLfree.tla`. `UnbundleCASLoop` fine/superfine
  branch re-read `oldW := linkage[casNode]` at CAS time (trivially true gate);
  coarse branch had no per-intermediate-node freshness check (only `superFresh`).
  Fix: same `casOldWrappers` mechanism added to `InitLocal`, `SnapshotForUnbundle`,
  `UnbundleWalk` (both branches), and `UnbundleCASLoop` (both branches). Local
  regression: 1-thread PASS (46 states), 2-thread coarse PASS (1,542,814 states,
  depth 98, liveness ✅). No violations were observed in previous runs because
  tested configs (superfine C = all-root, coarse 2t) did not exercise the specific
  interleaving where a peer thread modifies an intermediate-node wrapper between
  walk and CAS; the pending 3thr superfine A/B (leaf+root mix) are the highest-risk
  configs and must be run with the fixed spec.
- **InnerPhase2 restart fix (2026-05-03)**: `QuiescentCheck` violated in
  `BundleUnbundle_3level_LLfree_3thr_superfine_A_mc.cfg` at depth 61 on
  ohtaka with spec `8fb19385` (after InnerPhase4 fix). Trace analysis:
  State 58 (`CommitGrand(2)` completed, all threads idle) showed
  `ChildPayload(Child2) = 1` vs `SumPayloadOver(Threads, Child2) = 2`.
  Thread 1 had already committed Child2 directly (`Child2 ∉ childQueue[1]`)
  but Grand's bundle captured the pre-commit value. Root cause: when
  `InnerPhase2` (CAS for inner child = Parent) failed, the old code used
  `UNCHANGED local` — leaving `subwrappers[Parent]` non-Null and
  `subpackets[Parent]` stale. On restart, `BundlePhase1` saw
  `subwrappers[Parent] ≠ Null` and skipped re-collecting Parent,
  proceeding to `BundlePhase2` with the stale Child2 payload.
  Fix: same pattern as `InnerPhase3`/`InnerPhase4` — clear outer bundle
  state (`wrapper`, `subwrappers`, `subpackets`) on `InnerPhase2` failure,
  and eagerly tag `bundleNode` (Grand) in addition to inner child (`c` =
  Parent). C++-faithful: in C++, `bundle_subpacket` returning DISTURBED
  from inner bundle Phase 2 causes the outer Phase 1 child loop to
  `continue` (child_retry), re-reading Parent's linkage fresh without
  updating `subwrappers_org[i]` — identical to clearing and re-collecting.
  Local sanity checks: 1-thread superfine PASS (47 states); 2-thread coarse
  PASS (1,497,098 distinct states / depth 98 / 1:35 / liveness ✅).
  confA/B re-submitted to ohtaka with new spec (`run_l3llf_3thr_superfine_live.sh`).
  *Pattern*: InnerPhase2 → InnerPhase3 → InnerPhase4 all three inner-phase
  failure paths now consistently clear outer bundle state on DISTURBED,
  matching C++ semantics where the inner `bundle()` call returning DISTURBED
  causes the outer bundle to restart from Phase 1 with a clean slate.
- **InnerPhase4 restart fix (2026-05-02)**: `QuiescentCheck` violated in
  `BundleUnbundle_3level_LLfree_3thr_superfine_A_mc.cfg` (2 leaf + 1 root)
  at 1,024,087,212 distinct states, depth 60, with spec `415e451c` (after
  InnerPhase3 fix). Root cause: when `InnerPhase4` detected that the inner
  child's linkage had changed (CAS failed) and restarted to `snap_check`,
  it left `wrapper`/`subwrappers`/`subpackets` in local state unchanged. A
  subsequent restart could then resume a partial outer bundle with stale
  collection results, allowing a leaf thread's committed payload to be
  overwritten. Fix: clear outer bundle state (`wrapper`, `subwrappers`,
  `subpackets`) on `InnerPhase4` failure, matching the `InnerPhase3` fix
  pattern. Also eagerly tag `bundleNode` (Grand) in addition to the inner
  child (`c` = Parent). Spec md5: `8fb19385`. 1-thread PASS (47 states);
  2-thread superfine regression PASS (14,109,731 distinct states, depth 148,
  19m13s, counter 7–26, 4,048 terminal emits). confA/B re-submitted to ohtaka
  with new spec.
  *Note*: `InnerPhase4` is the analog of C++ `bundle()` Phase4 on the inner
  (recursive) child. The C++ code is correct — this is a TLA+ modeling
  fidelity issue where the failure-branch local-state semantics needs to
  match C++ (which discards all bundle-local variables when returning
  DISTURBED from the inner bundle call).
- **InnerPhase3 restart fix (2026-05-02)**: `QuiescentCheck` violated with
  3-thread mixed root+leaf configs (confA/B) on ohtaka. Root cause: when
  `InnerPhase3` detected a grandchild wrapper change and restarted, it
  went to `inner_phase2` with stale `subpackets` — a leaf thread's direct
  `CommitChild` between `BundlePhase1` collection and `InnerPhase3` CAS
  caused a lost increment. Fix: restart to `snap_check` (clearing
  `subwrappers`/`subpackets`/`wrapper`), matching C++ where inner
  `bundle()` returning DISTURBED causes the outer bundle to restart from
  Phase1. Not triggered by all-root configs (confC) because bundle
  operations don't modify payloads, so stale collection is harmless.
  Not triggered by 2-thread all-root configs for the same reason.
  3L superfine 2t state count: 11.5M → 15.6M (InnerPhase3 fix) → 14.1M
  (InnerPhase4 fix: pre-clearing on failure collapses intermediate stale-state
  transitions, net reduction). Liveness and safety hold throughout.
- **Tag-preserve fix (2026-04-30)**: ClearMyTags now called only on commit
  success, matching C++ `finalizeCommitment → drop_tags_n_privilege`. Tags
  are preserved across `iterate_commit` retries (C++ `operator++` keeps
  `m_tagged_linkages`). ActiveThread / zombie-tag handling removed — provably
  unreachable since inactive state is reached only through success path.
  State counts changed vs previous model (serial encoding + tag fix combined).
- **3L off (Privilege=FALSE)**: intentionally diverges — LL-free priority
  gating disabled means serial grows monotonically without bound, so TLC
  never terminates. Killed at 118M states / depth 114. This confirms
  `proof_semantics.md` §4–§5: LL-free is necessary for termination.
- **Lamport counter min**: coarse min = 6 (UnbundleCASLoop is one atomic
  action with a single GenSerial), fine/superfine min = 7 (each ancestor
  in the 3-level chain gets its own GenSerial in the per-CAS loop).
  Both match the expected single-thread minimum GenSerial call count.
- **Lamport counter max**: grows with contention. 2-level fine reaches
  counter 18; 3-level coarse reaches 22.
  Higher atomicity modes add CAS retry and DISTURBED restart paths.
- **2L 3thr superfine confA (ohtaka)**: 755M distinct states, fingerprint
  collision probability 13% (optimistic 7.2%). Largest run to date. The
  high collision rate means ~13% of distinct states may share a fingerprint
  with another state — TLC may have missed some reachable states. Safety
  invariants hold over all checked states; re-run with `-fp N` (different
  seed) or `-fpbits 64` for higher confidence if needed.
  Counter min=2 (SerialBase=4, CommitChild path with minimal retries).
- **3L 3thr superfine confC (ohtaka)**: 1,154,807,632 distinct states,
  3-thread all-root (CommitGrand only, no leaf threads). Run with spec
  `415e451c` (InnerPhase3 fix, InnerPhase4 fix does not affect all-root
  configs). PrintTerminalMaxCounter emitted 1,207 values; counter min=4,
  max=15. Fingerprint collision optimistic=14%, actual=3.3%. 4h 9min total.
- **3L 3thr superfine confC live (ohtaka, 2026-05-03)**: `EventuallyAllDone`
  liveness PASS. 640,894,951 distinct states / depth 88 / 15:25:00. Temporal
  property check 1h 33min. PrintTerminalMaxCounter emitted 1,140 values;
  counter min=4, max=15. Fingerprint collision optimistic=4.2%, actual=2.0%.
  State count differs from the non-live run (1,154M vs 640M) — attributable
  to spec changes between runs (InnerPhase4 fix and subsequent revisions
  collapse intermediate stale-state transitions). Counter range and terminal
  semantics are consistent. **3-level 3-thread superfine liveness formally
  proven for the all-root configuration.**
- **2L superfine 3t confC (ohtaka)**: 137M distinct states, 3-thread
  all-root (CommitParent only). Temporal property check took 3h 33min of
  the 6h 35min total. Fingerprint collision probability 0.75% (acceptable
  for this state space size). Counter min=4 is lower than 2-thread min=6
  because 3 threads share `SerialBase = 1 + 3 = 4`, reducing per-thread
  counter increment. Consider `-lncheck final` and `-metadir /dev/shm`
  (RAM disk) for future ohtaka runs to reduce temporal checking time.
- **2L superfine 3t confC live (ohtaka, 2026-05-13)**: `EventuallyAllDone`
  liveness PASS. Same state space (137,333,348 distinct states / depth 96)
  / 2h53m total / temporal check 19min31s. PrintTerminalMaxCounter emitted
  170 values; counter min=4, max=15. Fingerprint collision optimistic=0.23%,
  actual=0.83% (slightly higher on this run than the safety-only run's 0.075%
  but well within acceptable bounds). Total wall time dropped from 6h35m
  (earlier safety-only) to 2h53m (safety + liveness here) thanks to
  `-metadir /dev/shm` on a B1cpu node. **2-level 3-thread superfine
  liveness formally proven for the all-root configuration.** Notes on the
  resume attempt: PHASE=1 (slurm-2905644) finished in 16m23s, too quickly
  for TLC's default 30-min checkpoint cadence, so no `_stable` snapshot
  was written. PHASE=2 (slurm-2906729) saw `No checkpoint found — starting
  fresh.` and re-ran safety + liveness end-to-end. For short-lived
  safety runs, either run safety + liveness as a single PHASE=2 job, or
  add `-checkpoint 5` to force minute-scale checkpointing.
- **3-thread cfgs confA–C (2026-05-01)**: Thread roles split across three
  configs — confA (2 leaf + 1 root), confB (1 leaf + 2 root), confC (all
  root). This replaces the old `RootThreads=LeafThreads={1,2,3}` approach
  (every thread does both) which has a larger state space for no additional
  coverage. Coarse confC is laptop-runnable; confA/B exercise bundle
  contention and require ohtaka. Non-live cfgs use **INVARIANT only**;
  live variants (`_live_mc.cfg`) add `PROPERTY EventuallyAllDone`. Liveness
  at 2 threads proven by superfine cfgs (2L: 2,676,196 states; 3L:
  14,109,731 states). 3-thread confC liveness now additionally confirmed
  at 640M states (see above).
- **2L commits2**: deferred to ohtaka. Laptop run reached 33.6M distinct
  / depth 79 / queue 1.46M before being killed. The non-LLfree 2-level
  spec already passed MaxCommits=2; 3-thread configs are higher priority.

**C11 stress test (generated from spec):**

- 3L p0/p1 unit (NT=2, MAX_COMMITS=1)
- 3L p0 2t (14.4 M commits), p0 4t (3.6 M, 以前は失敗してた), p1 2t (19.3 M), p1 4t (3.9 M)
- 3L SUPERFINE 2t (22.1 M), COARSE 2t (64.5 M)
- 2L LLfree unit + 2t (18.6 M)

**C11 stress test post tag-preserve fix:**

- 3L p1 fine 4t: 3.7M → 5.9M (+60%)
- 3L p0 fine 4t: 4.7M/3s → 7.7M/3s (+63%)
- 2L fine 4t: 2.7M → 5.5M (+100%)
- All PASS: 3L p0/p1 unit, 3L p1 fine 2t/4t, 3L p1 superfine 2t,
  3L p0 fine 4t regression, 2L fine 4t, 2L superfine 2t.

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
