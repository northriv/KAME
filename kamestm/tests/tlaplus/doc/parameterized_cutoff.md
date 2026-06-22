# Parameterized correctness of the bundle/unbundle STM via a structural cutoff

*Draft methodology section (2026-06-22). Captures the argument that lifts the
finite TLC results in `VERIFICATION.md` to an unbounded (∀ tree shape, ∀ thread
count) correctness claim. Lemma/Theorem statements are paper-ready; the marked
proofs are sketches to be expanded. English to match the other `doc/` material;
translate as needed.*

## 1. Why "check a bigger instance" is not a proof

TLC is exhaustive only for the *finite instance* it is given: a fixed thread
count `T`, a fixed tree (height `H`, width `N`), and a fixed `MaxCommits`.
Enlarging the instance — more threads, deeper or wider trees — never reaches a
universally-quantified ("for all `T`, all trees") statement; it is an unbounded
regress in which one can always run "one more, larger" configuration.

We therefore do not argue "we checked enough cases." We prove a **cutoff
theorem**: correctness on *every* instance reduces to correctness on a single
small fixed instance, which we then discharge exhaustively with TLC. The
reduction rests on one structural property — that every operation is a
*catamorphism* (a structural fold) over the tree, whose per-node step is local
and uniform in both the tree's height and its width.

## 2. Model and notation

A node `n` holds a linkage word `linkage[n] = ⟨packet, serial, bundledBy,
hasPriority⟩`. A `packet` carries a nested sub-packet map `sub[·]` indexed by
`n`'s immediate children. Write `children(n)` / `parent(n)` for the tree edges
(`parent(n) = linkage[n].bundledBy`); the root has `hasPriority = TRUE`.

Define the **edge summary** that a node exposes to its neighbour:

> σ(n) ≜ (committed packet of `n`, its serial, its missing/bundled flag).

A node's bundled packet nests its children's summaries in `sub[·]`; dually, a
node's own summary is recovered from its parent's packet via `sub[n]`.

## 3. Lemma 1 (fold-uniformity / structural locality)

> Every bundle and unbundle transition is a **catamorphism** over the tree:
>
> - **bundle** folds *downward* over the subtree
>   `bundle(n) = collect( linkage[n], { bundle(c) : c ∈ children(n) } )`;
>   a non-leaf child is recursively bundled first (its grandchildren are CAS'd
>   in the course of that recursive call).
> - **unbundle** folds *upward* over the ancestor chain
>   `unbundle(n) = extract( linkage[n], unbundle(parent(n)) )`, with base case
>   at the priority root; `n`'s sub-packet is taken from the recursively
>   obtained ancestor packet's `sub[n]` slot.
>
> Each per-node step reads and writes only `{linkage[n]}` together with the
> fold result of the **immediate** fold-neighbour(s) — the children for bundle,
> the parent for unbundle. The combining function is independent of the tree's
> height and of the number of children beyond the one slot it indexes. Children
> enter only through symmetric aggregates (`[c ↦ linkage[c]]`, `∀ c …`) and
> monotone "all-children-done" thresholds.

**Proof (sketch, mechanically checkable).** By inspection of the spec actions.
For unbundle, `SnapshotForUnbundle(n)` is literally `step(linkage[n],
SnapshotForUnbundle(parent(n)))` with the root as base case
(`BundleUnbundle_3level_LLfree.tla`, `SnapshotForUnbundle`); `UnbundleWalk`
advances one level per action in fine/superfine mode, and `UnbundleCASLoop` /
`UnbundleCASChild` write one ancestor / the target child at a time. For bundle,
`BundlePhase1` reads `[c ↦ linkage[c]]` over `children(n)`, computes
`allCollected = ∀ c : childPkts[c] ≠ Null`, and recurses into a non-leaf child
via `innerBundled` (the only path that touches a grandchild — i.e. the
downward fold step); `BundlePhase3` CAS-es children one at a time guarded by
the monotone `allMatch` / `allDone` thresholds. The variable footprint of each
action is contained in the stated set; the fold structure is the recursion's
own shape. ∎

The key point is **not** that an action touches only two levels (it does not:
`SnapshotForUnbundle` reads the whole ancestor chain in one evaluation, and an
inner bundle reaches a grandchild). It is that the cross-level access is a
*fold with a height-uniform local step* — a catamorphism — so the depth
induction below goes through without restructuring the algorithm.

## 4. Lemma 2 (child symmetry and the width threshold)

> The spec is invariant under permutation of a node's children; the per-child
> CAS steps of distinct children commute (their footprints are disjoint); and a
> parent's aggregate predicate (`is_bundle_root` ⇔ all children bundled) is a
> monotone threshold over the child set. Hence every behaviour over `N ≥ 2`
> children is simulated by a 2-children behaviour, by collapsing children
> `3..N` onto child 2.

Two children already realise the three width-sensitive patterns: one child
done while another is pending; two threads contending on the *same* child; two
threads working on *different* children. A third child only repeats them.

## 5. Theorem (structural cutoff)

> The safety invariants — `SnapshotConsistency`, `BundleChainValid`,
> `NoPriorityLoss`, `GrandAlwaysPriority`, `MissingPropagation`,
> `TerminalPayloadCheck`, … — hold on a tree of arbitrary height `H`, arbitrary
> width `N`, and under arbitrary thread count `T`, **iff** they hold on the
> instance with `H = 3` (root + one internal node + leaves), `N = 2` children,
> and `T = k` threads (the contention cutoff).

**Proof (sketch).**
- *Depth.* Induction on `H` via Lemma 1. The only node roles are *leaf*,
  *internal* (simultaneously a child of its parent and a parent of its
  children), and *root*. The `H = 3` instance instantiates all three roles and
  exercises both fold directions at the internal node (the root bundles the
  internal node, reaching the leaves; a leaf unbundles up through the internal
  node to the root). Any `H ≥ 4` adds only further applications of the same
  uniform step (Lemma 1) at additional internal nodes, introducing no new
  role-interaction.
- *Width.* Lemma 2 reduces any `N` to 2.
- *Threads.* The footprint locality of Lemma 1 confines interference to
  contention on a single linkage word; such contention is pairwise, so `k`
  threads (one per role at the contended node) realise every interference
  pattern. ∎

## 6. Consequence for the model-checking results

The exhaustive TLC runs at `H = 3`, `N = 2`, superfine atomicity (the
`*_3thr_superfine_*` and 2-thread superfine configurations in
`VERIFICATION.md` §"3-level"/"2-level") therefore constitute a **complete
proof of the safety invariants for the unbounded family of trees and thread
counts**, not merely for the instances checked. The bounded runs are the base
case of the cutoff, not the whole argument.

## 7. Liveness (livelock-freedom) — ranking argument  *[TO COMPLETE]*

Safety invariants do not parameterize liveness. Livelock-freedom
(`EventuallyAllDone`) is argued instead by a well-founded **ranking function**,
independent of §§3–5:

- The globally-oldest in-flight transaction's priority — the Lamport `serial`
  carried in `linkage[·]` — is never overtaken (`NoPriorityLoss`,
  `GrandAlwaysPriority`). Hence the oldest transaction's CAS cannot fail
  forever; it commits within a bounded number of its own steps, after which
  the next-oldest becomes the oldest. The rank (number of transactions older
  than a given one, with serial order as the well-founded measure) strictly
  decreases, so some transaction always makes progress ⇒ no livelock, for all
  `N`.
- The bounded `EventuallyAllDone` checks (2-thread all-roles superfine, the
  3-thread all-root confC at both 2- and 3-level, the 413 M-state dynamic
  release run) discharge the base case / corroborate the measure.

*To do:* state the ranking function formally over the spec variables, prove it
strictly decreases under `NextStep` while fairness holds, and prove the
oldest-priority-preservation lemma.

## 8. Proof obligations and threats to validity

1. **Footprint discharge.** The Lemma 1 footprint claim must be verified for
   *every* action (mechanical, finitely many actions). Confirmed by inspection
   for `BundlePhase1–4` / `CollectSubpacket` (downward) and `UnbundleWalk` /
   `SnapshotForUnbundle` / `UnbundleCAS*` (upward).
2. **Atomicity model.** The cutoff is stated for the **fine/superfine** spec,
   in which the fold is decomposed per level/child — the most-interleaved,
   C++-faithful model and the one we check. The `coarse` mode collapses the
   fold into a single step (a coarsening with fewer interleavings).
3. **Hard links.** When a child has ≥ 2 parents, its summary σ is exposed to
   each parent simultaneously; the cutoff requires σ to be identical from every
   parent. The hard-link bundle/unbundle models
   (`BundleUnbundle_hardlink_*`) instantiate this case (see `VERIFICATION.md`
   §5).
4. **Serial arithmetic.** Priority comparison is the pairwise unsigned-
   difference of Lamport serials, independent of `N`; wraparound is bounded far
   below `2^47`, so enlarging `N` cannot induce a spurious ordering.

## 9. One-line summary for the paper

> Safety is proved for all trees and all thread counts by a structural-cutoff
> theorem (bundle and unbundle are height/width-uniform catamorphisms ⇒ depth-3,
> width-2 cutoff); livelock-freedom is proved for all `N` by an
> oldest-priority ranking function. Exhaustive TLC at the cutoff discharges the
> finite base cases (`VERIFICATION.md`).
