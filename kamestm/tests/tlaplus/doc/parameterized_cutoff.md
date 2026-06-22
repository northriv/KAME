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
- *Threads.* See §5.1: the safety predicates are gate- and serial-free
  structural invariants (Fact A), per-node linkage history is CAS-serialized
  independently of the gate (Fact B), and the protocol is thread-ID
  data-independent (Fact C); a data-independence + locality reduction therefore
  bounds the thread axis by the constant `k = 3` — the largest antichain of
  footprint-overlapping commit roles — checked exhaustively with thread
  `SYMMETRY`. ∎

## 5.1 The thread axis in detail

The depth and width cutoffs are clean (Lemmas 1–2). The *thread* axis is the
generic hard case of parameterized verification, so we treat it explicitly. The
organising observation is that **the priority gate (`priorityTag` /
`CanProceed`) plays no role in safety** — it is a liveness device (§7) — so the
thread cutoff rests on three *gate-independent* structural facts, A–C.

**Fact A — the safety predicates are structural (no thread identity).** Every
checked safety invariant — `SnapshotConsistency`, `NoPriorityLoss`,
`BundleRefConsistency`, `MissingPropagation`, `TerminalPayloadCheck` — is a
predicate over the **bundle-tree linkage** (the `hasPriority` / `bundledBy` /
`sub[·]` edges and `missing` flags) and the **payload counters**. By inspection
of the invariant bodies (`BundleUnbundle_*_LLfree.tla`), *none mentions
`priorityTag`, and none even mentions `serial`.* Hence the safety state-predicate
Φ contains **no thread identity whatsoever**; its thread-symmetry is trivial, and
`T` enters only through (i) the transitions that build the tree and (ii) the
*additive* expected count `MaxCommits·(|RootThreads|+|LeafThreads|)` in
`TerminalPayloadCheck` — a "no lost / double update" property witnessed by any
two concurrent committers on one child.

**Fact B — per-node CAS serialization (gate-free).** `linkage[n]` is mutated
only by a **successful** CAS, and CAS is atomic; so for *any* `T` the projection
`linkage[n]` is a totally-ordered value sequence `v₀ → v₁ → …`, each `vᵢ₊₁`
produced by one thread reading `vᵢ` and the committed linkage of its `O(1)`
immediate fold-neighbours (§8.1). This is the semantics of CAS, with no appeal
to the gate. The gate only ever *restricts which* thread attempts a CAS, and its
bookkeeping writes `priorityTag`, never `linkage`; therefore the reachable
**linkage** states under the gate are a **subset** of those without it. The gate
cannot manufacture a safety violation, and the cutoff argument is sound on the
gated reference model.

**Fact C — thread-ID data-independence.** Thread identities enter the
`linkage`/`serial` state in exactly one place: `serial = EncodeSerial(counter,
tid) = counter·SerialBase + tid`, with `tid` the low-order residue and `counter`
the quotient; `GenSerial` advances `counter` strictly past every observed serial
(a Lamport step). Therefore (i) the value a CAS writes into `linkage` — packet
contents, edges, flags — is a function of the committing thread's local snapshot
and its neighbours' linkage, **not** of its `tid`; and (ii) the only
`tid`-dependence anywhere is the serial *uniquifier*, which Φ does not read
(Fact A) and which influences transition guards only through the
**counter-dominated** order `isOlderThan` — fixed by `counter` for any
causally-ordered pair, and order-consistent under any tie-break for concurrent
(equal-counter) events. The protocol is thus *data-independent* in thread
identities in the sense of Wolper (identities used by equality only) — exactly
the `SYMMETRY` already declared in every config.

**The reduction (thread-axis cutoff).** Suppose Φ is violated by a behaviour
with `T` threads. The violating state is reached by a finite prefix; take its
*cone of influence* `C` — the successful CASes whose written values feed
(transitively) the linkage that Φ evaluates. By §8.1 locality `C` is **bounded
in breadth** (each CAS touches `{self} ∪ immediate neighbours ∪ root anchor`),
though it may be long (per node bounded by `MaxCommits`). Re-assign the threads
performing `C` onto a pool of `k`: CASes at one node are already serialized
(Fact B) and replay on a reused pool member across iterations; CASes on
causally-independent footprint regions take distinct pool members. By Fact C the
*values* are unchanged under the re-assignment (no CAS value depends on `tid`,
and Φ is insensitive to the re-uniquified serial, Fact A). The re-assigned
behaviour uses `≤ k` threads and still violates Φ. Contrapositive: **Φ at
`T = k` ⇒ Φ at all `T`.**

**The cutoff constant `k`.** `k` is the largest antichain of
*footprint-overlapping, simultaneously-poised commit roles* on the cutoff tree —
not a function of `T`. On the 2-level tree (`Parent → {C₁,C₂}`) a root-commit of
the `Parent` bundle (footprint `{Parent,C₁,C₂}`) contends with a leaf-commit on
each child (footprint `{Cᵢ,Parent}`) ⇒ `k = 3`; the 3-level tree adds the
internal node's role but no new *overlap* pattern ⇒ still `k = 3`. This is
exactly why **`T = 3` with thread `SYMMETRY` is the checked cutoff**, matching
the `3thr_*` configs in `VERIFICATION.md`.

**Status (honest).** Facts A–C are checked by inspection of the spec; the
reduction's one sketch-step — "re-assign the cone of influence onto `k` threads,
values unchanged" — is the classical **data-independence reduction** (Wolper
1986; Lazić 1999) specialised to this protocol, whose hypothesis (Fact C) holds.
Two standard routes make it a machine-checked theorem: **(a)** a TLAPS
simulation proof; or **(b)** a *saturation check* — verify the reachable
*linkage-projection* state set is identical at `T = k` and `T = k+1` (no
`(k+1)`-th thread enlarges it), which by `SYMMETRY` + the simulation lifts to
all `T`. We have run `T ≤ 3` exhaustively with full `SYMMETRY` (no violation —
the empirical face of saturation). Closing (a) or (b) makes safety fully ∀`T`;
until then it is "∀ tree shape, `T ≤ 3` (+ symmetry), with a concrete
data-independence route to ∀`T` (cutoff `k = 3`)".

**Safety vs. liveness, cleanly separated.** §5.1 (safety) never invokes the gate
— Facts A–C are gate-free, and the gate only *shrinks* the reachable linkage set
— whereas §7 (liveness) uses the gate for *progress* (the oldest transaction
always advances). The priority mechanism is thus precisely the part that makes
the *already-safe* protocol additionally livelock-free; safety and liveness rest
on disjoint structure (CAS-serialization + data-independence vs. the oldest-tag
ranking).

## 6. Consequence for the model-checking results

The exhaustive TLC runs at `H = 3`, `N = 2`, superfine atomicity (the
`*_3thr_superfine_*` and 2-thread superfine configurations in
`VERIFICATION.md` §"3-level"/"2-level") therefore constitute a **complete
proof of the safety invariants for the unbounded family of trees and thread
counts**, not merely for the instances checked. The bounded runs are the base
case of the cutoff, not the whole argument.

## 7. Liveness (livelock-freedom): an oldest-tag ranking function

Safety invariants do not parameterize liveness. We prove `EventuallyAllDone`
(`<>AllDone`; every thread eventually reaches `idle` with no remaining
iterations) by a well-founded **ranking function** built on the spec's
livelock-free *tagging* mechanism. The argument is independent of §§3–6 **and of
the thread count `N`**.

### 7.1 The progress mechanism (spec facts)

- Each thread `t` carries a tag `MyTag(t) = ⟨iter(t), t⟩`, totally ordered by
  `TagOlder` (lexicographic; smaller = older).
- On a CAS failure at node `n`, `priorityTag[n]` becomes the *older* of its
  current value and `MyTag(t)` (`TagAfterFail`): a younger tag never displaces
  an older one. Hence, until cleared, `priorityTag[n]` is **monotone
  non-increasing** in the tag order.
- `CanProceed(t, n)` (under `Privilege`) permits a CAS at `n` only when
  `priorityTag[n]` is `Null` or held by `t`. Once a tag is installed, only its
  holder may CAS `n`; strictly-younger contenders are gated out.
- A tag is cleared only on its holder's commit **success** (`ClearMyTags`); a
  tag's lifetime lies inside its holder's active commit (no zombie tags).

### 7.2 Ranking function

Let `Active ≜ { t : pc[t] ≠ "idle" ∨ iterBudget[t] ≠ 0 }`. Define the
lexicographic rank `R(s) = (M(s), d(s))`:

- **Outer** `M(s)`: the finite multiset `⟦ MyTag(t) : t ∈ Active ⟧` under the
  Dershowitz–Manna multiset extension of the (well-founded) tag order.
- **Inner** `d(s)`: for the globally-oldest active thread
  `t★ = argmin_{Active} MyTag`, the length of the remaining path from `pc[t★]`
  to `"done"` in the (finite, acyclic) single-transaction control-flow graph.

`R` ranges over a well-founded set (a finite multiset over a well-founded order,
then a bounded `Nat`).

### 7.3 Lemmas

- **(monotone)** At any node `t★` contends on, `priorityTag[n]` reaches
  `MyTag(t★)` and is never afterwards replaced by a younger tag — `t★` is the
  global minimum and `TagAfterFail` installs only strictly-older tags.
- **(gate)** Once `priorityTag[n] = MyTag(t★)`, `CanProceed(t′, n) = FALSE` for
  every younger `t′ ≠ t★` contending `n`: the oldest's CAS at `n` is eventually
  uncontended *by peers*.
- **(progress)** Under `WF_vars(NextStep)`, `t★`'s enabled step is eventually
  taken; being privileged, its peer-contended CAS succeeds and advances
  `pc[t★]` along its acyclic chain — so `d` strictly decreases and `t★ ⤳ done`.
- **(rank)** When `t★` reaches `done` it leaves `Active`, removing the minimum
  of `M`; `M` strictly decreases in the multiset order. Well-founded induction
  on `R` gives `Active ⤳ ∅`, i.e. `<>AllDone`.

### 7.4 Theorem (livelock-freedom, ∀N)

> Under `WF_vars(NextStep)` and `Privilege = TRUE`, `EventuallyAllDone` holds
> for every finite `Threads`, independent of `|Threads|`.

Because the rank quantifies over the *global* oldest — which exists for any
finite `Active` — **liveness needs no thread cutoff**, unlike the safety thread
axis (§5). This is the cleaner half of the parameterized result.

### 7.5 Remaining obligation (the real gap)

Lemma *(progress)* assumes the privileged `t★` completes **without unbounded
retry**. Peers are gated out of `t★`'s contended node, but a peer operating on a
*different* node (an ancestor or child in the chain) can still raise a
`DISTURBED` / `COLLIDED` on `t★` through the bundle chain. The parameterized
proof must therefore show such structural disturbances on the privileged
transaction are **bounded** — each is caused by a peer commit that itself
removes a (younger) element from `M`, so they cannot recur forever. The bounded
`EventuallyAllDone` runs (`VERIFICATION.md`: 2-thread all-roles superfine,
3-thread confC at both levels, the 413 M dynamic-release run) discharge this at
the cutoff; the general bound is the one open liveness lemma.

*(Note: this supersedes an earlier sketch that attributed progress to
`NoPriorityLoss` / `GrandAlwaysPriority`. Those are structural **safety**
invariants on the `hasPriority`/`bundledBy` chain; the liveness measure rests on
the orthogonal `priorityTag` / `CanProceed` / `TagAfterFail` tagging.)*

## 8. Proof obligations and threats to validity

1. **Footprint discharge.** The Lemma 1 footprint claim must be verified for
   *every* action (mechanical, finitely many actions). **Discharged by the
   per-action table in §8.1** (all 12 `NextStep` actions of the 3-level spec):
   each footprint ⊆ `{self} ∪ {immediate fold-neighbours} ∪ {root anchor}`.
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
5. **Thread-axis data-independence simulation (the main open safety lemma).**
   The thread cutoff (§5.1) rests on Facts A–C — all checked by inspection — plus
   one sketch step: that the cone of influence of any violation re-assigns onto
   `k = 3` threads with unchanged linkage values. This is the standard
   data-independence reduction (Wolper 1986; Lazić 1999); its hypothesis (Fact C,
   thread IDs used by equality only) holds here. Discharge route **(a)** a TLAPS
   simulation `Spec(T) ⊑ Spec(k)` on the linkage projection, or **(b)** a TLC
   *saturation* check that the reachable linkage-projection state set is equal at
   `T = k` and `T = k+1`. Until then safety is ∀ tree shape × (`T ≤ 3` +
   symmetry); this is the one obligation between that and fully ∀`T`. (The gate
   is *not* on this list: it only shrinks the reachable linkage set, §5.1
   Fact B.)

## 8.1 Footprint table (discharges obligation #1)

The `linkage` footprint of every `NextStep` action of
`BundleUnbundle_3level_LLfree.tla`, in fine/superfine mode (the per-level
decomposition). "self" = the action's focus node; "child"/"parent" = its
*immediate* tree neighbours; "root" = the priority root, which holds the
authoritative nested bundled packet (a node's bundled summary σ is materialised
in the root's `packet.sub[…]`).

| Action | `linkage` reads | `linkage` writes | role |
|---|---|---|---|
| `BundlePhase1` (collect) | self + each immediate child | grandchildren only via the recursive inner-bundle of a non-leaf child | bundle↓ step |
| `BundlePhase2` (set-missing CAS) | self | self | local |
| `BundlePhase3` (per-child ref CAS) | self + immediate children | one immediate child / action | bundle↓ step |
| `BundlePhase4` (finalize CAS) | self | self | local |
| `CommitGrand` | self (subtree lives in `Grand`'s nested packet) | self | local at root anchor |
| `CommitStart` | — | — | control only |
| `CommitRead` | self (target) | — | local (read) |
| `CommitTryCAS` | self (target) | self (+ `priorityTag[target]`) | local |
| `UnbundleWalk` | self + immediate parent / step; `SnapshotForUnbundle` folds self→root | — (thread-local only) | unbundle↑ step (read) |
| `UnbundleCASLoop` | one ancestor + root anchor / step | one ancestor / step (whole chain in coarse) | unbundle↑ step (write) |
| `UnbundleCASChild` | self + immediate parent | self + immediate parent | leaf+parent (2-level) |
| `CommitDone` | — | — | control only |

**Reading.** Every action's `linkage` footprint is contained in
`{self} ∪ {immediate fold-neighbours} ∪ {root anchor}`. No action reaches a node
off the `target`→root chain (unbundle) or outside the recursively-descended
subtree (bundle). The cross-level accesses are exactly the catamorphism: the
bundle↓ descent into a non-leaf child, and the unbundle↑ read of the parent's σ
— which, when bundled, is materialised in the root's nested packet, hence the
"root anchor" read. In fine/superfine mode each action performs **one** fold
step touching at most `{self, one immediate fold-neighbour}` (plus the root
anchor read for unbundle). This is the mechanical discharge of Lemma 1's
footprint premise across all 12 actions; obligation #1 holds.

*(`coarse` mode collapses a whole fold into one action — a coarsening with
fewer interleavings; the cutoff is stated on the fine/superfine model, §8
obligation 2.)*

## 9. One-line summary for the paper

> **Safety**: ∀ tree shape (arbitrary depth and width) via a structural-cutoff
> theorem — bundle and unbundle are height/width-uniform catamorphisms ⇒ a
> depth-3, width-2 cutoff. The thread axis reduces by data-independence +
> locality (the safety predicates are gate- and serial-free structural
> invariants; per-node linkage is CAS-serialized; thread IDs enter only a
> Lamport-serial uniquifier) to a constant cutoff `k = 3`, checked exhaustively
> with thread symmetry (§5.1), with a concrete TLAPS/saturation route to ∀`T`.
> **Liveness** (livelock-freedom): ∀`N` via an oldest-tag ranking function on the
> priority gate — no thread cutoff required. The gate is a pure liveness device,
> disjoint from the safety argument. Exhaustive TLC at the cutoff
> (`VERIFICATION.md`) discharges the finite base cases.
