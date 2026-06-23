# Parameterized correctness of the bundle/unbundle STM via a structural cutoff

*Draft methodology section (2026-06-22; thread axis revised 2026-06-23).
Captures the argument that lifts the finite TLC results in `VERIFICATION.md`
toward an unbounded claim. The **tree axis** (∀ height, ∀ width) is a
structural-cutoff theorem (catamorphism reduction). The **thread axis** is
deliberately scoped: not a ∀`T` theorem, but exhaustive checking to the
contention bound plus a measured structural-saturation result, with ∀`T` as a
strongly-evidenced conjecture (§5.1). Lemma/Theorem statements are paper-ready;
marked proofs are sketches. English to match the other `doc/` material.*

## 1. Why "check a bigger instance" is not a proof

TLC is exhaustive only for the *finite instance* it is given: a fixed thread
count `T`, a fixed tree (height `H`, width `N`), and a fixed `MaxCommits`.
Enlarging the instance — more threads, deeper or wider trees — never reaches a
universally-quantified ("for all `T`, all trees") statement; it is an unbounded
regress in which one can always run "one more, larger" configuration.

For the **tree axis** we therefore do not argue "we checked enough cases": we
prove a **cutoff theorem** — correctness on a tree of *any* height and width
reduces to correctness on one small fixed tree, discharged exhaustively with
TLC. The reduction rests on one structural property — that every operation is a
*catamorphism* (a structural fold) over the tree, whose per-node step is local
and uniform in both height and width (§§3–6).

The **thread axis** is harder and we are honest about it: there is no clean
mechanical ∀`T` proof (parameterized verification is undecidable in general), so
we do *not* claim one. We instead establish sound thread symmetry, check
exhaustively to the contention bound, and *measure* that the safety-relevant
state space saturates — presenting ∀`T` as a strongly-evidenced conjecture
(§5.1). The reader should read "∀ tree shape, all thread counts" as "proven for
trees; conjectured-with-strong-evidence for threads."

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
> `TerminalPayloadCheck`, … — hold on a tree of arbitrary height `H` and
> arbitrary width `N` **iff** they hold on the instance with `H = 3` (root + one
> internal node + leaves) and `N = 2` children (the structural cutoff, Lemmas
> 1–2). The **thread** axis is not closed to a theorem: we verify it up to the
> contention bound and present ∀`T` as a conjecture (§5.1).

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
- *Threads.* Not reduced to a theorem (§5.1). The safety predicates are gate-
  and serial-free structural invariants (Fact A), per-node linkage is
  CAS-serialized (Fact B), and thread identity enters only as a serial
  uniquifier (Fact C) — so thread `SYMMETRY` is *sound* but, being permutation
  only, does **not** reduce the thread count. We instead verify exhaustively to
  the contention bound (`T ≤ 3` full / `T = 4` core) and observe that the
  identity-free bundle-structure saturates (set-identical `T = 2 ≡ T = 3`),
  giving ∀`T` as a strongly-evidenced conjecture. ∎

## 5.1 The thread axis: exhaustive checking + structural saturation (∀`T` as a conjecture)

The depth and width axes admit clean structural reductions (Lemmas 1–2). The
**thread axis does not**, and we are deliberate about scope: *we do not claim a
∀`T` theorem.* Parameterized verification is undecidable in general
(Apt–Kozen 1986); a sound mechanical ∀`T` proof for this protocol would require a
guided TLAPS development or a separately-validated thread-free abstraction,
neither of which we mechanize. Instead we establish the thread axis by three
machine-checkable ingredients — **sound symmetry**, **exhaustive checking to the
contention bound**, and a **measured structural-saturation** result — and present
∀`T` safety as a strongly-evidenced **conjecture**. The priority gate
(`priorityTag` / `CanProceed`) plays no role here — it is a liveness device (§7);
the three facts below are gate-independent.

**Why the safety state carries no thread identity (Facts A–C).**
- **A — structural predicates.** Every checked safety invariant
  (`SnapshotConsistency`, `NoPriorityLoss`, `BundleChainValid` /
  `BundleRefConsistency`, `BundledByCorrect`, `GrandAlwaysPriority`,
  `MissingPropagation`, `TerminalPayloadCheck`) is a predicate over the
  **bundle-tree linkage** (`hasPriority` / `bundledBy` / `sub[·]` edges, `missing`
  flags) and **payload counts**. By inspection of the invariant bodies, *none
  reads `priorityTag` or `serial`* — none mentions a thread.
- **B — per-node CAS serialization.** `linkage[n]` is mutated only by a
  *successful* atomic CAS, so its history at each node is a single serialized
  value-sequence for any `T`. The gate only restricts *which* thread attempts a
  CAS (it writes `priorityTag`, never `linkage`), so it cannot manufacture a
  safety violation.
- **C — identity enters only as a uniquifier.** Thread identities appear in the
  state in exactly one place: the Lamport serial `serial = counter·Base + tid`,
  where `tid` is a low-order **uniquifier** (it makes serials distinct).
  `GenSerial` makes `counter` dominate causal order; `tid` only tie-breaks
  causally-*concurrent* events, where any consistent tie-break is sound (this is
  the order-sensitive data-independence of Lazić 1999, not Wolper's
  equality-only case). The safety-relevant behaviour is thus invariant under any
  permutation of thread identities.

**(i) Sound symmetry.** By A–C the system is symmetric under
`Permutations(Threads)`, so TLC's `SYMMETRY` reduction is **sound** for the
safety invariants. Crucially, symmetry reduces the state space at a *fixed* `T`;
it does **not** reduce `T` itself — collapsing thread *permutations* never maps a
`T`-thread run to a `(T-1)`-thread run. **Symmetry ≠ cutoff**; it makes the
bounded checks below sound and faster, nothing more.

**(ii) Exhaustive checking to the contention bound.** On the cutoff tree the
commit targets are just the root and the two leaves — the interior node is never
a direct commit target ("Parent is not targeted here", spec) — so the genuinely
concurrent contention is ≈ 3. We model-check the full **superfine** protocol
exhaustively at `T ≤ 3` with symmetry, and the coarse 2-level contention core at
`T = 4`: **136,366,732 distinct states, all safety invariants hold** (28 min, no
error). A dangerous `N`-thread CAS interleaving within these bounds would surface
as an invariant violation with a concrete counterexample trace; none does.

**(iii) Structural saturation (measured).** Project each reachable state onto its
**identity-free bundle structure** — per node ⟨`hasPriority`, `bundledBy`,
`missing`, which `sub` slots are populated⟩, dropping `serial` and payload value
(exactly the fields the structural invariants read). Computed from the exhaustive
dumps, this projection is a small finite set that **saturates**: it is
**set-identical at `T = 2` and `T = 3`** (verified by diff) — **4** structures
for the bundle-only workload, **6** once unbundle is exercised (the extra two are
the partial-unbundle intermediates: one child detached while the other is still
bundled) — even though the raw reachable-state count **explodes**
`1,093 → 339,744 → 136,366,732` across `T = 2,3,4`. More threads multiply
interleavings, serials, and increment counts but reach **no new safety-relevant
tree structure**. The one quantity that *does* scale with `T` is the payload
increment count; its **correctness** (no lost or doubled update) is a per-node
CAS-serialization property, and `TerminalPayloadCheck` passes at every checked
`T`, including the 136 M-state `T = 4` run. (Numbers: `VERIFICATION.md`, the
thread-saturation table.)

**Status (honest).** Safety is *mechanically verified* for all tree shapes
(Lemmas 1–2) and for thread counts up to the contention bound (`T ≤ 3` full
protocol; `T = 4` contention core, 136 M states), with sound symmetry (i) and
observed structural saturation (iii). We present **∀`T` safety as a conjecture**,
not a mechanized theorem — supported by (a) the safety-relevant structure being
finite and *observed* stable across `T` (set-identical `T=2 ≡ T=3`), and (b) the
identity-free, CAS-serialized structure of the protocol (Facts A–C). This scoping
is deliberate: parameterized verification is undecidable in general, and we do
not rest any claim on an unmechanized cutoff. Promoting the conjecture to a
theorem — a guided TLAPS simulation `Spec(T) ⊑ Spec(3)`, or a separately-validated
thread-free abstract CAS model whose reachable structure is checked equal to the
saturated projection — is future work.

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
`VERIFICATION.md` §"3-level"/"2-level") therefore constitute, **for the tree
axis**, a complete proof of the safety invariants on the unbounded family of
trees — not merely the instances checked: they are the base case of the
structural cutoff (§§3–5), and Lemmas 1–2 lift them to all `H`, `N`. **For the
thread axis** these same runs are the exhaustive base (up to the contention
bound) plus the structural-saturation measurement that *support the ∀`T`
conjecture* (§5.1) — they are not, and we do not present them as, a ∀`T` proof.

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
5. **Thread axis is a conjecture, not a theorem (deliberate scope).** Safety is
   mechanically verified to the contention bound (`T ≤ 3` full protocol; `T = 4`
   contention core, 136 M states) under sound symmetry, and the identity-free
   bundle-structure is *observed* to saturate (set-identical `T = 2 ≡ T = 3`;
   §5.1). ∀`T` is presented as a strongly-evidenced conjecture — not proven —
   consistent with the general undecidability of parameterized verification. No
   claim in this document rests on an unmechanized ∀`T` cutoff. Promoting it to a
   theorem (a guided TLAPS simulation `Spec(T) ⊑ Spec(3)`, or a
   separately-validated thread-free abstract CAS model checked equal to the
   saturated projection) is future work. (The gate is irrelevant here: it only
   shrinks the reachable linkage set, §5.1 Fact B.)

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

> **Safety, tree axis**: ∀ tree shape (arbitrary depth and width) via a
> structural-cutoff theorem — bundle and unbundle are height/width-uniform
> catamorphisms ⇒ a depth-3, width-2 cutoff, discharged exhaustively by TLC.
> **Safety, thread axis**: *not* claimed as a ∀`T` theorem. Thread symmetry is
> sound (IDs enter only a Lamport-serial uniquifier) but, being permutation only,
> does not reduce the thread count. We verify exhaustively to the contention
> bound (`T ≤ 3` full protocol; `T = 4` core = 136 M states, all pass) and
> *measure* that the identity-free bundle-structure saturates (set-identical
> `T = 2 ≡ T = 3`: 4 structures bundle-only, 6 with unbundle; raw states explode
> 1k→340k→136 M). ∀`T` is a strongly-evidenced **conjecture**, consistent with
> the undecidability of parameterized verification. **Liveness**
> (livelock-freedom): ∀`N` via an oldest-tag ranking-function argument on the
> priority gate — no thread cutoff required, a sketch verified exhaustively at
> small `T` (§7). The gate is a pure liveness device, disjoint from safety.
