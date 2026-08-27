# `transaction_dynamic_node_test`: the fault is a `Packet` use-after-free

Handoff for the TLA+ work.  Written 2026-08-24 from the x86-64 / g++ 15.2 /
Ubuntu 26.04 `PREEMPT_RT` session (2 cores).  Continues
`origin/claude/great-turing-Ufao2` and the header of `soak_dynnode.sh`, which
narrowed the fault to ILP32 + the pool allocator and ended with "no backtrace
yet" on LP64.

**There is now an LP64 reproducer that fires most runs, and the fault has been
caught in the act.  It is not an allocator defect and not a miscompile.**

> **Header note added later, and it contradicts the line above.**  §6's
> mixed-compiler table (clang-STM + gcc-pool **8/12**, gcc-STM + clang-pool
> **0/12**, allocator `-O2` **0/8**) puts the fault in the **allocator's** TU and
> makes its dependence on `-fipa-cp-clone` a codegen property.  §13.103 onwards
> works from that; this header's "not an allocator defect and not a miscompile"
> is the 2026-08-24 reading and is left in place for the record rather than
> quietly corrected.

## 0. Working convention for the two sessions

Both sides append numbered sections to this file, so **`git fetch` before
writing a section number and before every push.**  Four collisions have already
happened (§13.104, §13.110, §13.121, §13.132 were each written twice) and each
cost a rebase with a hand-resolved conflict — one of which
(`7899e2210`) went in with **unresolved conflict markers** and left the
allocator uncompilable on the branch tip for several commits, which is the worst
outcome available since a build-and-measure loop then reads stale artefacts as
results.

Taking the next free number after a fetch is enough; a parity split (one side
even, one odd) was considered and rejected as more bookkeeping than the problem
warrants.  If a collision does happen, **renumber the later-written section** and
say so in its text, so cross-references from earlier sections keep pointing at
what they meant.

---

## 1. The finding

A `Packet` is destroyed while a live `PacketWrapper` still holds
`m_packet` pointing at it.  The `local_shared_ptr<Packet>` refcount reaches
zero with an outstanding reference.

Caught by poisoning every slot with `0xBAADF00DBAADF00D` inside
`deallocate_pooled_or_free()` (allocator-side, so it sees the library's own
internal allocations too) and running under gdb:

```
si_signo = 11 (SIGSEGV)
si_code  = 128  (SI_KERNEL — general-protection fault, non-canonical address)
si_addr  = 0x0  (#GP does not report the address here; si_code is the
                 discriminator, not si_addr)

fault at  Transactional::Node<LongNode>::bundle(...)+113
          transaction_impl.h:2733   Node &supernode(supscope->packet()->node());

  mov 0x18(%r14),%rax   ; rax = supscope's PacketWrapper*   0x7fffeb5847a0  valid
  mov 0x10(%rax),%rsi   ; rsi = wrapper->m_packet (Packet*) 0x7ffff53a0420  valid
  mov 0x8(%rsi),%rdx    ; rdx = packet->m_payload           0xbaadf00dbaadf00d  POISON
  mov 0x10(%rdx),%rdx   ; deref payload->m_node             → #GP
```

Offsets: `PacketWrapper` = {refcnt @0, `m_bundledBy` @8, `m_packet` @0x10};
`Packet` = {refcnt @0, `m_payload` @8}; `Payload` = {vptr @0, refcnt @8,
`m_node` @0x10}.

The **wrapper is not freed** — reading `m_packet` out of it yields a good
pointer.  The block that went through `free` is the **`Packet`**: its
`m_payload` word reads back as poison, which only happens if that whole slot
was deallocated.

Call stack at the fault (the cloud session's exact signature):

```
snapshot → bundle → bundle_subpacket → bundle → bundle_subpacket → bundle
```

## 2. What to model

The property to check is:

> **A `Packet`'s reference count reaches zero while a reachable
> `PacketWrapper` still references it.**

No allocator model is needed.  Earlier work (both sessions) chased an
allocator identity-reuse story; it is **refuted** — see §5.  The allocator
only decides what the freed `Packet`'s memory *contains* when the stale read
happens, which is why every compiler and allocator knob moved the failure
rate without being the cause.

Add a refcount to `Packet` in the existing `BundleUnbundle_*` family and
check that invariant.  The existing specs allocate fresh identities forever,
so they cannot currently express it.

### Topology the fault needs

Established by ablation (§4), so the model does not need to be larger than
this:

- **A static three-level shared tree** `gn1 ← gn2 ← gn3`.  Two levels did
  **not** reproduce (`TWOLEVEL` 0/8 against a 44 % base — provisional, worth
  one confirmation run before you rely on it, since it decides whether you
  model `BundleUnbundle_2level_*` or `_3level_*`).
- **N threads**, each holding a **thread-local** root `p1` with a child `p2`.
- **A hard link**: `p2` is spliced under `gn2` inside a `gn1`-rooted
  transaction while still a child of `p1`, so `p2` has two parents — one
  thread-local, one in the shared tree.  This is the
  `BundleUnbundle_hardlink_external*` shape ("external" = the other parent is
  outside the tree).
- Concurrently: a multi-node Tx rooted at `gn2` writing `gn2`/`gn3`, and two
  single-node Tx on the leaf `gn3`.
- **Repeated thread creation and exit** under a tree that outlives them.
  One round with the same total work does **not** fire; the per-round tree
  rebuild is **not** needed.

The crash sits in the recursive bundle build, so the interleaving to explore
is a `bundle` / `bundle_subpacket` recursion racing against whatever drops
the last reference to a `Packet` the recursion is walking.

### Where to look in the C++

- `bundle()` / `bundle_subpacket()` recursion, `transaction_impl.h` ~2733,
  ~2841 (`bundle_subpacket` call), ~2488 (`snapshot`'s call into `bundle`).
- `scoped_atomic_view` ownership transfers in
  `kamepoolalloc/atomic_smart_ptr.h`: the `local_shared_ptr<T>&&` move-in
  constructor and `assign_from_local()`, which hand a `+1` refcount around
  with **zero atomic operations** and document that the caller is responsible
  for the moved-from pointer being the atomic's current value ("we do NOT
  verify").
- `Linkage::~Linkage()` — `this->reset()` with the comment "Packet should be
  freed before memory pools".

## 3. The reproducer

`kamestm/tests/tmin_dynnode.cpp` (staged in this commit, 148 lines, from the
original 365).

```bash
c++ -DKAMEPOOLALLOC_DYLIB -DA_NO_P1TREE \
    -I kamestm/tests -I kamestm -I kamepoolalloc \
    -O3 -g -DNDEBUG -std=gnu++17 -include kamestm/tests/support_standalone.h \
    kamestm/tests/tmin_dynnode.cpp kamestm/tests/support_standalone.cpp \
    kamestm/threadlocal.cpp \
    -o tmin -Wl,-rpath,<build>/kamepoolalloc-tests \
    <build>/kamepoolalloc-tests/libkamepoolalloc.so -ldl

./tmin 100 16 1250      # rounds threads iters — ~25 s, fires 40-65 %
```

`-DA_NO_P1TREE` is the recommended base (see §4).  Failure modes: SIGSEGV,
SIGABRT (uncaught `NodeNotFoundError`), hang, and `failed objcnt=N` — the
last being the ILP32 signature from `d76e5b2d9`, which does occur on LP64.

The original test needed `NUM_THREADS 4 → 8` to fire at all on LP64
(~50 %/60 s); this is 25 s and needs no source change.

### To turn the failure on and off

`-O3` vs `-O3 -fno-ipa-cp-clone` on `allocator.cpp` is a reliable switch:
**43/100 vs 0/167**.  Use it as an experimental control.  It is *not* a lead
— see §5.

To make a stale read fault immediately and identifiably, apply the poison
patch to `deallocate_pooled_or_free()` (scratch copy of `allocator.cpp`):

```c
if(p) {
    std::size_t _psz = kame_pool_malloc_usable_size(p);
    if(_psz >= 16 && _psz <= 4096) {
        std::uint64_t *_q = reinterpret_cast<std::uint64_t *>(p);
        for(std::size_t _i = 0; _i < _psz / 8; ++_i)
            _q[_i] = 0xBAADF00DBAADF00DULL;
    }
}
```

It must go **inside the library**.  An `operator new`/`malloc` interposer in
the executable does not work: the `.so` is built with
`-fno-semantic-interposition`, so its internal allocations bind to its own
definitions and never cross the boundary.  Four rounds of filtering left a
false-positive floor of ~450 per run, identical in failing and clean builds.

## 4. Ablation — what the fault needs

Each arm removes one element.  All runs `100 16 1250`.

**On the tree-rebuilt base (16 runs/arm):**

| removed | failed |
|---|---|
| nothing | 7 / 16 |
| main thread's `gn3->insert/release(gn4)` churn | 6 / 16 |
| `gn2->swap(p2, gn3)` | 6 / 16 |
| `gn4` transaction | 0 / 16 |
| `p1->insert/release(p2)` | 0 / 16 |
| `trans(*gn3)` pair | 0 / 16 |

**On the keep-tree base (15 runs/arm):**

| removed | failed |
|---|---|
| nothing | 7 / 15 |
| the `gn1` multi-node Tx | 10 / 15 |
| the `p1`-into-`gn1` splice | 8 / 15 |
| the `p2`-into-`gn2` splice | 6 / 15 |
| the `gn2` Tx | 0 / 15 |
| the private `p1`/`p2` churn | 0 / 14 |
| the `trans(*gn3)` pair | 0 / 15 |

**On the `A_NO_P1TREE` base (9 runs/arm):** base 4/9; every further cut 0/9;
`TWOLEVEL` 0/8.

**Cuts do not compose.**  Three arms that were individually free (10/15,
8/15, 6/15) gave 0/6 when applied together, and `gn1TX + p1TREE` gave 0/4.
Each cut also removes concurrency, so re-ablate after every adopted cut and
compensate with thread count — raising 8 → 16 threads restored 4/4 after
`gn4` was removed.

Removed and verified unnecessary: `ComplexNode`, the whole single-threaded
setup block, every `print_()`/`tr1.print()` dump, the main-thread churn,
`gn2->swap`, `gn4`, the per-round tree rebuild, and the `total` payload-sum
bookkeeping.

## 5. Retracted claims

Recorded so nobody re-derives them.  Several are mine from this session.

- **"Chunk identity reuse is the vehicle."**  The theory was that an
  `LRC_CHUNK` block sits in the recycle cache with its units still claimed
  and `chunk_header` intact, so a stale pointer still resolves through the
  radix.  Disabling the chunk cache entirely (`large_recycle_push` refuses
  `LRC_CHUNK`) gives **13/30 against a control of 12/31**.  Refuted.
- **"Aggregate codegen perturbation hides it."**  Refuted by perturbing
  without removing the clones: `-fno-ipa-sra` 10/16, `-fno-inline-functions`
  8/16, `--param inline-unit-growth=10` 7/16 (334 functions reshuffled),
  against `NC7` 0/16 (144 reshuffled).  The earlier monotone-in-perturbation
  trend was an artefact of 7–9 run samples.
- **"`-fipa-cp-clone` is a miscompile."**  It is an accelerator.  It changes
  what the freed `Packet`'s memory contains when the stale read happens.
  Same `mmap`/`munmap` counts and same wall time either way.
- **"The clone bodies accept an undersized block."**  `sz >= need` is present
  in the clone as `$262143`; checked on the ILP32 side.
- **"`noclone` on one function isolates that clone."**  It does not —
  suppressing one clone changes 72 other functions' sizes.
- **"`noclone`×5 ≡ `-fno-ipa-cp-clone`."**  True on i586/g++13, **false** on
  x86-64/g++15, where `bucket_release_chunk` (3 clones) and
  `find_training_zeros` (2) survive.  Seven functions are needed here, and
  clone-set equality is still not codegen equality (34 function sizes differ).
- **"The release fence is on the wrong side by construction."**  Wrong;
  `writeBarrier()` is `atomic_thread_fence(release)`, undeletable, and its
  partner is the client's later atomic publication.
- **`LOCAL_REF_CAPACITY`** — already withdrawn by the other session on
  arm64's 2400 clean runs; `KAME_LOCAL_REF_CAPACITY_OVERRIDE=4` on x86-64
  added nothing here either.

## 6. Experimental record

So these are not re-run.  All on the minimised reproducer unless noted.

| experiment | result |
|---|---|
| pool ON vs OFF (8-thread original) | fires 40–75 % vs **0/6**, 0/8 |
| `operator new` quarantine, reuse delayed 2 M allocations | **0/41** vs 13/41 |
| gcc-STM + clang-pool | **0/12** |
| clang-STM + gcc-pool | 8/12 |
| gcc vs clang, full test | 10/19 vs **0/19** |
| allocator `-O3` / `-O2` / `-Os` / `-O1` | 6/8, **0/8**, **0/8**, **0/7** |
| `-O2` + all 13 `-O3` flags / first 7 / last 6 | 13/28, 13/28, **0/27** |
| per-flag: only `-fipa-cp-clone` non-zero | 2/10, others 0/10 |
| `-O3 -fno-ipa-cp-clone` | **0/67**, and `NC7` **0/100** |
| `-O2 -fipa-cp-clone` | 19/35 |
| `seq_cst` on every atomic in the allocator | **55/100** — no effect |
| full fence at 4 acquire/release seams (each) | 48–54/100 — no effect |
| chunk recycle cache disabled | 13/30 — no effect |
| cross-thread batch `cap = 1` | **0/16**, 0/12 |
| batching kept, per-slot flush (no sort/merge/CAS-merge) | 5/16 — protocol innocent |
| pool exclusivity stress, 256 M ops, 3200 thread spawns | **0 violations** |

Note the non-monotonicity: reuse delayed ~0 (`cap = 1`) is clean, ~1024 frees
(default batching) fails, 2 M allocations (quarantine) is clean.  Both
extremes clean, middle fails.  That is consistent with the UAF reading — what
changes is only whether the freed `Packet`'s memory still looks valid at the
moment of the stale read — but it is not a monotone "reuse distance" effect
and should not be described as one.

## 7. Loose ends

- `TWOLEVEL` 0/8 is provisional and decides 2-level vs 3-level for the model.
- The STM throws `NodeNotFoundError` (a `std::domain_error`) out of the
  `Transaction` constructor uncaught, so an inconsistency it detects itself
  takes the process down.  Noted by the other session in `251edf60c`; still
  worth fixing independently of this bug.
- Whether the ILP32 fault is the same defect is now testable directly: apply
  the poison patch there and check `si_code` and the faulting register.

## 8. Refcount event tracing (`KAME_RC_TRACE`) — added 2026-08-24

The next step chosen over more modelling: instrument the `Packet` refcount
itself and let the reproducer name the guilty increment/decrement.  Hooks in
`kamepoolalloc/atomic_smart_ptr.h` (no-ops unless `-DKAME_RC_TRACE`), runtime
in `kamestm/tests/rc_trace.cpp`.

**Coverage.** Every strong-count change that goes through `local_shared_ptr`
(copy ctors, `reset()`), plus control-block birth via the `atomic_countable`
ctor.  `Packet`/`Payload` are intrusive `atomic_countable` and are never
installed in an `atomic_shared_ptr`, so their histories are **complete**.
`PacketWrapper` histories are **partial** (the atomic-side machinery — tag
drains, CAS transfers — is untraced).  Biased branches (`KAME_LSP_BIASED`)
are untraced; keep it off.

**Tripwires** (abort at the guilty call site, with history dump):

- `INC-FROM-ZERO`: copying a `local_shared_ptr` whose count is `0` **or
  ≥ 2^48** — the threshold matters because with the §3 poison patch a freed
  slot's count reads `0xBAADF00DBAADF00D`, not `0`.  This fires at the exact
  *stale copy* site, earlier than the poison `#GP`.
- `DEC-UNDERFLOW`: `reset()` on a count of `0` / poisoned — the exact *stale
  release* site.

Per-thread rings (64 rings × 16384 events, recycled round-robin — safe under
the reproducer's thread churn since events carry the real tid), TSC
sequencing on x86-64 (no shared cacheline on the record path).

**Build** (the §3 line plus two additions):

```bash
c++ -DKAMEPOOLALLOC_DYLIB -DA_NO_P1TREE -DKAME_RC_TRACE \
    -I kamestm/tests -I kamestm -I kamepoolalloc \
    -O3 -g -DNDEBUG -std=gnu++17 -include kamestm/tests/support_standalone.h \
    kamestm/tests/tmin_dynnode.cpp kamestm/tests/rc_trace.cpp \
    kamestm/tests/support_standalone.cpp kamestm/threadlocal.cpp \
    -o tmin_rct -Wl,-rpath,<build>/kamepoolalloc-tests \
    <build>/kamepoolalloc-tests/libkamepoolalloc.so -ldl
```

Keep the poisoned allocator `.so` from §3 — poison is what makes the stale
access *identifiable*; the thresholds make it *attributable*.

**Run**: `gdb --args ./tmin_rct 100 16 1250`, then either

1. a tripwire aborts first (the good case): the `abort()` backtrace **is** the
   guilty site; the stderr dump shows the object's whole inc/dec history with
   per-event `tid` and resolved sites; or
2. the poison `#GP` fires as in §1: then
   `call kame_rc_dump((const void*)$rsi)` — read the history for the freed
   `Packet`: look for the `DEC`/`DEAD` that has no matching owner, or the
   `DEAD → BORN` (slot recycled as a new Packet) followed by a `DEC` from a
   thread that should no longer hold it.  `call kame_rc_dump_recent(200)`
   prints cross-thread context around the fault.

**Perturbation caveat**: the hooks add a TLS ring store + (x86) a `rdtsc`
per refcount op.  §4's rule applies — if the fire rate collapses, raise the
thread count (16 → 24) rather than concluding anything.

What this decides: whether the premature zero is a *double release* (extra
`DEC` — its site named directly), a *missing increment* (history shows N
owners but N−1 `INC`s — the transfer that skipped its `INC` is the edge
between the last two events), or a *stale release onto a recycled slot*
(`DEAD → BORN → DEC` signature).  Each of the three points at a different
`bundle`/`bundle_subpacket` edge; §2's model can then be built around the
named edge instead of the whole recursion.

## 9. `KAME_RC_TRACE` result — the guilty site is named

Run on x86-64 / g++ 15.2 / 2 cores, 2026-08-24, using §8's build against the
§3 poisoned `.so`.  At 16 threads the tracing overhead suppressed the fault
(0/2 and 0/5); §8's rule applied, **24 threads** fires it.  `setarch -R`
(ASLR off) makes the raw site addresses directly resolvable with
`addr2line -e tmin_rct <addr - 0x555555554000>`.

The tripwire is what fires, not the poison `#GP` — exit (1), the good case.

```
kame_rc_trace: FATAL INC-FROM-ZERO **TRIPWIRE**
  obj=0x7ffff5de01e0  rc(before)=13451671603782742029  at 0x55555556635e
```

`13451671603782742029 == 0xBAADF00DBAADF00D`.  The `≥ 2^48` branch of the
double condition is what caught it — the `count==0`-only tripwire would have
slept through this, exactly as §8 predicted.

### The ordering (three threads, one object)

```
tid=3570406  BORN → INC → DEC → DEAD(unique) 1→0   site=0x555555562da3
tid=3570406  BORN, BORN                            site=0x555555566615   slot recycled
tid=3570417  INC/DEC ×2 → DEAD(unique) 1→0         site=0x555555562395
tid=3570408  INC-FROM-ZERO on the poisoned slot    site=0x55555556635e
```

Ledger for the object: `BORN 103 / DEAD 103 / INC 206 / DEC 204` — balanced,
so no class of decrement is unhooked and the history is trustworthy.

### Resolved sites

| addr | resolves to |
|---|---|
| `0x55555556635e` **(fatal)** | `local_shared_ptr<PacketList_>::reset()` `atomic_smart_ptr.h:763`, inlined into **`reverseLookupWithHint`, `transaction_impl.h:1720`** |
| `0x555555562395` | `fast_vector<local_shared_ptr<Packet>,1>::clear_fixed()` → `~PacketList_()`, `transaction.h:105` |
| `0x555555562da3` | `~local_weak_ptr<Linkage>` → `~PacketWrapper()`, `transaction.h:915` |
| `0x555555566615` | `local_shared_ptr<Packet>::operator=(&&)` → `reverseLookup`, `transaction_impl.h:1808` |
| `0x55555556632e` | `~local_shared_ptr<Packet>` → `reverseLookupWithHint`, `transaction_impl.h:1719` |
| `0x555555562ac1` | `forwardLookup`, `transaction_impl.h:1760` |

### What it says

`reset(Y*)` is `reset(); reset_unsafe(y)`.  The `INC` attributed to it comes
from its **argument**: `new PacketList( *(*foundpacket)->subpackets() )`
copy-constructs the list, and `PacketList_` derives from
`fast_vector<local_shared_ptr<PacketT>>`, so the copy runs a
`local_shared_ptr<Packet>` copy constructor **per element**.  The object's
last `DEAD` came from `~PacketList_()` releasing exactly such an element.

> **A `PacketList` still holds a `local_shared_ptr<Packet>` element pointing
> at a `Packet` that has already been destroyed.  The `copy_branch` clone at
> `transaction_impl.h:1720` copies that list, and the per-element copy
> increments a refcount in freed memory.**

```c
transaction_impl.h:1717-1721   (copy_branch block of reverseLookupWithHint)
    if(( *foundpacket)->subpackets()->m_serial != tr_serial) {
        *foundpacket = make_local_shared<Packet>( **foundpacket);          // 1719
        ( *foundpacket)->subpackets().reset(
            new PacketList( *( *foundpacket)->subpackets()));              // 1720  <-- stale read
        ( *foundpacket)->m_missing = ( *foundpacket)->m_missing || set_missing;
        ( *foundpacket)->subpackets()->m_serial = tr_serial;
```

Of §8's three scenarios this is the **third** — stale release onto a recycled
slot, signature `DEAD → BORN → DEAD → stale INC`.  Not a double `DEC`, not a
missing `INC`.  So §2's model wants the edge between this copy-branch clone
and whatever destroys a `PacketList` concurrently, **not** the whole
`bundle` recursion.  §1's `bundle()` crash at `transaction_impl.h:2733` is
the same object class observed later in the recursion.

### Status of this evidence

- One capture, symbolised.  A confirming capture with the same two sites is
  running; treat the attribution as strong-but-single until it lands.
- An earlier capture (ASLR on, `RCTP_HIT_24_1`) read `rc(before)=0` rather
  than poison, and its `1 → 0` transition was **absent** from the trace —
  same fault class, weaker instance; do not use it for attribution.
- Raw dump excerpt: `kamestm/tests/evidence/rc_trace_INC_FROM_ZERO.txt`.

### 9.1 Correction — §9's single-edge attribution is too strong

Four further captures (three plain, one under gdb) **do not confirm one
edge**.  Each run aborts at its own earliest *detected* anomaly, and those
differ:

| capture | tripwire | `rc(before)` | site |
|---|---|---|---|
| RCT2_3 | INC-FROM-ZERO | poison | `reverseLookupWithHint:1720` (PacketList clone) |
| RCT3_14 | DEC-UNDERFLOW | poison | `~PacketWrapper()` → `atomic_shared_ptr_base::deleter` |
| RCT3_22 | INC-FROM-ZERO | другой object's garbage | `Node::snapshot:2403` |
| RCT3_35 | DEC-UNDERFLOW | **0**, single tid | `~PacketList_()` → `clear_fixed`, `transaction.h:105` |
| RCTG_1 | INC-FROM-ZERO | 0 | `Node::snapshot:2423` |

(read "другой" as "another" — typo preserved rather than silently rewritten
in a record other people are reading.)

**The one repeat**: RCT3_22 and RCTG_1 both resolve through
`ScopedNegotiateLinkage::commit()` — but that is a line-table artefact,
`commit()` is `{ m_committed = true; }` and performs no refcount op.  The
real instruction is the adjacent
`snapshot.m_packet = scope->packet();` / `= *foundpacket;` at
`transaction_impl.h:2403` / `:2423`: a `local_shared_ptr<Packet>` copy-assign
that INCs a `Packet` reached **through the scoped view's wrapper**.  That is
§1's shape exactly — wrapper live, its `m_packet` already dead.

**Methodological caveat, and the reason §9 overreached**: once the heap is
corrupt the first tripwire is frequently a *secondary* access, not the root
release.  A single capture cannot distinguish them.  What is consistent
across all five is weaker but solid:

> An object is **reborn in a slot while a reference to the previous
> incarnation survives** (`DEAD → BORN → touch`, and in RCT3_35 a `DEC → BORN`
> with no intervening `DEAD` at all), and the surviving reference is always
> reached through the wrapper / list / scoped-view machinery of
> `snapshot` / `bundle` / `unbundle`.

RCT3_35 is the cleanest instance because it has no concurrency confound:
single tid, `rc(before) = 0` rather than poison, `BORN → DEAD(unique) 1→0 via
~PacketWrapper's chain → a second DEC from ~PacketList_()`.  That is a plain
**double release inside one thread** — scenario (1), not (3).  So §9's
"scenario (3), not (1)" claim is withdrawn; both shapes occur.

**Suggested change to the instrumentation** (for whoever owns it): aborting on
the first detected access maximises the chance of catching a secondary.
Recording anomalies and continuing — or gating the abort on the *first* one
per object per run and dumping all of them at exit — would let the earliest
anomaly be compared across runs rather than sampled one per run.

## 10. Next capture: dump the SOURCE LIST's history at the same abort

Written from the Mac session after auditing `fast_vector` (2026-08-24).

**`fast_vector` is clean.**  Copy assignment placement-news `T(r[i])` per
element (→ traced `INC`), moves transfer ownership without count changes,
`move_fixed_to_var` move-constructs then destroys the fixed slots exactly
once, `destroy()` destroys the active union member only.  (The union
discriminator had one historical inversion — see the `shrink_to_fit()`
comment — already fixed and unrelated to counting.)  So single-threaded
container accounting cannot produce "an element exists but its count does
not".  What remains is a **concurrent reader-copy vs writer-release on one
`PacketList`**: the §9 reader loaded the element's `m_ref`, the writer's
`reset()` ran `DEC → 1→0 → free`, the reader then `INC`'d the poisoned
count.  (`local_shared_ptr::reset()` nulls `m_ref`, so copying a *properly
destroyed* slot yields null, not a stale pointer — the reader must have
raced the release, or walked a list it should not reach.)  A list with a
concurrent writer is a list that is **shared when the code believes it is
private** — the `copy_branch` privacy argument (`m_serial != tr_serial`)
or a stale `*foundpacket` from the hint path are the candidate holes.

**The arbitrating evidence is already being recorded.**  `PacketList_` is
itself intrusive `atomic_countable` (`force_intrusive_ref`), so the LIST's
own `BORN/INC/DEC/DEAD` history with sites is captured by the same hooks.
`fast_vector` is `PacketList_`'s first base, so the list object address ==
the fast_vector address visible in the copy-loop frames.

At the §9 abort, additionally run:

```gdb
# frame: fast_vector<local_shared_ptr<Packet>,1>::operator=(const&) —
# `r` is the SOURCE list (== the PacketList_* the tracer keyed on):
frame <N-of-operator=>
call kame_rc_dump((const void *) &r)
# and the destination clone's list for contrast:
call kame_rc_dump((const void *) this)
```

Three outcomes, each decisive:

- **(A) source list shows `DEAD` before the fatal `INC`** — the reader is
  iterating a *freed* list (memory unreused, bytes intact).  The site of
  the list's last `DEC/DEAD` names who dropped it; the bug edge is a stale
  `*foundpacket` / hint that outlived the list's owner ⇒ model §2 around
  "lookup walks a chain whose interior list is released concurrently".
- **(B) source list alive, never shared (its `INC` history is all this
  thread)** — then the *element* was overwritten/reset concurrently by a
  writer that reached the same list through another path: the `m_serial`
  privacy check admitted a shared list as private.  Compare the list's
  `m_serial` (`print ((Transactional::PacketList_<...>*)&r)->m_serial`)
  against both threads' `tr_serial`.
- **(C) source list alive and INC'd from two threads** — direct proof the
  list is structurally shared while one side treats it as writable; the
  second thread's `INC` site names the aliasing edge.

No new code is needed for this capture.  If (B)/(C) point at the writer,
one further ring op (`OP_LIST_CLONE` at transaction_impl.h:1719-1721
recording `list, m_serial, tr_serial`) can be added, but only after the
free evidence above is in.

## 11. Instrumentation v2 — §9.1's two requests, implemented

§9.1 asked for (a) not aborting on the first *detected* access, so the
earliest anomaly per run can be compared across runs, and (b) a way to tell
a root release from a secondary access.  Both are in now; the default
behaviour is unchanged.

**(a) `KAME_RC_TRACE_ABORT=0` — record and continue.**  Default (unset or
`1`) is the old abort-with-dump, so §8/§10's gdb workflow is untouched.  With
`0`: full history dump for the **first** anomaly per object, a one-line note
for repeats, execution continues, and an `atexit` summary lists every
anomalous object with its full history.  A run can then be read as "these N
objects went wrong, in this order" instead of one sample per process.  (The
run may still die later on a poison dereference — everything above is
already on stderr by then.)

**(b) Destruction stack.**  `reset()` now brackets `deleter(pref)` with
`push_dtor`/`pop_dtor` (per-thread, 64 deep, depth counted past capacity so
truncation is visible).  Every anomaly report prints it:

```
  destruction stack (1 deep, innermost last):
    [0] destroying obj=0x1032c9b10  from 0x1001487ec (main+0x12c)
```

An empty stack means the anomaly is *not* reentrant — an independent access.
A non-empty stack means the anomaly happened **inside** the listed object's
destruction chain, which is exactly the discriminator §9.1's RCT3_35 needs:
`BORN → DEAD(unique) 1→0 → a second DEC from ~PacketList_()` is a double
release *within one chain* iff that second DEC reports the first object on
its stack.  Verified on a hand-built case (`Outer{local_shared_ptr<Inner>}`,
Inner's count poisoned, released from inside `~Outer`): tripwire fires at
`~local_shared_ptr<Inner>` with `[0] destroying obj=<the Outer>`.

Two notes for whoever writes the next smoke test: `atomic_countable`'s own
`assert(refcnt == 0)` is live in a `-DNDEBUG`-less build and fires before any
tripwire, and a **self-referential** type (`struct Q { local_shared_ptr<Q>
child; }`, i.e. the member declared while `Q` is incomplete) does not take
the intrusive path, so `q->refcnt` is not the counter in play.  Both cost a
detour here; the real `Packet`/`PacketList_` are complete at use and build
with `-DNDEBUG`, so neither affects the reproducer.

Suggested use for the next batch: run 10× with `KAME_RC_TRACE_ABORT=0` and
tabulate the **first** anomaly of each run (object, op, site, dtor-stack
depth).  If the earliest anomaly is stable across runs, §9's attribution can
be re-made on a distribution rather than one capture; if it is not, that is
itself the answer, and §2's model should target the invariant rather than an
edge.

### 11.1 That batch, run — the earliest anomaly is NOT stable

Ubuntu, x86-64 / g++ 15.2, `KAME_RC_TRACE_ABORT=0`, 9 runs of
`./tmin_rct 100 24 950`, ASLR off.

**First: continue-mode and the §3 poison are incompatible.**  With the
poisoned `.so`, runs 1-2 gave `rc=139 anomalies=0` — the tripwires only see
refcount *operations*, while the poison `#GP` is a plain dereference, so the
process dies before any anomaly is recorded and continue-mode never gets to
continue.  The batch below therefore uses the **unpoisoned** `.so`.  Anyone
running §11 should pair `ABORT=0` with an unpoisoned allocator and keep the
poison for `ABORT=1`.

| run | rc | anomalies | first anomaly | site resolves to |
|---|---|---|---|---|
| 1,2,4,9 | 139 | 0 | — | — |
| 5 | 134 | 0 | — | — |
| 6 | 255 | 0 | — | — (`failed objcnt`) |
| 3 | 139 | 3 | DEC-UNDERFLOW, `rc=0`, dtor-stack **1 deep** | `local_weak_ptr<Linkage>::reset()` `atomic_smart_ptr.h:911` → `~PacketWrapper()` |
| 7 | 139 | 2 | DEC-UNDERFLOW, `rc=0` | **same site**, `atomic_smart_ptr.h:911` |
| 8 | 139 | 2 | INC-FROM-ZERO, `rc=0` | `local_shared_ptr<Packet>::operator=(&&)` → `reverseLookup:1808` |

Two of three name the same site, which looks like the stability §11 hoped
for — but it does not survive the ledger check:

| capture | BORN | DEAD | INC | DEC | balanced? |
|---|---|---|---|---|---|
| §9 abort-mode captures (Packet / PacketList) | 103 | 103 | 206 | 204 | yes |
| run 8 object | **22** | **6** | **19** | **27** | **no** |

An object whose ledger is that far out is one the tracer is not seeing all of,
so its anomaly cannot be attributed.  And the 2-of-3 repeat is worse than
unbalanced: `atomic_smart_ptr.h:911` is `local_weak_ptr<Linkage>::reset()`,
i.e. a **weak** count on a `Linkage` — precisely the "atomic-side machinery is
untraced" class §8 excludes.  So that repeat is most likely an artefact of
partial coverage, not a finding.

**Also line-table noise in the new dtor stack**: run 3's frame reads
`destroying obj=0x7fffe124beb0 from 0x55555556f997`, and that site resolves to
`XThreadLocal<StampKind>::operator*()` / `~ScopedOpKind()` inlined in
`bundle:3060` — not a refcounted destruction at all.  Same class of
mis-attribution as `ScopedNegotiateLinkage::commit()` in §9.1.  The stack
*depth* (0 vs non-zero) is trustworthy; the *site string* is not, at `-O3`.

**Conclusion, by §11's own criterion**: the earliest anomaly is not stable, so
**§2's model should target the invariant, not an edge** —

> no slot may be reborn while a reference to its previous incarnation
> survives

with `Packet` refcount reaching zero under a live `PacketWrapper` (§1) as the
concrete instance to check first.

**Two changes that would make the next batch decisive:**

1. **Gate anomalies on ledger-complete classes.**  Only `Packet` and
   `PacketList_` are intrusive-and-`local_shared_ptr`-only, so only they have
   complete histories.  Either suppress anomalies on other types or print the
   object's `BORN/DEAD/INC/DEC` balance beside every report so a reader can
   discount an unbalanced one at a glance.
2. **Separate the two counters for `atomic_weakable` types.**  If the tracer
   keys on the control block but hooks strong and weak operations onto one
   counter, a `Linkage` will show spurious underflow whichever way the real
   counts move.  Worth checking before any `atomic_smart_ptr.h:911` report is
   believed.

### 11.2 Instrumentation v3 — and a correction to §11.1's ledger test

§11.1 asked for two things and flagged one suspicion.  All three are settled;
one of them **invalidates the ledger check as §11.1 applied it**.

**The ledger test does not mean what it was read to mean in continue-mode.**
The rings are circular *and* recycled across threads (slot = round-robin
`% MAX_RINGS`), so once a slot has taken `RING` (16384) writes, older events
are silently **evicted**.  In abort-mode the process usually dies before any
wrap — §9's `103/103/206/204` is a genuinely complete history — but a
continue-mode run of 100×24×950 wraps many times, so run 8's
`BORN 22 / DEAD 6 / INC 19 / DEC 27` is **the expected shape of an evicted
history, not evidence that the tracer misses ops**.  v3 makes this explicit:
every dump now states whether anything was evicted.

```
  ledger: strong BORN 0 / DEAD 0 / INC 2048 / DEC 2048   weak wINC 0 / ...
  ** 1 ring(s) have wrapped (40010 events recorded) -- older events were
     EVICTED, so an unbalanced ledger here says nothing about tracer coverage **
```
or, when nothing was lost:
```
  no ring has wrapped -- this history is complete
```
Run 8's anomaly therefore cannot be *confirmed* by its ledger, but neither is
it discredited by it.  For a ledger-based judgement, use abort-mode (or raise
`RING_LOG2`).

**The strong/weak suspicion is refuted, by construction.**  `atomic_smart_ptr.h:911`
is `local_weak_ptr<T>::reset()`, which touches `weak_refcnt` **only**; every
v1/v2 hook was on `refcnt` **only** (`atomic_countable` ctor,
`local_shared_ptr` copy ctors, `local_shared_ptr::reset`).  The tracer never
observed a weak counter, so it cannot have conflated the two — a `:911`
report could only ever have been line-table noise from an inlined strong op,
which is what §11.1 already suspected.  v3 nevertheless **traces the weak
side explicitly** as distinct ops (`wINC` / `wDEC` / `wDEAD`, three sites:
the ctor-from-shared, the weak copy ctor, `reset()`), so `PacketWrapper`
histories — it holds a `local_weak_ptr<Linkage>` — are now interpretable
instead of absent, and the ledger prints the two counters separately.

**Type identification (§11.1's item 1).**  Every event now carries its `T`
(from `__PRETTY_FUNCTION__` of a function template — one string literal per
instantiation, no runtime cost), and reports print it:

```
kame_rc_trace: FATAL DEC-UNDERFLOW  obj=0x...  rc(before)=0xBAADF00D...  type=Inner  at ...
```

so a report on a partially-covered type is discountable at a glance.  For
hard gating, `KAME_RC_TRACE_TYPES=<substr>` reports only anomalies whose type
label contains `<substr>` (recording is unaffected): `…=Packet` keeps
`Packet` / `PacketList_` / `PacketWrapper` and drops everything else.
Verified: `…=Inner` → report; `…=Nothing` → silent.

**Also worth stating**, since §9.1 and §11.1 each lost time to it: at `-O3`
every `site` string in this tool is `__builtin_return_address(0)` fed to
`addr2line`, and inlining makes that land on a *neighbouring* source line
often enough that no site string should be trusted without a second capture
agreeing.  What is reliable: the **op** class, the **object address**, the
**tid**, the **type label**, the dtor-stack **depth**, and the eviction flag.

**Suggested next batch** (unpoisoned `.so`, `ABORT=0`, `KAME_RC_TRACE_TYPES=Packet`):
tabulate first anomaly per run as before, but now recording type + eviction
state, and treat only `no ring has wrapped` runs as ledger-checkable.  If the
result is again unstable, that is the third independent confirmation and the
invariant-level model of §11.1 is the answer; no further instrumentation is
warranted.

## 12. Instrumentation v4 — crash-proof capture, and the first releaser in one line

Written after the §11.2 batch report: 30 v3 runs / 0 tripwires turned out to
be rate variation (an interleaved v2/v3 comparison had v3 firing on its
second run), and the capture that did land **lost its history to a peer
thread's SIGSEGV mid-dump** — header intact, ledger and events gone.

That capture nevertheless settled §11.1, via the one field v3 added:

```
FATAL DEC-UNDERFLOW  obj=0x7ffff5460640  rc(before)=0
  type=Transactional::Node<LongNode>::Packet      ← ledger-complete class
  destruction stack: (empty — not a reentrant release)
  at …  → local_weak_ptr<Linkage>::reset()  atomic_smart_ptr.h:937
         → ~PacketWrapper()  transaction.h:915
```

The op is on a **`Packet`**, not on a `Linkage` weak count — as §11.2's
structural argument required (the tracer never observes `weak_refcnt`).  The
`:911`/`:937` site is `~PacketWrapper()` destroying its `local_weak_ptr<Linkage>`
and its `local_shared_ptr<Packet> m_packet` interleaved by the optimiser, so
the Packet's `DEC` return address lands in the weak pointer's inlined code:
the same line-table noise as §9.1, now disambiguated by the type label.
Three captures (§11.1 runs 3 and 7, plus this one) at that site therefore read:

> `~PacketWrapper()` releases its `m_packet` and that `Packet`'s count is
> **already zero** — the wrapper is the SECOND releaser, and with an empty
> destruction stack it is an independent release, not a reentrant one.

Scenario (1), a double release, with one party named.  What is missing is the
**first** releaser, which needs a dump that survives.

### v4 changes

1. **Raw crash-proof sink.**  Every dump now goes out twice: first raw —
   plain `write(2)` of unsymbolised lines to a file, **newest event first** —
   then the symbolised version to stderr.  Newest-first is deliberate: if the
   process dies after three lines, those three are the ones nearest the fault.
   File: `$KAME_RC_TRACE_FILE`, else `rc_trace.<pid>.log`.  Nothing on the raw
   path calls `dladdr` (the loader-lock symbol walk that made the old window
   wide) or buffered stdio.

2. **`RC-PRIOR-RELEASE-FAST` — the first releaser, in O(1), emitted first.**
   A first attempt put this line after a ring scan and measured the failure:
   with a peer crash racing the dump, *only the anomaly header* reached the
   file, because `collect_()` walks 64×16384 slots before printing anything.
   So the datum is now maintained **at record time** — a 4096-slot
   direct-mapped cache of the last `DEC`/`DEAD` per object address, updated on
   every release, read in O(1) at anomaly time.  Verified against a deliberate
   peer-SIGSEGV race: in the worst run only two lines were written, and they
   were the header and the prior-release line.

   ```
   RC-ANOMALY #1 obj=0x… op=DEC-UNDERFLOW rc_before=0 tid=… site=0x… type=Pk dtor_depth=0
   RC-PRIOR-RELEASE-FAST obj=0x… op=DEC rc_before=2 tid=… seq=… site=0x… type=Pk
   ```

   (The cache is racy by construction — a torn slot is possible when two
   threads release into the same bucket.  The full `RC-EV` history that
   follows is the arbiter; this is the copy that survives.)

3. Full history retained as `RC-HIST` / `RC-EV` / `RC-END` (newest-first, with
   the eviction count), plus `RC-DTOR` frames; ring scan now bounded by the
   number of rings ever handed out.

### How to read the next capture

```bash
KAME_RC_TRACE_FILE=$PWD/rc.$$.log gdb --args ./tmin_rct 100 24 950
grep -E 'RC-ANOMALY|PRIOR-RELEASE-FAST' rc.*.log
```

`RC-ANOMALY` names the second releaser (expected: `~PacketWrapper()` on a
`Packet`), `RC-PRIOR-RELEASE-FAST` names the **first** — its `tid` says
whether the two releases are on the same thread (a genuine double release in
one call chain) or different ones (two owners both believing they hold the
last reference), and its `site` — treated as a hint, per §11.2, until a second
capture agrees — says through which edge.  Those two lines close the
attribution; everything after them is corroboration.

### 11.3 v4 batch — 40 runs, 9 captures: double release of a `Packet`, mostly same-thread

Ubuntu x86-64 / g++ 15.2, v4 tracer, **poisoned** `.so`, abort-mode,
`./tmin_v4 100 24 950`, ASLR off, `KAME_RC_TRACE_FILE` per run.
40 runs → 9 captures.  The two-stage raw sink worked: every capture kept both
`RC-ANOMALY` and `RC-PRIOR-RELEASE-FAST`, including runs that died mid-dump.

Using only the fields §11.2 marks trustworthy:

| field | result |
|---|---|
| `type` | **9/9 `Transactional::Node<LongNode>::Packet`** |
| prior release is the true freeing op (`DEAD(unique)`, `rc_before=1`) | 7/9 |
| prior release on the **same tid** as the anomaly | **6/9** |
| `dtor_depth` | 0 ×5, 1 ×1, 2 ×1, 4 ×2 |
| innermost site symbol | **scatters** (see below) |

The other 2/9 report `prior = DEC 2→1` on a different tid — there the fast
cache's last-traced DEC is simply not the freeing op, so they neither
corroborate nor contradict.

**Conclusion**

> A `Packet` is released twice, and in 6 of 9 captures the second release is
> on the **same thread** as the genuine `DEAD(unique) 1→0`.  That is a double
> release inside one call chain, not two owners racing for the last reference.

**The edge still cannot be named.** Innermost symbols across the nine:
`fast_vector::clear_fixed` ×3, `bundle` ×2, `local_weak_ptr<Linkage>::reset`
×2, `atomic_shared_ptr_base::deleter`, `~local_shared_ptr<Packet>`.  An
earlier reading of this batch (`reverseLookup:1808` → `~PacketList_`) was
drawn from one capture and does **not** survive the other eight — the exact
failure mode §11.2 predicts for site strings at `-O3`.  Recorded here so it is
not re-derived.

So §2's model should encode the invariant, now stated more precisely than in
§11.1:

> **No `Packet` may be released after its reference count has already reached
> zero** — and the model must admit the same-thread case, since that is the
> majority shape, rather than only the cross-thread race.

**Instrumentation status: sufficient.**  v4 answers what it was built to
answer.  Naming the source edge would need something site-strings cannot give
at `-O3` — a captured call chain (e.g. a few frames of return addresses
recorded at `DEAD` and at the anomaly, resolved offline) — and that is only
worth building if the model cannot be closed without it.

### 12.1 v5 — captured call chains, because one address cannot name the edge

§11.3 closed the attribution of the *second* releaser but not the **edge**:
the innermost `site` symbol scattered across nine captures (`clear_fixed` ×3,
`bundle` ×2, `local_weak_ptr::reset` ×2, …) because at `-O3` a return address
lands wherever the optimiser put the inlined code.  §11.3's own prescription —
"a few return addresses recorded at `DEAD` and at the anomaly, resolved
offline" — is now implemented.

**What it adds** (`KAME_RC_TRACE_CHAIN=1`, default OFF so §11.3's fire-rate
baseline stays comparable):

```
RC-ANOMALY #1 obj=0x… op=DEC-UNDERFLOW rc_before=0 tid=1 site=0x… type=Pk dtor_depth=0
RC-CHAIN-ANOM  obj=0x… frames=4 0x…76c 0x…804 0x…964 0x…
RC-PRIOR-RELEASE-FAST obj=0x… op=DEAD(unique) rc_before=1 tid=1 seq=4 site=0x… type=Pk
RC-CHAIN-PRIOR obj=0x… frames=3 0x…710 0x…958 0x…
```

A bounded frame-pointer walk (≤6 frames): `fp[0]` = caller's fp, `fp[1]` =
return address, true on x86-64 and aarch64 alike.  Captured on **release ops
only** (`DEC`/`DEAD`/`DEAD(unique)`, stored in the O(1) last-release cache
beside the event) and once **at the anomaly**.  Written on the raw path, so
both chains survive a peer crash — re-verified against the deliberate
peer-SIGSEGV race: 3/3 runs kept all four lines above.

**Requires `-fno-omit-frame-pointer`** on the reproducer; without it the walk
has nothing to follow.  It is written to fail SHORT rather than wild — each
candidate frame must be pointer-aligned, strictly above the previous one, and
within 1 MiB of it — so a garbage chain truncates instead of dereferencing
nonsense.  Re-check the fire rate after enabling either flag (§4's rule);
frame pointers alone perturb codegen.

**Verified to do the thing site strings could not.**  On a synthetic double
release with distinct real callers, compared against runtime function
addresses printed by the test itself:

| chain | frame[0] | frame[1] | frame[2] |
|---|---|---|---|
| `ANOM` | shared/inlined `reset()` body — **ambiguous, as §11.3 describes** | **`second_releaser`** | `main` |
| `PRIOR` | same shared body | **`main`** (the intermediate wrappers were *tail-called*, so they left no frames) | — |

frame[0] is exactly the useless address; **frames[1…] are the genuine
distinct callers, and the two chains diverge there.**  That divergence is the
disambiguation §11.3 asked for.

**Two caveats from that experiment**, both real:
- **Tail calls collapse frames.**  `outer→middle→inner` appeared as a single
  frame because clang turned the wrappers into jumps.  A chain can therefore
  *skip* intermediate functions — read it as "these frames are on the stack",
  never "these are all the frames".
- Resolve with `addr2line -e <binary> -f -C -i <addrs>`.  The **`-i`** matters:
  it expands the inline stack at each address, which is what a bare symbol
  lookup could not do — worth using on the old §11.3 site addresses too.

**Suggested run**

```bash
c++ … -fno-omit-frame-pointer -DKAME_RC_TRACE …            # rebuild
KAME_RC_TRACE_CHAIN=1 KAME_RC_TRACE_FILE=$PWD/rc.$$.log ./tmin_v5 100 24 950
grep -E 'RC-ANOMALY|RC-CHAIN|PRIOR-RELEASE-FAST' rc.*.log
addr2line -e ./tmin_v5 -f -C -i <the frames>
```

The question to put to the two resolved chains is narrow: **where do they
diverge?**  Their common prefix is the shared release code; the first frame
at which `ANOM` and `PRIOR` differ is the pair of edges that both released the
same `Packet`.  With §11.3's finding that 6/9 are same-tid, the expectation is
two different points in **one** thread's `snapshot`/`bundle` walk — and if the
divergent frames are stable across two or three captures, that names the edge
and §2's model can be built around it instead of the whole recursion.

### 11.4 v5 chains — the edge, named and stable

Ubuntu x86-64 / g++ 15.2, v5 tracer, `-fno-omit-frame-pointer`, poisoned
`.so`, `KAME_RC_TRACE_CHAIN=1`, ASLR off.

**§4 rate check first.**  Chain capture suppresses the fault: interleaved on
one binary, `CHAIN=1` gave **0/16** and chain-off **2/16**.  Not significant
alone (p≈0.48) but the wrong direction, so the chain arm was compensated with
threads per §4 — 24thr 1/11, 32thr 3/11, 40thr 2/11.  Run the chain arm at
32-40 threads, not 24.

**Four of seven captures are frame-for-frame identical** (`rc5b_24_3`,
`32_16`, `32_19`, `32_34`; all `DEC-UNDERFLOW`, all `dtor_depth=4`):

```
PRIOR  (the release that frees it, DEAD(unique) 1->0), read bottom-up:
  Node::bundle()                       transaction_impl.h:2893
    ScopedNegotiateLinkage::set_view() transaction_negotiation.h:877
    scoped_atomic_view::assign_from_local()  atomic_smart_ptr.h:1314
    scoped_atomic_view::release_()           atomic_smart_ptr.h:1613
    ~PacketWrapper()                   transaction.h:915
    ~Packet()                          transaction.h:252
    ~PacketList_()/clear_fixed         transaction.h:105 / fast_vector.h:236
    local_shared_ptr<Packet>::reset()  atomic_smart_ptr.h:1728   <-- frees

ANOM   (the double release, dtor_depth=4):
  ~PacketWrapper()
    ~Packet() -> ~PacketList_() -> ~Packet() -> ~PacketList_()
    local_shared_ptr<Packet>::reset()  atomic_smart_ptr.h:1739   <-- underflows
```

The other three captures differ (`dtor_depth` 0, 1, 2), so this is the
dominant shape, not the only one.

**What it says.**  Both releases are frames of **one recursive destruction
cascade**, not two unrelated code paths.  `set_view()` in `bundle()` drops the
old `PacketWrapper`; its destruction recurses `~PacketWrapper -> ~Packet ->
~PacketList_ -> ~Packet -> ...` through the packet graph, and a `Packet`
reachable at **two different depths** of that graph is released once at each
depth — the second time after its count already reached zero.

The graph is a DAG by construction in this reproducer: `p2` is hard-linked
under both the thread-local `p1` and the shared `gn2` (§2).  Refcounting
should absorb that — two list slots ⇒ count 2 ⇒ two DECs ⇒ one free — so the
implication is that **one of the two references was installed without an
increment**.

**This also explains §11.3's scatter.**  Every innermost symbol there —
`clear_fixed`, `~PacketWrapper`'s inlined weak reset, `~local_shared_ptr<Packet>`,
`bundle` — is a *frame of this same cascade*, sampled at whatever depth that
run aborted.  They were never different edges.  §11.3's "the edge cannot be
named" is therefore superseded: it could not be named *from single site
strings*, which is what §11.2 predicted; the chains name it.

**For §2's model**: the edge is `bundle()`'s `set_view()` releasing a
`PacketWrapper` whose packet DAG contains a shared `Packet`.  The invariant of
§11.3 still holds and is what to check; this narrows *where* to check it.

**Caveats.**  The 6-frame bound truncates the ANOM chain at `~PacketWrapper`,
so it is not proven that both releases belong to the *same* cascade instance
rather than two instances of the same shape.  Tail calls fold frames, so a
chain says "these frames were on the stack", not "these are all of them".

### 12.2 v6 — slot identity: testing §11.4's "installed without an increment"

§11.4's conclusion has an unexamined step.  "A Packet reachable at two
depths, released once at each" presumes **two holders**; the alternative —
the *same* holder reached twice — was not excluded by the chains, since the
two cascades' frames were truncated before their roots could be compared.

One deduction closes half of that gap by construction:
`local_shared_ptr::reset()` **nulls `m_ref`**, so destroying the same holder
twice is a **no-op** on the second pass, not an underflow.  A genuine
`DEC-UNDERFLOW` therefore implies either

- a **different slot** — two owners, one count: §11.4's missing increment, or
- a slot whose memory was **rewritten** with the stale value — a bitwise
  duplication of the holder (memcpy'd storage, torn move, relocated
  container bytes), which is a different bug in a different place.

v6 records the discriminator: every event now carries **`slot`** — the
address of the `local_shared_ptr`/`local_weak_ptr` performing the op
(`this`), null for `BORN`.  It appears in `RC-ANOMALY`,
`RC-PRIOR-RELEASE-FAST`, and every `RC-EV` line.  The release-time cache also
snapshots the destruction stack's object addresses, emitted as
`RC-PRIOR-DTOR`, so the PRIOR release's containment can be compared with the
anomaly's `RC-DTOR` frames.

**How to read the next capture** (three questions, in order):

1. `ANOM.slot == PRIOR.slot`?  Equal ⇒ the holder's memory was rewritten
   between the releases — look for who copies holder bytes, not for a missing
   increment.  Different ⇒ two owners, one count, §11.4 confirmed.
2. Does `ANOM.slot` appear as the `slot` of any `INC`/`BORN` in the (complete)
   history?  **If not, its ownership was installed invisibly** — by a move
   whose source was not emptied, or a bitwise copy — and the visible events
   bracket where: it happened after the last event whose slot chain accounts
   for all owners.
3. Which container holds each slot?  Compare the slot addresses against the
   `RC-DTOR` / `RC-PRIOR-DTOR` object addresses: a slot lying within a dying
   `PacketList_`'s storage identifies the list AND the element index
   (`(slot - list_base - header) / sizeof(local_shared_ptr)`), turning "two
   depths of the cascade" into two named list elements.

Verified on a synthetic invisible install (holder forged by `memcpy` beside
one visible copy): the anomaly's slot is the forged holder, the prior's slot
is the first owner, the visible `INC`'s slot is the third — and the forged
slot appears in **no** `INC`/`BORN` line, which is exactly the signature to
look for.  Crash-race re-verified after the change (2/2 runs kept all raw
lines).

If question 2 comes back "all slots accounted for", then the count itself was
wrong earlier than the installs — and the invariant for §2 stays as §11.3
stated it; the slots will have localised where not to look.

### 11.5 v6 slot identity — Q1 answered, Q2 answered, Q3 at n=1

Ubuntu x86-64 / g++ 15.2, v6, `-fno-omit-frame-pointer`, poisoned `.so`,
`KAME_RC_TRACE_CHAIN=1`, 32/40 threads, `100 <thr> 700`.
**60 runs → 7 captures.**

| capture | op | dtor_depth | RC-EV kept | Q1 slots | tid | Q2 |
|---|---|---|---|---|---|---|
| rc6_3 | DEC-UNDERFLOW | 0 | 16 | DIFF | cross | yes |
| rc6_18 | DEC-UNDERFLOW | 0 | **0** | DIFF | same | n/a |
| rc6_26 | INC-FROM-ZERO | 0 | 1 | DIFF | cross | n/a |
| rc6_28 | INC-FROM-ZERO | 0 | **0** | DIFF | cross | n/a |
| rc6_34 | DEC-UNDERFLOW | **1** | 322 | DIFF | cross | yes |
| rc6_59 | DEC-UNDERFLOW | 0 | **0** | DIFF | same | n/a |
| rc6_60 | INC-FROM-ZERO | 0 | 77 | DIFF | same | yes |

**Q1 — `ANOM.slot != PRIOR.slot` in 7/7.**  Always two holders.  The
holder-write-back branch (bitwise copy / torn move / container realloc) is
**excluded**; §11.4's two-holder reading survives.

**Q2 — 3/3 `yes`** among captures with a usable history: the ANOM slot always
has a matching `INC`.  The four "no" readings were **empty dumps**, not
invisible installs — a distinction worth stating, since an earlier draft of
this table scored them as evidence.  So the reference is installed *visibly
and with an increment*: this is the "count broken before installation" branch,
not the invisible-install one.

**Q3 — one capture only** (`rc6_34`, the only `dtor_depth>0`):

```
RC-DTOR [0] obj=0x7fffd09637f0   = a CASInfo, in fast_vector<CASInfo,32>::clear_fixed()
ANOM.slot  = 0x7fffd0963800      = that CASInfo + 16  ==  CASInfo::old_wrapper
PRIOR.slot = 0x7fffdf7f7330      = a scoped_atomic_view<PacketWrapper> local in
                                   Node::snapshot()  transaction_impl.h:2241
```

```cpp
struct CASInfo {                                   // transaction.h:1386
    local_shared_ptr<Linkage>          linkage;      // +0
    scoped_atomic_view<PacketWrapper>  old_wrapper;  // holder at +16
    local_shared_ptr<PacketWrapper>    new_wrapper;
};
```

Both releasers are `scoped_atomic_view<PacketWrapper>` instances — one owned by
a `CASInfo` in the CAS list, one a local in `snapshot()`.  That is the type
that carries a `+1` with **zero atomic operations** through
`assign_from_local()` and the `local_shared_ptr&&` move-in ctor ("caller is
responsible … we do NOT verify").  The same `+1` is handed
`parent_scope → CASInfo` at `transaction_impl.h:2178-2180` ("Extract
scoped_atomic_view from parent_scope into CASInfo … kept alive by the
CASInfo's view") and back out at `:3329` ("Move CASInfo's scoped_atomic_view
into ScopedNeg").  If either transfer leaves the source still owning, both
ends release — which is exactly "two holders, one count".

**Candidate, not conclusion.**  n=1, and it is the `depth=1` cross-thread
shape, not §11.4's `depth=4` same-thread cascade.

**A caution about all three shape claims.**  The dominant shape has moved with
every tracer version: v4 6/9 **same-thread**, v5 4/7 at **depth=4**, v6 5/7 at
**depth=0** with 4/7 cross-thread.  Three versions, three dominant shapes.
That is the instrumentation steering which shape gets sampled, not three
findings.  Weight the *invariant* (§11.3) over any shape.

**What limits the next batch**: 4 of 7 captures kept ≤1 `RC-EV` line, so Q2/Q3
are answerable only on the minority whose dump survives.  Writing the history
**before** the header, or capping it to the last ~40 events, would convert
those into answerable captures and reach a second Q3 data point far faster
than raising the run count.

### 12.3 v7 — recent-history survives the crash; and the three named transfers audited

**v7 tracer change (§11.5's ask).**  The reason 4 of 7 captures kept ≤1
`RC-EV` line is that the full history requires the 64×16384 ring scan, and
the peers are already corrupting while it runs.  v7 keeps the last 16 events
per object in a record-time direct-mapped cache (same hash as the
prior-release cache; racy on bucket collision, full scan stays the arbiter)
and emits them as `RC-RECENT`/`RC-R` lines **immediately after the header and
prior-release line, before anything that scans**.  16 events cover Q2 (the
matching `INC` with its slot) and the `DEAD → BORN` rebirth signature.  The
slow post-scan dump is additionally capped to the newest 40 events
(`KAME_RC_TRACE_FULL=1` restores all).  Crash-race result, honestly: of the
final 3 runs, **2 kept header + prior + the full `RC-R` block; 1 kept
nothing at all** — zero lines including the header, which is consistent with
the peer's SIGSEGV landing before the tripwire's first `write(2)` (a capture
that never started, which no output ordering can fix), but it caps what this
mechanism can promise: it protects captures that begin, it cannot create
ones that don't.  (The pushed commit message says "3/3"; this paragraph is
the correction.)

**The three §11.5-named transfer points, audited (Mac, branch tip).**  If
Q3's shape is real, one of the zero-atomic transfers must leave both ends
owning.  The named ones do not:

- `transaction_impl.h:2178` — `cas_infos->emplace_back(…,
  r.parent_scope->consume_scoped_view(), …)`: `consume` empties the source
  view; `emplace_back` move-constructs the member (move ctor nulls the
  temporary).  One owner at every step.
- `transaction_impl.h:3329` — `std::move(it->old_wrapper)` into the
  ScopedNeg **move-in ctor**: the ctor is `noexcept`, has no early return,
  and consumes unconditionally at `m_view = std::move(from)`
  (`transaction_negotiation.h`, view-variant); `_negotiate()` runs *before*
  the move and cannot skip it.  A mid-loop `DISTURBED` return leaves
  already-moved `CASInfo`s empty and unmoved ones owning — both correct.
- `scoped_atomic_view` move ctor / move assign / `assign_from_local` —
  audited in §10/§12: source nulled in every branch.

**Consequence**: if two holders end up sharing one `+1` (Q2 says the second
holder's `INC` is *visible*, so the count went wrong elsewhere), the defect
is **not** in the CASInfo hand-off pair; the remaining unaudited seam on that
path is where the `+1` **entered** `parent_scope` in the first place —
`set_view()` / the scoped-acquire path — or an earlier release that the
invariant (§11.3) will catch regardless of shape.

**On the shape instability** (v4 same-thread 6/9 → v5 depth-4 4/7 → v6
depth-0 5/7): agreed, and it is worth saying *why* the steering is expected —
every tracer version changes the per-op cost and therefore the interleaving
distribution; the captures are importance-sampled by the instrument itself.
The invariant is the only version-independent statement.  The next batch's
`RC-RECENT` blocks should be read for Q2/Q3 answers, not for a new dominant
shape.

### 12.4 v7 batch — Q3 confirmed at n=2, and the stale-view sequence

Ubuntu x86-64 / g++ 15.2, v7, chains on, 32/40 threads, `100 <thr> 700`.
**60 runs → 8 captures.**  v7 is markedly slower per run (~3 min vs ~40 s for
v6), which is itself worth recording: the instrument importance-samples the
interleaving harder at each version, so shape statistics from this batch are
not comparable across versions — read `RC-RECENT` for Q2/Q3 only, per §12.3.

| capture | op | depth | tid | rec | Q2 | dtor off |
|---|---|---|---|---|---|---|
| rc7_4 | DEC-UNDERFLOW | **1** | cross | 17 | 0 | **+16** |
| rc7_17 | DEC-UNDERFLOW | 0 | same | 17 | 1 | — |
| rc7_19 | DEC-UNDERFLOW | 4 | same | 17 | 0 | (slot not inside) |
| rc7_31 | INC-FROM-ZERO | 0 | cross | 51 | 3 | — |
| rc7_34 | INC-FROM-ZERO | 0 | cross | 17 | 1 | — |
| rc7_40 | DEC-UNDERFLOW | 0 | same | 17 | 1 | — |
| rc7_46 | DEC-UNDERFLOW | 0 | same | 17 | 1 | — |
| rc7_51 | INC-FROM-ZERO | 0 | cross | 6 | 2 | — |

**Q1: `DIFF` 8/8** — with §11.5's 7/7 that is **15/15**, two holders every time.

**Q3: confirmed, n=2.**  `rc7_4` is structurally identical to `rc6_34`:

```
RC-DTOR [0] obj=0x7fffcec1fa70 → CASInfo::~CASInfo()  transaction.h:1386
                                 in fast_vector<CASInfo,32>::clear_fixed()
ANOM.slot  = 0x7fffcec1fa80    = CASInfo + 16 = CASInfo::old_wrapper
PRIOR.slot = 0x7fffdaff6330    = ~scoped_atomic_view<PacketWrapper>()
                                 in Node::snapshot()  transaction_impl.h:2241
```

Across 120 runs these are the **only two captures with `dtor_depth > 0`**, and
both name the same member.  Both cross-thread, both with the poison as
`rc_before`.

**The sequence, from `rc7_4`'s recent block (chronological, last three):**

```
BORN          rc->1    tid=...251  slot=(nil)            <- the address is RE-BORN
DEAD(unique)  1->0     tid=...253  slot=…6330            <- snapshot():2241's view frees it
DEC-UNDERFLOW poison   tid=...251  slot=CASInfo+16       <- old_wrapper releases
```

`CASInfo::old_wrapper` releases an object **born after its own view was
established**, which then died at the hands of a *different* view.  That is
`DEAD → BORN → stale release`: the view **outlived its target**, rather than
sharing a count with it.  `Q2inc=0` for this capture is consistent — the
CASInfo slot has no `INC` inside the 16-event window because its view was
installed before the window opened.

**Consequence for §12.3's audit.**  This does not contradict the finding that
`consume_scoped_view()` → `emplace_back` → move-in-ctor are each clean.  It
says the `CASInfo` view is the **second releaser and the victim**: it holds a
`+1` that was not, or is no longer, backed by the target.  The remaining
unaudited seam §12.3 names — **where the `+1` entered `parent_scope`
(`set_view()` / the scoped-acquire path)** — is where a view could come to
hold a reference that does not keep its target alive.  That is now the single
most specific place to look, and it is reachable by reading code rather than
by more runs.

**Caveat**: n=2 for Q3, and both are the cross-thread `depth=1` shape.  The
`depth=4` same-thread cascade of §11.4 is *not* represented among the
container-knowable captures, so it is not established that both shapes share
this mechanism.

### 12.5 §12.4's slot identification withdrawn — the layout probe settles it

`rc_layout_probe.cpp` run on Ubuntu x86-64 / g++ 15.2 (needs
`-fno-access-control` there; the nested types are private and clang was more
permissive).  Measured, matching the Mac:

```
sizeof(PacketWrapper) = 40      sizeof(CASInfo) = 40      <- the trap
sizeof(scoped_atomic_view<PW>) = 24
PW::m_bundledBy @ +8   PW::m_packet @ +16   PW::m_reverse_index @ +24
CI::linkage     @ +0   CI::old_wrapper @ +8  CI::new_wrapper @ +32
```

`CASInfo + 16` lands **inside** `old_wrapper` (at its `m_pref`), an address no
hook passes as a slot.  So `type=Packet, slot=X+16` has one self-consistent
reading: **X is a `PacketWrapper` and the releaser is its `m_packet`.**
Re-derived:

| capture | dtor obj | anom slot | off | reading |
|---|---|---|---|---|
| rc6_34 | 0x7fffd09637f0 | 0x7fffd0963800 | **+16** | `PacketWrapper::m_packet` |
| rc7_4 | 0x7fffcec1fa70 | 0x7fffcec1fa80 | **+16** | `PacketWrapper::m_packet` |

**§12.4's "second releaser = `CASInfo::old_wrapper`" is withdrawn**, and with
it the inference that the CAS-list hand-off was implicated.  The `CASInfo`
reading came from symbolising the `RC-DTOR` *site* string, which §11.2 already
classifies as a hint; the two 40-byte structs made a wrong hint look
self-consistent.  The level-crossing was an attribution error on this side,
not a tracer inconsistency.

**What it becomes instead is more consistent, not less.**  The second releaser
is `~PacketWrapper()` releasing its `m_packet` — the same statement as §12's
three earlier captures and as §1's crash, where a live wrapper's `m_packet`
pointed at freed memory.  All five container-knowable observations now say one
thing, and the stale `+1` to hunt is the one a `PacketWrapper` should have held
on its `Packet`.

**Dual-keyed markers verified here**: the probe's `Packet` history carries
`VADOPT` and `VMOVE (src_slot=…)`, so a `Packet`-typed anomaly will now show
the custody chain of the wrapper that held it — which is what §13.1's reading
instruction needs and what v8 could not produce.

## 13. TLA+ scope-token model — the protocol is exonerated; hunt the departure

`kamestm/tests/tlaplus/atomic_shared_ptr_scopetoken.tla` + 6 cfgs.  Models
exactly the layer §11.5 converged on: a `+1` ownership token moving between
the atomic linkage word, `local_shared_ptr` holders, `scoped_atomic_view`
(TagHeld / Owned, with the promote split into its two real atomic steps),
and `CASInfo` parking (`:2178` park / `:3329` unpark), plus both `release_()`
branches and the linkage CAS.  No Packet tree: the tree only *drives* a
sequence of transfers, and the model generates all transfer sequences
nondeterministically — a superset of every walk.  Rebirth (`MakeNew` into a
freed identity) is included, so §11.3's invariant is checked in its token
form:

> per object `rc >= holders` (equality when no transfer has occurred),
> globally `Σrc + link_tag = holders + pins`, a freed object has **nothing**
> pointing at it, and no release ever fires against `rc = 0`.

`CONSTANT TagXfer` selects the one thing v1 cut: `FALSE` = CAS requires a
quiescent tag word; `TRUE` = the CAS **transfers** outstanding tags into the
displaced object's `rc` (Layer 1's `CASTransfer`) and the tag-release paths
gain their word-changed → global-decrement branch.  `TRUE` is the
**composition of the L1 drain machinery with the Owned-view layer** — the
seam neither Layer 1 (no Owned mode) nor v1 covered.  Backings are fungible
under transfer (a stolen word-tag and a transferred rc-unit swap roles),
which is why the per-object equality legitimately weakens to `>=`.

**Results (2 threads, 2 objects, 2 lsp + 2 views + 1 park per thread):**

| cfg | verdict |
|---|---|
| `none` (quiescent) | **PASS, exhaustive** — 11.7M generated / 748k distinct / depth 21 |
| `none_xfer` (transfer composition) | **PASS, exhaustive** — 111.7M / 7.2M / depth 27, 2m42s |
| `setview_noempty` ×2, `unpark_noempty` ×2 | **Conservation violated** within seconds, all four |

The bug knobs each break one documented contract the way a real defect
would (`set_view` adopting without emptying the source lsp; unpark leaving
the CASInfo owning) — so the detector provably fires on the fault class.
Two self-corrections during construction, recorded for honesty: the global
sum first double-counted "promoting" (it consumes a holder AND a pending
pin backing); and the first `TypeOK` rc bound counted holders only, tripping
on a transferred pin at 90M states — the bound was wrong, not the protocol.

**What this establishes.**  The token protocol **as documented** — including
the transfer composition — cannot produce §1's state at this scope, and the
scope is adequate for this fault class (both knobs violate within depth
~10; the fungibility argument is scope-independent).  Combined with §11.5
(Q2: the second holder's INC is visible), the defect is therefore a
**departure of the implementation from one of these finitely many
contracts**, not an emergent property of the protocol design.

**The checklist this reduces the hunt to** (model action → C++ → audit状態):

| action | C++ | audited? |
|---|---|---|
| `SetView` release-then-adopt | `set_view()` tn.h:877 + `assign_from_local` | transfer half clean (§12.3); release-ordering vs concurrent access **open** |
| `ParkToCAS` / `UnparkFromCAS` | `:2178` / `:3329` + move-in ctor | **clean** (§12.3) |
| `ViewPromoteAdd/Release` | scoped promote / `release_tagheld_zeroreset_` | L1-verified in isolation; **composition with Owned in-code: open** |
| `ViewReleaseOwned` | `release_()` Owned `fetch_sub` | hook-verified semantics |
| `LspCopy/Reset` | copy ctor / `reset()` | v1-hooked, §10 `fast_vector` clean |
| `CommitCAS` + step4 accounting | `compareAndSet_impl_` | L1-verified in isolation (dossier 🟡7 noted the fetch_sub(2) consume as unexamined detail) |
| wrapper ctor/dtor `m_packet` hand-off | `~PacketWrapper()` etc. | **open** — the cross-level seam (wrapper-count error becomes a Packet double-DEC) |

The two **open** rows plus a surviving v7 `RC-RECENT` capture are the
remaining moves.  A capture that names the deviating op empirically ends it;
failing that, the checklist is short enough to close by inspection.

### 13.1 The §12.4 seam, audited by hand — and v8: association markers

**Bulk TagHeld release (`release_tagheld_zeroreset_` + `release_tag_ref_`)
audited across interleavings.**  §13's model idealised the tag release as a
per-unit decrement; the implementation is a BULK protocol (pre-pay the other
`rcnt−1` pinners into global, then drain the whole word).  Hand-checked
cases, each netting exactly −1 for the releaser: (A) unchanged word, full
drain; (B) new acquires between read and drain (min() leaves them); (C) a
concurrent drainer shrank the word (excess-undo `fetch_sub(added−drained)`);
(D) pointer swapped with CASTransfer (full global undo); (E) two concurrent
zeroresets on the same word (second sees rcnt=0, undoes its own pre-pay +
releases via global); (F) swap-with-transfer between pre-pay and drain.
All conserve, GIVEN `release_tag_ref_`'s excess-undo delete-check
(`d93fc7de`).  This closes §13's "promote/release composition" open row by
inspection and discharges the dossier's 🟡7/🟡8 concern for these callers.

**The scoped-CAS Owned/TagHeld discrimination is present and correct.**
`compareAndSet_impl_`'s success path selects `sub = 2` for TagHeld (m_ref
share + tag share) and for Owned+RETAIN (m_ref share + the view's old +1,
since the view is reassigned to newr), `sub = 1` for plain Owned — and its
comment records a FIXED past bug of exactly the class we are hunting
("without this, OLD pref's refcnt leaks +1 per Owned-RETAIN call ... appears
at low LOCAL_REF_CAPACITY").  step4 = +T for SCOPED balances in both the
ABSORBED and DRAINED cases.  One stale comment found: the OldrT dispatch
header still says "SCOPED ... step4 = +(T−1)" — the implementation uses +T
with the sub-side discrimination; the block comment at the step-4 site is
the accurate one.  (This resolves the §12.3-era S6b comment contradiction:
it was doc drift, not two sub-cases.)

So every path §12.4 pointed at reads clean, three formal layers pass, and
yet the stale view exists at n=2.  The remaining blind spot is EXACTLY the
part with no events: zero-atomic transfers and view establishment are
count-neutral, so the tracer never sees them, and Q2inc=0 because the
16-event window opens long after the association was created.

**v8 therefore adds count-neutral association markers:**

- `VADOPT` — an Owned association is created: the lsp move-in ctor,
  `assign_from_local` (= `set_view`), the acquire-ctor promote, and the
  RETAIN_NEWR reassignment inside `compareAndSet_impl_`.  **The ADOPT site
  distinguishes the two §12.4 candidates directly** (promote vs
  set_view/lsp hand-off).
- `VMOVE` — the association transfers between view slots (move ctor / move
  assign; `consume_scoped_view` and the CASInfo park/unpark ride on these).
  The event's `rc_before` field carries the SOURCE slot address, so the
  custody chain `ADOPT → VMOVE → ... → CASInfo → release` is
  reconstructable from a single capture.  Smoke-verified: a three-hop
  chain reads back exactly, each hop naming its source slot.

Also in v8: the branch's tracer now hooks `scoped_atomic_view`'s own
release paths (`release_()` Owned branch and `release_tagheld_zeroreset_`'s
global-fallback branch) with the same threshold tripwires, `slot` = the
view address.  §11.5/§12.4's captures show view-address slots, so the
Ubuntu binary evidently carries equivalent local hooks — with this commit
the branch is self-sufficient and the two tracers agree on coverage.

**Reading the next capture**: for the underflowing object, the `RC-R` /
`RC-EV` history now contains its views' `VADOPT` (with site: promote vs
set_view) and every `VMOVE` hop.  A stale CASInfo view's chain ending in an
ADOPT whose object address was REBORN in between is the §12.4 sequence with
the origin attached — that names the deviating code path, and the hunt
ends there.  (Rate caveat as in §12.4: marker events add record-path work
on snapshot/bundle hot paths; do not compare shape statistics with earlier
tracer versions.)

### 13.2 v9 — dual-keyed markers, layout ground truth, and a tracer bug that
### invalidated every no-trace build of this branch

Response to the structural report ("markers keyed on the view's target,
every anomaly on a Packet").  The critique is accepted and implemented, and
chasing it uncovered two things bigger than the request.

**(1) The keying gap is real; markers were nonetheless alive.**  On a Mac
arm64 build of `tmin_dynnode` (20 rounds, 8 threads), the new op tally shows
VADOPT 1.1M / VMOVE 5.2M — the markers fire constantly; their absence from
the captures was purely per-object keying, as diagnosed.  Every anomaly
header now carries a receipt so this is never ambiguous again:
`RC-MARKERS-ALIVE adopt=<N> vmove=<N> (run totals)` — zero there means the
markers did not fire in that binary; nonzero means absence from a history is
a keying fact.  `KAME_RC_TRACE_STATS=1` prints the full per-op tally at exit
of any run, anomalous or not.

**(2) Option 2 implemented — markers are dual-keyed onto the payload.**
Every VADOPT/VMOVE now records twice: keyed on the view's target (the
wrapper) as before, and keyed on `PacketWrapper::m_packet.get()` (the
payload Packet), same slot, same site.  Wiring: `KAME_RC_EVT_TV` macro +
`rc_secondary_probe_` detection template in `atomic_smart_ptr.h` (layer-0
stays generic; intrusive-only), `PacketWrapper::rc_trace_secondary_()` in
`transaction.h` (tracer builds only; non-virtual member = no layout change).
End-to-end check lives in `rc_layout_probe.cpp`: the Packet's own history
reads

```
  BORN ... INC ...
  VADOPT (view association)  slot=0x16fd52770
  VMOVE  (...)               slot=0x16fd52758 src_slot=0x16fd52770
```

Interpretive rule: views of Packet do not exist in the STM, so **any
VADOPT/VMOVE in a Packet-keyed history is a dual-keyed record about a
wrapper that held that Packet** — the custody chain requested, in the same
capture, keyed by the address the anomaly is keyed by (so it also lands in
`RC-RECENT`'s O(1) crash-surviving cache).  The pretty dump now renders
markers with `slot=`/`src_slot=` instead of bogus rc arithmetic, and the
ledger counts them in their own `markers` column (they were previously
mislabelled `tripwires`).

**(3) Layout ground truth — the n=2 slot attribution does not survive it.**
`rc_layout_probe.cpp` (committed; run it per toolchain) prints, on
arm64/clang and by LP64 rules everywhere:

| thing | value |
|---|---|
| `CASInfo::linkage` | +0 |
| `CASInfo::old_wrapper` | **+8** (view = m_asp,m_pref,flags = 24 B) |
| `CASInfo::new_wrapper` | +32 |
| `PacketWrapper::m_packet` | **+16** |
| `sizeof(CASInfo)` = `sizeof(PacketWrapper)` | **40 = 40** |
| `sizeof(Packet)` | 32 |

So "ANOM.slot = CASInfo + 16 == old_wrapper" (§11.5, §12.4) is refuted:
old_wrapper's `this` is CASInfo+8, and CASInfo+16 is old_wrapper's
*interior* (`m_pref`), which no canonical hook ever passes as a slot.  The
reading that IS self-consistent with `type=Packet, slot=X+16`: **X is a
PacketWrapper and the slot is its `m_packet` member — the underflowing
release is `~PacketWrapper` (or wrapper packet reassignment) releasing the
Packet one level down.**  The equal sizes (40=40) make base-address
misidentification easy; the disambiguator is address class — CASInfos live
in `fast_vector<CASInfo,32>` (stack / Transaction frame), wrappers on the
heap.  Note the n=2 dtor-stack entry `obj=0x7fff...` is a *stack* address:
canonical v8+ `KAME_RC_DTOR_PUSH` only ever pushes the heap object being
deleted, so that entry came from local instrumentation — worth re-deriving
both rc6_34/rc7_4 attributions against the probe's numbers (run it under
g++ 15.2 to confirm x86-64 agrees).  If the reinterpretation holds, §12.4's
"CASInfo::old_wrapper is the second releaser" becomes "a PacketWrapper's
`m_packet` is the second releaser", and the stale +1 to hunt is the one the
wrapper's `m_packet` believed it still had — which is exactly what the
dual-keyed markers + the wrapper-era events in the Packet's history will
now show directly.

**(4) A tracer bug of mine, found by this turn's cross-checks, fixed here:
both `local_weak_ptr` weak-INC hooks had the `weak_refcnt.fetch_add(1)`
INSIDE the `KAME_RC_EVT_T` argument list.**  In a non-`KAME_RC_TRACE` build
the macro expands to `((void)0)` and the increment vanished: weak handles
never counted, the control block was freed under live strong references,
and every no-trace build of this branch died deterministically (Mac: 8/8)
on the `~gref_weakable_` `refcnt==0` assert — a use-after-free with
`-DNDEBUG`.  Master is unaffected (6/6 clean with identical test sources);
**your captures are unaffected** (every §8-recipe build defines
`KAME_RC_TRACE`, and with the macro live the behaviour was correct); but
any control run built from this branch WITHOUT the define was invalid
before this commit.  Both sites now hoist the fetch_add out of the macro,
and a whole-file audit confirms no other side-effectful expression sits in
any trace-macro argument.  (Meta-lesson filed: instrumentation macros that
compile out must never wrap the operation they observe.)

**Batch recommendation:** restart the 60-run batch from THIS commit rather
than letting it finish on 91a5bbf8b — a capture without dual-keys re-poses
the same question this section answers, and the markers-alive receipt +
slot-aware rendering only exist here.  Expect faster ring eviction (markers
are now ~35–40% of event traffic, and dual-keying adds a second record for
each): the `RC-RECENT`/`RC-EV` windows stay the right place to read, and
the eviction banner tells you when the ledger is not coverage.

### 13.3 Mac-runnable verification: GenMC test 11 mechanizes the §13.1 audit

The user asked for a verification test that runs on the Mac.  Timing-luck
reproduction is the wrong tool on arm64 (520 clean runs historically), so
the Mac's role is exhaustive interleaving exploration instead:
`cds_atomic_shared_ptr/cds_test_zeroreset.c` (GenMC test 11) models
`release_tagheld_zeroreset_` — the §12.4 seam — directly from the C++, and
mechanizes the §13.1 hand audit that was previously "closed by
inspection":

| scenario | covers (§13.1) | result |
|---|---|---|
| SCEN 1 zeroreset vs `load_shared_` | A, B, C | ✅ 43 executions |
| SCEN 2 dual zeroreset | E | ✅ 18 |
| SCEN 3 zeroreset vs `swap` | D, F | ✅ 89 |
| SCEN 4 three-way composition | all | ✅ 1,110,415 executions, 160 s |

Every global decrement funnels through an asserted helper, so the
DEC-UNDERFLOW tripwire the tracer checks at runtime is an exhaustive
assert here; double destroy and touch-after-destroy are asserted too.

**The teeth, and a modeling drift found on the way:** writing SCEN 3
against `cds_test_swap.c`'s swapper produced an immediate Safety violation
— because test 4's simplified swap transfers the tag shares AFTER the CAS,
while the real `lsp::swap(asp&)` pre-pays BEFORE it (acquire → pre-pay
rcnt−1 → CAS).  Against a concurrent TagHeld releaser the after-CAS order
lets the releaser's "pointer changed ⇒ my +1 is global" `fetch_sub(1)` hit
the implicit m_ref reference and destroy the object while the swapper
still holds it — the premature-destroy/UAF class we are hunting, found by
GenMC in milliseconds.  The faithful order passes exhaustively; the
unfaithful one is preserved as a bug knob (`-DSWAP_TRANSFER_AFTER_CAS`,
wired into `make run-test11` and REQUIRED to violate), so the test
provably has teeth against exactly this fault class.  Test 4 itself was
never wrong-in-effect (none of its threads hold a tag, so its transfer
never fires) — it now carries a comment saying so; test 7's swapper was
already faithful.  Also fixed: test 9's header still described the retired
`step4=+(T−1)` SCOPED protocol its own body no longer uses (same doc
drift as S6b).

Consequence for the hunt: the bulk-release/CASTransfer algebra is now
model-checked, not just audited — under RC11, with these actors, the
protocol conserves.  Combined with §13's TLA+ layer and tests 1–10, the
remaining habitat for the defect keeps shrinking toward (a) an
implementation path NOT equivalent to these models (the §13.2 layout
reinterpretation points at wrapper teardown, i.e. who was holding the
wrapper), or (b) a composition the models still do not contain (bundle
multi-linkage sequences).  `make run-test11` runs scenarios 1–3 + knob in
seconds on either machine; `make run-test11-full` adds SCEN 4.

### 13.4 Forensic poison — the freed memory itself names its killer
### (and the old poison had a blind spot exactly where it mattered)

User's idea, implemented: instead of a constant magic, the poison now
CARRIES INFORMATION.  Build the allocator with `-DKAME_POISON_FORENSIC`
(replaces the §3 hand-applied scratch patch; everything is inside the
ifdef, production builds unchanged — default-off ctest 18/18, flag-on
ctest 18/18, tmin 3×8-thread clean, no false anomalies):

- On every small-pool free (16..4096 B, the same window as §3), the block
  is filled with poison, but its FIRST word — where an intrusive refcnt
  sits, i.e. what the tripwires read — gets a token:
  `0xBAAD(16) | free-record counter(32) | 0x8000(16)`.  The low 16 bits
  absorb stale count operations, so the counter stays decodable and the
  decoder reports the DRIFT = how many stale ops already hit the block.
- The counter indexes a 2^18-record ring inside the allocator:
  `{ptr, size, tid, tsc, 4-frame call chain of the free}`.  A wrapped-out
  record is reported as such, never misattributed.
- `rc_trace` resolves `kame_poison_decode` via dlsym (binary and .so need
  not agree on flags) and the anomaly RAW phase prints, right under the
  header:

```
RC-ANOMALY #1 obj=0x110040080 op=DEC-UNDERFLOW ... rc_before=13451407662026227712 ...
RC-FREEREC freed_ptr=0x110040080 (=obj) size=64 free_tid=1 tsc=... drift=+0
           frames=4 <free call chain — addr2line -f -C -i>
```

Q3's question — WHO freed the thing the stale reference targets — is now
answered unconditionally at the anomaly, immune to tracer-ring eviction
and to `g_lastrel`/`g_recent` bucket theft: the block itself carries the
key, for as long as it stays free.

**The discovery that fell out (worth more than the feature): the pool's
owner-thread L1 freelist stores its next pointer IN the freed slot's
FIRST 8 bytes** (`freelist_push`, allocator_prv.h) — the exact word the
refcnt occupies and the §3 poison claimed.  Two consequences for every
capture to date:

1. **Plain poison had a blind spot in the most likely window.**  An
   L1-listed block's word 0 holds a HEAP POINTER, not poison — typically
   < 2^48, so a stale count op on a *freshly freed* block (the hottest
   reuse window) sailed under the tripwire threshold and was never
   reported.  Poison was only reliably present after the block drained to
   the bitmap.
2. **A stale strong-count op on an L1-listed block CORRUPTS THE FREELIST
   LINK** — the crash then happens later, in the owner's drain walk
   (`release_dll_chunks_for_thread`), far from the culprit.  Demonstrated
   here first-hand: the v11 smoke's two stale fetch_subs produced exactly
   that delayed teardown SIGSEGV.  Some of the §1-era "random"
   allocator-internal crash shapes may be this second-order effect rather
   than direct UAF derefs.

Under the flag, the link moves to the SECOND word (`kame_slot_link_()`;
all twelve in-block link accesses routed through it, and the remaining
`reinterpret_cast<char **>` casts in the pool are grep-audited to be
head-cell casts, never slot derefs).  So with forensic poison: the token
occupies word 0 for the block's ENTIRE freed lifetime (L1 → drain →
bitmap → realloc), the tripwire can no longer miss the fresh-free window,
and stale strong-count ops can no longer corrupt freelists.  (Stale WEAK
ops land on word 1 = the relocated link; unchanged exposure, noted.)

**Ubuntu action**: rebuild the poisoned `.so` from this commit with
`-DKAME_POISON_FORENSIC -fno-omit-frame-pointer` and drop the §3 scratch
patch.  Coverage is strictly wider than the old poison (fresh-free window
now included), every capture gains the RC-FREEREC culprit line, and the
n=3 capture's first releaser will be named even if it happened megabytes
of events ago.

### 13.5 Forensic poison on Ubuntu — the blind spot was real, and every catch is a PRIMARY event

Rebuilt the `.so` from `9c527ab4c` with `-DKAME_POISON_FORENSIC
-fno-omit-frame-pointer`, §3 scratch patch dropped; test built with
`-DKAME_RC_TRACE -fno-omit-frame-pointer` against it.  60 runs, 40 threads.

**The blind spot was costing us most of the signal.**

| tracer / poison | hits |
|---|---|
| v6 (constant poison) | 7 / 60 |
| v7 | 8 / 60 |
| v9 | 10 / 51 |
| **v10 = v9 + forensic poison** | **19 / 51 (37 %)** |

Three times v6's yield from the same reproducer, which is what §13.4 predicts
if the fresh-free window — where word 0 held an L1 freelist pointer under
2^48 — was previously invisible to the tripwire.  It also means the *silence*
in every earlier batch was partly instrumentation, not absence: "no anomaly"
runs in §11.3/§11.5/§12.4 cannot be read as clean.

**`drift=+0` in 14/14 captures carrying `RC-FREEREC`.**  Every catch is the
FIRST stale op on that block.  This is the discrimination no previous version
could make: earlier batches could not tell a primary event from a downstream
consequence of one, and §11.3's site scatter was read under that ambiguity.

**Who freed it — now answered unconditionally**, independent of ring
survival and of the `+16` offset inference §12.5 retracted:

| free-side chain tail | captures |
|---|---|
| `local_weak_ptr<Linkage>::reset()` → `~PacketWrapper()` | 6 / 14 |
| `fast_vector<local_shared_ptr<Packet>,1>::clear_fixed()` → `~PacketList_()` | 6 / 14 |
| `Snapshot<…>` / `local_shared_ptr<…>` | 2 / 14 |

**A reading withdrawn before it propagates.**  The first capture (`rcA_2`) had
BOTH the free and the stale release running through `~PacketWrapper()` →
`local_shared_ptr<Packet>::reset()`, with the stale side reached from
`Transaction::commit()` (`transaction.h:2616/2624`) — i.e. two distinct
`PacketWrapper`s each releasing the same `Packet`.  Across all 14 that shape
holds in **1** (partially 2).  It is a real capture, not a population.  Same
failure mode as §12.4's `CASInfo`: a single capture read as a mechanism.

**What survives at n=14**, on the fields §11.2 admits plus the self-identifying
token:

> Every anomaly is a `Packet`, is the FIRST stale op on its block
> (`drift=+0`), has two distinct holder slots (Q1 15/15), and the release that
> freed it is a destructor — `~PacketWrapper()` or `~PacketList_()` in equal
> measure — while the stale release arrives from an unrelated context.

That is §11.3's invariant with the primary-vs-secondary ambiguity removed,
which is the part that was missing.  It still does not name one edge, and the
free-side split 6/6 between two different destructors argues that it may not
be one edge.

### 13.6 The free-side 6/6 split is a lottery, not a second edge — and the
### habitat is now: plain lsp<Packet> algebra on shared state

Digesting §13.5 from the Mac side, one push-back and one narrowing.

**Push-back: the 6/6 destructor split carries (almost) no information
about the bug edge.**  The free-side chain names the LAST LEGITIMATE
holder — whoever happens to hold the final accounted unit when the count
(wrongly early) reaches zero.  A live Packet is legitimately held by its
parent list's `lsp<Packet>` entry, by wrapper(s') `m_packet`, and by
transients; if one unit is silently consumed upstream, which legitimate
holder performs the 1→0 DEC afterwards is a teardown-order lottery over
that population.  `~PacketWrapper()` 6 / `~PacketList_()` 6 / other 2 is
what a SINGLE upstream edge would also produce.  The constant across all
evidence is the STALE side — five container-knowable observations, §12.5
included, all say `~PacketWrapper()` releasing `m_packet` — so the
single-edge hypothesis is alive; it lives on the stale wrapper's share,
and the free-side table should not be read against it.

**Narrowing (structural, grep-verified on the branch): a Packet's strong
count is touched by NOTHING but `local_shared_ptr<Packet>`
copy/move/reset.**  The only `atomic_shared_ptr` in the tree is
`Linkage : asp<PacketWrapper>`; there is no `asp<Packet>` and no
`scoped_atomic_view<Packet>`.  Every mechanism exonerated so far — tags,
views, zeroreset, scoped CAS, park/unpark (GenMC tests 1–11, the
scope-token TLA+) — operates on the WRAPPER/Linkage layer and cannot
touch a Packet unit.  What remains is ordinary shared-pointer algebra
made unsound only by a broken immutability contract:

1. **Torn lsp copy** — an `lsp<Packet>` copy whose SOURCE is concurrently
   written (copy is load-then-INC, not atomic): suspects are every copy
   from shared state — `snapshot.m_packet = scope->packet()`
   (transaction_impl.h:2396/2402/2455/2495), PacketList clone entries
   (`reverseLookupWithHint` — RCT2_3's INC-FROM-ZERO already fired
   there), `reverseLookup` returns.
2. **Write into published state** — the writer side of the same race:
   `newwrapper->packet() = subpacket_new` (:1444),
   `newsubwrapper->packet() = *pit` (:1561), and any PacketList entry
   assignment — sound only while the wrapper/list is provably
   unpublished.  `PacketWrapper::packet()` returning a NON-const ref is
   what makes this class expressible at all.
3. **`fast_vector<lsp<Packet>>` lifecycle** under cross-thread
   publication (§10 audited the single-threaded semantics only).

`drift=+0` adds a hard fact here: between the legitimate free and the
stale op, NOTHING touched the block — no reallocation (a BORN would have
replaced the token), no earlier stale op.  The stale wrapper simply held
its `m_packet` across the entire freed window.  So pool double-hand-out
is effectively excluded for these 14, and the loss happened strictly
BEFORE the free.

**Age is already in your data.**  `Ev.seq` and `kame_freerec.tsc` are the
same rdtsc clock on x86, so `anomaly.seq − RC-FREEREC.tsc` gives each
capture's free→stale-op age; the tracer now prints it directly
(`age_tsc=` in RC-FREEREC).  The distribution is the cheap next
discriminator: tight-µs ⇒ the stale wrapper was racing the teardown that
freed the Packet (favors torn-copy/write-race suspects); long/bimodal ⇒ a
parked long-lived wrapper (favors a custody hand-off).  Computable for
the existing 14 without a rerun.

**Next capture reading** (v9+v10 build already produces everything): in
the underflowing Packet's history, pair every INC with its DEC by slot.
The defect is the DEC whose slot is NOT the slot that INC'd that unit —
with Q2 (the stale wrapper's INC is visible) that pair is the edge, and
the VADOPT/VMOVE custody on the same history names who was holding the
wrapper when it happened.

**arm64 control, now instrumentation-robust (Mac, 2026-08-25).**  §13.5
established that the old poison's silence cannot be read as clean — so the
historical "arm64 never reproduces" needed re-basing on the blind-spot-free
build.  Done: 200 runs × 40 threads, `-O3` on both the forensic `.so`
(Release + `KAME_POISON_FORENSIC`) and the test, tracer v9+v10+age —
**200/200 pass, zero tripwires** (the anomaly log was never created).
Against Ubuntu's 37 %/run on the same tracer generation, P(0/200) under
that rate is ~10⁻⁴⁰; even a 1.5 %/run arm64 rate is rejected at 95 %.
The fault needs something x86-64/g++ provides — the §4 `-O3` /
`-fno-ipa-cp-clone` codegen switch remains the strongest lever we have on
that axis — and arm64's weaker memory ordering does NOT summon it, which
argues against a plain missing-barrier reading and toward codegen
(store/load elision, cloning) or allocator address-pattern timing.

### 13.7 The age distribution — computed from the existing captures, no re-run

§13.6's observation that `Ev.seq` and `kame_freerec.tsc` share the x86 `rdtsc`
clock makes `free → stale op` computable from the §13.5 captures directly.
Host: i5-7500, TSC base **3.40 GHz** (invariant TSC; the 3701 MHz in
`/proc/cpuinfo` is the current core clock, not the TSC rate — the conversion
below uses 3.4).

| capture | ticks | age |
|---|---|---|
| rcA_42 | 258 | 0.08 µs |
| rcA_35 | 290 | 0.09 µs |
| rcA_54 | 532 | 0.16 µs |
| rcA_27 | 998 | 0.29 µs |
| rcA_10 | 1 108 | 0.33 µs |
| rcA_38 | 1 138 | 0.33 µs |
| rcA_32 | 2 372 | 0.70 µs |
| rcA_29 | 2 796 | 0.82 µs |
| rcA_51 | 3 436 | 1.01 µs |
| rcA_49 | 3 960 | 1.16 µs |
| rcA_22 | 5 870 | 1.73 µs |
| rcA_25 | 5 932 | 1.74 µs |
| rcA_2 | 6 036 | 1.78 µs |
| rcA_48 | 14 506 | 4.27 µs |
| rcA_47 | 462 048 | **135.9 µs** |

**13 of 15 under 2 µs, median ≈ 1.0 µs.**  By §13.6's discriminator that is
the **torn-copy class — the stale op racing the same teardown** — not a
long-parked wrapper handed around a custody chain.  The distribution is
single-peaked with one clear outlier (rcA_47, 136 µs) and one intermediate
(rcA_48, 4.3 µs); it is not bimodal.

Reading this against §13.6's three remaining habitats, all of which are plain
`lsp<Packet>` algebra since no `asp<Packet>` or `view<Packet>` exists in the
tree:

1. **torn `lsp` copy** — consistent with the whole distribution;
2. **writes to published state** (`newwrapper->packet() = …`, the non-const
   accessor) — also sub-µs, not separable by age alone;
3. **`fast_vector<lsp<Packet>>` lifecycle** — would need the copy to race a
   container operation, again sub-µs.

So age narrows to "racing the teardown" and rules the custody class *out* for
13/15, but does not separate (1)–(3) from each other.  What would: the
slot-pair audit §13.6 prescribes — in a `Packet`'s history, find the `DEC`
that consumes a unit its own slot never `INC`ed.

A methodological note, since it cost a wrong table above: the `RC-ANOMALY`
header carries no `seq`, so an extraction that falls back to
`RC-PRIOR-RELEASE-FAST` measures `DEAD → free` deleter latency (400–3 500
ticks here) rather than the age, and comes out **negative**.  Take the `seq`
from the tripwire's own `RC-EV`/`RC-R` line.

### 13.8 Reading the three habitats — habitat 2 eliminated; what the model should say

Per §13.7 the loss is strictly before the free and inside a ~1 µs teardown
window, so the remaining discrimination is cheaper by reading than by
measuring.  §13.6's three habitats, read:

**Habitat 2 — writes to published state through the non-const `packet()`
accessor — ELIMINATED.**  There are exactly two such writes in the tree:

```
transaction_impl.h:1442  auto newwrapper = make_local_shared<PacketWrapper>(…);
                  :1444  newwrapper->packet() = subpacket_new;      // not yet published
                  :1448  if(!scope.compareAndSetWithHint(newwrapper, …)) …  // publishes here

transaction_impl.h:1560  newsubwrapper = make_local_shared<PacketWrapper>(…);
                  :1561  newsubwrapper->packet() = *pit;            // not yet published
```

Both write into a wrapper created two lines earlier and published only by a
later CAS.  The non-const accessor makes the class *expressible* — worth
tightening on general principle — but neither instance is a write to published
state.

The adjacent write at `:1445` (`packet->subpackets()->back() = subpacket_new`,
which is *not* rolled back when the CAS at `:1448` fails) is also safe:
`packet` comes from `reverseLookup(tr.m_packet, true, tr.m_serial, true)` at
`:1387` — `copy_branch=true`, so it is this transaction's private clone, and a
failed CAS discards the whole branch on retry.

**Habitat 1 — torn `lsp<Packet>` copy — narrowed, not eliminated.**  The four
`snapshot.m_packet = scope->packet()` sites copy from a reference *into* the
`PacketWrapper` the scoped view holds.  Given habitat 2's result — published
wrappers are immutable, the only writes to `m_packet` happen pre-publish —
that source field is stable while the wrapper lives, and the view is what keeps
it alive.  So a tear there requires the **wrapper** to die under the view,
which is the wrapper-layer machinery already exonerated by GenMC 1–11 and the
scope-token TLA+.  This habitat survives only where the copy source is *not* a
published wrapper: `reverseLookup`'s returned reference and the `PacketList`
clone (`*foundpacket = make_local_shared<Packet>(**foundpacket)`), which point
into the packet tree rather than into a wrapper.

**Habitat 3 — `fast_vector<lsp<Packet>>` lifecycle — untouched.**  §10 audited
it under single-thread semantics only.

So the live surface after reading is: **copies whose source is a slot in the
packet tree (not in a wrapper), racing a concurrent structural mutation of that
tree.**

### What §2's model should encode

Version-independent, and it does not need the edge:

> A `Packet`'s strong count is pure `local_shared_ptr<Packet>` copy/reset
> algebra — there is no `atomic_shared_ptr<Packet>` and no
> `scoped_atomic_view<Packet>` anywhere in the tree (§13.6, grep-verified).
> **No `Packet` may be released after its count has reached zero**, and the
> model must admit the same-thread case (§11.3), the two-distinct-holder shape
> (Q1, 15/15), and a loss that occurs strictly *before* the free (`drift=+0`,
> 14/14).

The wrapper layer — tag refs, views, zeroreset, scoped CAS, park/unpark — is
**out of scope**: it cannot touch a `Packet`'s units, and it is separately
verified.  That makes the spec smaller than one built around a named edge.

### 13.9 The "wrongly-mine" precondition, made mechanically checkable
### (KAME_RC_TRACE_MINE_CHECK — a logic detector, no race needed)

§13.8 left the live surface at "copies whose source is a slot in the
packet tree, racing a concurrent structural mutation of that tree".  For
the WRITER of such a mutation to exist at all, copy_branch's already-mine
test (`m_serial == tr_serial` ⇒ skip the clone, mutate in place) must
have trusted a mark on a packet that other actors can reach.  Reading the
serial life cycle on this side:

- `tr_serial` is REGENERATED on every retry (`operator++` →
  `snapshot()` → `SerialGenerator::gen()`), so a stale mark from a failed
  attempt can never match — the wrongly-mine condition requires a mark
  that ESCAPES the private branch **within one attempt**.
- The complete writer set for `PacketList::m_serial` is five sites:
  `insert` (:1389), `release` (:1522), `swap` (:1649), and the two
  copy_branch clone blocks.  bundle/unbundle stamp only the WRAPPER field
  `m_bundle_serial` — never a list — so mid-transaction bundle
  publications carry no tr marks.
- `insert(online)` audited: both interim `commit(tr)` calls publish a
  fresh clone OF THE OLD packet (shallow, old serials), and the
  child-linkage CAS publishes `subpacket_new` (bundle-built, no list
  marks).  Object SHARING between the private branch and the live tree
  does begin here (`packet->subpackets()->back() = subpacket_new`), but
  every shared list still carries a non-tr serial, so re-encounters clone
  properly.  No escape found on this path.
- **`eraseSerials()` is the design's own acknowledgment of the hazard**:
  it exists to strip tr marks from a RELEASED subtree, precisely because
  that subtree stays alive (the released node's own linkage still serves
  it) and a surviving mark would be wrongly trusted.  It is the ONLY
  escape route that is patched — and its walk recurses through
  `subpackets()` slots, so in a hard-link topology, where a slot on this
  branch is NULL (missing) while the same child is reachable through a
  sibling parent, **the walk cannot reach everything the tree can**.  A
  mark surviving on such a shared packet + a same-tr re-encounter is the
  cleanest wrongly-mine candidate left.  (The dynamic test creates
  exactly these shapes: insert of an already-parented node = hard link,
  plus release/swap in the same closures.)

Rather than keep enumerating by hand, the condition is now checked
mechanically: `KAME_RC_TRACE_MINE_CHECK=1` (tracer builds) arms a
detector at all three already-mine skip sites (`reverseLookupWithHint`,
`forwardLookup`, `reverseLookup`'s payload-level skip).  On a skip it
walks the CURRENT COMMITTED tree (root node's linkage, tag-view read) and
records an `OP_MINE_SHARED` anomaly — with the packet's full history —
if the packet about to be treated as private is reachable from it.  This
is a **logic condition, not a race**: it fires deterministically on any
host where the precondition occurs, independent of whether the downstream
torn-copy race would have been lost or won.  Known coverage gaps: the
check self-disables while the root wrapper lacks priority (mid-bundle),
and `forwardLookup`'s interior recursion levels check only their local
root.

Mac result: 100 runs × 40 threads, `-O2` test / `-O3` forensic pool —
**zero MINE-SHARED hits** (and zero anomalies).  So on arm64 the
precondition does not occur in this workload, at least outside the
gated windows.

**Ubuntu next**: run the reproducer with `KAME_RC_TRACE_MINE_CHECK=1` on
the v10 build.  Interpretation:
- MINE-SHARED fires (with or without a subsequent underflow) → the edge
  is the wrongly-mine class; the anomaly's site + packet history name the
  specific escape (watch for release-of-hard-linked-subtree shapes).
- The 37 %/run underflows keep firing with ZERO MINE-SHARED → the
  wrongly-mine class is refuted, and §13.8's surface narrows to the
  reader side: torn copies through slot REFERENCES already in hand
  (`local_shared_ptr<Packet> &` returned by lookups, `fast_vector`
  entry references across a concurrent in-place legitimate clone of the
  same list) — plus habitat 3's container lifecycle.

### 13.10 TSan — the missing enumeration, and why it must run on Ubuntu

Reflecting on why the hunt narrows without converging: every instrument so
far (GenMC, TLA+, the hand audits, §13.8's reading) verifies SOURCE
SEMANTICS, and at that level the code keeps coming up clean.  The
observables, though — x86-64/g++ only, flipped by ONE optimization pass
(`-fno-ipa-cp-clone`), invisible on weaker-ordered arm64 (0/200,
blind-spot-free) — point BELOW that level: either a genuine g++
miscompile, or, more likely, a PLAIN-field data race in the source
(formal UB) that ipa-cp-clone's changed inlining/cloning contexts license
the optimizer to weaponize (merged/hoisted loads, sunk stores).  Sanctioned
plain-field shared accesses exist: `eraseSerials()` resets
`PacketList::m_serial` / `Payload::m_serial` IN PLACE on a
released-but-still-served subtree; `Packet::m_missing` (plain bool) is
written on packets other threads read; `setReverseIndex()` is a plain int
write.  GenMC cannot see any of this as g++ codegen (it compiles with
clang), and the models abstract the fields entirely.

The mechanical enumerator for exactly this class is ThreadSanitizer, and
it has never been run on this codebase.  **It cannot run on the Mac**:
on macOS 26.4, Apple clang 17's TSan runtime crashes pre-init in
`__tsan::SlotLock` even on `int main(){return 0;}`, and Homebrew LLVM
20's TSan links but never initializes (verbosity=2 prints nothing; a
guaranteed 10^6-increment race goes unreported).  So this lands on
Ubuntu, which is the right host anyway (g++, canonical TSan, and the
races of interest are observable INDEPENDENT of whether the fault
manifests — no timing luck needed):

```
g++ -DA_NO_P1TREE -fsanitize=thread -O2 -g -std=gnu++17 \
    -I kamestm/tests -I kamestm -I kamepoolalloc \
    -include kamestm/tests/support_standalone.h \
    kamestm/tests/tmin_dynnode.cpp kamestm/tests/support_standalone.cpp \
    kamestm/threadlocal.cpp -o tmin_tsan
TSAN_OPTIONS="halt_on_error=0 log_path=tsan history_size=7" ./tmin_tsan 20 8 200
```

Constraints: NO pool `.so` and no `-DKAMEPOOLALLOC_DYLIB` (TSan must own
malloc), NO `-DKAME_RC_TRACE` (the tracer's racy-by-design caches would
flood the output with known-benign reports).  A few runs at 8–40 threads
suffice.

Reading the output: TSan models `std::atomic` — the intentional
relaxed/acq-rel machinery stays silent.  **Every report on a PLAIN field
is formal UB** and is exactly the enumeration we have been missing.
Expected/priority suspects: `PacketList::m_serial` (eraseSerials reset vs
copy_branch's serial checks), `Packet::m_missing`, `m_reverse_index`,
`fast_vector` internals.  For each report keep both stacks and the field;
then cross-reference which side lives in a function ipa-cp-clone would
clone (hot templates with constant-ish arguments).  Outcomes:
- Reports on plain fields exist (expected) → each is a candidate for the
  weaponization story; fixing is per-field atomic<> with relaxed loads —
  cheap — and re-running the reproducer per fix bisects WHICH race is
  load-bearing.
- Zero reports (unlikely) → source is race-free, H1 (real g++ miscompile)
  is promoted; next step is the `__attribute__((noipa))` per-function
  bisect of the 43%↔0% switch, then a g++ version sweep (12/13/14/15) and
  an asm diff of the named function.

### 13.11 TSan on Ubuntu — one racing variable, already fixed elsewhere; then silence

§13.10's recipe run on Ubuntu 26.04 / g++ 15.2 (TSan works here; the Mac
finding stands).  Built without the pool (TSan owns `malloc`) and without
`KAME_RC_TRACE`, per the recipe.

**Pass 1 — `./tmin_tsan 5 8 200`: 60 data races, ALL the same variable.**

```
Linkage::m_tx_commit_count      transaction.h:1003   mutable uint64_t (non-atomic)
  write  ++node.m_link->m_tx_commit_count   finalizeCommitment      transaction.h:2767
  read   self->m_tx_commit_count            livelock_probe_tx_tick  transaction_neg_impl.h:1546
```

56 of 60 are that write/read pair, 4 are write/write on the same word.  This
is exactly the race fixed by `cc227fafd` on `claude/great-turing-Ufao2`
("make `Linkage::m_tx_commit_count` a relaxed pointer-width atomic") — **and
that fix is not on this branch**, which forked from the rt-linux line.  It
also matches that commit's own note that TSan reported this as the only racing
variable in the STM.

**Pass 2 — same fix applied locally to unmask, `./tmin_tsan2 20 8 200`
(4× the work): ZERO races.  No log file emitted at all.**

So **none of §13.10's priority suspects is a TSan-visible race**: not
`PacketList::m_serial`, not `Payload::m_serial` (the `eraseSerials()`
in-place resets), not `Packet::m_missing`, not `setReverseIndex()`'s plain
int, not `fast_vector` internals.  Whatever those accesses are, in this
workload they do not execute concurrently in a way TSan can flag.

**Branch decision per §13.10: H1 (genuine miscompile) is promoted.**  The
plain-field-race explanation is not supported by the mechanical enumerator
built for exactly that class.

**Two caveats on the negative**, so it is not over-read:

1. Coverage is `20 × 8 × 200` without the pool — the configuration in which
   the fault does not manifest at all.  §13.10's argument that race
   observation is manifestation-independent holds for races that *execute*,
   but a plain-field race gated behind a code path the pool-less build never
   takes would be invisible.  A larger run (more rounds/threads) would
   strengthen it; TSan at 8 threads is ~30× slower than the bare build.
2. TSan sees the *compiled* accesses.  If g++ has already merged or hoisted a
   plain load such that two source-level accesses became one, TSan observes
   the optimized form — which is the very transformation H1 posits.  So a
   TSan negative is weaker evidence against H1's *mechanism* than it looks:
   it cannot see a race the optimizer created.

**Action item independent of this hunt**: port `cc227fafd` to this branch and
to master.  It is real UB (`++` on a shared non-atomic `uint64_t`), it is
qualitatively worse on ILP32 (`add`+`adc`, readers can observe torn halves),
and on this branch it is currently the only thing TSan reports.

### 13.12 The counter fix is now ON this branch — and it is a precondition, not a fix

`cc227fafd` cherry-picked here (the branch's only TSan-reported race is
gone at the source).  Expectations kept honest: that commit's own analysis
says the counter feeds only the livelock probe → privilege claiming, no
safety property reads it, and on LP64 the race merely loses updates — so
it is NOT predicted to move the 37 %/run rate.  Its importance is
different: **an H1 (genuine-miscompile) case cannot be made from a binary
containing known UB** — neither internally nor as a GCC report, since UB
licenses the very transformations we would be attributing to a compiler
bug.  With this port, the reproducing build is TSan-clean end to end.

**The pre-registered causal test, now runnable**: rebuild the reproducing
configuration (pool `.so`, `-O3`, forensic) from THIS commit and re-run
the 60-run batch.
- Rate unchanged (expected) → H1 stands on a UB-free binary; proceed to
  the `__attribute__((noipa))` per-function bisect of the
  `-fno-ipa-cp-clone` switch (start with `finalizeCommitment`,
  `livelock_probe_tx_tick`, `reverseLookupWithHint`, `bundle` — the hot
  cloned-template set), then the g++ 12/13/14/15 sweep, then the asm diff
  of whichever function the bisect names.
- Rate drops to zero (unlikely per the analysis above, but the test is
  cheap) → the "benign" counter race was load-bearing after all, and the
  mechanism to write up is optimizer treatment of the racy `++` in the
  cloned commit path.

### 13.13 Where does the anomaly sit on the POOL's timeline?  (asked, and
### until now unmeasurable — plus a reconciliation the record needs)

The user asked: what is the temporal relation between the anomalies and
the pool's batch processing / chunk claim & release — unrelated, or right
after?  Honest answer from the existing record:

- **Chunk-CACHE reuse is already refuted** by §5's ablation (chunk cache
  disabled: 13/30 vs control 12/31).
- **`drift=+0` (14/14)** says the block was never re-handed-out between
  the free and the stale op — so the stale op is NOT "right after a
  re-allocation" of that block.
- **Ages ~0.1–1.8 µs** are far below chunk-lifecycle timescales
  (release needs a fully-empty chunk + madvise) but exactly at slot-batch
  timescales (cross-thread flush, L1→bitmap batch returns, BMWIN claims).
- Beyond that, **nothing is known**: no instrument has ever recorded pool
  batch/chunk events, and the free-side call chains terminate ABOVE the
  pool internals by construction.  The question was unmeasurable.

**A reconciliation §13.11 needs**: promoting H1 (genuine miscompile) sits
uneasily with §5's own earlier retraction — "`-fipa-cp-clone` is a
miscompile" was withdrawn there, with the flag reclassified as an
ACCELERATOR that "changes what the freed Packet's memory contains when
the stale read happens" (same mmap/munmap counts, same wall time).  A
TSan-clean, model-clean codebase whose failure rate depends on the
CONTENT of freed memory points at a third reading alongside H1/H2:
**somewhere upstream there is a read of freed memory whose consequences
depend on what it finds** — the count corruption would then be a
second-order effect of a first-order stale READ, and the flag merely
changes how often the read finds something plausible.  Note who writes
into freed memory as a matter of course: the pool itself (L1 freelist
links, §13.4 — and with the old constant poison an L1-listed block's
word 0 was a valid-looking heap pointer).  Slot-level double hand-out
(the BMWIN class, f104768b) is also NOT excluded by the chunk-cache
ablation.

**New instrument** (this commit): the pool now records a 4096-entry
timeline of its lifecycle events on the same clock as the free records —
`CHUNK-ALLOC`, `CHUNK-RECYCLE` (LRC cache claim), `CHUNK-RELEASE`,
`BATCH-RETURN` (per-chunk bitmap batch), `DLL-DRAIN` (owner exit),
`CROSS-FLUSH` (cross-thread free batch, with entry count) — exported as
`kame_pool_recent_events()`, and the anomaly raw phase prints the 12
newest as `RC-POOLEV` lines with per-event `age_tsc` and a
`SAME-UNIT` tag when the event's address falls in the anomaly object's
256 KiB unit.  (Also fixed: age computations now use a dedicated
wall-clock helper — on arm64 `Ev.seq` is a plain counter and must not be
differenced against the allocator's cntvct.)

Verified on Mac: pool ctest 18/18 under the flag, tmin 3× clean,
RC-POOLEV renders with sane ages in the tripwire smoke.

**Ubuntu**: the next batch (from this commit) answers the question
directly — for each capture, read the RC-POOLEV block: CROSS-FLUSH /
BATCH-RETURN with SAME-UNIT tags inside the free→stale window (≲ 6 000
ticks) mean the anomaly rides the batch machinery; an empty/far timeline
means the pool is a bystander and the §13.12 causal test + noipa bisect
continue on the STM side.

### 13.14 clang and `-O2` re-checked under blind-spot-free instrumentation — still clean

§13.5 established that the constant poison missed stale ops on L1-listed
blocks, so every "clean" arm measured before it — including clang and `-O2` —
had to be re-checked before it could be trusted.  Done: three `.so`s built
from the same source with `-DKAME_POISON_FORENSIC -fno-omit-frame-pointer`,
linked against one identical gcc `-O3` test binary so only the allocator's
codegen varies, interleaved on one host.

| arm | tripwires | runs | outcomes |
|---|---|---|---|
| **gcc `-O3`** | **13** | 42 | 19× `rc=134`, 23× `rc=139` — **0 clean runs** |
| gcc `-O2` | **0** | 41 | **41 × `rc=0`** |
| clang `-O3` | **0** | 41 | **41 × `rc=0`** |

Not a rate difference — a total separation.  gcc `-O3` never completed a run;
`-O2` and clang never failed one.  Fisher on 42/42 vs 0/41 is beyond
meaningful quotation.

**The blind-spot doubt is resolved in the negative**: clang and `-O2` are not
"quietly corrupting and surviving".  With the instrumentation that catches the
fresh-free window at 31 % on gcc `-O3` (13/42, consistent with §13.5's 35 %),
they produce **zero tripwires** — no `INC-FROM-ZERO`, no `DEC-UNDERFLOW`, no
refcount anomaly of any kind.  The defect is absent from their codegen, not
merely non-fatal in it.

**Combined with §13.11** (TSan clean once `m_tx_commit_count` is fixed) this
is the strongest form of the H1 case the available instruments can produce:

- no TSan-visible plain-field race (§13.11);
- no refcount anomaly at all under clang or gcc `-O2`, blind-spot-free (here);
- the fault flipped by exactly one pass, `-fno-ipa-cp-clone`, 0/167 (§11.x);
- arm64 silent at 200×40t, blind-spot-free (§13.6 addendum).

The one reading these do not exclude, and §13.11's second caveat is the reason:
a plain-field race that **only gcc `-O3` codegen makes reachable** — the
optimizer merging or hoisting a load so two source accesses become one — is
invisible to TSan (which sees the compiled form) and absent from `-O2`/clang
(which never perform the transform).  That is not distinguishable from a
miscompile by any instrument used so far; it is distinguishable by reading the
`-O3` asm of the affected paths against `-O2`.

### 13.15 The allocator was never under TSan — and its bitmap machinery is
### plain-load + __sync CAS.  A named mechanism, and a zero-cost falsifier.

Two things line up against §13.14's localization (only the ALLOCATOR's
codegen varies across the total separation):

1. **§13.11's "TSan clean" does not cover the allocator.**  The TSan
   build was pool-less BY NECESSITY (TSan must own malloc).  The half of
   the program the fault was just localized to has never been enumerated.
2. **The bitmap machinery is pre-C++11 idiom**: `m_flags` is a plain
   `uintptr_t *`, CASed via `__sync_bool_compare_and_swap`, and READ via
   plain loads.  Every `oldv = *pflag` racing a peer's CAS is a data race
   — formal UB — in exactly the component the fault lives in.

**The named mechanism (candidate):** under UB, gcc may REMATERIALIZE a
plain load — re-read memory instead of spilling a register — so a value
the source reads once is read twice.  In the claim loops the derived mask
and the CAS expected value can then come from DIFFERENT reads:

- word-grab loop (word cache): `mask = ~read1`, CAS expected = `read2`.
  A peer CLAIM landing between the reads → CAS succeeds (against read2),
  the word goes to ~0, and `mask` still contains the peer's bit — **the
  same slot is handed to two owners.**  A codegen-induced twin of the
  BMWIN double-payout (`f104768b`), in the same machinery.
- N-run claim loop (FS=false): `cand/ones` from read1, expected from
  read2 → `newv = read2 | ones` claims a run overlapping the peer's.

Two owners, one refcount backing → a Packet and its neighbours built in
overlapping slots → the count reaches zero with a live holder → the
~1 µs-later DEC-UNDERFLOW at teardown, and every §11.3 field: two
distinct holder slots, loss strictly before the free, same-thread cases
admitted.  Rematerialization is a register-pressure response, which is
precisely what ipa-cp-clone's constant-N inlining changes — and why -O2
and clang (which do not perform it here) are TOTALLY clean rather than
rate-reduced, and why the flag looked like an "accelerator": different
clone shapes, different split-read windows.

**The falsifier costs nothing**: this commit converts the four
`m_flags`-word loads that feed CASes (word-grab :1528-area, FS=true
single-bit claim, FS=false N-run claim, batch_clear_impl) to
`__atomic_load_n(..., __ATOMIC_RELAXED)` — the same single `mov`/`ldr`
on every target, no ordering change, only the OPTIMIZER'S LICENSE to
duplicate/stabilize the read is revoked.  no-DCAS is unaffected
(pointer-width).  Mac: pool ctest 18/18, tmin 3×40t clean.

**Ubuntu, the decisive run**: rebuild the gcc `-O3` arm from THIS commit
and re-run §13.14's batch.
- 42/42-fail → 0 clean: root cause proven at the construct, no asm
  reading needed (and the fix is already in).  A revert-and-objdump of
  one clone would then document the exact split read for the record.
- Still failing: the mechanism is refuted, the atomic loads stay (they
  fix real UB regardless), and the remaining suspects are the OTHER
  plain shared fields in the allocator (`m_idx`, `m_flags_filled_cnt`
  gate reads, freelist heads) — same treatment, or proceed to the asm
  diff of the NC7 clone set (`-O2` vs `-O2 -fipa-cp-clone`, the 19/35
  minimal pair).  Either way, please also record the full NC7 function
  list in this file — only 2 of 7 are named so far.

### 13.16 The split-read falsifier — REFUTED (the UB fix stands on its own)

§13.15's prediction was explicit and all-or-nothing: if gcc rematerializes a
racy plain `m_flags` load so mask and CAS-expected come from different reads,
converting those four loads to `__atomic_load_n(RELAXED)` should take the gcc
`-O3` arm from 42/42-failing to **zero** — the same total cleanliness `-O2`
and clang show, not a reduced rate.

Run as a **paired A/B** rather than against §13.14's historical numbers: same
test-binary source, same `-DKAME_POISON_FORENSIC`, same host, alternating
`PRE` (parent commit) / `POST` (`c2c5be471`) per cycle so both arms see
identical load.  The `.so`s differ (4 072 112 → 4 097 904 bytes), so the
relaxed loads did change codegen.

| arm | runs | failed | tripwires |
|---|---|---|---|
| `PRE` | 13 | **13** | 5 |
| `POST` | 12 | **12** | 3 |

**100 % failure in both arms.**  Not a reduced rate — no effect.  At a
baseline where every single run crashes, 12/12 is as far from the prediction
as the experiment can get.

**So the named mechanism is refuted.**  The rematerialized-split-read twin of
the BMWIN double-payout is not what produces the fault, and the story it told
so well — why `-O2`/clang are totally clean rather than rate-reduced, why
`-fipa-cp-clone` acts as an accelerator via register pressure — is
unfortunately not load-bearing.

**The UB fix stands regardless**, on the grounds §13.15 gives independently: a
plain load feeding `__sync_bool_compare_and_swap` is formal UB whatever the
optimizer does with it, the relaxed load is the same instruction, and no-DCAS
is unaffected.  Keep it; it just isn't the root cause.

**What the refutation costs and what it leaves.**  It removes the only named,
mechanically-checkable candidate the H1 case had produced.  What survives is
the §13.14 boundary — the fault is in the allocator's gcc `-O3` codegen,
absent from `-O2` and clang, blind-spot-free — with no mechanism attached.  Of
§13.15's own "what comes next" list, the remaining items are the other plain
fields in the same machinery, and the `-O2` vs `-O2 +ipa-cp-clone` asm diff of
the NC7 clones.  The asm diff is now the more valuable of the two: it is the
only instrument left that can distinguish a miscompile from an
optimizer-created race, and §13.14 already narrowed where to look.

### 13.17 After the refutation: falsifier #2 (the ring outside m_flags) and
### a one-command asm-diff runner for the NC7 clones

§13.16 removed the only named mechanism.  This commit supplies both of
§13.15's surviving next steps in runnable form.

**Falsifier #2 — the remaining plain shared reads, one ring out.**  The
§13.15 conversion covered only the four `m_flags` WORD loads.  The free
fast path and the claim gate still read, with PLAIN loads, fields that
chunk-lifecycle writers mutate concurrently:

- `rmeta->back_offset[unit_idx]` (hot `deallocate`) — written by
  `claim_chunk`'s post-CAS publish, cleared by `deallocate_chunk`.  A
  stale/merged read mis-derives `chunk_base`: the free lands in the
  WRONG chunk's bitmap — a bit cleared in a chunk that still owns the
  slot = that slot handed out twice.  This is the highest-value
  conversion of the set.
- `chunk_obj->m_owner_id` (both owner-check sites) — racing owner exit /
  release / re-claim.  A stale match sends a free down the OWNER fast
  path of a chunk this thread no longer owns: `freelist_push` into
  another owner's L1 cells.
- `chunk_obj->m_fs_flag`, and the `m_flags_filled_cnt` claim-gate read
  (races its own `atomicInc/Dec` writers).

All become `__atomic_load_n(RELAXED)` — same single instruction, no
ordering change, only the optimizer's license revoked, no-DCAS
unaffected.  Mac: ctest 18/18 (forensic ON), tmin 3×40t clean.
Prediction discipline as before: if gcc -O3 drops from 100 %-failing to
zero, the guilty read is in this set (bisect by reverting one at a
time); if it keeps failing, the UB fixes stay and the asm diff decides.

**The asm-diff runner** (`kamepoolalloc/tests/asm_diff_ipa_clone.sh`):
one command on Ubuntu builds `allocator.cpp` as the 19/35 minimal pair
(`-O2` vs `-O2 -fipa-cp-clone`; arms overridable via
`ARM_A_FLAGS`/`ARM_B_FLAGS`), splits both disassemblies per symbol, and
emits `summary.txt` (clone symbols only-in-B; changed base symbols) plus
a unified diff per changed symbol.  Plumbing verified on the Mac
(clang `-O2` vs `-O3`: 1117 per-symbol diffs).  Suggested reading order:
the `.constprop` clone families first (`bucket_release_chunk` ×3,
`find_training_zeros` ×2, and the rest of NC7 — please record the full
list), looking for (a) a load of the same address issued twice where the
source reads once, (b) plain stores sunk across a `lock cmpxchg`, (c) a
dropped or displaced `mfence`/`lock` relative to the `-O2` body.

### 13.18 Decision: ship the mitigation now — `-fno-ipa-cp-clone` on gcc
### production builds of allocator.cpp (a mitigation, not a fix)

Two-plus days in, root cause still unnamed.  The user's call: if hours-
scale resolution is not assured, gcc builds should carry the flag.  Done
— and the record should be explicit about what this buys and what it
does not:

- **Empirical strength**: `-O3 -fno-ipa-cp-clone` 0/167 and NC7 0/100
  against a 40–100 %/run baseline; §13.14's blind-spot-free total
  separation.  Combined ≈ 0/267.  As a shipping posture this is as
  strong as an unnamed-root-cause mitigation gets.
- **What it is NOT**: a fix.  §5 stands — the pass is an accelerator;
  the defect (optimizer-created race or miscompile, §13.16) remains in
  the source/toolchain pair and may resurface under a different gcc
  version, flag set, or workload.  Tracking note: remove this flag only
  when the root cause is named and fixed, and re-run §13.14's A/B to
  prove it.
- **Scope**: production builds that compile `allocator.cpp` under gcc —
  `kame/kame.pro` (Linux g++ AND Windows mingw-g++ production; `*g++*`
  scope, macOS is clang and refuses gcc anyway per `cdb70d2cf`),
  `kamepoolalloc/kamepoolalloc.pro`, and the standalone
  `kamepoolalloc/CMakeLists.txt` dylib (flows to the standalone repo via
  the subtree sync; tag the next standalone release with it).
- **Deliberately NOT flagged**: every test tree (`tests/`,
  `kamestm/tests/`, `kamepoolalloc/tests/`, the handoff's manual .so
  builds).  The reproducer must keep reproducing or the hunt stops.
- **Cost**: ipa-cp-clone is typically worth at most a few percent here;
  if it matters, measure on Ohtaka later (correctness first).  The Mac
  cmake build is unaffected (GNU-gated generator expression).

The hunt continues underneath: falsifier #2's A/B and the NC7 asm diff
(§13.17) still decide between optimizer-created race and miscompile —
whichever lands also decides whether this flag can ever come off.

### 13.19 The three textbook ipa-cp-clone failure modes, checked against
### this code — and the indirect-call refinement for the asm diff

A generic list of "why ipa-cp-clone breaks programs" (miscompile with
wrongly-assumed constant args; function-ADDRESS identity broken by
clones; inline-asm/Windows interactions) was checked against
allocator.cpp:

1. **Function-address identity — does not apply as stated.**  The chunk
   header stores `DeallocateFn`/`SizeOfFn` and the cold paths CALL
   THROUGH them (:4071/:4166); nothing compares a function pointer's
   address (the "identity-compares" at :577 is the DATA pointer
   `&g_teardown_page`).  An address-taken function keeps its original
   address under cloning, and calls through it are semantically safe.
2. **But the inverse of (1) is live and CHECKABLE**: gcc performs
   SPECULATIVE INDIRECT-CALL PROMOTION — at an indirect call whose
   target set it thinks it knows, it emits `cmp fn, $target; je
   direct_or_clone; else indirect` GUARDS of its own.  The stored-fn
   dispatch at :4071/:4166 (one stamp site per template instantiation)
   is exactly the shape that invites it, and a wrong assumption there
   (mis-narrowed target set, or a clone specialized on a mis-proven
   constant argument) is a PR110282/118138-class wrong-code.  **Asm-diff
   refinement: in the `-O2 -fipa-cp-clone` arm, look FIRST at the
   deallocate_cold / size_of call sites through the header-stored
   pointers for compiler-inserted compare-and-branch guards and
   .constprop targets.**  gcc 15.2 postdates the PR118138 ipa-cp fix
   (2025-01), so that exact bug is likely already fixed — but the family
   is the right neighborhood.
3. **Inline asm / MinGW modes — not applicable** (production repro is
   Linux ELF; the only asm in the TU is rdtsc/cntvct in the debug-gated
   forensic block and pause4spin).

Cheap additional discriminator for Ubuntu (E3): an A/B of
`-O3 -fno-indirect-inlining` (clones still allowed, indirect-call
promotion suppressed).  If THIS also fully suppresses, the defect sits
in the promoted-indirect-call machinery around the stored-fn dispatch,
and the asm reading narrows to those two call sites; if it keeps
failing, the promotion reading dies and the clones themselves remain.

**§13.18 addendum — ecosystem precedent for the flag (no apology needed).**
Surveyed who else disables function cloning / related IPA passes:

- **Linux kernel**: carries `-fno-ipa-cp-clone` (CONFIG_READABLE_ASM) and
  a `-fno-indirect-inlining` bundle (CC_DISABLE_AUTO_INLINE) — and builds
  at `-O2`, where the pass is off by default anyway.
- **illumos**: since 2018 builds the entire gate with function cloning
  and IPA-ICF disabled in `Makefile.master` (`CCNOAUTOINLINE`) as
  permanent policy — clones break dtrace/mdb symbol integrity.
- **GCC itself** disables IPA-CP cloning on functions with
  `target_clones` (feature interaction fix), and has a wrong-code
  lineage in the pass (PR110282-family; PR118138 fixed 2025-01).
- Distro-level breakage records exist (RH bz#1340377, `-fipa-cp-clone`
  + `-flto` crash; Valgrind false positives, KDE bz#378627).
- The deeper norm: mainstream OSS and distro packaging ship `-O2`, where
  `ipa-cp-clone` (and `-findirect-inlining` beyond `-O2`'s subset) never
  runs.  Running "`-O3` minus this one pass" on one TU is therefore MORE
  aggressive than the ecosystem default, not less.

So the shipping posture needs no defense.  What the precedent does NOT
yet license is the verdict "gcc is at fault" — most documented disables
are for observability/tooling or specific ICEs, and our own §5/§13.16
record says the pass is an accelerator here.  The verdict still comes
from E3 + the asm diff; the flag is correct to ship either way.

**§13.18 addendum 2 — allocators are the -O3-by-default corner of OSS,
and that shifts the prior.**  Asked whether any OSS ALLOCATOR defaults to
-O3: yes, essentially all of the modern ones.  jemalloc's configure
appends `-O3 -funroll-loops` on gcc when CFLAGS is unset; mimalloc and
snmalloc build as CMake Release, which is `-O3 -DNDEBUG` on gcc; rpmalloc
likewise.  These run under gcc -O3 — ipa-cp-clone ON — at planetary
scale (Redis vendors jemalloc; Firefox ships mozjemalloc; Rust shipped
jemalloc for years) without endemic miscompiles.

Consequence for the verdict (as opposed to the shipping posture): the
ecosystem does NOT support "gcc at -O3 cannot be trusted over allocator
hot code".  What separates kamepoolalloc from the peers is the INPUT
class: they are C with strict C11-atomics discipline throughout; this TU
is heavily-templated C++ (PoolAllocator<ALIGN,FS,DUMMY>, constant-N
claim loops — far richer cloning food) whose shared accesses were, until
§13.15/§13.17, plain-load + __sync (formal UB).  The prior therefore
tilts back toward "our residual UB × gcc's license" over "gcc bug" —
falsifier #1 is refuted but #2 is pending, and plain fields remain
beyond it (freelist links via kame_slot_link_, the L0 FIFO/STASH cells,
m_idx, the write sides of owner/back_offset).  The flag ships either
way; the verdict still belongs to E3 + the asm diff, now with this
prior attached.

### 13.20 DRF-migration inventory — what §13.15/§13.17 did NOT convert,
### and the queued falsifier #3

Asked whether "mark the concurrent accesses, keep plain inside sync
edges" is already done: only the READ half of the hot paths is.

Converted: the four m_flags word loads (§13.15); back_offset /
m_owner_id ×2 / m_fs_flag / filled_cnt-gate READS (§13.17, verdict
pending).

NOT converted, classified:
1. **`m_idx` — reads AND writes, the clearest live race.**  The code's
   own comment licenses concurrent allocate_pooled on one chunk, so the
   `this->m_idx = idx` stores (:1537/:1618/:1808) race the loop-head
   reads across threads.  Queued as **falsifier #3** (convert both
   sides, relaxed).
2. **Post-publication write sides** of m_owner_id / back_offset /
   m_fs_flag (owner-exit clear, release zeroing, restamp) — formal race
   against the §13.17 atomic reads; also falsifier #3 scope.
   Construction-time stamps stay plain (correct publication pattern).
3. **Freelist links / L0 FIFO+STASH cells / hdr_word** — single-writer
   or publication-patterned BY DESIGN; correctly plain, do not convert
   (converting would bury design intent).

Discipline note: #3 waits for #2's verdict — one all-or-nothing
falsifier per commit is what gives these experiments evidential value,
and Ubuntu's paired A/B baselines must stay clean.  If #3 also comes
back refuted, switch strategy: hygiene-first full DRF migration of
allocator.cpp (the mimalloc/jemalloc discipline, §13.18 addendum 2) and
let the asm diff carry the verdict alone, for the record and any GCC
report.

**§13.20 addendum — spelling fix: `atomicLoadRelaxed()` shim (MSVC
compatibility restored).**  Answering "so it's atomic<> everywhere OR the
old barrier+volatile school?": the two schools are ISOMORPHIC — the
kernel's READ_ONCE/WRITE_ONCE + barrier() is a hand-rolled pre-C11
spelling of relaxed-atomic + fences, same generated code — and in
userland C++ the standard spelling is strictly better (ISO-defined,
TSan-visible, no volatile over-constraint).  It is also not "everywhere":
§13.20's classification stands (concurrent words atomic; owner-only /
pre-publication words plain, by design).

While answering, an own-goal surfaced: the §13.15/§13.17 falsifiers used
`__atomic_load_n` directly — a GNU extension absent on MSVC, which the
standalone lib supports (the `_Interlocked*` shims, `/utf-8`).  Fixed:
`atomicLoadRelaxed()` in allocator_prv.h next to the other shims — GNU
arm `__atomic_load_n(RELAXED)`, MSVC arm an aligned volatile scalar read
(the idiomatic MSVC relaxed load; a live illustration that the volatile
school and the atomic school are one discipline in two dialects).  All 9
call sites converted; ctest 18/18 forensic ON, tmin 40t clean.

### 13.21 The decision procedure — audit the simple rule; what survives a
### clean audit is a gcc bug

Agreed framing (user): the rule is simple — concurrent words atomic,
owner-only/pre-publication words plain — so it should be settled by
AUDIT, and whatever slips through a clean audit is gcc's fault.
Formalized as the procedure:

1. **Dynamic audit — runnable on Ubuntu TODAY, closes §13.11's gap.**
   `KAMEPOOLALLOC_NO_LIBC_INTERPOSE` (already in the tree; Linux gates
   the strong-symbol malloc family off, operator new/delete still pool)
   lets TSan own libc malloc while the POOL runs fully instrumented
   under it:

   ```
   g++ -DA_NO_P1TREE -DKAMEPOOLALLOC_NO_LIBC_INTERPOSE -fsanitize=thread \
       -O2 -g -std=gnu++17 -I kamestm/tests -I kamestm -I kamepoolalloc \
       -include kamestm/tests/support_standalone.h \
       kamestm/tests/tmin_dynnode.cpp ../kamepoolalloc/allocator.cpp \
       kamestm/tests/support_standalone.cpp kamestm/threadlocal.cpp -o tmin_tsan_pool
   ```

   (inline-compile the allocator, as production does; no .so, no
   KAME_RC_TRACE.)  The pool's __sync/__atomic ops are TSan-visible
   sync, so claim/free handoffs form recognized happens-before — every
   report on a PLAIN allocator field is a genuine rule violation, named
   with both stacks.  This is the mechanical audit of the simple rule
   for all executed paths.
2. **Fix what it names** (atomicLoadRelaxed / relaxed stores — the
   §13.20 falsifier-#3 set is the predicted catch: m_idx both sides,
   post-publication owner/back_offset writes).
3. **Static ratchet afterwards**: a tools/audit checker over
   allocator.cpp flagging loads/stores of the classified-shared field
   list that bypass the atomic helpers, baselined like stm_closures, so
   the rule cannot regress.
4. **Then the verdict is mechanical**: if the reproducer still fails
   at gcc -O3 on a TSan-clean-WITH-POOL, audit-clean build, the program
   is DRF and the compiled behavior is nonconforming — a reportable gcc
   bug, with the §13.17 asm-diff artifacts as the exhibit.  (One honest
   caveat: TSan certifies executed pairs; the static ratchet plus the
   classification list is what covers the rest.)

Mac-side verification this commit: the define compiles clean (Release).

### 13.22 Asm diff run on Ubuntu — signature (c) hit: `orphan_chain_pop` loses one refcount `lock add` in the clone arm

`kamepoolalloc/tests/asm_diff_ipa_clone.sh $OUT /usr/bin/c++
-DKAME_POISON_FORENSIC` on Ubuntu / g++ 15.2.  Output: **40 clone symbols
only in B**, **380 changed base symbols**.

Scanned mechanically for §13.17's signature (c) — a dropped or displaced
`lock`/`mfence` relative to the `-O2` body — by comparing per-symbol atomic
counts.  Two hits; one is benign, one is not obviously so.

**Benign: `recycle_push`** (A=6, B=5).  The missing `lock xadd` is inside
B's `l1_base.part.0`, a B-only outlined symbol; A inlines `l1_base` entirely
(A has no separate `l1_base` symbol at all).  Moved, not dropped.

**Not benign-looking: `orphan_chain_pop`.**

```
A (-O2)                    B (-O2 -fipa-cp-clone)
  je   …                     (block absent)
  mov  %rax,%r14
  lock add %rax,0xd8(%r14)   <-- early-path refcount add, NOT in B
  …
  lock cmpxchg %rdx,(%rsi)   lock cmpxchg %rcx,(%rsi)
  jne  …                     jne  …
  lea  0xd8(%r15),%r9        lea  0xd8(%r14),%rdi
  lock add %rcx,0xd8(%r15)   lock add %rdx,0xd8(%r14)   <-- CAS-path add, in both
```

It is **systematic, and the instructions are not merely relocated**:

| | A | B |
|---|---|---|
| `lock add …0xd8` instructions, whole object | **198** | **176** |
| symbols containing one | 154 | 154 |
| `orphan_chain_pop` instantiations losing exactly one | — | **22 of 22** |

Same symbol count, 22 fewer instructions, and every one of the 22
`orphan_chain_pop` instantiations accounts for exactly one.  So B is not
outlining them elsewhere — it is not emitting them.

`0xd8` is a refcount word on the popped chunk object (`orphan_chain_pop`
returns `local_shared_ptr<PoolAllocator<…>>`; the adoption path re-owns the
chunk).  A performs the increment on a null-checked early path *and* on the
CAS-success path; B performs only the latter.

**What this is NOT yet.**  `-fipa-cp-clone` legitimately deletes code it has
proven unreachable in a specialisation, so a constant-folded condition
eliminating one of two source paths is an entirely lawful explanation and is
the null hypothesis here.  Deciding it needs the source read that this session
cannot responsibly do from disassembly alone: **are both increments reachable
in the specialised context, and does the early one correspond to a reference
that the adoption path still relies on?**  `orphan_chain_pop`'s caller is at
`allocator.cpp:2952` (`oc_hold = orphan_chain_pop()`, then the BIT_OWNED claim
loop and owner re-arm).

It is worth prioritising because of *where* it sits: chunk adoption is the
path the reproducer hammers (§11.4 — repeated thread create/exit under a
persistent tree), and a lost increment on a chunk's refcount in the failing
arm only is the right shape for "two owners, one count", which is what Q1
established 15/15.

**Falsifier #2 is running in parallel** (paired PRE vs `2d8a9e5b8`); §13.16's
discipline applies unchanged — 100 %-failing to zero confirms, still-failing
refutes and keeps the UB fixes.

### 13.23 E3 run — and a flag that did nothing, nearly reported as a refutation

**First, a methodological correction that matters more than the result.**
E3 as specified used `-fno-indirect-inlining`.  On this TU that flag is very
nearly a **no-op**:

| | `-O3` | `+ -fno-indirect-inlining` |
|---|---|---|
| differing disassembly lines | — | **2** |
| symbols | 1143 | 1143 |
| indirect calls (`call *%r…`) | 90 | 90 |
| `constprop` symbols | 585 | 585 |

Two instructions.  Its arm duly "kept failing" (6/6) — a result with **no
information**, which was about to be written up as a refutation.  Probing which
flag actually engages the machinery §13.19 names:

| flag | diff lines | constprop | indirect calls |
|---|---|---|---|
| `-fno-indirect-inlining` | **2** | 585 | 90 |
| `-fno-devirtualize-speculatively` | **62 425** | 558 | **5** |
| `-fno-devirtualize` | 62 425 | 558 | 5 |
| `-fno-ipa-cp` | 62 434 | 0 | 69 |
| `-fno-ipa-cp-clone` | 58 779 | **12** | 93 |

**Standing practice from here: verify a falsifier flag moved codegen before
believing either outcome.**

**E3, re-run with `-fno-devirtualize-speculatively`** (paired, same HEAD, only
the flag differing):

| arm | runs | failed | tripwires |
|---|---|---|---|
| `-O3` | 28 | **28** | 25 |
| `-O3 -fno-devirtualize-speculatively` | 27 | **27** | 13 |

**100 % failure in both — the speculative-promotion reading is refuted**, and
this time the negative is real: the flag changed 62 425 lines and collapsed
indirect calls 90 → 5.

Two observations worth keeping:

- The flag **halves the tripwire count** (25 → 13) without touching the
  failure rate.  A rate effect on anomaly *frequency* but not on whether the
  run dies — consistent with everything since §13.5, where instrument and
  codegen changes move rates while the fault survives.
- `-fno-devirtualize-speculatively` collapsing indirect calls 90 → 5 shows
  speculative promotion is **pervasive** in this TU, not confined to the
  `:4071`/`:4166` dispatch sites.  Had the arm gone clean, it would still have
  needed narrowing before naming a site.

Also noted from the probe table: `-fno-ipa-cp-clone` leaves **12** `constprop`
symbols rather than zero — independent confirmation of §12.5's point that
clone-set equality is not codegen equality.

**Tally of named mechanisms refuted by mechanical test: five.**
`CASInfo::old_wrapper` (§12.5), wrongly-mine (§13.9, Mac 0/100), split-read
(§13.16, 40/40 both arms), falsifier #2's `back_offset`/owner/gate reads
(11/11 both arms), and speculative indirect-call promotion (here).  The
§13.14 boundary — allocator, gcc `-O3` only, absent from `-O2` and clang,
blind-spot-free — has never moved.

### 13.24 The §13.22 source read — no constant can kill either increment;
### one benign escape hatch remains, and falsifier #4 decides at runtime

The requested read of `orphan_chain_pop` (allocator.cpp:8242) against the
two `lock add …,0xd8` sites:

```
local_shared_ptr<PoolAllocator> old(s_orphan_chain_head());
        // lsp(asp&) ctor = load_shared_: acquire_tag_ref_ (CAS loop) →
        // null check → PROMOTE refcnt.fetch_add(rcnt)  == the EARLY,
        // null-checked lock add (variable amount = the tag count in %rax)
for(;;) {
    if(!old) return {};
    local_shared_ptr<PoolAllocator> nxt(old->m_orphan_next);
        // second load_shared_ — same shape, on NXT's refcnt
    if(s_orphan_chain_head().compareAndSwap(old, nxt)) {
        // Swap variant: step4 pre-pay fetch_add(rcnt_old−1) on OLD's
        // refcnt == the CAS-path lock add (present in BOTH arms)
        old->m_orphan_next = lsp();   // asp assign = more asp machinery
        return old;
    }
    // failure: compareAndSwap re-acquires into `old` (another promote)
}
```

**Verdict on the null hypothesis: DEAD at source level.**
`orphan_chain_pop` takes NO arguments — IPA-CP has no constants to
specialize on — and neither promote is guarded by anything a template
parameter or propagated constant could fold: `load_shared_`'s promote is
unconditional after the null check, the chain head is a global atomic
(non-emptiness unprovable), and the only constexpr branch in the ctor
path (`is_biased_directpublish<T>`) is OFF identically in both arms.
There is no lawful "specialisation removed a source path" story for this
function.

**One benign escape hatch their exclusion does not close.**  "Symbols
containing one add = 154 in both" rules out relocation into a NEW
symbol, but not into an EXISTING one: if B replaced the inlined
`load_shared_` with a CALL to an out-of-line copy that already carried
an add in A's census, totals drop by exactly 22 while the
symbols-containing count stays fixed — the same shape as the benign
`recycle_push`/`l1_base.part.0` case.  **The one-line check: does B's
`orphan_chain_pop` body contain a `call` where A had the inline add?**
(e.g. `for f in B/sym/*orphan_chain_pop*.s; do grep -c "call" $f; done`
against the same over A, and eyeball the callee names.)

**If there is no call either — the increment is gone, and this is the
bug**: the popped head loses its pin, `orphan_chain_scrub` /
`atomic_intrusive_dispose` can free the chunk while the adopter holds
and re-owns it, the region is reused, and every slot in it is handed out
twice — Packet underflows downstream with exactly Q1's
two-holders-one-count shape, at the thread-exit cadence the reproducer
hammers (§11.4).  clang/-O2 keep the add; 42/42 vs 0/41 follows.

**Falsifier #4 (runtime, cheap, decisive either way):** rebuild the B
arm with `__attribute__((optimize("O2")))` (or `noipa`) on
`orphan_chain_pop` alone — §5's warning about noclone side-effects is
about attributing CAUSE to a clone set, but here the prediction is
sharp and pre-registered: if pinning this ONE function to -O2 codegen
takes the arm from 100 %-failing to zero, the lost increment is the
defect (then extract the minimal TU around orphan_chain_pop + the asp
machinery for the GCC report); if it keeps failing with the add
restored (verify in the disassembly first, per §13.23's standing
practice), the site is exonerated and the census scan continues to
signatures (a)/(b).

### 13.26 The call: run (b) now — and (a) only as a census, with the unit
### of analysis corrected for -O3

Decision requested on §13.25's fork.  **Run (b)** — falsifier #4 on the
-O2 proxy pair, where the missing increment demonstrably exists and the
configuration demonstrably fails (19/35).  It is the well-posed
experiment, and it completes the GCC-report package either way: the
-O2-pair exhibit is already sufficient for a wrong-code report (one
pass, one no-argument symbol, 22 deleted atomic RMWs, source read on
record showing no constant can justify any of them, call-census showing
no relocation) — (b) adds the runtime consequence if it flips, and if it
does NOT flip, the deletion is still reportable wrong code, just not our
crash.  Please record the proxy arm's tripwire phenotype (op types,
anomaly object type) alongside, so proxy-vs--O3 comparability stops
being an assumption.

**(a) should not be run as a full A/B — only as a census — and with a
corrected target.**  Two structural points:

1. **-O3's "5 lock adds" in the standalone symbol is likely the COMPLETE
   set, not an excess.**  The §13.24 source map has exactly five static
   variable-amount fetch_add classes inside orphan_chain_pop: head-load
   promote, nxt-load promote, CAS step4 pre-pay, CAS-failure re-acquire
   promote, and the m_orphan_next-assign pre-pay.  -O2's "2" means three
   of them lived behind calls there; -O3 inlines them all.  So the
   standalone -O3 symbol being "intact" is expected and says nothing.
2. **At -O3 the unit of analysis is probably the CALLER.**  If
   allocate_chunk_path (:2952) inlines orphan_chain_pop at -O3, the
   standalone symbol is dead weight and the place an analogous deletion
   would live is the INLINED copy.  Check first: does the -O3 caller
   `call orphan_chain_pop` at all?  If not, run the census on
   allocate_chunk_path's body (all instantiations): count the five site
   classes in -O3 vs -O3 -fno-ipa-cp-clone.  Corollary: §13.25's "noipa
   is a no-op at -O3" was verified on the SYMBOL being byte-identical —
   but noipa also forbids INLINING, so the CALLER must have changed
   (now calling the complete out-of-line copy).  If the caller did
   change, falsifier #4 at -O3 is not a no-op after all — it forces the
   never-cloned, all-five-adds copy into use, which is exactly the
   experiment we want; re-check the caller's disassembly before
   discarding that arm.

Sequence: (b) first (cheap, decisive for the proxy), the corrected (a)
census second (one build + objdump, no runs), full -O3 A/B only if the
census finds an asymmetry to attribute.

### 13.27 §13.24's call-check answered — and falsifier #4 is untestable as specified

**The call-check: no inline-to-call conversion.**  Comparing
`orphan_chain_pop` bodies in the §13.22 pair:

```
call targets in A: s_owner_id_next ×1, self-relative ×13
call targets in B: s_owner_id_next ×1, self-relative ×11
calls present only in B: NONE
totals across all 22 instantiations: A=330 calls, B=307
```

B has **fewer** calls, and no target appears in B that is absent from A.  The
benign escape §13.24 names is excluded: the increment is **gone**, not moved
into a callee.  §13.24's source read (no foldable constant guards either add;
`orphan_chain_pop` takes no arguments so IPA-CP has nothing to specialise on)
plus this leaves no lawful explanation *within the `-O2` pair*.

**But falsifier #4 cannot be run as written, and the reason matters.**
`__attribute__((noipa))` on `orphan_chain_pop` produced **byte-identical**
codegen: 197 `lock add …0xd8` object-wide in both, and the function body
identical (5 adds, 219 lines).  Per §13.23's standing practice that is a
no-op, so the arm would have returned a null — the `-fno-indirect-inlining`
trap again.

**Why it is a no-op exposes a gap in §13.22's own reasoning**, which is mine
to correct:

| build | `lock add` in `orphan_chain_pop` | body lines |
|---|---|---|
| `-O2` (asm-diff A) | 2 | 158 |
| `-O2 -fipa-cp-clone` (asm-diff B) | **1** | 136 |
| **`-O3` (the arm that actually fails)** | **5** | 219 |
| `-O3` + `noipa` | 5 | 219 |

The §13.22 asymmetry lives entirely in the **`-O2` pair**.  The build that
actually fails is `-O3`, and there `orphan_chain_pop` carries **five**
increments — more than `-O2`'s two, not fewer.  The `-O2 +ipa-cp-clone` pair
is a *proxy* for the failing configuration (chosen in §13.17 as the 19/35
minimal pair), and on this symbol the proxy and the real failing build do not
agree.

So §13.22's "the failing arm loses a refcount increment" **does not transfer
to `-O3` as stated**.  What survives is narrower: *in the minimal pair*,
enabling one pass deletes 22 increments that no source-level constant can
justify.  Whether anything analogous happens at `-O3` is unmeasured — the
symbol there is a different shape and would need its own comparison, against
what baseline is not obvious, since `-O3 -fno-ipa-cp-clone` is the natural
control but differs from `-O3` in far more than this symbol.

**Suggested reframing before more effort goes here**: either (a) re-establish
the asymmetry directly in the failing configuration (`-O3` vs
`-O3 -fno-ipa-cp-clone`, this symbol, same census), or (b) demonstrate that
the `-O2 +ipa-cp-clone` proxy actually fails — §13.17 cites 19/35, so it does
fail, and then falsifier #4 should be run **on the proxy**, where the missing
increment demonstrably exists, rather than at `-O3` where it does not.

### 13.28 (a) and (b) both run — and falsifier #4's target is the wrong function

**(a), the corrected `-O3` census — comes back empty.**  §13.26 predicted the
`-O3` deletion, if any, would live in `allocate_chunk_path`'s *inlined* copy,
and that `noipa` must have changed the caller.  Measured:

```
calls to orphan_chain_pop:   -O3 = 22        -O3 + noipa = 22
allocate_chunk_path bodies:  13153 lines     13014 lines
  lock add inside caller:    0               0
```

At `-O3` the function is **already out-of-line** — 22 call sites — so `noipa`
had no inlining to forbid, and the caller contains **no** `lock add` at all,
so there is no inlined copy for a deletion to hide in.  Falsifier #4 at `-O3`
is a null, confirmed on the caller as §13.26 asked, not just on the symbol.

**(b), falsifier #4 on the proxy — fails its own verification step.**
Baseline established first:

| build | `orphan_chain_pop` symbols | their `lock add` |
|---|---|---|
| `-O2` | 44 | **44** |
| `-O2 -fipa-cp-clone` (the proxy) | 44 | **22** |

The deletion is real, and the restore target is 44.  Then, on the proxy, with
the attribute applied to `orphan_chain_pop`:

| attribute | resulting `lock add` |
|---|---|
| `noipa` | 22 |
| `optimize("O2")` | 22 |
| `optimize("no-ipa-cp-clone")` | 22 |
| `noclone` | 22 |

**None restores the increment** — though `noipa` changed 35 777 disassembly
lines elsewhere, so the attributes are being applied, they simply do not
affect this.

**What that tells us, and it is the useful part.**  If per-function
suppression on `orphan_chain_pop` cannot bring the increment back, the
deletion is **not** performed on `orphan_chain_pop` as a cloning target.
§13.24 already mapped the early add to `load_shared_`'s unconditional promote
inside the `lsp(asp&)` constructor of the chain-head load — i.e. it arrives
*inlined from a callee*.  IPA-CP specialising **that callee** would delete it
in the inlined copy, and no attribute on the enclosing function can prevent
it.

**So falsifier #4 should retarget**: apply the suppression to the callee that
supplies the increment — the `local_shared_ptr(atomic_shared_ptr&)`
constructor / `load_shared_` path in `atomic_smart_ptr.h` — and verify against
the same 22→44 census before running anything.  That is a header-inline
template, so the practical lever is likely `-fno-ipa-cp-clone` scoped to the
TU (already known to restore it) or an `optimize` attribute on the smart-pointer
member, not on the allocator function.

Standing practice (§13.23) has now caught three untestable falsifier arms in a
row — `-fno-indirect-inlining`, `noipa` at `-O3`, and all four attributes on
the proxy.  Each would have been reported as a refutation.

### 13.29 Falsifier #4, retargeted at the source: KAME_ASP_NOCLONE forbids
### cloning of the refcount-protocol members themselves

§13.28's conclusion implemented where it belongs (the smart-pointer
header): `-DKAME_ASP_NOCLONE` (gcc-only; expands to nothing on clang and
when unset) puts `__attribute__((noclone))` on the three protocol members
whose inlined copies supply the deleted increment —

- `atomic_shared_ptr<T>::load_shared_()`   (the unconditional promote),
- `acquire_tag_ref_()`,
- `release_tag_ref_()`.

Rationale: (b) proved no attribute on the ENCLOSING function can reach a
callee's specialisation; the specialisation to forbid is of the callees.
`noclone` on a callee removes IPA-CP's licence to make the `.constprop`
copy that gets inlined with the increment folded away, while leaving
ordinary inlining untouched (so the arm stays comparable — a `noipa`
here would also block inlining and change far more).

**Protocol, per §13.23's standing practice**: on the proxy
(`-O2 -fipa-cp-clone -DKAME_ASP_NOCLONE`), FIRST re-run the census —
expect the orphan_chain_pop family back at **44/44** lock adds; only if
it verifies, run the paired A/B (proxy vs proxy+knob).  Prediction
stays all-or-nothing: 19/35-class failing → 0 names the deleted promote
as the proxy's defect and completes the GCC-report package (asm exhibit
+ source read + call census + runtime flip, all on one no-argument
symbol family); census-unverified or still-failing gets reported as
exactly that.

Mac-side verification: clang builds with and without the define
(attribute correctly compiles away), pool Release build clean.  If the
knob verifies and flips, note it is a DIAGNOSTIC, not the mitigation —
§13.18's production flag stays until the root cause is fixed in gcc or
the protocol members carry the attribute permanently by decision.

### 13.30 Falsifier #4, retargeted and properly armed — refuted; the missing increment is real but not causal

First arm in four whose **census passes** before the run, exactly as §13.29
specified:

| build | `orphan_chain_pop` `lock add` |
|---|---|
| `-O2` (baseline) | 44 |
| `-O2 -fipa-cp-clone` (proxy) | **22** |
| proxy `+ -DKAME_ASP_NOCLONE` | **44** — restored |

That by itself confirms §13.28's inference: the deletion is performed on the
**callee's** specialisation (`load_shared_` / `acquire_tag_ref_` /
`release_tag_ref_`), not on `orphan_chain_pop`, which is why no attribute on
the enclosing function could reach it.

**The run, paired on the proxy pair:**

| arm | runs | failed | tripwires |
|---|---|---|---|
| `PXctl` | 11 | **11** | 8 |
| `PXasp` (increment restored) | 11 | **11** | 4 |

**100 % failure in both — refuted.**  And this refutation is trustworthy in a
way the last three arms were not: the knob demonstrably does the thing it was
built to do, verified before the batch started.

**So the §13.22 asymmetry is real but not causal.**  `-fipa-cp-clone` does
delete 22 refcount increments that no source-level constant justifies (§13.24
established that at source level, §13.25 excluded inline-to-call conversion,
§13.28 established the baseline).  Restoring every one of them changes
nothing.  Whatever those increments were protecting, the fault does not
depend on them.

Two secondary observations:

- **The proxy fails 11/11, not 19/35.**  §13.17 cites 19/35 for
  `-O2 -fipa-cp-clone`; here it is 100 %, the same as `-O3`.  Worth
  reconciling — it may be host/thread-count dependent (this batch is 40
  threads) — but it makes the proxy a *stronger* control than assumed, since
  a zero on the treated arm would have been unambiguous.
- **Tripwires halve (8 → 4) while failures do not move.**  The §13.21 pattern
  again: anomaly frequency responds to codegen knobs, the fault does not.
  Three separate knobs have now shown this shape (`-fno-devirtualize-
  speculatively` 25→13, `KAME_ASP_NOCLONE` 8→4, and the instrument versions
  themselves).  That is worth taking seriously as a finding in its own right:
  the tripwire counts a *symptom whose rate is tunable*, while the failure is
  not — which is hard to reconcile with the anomalies being the fault's
  direct cause, and easier to reconcile with both being downstream of
  something the knobs do not touch.

**Tally: six named mechanisms refuted by mechanical test.**  The §13.14
boundary has still never moved.  The one lead that came from measurement
rather than reasoning — §13.22's deleted increment — is now closed too.

### 13.31 The TSan-over-the-pool run is now THE move — decision tree

With six named mechanisms refuted and §13.30's observation that three
independent knobs tune the tripwire RATE without touching the failure
rate (the anomalies are a symptom cloud, the fault sits where the knobs
do not reach), mechanism-guessing has exhausted its ladder.  What
remains is the §13.21 dynamic audit, which enumerates instead of
guessing: TSan with the pool COMPILED IN and instrumented
(`-DKAMEPOOLALLOC_NO_LIBC_INTERPOSE`, no `.so`, no `KAME_RC_TRACE` —
recipe in §13.21).  The pool's `__sync`/`__atomic` ops are TSan-visible
sync, so claim/free handoffs form recognized happens-before; races on
POOL METADATA and on STM/user data alike get named with both stacks.

Practicalities: build the failing shape (gcc, `-O2 -fipa-cp-clone` or
`-O3` — TSan composes with either; keep 40 threads).  TSan is 5–15×
slower, but at 100 %-failure-per-run a handful of runs decides.
`TSAN_OPTIONS="halt_on_error=0 log_path=tsan_pool history_size=7"`.

Decision tree:
- **Reports on pool/STM PLAIN fields** → the enumeration we have been
  missing; fix per §13.20's classification (relaxed loads/stores), rerun
  — repeat until TSan-silent, then the reproducer verdict is clean.
- **TSan silent AND the run still fails** → the program is
  DRF-on-executed-paths with the pool included, and the compiled
  behavior is nonconforming: the gcc-bug verdict is effectively sealed
  (finish with the §13.20 static ratchet for the unexecuted-path
  caveat, and take the §13.22/§13.30 asm exhibits — real deletion,
  proven non-causal, but proof the pass rewrites this TU unsoundly-
  looking — into the report as supporting material).
- **TSan itself perturbs the fault away** (runs stop failing under
  TSan) → still informative: the fault needs a timing TSan destroys;
  record it and fall back to the static ratchet as the audit arm.
Also worth capturing while there: the non-tripwire death signature
(which assert / which SIGSEGV site) — 11 failures vs 8 tripwires says
some runs die without ever touching a tripwire, and that signature is
the closest thing we have to the fault's direct voice.

### 13.33 The §13.32 prerequisite, installed: TSan allocator annotations

The fourth state is closed.  Under a TSan-instrumented build (and ONLY
then — `__SANITIZE_THREAD__` / `__has_feature(thread_sanitizer)`, so
production codegen is untouched and the entry points keep their exact
names and shapes otherwise):

- **hand-out**: `new_redirected` gains a single-choke-point wrapper
  (`new_redirected_body_` conditional rename) issuing
  `__tsan_acquire(p)` — every route (freelist pop, L0 FIFO/STASH take,
  cold, large tail-call) returns through it, so one acquire covers all;
  `new_redirected_aligned` gets the same wrapper pattern at its
  definition.
- **free**: `PoolAllocatorBase::deallocate` — the one entry every free
  passes — issues `__tsan_release(p)` first thing (a release on a
  foreign pointer is a harmless extra edge).

This gives TSan the free→alloc happens-before that recycled addresses
need, which should collapse the 1 573 Zero-Location reports and the
vptr/atomic-vs-atomic impossibilities.  Verified on the Mac: default
Release build byte-for-byte unaffected path (ctest 18/18), the TSan TU
compiles with both gcc-style and clang-style gates, and the object
references both `__tsan_acquire` and `__tsan_release`.

Residual seams if Zero-Location noise persists (extend the same two
macros there): the LRC chunk-recycle push/pop (a region reused as a NEW
chunk without munmap keeps stale shadow), and realloc's grow-in-place
path (same identity, likely fine).  With this in place §13.31's decision
tree is live again — outcome 2 (TSan silent, run still fails) now
genuinely seals the gcc verdict for executed paths.

### 13.36 The two residual seams annotated — plus the third one both of us
### missed (re-carve granularity)

§13.35's discriminator, installed, with one addition.  Per-address
release/acquire cannot order a region that is REUSED AS A DIFFERENT
ALIGN CLASS: interior slot bases shift, so the new owner's acquire(base')
never pairs with the old owner's release(base).  Three edges close
everything:

- `deallocate_chunk` entry: `__tsan_release(chunk_base)` — the region
  identity leaves its old life; prior slot traffic happens-before this
  point via the bitmap/MASK_CNT atomics TSan already models, so the edge
  is transitive over all old accesses.
- `construct_chunk_at`: `__tsan_acquire(addr)` — every new chunk (fresh
  claim AND LRC-recycled) passes through here.
- `large_recycle_push`/`pop`: release(base)/acquire(base) — covers
  dedicated-origin regions whose only per-address release was the USER
  block (base+header), not the region base.

In-place realloc is left un-annotated deliberately (same identity, no
reuse).  Verified: TSan TU compiles with both symbols referenced;
default Release ctest 18/18.

Interpretation unchanged from §13.35: if the 10-survivor set — the
`~PacketWrapper()`-writes-into-recycled-storage shape at
`set_view ← Node::bundle:2948`, where the tracer and TSan now agree for
the first time — SURVIVES these edges, it is a genuine use-after-free
enumeration, the thing §13.21 was built to obtain.  If it collapses,
the survivor was chunk-recycle shadow, and the audit returns to
outcome-2 territory.

### 13.37 TSan-over-the-pool: run performed, but NOT interpretable without allocator annotations

Built and ran §13.31's recipe (gcc `-O3`, `-DKAMEPOOLALLOC_NO_LIBC_INTERPOSE`,
pool compiled in and instrumented, no `.so`, no `KAME_RC_TRACE`, 40 threads,
`20 40 700`).  It produces reports in volume:

```
1573  data race
  66  data race on vptr (ctor/dtor vs virtual call)
 217  distinct racing pairs, overwhelmingly in atomic_smart_ptr.h
```

**These cannot be entered into the decision tree as outcome 1, and the reason
is mechanical, not a judgement call.**

1. **Zero `Location is …` attributions.**  Across all 1573 reports TSan does
   not characterise a single racing address as heap / stack / global.  It has
   no allocation record for them — they come from the pool's `mmap`'d regions,
   which TSan never saw allocated.
2. **The pool carries no TSan annotations.**  `grep` for `__tsan_acquire` /
   `__tsan_release` / `ANNOTATE_HAPPENS_BEFORE` over `allocator.cpp` and
   `allocator_prv.h` returns nothing.

Without those, a custom allocator makes TSan report a **false race for every
recycled address**: it cannot know that `free → alloc` orders the previous
owner's accesses before the new owner's, so the new object's writes race the
old object's reads in stale shadow.  The 66 vptr races are the textbook
signature — e.g. `PayloadWrapper`'s copy constructor writing a vptr at
`0x7fffd4001010`, reached from `Transaction::operator[]` → `clone` → a fresh
`make_local_shared`, i.e. a **newly constructed object at a recycled
address**, which is precisely what this reproducer does thousands of times a
second.

A further tell: many of the top pairs are `<atomic_base.h:358>` against
`:477/:501/:641` — i.e. TSan reporting races *between `std::atomic`
operations*.  TSan models atomics correctly and does not report those; seeing
them means the shadow state for those addresses is not trustworthy.

**What is needed before this arm can decide anything** — small, standard, and
entirely inside the pool:

```c
#if defined(__SANITIZE_THREAD__)
  /* on hand-out */   __tsan_acquire(p);
  /* on free      */  __tsan_release(p);
#endif
```

on the slot alloc/free paths (and the chunk claim/release paths), so recycled
memory carries a happens-before edge TSan can see.  §13.31's own framing —
"the pool's `__sync`/`__atomic` ops are TSan-visible sync, so claim/free
handoffs form recognized happens-before" — holds for the *metadata* words, but
not for the **payload bytes**, which is where object reuse lives and where
every one of these reports sits.

**So the decision tree is not yet entered.**  This is neither outcome 1
(reports we can act on), nor outcome 2 (TSan silent), nor outcome 3 (TSan
perturbs the fault away — the run still fails).  It is a fourth state:
instrument not yet valid for this configuration.  Annotating the pool is the
prerequisite, and it is a genuinely small change.

Recorded also because §13.11's *pool-less* TSan silence remains the only
trustworthy TSan datum so far, and its caveat stands unchanged: it was
measured in the configuration where the fault never manifests.

### 13.38 Annotated TSan-over-the-pool — the false-race class collapses, and outcome 1 arrives

§13.33's annotations rebuilt and **verified before running** (§13.23): 14 call
sites to `__tsan_acquire`/`__tsan_release` in the binary, 2 undefined symbols.
(A first build reported 0 — it predated the merge; the verification caught it.)

**The false-race class collapses as predicted:**

| build | TSan warnings, `20 40 700` |
|---|---|
| unannotated (§13.34) | **5 412** |
| annotated | **10** (7 data race + 3 vptr) |

**The dominant survivor is a use-after-free, and it is the same shape the
tracer has been reporting since §1:**

```
Write      T74:  Node::Node()  transaction_impl.h:1320   ← vptr of a NEW LongNode
                 at 0x7fff90000920
Prev write T65:  lsp<Packet>::reset()  atomic_smart_ptr.h:1903   ← SAME address
                 ~lsp<Packet> ← ~PacketWrapper()  transaction.h:915
                 ← deleter(PacketWrapper*)  ← release_tagheld_zeroreset_  :1661
                 ← release_()  ← assign_from_local  :1420
                 ← ScopedNegotiateLinkage::set_view  transaction_negotiation.h:877
                 ← Node::bundle  transaction_impl.h:2948
```

T65 runs `~PacketWrapper()`, whose member `~local_shared_ptr<Packet>` writes
into the wrapper's **own storage**, while T74 constructs a **new `Node` at that
address**.  With `__tsan_release` at deallocate entry and `__tsan_acquire` on
hand-out, a correct `destruct → free → alloc → construct` sequence is fully
ordered and would not be reported.  A report means the wrapper's destructor is
touching storage **already recycled to a new owner**.

Note where it is reached from: `bundle → set_view → assign_from_local →
release_ → release_tagheld_zeroreset_`.  That is the same
`scoped_atomic_view` release path §12.4/§12.5 kept landing on from the
tracer side, arrived at here by a completely independent instrument.

**Caveat, stated because it is the one thing that could still explain it
benignly**: locations are still un-attributed (`Location is …` absent in all
10), so the residual seams §13.33 documents — LRC chunk recycle and in-place
realloc, both un-annotated — remain possible sources.  A wrapper freed through
the chunk-recycle path rather than the slot path would not carry the
`__tsan_release` edge, and would produce exactly this report.  **Annotating
those two seams is the next step**, and if the report survives it, this is the
enumeration the whole §13.21 line was after.

Remaining survivors for completeness: `atomic_base.h:358 ↔ transaction.h:2614`,
`shared_ptr_base.h:1073 ↔ tmin_dynnode.cpp:46`, `atomic_smart_ptr.h:937 ↔
transaction_impl.h:1276`, `:937 ↔ transaction.h:252`, and three pairs internal
to `atomic_smart_ptr.h` (`:1810`, `:1037`, `:950`, `:501`).

### 13.39 All seams annotated — noise 5412 → 2, and the survivor is `~PacketWrapper` on recycled storage

Rebuilt with §13.36's seam annotations, verified first (§13.23): **91**
`__tsan_acquire`/`__tsan_release` call sites, up from 14 with slot-only edges.

| build | TSan warnings, `20 40 700` |
|---|---|
| unannotated | 5 412 |
| slot annotations | 10 |
| **+ chunk teardown/construct, LRC push/pop, re-carve** | **2** |

**The `~PacketWrapper`-into-recycled-storage report survives all three seam
classes.**  Both survivors are the same object (`0x7fffa0002a20` /
`…2a28`, adjacent words):

```
Write  T78 @0x7fffa0002a20   PacketList_::PacketList_(const&)   transaction.h:104
                             (copying the m_subnodes shared_ptr)
                             <- Node::bundle   transaction_impl.h:2870

Prev   T72 @0x7fffa0002a20   atomic_countable::~atomic_countable  atomic_smart_ptr.h:440
       (atomic read of       (reads refcnt)
        refcnt)              <- ~PacketWrapper()            transaction.h:915
                             <- deleter(PacketWrapper*)     atomic_smart_ptr.h:756
                             <- lsp<PacketWrapper>::reset() atomic_smart_ptr.h:1897
                             <- ~local_shared_ptr           atomic_smart_ptr.h:1856
                             <- Node::bundle   transaction_impl.h:2840
```

**Both stacks are inside `Node::bundle`, thirty lines apart** — `:2840`
releasing a wrapper, `:2870` copy-constructing a `PacketList_`.  One thread's
wrapper destructor is reading `refcnt` in storage another thread has already
been handed and is constructing into.

**Why this is now hard to dismiss.**  The destructor chain runs *before* the
free (`deleter` → `~PacketWrapper` → `~atomic_countable`), so in a correct
sequence that read strictly precedes the `__tsan_release` at deallocate and
strictly precedes any later `__tsan_acquire` on hand-out.  For TSan to report
it with all four annotation classes installed, the block must have been
**freed and re-handed-out before this destructor ran** — i.e. a second
destruction of an object already released, which is exactly the
two-holders-one-count shape Q1 established 15/15 and §1 saw as a live wrapper
pointing at freed storage.

**Convergence worth stating plainly.**  Three independent instruments now name
the same site: the refcount tracer (§12.4/§12.5, `~PacketWrapper` releasing
`m_packet`), the forensic free-record (§13.5, `~PacketWrapper()` 6/14 of free
chains), and annotated TSan (here).  No mechanism hypothesis produced this —
all six were refuted — but three measurement instruments agree on the
location.

**Remaining caveat**: locations are still un-attributed, and `bundle` is the
hottest function in the reproducer, so coincidental address reuse inside one
function cannot be excluded from a single run.  The batch is running to six;
if both survivors recur with `bundle:2840` / `:2870` across runs, that is the
enumeration §13.21 was built to produce.

### 13.40 Correction to §13.39 — the "2 survivors" was a partial log; the real count is ~51, still scattered

**§13.39's headline number was wrong and the conclusion drawn from it does not
stand.**  The "2 warnings" was read from a log while the run was still
executing.  Run 1 completed with **51** (46 data race + 5 vptr); run 2 reached
20 before this was written.  So the collapse is **5 412 → ~51**, not → 2.

**And the survivors do not converge.**  44 distinct pairs across 51 reports,
maximum 3 occurrences for any pair:

```
  3  atomic_base.h:358  <->  atomic_base.h:477      <-- atomic vs atomic
  3  atomic_base.h:501  <->  transaction.h:104
  2  atomic_smart_ptr.h:1903  <->  transaction.h:980
  2  atomic_base.h:358  <->  atomic_smart_ptr.h:950
  2  atomic_base.h:501  <->  transaction_impl.h:1276
  … 39 more pairs at 1 each
```

That is a long tail, the same shape §11.3 and §13.34 showed — not a converged
finding.  §13.39 read two reports out of fifty-one and described them as "the
survivor".

**The `atomic_base.h:358 ↔ :477` pair is the diagnostic one**: those are
`std::atomic` load and store.  TSan models atomics and never reports races
between them.  Its presence means the shadow state for those addresses is
still untrustworthy — i.e. **annotation coverage remains incomplete** even
with §13.36's chunk, LRC and re-carve edges.  This is exactly the tell §13.34
used to disqualify the unannotated run, and it has not gone away; it has only
become rarer.

**What survives from §13.39, and what does not.**  The `~PacketWrapper` /
`bundle:2840` vs `bundle:2870` report is real and did appear (9 and 7 line
occurrences in run 1).  What does not survive is the inference that it is *the
survivor* and therefore the enumeration — it is one pair among 44, in a set
that still contains provably impossible reports.

**Consequence for §13.31's decision tree**: still not entered.  Outcome 1
requires reports we can act on, and a set containing atomic-vs-atomic
impossibilities cannot be acted on selectively without a principled way to
tell the real ones from the residue.  The remaining un-annotated seam should
be found first — the atomic-vs-atomic pairs are the trail, since each names an
address whose shadow is wrong.

Recorded at length because this is the second time in this section a partial
or unverified read produced a conclusion that had to be withdrawn (§13.21's
flag that did nothing, and now this).  Both were caught, but only after being
reported.

### 13.41 The TSan-over-the-pool avenue has a ceiling: no shadow reset on reuse, so it cannot converge

The six-run batch finished: **51, 44, 43, 48, 21, 48** warnings (all `rc=134`).
Aggregated: **255 reports over 105 distinct pairs, 64 of them singletons**,
22 atomic-vs-atomic.  Eight pairs recur in >=5 of 6 runs, led by
`atomic_smart_ptr.h:1903 <-> transaction_impl.h:1320` (6/6, 12x).  So the
residue is *stable*, not run-to-run noise — which pointed at a specific
un-annotated seam, per §13.40.

It is not a seam.  Two mechanical checks close this avenue:

**(a) Zero `Location is` attributions in all 255 reports** — same as the
unannotated run (§13.34).  And the impossible pairs and the ordinary pairs
draw from *the same address prefixes* (`0x7fff70..`–`0x7fffd4..`); there is no
region that is attributed and another that is not.  TSan has no allocation
record for any pool address.

**(b) The public TSan interface has no shadow-reset call.**  Checked against
`/usr/lib/gcc/x86_64-linux-gnu/15/include/sanitizer/tsan_interface.h`: the
exported user API is `__tsan_acquire`/`__tsan_release`, the `__tsan_mutex_*`
family, the fiber family, and report accessors.  Nothing equivalent to
`__tsan_malloc`/`__tsan_free`.  `common_interface_defs.h` offers only the
contiguous-container annotations (an ASan-side facility).

Together those give the mechanism.  `__tsan_acquire`/`__tsan_release` add
happens-before *edges*; they never reset *shadow*.  When a byte is freed by
thread A and re-carved to thread B, B's write meets A's stale shadow entry
for that address — a race is reported between two accesses that never
coexisted, and it reproduces stably because the carve pattern is stable.
That is exactly the observed signature: repeatable atomic-vs-atomic pairs,
no location attribution, a long singleton tail.  §13.36's seam annotations
were not wrong, and they did real work (5412 -> ~45); they are simply
**insufficient in kind**.  No number of additional acquire/release edges
resets shadow, so adding more cannot drive the residue to zero.

This corrects §13.40's reading.  The problem is not that annotation coverage
is incomplete — it is that the available annotation *primitive* cannot express
"this storage is fresh".  Under that ceiling, **no report in the set is
individually trustworthy**, including the `~PacketWrapper` / `bundle:2840` vs
`:2870` pair.  §13.31's decision tree therefore stays un-entered, and
outcome 1 was never actually reached.

**The one escape hatch, and its cost.**  `libtsan.so` *does* export a
shadow-resetting pair — `__tsan_java_alloc` / `__tsan_java_free`
(`MemoryResetRange` / `MemoryRangeFreed` internally), plus `__tsan_java_move`.
They ship without a header on this install but are declarable by hand.  The
blocker is `__tsan_java_init(heap_begin, heap_size)`: it registers **one
contiguous** heap span, and every annotated pointer must fall inside it.
kamepoolalloc does not have one — it takes repeated 32 MiB regions
(`ALLOC_MIN_MMAP_SIZE`, tracked per-index and derived by masking any in-region
pointer), at whatever addresses mmap returns.  Using the java API therefore
requires a TSan-only build mode that reserves a single large VA span up front
and carves all regions from it (`PROT_NONE` reserve + commit-on-demand), so
that `__tsan_java_init` can cover it.  That is a real change to the region
layer, not an instrumentation add-on.

**Recommendation.**  Do not spend further effort widening acquire/release
coverage — the ceiling is structural.  Either commit to the single-arena TSan
build so `__tsan_java_alloc`/`_free` can reset shadow on carve and recycle, or
drop TSan and return to the differential evidence that has never moved: the
`-O3` / `-fipa-cp-clone` separation (§6, §7), the two-holder-slot result
(Q1 15/15 `DIFF`), and `drift=+0` 14/14.  Of the two, the differential line is
cheaper and has produced every boundary that still stands; the TSan line
would need the arena work before it produces anything admissible at all.

### 13.42 The escape hatch, built: single-arena TSan mode with
### __tsan_java shadow reset

§13.41's fork, taken on the arena side — the differential line is
mechanism-starved (six refutations, boundary never moved), while the
enumeration line dies only for want of a shadow-reset primitive; the
region-layer change that unlocks it is contained, and here it is.

Under `KAME_TSAN_ENABLED` (and POSIX) only — production and plain builds
byte-identical, gated exactly like §13.33:

- **One arena**: first region claim reserves a PROT_NONE,
  32 MiB-aligned span (default 64 GiB VA, `KAME_TSAN_ARENA_GB`
  overrides; `MAP_NORESERVE`), registers it ONCE with
  `__tsan_java_init`, and both `mmap_new_region` (pool regions) and
  `large_va_raw_map` (large tier) carve 32 MiB-granule spans from it
  (mutex + small freelist + bump; carve = `mprotect(RW)`).
  `large_va_raw_unmap` returns spans with `madvise(DONTNEED)` — VA and
  shadow persist, pages are given back, and released-chunk headers stay
  READABLE for the concurrent-lookup contract.
- **Shadow resets**: `__tsan_java_alloc(p, size)` at both hand-out
  wrappers (`new_redirected`, `new_redirected_aligned`) and over the
  whole region at `construct_chunk_at` (covers re-carve into a different
  ALIGN class); `__tsan_java_free(p, size_of(p))` at `deallocate` entry,
  before the forensic poison overwrites anything.  All java calls are
  RANGE-GUARDED to the arena, so libc-fallback pointers never reach the
  java API (which would CHECK-fail).
- The §13.33/§13.36 acquire/release edges stay — the java calls reset
  shadow, the edges order genuine handoffs; they compose.

Mac verification: TSan TU compiles and a full TSan link resolves all
three `__tsan_java_*` from the runtime; default Release build unchanged
(ctest 18/18).  Runtime validation belongs to Ubuntu (macOS 26 TSan is
dead, §13.10): expect the Zero-Location class to vanish entirely — every
pool address now has java-heap standing — and the report count to become
the true enumeration.  §13.31's decision tree re-enters service with
outcome 1/2 finally distinguishable; §13.38's `~PacketWrapper` pair is
the first thing to look for in the survivor set.

### 13.43 §13.42's arena never activates under the §13.10 recipe — and neither recipe gives TSan over a live pool

§13.42 built the single-arena mode I asked for in §13.41.  It is correct code,
but **it does not run** in the binary the TSan work has been using, so it
changed nothing.  Measured, not inferred.

**The null result first.**  Rebuilt `tmin_dynnode` exactly as §13.10 specifies
plus `-DKAME_TSAN_ENABLED=1`, verified the annotations linked (**40
`__tsan_java_alloc`, 3 `__tsan_java_free`, 1 `__tsan_java_init`** call sites;
the pre-arena binary has 0).  Two runs of the six-run batch before I stopped
it: **52 and 40** warnings, **0** `Location is` — statistically identical to
the pre-arena 51/44/43/48/21/48, with the atomic-vs-atomic pairs at the same
~10% share.

**Why.**  Direct `fprintf` probes at the entry of `kame_tsan_arena_map()` and
`PoolAllocatorBase::mmap_new_region()` (patched, measured, reverted):

| build | pool `operator new` | `mmap_new_region` | arena |
|---|---|---|---|
| §13.10 inline (`NO_LIBC_INTERPOSE`), TSan | **live** | **never entered** | never |
| §13.10 inline, no TSan | live | **never entered** | n/a |
| `.so` (`KAMEPOOLALLOC_DYLIB`), no TSan | live | **entered** | n/a |
| `.so`, TSan | — | never entered | never |

Corroborated externally: no 64 GiB `PROT_NONE` reservation in
`/proc/<pid>/maps` and no `mmap(…, 68753031168, …)` in an `strace -f` of the
run.  Since `g_tsan_arena_base` stays null, `kame_tsan_arena_contains()` is
false for every pointer, and **all 40 `JAVA_ALLOC`/`JAVA_FREE` sites are
unconditionally no-ops** — they are guarded by that predicate.  The same
holds for any §13.33/§13.36 edge placed inside the region path.

**And the `.so` recipe is not the fix.**  A clean A/B — same source, same
workload (`10 8 400`), only `-fsanitize=thread` differing — gives
`mmap_new_region entered` without TSan and **0 warnings, no pool activity**
with it.  In the `.so` layout libtsan's `operator new`/`delete` (linked
first) preempt the pool's; in the inline layout the executable's definitions
win, which is why the inline build's pool `operator new` *is* live.  So:

* inline → pool `operator new` live, region/chunk path dead;
* `.so` → pool bypassed entirely.

Neither configuration is "TSan over a live pool", which is what §13.10
promised and what §13.33 onward assumed.

**Open question I did not resolve.**  If the inline build's pool `operator
new` is live but `mmap_new_region` never runs, its chunk backing comes from
some other path.  I did not find it, and I am not going to assert the pool
is "unused" — the honest statement is narrower: *the region/arena path is
dead in that build, so annotations placed there cannot fire.*

**Two incidental defects.**
1. `allocator.cpp` uses `pthread_mutex_*` / `PTHREAD_MUTEX_INITIALIZER` in the
   §13.42 block without including `<pthread.h>`.  It compiles only when the TU
   drags pthread in (`support_standalone.h` does); a bare
   `g++ -I kamepoolalloc allocator.cpp` fails with five errors.  One-line fix.
2. `-Wtsan`: **`atomic_thread_fence` is not supported under
   `-fsanitize=thread`.**  The allocator has 6 `writeBarrier()` call sites and
   **0** `readBarrier()` ones, so fence-published data (`m_idx` at :1599,
   `m_owner_id` at :4463/:4495, `LargeAllocMeta` at :8149) is invisible to
   TSan's model no matter how the allocation is annotated.  This is a second
   ceiling independent of §13.41's, and it would bite even after the arena
   works.

**Also worth recording:** `rc=134` in every TSan run is glibc `free():
invalid size`, **not** the test's own value-check assertion — it is present
in the pre-arena binary too, so the TSan runs have been terminating on heap
corruption rather than on the STM detector this whole time.

**Recommendation unchanged from §13.41, now with a cost estimate.**  Making
TSan viable needs (a) a build where the pool is the sole `operator new` *and*
its region path is live, (b) the arena from §13.42 actually reached, and
(c) the fence sites converted to load-acquire/store-release so TSan can see
the ordering.  That is three fixes before the first admissible report.  The
differential line (`-O3` / `-fipa-cp-clone`, two-holder-slot 15/15 `DIFF`,
`drift=+0` 14/14) still costs nothing and still holds.

*Process note:* twice in this section I started building a conclusion on an
under-powered probe — a tiny workload without `strace -f`, then a global
`new char[48]` that never reaches the pool — and had to retract mid-
investigation.  Both were caught by a control run before reporting, which is
the only reason they are footnotes rather than another §13.40.

### 13.44 §13.43's three blockers, fixed — and the answer to its open question

**The open question is answered: the pool was never ACTIVATED.**  Inline
(non-dylib) mode requires an explicit `activateAllocator()` call —
production does it via `KamePooledAllocGuard` in kame/main.cpp, the old
standalone tests via a static shim — and the §13.10 recipe contains
neither.  So `g_sys_image_loaded` stayed false, every `operator new` took
the pre-activation FALLBACK (the symbol is "live", the pool is inert),
`mmap_new_region` never ran, and §13.42's arena sat behind a door that
was never opened.  Every TSan-over-the-pool number since §13.34 —
5 412, 45, 255, 52/40 — was measured against an inert pool.  (This also
reframes the `rc=134` = glibc `free(): invalid size` observation: those
crashes happened on a libc-backed run, which deserves its own §: see the
note below.)

Fixes, all in this commit:

1. **TSan builds auto-activate.**  Under `KAME_TSAN_ENABLED`, inline mode
   gains the same `constructor(101)` auto-activation as the dylib mode —
   an analysis build with an inert pool measures nothing, so the whole
   "recipe forgot the shim" class is now impossible for this line.
   (Non-TSan inline builds keep the explicit-activation contract.)
2. **`<pthread.h>` included** by the §13.42 arena block directly.
3. **Fence visibility** (`-Wtsan`, the second ceiling): under a TSan
   build only, `writeBarrier()` / `readBarrier()` / `memoryBarrier()`
   additionally release/acquire a process-wide proxy token
   (`kame_tsan_fence_token_` in atomic_mfence.h, self-contained gate),
   and `atomicLoadRelaxed()` acquires the token after its load — so the
   fence-published fields §13.43 lists (`m_owner_id`, `m_idx`,
   `LargeAllocMeta`) are ordered in TSan's model.  Conservative by
   construction: proxy edges can only HIDE races along orderings the
   fences already claim, never invent one.

Mac verification: TSan TU compiles (java + acquire/release + token all
referenced), bare `clang++ -c allocator.cpp` with no harness compiles
(defect 1 gone), default Release ctest 18/18.

**Worth a run of its own once TSan works**: §13.43 noted every TSan run
died `rc=134` = glibc `free(): invalid size` — on what we now know was a
LIBC-BACKED binary.  If that reproduces on a pool-INERT, TSan-OFF build
(§13.10 recipe minus `-fsanitize=thread`, activation still absent), the
fault corrupts the GLIBC heap with the pool bypassed — which would be a
major boundary shift all by itself (the §13.14 "allocator codegen"
boundary would need re-reading, since the allocator's pool paths would
be out of the loop).  One cheap batch answers it.

### 13.45 §13.44 verified on Ubuntu: the pool is live, attribution works, and the first admissible report is `m_owner_id`

All three §13.43 blockers confirmed fixed on Linux/g++ 15.2, each checked
against the behaviour it was supposed to move (not just "it built"):

1. **pthread include** — `g++ -I kamepoolalloc -c allocator.cpp` with no
   harness: **0 errors** (was 5).
2. **Auto-activation + arena** — rebuilt `tmin` to the §13.10 recipe plus
   `-DKAME_TSAN_ENABLED=1`: `strace -f` now shows the
   **64 GiB reservation (`mmap(…, 68753031168, …)`) exactly once**, where
   §13.43 measured zero.  The pool's region path is live.
3. **Attribution** — the decisive one.  Every report now carries
   **`Location is heap block of size 262144 …`** naming the allocating stack.
   All 255 pre-§13.44 reports had **zero** location lines (§13.41(a)); that
   was the tell for stale shadow, and it is gone.

**The first admissible finding.**  At `5 4 200` the run completed cleanly
(`succeeded`) with **2 warnings, 2 locations — and both are the same pair**:

* **Write of 4 B**, `PoolAllocator<48u,true,true>::release_dll_chunks_for_thread()`
  `allocator.cpp:3366` ← `~AllocThreadExitCleanup` ← `__call_tls_dtors`
  — the plain store `c->m_owner_id = 0;` on the **non-empty orphaned-chunk**
  branch at thread exit.
* **Previous atomic read of 4 B**, `atomicLoadRelaxed<unsigned>`
  (`allocator_prv.h:346`) in `PoolAllocatorBase::deallocate` **`allocator.cpp:3855`**
  — the deallocate **fast-path routing test**
  `atomicLoadRelaxed(&chunk_obj->m_owner_id) == page_owner_id && page_owner_id != 0`
  — reached via `operator delete` ← `atomic_shared_ptr_base<PacketWrapper>::deleter`
  ← `scoped_atomic_view::release_` ← `assign_from_local` ← `set_view` ←
  **`Node::bundle`** (`transaction_impl.h:2835`) in one report, and
  ← `PayloadWrapper::~PayloadWrapper` (`transaction.h:231`) in the other.

The two reports involve **different threads, different chunks, and different
STM callers**, and in the second the chunk was allocated by a *third* thread —
so this is one mechanism, not one unlucky object.

**Why it is a real race, not an annotation gap.**  A plain 4-byte store
concurrent with any other access to the same location is a data race by the
model; the in-tree comment ("`atomicFetchAnd` provides a full barrier
ordering this store") justifies *ordering*, but a barrier orders the
operations around it — it does not make the store itself atomic, and it is
invisible to TSan besides (§13.43(2)).

**Why it is a plausible fault, not just a defect.**  The routing test's
*intended* failure direction is safe: the comment states a zeroed
`m_owner_id` fails the test and tail-calls the cold path, which re-validates.
The unsafe direction is the reader observing the **stale non-zero** owner id
and taking the owner-assumed fast path into a chunk whose ownership is being
handed to the orphan chain. That is a free-path mis-route — the mechanism
class that produces both the "released after count reached zero" signature
and libc heap corruption.  A **relaxed** load is also precisely what `-O3`
may cache or duplicate across the branch, which is how a fault acquires
codegen sensitivity.  It further coincides with the independently-derived
teardown timing (§12: 13/15 under 2 µs, median 1.0 µs) and with §13.20's
falsifier #3, which predicted post-publication `m_owner_id` writes.

**Boundary test §13.44 asked for — negative, and that is good news.**
Pool-inert, TSan-OFF, `20 40 700` × 12: **12/12 clean**, no
`free(): invalid size`, no failures.  So **§13.14's "requires the pool"
boundary is intact** and needs no re-reading.  It also localises §13.43's
`rc=134`: with the pool inert, glibc corruption appears only when TSan is
present, so that abort was an artifact of the inert-pool + TSan
configuration, not the fault.

**Status / next.**  A six-run `20 40 700` batch is in flight; at that scale
the report count is larger than 2, so **convergence is not yet claimed** —
the small-workload result is what is verified here.  The decisive experiment
is differential, not more TSan: make the `:3366` store a release atomic (and
the `:3855` load acquire), then re-run the **original** `-O3` + `.so` + gcc
reproducer and see whether 42/42 failures moves. Only that tests causation.

### 13.46 The §13.45 differential, implemented: the m_owner_id handoff
### goes release/acquire (falsifier #5 — and a correct fix regardless)

The store and read §13.45 attributed are now a proper atomic pair, plus
every other POST-PUBLICATION `m_owner_id` write found by census:

- release stores (`atomicStoreRelease`, new shim beside the load
  helpers; MSVC arm = volatile store, its idiomatic release):
  the §13.45 store itself (`release_dll_chunks_for_thread`, non-empty
  orphan branch), the empty-branch zero, the DLL-neighbour zero, the
  orphan-ADOPT re-arm (`oc->m_owner_id = kame_owner_id()`), and both
  dedicated/recycle restamps.  The construction-time stamp (:1304) stays
  plain — pre-publication, correctly so.
- acquire loads at both deallocate routing tests
  (`atomicLoadAcquire`, which also takes the §13.43 fence-proxy token
  under TSan).  x86: mov, zero cost.  arm64: `ldar` on the free fast
  path — accepted pending the verdict; revisit if the reproducer flips.

Two things this is, one thing it is not:
1. **A correct fix regardless of the verdict** — a plain 4 B store
   racing an atomic load is UB, full stop; §13.45's reasoning about the
   dangerous direction (stale NON-ZERO owner id routing a foreign free
   down the owner fast path mid-handoff) stands on its own.
2. **The pre-registered causation experiment**: rebuild the ORIGINAL
   failing configuration (gcc `-O3`, `.so`, forensic, no TSan) from this
   commit and re-run the 42/42 batch.  All-or-nothing as always:
   → 0 = the handoff race is THE fault; the mechanism is named
   (§13.45), the §13.18 mitigation can be re-examined, and the story
   closes as "post-publication plain store, exposed by -O3 codegen" —
   note falsifier #2 (§13.17) had already atomicized the READ side and
   was refuted, so the causal half was the WRITE side (or the acquire
   ordering), which is consistent, not contradictory.
   → still 42/42 = the UB fix stays, the enumeration continues on the
   now-live pool (§13.45's six-run batch), and the survivor list is the
   worklist.
3. It is NOT a full DRF certificate — TSan's live-pool enumeration
   remains the authority for what else is racing.

Mac: both compile modes clean (TSan TU + bare), default Release ctest
18/18; remaining plain owner writes are the construction stamp only.
### 13.47 First real enumeration: 104 reports → 5 pairs, zero impossible. Two mechanisms, both in the free/reuse path

Run 1 of the live-pool batch (`20 40 700`) completed — **`rc=66`**, which is
TSan's "warnings were reported" exit, i.e. the run finished normally.  The
`free(): invalid size` abort that ended every pre-§13.44 run is **gone**,
consistent with §13.45's finding that it was an artifact of the inert-pool +
TSan configuration.

**The report set is finally an enumeration:**

| | pre-§13.44 (§13.41) | now |
|---|---|---|
| reports | 255 | 104 |
| **distinct pairs** | **105** | **5** |
| singletons | 64 | 3 |
| atomic-vs-atomic (impossible) | 22 | **0** |
| with `Location is` | **0** | 80 |
| vptr races | 5 | 0 |

Five pairs, and 101 of the 104 reports fall in just two of them.

**Pair A — 61×: slot header written while the same slot's size is read.**
Addresses **overlap**: write of 8 B at `…0f8`, previous read of 4 B at `…0fc`.

* read (first, T29): `size_of_static` `allocator.cpp:2060`
  (`*(uint32_t*)(p-4)`) ← `PoolAllocatorBase::size_of` `:4281`
  ← **`deallocate` `:3801`** ← `operator delete` ←
  `fast_vector<lsp<Packet>>::destroy` (`fast_vector.h:231`) ←
  `PacketList_::~PacketList_` (`transaction.h:105`) ← `Packet::~Packet`
  (`transaction.h:252`) — the recursive `local_shared_ptr::reset()` teardown.
* write (after, T3): `allocate_pooled` `allocator.cpp:1848`
  (`*(uint64_t*)(slot_start-8) = hdr_word`) ←
  `fast_vector<lsp<Packet>>::operator=` (`fast_vector.h:96`) ←
  **`PacketList_` copy ctor** (`transaction.h:104`) ←
  `reverseLookupWithHint` (`transaction_impl.h:1759`) ← `reverseLookup` ←
  `Transaction::operator[]`.

Both sides are `fast_vector<local_shared_ptr<Packet>>` inside `PacketList_`
— **habitat 3** of §13.28, the one never audited beyond single-thread
semantics.  One thread frees a slot and another is handed it with **no
happens-before edge TSan can see**.  `deallocate` reads the size header to
route the free; if that header has already been rewritten by a concurrent
allocation, the free routes to the wrong size class — freelist/bitmap
corruption, which is the "released after count reached zero" signature and a
libc-visible corruption both.

*I am not calling this the fault yet.*  It is one of two things, and the
allocator owner should adjudicate: **(i)** a genuinely missing release/acquire
on the cross-thread free → reuse handoff (a real bug: the reusing thread may
observe stale contents), or **(ii)** a remaining annotation gap on that
specific handoff path, the §13.36 seams not covering free→realloc.  The
overlapping-address detail favours (i) — the two accesses touch the *same
header word*, not merely the same cache line — but that is an argument, not
a proof.

**Pair B — 40×: `m_owner_id`,** exactly as §13.45 described from the
small-workload run (`:3366` plain store at thread exit ↔ `:3855` relaxed load
in the deallocate routing test).  It reproduces at scale.  Unlike Pair A this
one is unambiguous: a plain 4-byte store concurrent with any access is a race
by the model, no annotation question arises.

**Three singletons:** `:8354 ↔ prv:252`, `:8354 ↔ prv:240`,
`:3004 ↔ prv:346`.  Left un-analysed pending more runs — with 1 occurrence
each they may not recur.

**Next.**  Pair B is cheap and unambiguous, so test it first and
*differentially*: make `:3366` a release store and `:3855` an acquire load,
then re-run the **original** `-O3` + `.so` + gcc reproducer against the 42/42
baseline.  If the rate moves, causation is established; if not, Pair B is a
real defect that is not this fault, and Pair A becomes the candidate.  Runs
2–6 are in flight to confirm the 5-pair structure is stable.

### 13.48 Pair A adjudicated: two defects, two commits (bisectable)

The allocator-owner read §13.47 asked for.  The overlap arithmetic
(write 8 B @…0f8 ⊇ read 4 B @…0fc, with the read = `*(uint32_t*)(p-4)`
and the write = `*(uint64_t*)(q-8)`) forces **p == q: both sides touch
the SAME slot's header word** — the freer routing slot q's free while a
claimer initialises slot q's header.  Verdict: **(i), a real defect —
in fact two**, shipped as two commits so the causal batch can bisect:

**(A) The relaxed claim load is the missing edge.**  §13.15's relaxed
conversion bought load integrity, not ordering: a claimer that observes
a freed bit through a RELAXED `m_flags` load gets visibility without
happens-before, so its header write is unordered against the freer's
last header read — exactly Pair A as TSan states it, and formal UB even
where hardware behaves.  All four CAS-loop loads go relaxed → acquire
(free on x86; arm64 pays `ldar` in the claim loops, accepted pending
the verdict).

**(B) The pre-CAS header write is a store to unowned memory, and it can
clobber a LIVE slot.**  Two claimants computing candidates from the same
`oldv` pick overlapping runs (a lowest-zero-run start is shared;
constant-N clones make different-N claims of one chunk routine common at
-O3).  Both write their `{bucket,SIZE}` header BEFORE their CAS; the
loser's store can land AFTER the winner's CAS — the winner's live slot
now carries the LOSER's size, and its eventual free mis-routes (wrong
size class → freelist/bitmap corruption → the released-after-zero
signature and libc-visible damage).  The old "write before the CAS so
the CAS publishes it" rationale does not hold up: every header reader
reaches the slot through the allocator's RETURN VALUE plus an
application-level synchronized handoff, which orders a post-CAS write
just as well.  The metadata now goes in AFTER CAS success.

Note what (B) is: a plain LOGIC bug — no UB required, timing-window
dependent (contention on one m_flags word with overlapping candidates),
which is exactly the kind of window that codegen reshaping (ipa-cp-clone
constant-N specialisation) widens or narrows without being the "cause"
— consistent with every knob-tunes-rate-but-not-failure observation
(§13.30) IF (B) is the fault, and with the -O2/clang total cleanliness
if their windows round to zero.

**Causal protocol (all-or-nothing, as always):** re-run the ORIGINAL
gcc `-O3` + `.so` + forensic reproducer at HEAD (A+B+§13.46's Pair-B
fix all in).  If 42/42 → 0: revert one commit at a time to name which
of the three carries it.  If still failing: all three stay as correct
fixes, and the TSan enumeration (runs 2–6, the three singletons) is the
worklist.  Mac: ctest 18/18 both builds, tmin 3×40t clean on the
patched pool.

### 13.49 Both TSan-derived candidates refuted by differential test: 18/20 vs 13/20 (p=0.13), then 20/20 vs 20/20

The pre-registered causation tests from §13.46 and §13.48 have both run on
Ubuntu.  Method each time: **one** `-O3` `.so` test binary (`tmin_ab`, built
once), two allocator arms compiled with identical flags
(`-O3 -g -DNDEBUG -fPIC -fvisibility-inlines-hidden -fno-semantic-interposition`)
and swapped via a symlink **interleaved run-by-run** in a single job, 20 pairs
per experiment.  Arms confirmed to differ (`cmp` + differing `.so` sizes)
before each run.

**Test 1 — §13.46 `m_owner_id` release/acquire (falsifier #5).**

| arm | failures |
|---|---|
| BASE (`25d732690^`) | **18/20** (90%) |
| FIXED (`25d732690`) | **13/20** (65%) |

Fisher exact two-tailed **p = 0.127** — not significant.  By §13.46's own
criterion ("still-failing keeps the fix and returns to the enumeration"),
**refuted as the cause**.

The fix nonetheless *worked*, which is what makes this a clean refutation
rather than a null result: a TSan build from the fixed commit shows **Pair B
gone** (1 report, 1 pair, at `8 8 400`).  The race was real, was eliminated,
and the fault continued.

The 90% → 65% gap is a **load artifact, not an effect** — confirmed by Test 2,
where the *same* fixed code (as the PREV arm) failed **20/20** once the
concurrent TSan batch stopped competing for CPU.  Absolute rates here track
machine load; only the interleaved within-job comparison is meaningful.

**Test 2 — §13.48(A) claim-side acquire + §13.48(B) metadata-after-CAS.**

| arm | failures |
|---|---|
| PREV (`e23e5b2c4`, §13.46 fix only) | **20/20** |
| HEAD (`d3c9a2176`, +§13.48 A+B) | **20/20** |

**No effect at all.**  §13.48(B) was the strongest candidate yet — a pure
logic bug needing no UB, with a direct path to the observed signature (loser's
pre-CAS header store re-types the winner's live slot → wrong size on free →
mis-route → freelist/bitmap corruption → released-after-zero).  The reasoning
is sound and the fix is worth keeping; it is simply **not this fault**.

**Where that leaves the method.**  §13.44 made TSan-over-the-pool work, and it
delivered a genuine enumeration — 5 pairs, 0 impossible, stable across five
runs (104/127/75/83/85 reports).  Both dominant pairs were then real races,
both were fixed, and **neither was the fault**.  So the fault is not a
TSan-visible data race on allocator metadata.  That is a substantive negative
result: it eliminates the entire class the last several sections have been
searching, and it is consistent with §13.30's pattern — the codegen knobs and
now the race fixes tune symptom rates without touching the failure rate.

Refuted mechanisms now total **nine**: `CASInfo::old_wrapper`, wrongly-mine,
split-read, falsifier #2, speculative devirtualization, `orphan_chain_pop`,
`m_owner_id` handoff, claim-side ordering, pre-CAS metadata.

**Recommendation.**  Stop generating allocator-race candidates; that well is
now measured dry.  The evidence that has never moved is differential and
points at codegen: total `-O3` vs `-O2`/clang separation, the single-pass
`-fipa-cp-clone` flip (0/167), arm64 silence (0/2400). The unexamined
habitat is §13.28's habitat 3 — `fast_vector<lsp<Packet>>` / `PacketList_`
lifecycle **as STM logic**, not as an allocator race — which is exactly what
Pair A's stacks kept pointing at on *both* sides even though the allocator-side
fix did nothing.  A post-§13.48 TSan batch at matched workload is running to
confirm Pair A is actually gone (the Pair B check pattern); that result is
pending and is not assumed here.

### 13.50 The instrument for what TSan cannot see: ASan use-after-poison
### over the live pool (same-thread UAF finally observable)

§13.49's negative is accepted, and its recommendation sharpened by one
structural fact: **a same-thread use-after-free has no happens-before
violation, so TSan can never report it** — while §11.3 explicitly admits
same-thread cases and Pair A's stacks point at habitat-3 STM lifecycle
on both sides.  The class that remains is precisely the class the
enumeration instrument was blind to.  The instrument for it is ASan's
manual poisoning, and the pool now cooperates (KAME_ASAN_ENABLED,
`__SANITIZE_ADDRESS__` / `__has_feature(address_sanitizer)` gate,
production builds byte-identical):

- **free** (`deallocate` entry, after the forensic fill): the slot's
  bytes are poisoned (same 16..4096 window).  The freelist-link word
  stays unpoisoned — the pool owns it while the slot is free; run with
  `-DKAME_POISON_FORENSIC` so the link is word 1 and **word 0
  (refcnt / vptr) stays covered**.
- **hand-out** (the `new_redirected`/`_aligned` wrappers, which now
  exist under either sanitizer): exactly the requested bytes are
  unpoisoned.
- **re-carve** (`construct_chunk_at`): the whole region is unpoisoned
  (old slot poison from a previous ALIGN class would otherwise trip the
  pool's own header writes).
- Auto-activation (§13.44) now covers ASan builds too.

Effect: **the FIRST touch of freed pool memory — any thread, any byte,
same-thread included — aborts with the accessing stack.**  That is a
strictly stronger detector than the refcnt tripwires (all bytes, not
one word) and complementary to them (the link-word hole is covered by
the tripwires).

**Mac cannot runtime-validate this either**: on macOS 26.4, Apple
clang 17's AND Homebrew LLVM 20's ASan both hang inside
`InitializeShadowMemory → get_dyld_hdr → dyld_shared_cache_iterate` —
the same OS-version breakage family as TSan (§13.10).  Verified here:
the four-way compile matrix (plain / forensic / ASan+forensic / TSan)
is clean and default Release ctest is 18/18.  Runtime validation and
the run itself belong to Ubuntu:

```
g++ -O3 -g -DNDEBUG -fPIC -fvisibility-inlines-hidden -fno-semantic-interposition \
    -fsanitize=address -DKAME_POISON_FORENSIC ... allocator.cpp  (the .so)
g++ ... -fsanitize=address ... tmin_dynnode.cpp ...              (the binary)
ASAN_OPTIONS=halt_on_error=0 ./tmin 20 40 700
```

(gcc -O3 arm — the failing codegen — WITH ASan; if ASan's
instrumentation suppresses the fault the way heavy tracing once did,
fall back to `-O2 -fipa-cp-clone`, the 19/35 proxy.)  Expected outcomes:
a use-after-poison report whose access stack lands in habitat-3 STM
code = the fault's first direct observation; silence while runs still
fail = the corruption is NOT written through freed-memory bytes at all,
which would leave wild writes through live-but-wrong pointers (Pair-A's
(B)-style retyping survivors) and genuine miscompile as the last two
readings.

### 13.51 ASan suppresses the fault (0/17), post-§13.48 TSan is near-silent, and this box has been running on half its cores

**§13.50's ASan mode builds and its hooks are live** — 10 `__asan_poison_memory_region`
and 48 `__asan_unpoison_memory_region` call sites in the `-fsanitize=address`
`.so`.  It never fired.

| build (same 4 cores, same session) | failures | ASan errors |
|---|---|---|
| non-ASan `-O3` `.so` | **7/9** (3 SIGSEGV, 4 abort) | — |
| ASan `-O3` `.so` | **0/8** | **0** |

Fisher exact two-tailed **p = 0.0023**.  Adding the 2-core runs, ASan is
**0/17 with zero use-after-poison reports** while the same code without ASan
fails routinely.  **ASan suppresses the fault**, joining `-O2` and clang in
the "cannot observe it" set.  §13.50's instrument is sound — it simply has no
window into this fault, and the same-thread-UAF hypothesis is untested rather
than refuted.

**Post-§13.48 TSan, matched `20 40 700` workload** (the §13.49 pending item):
reports fall **104–127 → 2–3**, and **Pair A and Pair B are both gone**,
confirming both fixes did what they claimed.  One residual pair remains,
3× and 2× in two runs: `orphan_chain_scrub()` `allocator.cpp:8364` reading
`cur->m_flags_packed & MASK_CNT` as a **plain load** against the `__sync_*`
RMWs (`allocator_prv.h:240`) that maintain it — same class as Pair B, worth
fixing on correctness grounds, but §13.49's "well is dry" verdict stands and
it should not be chased as the cause.

**Environment finding that affects every rate in this document.**  This box is
an i5-7500, **4 cores online** — but processes in this session inherit an
affinity mask of **CPUs 0,1 only** (`nproc` = 2; the other two cores idle at
800 MHz).  Every run before this section used **half the machine**.
`taskset -c 0-3` lifts it, and the failure *character* changes with it: on
2 cores failures were assertion aborts; on 4 cores **3 of 7 were SIGSEGV**.
Governor is `powersave` with scaling at 59%, so clock varies with load too.

**Consequence — treat single-arm rates as uninformative.**  With `.text`
**byte-identical** (verified: `md5 803ecb9e…` for both `c25fa8244` and
`d3c9a2176` compiled ASan-off — so §13.50's "production byte-identical" claim
holds), the same binary measured **20/20** in one session and **1/3** in
another.  I started to report that swing as a §13.50 regression and stopped
to check the `.text` bytes first; it was environment, not code.  Only
**interleaved within-job** comparisons are admissible here, which is what
§13.49's two differentials were — those results stand.  Any absolute rate
quoted anywhere in §13 should be read as conditional on the core count,
affinity mask and load of that session.

**Recommendation.**  Re-run the key differentials with `taskset -c 0-3`, where
the fault reproduces harder and produces SIGSEGV rather than a caught
assertion — a segfault carries a faulting address and a stack, which is
strictly more evidence than the value-check gives.  That is the cheapest
available upgrade to the signal, and it needs no new instrumentation.

### 13.52 The residual pair fixed — and a unification hypothesis for the
### suppressor set, testable with one census

**Code**: §13.51's residual TSan pair (`orphan_chain_scrub`'s plain
`m_flags_packed & MASK_CNT` read racing the `__sync` RMWs) is fixed with
an acquire load, plus its two same-class siblings found by census (the
DLL-neighbour emptiness test and `try_adopt_orphan`'s precondition).
Correctness fixes per §13.49's verdict — not chased as the cause.

**The suppressor pattern, unified into one testable hypothesis.**  Sort
everything by whether it can observe the fault:

| suppresses (0 failures) | does not suppress |
|---|---|
| gcc `-O2` | KAME_RC_TRACE tracer |
| clang `-O3` | forensic poison + free records |
| `-fno-ipa-cp-clone` | pool-event ring |
| arm64 | §13.46/§13.48 race fixes |
| **ASan** (§13.51, p=0.0023) | 2-vs-4-core affinity (rate only) |
| TSan (runs complete) | |

Every suppressor CHANGES THE CLONE SET of the allocator TU (different
compiler, different pass set, different arch — and sanitizer
instrumentation bloats function bodies, which flips inlining/cloning
profitability heuristics).  Every non-suppressor leaves the clone set
alone (side-table writes, extra data, scheduling).  So the cheap
hypothesis is that **ASan/TSan suppression is the `-fipa-cp-clone` flag
effect in disguise** — not "no window into the fault" but "the fault's
codegen precondition evaporated".  One command decides it, no runs:

```
ARM_A_FLAGS="-O3" ARM_B_FLAGS="-O3 -fsanitize=address" \
  kamepoolalloc/tests/asm_diff_ipa_clone.sh /tmp/asan_census g++
grep -c constprop /tmp/asan_census/{A,B}/sym/ -r   # A ≈ 585, B ≈ ?
```

If the ASan arm's `.constprop` count collapses toward the
`-fno-ipa-cp-clone` value (~12), the suppressor set has ONE explanation
and the fault's necessary condition sharpens to "the -O3 clone set of
allocator.cpp, on x86-64 gcc" — with §13.49 having eliminated its
TSan-visible races, the remaining readings are (a) a genuine
gcc wrong-code in one of those clones (the §13.22-style census, now
run at `-O3` vs `-O3 -fno-ipa-cp-clone` with the §13.26 caller-unit
correction, is the exhibit hunt), or (b) a latent app/pool bug whose
window only those clones' instruction scheduling opens (the taskset
SIGSEGVs are the cheapest new evidence for this — a faulting address
plus stack, strictly more than the assertion gives).

**Endorsed**: §13.51's taskset recommendation.  With 4 cores the fault
speaks in SIGSEGV — take the faulting-address/stack pairs (forensic
build, so RC-FREEREC and RC-POOLEV annotate each one) before any more
instrument-building.

### 13.53 The clone census: §13.52's hypothesis refuted, a perfect 6-arm correlation found, and that correlation refuted too

**The census (no runs).**  `objdump` clone-body and `constprop`-reference
counts for `allocator.cpp`, all arms compiled identically otherwise.  The
reference column reproduces the known baselines (`-O2 -fipa-cp-clone` → 558 ≈
"~585"; `-fno-ipa-cp-clone` → 14 ≈ "~12"), confirming the metric matches.

| arm | verdict | clone bodies | constprop refs |
|---|---|---|---|
| `-O3` | **FAULT** | 8 | 121 |
| `-O2 -fipa-cp-clone` | **FAULT** | 24 | 558 |
| `-O3 -fno-ipa-cp-clone` | suppress | **2** | **14** |
| `-O2` | suppress | **3** | **25** |
| `-O3 -fsanitize=address` | suppress | 7 | 183 |
| `-O3 -fsanitize=thread` | suppress | **8** | **123** |

**§13.52's hypothesis is refuted.**  TSan does **not** collapse the clone set —
it carries `-O3`'s count almost exactly (8 bodies / 123 refs vs 8 / 121), and
ASan's reference count is *higher* than `-O3`'s.  Only the genuine
clone-suppressors collapse (2/14, 3/25).  So sanitizer suppression is **not**
the `-fipa-cp-clone` effect in disguise; the two act by different mechanisms,
and §13.51's reading (sanitizers perturb timing) stands.

**But the census found something sharper — clone *membership*.**  Exactly one
clone family separates the arms perfectly:

| | `-O3` | `-O2+clone` | ASan | TSan | `-O2` | `-fno-ipa-cp-clone` |
|---|---|---|---|---|---|---|
| verdict | FAULT | FAULT | suppr | suppr | suppr | suppr |
| **`global_pop_fit`** | **1** | **1** | **0** | **0** | **0** | **0** |
| `l1_pop_fit` | 1 | 0 | 1 | 0 | 0 | 0 |
| `recycle_pop_fit` | 1 | 1 | 1 | 1 | 0 | 0 |

`l1_pop_fit` fails (absent from a FAULT arm), `recycle_pop_fit` fails (present
in suppressors).  **`global_pop_fit`'s clone is present in both fault arms and
absent from all four suppressors — 2/2 vs 0/4, perfect across every arm.**

**And it is not causal.**  `__attribute__((noclone,noinline))` on
`global_pop_fit`, verified surgical (its clone 1 → 0, the other 7 clone bodies
retained, refs 121 → 85), interleaved 16 pairs under `taskset -c 0-3`:

| arm | failures |
|---|---|
| BASE (clone present) | 11/16 |
| NOCLONE `global_pop_fit` | 10/16 |

Fisher **p = 1.000**.  **Refuted** — the tenth mechanism to die on a
differential test.

**Positive control, same session, same 4 cores** — because a null result is
only meaningful if the framing still reproduces:

| arm | failures |
|---|---|
| BASE | 6/14 |
| `-O3 -fno-ipa-cp-clone` | **0/14** |

Fisher **p = 0.016**.  The flag effect is **real and still reproduces here**.

**What this establishes.**  `-fipa-cp-clone` genuinely gates the fault, but its
effect is **not reducible to the presence of any single clone** — removing the
one clone that correlates perfectly across six arms changes nothing, while
removing the pass entirely suppresses completely.  The effect is distributed
across the clone set, or is not about clone *presence* at all but about the
whole-TU codegen the pass induces.  Either way, **the "find the one bad clone"
shortcut is closed**, and a census — however clean its correlation — cannot
substitute for the differential.  That is the second time in this section a
perfect structural correlation (§13.52's suppressor/clone-set alignment, then
`global_pop_fit`) has failed its causal test.

### 13.54 After the tenth refutation: stop asking "which mechanism" and
### ask the execution itself — rr, plus a dose-response probe

§13.53 accepted in full, including the demolition of §13.52 — that is
the second perfect correlation to fail its causal test here, and the
lesson is now structural: with the effect distributed across the clone
set (or not about clone presence at all), *hypothesis-first* testing has
hit its floor.  Ten mechanisms died on differentials; the flag effect
alone survives every control.  Two moves remain that do not require
guessing a mechanism first:

**1. rr — the tool this situation is the textbook case for.**  A
`SIGSEGV` with an unknown writer, µs windows, 40 threads, and a fault
that survives light instrumentation: record once, then interrogate the
execution *backwards*.

```
sudo sysctl kernel.perf_event_paranoid=1
rr record -h ./tmin 20 40 700          # -h = chaos mode; repeat until a crash lands
rr replay                              # at the SIGSEGV:
  (rr) watch -l *(uintptr_t*)<corrupted address>   # hardware watchpoint
  (rr) reverse-continue                # lands ON the writing instruction
```

That names the writer of the corrupted word — refcnt, header, pointer,
whatever the crash exposes — with zero hypotheses, and the forensic
annotations date-stamp the block's free/pool history around it.  Two
caveats to pre-register: rr serializes onto one core (chaos mode
compensates by adversarial scheduling; if the fault needs true
parallelism it may suppress — record the attempt count either way), and
rr's syscall interposition may perturb like ASan did (same fallback:
`-O2 -fipa-cp-clone` proxy, which fails at the same rates).

**2. The dose-response probe (cheap, structural).**  The distributed
reading predicts a *graded* rate: `noclone` on half the clone parents ≈
half the failure rate; the whole-TU-phase-change reading predicts a
*step* (full rate until the pass is off).  Three arms, interleaved as
usual: BASE / NC-half (§5's seven-function list, take 3–4) / flag-off.
Graded ⇒ many small windows, each clone contributing — consistent with
a latent timing bug the pass's codegen widens everywhere at once.
Step ⇒ a whole-TU property (register allocation, section layout,
alignment), which is also what a *distributed* miscompile would look
like.  Either shape kills half the remaining hypothesis space.

**3. Standing offer**: paste any 4-core SIGSEGV capture (fault address,
backtrace, the RC-FREEREC / RC-POOLEV block) — the Mac side reads them
against the source, §13.24-style, as they arrive.

### 13.55 The dose-response is GRADED — and rr now records this program (glibc 2.43 workaround)

**Result 1 — the probe §13.54 asked for, three arms interleaved, `taskset -c 0-3`, 14 rounds each:**

| arm | clone bodies | constprop refs | failures |
|---|---|---|---|
| BASE | 8 | 121 | **10/14 (71%)** |
| NC-HALF (`noclone` on `l1_`/`global_`/`recycle_pop_fit`) | 5 | 46 | **5/14 (36%)** |
| FLAG-OFF (`-fno-ipa-cp-clone`) | 2 | 14 | **0/14 (0%)** |

Monotone in clone count, **r = 1.000**.  BASE vs OFF **p = 0.00015**;
NC-HALF vs OFF **p = 0.041**; BASE vs NC-HALF p = 0.128 (the adjacent step is
underpowered at n=14, the endpoints are not).

**This is the graded shape, not the step.**  By §13.54's pre-registration that
means **many small windows, each clone contributing** — a latent timing bug
whose windows the pass's codegen widens everywhere at once — and it argues
against a whole-TU phase change (register allocation, section layout,
alignment) and against a single miscompiled clone.

**It also re-reads §13.53 rather than contradicting it.**  Removing *one*
clone (8 → 7) moved 11/16 → 10/16, p = 1.000; removing *three* (8 → 5) moved
71% → 36%.  Those are consistent with a roughly additive per-clone
contribution: §13.53's null was **underpowered for one clone's share**, not
evidence that clones are irrelevant.  `global_pop_fit` is not special — no
single clone is.  §13.53's stated conclusion ("the effect is distributed
across the clone set") is exactly what the dose-response now confirms
independently, and the "find the one bad clone" shortcut stays closed.

**Result 2 — rr works here now, with one caveat worth recording.**
`rr` 5.9.0 (installed) **cannot record any threaded program** on this box:
glibc **2.43** installs pthread stack guards with
`madvise(…, MADV_GUARD_INSTALL)` (advice **102**, Linux 6.13+), which rr 5.9
does not know, and it aborts at the first `pthread_create`.  Confirmed on a
three-line `std::thread` test, so it is not the allocator's doing.  An
`LD_PRELOAD` shim does **not** help — glibc issues that `madvise` by inline
syscall, bypassing the interposable symbol.

The workaround is test-side and contained: create the threads with
`pthread_attr_setguardsize(&attr, 0)` (guard pages are what triggers the
syscall).  A scratchpad copy of `tmin_dynnode.cpp` with the `std::thread`
loop replaced by `pthread_create`/`pthread_join` at guardsize 0 — body and
join order unchanged — **records cleanly under `rr record -h`**, and still
reproduces natively: **3/4, all SIGSEGV**.  The repo copy is untouched.

`kernel.perf_event_paranoid` must be **≤ 3** (was 4; the user set it).

A chaos-mode capture loop is running against the reproducing workload.  The
pre-registered caveat applies — rr serializes threads, which may suppress the
fault; if 12 chaos runs come back clean, that is the answer to record, and the
`-O2 -fipa-cp-clone` proxy is the fallback arm.

### 13.56 Reading the graded result — one recorded tension that narrows
### the window's habitat, and what to ask rr

Accepted in full.  The many-small-windows model now carries the flag
effect, the knob insensitivity (§13.30: windows narrow, never close),
the sanitizer/timing suppression (§13.51/§13.53), arm64/clang silence,
and the single-clone nulls, all at once.  Three additions for the
record:

1. **A tension worth keeping visible.**  NC-HALF's three functions are
   the `*_pop_fit` family — the large-recycle cache's pop side.  But §5
   ablated the LRC **chunk** recycling entirely (13/30 vs control
   12/31, no effect).  Both results stand only if the pop_fit windows
   act through a **non-LRC_CHUNK path** — the large/dedicated BLOCK
   recycle tiers, which §5's ablation left running.  That narrows the
   habitat of at least those three windows to the large-tier block
   lifecycle, not chunk identity reuse.  (Consistent with §5, which
   refuted chunk-identity reuse specifically.)
2. **Please record the full 8-clone-parent list** (only
   `l1_/global_/recycle_pop_fit` are named).  Not to resume
   single-clone hunting — the dose-response closed that — but because
   the family membership itself says which machinery hosts windows,
   and the rr capture should be read against that list.
3. **What to ask rr when a capture lands.**  The model predicts the
   crash is the SECOND actor of an interleaving whose FIRST actor ran
   ~µs earlier inside one of the cloned hot paths.  So: (a) hardware
   watchpoint on the corrupted word, reverse-continue → the writing
   instruction; (b) then AGAIN reverse-continue on the same word →
   the previous legitimate writer; the pair of stacks IS the
   interleaving, and with forensic annotations the block's free/carve
   history dates it.  If chaos mode comes back 12/12 clean, record
   that as the pre-registered suppression answer and fall back to the
   `-O2 -fipa-cp-clone` proxy under rr before abandoning the line.

### 13.57 CAPTURED — the corrupted word is `PacketWrapper::m_bundledBy.m_ref == 1`, in `unbundle`'s CAS deleter

**rr recipe that works** (chaos alone does not).  `rr record -h` at
`20 40 700`: **17/17 clean** — the pre-registered suppression.  Adding forced
preemption **`-c 10000`** flips it: the fault reproduces on essentially every
run (`rc=139` on run 1 of two independent hunts, and 2/2 in a third).  That
knob, not chaos, is what defeats rr's serialization.

**Replay diverges — read the recording instead.**  `rr replay` aborts with
`PerfCounters.cc:1147 read_ticks(): Detected 31656 ticks, expected no more
than 734` on every crashing trace (and on autopilot).  A *short successful*
run of the same workload replays fine, so it is not the workload per se;
this box runs a **PREEMPT_RT kernel (7.0.0-29-realtime)** and PMU tick
accounting under RT + forced preemption is the obvious suspect.  No matter —
**the crash registers and the module map are in the trace already**:
`rr dump <trace>` gives the `SIGNAL: SIGSEGV(det)` frame with full registers,
and `rr dump -p` gives the mmap bases.  No replay required.

**The fault, fully resolved.**  `rip = 0x5fc583993bf2`, exe base
`0x5fc58397c000` → file offset **`0x17bf2`**, inside `Node<LongNode>::unbundle`:

```
17be0: lea    0x10(%rbp),%rdi
17be4: call   local_shared_ptr<Packet>::reset()   ; m_packet at +0x10, OK
17be9: mov    0x8(%rbp),%rax                      ; m_bundledBy.m_ref at +0x8
17bed: test   %rax,%rax                           ; null check
17bf0: je     17c10
17bf2: lock subq $0x1,0x8(%rax)                   ; <-- SIGSEGV
```

At the fault **`rax = 0x1`**, so the `lock subq` targets address `0x9`.

Inline chain (`addr2line -f -C -i`), innermost first:
`fetch_sub` → `local_weak_ptr<Linkage>::reset()` (`atomic_smart_ptr.h:1043`)
→ `~local_weak_ptr` (`:1012`) → **`~PacketWrapper`** (`transaction.h:915`)
→ `atomic_shared_ptr_base<PacketWrapper>::deleter` (`:756`)
→ `compareAndSet_impl_<scoped_atomic_view, …>` (`:2267`)
→ `compareAndSetWeak` (`:2363`)
→ `ScopedNegotiateLinkage<LongNode>::compareAndSet` (`transaction_negotiation.h:917`)
→ **`Node<LongNode>::unbundle`** (`transaction_impl.h:3393`).

**PacketWrapper layout**: `+0x0` `atomic_countable::refcnt`,
**`+0x8` `local_weak_ptr<Linkage> const m_bundledBy`**, `+0x10`
`local_shared_ptr<Packet> m_packet`.

**Why this is proof of premature reuse, not a logic bug in the dtor.**
`m_bundledBy` is declared **`const`** — no PacketWrapper method writes it
after construction, and `m_ref` is a plain `gref_weak_base_ *` with no tagging
(so `1` cannot be a legitimate tagged value).  A `const` member observed
holding a value it was never constructed with means **the storage was written
by something other than this object** — i.e. the slot was recycled and
re-occupied while this destructor was still running.  Note also that the
preceding `m_packet.reset()` at `+0x10` completed normally, so the object was
coherent moments earlier.

`1` is exactly `atomic_countable::refcnt`'s construction value, which is
consistent with a fresh object being constructed in overlapping storage — but
which occupant wrote it is **not** established, and I am not asserting it.

**What this does and does not settle.**  It identifies the *victim* (a
`PacketWrapper` freed through `unbundle`'s CAS), the *corrupted word*
(`m_bundledBy.m_ref` at `+0x8`), and the *manifestation* (a null check that
passes on a non-pointer value, then a `lock subq` into unmapped memory).  It
does **not** yet name the writer.  §13.56's protocol (watchpoint +
reverse-continue to the writer, then again to the previous legitimate writer)
is exactly the right next step and is **blocked only by the replay
divergence** — worth one attempt on a non-RT kernel, or with rr's PMU checks
relaxed, before anything else.

### 13.58 The capture read against the source — and a native SIGSEGV
### forensics handler, since replay is blocked

**§13.57's read is confirmed against the source, and it is stronger than
stated.**  `PacketWrapper::m_bundledBy` is `local_weak_ptr<Linkage> const`
— and the *only* member of the whole class that a `const_cast`-free
program cannot touch after construction.  Checked in the tree: no
`PacketWrapper` method writes it; `local_weak_ptr::m_ref` is a plain
`gref_weak_base_ *` with no tag bits (unlike `local_shared_ptr`'s tagged
`m_ref`), so `1` is not a legitimate encoding of anything.  Two further
details make the reuse reading tight:

- `1` is `atomic_countable::refcnt`'s **constructor** value, and the
  fault is at `+0x8`.  A `PacketWrapper` re-constructed at
  `victim + 0x8` would put its `refcnt` exactly there — i.e. a
  **misaligned-by-8 re-occupation**, not a clean same-base reuse.
  (`PacketWrapper` is 40 B; the 48 B class the smoke exercises has
  0x8-offset neighbours.)  Worth checking against the pool's slot stride
  for that size class in the capture.
- `m_packet.reset()` at `+0x10` completed **normally** immediately
  before, so the object was coherent µs-scale earlier — matching §13.7's
  0.1–1.8 µs free→stale distribution exactly.

**The blocker is replay, so instrument the NATIVE crash instead.**  This
commit adds a `SIGSEGV`/`SIGBUS` handler to the tracer (default ON in
tracer builds, `KAME_RC_TRACE_SEGV=0` off) that answers, at the crash,
what reverse-execution would have been asked first:

- fault address + all plausible object-base registers (x86-64:
  rbp/rdi/rbx/r12–r15/rax; arm64: fp/x0/x19–x24) — §13.57's frame kept
  `this` in `rbp`, but that is codegen-dependent, so it scans;
- for every register the O(1) caches RECOGNISE (cache-HIT filter, so a
  9-register dump does not bury the signal): `RC-PRIOR-RELEASE-FAST`
  (who released it last, with call chain) and `RC-RECENT` (its last 16
  events) — pure table lookups, no dereference of crash memory;
- the pool-event tail (`RC-POOLEV`, §13.13) — was a batch/chunk
  operation in flight on that address's unit?
- and LAST, with `SA_RESETHAND` already armed so a nested fault merely
  ends the process with everything above already on disk: word 0 of each
  aligned candidate, decoded for the forensic token → `RC-FREEREC`
  ("who freed this slot, when", §13.4).

That yields the victim's release history and the freeing stack on every
native crash, without rr.  It does not name the writer — only replay or
a watchpoint can — but combined with §13.57's disassembly it should show
whether the victim was freed legitimately just before (reuse race) or
never freed at all (wild write).

Mac verification: a synthetic crash in the §13.57 shape
(`fault_addr=0x9`, deref of a `1` read from a freed object) produces the
full `RC-SEGV` / `RC-SEGV-W0` sequence; the cache-HIT filter correctly
stays silent for registers holding no traced object; tmin 3×40t with the
handler armed is clean and silent.  Ubuntu: it needs no new flags — the
existing forensic tracer build arms it, so the next `-c 10000` capture
carries the block's history automatically.

### 13.59 The writer is named: `bundle` Phase 1's PacketList copy resurrects a dead `Packet` (`transaction_impl.h:2870`)

§13.58's handler works.  **Operational note:** its output goes to the raw sink
(`rc_trace.<pid>.log` in CWD, or `KAME_RC_TRACE_FILE`), **not** stderr — I
first grepped stderr and saw nothing.  Run with `setarch -R` so every `site=`
resolves against the PIE base `0x555555554000`.

**Captured natively** (`-O2 -fipa-cp-clone` `.so`, forensic poison + tracer,
`taskset -c 0-3`, `10 40 500`, crash on run 2): two `INC-FROM-ZERO` tripwires
on the same `Packet` `0x7ffff58d6580`, size 32, each with the full triple.

| record | content |
|---|---|
| `RC-PRIOR-RELEASE-FAST` | `DEAD(unique) rc_before=1` tid **788730** site `0xf856` |
| `RC-FREEREC` | `freed_ptr=(=obj) size=32` **`drift=+0`** frames `0xf139 …` |
| `RC-ANOMALY #1` | `INC-FROM-ZERO` rc_before=**poison** tid **788756** site `0x1cbbe` |
| `RC-ANOMALY #2` | same object, same site, tid **788731**, `drift=+1` |

**`drift=+0` at free time is the key qualifier**: the reference accounting was
*correct* when the block was released.  This is **not** a lost decrement.  The
object was legitimately released and freed, and *then* a stale reference
incremented it — resurrection, not miscounting.  The increments come from OS
threads (788756, 788731) distinct from the releasing thread (788730).

**The three sites, resolved (`addr2line -f -C -i`):**

* **Last release** `0xf856` — `local_weak_ptr<Linkage>::reset()` ←
  `~local_weak_ptr` ← **`~PacketWrapper`** (`transaction.h:915`) ←
  `atomic_shared_ptr_base<PacketWrapper>::deleter` ←
  `local_shared_ptr<PacketWrapper>::reset()`.  **The same shape as §13.57's
  crash frame.**
* **Free** `0xf139` — `local_shared_ptr<Packet>::reset()`
  (`atomic_smart_ptr.h:1891`).
* **The resurrecting increment** `0x1cbbe` —
  `local_shared_ptr<PacketList_>::reset<PacketList_>(…)`
  (`atomic_smart_ptr.h:895`) inlined into **`Node<LongNode>::bundle`,
  `transaction_impl.h:2870`**:

```cpp
//--- Phase 1: collect sub-packets from child nodes ---
newpacket->subpackets().reset(new PacketList( *newpacket->subpackets()));
```

That copy-constructs a `PacketList` from the live one, and the copy
**increments every element `local_shared_ptr<Packet>`**.  When one element
already refers to a released-and-freed `Packet`, the copy's increment lands on
poisoned storage — exactly the tripwire that fired, twice, from two threads.

**This is where every independent line has been pointing.**
§13.39/§13.40's earliest TSan pair named **`bundle:2870`** by line number.
§13.47's Pair A (61×) had `PacketList_`'s copy ctor on the allocating side and
`PacketList_`'s dtor on the freeing side.  §13.28's habitat 3 was
"`fast_vector<lsp<Packet>>` / `PacketList_` lifecycle, never audited beyond
single-thread semantics".  §13.57's crash was `~PacketWrapper` on reused
storage.  They are all the same site.

**What is established, and what is not.**  Established: the *writer* is
`bundle` Phase 1's PacketList copy; the victim is a `Packet` element of the
copied list; the accounting was correct at free time, so the defect is a
liveness/ownership assumption in the copy, not a refcount leak.  **Not**
established: *why* an element is already dead at that moment.  The obvious
suspect is documented in CLAUDE.md — the bundle Phase-3 rule (skip child
wrappers whose `local.subpackets[c] == nullptr`) needed "when hard-link
references coexist with `is_bundle_root` bundles" — and **this reproducer
carries a hard link by construction** (`p2` under `gn2`).  That is the next
thing to test, and it is an STM-side question, not an allocator one.

### 13.60 Reading `bundle` Phase 1 around the named writer — the reference
### `newpacket` is, and a deterministic detector for it

§13.59 named the writer.  The remaining question ("why is an element
already dead?") is STM-side, so here is the source read of the four
lines above `:2870`:

```cpp
local_shared_ptr<PacketWrapper> superwrapper(              // :2861 pins a COPY
    make_local_shared<PacketWrapper>( *supscope, bundle_serial));
local_shared_ptr<Packet> &newpacket(                       // :2863 a REFERENCE,
    reverseLookup(superwrapper->packet(), true, SerialGenerator::gen()));
...
newpacket->subpackets().reset(new PacketList( *newpacket->subpackets()));  // :2870
```

**`newpacket` is a reference to a slot inside whatever list the lookup
navigated to — the frame owns no counted reference to that list.**  What
it pins is `superwrapper` (hence the root packet).  That is sound only
while the returned slot is inside `superwrapper->packet()`'s own tree.

`reverseLookupWithHint` navigates by the child wrapper's `bundledBy`
back-reference chain, and it validates NODE identity at the end
(`p->node().m_link != linkage → nullptr`) — but **not list ownership**.
With a hard link (a child with two parents — which this reproducer builds
by construction, `p2` under `gn2`) the hint chain can resolve through the
SIBLING parent, returning a slot in a list that only the sibling's
wrapper keeps alive.  When that wrapper dies (the §13.57/§13.59 last
release is exactly `~PacketWrapper`), the list dies with it, its dtor
releases every element (`drift=+0`: accounting correct at that moment),
and Phase 1's copy then increments elements from zero — the
INC-FROM-ZERO that fired, from two threads, on one `Packet`.

That is a complete, single mechanism consistent with **every** surviving
observation: the site (§13.59), the shape (§13.47 Pair A: PacketList copy
ctor allocating / PacketList dtor freeing), habitat 3 (§13.28), the
victim identity (§13.57), `drift=+0` (§13.5), the two-holder result
(Q1 15/15), the hard-link caveat already in CLAUDE.md, and — because the
window is a µs-scale race between the sibling wrapper's death and Phase
1's copy — the timing distribution (§13.7) and the many-small-windows
dose-response (§13.55).

**Deterministic detector, this commit** (`KAME_RC_TRACE_ESCAPE_CHECK=1`,
tracer builds): after the `:2863` lookup, walk `superwrapper->packet()`'s
tree and check that `&newpacket` is a slot within it; if not, record an
`OP_LOOKUP_ESCAPE` anomaly with the site.  A LOGIC condition — no race
needed, so it fires wherever the precondition occurs, exactly like the
MINE-SHARED detector.

**Mac result: 0 escapes** (1×8t + 5×40t, all clean).  Note what that
does and does not mean: arm64 never reproduces the fault either
(§13.6/§13.53), so a null here is consistent with the hypothesis; the
check needs to run **on Ubuntu**, ideally on the `-O2 -fipa-cp-clone`
proxy that produced §13.59's capture, where it is decisive:
- escape fires → the mechanism is confirmed and the fix is scoped (pin
  the list: hold a `local_shared_ptr<PacketList>` for the returned
  slot's container across Phase 1, or reject a hint that leaves the
  pinned tree and fall back to `forwardLookup`);
- escape never fires while INC-FROM-ZERO still does → the dead element
  arrives some other way, and the next question is which list the victim
  belonged to (dump `&newpacket`'s container identity alongside the
  tripwire).

### 13.61 The escape detector is live and reports **zero** escapes across four reproductions — §13.60 refuted as stated

Built the §13.60 tracer on Ubuntu against the `-O2 -fipa-cp-clone` forensic
`.so` (the arm that produced §13.59's capture), `setarch -R`,
`taskset -c 0-3`, `KAME_RC_TRACE_ESCAPE_CHECK=1`.

**Liveness proved first, because `escapes=0` is ambiguous** between "ran and
found nothing" and "never ran".  Temporarily patching `rcSlotWithin_` to
`return false` (build, measure, revert — tree clean) makes the detector report
**3 `LOOKUP-ESCAPE`** on a small run.  The call site at
`transaction_impl.h:2902` executes, the env gate opens, and the anomaly
reporter fires.

**Result, 9 runs, 4 reproductions:**

| run | rc | escapes | anomalies |
|---|---|---|---|
| 4 | 134 | **0** | 2 (`INC-FROM-ZERO`) |
| 6 | 139 | **0** | 0 |
| 7 | 134 | **0** | 0 |
| 9 | 139 | **0** | 0 |

**Zero escapes in every run, including the one that reproduced §13.59's
`INC-FROM-ZERO` pair.**  At the moment of the check, `&newpacket` *was* a slot
within `superwrapper->packet()`'s tree — the lookup did **not** navigate into
a sibling parent's subtree.

**So §13.60 is refuted in the form it was stated** — "a logic condition, no
race needed".  The eleventh mechanism to die on a mechanical test.  The
hard-link hint path is not returning an unpinned slot.

**What is *not* refuted, and the distinction matters.**  The check is a
point-in-time containment test taken immediately after the lookup.  It
establishes that the slot was inside the pinned tree *then*; it cannot exclude
a **time-of-check/time-of-use** variant in which the slot is legitimately
inside the tree at lookup and the list holding it dies before Phase 1's copy
at `:2870` runs.  That variant keeps everything else §13.60 assembled — the
site, `drift=+0`, the two-holder Q1 result, the µs window — and would need a
check placed *at the copy* (or a pin taken across it) rather than after the
lookup.  Since §13.59 established the increment happens on already-freed
storage, something does die in that interval; the escape check just shows it
is not because the slot started outside the tree.

**Recommendation.**  Move the containment check (or better, an explicit pin of
the owning list) to immediately before the `:2870` copy and re-run this same
matrix.  If it then reports, the TOCTOU variant is confirmed and the fix is a
pin across Phase 1; if it still reports zero across several reproductions, the
list is dying by some route other than the copy's own navigation, and the next
question is who releases it.

### 13.62 Check the ELEMENTS, one instruction before the resurrection
### (the TOCTOU question answered at the scene, with full forensics)

§13.61's result taken as decisive for §13.60-as-stated, and its
liveness-first discipline adopted below.  Rather than move the
containment check to `:2870`, this commit checks the thing §13.59
actually indicted — **the elements** — at exactly that instant:

```cpp
rcPreCopyCheck(newpacket->subpackets(), …);                 // §13.62
newpacket->subpackets().reset(new PacketList( *newpacket->subpackets()));
```

`rcPreCopyCheck` walks the list and tests every element's refcount for
the same predicate the `local_shared_ptr` tripwires use (`0` or
`>= 2^48` = poisoned).  Why this beats a containment/pin check:

- **It answers TOCTOU directly.**  If an element is dead *here*, the
  copy on the next line is the resurrection — no inference from
  containment needed.  If every element is live *here* and
  `INC-FROM-ZERO` still fires from `:2870`, then the element dies
  **between this check and the copy**, i.e. inside the copy's own
  execution window — which narrows the race to a handful of
  instructions and rules out "it was already dead when we got here".
- **It arrives with the complete case file.**  The report keys on the
  ELEMENT, so the existing anomaly path prints, for that element:
  `RC-PRIOR-RELEASE-FAST` (who released it last, with call chain),
  `RC-RECENT` (its last 16 events), and `RC-FREEREC` (who freed the
  storage, when — §13.4), plus `slot=` = the list slot address, which
  says whether the list is a private clone or shared with the committed
  tree.  That is the whole §13.59 triple *and* the container identity,
  in one report.

**Liveness proved before trusting any zero** (§13.61's rule): patching
the predicate to `if(true)` yields **144 138** `DEAD-ELEMENT` reports on
a small run, with well-formed anomaly lines (`rc_before=1` for live
elements, `slot=`, `type=…::Packet`) — the call site executes on every
bundle, the gate opens, the reporter fires.  Reverted; tree carries only
the feature.

**Mac: 0 dead elements** (6 runs, 1×+5× at 40 threads) — as expected
where the fault never reproduces.

**Ubuntu, the decisive matrix** — same arm as §13.61
(`-O2 -fipa-cp-clone` forensic, `setarch -R`, `taskset -c 0-3`,
`KAME_RC_TRACE_PRECOPY_CHECK=1`), enough runs to include reproductions:
- **DEAD-ELEMENT fires** → the element was already dead on entry to
  Phase 1; read its `RC-PRIOR-RELEASE-FAST` + `RC-FREEREC` and the
  releaser is named — that is the last unknown in the chain.  Note the
  `slot=` value: inside a private clone means the dead pointer was
  *copied in* earlier (look one level up, at the lookup's own
  `PacketList` copy); shared with the committed tree means a reachable
  list holds a dead element, which is the STM-invariant break.
- **No DEAD-ELEMENT but INC-FROM-ZERO still fires at `:2870`** → the
  element dies *during* the copy: a concurrent releaser wins a race
  against the copy's per-element increment, which points at the copy
  needing the container pinned (or the elements' release ordered) rather
  than at any earlier navigation — and the fix scope is then a pin/CAS
  around this one copy.
Either branch ends with a named releaser or a named window; there is no
third outcome that leaves the chain open.

### 13.63 Elements are LIVE at the pre-copy check — and the first resurrection is not the Phase 1 copy

Ubuntu matrix, `-O2 -fipa-cp-clone` forensic `.so`, `setarch -R`,
`taskset -c 0-3`, both gates on.  Liveness proved locally first (§13.61's
rule): forcing the predicate to `if(true)` yields **3** well-formed
`DEAD-ELEMENT` reports (the reporter caps; §13.61's control capped the same
way), then reverted — tree clean.

| run | rc | DEAD-ELEMENT | LOOKUP-ESCAPE | INC-FROM-ZERO |
|---|---|---|---|---|
| 1 | 139 | 0 | 0 | 0 |
| 2 | 134 | 0 | 0 | 0 |
| 3 | 134 | **0** | 0 | **12** |
| 4 | 139 | 0 | 0 | 0 |

**Four reproductions, zero dead elements — including the run that fired 12
`INC-FROM-ZERO`.**  Every element of the list was live one instruction before
the copy.  By §13.62's own reading this is outcome 2: the element dies inside
a window of a few instructions.

**But the case file says something sharper.**  All 12 events are on **one**
`Packet` (`0x7fffd672bc20`), and the two anomaly records differ in kind:

* **`#1`, the first: `rc_before=0`** — a *clean zero*, not poison.  The object
  had reached refcount 0 but had **not yet been freed**.  Site resolves to
  `Node<LongNode>::bundle`, **`transaction_impl.h:2851`**:

  ```cpp
  local_shared_ptr<PacketWrapper> superwrapper(
      make_local_shared<PacketWrapper>(supscope->packet(), bundle_serial));
  ```

  The `PacketWrapper` constructor copies that `local_shared_ptr<Packet>`, so
  this increments **`supscope->packet()`'s Packet — which is already at zero.**

* **`#2`, later: `rc_before=` poison** — the same object after the free, from a
  different thread.  Its site attributes to `ScopedNegotiateLinkage::commit()`
  / `Node::snapshot` (`transaction_impl.h:2507`); with inlining this frame
  attribution is the least trustworthy datum here and I would not build on it.

**This revises §13.59's single-site reading.**  §13.59 caught the increment at
Phase 1's copy (`:2870`); this capture catches an *earlier* one, at the
super-wrapper construction, on a **not-yet-freed** object.  So the copy at
`:2870` is not uniquely guilty — it is one of several sites that increment a
`Packet` already at zero, and the earliest observed one is upstream of it.
That is exactly the shape §13.55's graded dose-response predicted: **many
small windows, not one**.

**What this establishes.**  The defect is not "the copy navigates to an
unpinned list" (§13.60/§13.61, refuted) and not "the list elements are already
dead when the copy starts" (§13.62's outcome 1, refuted here).  It is that
**a `Packet` reaches refcount zero while a scope still holds a path to it**,
and whichever site next copies that `local_shared_ptr` resurrects it —
`bundle:2851` in this capture, `bundle:2870` in §13.59's.  The question is now
squarely upstream: **why does `supscope->packet()` yield a zero-count
Packet** — i.e. what dropped the last reference while the scope was live.

**Suggested next probe**, in the same style that has been working: validate
`supscope->packet()`'s refcount at scope *entry* and at each use, keyed on the
Packet so the existing anomaly path prints its release history.  If it is
already zero at entry, the defect is in how the scope acquires its view; if it
goes zero during the scope, the release that does it is the culprit and its
`RC-PRIOR-RELEASE-FAST` record will name it.

### 13.64 TLA+ at the packet layer: the protocol does NOT permit §13.63's
### observation — so the C++ departs from it (and the fix has a constraint)

Answering "can't TLA+ prove this?": **now it can be asked**, and this
commit asks it.  Every earlier layer abstracts packets as VALUES; none
tracks a packet's refcount, so none could express §13.63's finding.  That
finding is also logically over-determined, which is what makes it
modelable: a scope pins its wrapper, a live wrapper's `m_packet` is a
counted reference, so `rc[wrapper.m_packet] >= 1` should be a theorem.
Observing 0 leaves exactly three possibilities —

1. **LIFETIME** — the scope does not keep its wrapper alive across a
   consuming CAS (view cleared, never restored, still dereferenced);
2. **DOUBLE** — the packet is released through a second, uncounted path;
3. **UNCOUNTED** — a published wrapper's `m_packet` was installed without
   taking the +1.

`kamestm/tests/tlaplus/PacketRefcount.tla` models the bundle protocol's
packet ownership (scope acquire → super-wrapper copy at `:2851` →
Phase 1 list copy at `:2870` → commit CAS with explicit ownership
TRANSFER → `set_view` restore → scope release, with `~PacketWrapper`
cascading into `m_packet`), on the reproducer's hard-link topology, and
offers each candidate as a bug knob.

| arm | result |
|---|---|
| faithful (`HardLink=TRUE`) | **PASS**, exhaustive: 3 455 353 states / 703 620 distinct, queue empty |
| faithful (`HardLink=FALSE`) | **PASS**, same |
| knob 1 LIFETIME | **NoResurrection violated** |
| knob 2 DOUBLE | **LiveWrapperPinsPacket violated** |
| knob 3 UNCOUNTED | **LiveWrapperPinsPacket violated** |

**So the protocol as specified cannot produce the observation, and all
three candidate departures are caught.**  Same verdict shape as §13's
scope-token model: the defect is an implementation departure, not a
design hole — and now with a three-way partition of *which* departure,
each mechanically falsifiable in the C++.

**Honest limits, recorded so the PASS is not over-read:**
- Two of the three knobs were **toothless in their first formulation**
  (UNCOUNTED starved the protocol of commits; LIFETIME had no action that
  dereferences a cleared view).  Both were repaired until they violated —
  §13.61's rule applied to my own instrument.
- `CountMatchesHolders` is a **model-internal** identity, not a protocol
  invariant, and is deliberately **not asserted**: ownership transfer
  reuses holder ids, so `holders` (a set) cannot track `rc` (a count)
  exactly.  It stays as a definition for a future tightening with
  unique-per-acquisition ids.
- **The `HardLink` switch currently expresses only "one extra initial
  reference", not the dual-PATH structure** that makes hard links
  special.  Proof that this matters: both settings explore an
  *identical* 703 620-state graph — isomorphic, i.e. the model cannot yet
  tell the two topologies apart.  Modelling two reverseLookup routes to
  one node is the next tightening, and it is a prerequisite for using
  this model on the question below.

**The user's constraint, and how the model serves it.**  A fix must not
regress the NON-hard-link path.  That is a structural claim, and this is
the right instrument for it once the dual-path tightening lands: check any
candidate fix under `HardLink=TRUE` (must repair) *and* `HardLink=FALSE`
(must not change the reachable behaviour).  Scoping guidance per
candidate, in the order that preserves the common path best:
- **UNCOUNTED** — if the missing +1 is at a specific install site, the fix
  is that one `+1`: cost is one atomic increment on a path that already
  performs several, and it is topology-independent (no regression risk,
  no hard-link special case).
- **DOUBLE** — a release that does not own its reference; the fix removes
  or conditions that release.  Also topology-independent.
- **LIFETIME** — the only candidate whose natural fix (pin/restore the
  view across the CAS, or re-read after it) touches the hot bundle path
  for every topology.  If this is the one, prefer the narrowest form:
  restore the view exactly where the CAS consumed it (the `set_view`
  pattern already in `bundle_subpacket`), rather than holding an extra
  reference across Phase 1 — the former is a no-op for anyone who does
  not take the consuming branch, the latter is a permanent extra
  increment per bundle.

Run: `cd kamestm/tests/tlaplus && java -cp tla2tools.jar tlc2.TLC
-workers 4 -config PacketRefcount_none.cfg PacketRefcount.tla`
(≈30 s), and the four other cfgs likewise.

### 13.65 "Why does discrimination need Ubuntu?  Run TLC here." — correct;
### the model now transcribes the real two-scope sequence

The objection is right and §13.64's closing sentence was wrong: choosing
between LIFETIME / DOUBLE / UNCOUNTED is a question about the C++ code's
STRUCTURE, not about the failing hardware, so it belongs here.  What was
missing was fidelity, not a machine.  §13.64's model idealised each
thread as holding ONE view; the real serial-tag block holds TWO views of
the SAME wrapper:

```cpp
local_shared_ptr<PacketWrapper> superwrapper(                       // :2850
    make_local_shared<PacketWrapper>(supscope->packet(), bundle_serial));
ScopedNegotiateLinkage<XN> scope(supernode.m_link, snap, -1, OnExit); // :2851
...
if(scope.operator->() != supscope.operator->()) return DISTURBED;    // :2875
if( !scope.compareAndSet(superwrapper))       return DISTURBED;      // consumes `scope`'s view
supscope.set_view(std::move(superwrapper));                          // releases supscope's OLD view
```

The `:2875` pointer check *establishes* that both scopes view one
wrapper, so that wrapper carries **three** references (linkage + two
views), and the CAS consumes one while `set_view` releases another.  The
model now encodes exactly that (`AcquireInner`, a CAS that drops the
linkage's ref AND the inner view, and `RestoreView` = `set_view` releasing
the outer view first).  Two source facts settled by reading while doing
it, both narrowing the partition:

- **All three `PacketWrapper` constructors take COUNTED copies**
  (`m_packet(x)`, `m_packet(x.m_packet)`) — so UNCOUNTED is refuted at
  the constructor level.  The §13.63 increment at `:2850` is the
  *victim's* resurrection, not the cause.
- Therefore the cause is upstream of `:2850`: the packet reached zero
  while `supscope`'s wrapper was still alive holding it — which is
  DOUBLE, or a concurrent overwrite of a live wrapper's non-`const`
  `m_packet`.

**A model bug caught in the act, recorded because it nearly became a
"finding".**  The first faithful two-scope run reported
`NoResurrection` violated.  Reading the 12-state trace showed the cause
was mine: `set_view` performs a **zero-atomic transfer** of the
`superwrapper` local's reference into the view, so the installed wrapper
carries TWO references (linkage + transferred local).  The model counted
only the linkage's, so the new wrapper died one release early and killed
its packet.  Fixed (`wrc[new] = 2` at the CAS, no increment at the
transfer).  This is the same trap as §13.61's and §13.30's: a violation
in an under-specified model is a model result, not a code result — and
the only defence is reading the counterexample against the source before
believing it.

**Effect on the user's constraint** (a fix must not regress the
non-hard-link path): the partition now favours it strongly.  With
UNCOUNTED refuted at the constructors and LIFETIME's natural fix being
the narrow `set_view`-at-the-consuming-CAS form already used by
`bundle_subpacket`, the two surviving candidates are both **one-site,
topology-independent** changes:
- DOUBLE — remove or condition a release that does not own its
  reference: no new work on any path, no hard-link special case;
- a concurrent write to a live wrapper's `m_packet` — the fix is to make
  that field `const` (it already is for `m_bundledBy`), which is a
  compile-time change with **zero** runtime cost anywhere.
Neither adds an atomic operation to the common path, which is exactly the
property asked for.  Only if both are refuted does the LIFETIME-style
pin come back into scope, and §13.64 already records how to bound that
one.

Run matrix (all five arms, this machine, no Ubuntu needed):
`cd kamestm/tests/tlaplus && for a in none none_nohardlink lifetime
double uncounted; do java -cp tla2tools.jar tlc2.TLC -workers 4 -config
PacketRefcount_$a.cfg PacketRefcount.tla; done`

**Results of the faithful two-scope model** (this machine, seconds per arm):

| arm | result |
|---|---|
| faithful, `HardLink=TRUE` | **PASS** exhaustive — 2 673 states / 1 297 distinct, queue empty |
| faithful, `HardLink=FALSE` | **PASS** exhaustive — identical counts |
| knob LIFETIME | **NoResurrection violated** |
| knob DOUBLE | **LiveWrapperPinsPacket violated** |
| knob UNCOUNTED | **LiveWrapperPinsPacket violated** |

Finiteness is by **precondition, not StateConstraint** (this project's
rule): each thread performs at most one bundle sequence (`done[t]`,
mirroring the hard-link models' `bundleDone`), and `ScopeNodes` restricts
which linkages threads bundle.  **That bound is the result's main
limitation**: with one bundle per thread the search never reaches a retry
loop or a second bundle on the same linkage, so "PASS" means "no
departure within that depth", not "the protocol is safe at any depth".
Raising it is the obvious next step and is a cfg-level change
(`done` → a small counter).

Two further model errors of mine were caught and fixed while getting
here — the `set_view` zero-atomic transfer (above) and unclamped
`wrc` arithmetic that made the LIFETIME knob trip `TypeOK` instead of the
invariant it was built to trip.  Both were found by reading TLC's output
rather than by assuming; the pattern is now three-for-three in this
section (§13.61, §13.62, here).

**A source finding that constrains the fix, discovered while modelling.**
`m_packet` **cannot simply be made `const`**: `reverseLookup` takes
`local_shared_ptr<Packet> &superpacket` and rewrites it in place
(copy-on-write along the path), and `bundle` passes
`superwrapper->packet()` to it directly (`:2912`, `:3120`).  So the
"harden the field" route needs `reverseLookup`'s contract changed first
(return the new root instead of mutating the caller's slot), which is a
larger change than the const keyword suggests — worth knowing before
anyone proposes it as the cheap fix.  The two writes §13.36 found
(`:1444`, `:1561`) are the easy half: both immediately follow
`make_local_shared<PacketWrapper>(m_link, idx, serial)` and would become
a 4-argument constructor, at zero runtime cost.

### 13.66 CORRECTION to §13.65 — its PASS was over a crippled model; the
### repaired one passes exhaustively to depth 3, and a checker now prevents
### the class of mistake

**§13.65's numbers were wrong and are withdrawn.**  `ScopeRelease` both
assigned `done'` and listed `done` in its `UNCHANGED` tuple — a
contradiction, so **that action was never enabled**: scopes were never
released, wrappers never died from scope exit, and the "exhaustive PASS
over 1 297 distinct states" was over a model that could not exercise the
mechanism it was built for.  Caught by asking why raising the per-thread
bound from 1 to 2 left the state count *identical*.

Repaired (`done` removed from that tuple).  What the corrected model says,
with `ScopeNodes = {Root, C}` and two threads:

| depth (`MaxBundles`) | faithful arm | states / distinct |
|---|---|---|
| 1 | **PASS** exhaustive | 8 821 / 3 623 |
| 2 | **PASS** exhaustive | 570 741 / **156 927** |
| 3 | **PASS** exhaustive | 5 383 721 / **1 333 183** |

and with the bug knobs at depth 2: LIFETIME violates `NoResurrection`,
DOUBLE and UNCOUNTED violate `LiveWrapperPinsPacket`.  So the faithful
transcription of the two-scope sequence — including scope release,
wrapper death, `~PacketWrapper` cascading into `m_packet`, and up to
three bundle sequences per thread — **admits no resurrection**, while all
three candidate departures are caught.  That is now a real result rather
than an artefact: 1.33 M distinct states with the mechanism live.

**Prevention, since this is the fourth model error in three sections.**  A
mechanical check now runs over the module: for every action, the set of
primed variables must be disjoint from every `UNCHANGED` tuple it lists.
It reported the contradiction immediately and reports NONE after the fix.
Anyone extending this model should run it (it is six lines of Python over
the `.tla`, split on action headers) before trusting a PASS — the
TLA+ analogue of §13.61's "prove the detector fires first".

**Where that leaves the partition.**  The protocol, faithfully
transcribed at the depth the reproducer needs, does not permit
§13.63's observation.  Combined with the constructor reading
(UNCOUNTED refuted at source) the surviving candidates are:
1. **DOUBLE** — a release of a `Packet` by code that does not own a
   reference to it;
2. a **concurrent overwrite** of a live wrapper's non-`const` `m_packet`
   (which the model does not represent — it has no action that writes a
   published wrapper's packet field, because §13.36's audit found both
   writes pre-publish).

(2) is now the more interesting gap precisely *because* the model omits
it on the strength of that audit — and the audit predates everything
learned since.  Re-auditing it is cheap and is the natural next Mac-side
step; adding the write as an action would then let TLC say whether it is
sufficient to produce the observation.

### 13.67 The §13.36 re-audit finds it: `unbundle` dereferences a list slot
### AFTER the CAS loop has let its container die

Re-auditing "all `m_packet` writes are pre-publish" as §13.66 asked.  The
original audit grepped for `packet() = ` assignments; the re-audit
classified **every** `packet()` use, including MUTABLE BORROWS (address
taken, or passed as a non-const reference), which the first pass missed.
Inventory:

| kind | sites | verdict |
|---|---|---|
| assignments | `:1444`, `:1561` | pre-publish (fresh wrappers) — audit stands |
| `reverseLookup(superwrapper->packet(), …)` | `:2808`, `:2994` | pre-publish (fresh `superwrapper`) |
| `const_cast` to a mutable slot pointer | `:3303`, `:3307` | read-only use; type-unification only |
| **address of a slot inside a scope-held packet** | `:1948`/`:1952` → `:1996` | **the finding, below** |

**The sequence.**  `snapshotForUnbundle` sets the caller's
`newsubpacket` to `&(*(*parent_packet)->subpackets())[i]` (`:1996`) —
a slot **inside an ancestor's packet**, where `parent_packet` points into
`(*r.parent_scope)->packet()` (`:1948`).  The source's own lifetime
argument is *"r.parent_scope … keeps it alive"*, and that view survives
the return by being **parked into `cas_infos`** (§12.4's park).  Then:

```cpp
for(auto it = cas_infos.begin(); it != cas_infos.end(); ++it) {
    ScopedNegotiateLinkage<XN> scope(it->linkage, snap, -1,
        std::move(it->old_wrapper), …);      // parked view MOVED OUT, :3333
    if( !scope.compareAndSet(it->new_wrapper)) return DISTURBED;
    …
}                                            // <-- scope DIES here, each iteration
…
newsubwrapper = make_local_shared<PacketWrapper>( *newsubpacket, …);  // :3379
```

The loop moves each parked view into a **loop-local** `ScopedNeg` and
lets it die at the end of its iteration.  The linkage's reference to that
old ancestor wrapper was just replaced by the CAS, so the loop-local view
was its **last** reference: the old wrapper dies, `~PacketWrapper`
releases its `m_packet`, the old ancestor packet dies, **its PacketList
dies, and the list's element references are released**.  `:3379` then
dereferences `newsubpacket` — a slot in that list — and copies the
element, incrementing an already-freed `Packet`.

**This matches every surviving observation**: the resurrection is at a
`make_local_shared<PacketWrapper>(…packet…)` site (§13.63); the victim
was released with CORRECT accounting (`drift=+0`, 14/14 — the ancestor's
packet genuinely reached zero when its wrapper died); the last releaser
is `~PacketWrapper` (§13.57/§13.59 and all five container-knowable
observations); two holders for one count (Q1 15/15 — the dead slot plus
the new copy); a µs window (the loop is short); and deeper hard-link
walk-up chains mean more `cas_infos` entries, i.e. more chances — which
is why the hard-linked reproducer hits it.  It also explains why
§13.60/§13.62's detectors stayed silent: they were placed in **`bundle`
Phase 1**, and this is in **`unbundle`**.

`oldsubpacket` is `nullptr` on the live path (`bundle_subpacket` passes
`nullptr`, `:2622`), so the dangerous branch is the one that runs.

**Detector, this commit** (`KAME_RC_TRACE_SLOT_CHECK=1`): validate the
slot's target immediately before the `:3379` copy, keyed on the ELEMENT so
the anomaly prints its release history and the freeing stack.  Liveness
proved first (§13.61's rule): forcing the predicate true yields **46 435**
reports; reverted.  Mac: 0 hits over 3×40-thread runs, as expected where
the fault does not reproduce.

**The fix is a REORDERING, which is what the no-regression constraint
wants.**  Nothing in the CAS loop changes the slot's value (the old
ancestor packet is immutable; the loop only republishes wrappers), so the
copy can simply be taken BEFORE the loop — same single increment, no new
atomic operation, no hard-link special case, no cost on any path that
does not unbundle.  That lands as the next commit, kept separate so the
causal batch can bisect detector-vs-fix.

### 13.68 The fix: take the value before the CAS loop (a pure reordering)

```cpp
    // BEFORE the cas_infos loop:
    const local_shared_ptr<Packet> newsubpacket_val(
        oldsubpacket ? local_shared_ptr<Packet>() : *newsubpacket);
    for(auto it = cas_infos.begin(); …)   // views move out, wrappers die
    …
    newsubwrapper = make_local_shared<PacketWrapper>(newsubpacket_val, …);
```

Why this is the right shape for the constraint the user set (no
regression on the non-hard-link path):

- **No new atomic operation, on any path.**  The copy at `:3379` already
  took one reference on that `Packet`; this takes the same one earlier.
  The old code's dereference-after-the-loop was not just unsafe, it was
  unnecessary — nothing in the loop can change the slot's value (the old
  ancestor packet is immutable; the loop only republishes wrappers).
- **No topology special case.**  It is the same three lines whether or
  not a hard link exists; hard links only made the window likelier by
  lengthening the walk-up chain.
- **Nothing outside `unbundle` changes**, so the common bundle-only path
  is untouched.
- The `oldsubpacket ? …` guard keeps the other branch allocation-free:
  when the caller supplied an expected value, `newsubwrapper` comes from
  `*newsubwrapper_returned` and no capture is needed.

The §13.67 detector is retained and re-pointed at the pre-loop capture:
after the fix, a report there would mean the slot was **already** dead
before the loop — a different and worse defect — so it keeps earning its
place.

Mac verification: `tmin` 5×40 threads clean with zero anomalies, and the
standalone `transaction_dynamic_node_test` (3×), `transaction_test`,
`transaction_negotiation_test` and `transaction_lookup_memo_test` all
pass.

**Ubuntu, the causal test** — the same interleaved discipline as §13.49:
one `-O3` `.so` test binary, two allocator/STM arms differing only by
this commit, alternated run-by-run under `taskset -c 0-3`.
- rate → 0: the mechanism is confirmed and the hunt is over; §13.18's
  `-fno-ipa-cp-clone` mitigation can then be re-examined (keep it until
  a soak confirms, then decide).
- rate unchanged: the fix stays (it removes a real dangling dereference
  that the source's own lifetime comment does not justify), and the
  detector output from the §13.67 build tells us whether the slot was
  dead before the loop instead.

### 13.69 Which TLA+ models the §13.68 rewrite touches — and the surprise:
### the specs already described the FIXED order

Asked directly.  Surveyed all thirteen `BundleUnbundle*` models plus
`atomic_shared_ptr.tla` and `PacketRefcount.tla`:

| model family | represents unbundle's CAS loop | tracks refcounts | affected? |
|---|---|---|---|
| `BundleUnbundle*` (13 files) | yes (`cas_infos` in 5 of them) | **no** | **not invalidated** |
| `atomic_shared_ptr.tla` | no | yes (wrapper layer) | no |
| `PacketRefcount.tla` | no (bundle only) | yes (packet layer) | extended here |

**No existing model's result is invalidated**, for a structural reason:
the `BundleUnbundle` family abstracts packets as VALUES.  It has no
references, no aliasing and no refcounts, so "when is the reference
taken" is not expressible there, and a reordering that changes only that
cannot change any of their conclusions.

**The surprise is stronger than "unaffected".**  `BundleUnbundle_3level_LLfree`
carries the sub-packet as `local[t].newpacket` — a VALUE captured into
thread-local state before `UnbundleCASLoop`, then used by
`UnbundleCASChild`.  TLA+ has no pointers, so the spec could only ever
describe *capture, then CAS the ancestors* — i.e. **the specification has
always described §13.68's fixed order.  It was the C++ that drifted**, by
holding a pointer whose container the loop kills.  Pre-fix, the code
diverged from every unbundle model at exactly this point, and nobody
noticed because the divergence is invisible in the models' vocabulary.
The fix restores correspondence rather than requiring a spec change.

Corroboration from the code itself, worth quoting because it states the
broken contract explicitly (`transaction_impl.h:2178-2180`):

> *"Extract scoped_atomic_view from parent_scope into CASInfo.
> parent_scope is not used after this point (parent_packet still points
> into the PacketWrapper kept alive by the CASInfo's view)."*

The lifetime argument is "the CASInfo's view keeps it alive" — and the
CAS loop is precisely what moves that view out and lets it die.

**What this commit adds: a model-level regression test for the fix.**
`PacketRefcount.tla` (the only model with the packet-refcount layer) now
models the handoff as `ParkAndKill` + `CaptureEarly` / `DerefLate`, with
`BUG_LATE_DEREF` selecting the order:

| arm | result |
|---|---|
| `BUG_LATE_DEREF = FALSE` (§13.68's order) | **PASS** exhaustive — 3 492 281 states / **1 045 892 distinct**, queue empty |
| `BUG_LATE_DEREF = TRUE` (pre-fix order) | **NoResurrection violated** in 163 states |

So the reordering is now verified in both directions at the layer where
it matters: the old order provably resurrects a released packet, the new
one provably does not, over a million distinct states.  (One model error
of mine on the way — `PacketOf[slotIn[t]][1]` for
`PacketOf[slotIn[t][1]]`, which TLC reported as a domain error rather
than a violation.  Caught before it became a "PASS".)

**Documentation follow-up (not done here, small):** the fidelity dossier
and `BundleUnbundle_3level_LLfree`'s `UnbundleCASChild` note should record
that `local[t].newpacket` corresponds to §13.68's pre-loop capture, and
that the pre-fix pointer-after-loop was an unmodelled divergence.  That is
the one place where the rewrite changes what the docs should say.

### 13.70 Full test sweep for the §13.68 rewrite (the earlier five were not enough)

Asked whether the other tests were run — they had not been: §13.68
reported only `tmin` plus four transaction tests, which is thin for a
change to a core STM header.  Complete sweep, all on this commit:

| suite | result |
|---|---|
| `cmake -S tests` tree, `ctest -j4` | **40/40 pass** (40.7 s) — includes the 10 `alloc_*`, `atomic_*`, `mutex`, `fast_vector`, `xwaitcell`, `xnode_ctorthrow`, `c_api`, `malloc_intercept`, `test_pmr_resource`, `test_ros2_allocator`, `bench_rt_wcet_smoke`, and 15 `transaction_*` |
| `tools/audit/run_audits.sh` | **exit 0** (node-name collisions, iterate_commit side effects, pybind GIL, UI listeners, non-const Payload pointees, conditional layout, no-DCAS) |
| `transaction_lookup_memo_test` | pass — **absent from the cmake tree**, built by hand |
| `transaction_payload_integrity_mixed_test` | pass, default **and** `3 8 3 30` |
| `transaction_payload_integrity_3level_mixed_test` | pass, default **and** `3 8 3 30` |
| `transaction_dynamic_node_test` ×20 | **20/20** |
| `transaction_payload_integrity_3level_test` ×5 | **5/5** |
| `PacketRefcount.tla`, both orders | fixed PASS 1 045 892 distinct / pre-fix violates (§13.69) |

Worth recording as a gap in its own right: **three tests in CLAUDE.md's
list are not in the `tests/` cmake tree** — `transaction_lookup_memo_test`
and the two `*_mixed_test` payload-integrity variants.  They build and
pass standalone (and the mixed ones need their arguments as four separate
tokens, per the bench-methodology note), but nothing runs them
automatically, so a regression there would be invisible to `ctest` and to
CI.  Adding them to `tests/CMakeLists.txt` is a small, separate change
worth doing.

Not covered on this machine, and stated so rather than implied: the Qt
app build (`kame.pro`) — the change is header-only inside `kamestm/`, and
`libkame`/module compilation of that header is exercised on Ubuntu; the
GenMC suites (unaffected: they model the allocator, not this path); and
the reproducer itself, which does not fire on arm64.

### 13.71 FIXED still crashes → the same class, audited again: one candidate
### refuted by reading, one narrower case closed by a free hardening

`FIXED` at 3/30 vs `BASE` 30% (p = 0.235) is exactly what §13.55's
graded dose-response predicts if §13.68 closed **one of several** windows
of the same class, so the audit continued on the generalisation:
*a reference or pointer into a packet / list held across a point where a
wrapper can die.*

**Verified first that §13.68 covers all of `unbundle`'s callers**: three
call sites, two pass `oldsubpacket = nullptr` (the branch that was fixed),
the third passes `&tr.m_oldpacket` — a `Transaction` member, alive by
construction, and it takes the other branch.  No further exposure there.

**The bundle-side candidate, and its honest resolution.**  Phase 1 binds
two REFERENCES into the packet it is building (`:2816-2817`):

```cpp
local_shared_ptr<PacketList> &subpackets(newpacket->subpackets());
shared_ptr<NodeList>        &subnodes(newpacket->subnodes());   // = subpackets()->m_subnodes
```

`subpackets` is used only through Phase 3 (`:2902`), before anything can
die — clean.  `subnodes` is used again at **`:3050`, after Phase 4 and
after `supscope.set_view(std::move(superwrapper))` drops the last holder
of the Phase-1 wrapper**.  Chasing whether that dangles:

- **Root path (is_bundle_root): SAFE, and the reason is subtle** —
  Phase 4's `reverseLookup` takes the self-alias branch, whose only
  clone is the *payload-level* `make_local_shared<Packet>(**foundpacket)`.
  That copy is SHALLOW: it shares `m_subpackets`, so the PacketList that
  actually owns `m_subnodes` survives the Phase-1 packet's death.  I had
  this written up as a second use-after-free before tracing
  `Packet::subnodes()` through `subpackets()` — it would have been the
  second false finding in this section, and the trace is what stopped it.
- **Non-root path: NOT excluded** — there `reverseLookupWithHint`'s
  copy_branch does the *list-level* clone
  (`subpackets().reset(new PacketList(...))`), which REPLACES the very
  list `subnodes` points into, and nothing else need hold the old one.

Rather than argue over which branch the reproducer reaches, `:3050` now
reads the children through the **live view** —
`supscope->packet()->subnodes()` — which is valid in every branch by
construction (the view was just installed), names the same children (the
clone shares the NodeList), and **costs nothing**: no new atomic
operation, no allocation, no topology special case.  Same shape of remedy
as §13.68: stop holding a reference across a lifetime edge you do not
own.

Verified: `transaction_dynamic_node_test` 10/10, full `tests/` tree
**40/40**.

**What this does and does not claim.**  It closes a reachable
dangling-reference path; it is NOT claimed to be the remaining 10 %.  The
discriminator is on the Ubuntu side and needs no new instrumentation: the
FIXED arm's surviving SIGSEGVs already carry §13.58's `RC-SEGV` block
(fault address, candidate registers, `RC-PRIOR-RELEASE-FAST`,
`RC-RECENT`, `RC-FREEREC`).  **Reading one of those tells us whether the
survivor is this class again (a member read out of freed storage, as in
§13.57) or something new** — please paste one when the higher-workload
batch lands, and the higher-power BASE/FIXED comparison at
`20 40 700` will also say whether §13.68 moved the rate at all.
### 13.72 The §13.68 causal test: rate does NOT go to zero — 40 interleaved pairs, two workloads, p = 0.82

§13.68's pre-registered Ubuntu test, run per §13.49's protocol.  The fix is in
`transaction_impl.h`, so the arms differ in the **test binary**, not the `.so`:
one allocator (`-O2 -fipa-cp-clone`, the arm that produced §13.59's and
§13.63's captures), two binaries built with identical flags, the BASE arm
compiled against `b1cb96327`'s header via an `-I` override.  Arms verified
distinct (`newsubpacket_val`: 3 occurrences in FIXED, 0 in BASE; binaries
differ, and FIXED is the larger — consistent with added code).  Interleaved
run-by-run, `taskset -c 0-3`.

| workload | BASE (pre-fix) | FIXED (§13.68) | Fisher p |
|---|---|---|---|
| `10 40 500` | 6/20 (30%) | 2/20 (10%) | 0.235 |
| `20 40 700` | 10/20 (50%) | **12/20 (60%)** | 0.751 |
| **pooled** | **16/40 (40%)** | **14/40 (35%)** | **0.818** |

**The two workloads point in opposite directions**, which is the signature of
noise, not of a partial effect — and neither is significant on its own.  A
separate 10-run batch of the FIXED arm produced another SIGSEGV (3/30 overall
at the smaller workload).  **The rate does not go to zero.**

By §13.68's own pre-registration this is the "unchanged" branch: **the fix
stands as a real dangling-dereference removal** — §13.67 identified a genuine
defect and the reordering is sound and free — **but it is not this fault**, and
the hunt does not end here.  Twelfth mechanism to survive its author's
reasoning and die on a differential.

**A methodological note worth keeping.**  The first workload alone would have
read as a 3× reduction, and had I stopped there I would have reported the fix
as confirmed on a p = 0.235 result. The second workload reversed the sign.
Single-workload, single-session rates in this document have repeatedly proven
unstable (§13.51: identical `.text` measured 20/20 and 1/3); the pooled,
two-workload, interleaved form is the weakest design I would now trust for a
"went to zero" claim, and even it only bounds the effect rather than
confirming absence.

**Where that leaves the chain.**  §13.63's finding is untouched and remains the
live thread: a `Packet` reaches refcount zero while a scope still holds a path
to it, and the *earliest observed* resurrection was `bundle:2851`'s
`make_local_shared<PacketWrapper>(supscope->packet(), …)` — upstream of both
the Phase 1 copy and the unbundle slot §13.68 fixed.  The open question is
unchanged and is the one §13.63 posed: **what drops the last reference to
`supscope->packet()` while the scope is live.**  The detector §13.68 retained
(now pointed at the pre-loop capture) has not reported, which is consistent:
that slot was not the one dying.

### 13.73 §13.71's discriminator, answered: the survivor is a DIFFERENT signature — a same-thread DEC-UNDERFLOW

Built HEAD (with §13.68 + §13.71; both verified present in the compiled
header: `newsubpacket_val` ×3, `supscope->packet()->subnodes()` ×1) as the
tracer/forensic binary and hunted survivors at `20 40 700`,
`setarch -R taskset -c 0-3`.

**Three surviving failures analysed: `rc=134` every time, and in each one
`RC-SEGV` count = 0 and `INC-FROM-ZERO` count = 0.**  The §13.57/§13.59/§13.63
signature is absent from the survivors.  What fires instead is
**`DEC-UNDERFLOW`** — and the two events are on **the same thread**:

```
tid=844320  seq …927802  DEAD(unique)               rc 1 -> 0
            (free at tsc …928276, free_tid=88, drift=+0)
tid=844320  seq …929578  DEC-UNDERFLOW **TRIPWIRE** rc <poison> -> poison-1
```

One thread releases the last reference, the block is freed with **`drift=+0`**
(accounting correct, as in every prior capture), and **the same thread
decrements it again** ~1776 seq units later.  That is a **same-thread
double-release** — precisely the class §13.50 built the ASan mode for and
which §13.50 correctly noted TSan structurally cannot see.

**The two sites.**

* **Last release** (`0x1249d`) — `local_weak_ptr<Linkage>::reset()` ←
  `~local_weak_ptr` ← `~PacketWrapper` ←
  `atomic_shared_ptr_base<PacketWrapper>::deleter` ←
  `compareAndSet_impl_`.  **The same shape as §13.57's crash frame and
  §13.59's legitimate release.**
* **The second decrement** (`0x605a`) —
  `local_shared_ptr<Packet>::operator=(const local_shared_ptr&)`
  (`atomic_smart_ptr.h:863`), i.e. an assignment releasing its *old* pointee.
  Its `slot=0x7ffff37f7e50` is a **stack address**: a stack-local
  `local_shared_ptr<Packet>` still held a countable reference it no longer
  owned, and assigning over it paid the reference a second time.

**Object life story, all on the one thread** (sites resolved):

| seq | op | resolved site |
|---|---|---|
| …910962 | `BORN` | `reset_unsafe` ← `local_shared_ptr(Packet*)` ctor |
| …913086 | `VADOPT` | `atomic_shared_ptr_base::get()` ← `lsp::get()` |
| …915796 | `INC` (1→2) | `lsp::swap` ← `lsp::operator=` |
| …916556 | `INC` (2→3) | `lsp::swap` ← `lsp::operator=` |
| …927472 | `VMOVE` | `~scoped_atomic_view` ← **`ScopedNegotiateLinkage` ctor** (`transaction_negotiation.h:555`) |
| …927802 | `DEAD(unique)` | `~PacketWrapper` ← `compareAndSet_impl_` |
| …929578 | `DEC-UNDERFLOW` | `lsp::operator=` (`:863`), stack slot |

The `VMOVE` — a view-custody move out of a `scoped_atomic_view` during
`ScopedNegotiateLinkage` construction — is the last event before the death,
and is the natural place to look for where a stack-local's ownership stopped
matching its count.

**Caveat on type labels:** the tracer keys on address, and this address
carries `type=PacketWrapper` on the `VADOPT`/`VMOVE` lines and
`type=Packet` on the `INC`/`DEAD` ones.  I have not resolved whether that is
custody markers being recorded against the view's type or genuine reuse across
the boundary, so **the type column should not be over-read**; the seq ordering
and the site resolutions are the solid parts.

**What this means for §13.71's question.**  The survivor is *not* the same
class.  §13.68 and §13.71 may well have closed the resurrection windows —
those signatures are gone from the survivors — while a **distinct
same-thread double-release** remains, sharing only the `~PacketWrapper`
last-release frame and `drift=+0`.  The next probe should target the
stack-local's ownership across the `VMOVE`, not the resurrection sites.

### 13.74 Two answers the instrument's author owes §13.73 — and the last
### decrement-coverage hole, closed

**1. The type-label caveat is resolved: it is the dual-keyed markers, not
reuse.**  §13.32 made `VADOPT` / `VMOVE` record TWICE — once keyed on the
view's target (a `PacketWrapper`) and once on `secondary_obj_`, the
target's `m_packet` (the `Packet`) — and **both records carry
`type_name_<T>()` = the VIEW's T**, i.e. `PacketWrapper`.  So in §13.73's
history the `type=PacketWrapper` lines on a `Packet` address are the
SECONDARY records: they mean *"a PacketWrapper whose `m_packet` is this
Packet adopted / moved a view"*.  The object is a `Packet` throughout; no
cross-boundary reuse is implied.  That makes those two lines **positively
useful** rather than suspect: they name the wrapper custody events
bracketing the death, which is why the `VMOVE` sits where it does.

**2. The arithmetic gap (BORN 1 → INC → INC → `DEAD(unique)` at 1) has
two possible causes, and one of them was my instrument.**

*Cause A — cache bucket theft (most likely for a `Packet`).*  `RC-RECENT`
is an O(1) **direct-mapped** cache: 4096 buckets keyed by an address
hash, 16 events each.  A second object hashing to the same bucket **takes
it over and discards the first object's events**; when the original
object is touched again it re-claims the bucket and starts from an empty
index.  So a `RC-RECENT` history can legitimately be missing intermediate
events.  **The authoritative source is `kame_rc_dump`'s ring scan** — its
`ledger:` line (`strong BORN n / DEAD n / INC n / DEC n`) is computed over
all 64 rings and is what settles whether decrements are missing or merely
unlisted.  Please read that line for the survivor's object; if
`INC + BORN − DEC − DEAD ≠ 0` the arithmetic really is open.

*Cause B — six decrement paths had NO event at all, now fixed here.*
Audited every `refcnt.fetch_sub` / `decAndTest` in `atomic_smart_ptr.h`:
`lsp::reset()`, `release_()`'s Owned branch and `release_tagheld_zeroreset_`'s
fallback were hooked (v8), but **six were silent** — the CAS success-path
release, the step-4 undo, `release_tag_ref_`'s excess-undo,
`new_refcnt_undo`, the Swap path's `decAndTest`, and
`try_release_single_attempt`'s helper.  A new `KAME_RC_DEC_N(obj, amount,
T)` macro now traces all six with the standard threshold tripwire and
records the pre-value, so multi-unit decrements are visible and any
history's arithmetic is closable.  Verified: a CAS smoke's ledger now
shows the previously invisible CAS-path `DEC`, no false anomalies, and the
full `tests/` tree is **40/40**.

**Scope note that matters for the survivor.**  All six are
`atomic_shared_ptr<T>` paths, and there is no `atomic_shared_ptr<Packet>`
(§13.6, grep-verified) — so for a **Packet** they cannot fire.  Meaning:
if the survivor's `Packet` history still fails to add up after this
commit, the missing decrements are **cache eviction (Cause A), not an
unhooked path** — every `Packet` decrement goes through `lsp`, and all of
those were already traced.  Where the new hooks DO change things is
**wrapper** histories, which §13.57's crash (`~PacketWrapper` on reused
storage) and §13.73's `VMOVE` make central: a wrapper's life can now be
reconstructed completely.

**Suggested next read, no new runs needed**: for the survivor's object,
the `ledger:` line plus the full `RC-EV` list (newest-40, ring-scanned).
If `DEAD(unique)` really fired at a true count of 1 while a stack-local
`lsp<Packet>` still held it, that is **two holders / one count on a
Packet** — the UNCOUNTED shape §13.64 partitioned out and refuted at the
`PacketWrapper` constructors, which would mean the missing `+1` is on a
different acquisition path (the stack-local's), and the `VMOVE` line
names the custody event to inspect.

### 13.75 With all six paths traced: the survivor is `PacketList_`'s destructor releasing an element that is already dead

Rebuilt HEAD (§13.74's `KAME_RC_DEC_N` present, 8 sites in
`atomic_smart_ptr.h`) and hunted survivors at `20 40 700`.  **Two captures,
both `rc=134`, both `DEC-UNDERFLOW` on a `Packet`, no `INC-FROM-ZERO` and no
`RC-SEGV` in either** — §13.73's survivor signature reproduces 2/2, and the
resurrection signature stays absent from the FIXED arm.

**Operational note:** the `ledger:` line is written by `kame_rc_dump` to
**stderr**, not to the `KAME_RC_TRACE_FILE` raw sink — it is in the run's
`.out`, not the `.log`.  (I looked in the log first and reported it missing.)

**The authoritative ledgers are balanced:**

| capture | ledger | tripwires |
|---|---|---|
| 1 | `BORN 31 / DEAD 31 / INC 72 / DEC 72` | 1 |
| 2 | `BORN 8 / DEAD 7 / INC 13 / DEC 13` | 1 |

Balanced strong counts, consistent with `drift=+0` in every capture so far:
the accounting is not leaking, one decrement simply arrives at an object that
has already reached zero.

**Capture 1 — `dtor_depth=2`, and the site is habitat 3 itself:**

```
fast_vector<lsp<Packet>,1>::clear_fixed()     fast_vector.h:236
fast_vector<lsp<Packet>,1>::clear()           fast_vector.h:180
PacketList_::~PacketList_()                   transaction.h:105
atomic_shared_ptr_base<PacketList_>::deleter  atomic_smart_ptr.h:795
```

A `PacketList_` is being destroyed; its `fast_vector<local_shared_ptr<Packet>>`
clears, releasing each element — **and one element's `Packet` is already at
zero and freed.**  `dtor_depth=2` matches exactly (list dtor → vector clear →
element release).

**This is the mirror image of §13.59.**  There, `bundle` Phase 1's PacketList
*copy* **incremented** an element whose `Packet` was already dead.  Here the
PacketList *destructor* **decrements** one.  Both say the same thing about the
same structure: **a `PacketList_`'s element array holds a
`local_shared_ptr<Packet>` whose ownership does not match its count** — copy it
and you resurrect; destroy it and you double-release.  §13.28 flagged this
container as habitat 3 and it has now been indicted from both directions.

**Capture 2** attributes to `Transaction::finalizeCommitment`
(`transaction.h:2814`), but its leaf frames are `is_fixed`/`size`/`empty` on an
unrelated `fast_vector<shared_ptr<Message_>>` — inlining has smeared this one,
so I would treat only the `finalizeCommitment` frame as meaningful and would
not build on the leaf.

**Suggested next probe.**  The two indictments meet at the element array, so
instrument *there* rather than at either caller: on `PacketList_` construction
record each element's `Packet` and its refcount, and re-check them in
`~PacketList_` before `clear()` runs.  A mismatch names the interval in which
the element's ownership was lost, and the existing anomaly path will print the
offending release's history keyed on that `Packet`.

### 13.76 Instrumenting the element array itself — and how to read a ledger
### whose `BORN` is 31

§13.75's suggestion, implemented, plus one reading correction that changes
what "balanced" means.

**The ledger is per-ADDRESS, not per-object.**  `BORN 31` means that
address was born **31 times** — 31 incarnations of pool storage.  So
`INC 72 / DEC 72` balancing across 31 lives does **not** imply any single
life balanced: a decrement that lands one incarnation too late leaves life
N−1 short a `DEC` and life N with a spare, and the totals still add up.
That is precisely the stale-reference shape, and it means the balanced
ledger is **consistent with** §13.75's finding rather than a puzzle
against it.  (Also recorded: the `ledger:` line goes to stderr via
`kame_rc_dump`, not the raw sink — §13.75 found that the hard way.)

**The instrument, at the place the two indictments meet.**  Rather than at
either caller, the check now lives in `PacketList_` itself
(`KAME_RC_TRACE_LIST_CHECK=1`), at **both ends of a list's life**:

- its **copy constructor**, before copying the elements — §13.59's
  direction (a copy that would resurrect a dead element);
- its **destructor**, before `clear()` — §13.75's direction (a release
  that would double-pay one).

Each element's `Packet` is tested with the same predicate the `lsp`
tripwires use, keyed on the ELEMENT, so a report brings the element's
release history, its `RC-FREEREC` (who freed the storage, when) and
`slot=` (the element's address, i.e. which list held it).  The `type`
field carries the check site (`PacketList_ copy-ctor element` /
`PacketList_ dtor element`) so the two directions are distinguishable at a
glance.

Deliberately **no new data member**: a class layout must not depend on a
build macro (CLAUDE.md's `USE_RUBY` lesson), so the traced build adds only
member functions and a hand-written copy ctor that forwards exactly what
`= default` did (`fast_vector` copy, `atomic_countable(x)` for
`refcnt = 1`, `m_subnodes`, `m_serial`).

Liveness proved before trusting the zero (§13.61's rule): forcing the
predicate true yields **1 663 344** well-formed reports carrying the check
site.  Reverted.  Mac: 0 reports over 3 dynamic-node runs, full `tests/`
tree **40/40**.

**Ubuntu**: add `KAME_RC_TRACE_LIST_CHECK=1` to the survivor hunt — no
rebuild flags beyond the tracer build already in use.
- Fires in the **dtor** → confirms §13.75 at the array and names the
  releaser that emptied the element while the list still held it;
- fires in the **copy ctor** → §13.59's direction is still live too, and
  the same array is the common cause;
- fires in **neither** while `DEC-UNDERFLOW` still lands in
  `~PacketList_` → the element was live at dtor entry and died *during*
  `clear()`, i.e. the double-pay is inside the vector teardown itself
  (`erase`/spill were audited clean here — `erase`'s swap-based
  move-assign carries the erased value to the tail and destroys it once;
  `move_fixed_to_var` move-constructs then destroys an emptied source),
  which would point at concurrent access to a list presumed private.

### 13.77 The list check is silent because the victim was never a list element — it is `supscope->packet()` (§13.63's site, again)

Ran §13.76's probe (`KAME_RC_TRACE_LIST_CHECK=1`, forensic `.so`,
`setarch -R taskset -c 0-3`, `20 40 700`).  Three runs, two failures; the
`rc=139` one carried **2 `INC-FROM-ZERO` tripwires and ZERO list-check
reports** — neither the copy-ctor direction nor the dtor direction fired.

**Liveness proved locally first** (§13.61's rule): forcing the predicate to
`if(true)` gives **3** well-formed reports carrying the check site
(`type=PacketList_ copy-ctor element`), then reverted — tree clean.  So the
silence is a real negative, not a dead probe.

**And the reason is in the anomaly's own site.**  Both tripwires resolve to
`Node<LongNode>::bundle`, **`transaction_impl.h:2868`**:

```cpp
local_shared_ptr<PacketWrapper> superwrapper(
    make_local_shared<PacketWrapper>(supscope->packet(), bundle_serial));
```

That is **§13.63's anomaly #1 site** (line 2851 before the intervening edits
shifted it).  The `PacketWrapper` constructor copies `supscope->packet()` — so
the resurrected `Packet` is **the scope's own view packet, not an element of
any `PacketList_`**.  §13.76's checks walk the element array; the victim is
never in it.  That is why both directions are silent, and it is a clean
answer rather than a null one: **the element array is excluded as the locus.**

**A correction I owe on my own earlier reading.**  In §13.73 and §13.75 I wrote
that the resurrection signature was *absent* from the FIXED-arm survivors,
on five captures that were all `DEC-UNDERFLOW`.  This capture is a FIXED-arm
survivor with `INC-FROM-ZERO`.  **The survivor population carries both
signatures**; they are not partitioned by fix state, and my "absent" was an
over-read of a small sample.  The two directions are the same defect seen from
either side, exactly as §13.75 argued: a `Packet` whose ownership does not
match its count — copy the holder and you resurrect it, destroy the holder and
you double-release it.

**Where the evidence now converges.**  Three independent captures
(§13.63 #1, and both tripwires here) put the resurrection at the *same*
construction, and it is not a list operation:
`make_local_shared<PacketWrapper>(supscope->packet(), …)`.  The open question
is the one §13.63 posed and it is now the only one standing:
**what drops the last reference to `supscope->packet()` while the scope is
live.**

**Suggested probe** — the same shape that has worked, moved to the right
object: check `supscope->packet()`'s refcount at `ScopedNegotiateLinkage`
construction, at each `packet()` use, and immediately before `:2868`, keyed on
that `Packet` so the anomaly path prints its `RC-PRIOR-RELEASE-FAST`.
Zero-at-entry indicts how the scope acquires its view; going-zero-during
indicts the release that does it, and names it directly.

### 13.78 The probe at the one remaining question — and it dumps the WRAPPER
### too, which is the part that discriminates

§13.77's correction accepted: the survivor population carries **both**
signatures, the element array is excluded as the locus, and three
independent captures put the resurrection at the same construction —
`make_local_shared<PacketWrapper>(supscope->packet(), …)`.  So the single
open question is the §13.63 one: **what drops the last reference to
`supscope->packet()` while the scope holds its wrapper.**

Implemented as suggested, with one addition that changes what a hit can
tell us.  `KAME_RC_TRACE_SCOPEPKT_CHECK=1` arms
`rcScopePacketCheck(scope.operator->(), where)` at two points in `bundle`:

- **entry** (`supscope->packet()` as received from the caller) — a hit
  here indicts how the CALLER's scope acquired its view;
- **immediately before `:2867`'s construction** — a hit only here means
  the packet died between entry and the copy, and the release that did it
  is in the middle.

The addition: on a hit the probe reports the **PACKET** (so the anomaly
path prints its `RC-PRIOR-RELEASE-FAST` — the releaser — plus
`RC-FREEREC`), and then **dumps the WRAPPER's history too**
(`kame_rc_dump(w)`, with `slot=` = the wrapper address).  That second dump
is the discriminator, and it only became possible with §13.74: a wrapper's
life is now completely traced, so its history separates

- **"this wrapper was destroyed twice"** (two `DEC`/`DEAD` on the wrapper
  — a wrapper-level double release, which would explain a live scope
  whose packet is gone), from
- **"this wrapper never took a count on the packet it names"** (the
  wrapper's own `BORN`/`INC` pattern showing an uncounted acquisition —
  the UNCOUNTED shape §13.64 partitioned out and refuted at the three
  `PacketWrapper` constructors, which would then have to be arriving by
  some other route), from
- **"the wrapper is fine and someone else released the packet"** (a
  clean wrapper life, with the packet's own `RC-PRIOR-RELEASE-FAST`
  naming the third party).

All three are distinguishable from ONE capture, which is why the extra
dump is worth its (debug-only, on-hit) cost.

Liveness proved before trusting the zero (§13.61's rule): forcing the
predicate gives **60 028** well-formed reports carrying the check site and
the wrapper address, then reverted.  Mac: 0 hits over 3 dynamic-node runs;
the plain (non-traced) build is unchanged; `tests/` tree **40/40**.

Ubuntu: add `KAME_RC_TRACE_SCOPEPKT_CHECK=1` alongside the existing
tracer flags — no rebuild beyond that.  Whichever of the two sites fires
narrows the interval, and the wrapper dump then names the mechanism.

### 13.79 The scope-packet probe HITS — at bundle ENTRY — and the releaser is `finalizeCommitment`

Ran §13.78's probe (`KAME_RC_TRACE_SCOPEPKT_CHECK=1`, forensic `.so`,
`setarch -R taskset -c 0-3`, `20 40 700`).  Two runs; the `rc=139` one gives
**5 scope-packet hits and 2 anomalies**.

**Every hit is at the ENTRY check** — `type=bundle entry supscope->packet()`,
none at the pre-`:2867` check.  By §13.78's pre-registration that is the
branch which **indicts how the caller's scope acquired its view**: the packet
is already dead when `bundle` is entered, so nothing inside `bundle` kills it
and the resurrection at `:2867/:2868` is a *consequence*, not the cause.  Two
of §13.63/§13.77's three captures pointed inside `bundle`; this says the
defect is upstream of all of them.

Further, **two different threads (867556, 867554) hit on the same `Packet`
(`0x7fffe041f9a0`) at the same slot (`0x7fffc51eb7a0`)** — the dead packet is
reachable from more than one thread's scope, not a single thread's stale local.

**The releaser, from the same capture:**

```
RC-PRIOR-RELEASE-FAST obj=0x7fffe041f9a0 op=DEAD(unique) rc_before=1
                      tid=867556  site=0x1681b
RC-FREEREC            freed_ptr=(=obj) size=32 drift=+0
```

`drift=+0` once more, and **the releasing thread is the same one that then hit
the entry check**.  Site `0x1681b` resolves to
**`Transaction<LongNode>::finalizeCommitment`** (`transaction.h:2863`); its
leaf frames are `is_fixed`/`size`/`empty` on the unrelated
`fast_vector<shared_ptr<Message_>>`, i.e. the usual inlining smear, so the
**function** is the solid datum and the exact line is not.  Reading the
function, the release in it is **`m_oldpacket.reset()` at `transaction.h:2852`**
— the `Transaction`'s own old-packet reference.  (§13.68 already noted that one
`unbundle` caller passes `&tr.m_oldpacket`.)

**This is the second independent capture to name `finalizeCommitment`:**
§13.75's capture 2 attributed its `DEC-UNDERFLOW` to the same function.  Two
captures, two different tripwire directions, one releasing frame.

**The chain now reads end to end:** `finalizeCommitment` drops the last
reference to a `Packet` (`m_oldpacket.reset()`, accounting correct — every
capture has `drift=+0`) **while a live `ScopedNegotiateLinkage`'s wrapper still
names that same `Packet`**; the block is freed and poisoned; whichever code
next touches the wrapper's packet pays for it — `bundle` copying it at
`:2867` (`INC-FROM-ZERO`), or a holder's destructor releasing it again
(`DEC-UNDERFLOW`).  One defect, both signatures, matching §13.77.

**What is still open.**  Why the wrapper's reference is not counted — whether
the scope's view was taken without acquiring, or `finalizeCommitment` releases
a reference the wrapper had already assumed. §13.78's wrapper-history dump is
the right discriminator but **did not reach the log in this capture**: the
process died at the FATAL with only 512 bytes of stderr, so the wrapper dump
was not flushed.  Re-running with the wrapper dump routed to the raw sink (or
flushed before abort) should settle which of §13.78's three mechanisms it is,
from a capture that already reproduces readily.

### 13.80 Two of §13.78's three mechanisms refuted; and why the wrapper dump never printed

Working the §13.79 capture further on Ubuntu.  Two detectors added, both
gated to `KAME_RC_TRACE` (production parity verified below).

**First, a correction to §13.79.**  I attributed the missing wrapper dump to
cache eviction.  Wrong: **`anomaly()` aborts by default** —
`abort_mode_()` reads `KAME_RC_TRACE_ABORT` and returns `true` unless it is
`0` ("default: abort (gdb workflow)").  §13.78's `kame_rc_dump(w)` sits
*after* the `anomaly()` call, so it never ran; neither did anything else
placed there.  **Run these probes with `KAME_RC_TRACE_ABORT=0`.**  With that
set, the follow-up code executes and the capture rate is unchanged.

**Mechanism "the wrapper itself is dead" — REFUTED.**  Added a one-load
discriminator reporting the wrapper's own `refcnt` beside the packet's.  In
the capturing run: **`wrapper-ALIVE = 5`, `wrapper-DEAD = 0`**, with the
wrapper at `refcnt = 1` (alive, singly held by the scope) while its
`m_packet` was poisoned.  The scope is not holding a freed wrapper.

**Mechanism "the wrapper was destructed twice" — REFUTED.**  `PacketWrapper`
had no explicit destructor; added one under `KAME_RC_TRACE` that reads its own
`refcnt` at entry (0 is correct; poison means the storage was already freed)
and only observes.  Across runs that hit the entry check **8 and 4 times**,
`DTOR ON FREED STORAGE = 0`.

**What that leaves — and it sharpens the question.**  All three
`PacketWrapper` constructors initialise `m_packet` **by copy**
(`m_packet(x)`, `m_packet()`, `m_packet(x.m_packet)`), so a live wrapper
*always* holds a count on the Packet it names.  The wrapper is alive, it was
not destructed twice, and its constructor took a count — yet the fatal release
in `finalizeCommitment` finds `rc_before = 1`.  So the wrapper's count did not
fail to be taken: **it was taken away.**  An extra decrement while the Packet
is still live leaves no tripwire (the tripwires fire on zero/poison), which is
exactly why every capture so far has shown the *consequences* and never the
act.

**The next suspect, from reading rather than measurement.**
`PacketWrapper::packet()` has a **non-const overload returning a mutable
reference** (`transaction.h:964`), and `bundle` hands the wrapper's own member
straight into a function that writes through it:

```cpp
local_shared_ptr<Packet> &newpacket(
    reverseLookup(superwrapper->packet(), true, SerialGenerator::gen()));
```

That is a path by which a live wrapper's counted member can be overwritten
without the wrapper knowing — the shape that would produce precisely this
evidence.  I have not demonstrated it, and say so; it is where I would
instrument next (record `w->m_packet.get()` at wrapper construction, compare at
the entry check, and report on mismatch).

**Production parity, with a caveat worth recording.**  Both detectors are
`#ifdef KAME_RC_TRACE`.  Verifying that took a control: a plain build's
`.text` *did* differ from pristine — but inserting **15 pure comment lines** at
the same spot changes `.text` too (same size, one differing region), so
something in this TU embeds `__LINE__`.  Two pristine builds are byte-identical,
so the comparison is otherwise sound.  **`.text` equality is not a usable
production-parity test for edits to this header** unless line counts are held
constant; use it only with a comments-only control alongside.

### 13.81 Catch the ACT, not the consequence: check at the named releaser

§13.80's conclusion — *the wrapper is alive, was not destructed twice, and
its constructor took a count, so the count was **taken away*** — makes the
remaining act **an extra decrement while the Packet is still live**.  That
crosses no zero, so no tripwire can see it; every capture so far has been
a consequence.  Two things follow, and both are actionable now.

**The named releaser has exactly one Packet release in it.**
`finalizeCommitment` touches a Packet in a single place:

```cpp
m_oldpacket.reset();
```

So "the fatal release resolves to `finalizeCommitment`" means precisely:
*that* reset found the count at 1 and took it to 0, while a live wrapper
still named the same Packet.  This commit checks that condition **at that
line, before the reset** (`KAME_RC_TRACE_OLDPKT_CHECK=1`):

- `m_oldpacket.use_count() == 1` — this reset is about to be fatal, **and**
- the node's **published** wrapper (read through a `scoped_atomic_view` on
  `node.m_link`) still names the same `Packet`.

Both true ⇒ a live holder exists whose reference is not in the count, and
we are standing on the act.  Reported keyed on the **Packet** (so the
anomaly prints its own release history — the earlier decrement that
removed the wrapper's count) with `slot` = the published wrapper.

**Why this is the right next capture.**  Everything so far names either the
victim (§13.63/§13.77) or a downstream release (§13.75/§13.79).  This
condition is checked *before* the damage, on the live path the capture
already reaches, so its `RC-PRIOR-RELEASE-FAST` block points at **the
decrement that stole the wrapper's count** — the one event the whole
investigation has never observed.

**Operational notes carried forward from §13.80**: run with
`KAME_RC_TRACE_ABORT=0` (otherwise `anomaly()` aborts before any follow-up
output), and remember the `ledger:` line goes to stderr.

Liveness proved before trusting the zero (§13.61's rule): dropping the
wrapper-match condition yields **565 774** well-formed reports carrying the
check-site string; reverted.  Mac: 0 hits over 3 dynamic-node runs, plain
build unaffected, `tests/` tree **40/40**.

**Also noted for the record**, since §13.80 raised it: `.text` comparison
is not a usable production-parity test for edits to these headers —
inserting comment-only lines changes `.text` too, so something in the TU
embeds `__LINE__`.  The parity argument here rests on the `#ifdef
KAME_RC_TRACE` gate plus a clean plain build, not on byte equality.

### 13.82 The act check does not fire — and the releaser is not consistently `finalizeCommitment`

Ran §13.81's check (`KAME_RC_TRACE_OLDPKT_CHECK=1`, plus
`KAME_RC_TRACE_ABORT=0` and the scope check), forensic `.so`, `setarch -R`,
`taskset -c 0-3`, `20 40 700`.  **14 runs, 9 failures, `ACT = 0` in every
one** — including runs where the downstream entry check fired **5** and **3**
times.

**Read this as a bounded negative, not a refutation.**  §13.81's condition is
`m_oldpacket.use_count() == 1` **and** the **published** wrapper (read via a
`scoped_atomic_view` on `node.m_link`) naming the same `Packet`.  The wrapper
in evidence is the one the *scope* holds (`supscope`), and nothing establishes
that it is still the published wrapper at that moment — a scope can hold a
wrapper the linkage has already replaced.  So `ACT = 0` is consistent both
with "the act is not there" and with "the act is there but the wrapper is no
longer the published one".  Widening to *any* wrapper naming the packet is the
obvious next form.

**A correction that matters more.**  Re-resolving the fatal releaser on the
capturing run gives a **different site from §13.79's**:

```
local_weak_ptr<Linkage>::reset()                atomic_smart_ptr.h:1076
~local_weak_ptr()                               :1051
PacketWrapper::~PacketWrapper()                 transaction.h:1016
atomic_shared_ptr_base<PacketWrapper>::deleter  :795
```

The last count is taken by **a `PacketWrapper` being destroyed**, releasing its
own `m_packet` — not by `finalizeCommitment`.  And this chain is *clean*: every
frame belongs to the release path.  The two captures that named
`finalizeCommitment` (§13.75 cap 2, §13.79) both had leaf frames from an
unrelated `fast_vector<shared_ptr<Message_>>`, i.e. exactly the inlining smear
I flagged at the time and then leaned on anyway.  **The releasing site is not
an invariant of this bug**; three captures give two different sites, and the
cleanest resolution points at `~PacketWrapper`.

**What IS invariant across every capture** — and is what a fix must explain:
`rc_before = 1` at the fatal release, `drift = +0` at the free, and a **live**
wrapper (§13.80: `refcnt = 1`, not double-destructed) still naming the freed
`Packet`.  Two wrappers name one `Packet`; one dies and legitimately releases;
the other's count is simply not there.  Since every ctor copies `m_packet`,
the second wrapper's count was taken away, and §13.80's suspect — `packet()`'s
**mutable reference** handed to `reverseLookup(superwrapper->packet(), …)`,
which can overwrite a live wrapper's counted member — remains the only
mechanism proposed that would produce exactly this.

**Recommended next form of the check**, given the above: at the fatal release
(wherever it occurs — hook `~PacketWrapper`'s `m_packet` release as well as
`m_oldpacket.reset()`), report when the count is about to reach zero while
**any** reachable wrapper names the packet; and separately, record
`w->m_packet.get()` at wrapper construction and compare it at the entry check,
which tests the overwrite hypothesis directly rather than by elimination.

*Process note:* I again lost a capture to a too-narrow log-retention condition
(kept only runs where the new check fired, discarding one that had 5 entry
hits).  Fixed to keep every failing run; worth stating because it is the
second time in this section.

### 13.83 The overwrite hypothesis is not testable by "did m_packet change?" —
### measured: that happens 195 000 times in 40 s

Implemented §13.82's recommendation (record `w->m_packet.get()` at wrapper
construction, compare later) as a tracer-side association table
(`wp_note`/`wp_check`, 8192 direct-mapped slots — no data member, since a
class layout must not depend on a build macro).  It reports **195 171
changes in a single 40 s arm64 run**.

**So "m_packet changed since construction" is not a defect predicate.**
Copy-on-write through the non-const `packet()` is exactly how `bundle`
builds its *private* wrapper, and it does so constantly.  The reporter is
therefore left in place but **OFF by default**
(`KAME_RC_TRACE_WP_REPORT=1`) as a frequency probe only.  This is a
substantive negative: §13.80/§13.82's "only mechanism standing" is not
anomalous *per se*, and the anomaly must be narrower.

**Two instrument bugs of mine, recorded because both cost a run.**
(1) `anomaly()`'s `tname` must have **static lifetime** — it is stored in
the ring and read at dump time — and I passed a `snprintf` stack buffer,
which crashed at the dump (dangling read).  The two pointers now travel in
the numeric fields (`rc_before` = born packet, `slot` = current).  (2) The
`bundledBy` constructor legitimately starts with `m_packet == nullptr` and
is assigned right after (`:1444`/`:1561`, both pre-publish), so without a
born-non-null guard every such wrapper "changed" — that was the flood.

### 13.84 The narrower predicate: writing through a PUBLISHED wrapper

If mutating a *private* wrapper is the design, the defect can only be
mutating the **published** one — that drops a counted reference other
holders still rely on, with no zero crossing and hence no tripwire, which
is exactly the evidence §13.80/§13.82 assembled.

`KAME_RC_TRACE_PUBWRITE_CHECK=1` arms `rcPublishedWriteCheck(node, w,
where)` at **both** `reverseLookup(superwrapper->packet(), …)` call sites
in `bundle`: it reads the linkage's current wrapper through a
`scoped_atomic_view` and reports only when that wrapper **is** the one
about to be written through.  Keyed on the packet, with the wrapper in
`slot` and `rc_before` = the wrapper address.

Liveness proved (§13.61's rule): inverting the private-case early-out
gives **434 719** reports; reverted.  Mac: **0 hits** over 3 dynamic-node
runs, `tests/` tree **40/40**, plain build unaffected.

### 13.85 Division of labour — the Linux side should build instruments too

Raised by the user, and the record supports it.  My last two probes were
designed without a reproducer and **each needed a correction from your
runs**: §13.78's follow-up dump sat after an `anomaly()` that aborts by
default (§13.80 found it), and §13.83's predicate turned out to fire
195 k times (found here only because arm64 runs the same code).  A probe
whose base rate must be measured before it can be trusted belongs on the
machine that can measure it in one cycle instead of three round trips.

Suggested split from here:

- **Ubuntu (has the reproducer)** — write and calibrate detectors
  directly: measure a candidate predicate's base rate BEFORE trusting a
  zero (the §13.83 lesson); iterate probe placement against a live
  failure; own the `rr` line (recording works with `-c 10000`; the replay
  divergence is your PREEMPT_RT/PMU issue, and the watchpoint +
  reverse-continue to the writer is still the highest-value unexplored
  step); run the interleaved A/B for any fix.
- **Mac (no reproduction, ever)** — formal work where an arm64 result is
  the datum: TLA+/GenMC models (`PacketRefcount.tla` is the only one with
  the packet-refcount layer, and the depth-4+/3-thread run is still
  outstanding), source audits of the kind that produced §13.67 and
  §13.71, and the tracer's *mechanics* (ring/cache/marker semantics, the
  `tname` lifetime contract, decrement coverage) where I am the author.
- **Both** — keep the discipline that has actually worked: prove a
  detector fires before believing its zero; measure a predicate's base
  rate before believing its hit; and state which of the two you did.

### 13.86 "Why doesn't heavy preemption fire it?" — it does; and the window
### survives full serialization, so reordering is not what closes it

The question: if this is an algorithmic hole, why does it not fire under
plenty of OS preemption — is the window one that processor reordering
always destroys?  The record answers both halves, and the answer inverts
the usual expectation.

**It DOES fire under preemption — that is what makes it reproduce at all.**
§13.57: `rr record -h` (chaos scheduling) was **17/17 clean**; adding
**`-c 10000`** — a forced preemption every 10 000 instructions — made it
reproduce on essentially every run.  Chaos mode randomises *which* thread
runs; `-c` multiplies *how often* the switch happens.  The fault responds
to preemption **frequency**, not to scheduling randomness.  Ordinary OS
preemption on a 4-core box is orders of magnitude rarer than one switch
per 10 k instructions, which is exactly why the native rate is 10–37 % per
run instead of ~100 %.

**And the window survives complete serialization, so it is not a
reordering window.**  Under `rr` every thread is serialized onto one core
and replayed deterministically: no store-buffer interleaving, no
cross-core visibility delay, no speculative reordering to exploit.  It
reproduces there anyway.  Therefore:

- it is an **interleaving hole** — a logical race between two threads'
  instruction streams — **not** a memory-model artifact;
- "processor reordering destroys the window" is **refuted**: reordering is
  not needed to hit it, so it cannot be what suppresses it elsewhere;
- a missing-barrier-style fix would not address it, which is consistent
  with §13.46/§13.48's ordering fixes not moving the rate.

**Then why is arm64 silent, and why do ASan/TSan/`-O2`/clang suppress?**
Because preemption is *necessary but not sufficient*: the window must
EXIST in the compiled code first.

- arm64/clang and gcc `-O2` are not preempted less — they lack the code
  shape (§13.14's total separation, §13.53's clone census, §13.55's
  dose-response graded with clone count, r = 1.000).  No scheduling
  pressure opens a window that is not there.
- ASan/TSan keep the clone set (§13.53: TSan 8 bodies / 123 refs vs
  `-O3`'s 8 / 121) yet suppress, so they act by **timing**: every shared
  access gains an instrumentation prologue, moving the two racing streams
  apart relative to the window.

**What this says about the defect's shape.**  An interleaving hole that
exists in only one code shape is the signature of a **check-then-act whose
two uses of a shared value can see different values in one shape and the
same value in another** — a source-level read the optimizer may split or
cache differently per clone.  That is the class §13.15's falsifier
targeted (and refuted) for `m_flags`; it has **never been tested on the
STM side**, where it would look like a plain field read once in the source
and used twice, with the compiled clone re-reading it in between.

**Cheap deterministic measurement for the Linux side — sizes the window
instead of guessing at it.**  Sweep `rr`'s `-c`: 200 000 / 100 000 /
50 000 / 10 000 / 2 000, recording the reproduction rate at each.  Because
`rr` replays deterministically the sweep needs no statistics, and the
frequency at which the rate collapses gives the window's **instruction
width** directly:

- still reproducing at `-c 200000` ⇒ a **wide** window (thousands of
  instructions) ⇒ a protocol-level interleaving between two named phases,
  which is findable by reading the phase boundaries;
- needs `-c ≤ 10000` ⇒ a **narrow** window (hundreds) ⇒ the split-read
  class above, which is findable by diffing the `-O3` clone's asm for a
  shared load issued twice where the source reads once;
- finer granularity *reducing* the rate ⇒ the interaction is with switch
  cost, not window position, and the whole framing needs revisiting.

That single number constrains the remaining search more than another
detector would.
### 13.87 The published-write predicate never fires — my overwrite hypothesis is dead in both forms

§13.83's measurement already killed the broad form (195 k "m_packet changed"
events in 40 s — copy-on-write through the non-const `packet()` is just how
`bundle` builds a private wrapper).  The narrow form is now measured here too.

**Base rate first, per §13.85.**  Three short clean runs with
`KAME_RC_TRACE_PUBWRITE_CHECK=1` wrote **no events at all** — the raw sink file
was never even created.  Base rate zero, so a hit would mean something.

**Result: 9 runs, 9 failures, `PUBWRITE = 0` in every one**, including a run
that hit the scope-packet entry check **4** times.

**Liveness proved locally** (§13.61's rule, and the same control the Mac used):
inverting the private-case early-out yields **267 648** reports on a *2 8 200*
run.  The check is reachable, the predicate evaluates, the reporter works.

So `bundle` **never** copies-on-write through a wrapper that is currently
published at the linkage — the write is always to a private wrapper.  My
§13.80/§13.82 suspicion that `reverseLookup(superwrapper->packet(), …)` steals
a live wrapper's counted reference is **refuted in both the broad and the
narrow form**.  It was the only mechanism I had proposed; it is gone.

**The contradiction that survives, stated sharply.**  Every capture shows all
four of these at once:

1. the fatal release finds `rc_before == 1`;
2. `drift == +0` at the free (accounting correct);
3. a wrapper naming that `Packet` is **alive**, `refcnt == 1` (§13.80);
4. that wrapper was not double-destructed (§13.80), and every ctor copies
   `m_packet`, so it should hold a count.

(1) and (3)+(4) cannot both be true of the *same* wrapper instance holding a
*counted* reference.  Two ways out remain, and they are distinguishable:

* **(A) the wrapper's `m_packet` is not a counted reference at that moment** —
  e.g. it was installed by a path that transfers the pointer without the
  count.  §13.83's 195 k says pointer-changing is normal, so only a
  *count-losing* variant qualifies, and the published-write form is now
  excluded.
* **(B) the wrapper observed at the entry check is not the instance that
  named the packet at release time** — pool storage is reused, so a *new*
  wrapper can occupy the same address.  Nothing tested so far distinguishes
  instance identity; `refcnt == 1` is equally consistent with a fresh tenant.

**(B) is the cheaper test and has never been run.**  Give each wrapper an
identity at construction (its `m_bundle_serial` is already there, or record
construction seq in the §13.83 association table) and compare it at the entry
check.  If the identity differs, every "live wrapper names a dead packet"
observation in §13.79–§13.82 is really *address reuse*, and the investigation
has been reading a tenant boundary as a liveness contradiction — which would
also explain why every mechanism proposed against (A) has been refuted.

### 13.88 The `-c` sweep: a graded preemption dose-response, collapse centred near 150 000 instructions

§13.86's pre-registered measurement, run here.  One `.so`
(`-O2 -fipa-cp-clone`), one binary (`tmin_rr`, the guardsize-0 variant),
`rr record -h -c <N> ./tmin_rr 10 40 500`, arms **interleaved round-robin**
so drift cannot bias one arm.  Six rounds.

| `-c` (instructions between forced preemptions) | reproductions |
|---|---|
| 400 000 | 1/6 (17%) |
| 200 000 | 2/6 (33%) |
| 150 000 | 4/6 (67%) |
| 100 000 | 5/6 (83%) |
| 50 000 | 5/6 (83%) |

**Monotone in preemption frequency, r = +0.93.**  Not a step: the rate climbs
smoothly from 17% to ~83% and then saturates by 100 000.  The half-height
point sits near **150 000 instructions**.

**A correction to my own first attempt.**  My initial sweep counted `rc != 0`
as a reproduction, which silently counted **timeouts** as hits — and the
`-c 2000` arm was so slow that runs were exceeding the 900 s limit (one was
still running at 12.5 minutes).  That arm's apparent "1/1" was very likely a
timeout, not a fault.  The rerun records the exit code per run and **excludes
`rc=124` from both numerator and denominator** (zero timeouts occurred in the
final arms).  `-c 2000` is dropped as impractical; it is not needed, because
the collapse is fully contained between 400 000 and 100 000.

**What the shape says.**  A saturating curve rather than a threshold means the
window is not a single fixed-width hole that either does or does not get hit:
more preemption points keep buying more chances up to ~100 000, after which
extra preemptions add nothing.  Read as an order-of-magnitude estimate, the
hole is wide enough that roughly one preemption per **10^5** instructions is
already near-certain to land in it, while one per 4×10^5 lands only ~1 time in
6.  That is a large window in instruction terms — consistent with §13.55's
graded clone dose-response (many small windows summing) and inconsistent with
a single tight check-then-act of a few instructions, which would show a much
sharper knee.

It also confirms §13.86's central point independently: this reproduces under
rr's **full serialization**, at every `-c` tried, so the window is an
interleaving hole between two instruction streams and not a memory-model
artifact — and its width is now measured rather than assumed.

### 13.89 Address reuse refuted too: the wrapper at the entry check IS the instance that was constructed

§13.87 left two ways out of the four-way contradiction, and **(B) — "the live
wrapper seen at the entry check is a different instance from the one that named
the packet at release time, because pool storage is reused"** — had never been
tested.  It is now.

**Implementation.**  Extended §13.83's wrapper→packet association table with an
**incarnation id**: `m_bundle_serial`, which every ctor sets and nothing ever
changes.  `wp_note_id()` records it at all three constructors; `wp_check_id()`
runs at the scope-packet entry check and reports **only** when the address
still carries our note *and* the serial differs — i.e. this storage was reused
since the note.  Tracer-side table only, so no data member and no layout
change; plain builds still compile unchanged.

**Both controls run before reading the result** (§13.61/§13.85):

* **Liveness** — with the `s.id == id_now` early-out removed, a failing run
  reports **12**.  The check is reachable and the reporter works.
* **Base rate** — three short clean runs wrote **no events at all**.

**Result: `REUSE = 0` in every run**, including one with **13** entry-check
hits and another with 4.

| run | rc | entry hits | address-reuse reports |
|---|---|---|---|
| 2 | 139 | **13** | **0** |
| 6 | 139 | 4 | **0** |
| 1,3,4 | 139 | 0 | 0 |

**Coverage caveat, stated because the zero depends on it.**  `wp_check_id`
returns early when the slot no longer carries our note (8192 direct-mapped
slots under heavy churn), so a miss is silent.  The liveness build measures
exactly that population: **12 notes were present** on a comparable failing
run, against **0** mismatches in the real build.  So the sample is small — of
order ten observed opportunities, not hundreds — but it is real, and every one
of them said *same incarnation*.

**So (B) is refuted.**  The wrapper observed alive at `refcnt == 1`, naming a
freed `Packet`, is the same instance that was constructed with that member —
not a fresh tenant at a recycled address.  The four facts of §13.87 stand
together and are not an artifact of reading a tenant boundary.

**Which leaves (A) alone, and cornered.**  The wrapper's `m_packet` must be
losing its count by some path that is neither a wrapper destructor (§13.80),
nor a write through a published wrapper (§13.87), nor ordinary copy-on-write
(§13.83's 195 k base rate). Every mechanism proposed so far is refuted, and
the remaining shape is a decrement that is *attributed to nobody* — which is
consistent with §13.88's measurement that the window is wide (half-height near
150 000 instructions) rather than a tight check-then-act.

**The probe that follows from this**, and the first one that does not need a
mechanism guessed in advance: instrument the *`Packet`* rather than its
holders — record every INC/DEC on the packet a scope's wrapper names, from the
moment the wrapper is constructed, and dump that ledger when the entry check
fires.  §13.74 completed the decrement coverage, so such a ledger is now
complete; the missing decrement will be in it, with its site.

### 13.90 THE ACT, CAPTURED: two decrements with `rc_before == 0`, after the packet was already dead

§13.89 cornered the defect at "a decrement attributed to nobody", and named the
reason it had never been seen: the per-object rings **evict** (§13.74), so by
the time a corpse is found its own history is gone.  So I built a dedicated
ledger that cannot evict by object.

**Instrument.**  `record()` is the single funnel for every traced event, so one
hook there feeds a direct-mapped table (16384 slots × 8 entries) holding the
last decrements per object — site, `rc_before`, tid.  The object address is
stored in the slot, so a collision **resets** the slot rather than silently
attributing another object's decrements.  `dec_dump(packet)` is called from the
scope-packet entry check, i.e. at the moment the dead packet is discovered.
Tracer-side only; plain builds compile and are untouched.

**The capture** (`20 40 700`, forensic `.so`, `KAME_RC_TRACE_ABORT=0`):

```
RC-DECLEDGER obj=0x7ffff4950860 total_decs=26 showing=8 (oldest first)
  DEC[18] site=0x…68b72  rc_before=2  tid=926640
  DEC[19..22] site=0x…63e72 rc_before=2  tid=926643     (churn: INC/DEC pairs)
  DEC[23] site=0x…5a0f7  rc_before=1  tid=926640   <-- legitimate last release, 1 -> 0
  DEC[24] site=0x…63e72  rc_before=0  tid=926643   <-- DECREMENT FROM ZERO
  DEC[25] site=0x…63e72  rc_before=0  tid=926643   <-- again
```

**This is the event the investigation has been chasing since §13.79.**  Thread
`926640` takes the count to zero; thread `926643` then decrements the same
packet **twice more, from zero**, at one site.  Every earlier capture saw only
the consequences (`INC-FROM-ZERO` when someone copied the corpse,
`DEC-UNDERFLOW` when someone released it again after the free); this is the
act itself, and it explains the whole §13.87 contradiction: the wrapper's
count was never "not taken", it was **decremented away by a second party**
while the wrapper was alive and holding it.

**Attribution, with the caveat this section has earned.**  Both the legitimate
release and the from-zero pair resolve into `~PacketWrapper`
(`local_weak_ptr<Linkage>::reset()` ← `~local_weak_ptr` ← `~PacketWrapper`);
the legitimate one additionally carries the
`atomic_shared_ptr_base<PacketWrapper>::deleter` frame, the from-zero pair does
not.  That difference is suggestive — a wrapper destructor running *without*
going through the deleter would do exactly this — but §13.75/§13.82 have twice
shown these `site=` resolutions smeared by inlining, and the innermost frame
here (a **weak** `local_weak_ptr<Linkage>` reset) does not even match the
strong `Packet` decrement being recorded.  **I am not naming the caller on this
evidence.**  A run with `KAME_RC_TRACE_CHAIN=1` is in progress to get real call
chains rather than a single smeared address; the entry check fires in roughly
one run in six, and chain capture slows each run substantially.

**Note also** that §13.80's double-destruction detector cannot see this case:
it tests the wrapper's storage for *poison*, and a destructor invoked on a
wrapper whose refcount reached zero but whose storage has not yet been freed
passes that test unremarked.  That is consistent with its zero result and with
this ledger at once.

**Standing:** the act is observed and reproducible-in-principle (one capture so
far, from a probe with zero base rate and proven liveness).  What remains is
one question, now a narrow one: **which call path performs a `~PacketWrapper`
whose `m_packet` release lands on an already-zero count.**

### 13.91 Deleter-context recording: every post-free decrement happens OUTSIDE the deleter, and one resolves cleanly to `unbundle`'s function exit

§13.90 deferred attribution because the `site=` resolution was smeared and the
chain alternative failed — `KAME_RC_TRACE_CHAIN=1` ran **14 runs with zero
entry hits** (chain capture perturbs timing enough to suppress the window), so
stack walking is not available here.  Recorded the fact directly instead.

**Instrument.**  An RAII marker inside `atomic_shared_ptr_base<T>::deleter`
bumps a thread-local depth (`KAME_RC_TRACE` only), and the §13.90 ledger stores
that depth with every decrement.  No unwinding, no smear: each DEC now says
whether it was reached *through the refcount deleter*.

**Capture** (`20 40 700`, forensic `.so`):

```
RC-DECLEDGER obj=0x7ffff4a90980 total_decs=31 showing=8
  DEC[23] rc_before=1       tid=937109 in_deleter=0
  DEC[25] rc_before=1       tid=937109 in_deleter=1
  DEC[27] rc_before=1       tid=937109 in_deleter=1
  DEC[28] rc_before=POISON  tid=937109 in_deleter=0   <-- on freed storage
  DEC[29] rc_before=POISON  tid=937124 in_deleter=0   <-- on freed storage
  DEC[30] rc_before=POISON  tid=937124 in_deleter=0   <-- on freed storage
```

**Reading the `in_deleter` column correctly:** a `1 -> 0` decrement with
`in_deleter=0` is *normal* — the decrement that reaches zero is recorded before
the deleter is entered.  `in_deleter=1` marks a nested release while destroying
an owner, also normal.  The signal is the three decrements on **poisoned**
storage, and **all three are `in_deleter=0`** — nothing is destroying these
through the refcount path; they are plain `local_shared_ptr` destructions whose
target is already gone.

**One of them resolves cleanly.**  `0x1a73a` →
`~local_shared_ptr<Packet>` ← **`Node<LongNode>::unbundle`,
`transaction_impl.h:3691`** — and 3691 is `unbundle`'s **closing brace**, i.e.
its local `local_shared_ptr<Packet>` objects being destroyed at function exit.
Two such decrements, one thread.  (The other, `0x16c41`, is the familiar
`finalizeCommitment` attribution with unrelated `fast_vector<Message_>` leaf
frames — smeared, and I am not relying on it.)

**What that makes of the two signatures.**  A local in `unbundle` that copied a
slot whose `Packet` was already at zero would show **exactly** the pair this
investigation has been collecting: `INC-FROM-ZERO` at the copy (§13.59 caught
one at precisely this function), then `DEC-UNDERFLOW` at function exit when the
local dies (this capture).  They are plausibly the **acquire and release of one
local variable**, not two independent defects — which would explain why fixing
either end (§13.68, §13.71) moved nothing: both are victims of a slot that is
already dead when `unbundle` reads it.

**So the root question is unchanged but now sharply placed:** what drives a
`Packet` to zero *while it is still referenced from the slot `unbundle` is
about to copy*.  Note this is consistent with §13.88's wide window and with
every refuted mechanism so far, all of which asked "who mishandles the
reference" rather than "who releases the referent early".

**Do I need the Mac for this?**  Not to proceed — the next step is another
Linux-side measurement.  Where it would genuinely help: an audit of which
locals in `unbundle` hold `Packet` references across the CAS loop and whether
any can outlive their referent, and a model of the acquire/release pairing.
Both are source/formal work, which is where §13.85 put the division.

### 13.92 The audit §13.91 asked for — and the lifetime claim it breaks:
### an EMPTY view parked into a CASInfo (base rate on arm64: 0 in 4.1 M)

**The audit.**  Every local in `unbundle` that can reach a `Packet`:

| local | line | holds a count on a Packet? | released at function exit? |
|---|---|---|---|
| `newsubpacket` | 3551 | **no** — a raw `local_shared_ptr<Packet>*` | — |
| **`newsubpacket_val`** | 3595 | **yes** (counted copy, §13.68's fix) | **yes, directly** |
| `newsubwrapper` | 3647 | indirectly, via the wrapper's `m_packet` | yes, but **through the wrapper's deleter** |
| `cas_infos` | 3552 | `Linkage` / views / `PacketWrapper` | via their own deleters |

So the only local that can produce a **Packet** decrement at `unbundle`'s
closing brace **with `in_deleter = 0`** is `newsubpacket_val`.  §13.91's
cleanly-resolved post-free decrement is therefore that local's release, and
its `INC-FROM-ZERO` partner is the copy at `:3595` — one variable's acquire
and release, exactly as §13.91 suspected.

**And that dates the damage earlier than §13.68 assumed.**  `:3595` runs
**before** the CAS loop.  If the slot's `Packet` is already dead there,
§13.68's premise — *"the parked views still hold everything up before the
loop"* — is **false**, and the defect is upstream, in the walk that produced
the slot.

**Where that premise comes from, and how it can fail.**  `walkUpChainImpl`
states it in a comment at the park site:

> *"parent_scope is not used after this point (parent_packet still points into
> the PacketWrapper kept alive by the CASInfo's view)."*

That holds **only if the parked view is non-empty**.  And
`consume_scoped_view()` is a bare `return std::move(m_view);` — **no emptiness
check** — while the CAS loop downstream contains
`if(!scope) return UnbundledStatus::DISTURBED; // view was empty`, i.e. **the
code itself acknowledges an empty parked view is reachable**.  If it is empty,
nothing protects that ancestor wrapper: it can die, its packet dies, and both
the walk's own `*p = make_local_shared<Packet>(**p)` write and the slot handed
back to `unbundle` touch freed memory.  That is "who releases the referent
early" answered as **nobody — it was never held**, which is why every
mechanism asking "who mishandles the reference" was refuted.

**Two checks added, and — per §13.83's lesson — the base rate measured first.**

- `park_note()` counts every parked view, empty or held, and the count is
  printed at exit **whether or not anything fires** (registered from
  `park_note`, not from `anomaly()`, so a silent run still states it).
  **arm64, one run: `EMPTY 0 / held 4 101 676`.**  Zero in 4.1 M samples
  makes an empty park genuinely anomalous — a hit on Linux is signal, not
  noise, and the detector's zero is meaningful.
- An `rcSlotLiveCheck` on the **slot** immediately *before* `:3595`'s copy
  (the previous one checked the copy, i.e. after the fact).  A hit says the
  slot handed back was already dead.

Liveness proved for both (§13.61's rule): forcing the empty predicate gives
**176 182** reports; the pre-copy call site is reached (9 hits with a capped
counter).  Reverted.  `tests/` tree **40/40**.

**Ubuntu, two flags, no rebuild beyond the tracer**:
`KAME_RC_TRACE_SLOT_CHECK=1` and (always on with `KAME_RC_TRACE`) the park
counters — the exit line states the base rate for your build too, which is
worth having even from a clean run.
- **`EMPTY n>0`** → the lifetime claim is broken and the walk is the locus;
  the fix is to reject an empty parked view at the park site (return
  `DISTURBED` there, as the loop already does later) rather than to defend
  its consumers.
- **`EMPTY 0` with the pre-copy check firing** → the slot dies for another
  reason while a non-empty view is parked, which points at the ancestor
  wrapper being released *despite* the view — i.e. back at the wrapper
  layer, but now with a specific interval to look in.

### 13.93 Fault INJECTION on arm64 — §13.92's own hypothesis refuted, cheaply,
### on the machine that never reproduces anything

Asked why the A/B mimicry (hoist a read to fake caching / duplicate a read
to fake splitting) had not been run.  Two honest reasons, then the better
experiment that replaced it.

1. **§13.88 moved the ground under it.**  The `rr -c` sweep is graded with
   half-height near **150 000 instructions** — by §13.86's own
   pre-registration that is the **wide-window** branch, i.e. a
   protocol-level interleaving, whereas A and B are *narrow*-window
   mechanisms.  The measurement weakened the hypothesis the mimicry was
   built to test.
2. **A/B mimicry needs a named site.**  Hoisting or duplicating a read is
   only meaningful at a specific shared read; a shotgun edit tests
   nothing.  No site is named yet.

**What is nameable is a STATE**, and §13.92 named one: an EMPTY view parked
into a `CASInfo`, which the walk's own comment assumes cannot happen.  So
the mimicry moved from "fake the transformation" to **"inject the state"**
— a sufficiency test, and exactly the kind of thing arm64 can do since it
never produces the fault on its own.

**Injector** (`KAME_STM_INJECT_EMPTY_PARK=N`, `KAME_RC_TRACE` only, default
off): every N-th park drops the view's protection (consume into a
temporary that dies) and parks an **empty** view — manufacturing the state
whose natural base rate here is 0 in 4.1 M.

| N (1 in N parks) | outcome (3 runs each) |
|---|---|
| off | 3/3 clean |
| 1000 | 3/3 clean, no tripwires |
| 100 | **3/3 abort** |
| 10 | **3/3 abort** |

**And the abort is loud, immediate and deterministic** — not the fault's
signature:

```
Assertion failed: (*this), atomic_smart_ptr.h:984   (lsp<Packet>::operator->)
  Node<LongNode>::snapshotForUnbundle  transaction_impl.h:2446
     int size = ( *r.parent_packet)->size();
```

So when the walk's `parent_packet` loses its owner, the **very next line of
the walk** dereferences it and asserts.  That is the discriminator:

- the reproducer builds run with **asserts enabled** (§13.10's recipe adds
  no `-DNDEBUG`), and no capture has ever shown this assert;
- therefore `*r.parent_packet` is *not* empty/dead there in the failing
  runs, and **§13.92's empty-park hypothesis is refuted as the fault** —
  it would announce itself instantly instead of corrupting silently and
  surfacing thousands of instructions later.

The hypothesis also splits cleanly, and both halves are dead: *empty park
with the owner dead* → this loud assert (not observed); *empty park with
the owner still held elsewhere* → the packet stays valid, no fault.

**What the injector is worth keeping for.**  It is the first *artificial*
reproduction on arm64 of the general shape "the walk touches a `Packet`
slot whose owner has died", and it shows that shape is **guarded** — which
is itself a constraint on the real fault: whatever it is, it does not pass
through `(*r.parent_packet)->size()` with a dead owner.  Left in place,
gated and off by default, as a sufficiency-test lever for the next named
state; `tests/` tree **40/40** with it present.

**Where this leaves the A/B mimicry.**  Still unactionable for want of a
site, and now also less motivated by §13.88's wide window — but the
*method* has proven itself in the injection form: name a state, manufacture
it here, and see whether the observable matches the real signature.  That
is a Mac-side capability worth using on every future candidate, and it
costs one gated `if`.
### 13.96 Empty parked views: `EMPTY 0` on four crashing runs — the §13.92 premise holds

§13.92's audit confirmed §13.91's reading (one variable, two ends: the
`INC-FROM-ZERO` at `newsubpacket_val`'s copy and the post-free decrement at
its destruction) and proposed the locus: an **empty parked `CASInfo` view**,
leaving the ancestor wrapper unprotected.  Measured here.

**An instrument fix was needed first.**  `park_note`'s base-rate report is
registered with `atexit`, which **does not run after `SIGSEGV`** — and a
crashing run is precisely where `EMPTY > 0` would appear.  Every failing run
reported an empty string.  Added an async-signal-safe readback
(`park_counts()`) emitted from §13.58's crash handler as `RC-SEGV-PARK`, so
the counters survive the crash that motivated them.

**Result — four crashing runs (`rc=139`), counters read from the crash path:**

| run | parked views |
|---|---|
| 1 | `EMPTY 0 / held 91 835` |
| 2 | `EMPTY 0 / held 190 359` |
| 3 | `EMPTY 0 / held 188 262` |
| 4 | `EMPTY 0 / held 771 404` |

Plus a clean run at `EMPTY 0 / held 50 096`, and arm64's `0 / 4 101 676`.
**Over 1.2 M parks observed on this machine across runs that actually
crashed, not one was empty.**

The instrument is live in the sense that matters: `held` and `EMPTY` are
incremented by the *same* call in the *same* function, and `held` counts in
the hundreds of thousands, so the call site is reached constantly — only the
boolean's value is in question, and it is never true.

**The pre-copy `rcSlotLiveCheck` also reported zero** (`DEAD-ELEMENT` = 0 in
all four logs): at the moment `unbundle` is about to copy the slot, the
`Packet` in it is still alive.

**So §13.92's premise holds, and its hypothesis is refuted.**  The parked
views are never empty, and the slot is not already dead at the copy.  That
places the death **after** the copy — i.e. between `newsubpacket_val`'s
acquisition and `unbundle`'s exit — which is the CAS loop, exactly the
interval §13.68 moved the copy *out of* in order to protect it.  §13.68's
reordering was therefore correct in intent and still did not help (§13.72:
16/40 vs 14/40), which now reads as: the copy is safe where it is, but the
count it takes is being cancelled by something during the loop.

**What I would measure next**, and it is again a Linux-side measurement:
sample the packet's refcount at three fixed points in `unbundle` — right
after the copy, once per CAS-loop iteration, and at the closing brace — and
report the first point at which it has reached zero while `newsubpacket_val`
still holds it.  That converts "somewhere in the loop" into a specific
iteration, without needing a mechanism guessed in advance.

### 13.97 Three-point sampling inside `unbundle`: the captured packet never dies there

§13.93 placed the death "after the copy, before the closing brace".  Measured
that directly: sample `newsubpacket_val`'s own `Packet` refcount at three fixed
points — **immediately after the copy (0)**, **at the top of each CAS-loop
iteration (1000+n)**, and **at the closing brace (2000)** — reporting the first
point that sees zero or poison, at most once per call so a hit names an
iteration instead of flooding.

**Controls first:** liveness **39 025** reports with the predicate forced true;
base rate **zero** on a clean run.

**Result: 8 crashing runs (`rc=139`), `DIED = 0` in every one.**  The packet
`newsubpacket_val` holds is alive at the copy, alive at every CAS-loop
iteration, and alive at the closing brace.

**So the §13.93 inference is refuted**: the death is not inside `unbundle`'s
span for the packet that `unbundle` captured.  Combined with §13.93 (slot alive
at the copy, parked views never empty), `unbundle` is now measured clean from
entry to exit with respect to this reference.

**A coverage limitation I have to state, because it may be the whole answer.**
The capture is

```cpp
const local_shared_ptr<Packet> newsubpacket_val(
    oldsubpacket ? local_shared_ptr<Packet>() : *newsubpacket);
```

so on the **`oldsubpacket` branch it is EMPTY** — `_wpkt` is null and my probe
does nothing at all.  Every "DIED = 0" above is therefore silent about that
branch, and that branch is precisely the one §13.68 left allocation-free and
un-copied.  §13.91's cleanly-resolved post-free decrement was
`~local_shared_ptr<Packet>` at `unbundle`'s closing brace; if the *other*
branch is the one running there, the decrementing local is **not**
`newsubpacket_val` and §13.92's audit conclusion ("only `newsubpacket_val` can
decrement a Packet at the closing brace with `in_deleter=0`") needs re-checking
against the `oldsubpacket` case, where that local is empty and some other
`local_shared_ptr<Packet>` must be doing it.

**Next measurement, and it is small:** count how often each branch is taken,
and extend the sampler to whichever `local_shared_ptr<Packet>` is live on the
`oldsubpacket` path (`*oldsubpacket` itself is a caller-owned pointer —
`&tr.m_oldpacket` for one of the three callers, per §13.68 — so the interesting
sample is the caller's object, not a local).  That closes the one blind spot
these three sections have left.

### 13.98 The `oldsubpacket` branch is 11% of calls — and covering it changes nothing

§13.97 left one blind spot: on the `oldsubpacket` branch `newsubpacket_val` is
empty, so the three-point sampler sampled nothing.  Both halves measured now.

**Branch census** (counters emitted from the crash handler, so they survive the
`SIGSEGV` that `atexit` misses — §13.93's fix):

| run | `oldsubpacket` | `newsubpacket` | old share |
|---|---|---|---|
| 1 | 96 468 | 799 931 | 10.8% |
| 2 | 32 074 | 267 334 | 10.7% |
| 3 | 48 869 | 404 593 | 10.8% |
| 5 | 81 202 | 674 334 | 10.7% |
| 6 | 48 748 | 403 507 | 10.8% |

**Remarkably stable at ~10.8%.**  So §13.97 was blind on roughly a ninth of all
`unbundle` calls — a real gap, not a corner case.

**Extended the sampler** to take `*oldsubpacket` (the caller-owned reference
that *is* live on that path) when the branch is taken, keeping the same three
points and the same once-per-call reporting.

**Result: 6 crashing runs, `DIED = 0` on both branches.**  Neither the copied
`newsubpacket_val` nor the caller's `*oldsubpacket` ever sees its `Packet` at
zero or poisoned — not after the copy, not at any CAS-loop iteration, not at
the closing brace.

**So `unbundle` is now measured clean for every `Packet` reference it holds,
on both paths.**  Which sharpens §13.91's finding rather than confirming it:
the post-free decrement that resolved to `unbundle`'s closing brace is **not**
a reference `unbundle` held and lost — every reference it holds is alive
throughout.  Either that attribution was smeared after all (the third time this
section has had to discount a `site=`), or the decrementing object is a
`local_shared_ptr<Packet>` that neither §13.92's audit nor this sampler has
identified.

**Known instrument gap, stated for whoever runs this next:** the crash handler
covers `SIGSEGV`/`SIGBUS` only, so `rc=134` (abort) runs still lose the
counters — two of the eight runs above show empty census fields for that
reason.  Extending §13.58's handler to `SIGABRT` would close it, and is worth
doing before the next census-style measurement.

### 13.99 Two instrument fixes §13.98 asked for: `SIGABRT` coverage, and
### dynamic STM-scope tags that cannot be smeared

§13.98's result — `unbundle` measured clean for **every** `Packet` reference
it holds, on both branches — means either §13.91's `site=` was smeared (the
**third** discounted attribution in this section, after §13.75 and §13.79)
or the decrementing `local_shared_ptr<Packet>` has not been identified.
Both possibilities are attribution problems, so this commit attacks
attribution itself rather than adding another predicate.

**1. `SIGABRT` is now covered** (§13.98's explicit request).  `atexit` does
not run after a fatal signal, and §13.58's handler took `SIGSEGV`/`SIGBUS`
only — so every `rc=134` run lost its census counters, which is exactly
where they were needed.  The handler now also takes `SIGABRT`, which covers
both `assert()` and `anomaly()`'s own abort path.  Verified: an injected
abort (§13.93) now produces the full crash block instead of nothing.

**2. Dynamic STM-scope tags — the `in_deleter` trick generalised.**
§13.91's `in_deleter` depth worked precisely because it needs no unwinding
and no line table: it is set by RAII at the place that matters.  The same
device now marks the STM entry points (`bundle`, `unbundle`,
`finalizeCommitment`, with slots reserved for `snapshotForUnbundle`,
`commit`, `snapshot`), each setting a thread-local bit for the duration of
its call (recursion-safe: an inner RAII that finds the bit already set
leaves it alone on exit).  Every recorded event carries the mask, so:

- `RC-R` history lines gained `scope=…`;
- `anomaly()` emits `RC-STM-SCOPE …`;
- the crash handler emits `RC-SEGV-SCOPE …`, placed **after** the
  architecture split so it appears on both x86-64 and arm64 (the first
  attempt landed inside the x86 branch and was invisible here — caught by
  testing on this machine).

**A post-free decrement now says which STM function was on the stack, and
that statement cannot be smeared by inlining**, because it does not come
from the line table at all.  Applied to §13.91's capture, this settles the
open question directly: if the DEC carries `scope=unbundle` the attribution
was right and the object is one this audit has not found; if it carries
something else (or `(none)`), §13.91's `site=` was the third smear and the
search moves to whatever scope it names.

Verified on arm64: forcing the §13.93 injector produces
`RC-SEGV sig=6` followed by **`RC-SEGV-SCOPE unbundle`** — the tag is live
and correct, on the abort path that previously produced nothing at all.
`tests/` tree **40/40**; the plain (non-traced) build is unaffected.

**For the next Ubuntu run**: no new flags — both fixes are part of
`KAME_RC_TRACE`.  Read `RC-SEGV-SCOPE` on the crash and `scope=` on the
`RC-R` lines of the underflowing object; between them they name the dynamic
context of every recorded decrement without relying on a single `site=`.

### 13.100 Scope tags corrected — and the victims are `Linkage` and `PacketWrapper`, not `Packet`

§13.99's tags work, but the first Ubuntu run read `scope=(none)` on the
bundle-entry check — a probe that is inside `bundle` by construction.

**Instrument fix.**  The entry check sat **three lines above** the
`ScopedStmScope` marker, so its own record was written before the tag existed.
A probe that can report must be placed *after* the marker or its attribution is
silently empty.  Moved the marker above it; the same check now reports real
tags.  (Worth stating as a rule: every scope-tagged probe needs its marker
entered first, and `(none)` on a probe you know the location of is an
instrument bug, not a finding.)

**With that corrected, one capture (`rc=139`, 7 tripwires):**

| count | op | type | scope |
|---|---|---|---|
| 3 | `INC-FROM-ZERO` | **`Linkage`** | **`bundle,unbundle`** |
| 1 | `DEC-UNDERFLOW` | **`PacketWrapper`** | `bundle` |

`RC-STM-SCOPE` shows `bundle` and `bundle,unbundle`; the crash itself carries
`RC-SEGV-SCOPE bundle,unbundle`.

**Two things change here.**

1. **The victims are not `Packet`.**  Every capture from §13.59 to §13.95 was a
   `Packet`, and every probe I built chased `Packet` references — which is
   consistent with §13.93/§13.97/§13.98 all coming back clean.  In this
   capture the resurrections are on **`Linkage`** and the double-release on
   **`PacketWrapper`**.  If `Linkage` is the primary victim, the `Packet`
   damage seen earlier is downstream, and the whole `unbundle`-locals line of
   inquiry (§13.91–§13.98) was auditing a symptom.
2. **The context is the nested descent.**  `bundle,unbundle` means both frames
   are live — `bundle` recursing into `unbundle` — which is an interval nobody
   has instrumented as a unit, and it is exactly where §13.88's wide window
   (half-height ~150 000 instructions) would sit.

**Caveat, stated plainly:** this is **one** capture.  The type distribution is
a real observation from trustworthy attribution, but whether `Linkage` is the
*primary* victim or just another downstream casualty needs more captures — the
tripwire population has been mixed before (§13.77 corrected exactly this kind
of over-read on a five-capture sample).

**Next:** collect several more tagged captures and tabulate op × type × scope.
If `Linkage` resurrections dominate and precede the `Packet` ones in `seq`
order, the investigation should move to `Linkage`'s lifetime in the
bundle→unbundle descent, and the `Packet`-side probes can be retired.

### 13.101 Correction to §13.100: victim types are MIXED — but `finalize` is now confirmed unsmeared

Collected further tagged captures.  Three with tripwires:

| capture | tripwires |
|---|---|
| A | 3× `INC-FROM-ZERO` **`Linkage`** `scope=bundle,unbundle`; 1× `DEC-UNDERFLOW` **`PacketWrapper`** `scope=bundle` |
| B | 1× `DEC-UNDERFLOW` **`Packet`** **`scope=finalize`** |
| C | 4× `DEAD-ELEMENT` (the entry-check probes) `scope=bundle` |

**§13.100 over-read its single capture.**  I wrote that "the victims are
`Linkage` and `PacketWrapper`, not `Packet`" and suggested the `Packet`-side
probes could be retired.  Capture B is a `Packet`.  **The victim population is
mixed** — `Linkage`, `PacketWrapper` and `Packet` all appear across three
captures — so no type is established as primary, and the §13.91–§13.98
`Packet` work was not chasing a symptom on this evidence.  I flagged that
caveat in §13.100 and it is now realized; this is the same over-read §13.77
had to correct, on the same kind of small sample.

**The instrument fix is confirmed working.**  Capture C's entry-check probes
now report `scope=bundle`, where before the reorder every one of them read
`(none)` — so §13.100's marker-ordering fix does what it claimed.

**And one earlier discounted result is rehabilitated.**  Capture B's `Packet`
double-release carries **`scope=finalize`**, i.e. `finalizeCommitment` was on
the stack.  §13.75 and §13.79 both attributed a release there from the line
table, and §13.82 **discounted both as inlining smear** because their leaf
frames were unrelated `fast_vector<Message_>` helpers.  A scope tag is a
thread-local bit set by an RAII marker; it cannot be smeared by inlining.  So
`finalizeCommitment` is now confirmed as a genuine site of `Packet`
double-release, by a mechanism independent of the line table — and §13.82's
dismissal was too strong.

**Standing.**  What the tags have bought is trustworthy attribution, and the
first two things they attribute are: `Linkage` resurrections inside the
**nested `bundle,unbundle` descent**, and a `Packet` double-release inside
**`finalizeCommitment`**.  Whether those are one defect or two is not
established.  The cheap next step is ordering: with `seq` on every record,
tabulate whether `Linkage` events precede the `Packet`/`PacketWrapper` ones
within a capture.  A consistent order would separate cause from consequence;
an inconsistent one would say they are independent.

### 13.102 Installing the three reserved scope markers: `(none)` collapses, and the damage spans four object types across five scopes

§13.101's ordering pass turned up something more useful than the ordering:
**most `Packet` tripwires carried `scope=(none)`.**  In one capture all nine
did.  §13.99 defined six scope bits but installed only three markers
(`BUNDLE`, `UNBUNDLE`, `FINALIZE`); `SNAPFORUNB`, `COMMIT` and `SNAPSHOT` were
reserved and never placed — so `(none)` did not mean "outside the STM", it
meant "in a function nobody tagged".

**Installed the missing three** (`snapshotForUnbundle`, both `Node::snapshot`
overloads, `Node::commit`), each immediately at function entry per §13.100's
ordering rule.  Plain builds unaffected; markers are `KAME_RC_TRACE`-only.

**Result — tripwires now, across captures:**

| op | victim | scope |
|---|---|---|
| `INC-FROM-ZERO` | `Linkage` | `bundle,unbundle,snapforunb` |
| `INC-FROM-ZERO` | `Linkage` | **`snapshot`** |
| `DEC-UNDERFLOW` | `Packet` | **`snapshot`** |
| `DEC-UNDERFLOW` | `Packet` | `finalize` |
| `INC-FROM-ZERO` | **`Payload`** | `(none)` |

**`(none)` fell from the majority to 1 of 5.**  The markers did what they were
reserved for, and `snapshot` — never previously implicated — is now named
twice, for two different victim types.

**The damage is broader than any single hypothesis has assumed.**  Four victim
types are now attested (`Packet`, `Linkage`, `PacketWrapper`, `Payload`) across
five scopes (`bundle`, `unbundle`, `snapshotForUnbundle`, `snapshot`,
`finalizeCommitment`).  That is hard to reconcile with a single narrow defect
at one site, and easy to reconcile with §13.55's graded clone dose-response
(many small windows) and §13.88's wide window (half-height ~150 000
instructions): a refcounting hazard that the `-O3` clone set opens in *several*
STM paths at once, rather than one mishandled reference.

**Sample caveat, again explicitly.**  Five tripwire records.  §13.100 over-read
a single capture and §13.101 had to correct it; I am not claiming a
distribution from five, only that **no scope or type is exclusive** — which is
itself enough to retire "find the one site" as a strategy, the same way §13.53
retired "find the one clone".

**Next**: the remaining `(none)` is a `Payload` resurrection, so at least one
more path still needs a marker. Worth finding it before drawing the
distribution, since it is the only unattributed class left.

### 13.103 The `DOUBLE-LIVE` probe: §6's mixed-compiler result says ALLOCATOR, and §13.102's four victim types are what one double hand-out looks like

**Reading §13.102 against §6.**  §13.102 concluded from four victim types
(`Packet`, `Linkage`, `PacketWrapper`, `Payload`) across five scopes that the
damage is "broader than any single hypothesis has assumed" and retired
"find the one site".  Correct about the site — but the type spread is not
evidence for *many* defects.  **One pool slot handed to two owners produces
exactly this signature**: whichever types happen to share the block are the
ones whose refcounts corrupt, and whichever scope happens to touch them is the
scope that reports.  Four types and five scopes is the *expected* fingerprint
of one address-level fault, not of four protocol faults.

And §6's table already localises the layer, which the last ~40 sections have
not been treating as binding:

| experiment (§6) | result |
|---|---|
| pool ON vs OFF | fires 40–75 % vs **0/6, 0/8** |
| **gcc-STM + clang-pool** | **0/12** |
| **clang-STM + gcc-pool** | **8/12** |
| allocator `-O3` / `-O2` / `-Os` / `-O1` | 6/8, **0/8**, **0/8**, **0/7** |

The fault follows **the allocator's compiler and the allocator's `-O` level**,
not the STM's: a clang-built STM over a gcc-built pool still fires 8/12, and
the reverse is 0/12.  So the `Packet`-lifetime audits (§13.91–§13.98), the
`unbundle`-locals work, and `PacketRefcount.tla` were all auditing a layer the
evidence had already excluded as the *origin* — they were right to find it
clean.  §13.x's own word-cache reading names the mechanism: the read1/read2 gap
in the word-grab loop as "a codegen-induced twin of the BMWIN double-payout
(`f104768b`), in the same machinery."

**What was never done: test that claim in the configuration that fires.**
`alloc_stress_test`'s "256 M ops, 0 violations" is synthetic — a different
workload, different size classes, different thread pattern.  No instrument has
ever asked, under the reproducer, whether one address is ever occupied by two
live objects.

**The probe.**  Every refcounted object announces `OP_BORN` in its constructor
and `OP_DEAD`/`OP_DEAD_UNIQUE` at death, so a live-set keyed on the object
address answers it directly: **a BORN on an address that is still live means
two objects occupy it at once.**  This fires *at* the double hand-out —
upstream of every refcount victim — so it names a cause rather than a
consequence.  `KAME_RC_TRACE_DLIVE=1` counts, `=2` reports through
`anomaly()`; default 1.

**Confined to `rc_trace.cpp` by design.**  A check inside `allocator.cpp`
would change the ipa-cp-clone set under test — §5 records that `noclone` on
one function moved 72 other function sizes — i.e. it could hide the fault it
is looking for.  `rc_trace.cpp` is a separate TU and the pool is a separate
library, so the allocator's codegen is untouched.  Plain builds contain none of
this (`rc_trace.cpp` is not in `support_SRCS`).

**Validated in both directions before any zero is trusted (§13.61, §13.83).**

| step | result |
|---|---|
| first version, base rate | **22.2 M hits / 96.3 M births — instrument error** |
| self-calibrating version, arm64 4 t | 0 hits / **74.2 M enforced** |
| arm64 40 t, 5 runs | **0 hits / 371 M enforced**, dead-miss 0, table-full 0 |
| positive control: 1-in-1 M injected double occupancy | **75 hits** (74 expected) |
| report path (`=2`) | full `RC-ANOMALY` + prior occupant's release record |

The first version's 23 % "hit" rate is worth recording as the failure mode it
was: a stack or embedded `atomic_countable` announces BORN but never DEAD, and
its address recycles every call.  The fix makes the probe **self-calibrating**
— it enforces only on addresses a *traced death* has proven to be recycling
heap slots (`DL_EVERDEAD`).  That keeps 77 % of births under enforcement while
taking the false-positive rate to zero across 371 M checks.  It **under**-detects
in one known way, stated so it is not over-read: the key is the
`atomic_countable` subobject address, so two types whose subobject sits at
different offsets inside one block would overlap without colliding here.

**For the Ubuntu side — one env var, no rebuild beyond this commit:**

```
KAME_RC_TRACE_DLIVE=2 KAME_RC_TRACE_ABORT=0 ./tmin ...
```

- **HITS > 0** → the allocator handed one slot to two live owners in the
  failing configuration.  That closes the hunt at the layer §6 pointed to, and
  the `RC-PRIOR-RELEASE-FAST` line names the other occupant.
- **HITS = 0 across the failing runs** → the whole double-hand-out class is
  refuted *at the address level*, on an instrument with a 371 M-check clean
  baseline and a working positive control.  The STM then genuinely holds a
  stale reference, and the remaining allocator story has to be about *reuse
  timing* (which §6's non-monotone quarantine result already hints at) rather
  than about exclusivity.

Either outcome is decisive, which is why this is worth running before any
further site-level audit.

### 13.103b Targeted preemption injection on arm64: null — and `sched_yield` is not a perturbation on macOS

Testing the standing intuition that the window is one macOS preemption cannot
enter (rather than one clang never emits).  `KAME_STM_YIELD_AT=<site>` with
`KAME_STM_YIELD_EVERY` / `KAME_STM_YIELD_US` deschedules at six hand-picked
points: 1 `unbundle` after the pre-loop capture, before the CAS loop; 2 inside
that loop after `compareAndSet` (loop-local scope alive); 3 `bundle` Phase 2
published, before `set_view`; 4 Phase 4 CASed, `sw1` still held by `supscope`;
5 `finalizeCommitment` before `m_oldpacket.reset()`; 6 walk, just after parking
the view.  `KAME_RC_TRACE`-only; plain builds unaffected.

**Result: 0 failures, 0 anomalies at every site.**  But the first sweep was
worthless and that is the more useful finding: **`sched_yield()` barely
perturbs anything on macOS** — 21.16 s → 21.51 s with a yield at *every*
`unbundle` — because with idle cores there is nobody to yield to.  `usleep(1)`
is a real deschedule (21.16 s → 36.07 s), and `usleep(50)` at every call slows
the test past a 180 s timeout, so that arm carries no information either (all
its "failures" are exit 124).  The informative arm is therefore
`us=1 × 6 sites`, which is clean.

So this does not separate "clang lacks the window" from "macOS never lands in
it" — it only says that six specific points, deschedule-injected, do not open
it. Recorded as a null result with its coverage stated, and as a standing
caution: **measure that a perturbation perturbs before believing its zero**
(the same rule §13.83 established for predicates).

### 13.104 DOUBLE-LIVE FIRES — a crashing run hands out an occupied address, and the refcount damage follows it

§13.103's probe run on Ubuntu.  **Instrument fix first, same lesson as
§13.93/§13.99:** `dl_report_` was reachable only via `atexit`, which does not
run after a fatal signal, so every crashing run reported nothing.  Added a
signal-safe `dl_counts()` readback emitted from the crash handler as
`RC-SEGV-DLIVE`.  Without it this section's result would have been invisible.

**Base rate — zero.**  Clean runs: `born 1 234 375 / enforced 837 948 /
HITS 0`, and a longer clean run `born 51 443 040 / enforced 20 202 912 /
HITS 0`.  **Zero double-live hits in ~21 M enforced checks on runs that
succeeded.**

**Crashing runs:**

| run | rc | born | enforced | **HITS** |
|---|---|---|---|---|
| 2 | 139 | 25 689 514 | 9 011 912 | **3** |
| 3 | 139 | 12 794 248 | 5 707 402 | 0 |
| 4 | 139 | 7 697 087 | 3 926 128 | 0 |

**And the three hits are not incidental — they lead the damage.**  On one
address, in `seq` order, one thread:

```
obj=0x7fffd0676f90
  #1  DOUBLE-LIVE   BORN at an address already occupied by a live object
                    scope=bundle,snapshot
  #2  DEC-UNDERFLOW type=PacketWrapper
  #3  DEC-UNDERFLOW type=PacketWrapper
```

**The double hand-out comes first, and the refcount corruption is on the same
address afterwards.**  That is the ordering §13.101 asked for and never got
from the victim types: cause before consequence, one address, one thread.  It
also explains §13.102's four victim types in one stroke — whichever types
happen to share a doubly-handed-out block are the ones that corrupt, which is
§13.103's prediction, made before this run.

**Site attribution, with the standing caveat.**  The second occupant's `BORN`
resolves through `lsp<PacketWrapper>::swap` ← `operator=(&&)` ← `bundle`
(`transaction_impl.h:3304`), and the recorded previous-occupant slots to
`bundle:3032` and `snapshotForUnbundle:2367`.  These are line-table
resolutions and this section has discounted three of them (§13.75, §13.79,
§13.82); the **scope tag** `bundle,snapshot` is the part that cannot smear.

**Two coverage caveats, both real.**  (1) The probe's live-set saturates:
`table-full 26 851 369` and `dead-miss 25 616 655` against 20 M enforced, so
roughly half the births are untracked and **every HITS figure is a lower
bound** — including the zeros.  (2) Only 1 of 3 crashing runs showed hits,
which given (1) is as consistent with coverage as with distinct failure modes.

**What this establishes.**  A double hand-out is **real, observed, and ordered
ahead of the refcount damage** — it is no longer a hypothesis.  Combined with
§6's table (fault follows the *allocator's* compiler: clang-STM+gcc-pool 8/12,
gcc-STM+clang-pool 0/12, allocator `-O3` 6/8 vs `-O2` 0/8), the layer and the
mechanism now agree, and every STM-side lifetime audit coming back clean
(§13.91–§13.98) reads as correct rather than puzzling.

**Next, in order:** enlarge the probe's table (or key it more sparsely) so the
zeros become trustworthy, then re-run to establish how often hits accompany a
crash; and take the `bundle,snapshot` scope as the place to look on the
allocator side, since that is the one attribution here that cannot be smeared.
### 13.105 `DOUBLE-LIVE`, second key: the subobject offsets are NOT uniform, so the exact key has a structural blind spot

§13.103 flagged one honest limit — the probe keys on the `atomic_countable`
subobject address, so two types whose subobject sits at a different offset in
one block would overlap without colliding.  Measured, rather than left as a
caveat:

| type | `sizeof` | `atomic_countable` offset |
|---|---|---|
| `Packet` | 32 | **0** |
| `PacketWrapper` | 40 | **0** |
| `Payload` (base) | 40 | **8** |
| `LongNode::Payload` | 48 | **8** |
| `PacketList` | 104 | **72** |

So the blind spot is real and it lands exactly on the pair §13.102 names:
`PacketWrapper` (offset 0) and `Payload` (offset 8) sharing one size class are
invisible to an exact-address key, while `Packet` vs `PacketWrapper` (both 0)
are covered.  23 % of all births are at non-16-aligned addresses — that is the
offset-8 population, and it is now reported as `nonaligned`.

**First attempt, rejected on cost.**  Probing the +/-8 neighbours explicitly is
sound (two live countables 8 bytes apart cannot be distinct blocks when the
smallest class is >= 32) but tripled the lookups per BORN and pushed the
40-thread run past a 300 s timeout — no result at all.  Recorded because the
soundness argument is worth keeping even though the implementation is not.

**Second attempt: round the key down to 16 bytes.**  Folds the {0, 8} family
onto one key at *zero* extra probes, and raises enforcement coverage 15 %
(74 M → 85 M per run) because more keys acquire the `EVERDEAD` proof.

**But it does not keep the zero base rate, so it is opt-in, not the default:**

| key | base rate on arm64 (nothing fails here) | coverage |
|---|---|---|
| **exact** (default) | **0 hits / 518 M enforced, 7 runs** | 74 M/run |
| `KAME_RC_TRACE_DLIVE_FOLD=1` | **3 hits / 509 M enforced, 6 runs** | 85 M/run |

The folded hits are not obviously instrument error: one came from
`Transaction::operator[]` constructing a `Payload` at an offset-8 address —
precisely the cross-offset event the fold exists to see.  Genuine co-tenancy
and a stale `LIVE` bit left by a death that reaches no hook are both plausible
at ~1 in 10^8 births, and this is not settled here.

**Both are kept because they buy different things.**  The exact key is the
DECISIVE instrument: one hit means something, because its baseline is zero over
518 M checks and its positive control still fires at the predicted rate (75
hits at 1-in-1 M injection).  The fold is the SENSITIVE one: it can see an
overlap the exact key structurally cannot, at the price that a single hit needs
corroboration.  Ubuntu should run **exact first** — a hit there is the decisive
outcome — and reach for the fold only if exact stays clean, where its extra
reach is the whole point and its 3-in-509 M baseline is the number to beat.

### 13.106 Making §13.104's zeros trustworthy: the live-set was leaking, and saturation collapses coverage to nothing

§13.104's caveat (1) is the one thing standing between "DOUBLE-LIVE fires on a
crashing run" and "DOUBLE-LIVE fires on N of M crashing runs":
`table-full 26 851 369` / `dead-miss 25 616 655` against 20 M enforced, i.e.
roughly half the births untracked and every HITS figure a lower bound.  Two
causes, both fixed.

**1. The table leaked.**  An `EVERDEAD`-without-`LIVE` entry was never
reclaimed, and `dl_dead_` *inserts* one for every death whose birth it never
saw — so the table filled with dead marks and then refused new claims.  A full
probe window now **steals the first non-`LIVE` slot in it**.  Stealing a dead
mark costs only that address's `EVERDEAD` proof, which must be re-earned before
enforcement resumes there: it can **under**-detect and cannot manufacture a
hit.  `steals` reports the rate.

**2. 2^20 slots was too few for a sparse address set.**  `DL_BITS` now comes
from `KAME_RC_TRACE_DLIVE_BITS` (default **22**, clamped 8..26), so Ubuntu can
turn it without a rebuild.  The mapping is lazy — an oversized table costs only
the pages touched.

**Why macOS never saw this, which is itself worth recording.**  Forcing
saturation by shrinking the table does not work here: `steals 0, table-full 0`
at **bits=10 (1024 slots)**.  So this reproducer's 96 M births recycle **fewer
than ~1000 distinct addresses** on macOS/arm64, while Ubuntu's run saturates a
million-slot table.  Same test, same pool, and the reuse distance differs by
orders of magnitude.  Noted as an observation, not a claim about the fault --
but it is the kind of difference the arm64 silence would live in, since a
freed block being handed straight back leaves a much narrower window in which
"freed but still referenced" is distinguishable from "still valid".

**Positive control for the steal path** (`KAME_RC_TRACE_DLIVE_HASHOFF=1`
collapses every key onto one probe window, since table size cannot force
saturation here):

```
enforced 16   HITS 0   steals 114   table-full 94 874 884   dead-miss 96 215 947
```

- The steal path **executes** (114) and produces **no false positives** (HITS 0)
  — the claim that stealing only under-detects, measured rather than argued.
- Only 114 steals against 94 M full windows is correct behaviour: once the
  single window holds 8 *live* entries there is nothing stealable, and a `LIVE`
  slot is never taken.
- **Enforcement collapses to 16.**  That is the cost of saturation demonstrated
  directly, and it is the regime §13.104 was measuring in — so its zeros were
  indeed lower bounds, now confirmed from the other side.

**Regression check on the clean path** (arm64, 40 threads): exact key at
bits=22 and bits=20, folded key at bits=22 — all `HITS 0, steals 0,
table-full 0, dead-miss 0`; positive control still 75 hits at 1-in-1 M
injection.  `steals 0` matters: the new path is **inert without pressure**, so
it cannot have perturbed the 518 M-check baseline §13.105 rests on.

**For Ubuntu:** re-run §13.104 with `KAME_RC_TRACE_DLIVE_BITS=24` and check
that `table-full` and `dead-miss` fall to near zero *before* reading the HITS
column.  Only then is "hits accompany a crash N of M times" a measurement
rather than a lower bound.

### 13.107 A destructor hook makes the live-set exact — and that is what upgrades §13.104's hit from "a double hand-out" to "a slot recycled under a constructed object"

**The gap §13.104 could not exclude.**  `DOUBLE-LIVE` means "BORN at an address
whose previous occupant never announced DEAD", and three readings reach that:

- **(a)** the allocator handed out an occupied block;
- **(b)** the block was freed while its object was still live, then legitimately
  reallocated — which is the *original* UAF hypothesis, not a new mechanism;
- **(c)** the object legitimately died and the tracer simply missed its `DEAD`
  — a stale `LIVE` bit, i.e. instrument error.

§13.104 read its hit as (a).  On the evidence available then, (b) fits the
observed order (`DOUBLE-LIVE` then two `DEC-UNDERFLOW` on the same address)
exactly as well, and (c) could not be bounded: `born` and `dead` balanced only
to ~2 e-5, a gap **1000x larger** than the probe's own hit rate.  §13.105's own
3-in-509 M folded-key hits were the same ambiguity on the Mac side.

**The fix is one line in the one place every death must pass.**
`~atomic_countable()` runs whatever released the object, so a hook there feeds
the live-set independently of how complete the release hooks are.  It feeds the
live-set ONLY — no event-ring record, no ledger entry — so every prior analysis
keeps the semantics it was validated with.

**Result (arm64, 40 threads):**

| | before | after |
|---|---|---|
| `dtor` vs `born` | (no dtor feed) | **96 588 856 vs 96 588 856 — exactly equal** |
| enforced / born | 74 M / 96 M = **77 %** | 96 588 542 / 96 588 856 = **99.9997 %** |
| base rate, exact key | 0 / 518 M | **0 / 290 M (3 runs)** |
| base rate, folded key | **3 / 509 M** | **0 / 96.8 M** |
| positive control | 75 @ 1-in-1 M | **97 @ 1-in-1 M** (96.6 M enforced) |

`dtor == born` exactly, on every run, is the completeness proof: the
bookkeeping is now exact rather than balanced-on-average.  And **§13.105's
folded-key hits are gone**, which settles that open question — they were (c),
untraced deaths, not genuine co-tenancy.  Both keys now have a zero baseline,
so the fold is usable without corroboration.

**What a hit means now — narrower and stronger than §13.104 claimed.**  With
the destructor feed, (c) is closed by construction.  And (b) collapses too, on
inspection: a premature free *through the smart-pointer path* is impossible to
reach here, because that path frees only after the refcount hits zero, which
runs `DEAD` **and** the destructor — so it could never present as a
`DOUBLE-LIVE`.  A (b) that survives therefore requires a free that bypasses the
object protocol entirely.  So:

> A `DOUBLE-LIVE` hit means **a slot was recycled while its previous occupant
> was still a constructed object** — the destructor never ran.

That is an allocator-level statement, and it supports §13.104's conclusion for a
sharper reason than §13.104 gave.  It also stays agnostic about *which*
allocator path: a claim handing out an occupied slot, and a reclaim taking a
slot back under a live object, both land here.

**Source note, closing one named suspect.**  The two claim paths §13.x
nominated are **sound as written**: the word-grab loop takes ONE
`atomicLoadAcquire` into `oldv`, derives `mask = ~oldv`, and CASes
`oldv -> ~0`, so a successful CAS proves the word transitioned exactly from
`oldv` and the claimed bits are precisely `mask`; the N-run loop has the same
single-load shape (`one` and `newv` both from `oldv`).  An atomic load is not a
rematerializable pure expression, so after §13.15's conversion gcc cannot split
these into two reads — consistent with that conversion changing nothing.
Whatever the `-O3` clone set does, it is not this.

**For Ubuntu, in order:** (1) re-run with `KAME_RC_TRACE_DLIVE_BITS=24` and
confirm `table-full`/`dead-miss` are ~0 *before* reading HITS (§13.106);
(2) confirm `dtor == born` in the report — if they differ, some death is
escaping even the destructor and that is its own finding; (3) then the HITS
column is a measurement, and `scope=` on each hit is the one attribution that
cannot smear.

### 13.108 The tightened probe on Ubuntu: DOUBLE-LIVE accompanies **every** failing run, zero on clean

§13.107's destructor-hooked live-set, run in the firing configuration.  Its
soundness reproduces here: **`dtor == born` exactly** (1 230 350 both on a
clean run), `table-full 0`, `dead-miss 0`, `steals 0` — the ~43% coverage and
the 2e-5 born/dead gap that made §13.104's zeros unreliable are gone.  The
`KAME_RC_TRACE_DLIVE_INJECT` positive control fires, so the probe is live.

**Clean runs: `HITS 0`** (1 047 796 enforced).

**Failing runs — five of five:**

| run | rc | enforced | **HITS** |
|---|---|---|---|
| 1 | 139 | 25 853 245 | **2** |
| 2 | 139 | 1 884 575 | **1** |
| 3 | 139 | 15 792 428 | **2** |
| 4 | **255** | 48 101 233 | **1** |
| 5 | 139 | 25 379 050 | **2** |

**Every failing run has at least one hit; clean runs have none.**  §13.104 saw
1 of 3, which I attributed to the probe's saturation — that reading is
confirmed: with coverage at ~100% the association is complete.

**Run 4 matters disproportionately.**  `rc=255` is the test's *own*
`objcnt` consistency check failing, not a segfault — so a double-live hit
accompanies the STM-level correctness failure as well as the crashes.  The
probe is not merely tracking "this run happened to segfault".

**What §13.107 makes of a hit is what makes this worth stating.**  With the
destructor hook, reading (c) — a stale LIVE bit from an untraced death — is
closed by construction, and (b) — a premature free under a live object —
cannot present as DOUBLE-LIVE, because freeing through the smart-pointer path
happens only after the refcount reaches zero, which runs DEAD *and* the
destructor.  So a hit now means what §13.107 states: **a slot recycled under a
still-constructed object.**

**Caveat kept:** enforced/born is ~92–99% here (`nonaligned` accounts for the
remainder), so hits remain a lower bound — but the zeros on clean runs are now
measurements rather than artefacts, which is the property §13.104 lacked.

**Where this leaves the investigation.**  The association is now: failing run
⇔ slot recycled under a live object, with the recycle ordered *ahead* of the
refcount damage (§13.104's per-address sequence) and the fault following the
allocator's compiler (§6).  Three independent lines agree on the allocator
layer.  The open question is no longer *whether* but *which* recycle path —
and that is an allocator-side question, on the arm where it fires.

### 13.109 Which recycle path? — a free-record discriminator on every hit. Plus a correction: the Mac baselines were measured with the pool INACTIVE

**Correction first, because it touches §13.103–§13.107's numbers.**  The
hand-rolled Mac build used for every DOUBLE-LIVE baseline compiled
`allocator.cpp` straight into the test executable.  That links the pool in but
does **not activate it**: `kame_pool_reserved_bytes()` returned **0** and freed
blocks carried **no forensic poison** — `new`/`delete` were libc's.  The
activation (`constructor(101)` + `__DATA,__interpose`) exists only under
`KAMEPOOLALLOC_DYLIB` in a **shared** library, which is how
`kamepoolalloc/tests/CMakeLists.txt` builds it.  **This is the second time this
trap has bitten this investigation** (the first invalidated a round of TSan
numbers); it is worth a standing rule: any hand-rolled build must be checked
with `kame_pool_reserved_bytes() != 0` before its numbers are quoted.

What survives, what changes:

- **Survives** — everything internal to the tracer, which does not depend on
  which allocator runs: `dtor == born` exactly (§13.107), the positive control
  firing at the injected rate, and §13.105's folded-key hits being untraced
  deaths rather than co-tenancy.
- **Withdrawn** — §13.106's remark that "96 M births recycle fewer than ~1000
  distinct addresses on macOS", offered as a contrast with Ubuntu's saturation.
  That was **libc malloc's** reuse pattern, not the pool's, and it cannot be
  compared with a pool-active run.  The saturation *fix* stands; the
  reuse-distance observation does not.
- **Re-measured with the pool active** (`reserved = 33 554 432`, poison present
  on freed blocks), arm64 40 threads:

| key | runs | born | enforced | HITS |
|---|---|---|---|---|
| exact | 3 | 96.1–97.2 M each | 99.85 % of born | **0** |
| 16 B-folded | 1 | 96.2 M | 99.9 % | **0** |
| exact + inject 1-in-20 M | 1 | 97.1 M | — | **5** (control fires) |

`dtor == born` exactly on every one of them.

**Now the discriminator.**  §13.108 ends at "no longer *whether* but *which*
recycle path", and §13.107's narrowing makes that question answerable, because
a **legitimate** free cannot produce a DOUBLE-LIVE at all (freeing through the
smart-pointer path happens only after the refcount reaches zero, which runs
`DEAD` *and* the destructor).  So the slot reached the bitmap without this
object being freed, and the forensic poison separates the ways that can happen:

| what the hit shows | reading |
|---|---|
| tag present, `freed_ptr == obj` | the block **was** freed while its object was live — a premature free |
| tag present, `freed_ptr != obj` | **someone else's** free carries this block's ring record — the mis-derived `chunk_base` shape (§13.x's `back_offset[unit_idx]`): a bit cleared in the wrong chunk |
| no tag at all | the block was **never freed**, yet was handed out again — a claim-side double hand-out |

**The pre-store capture is what makes it work on Ubuntu's hit shape.**  A freed
block carries the token in word 0, but for every type whose `atomic_countable`
sits at offset 0 (`Packet`, `PacketWrapper`) the constructor's `refcnt = 1`
lands on exactly that word before `OP_BORN` can be observed — and §13.104's hit
(`0x7fffd0676f90`, offset 0) is that shape, so the probe would have been blind
on the one capture it exists to read.  Under `KAME_RC_TRACE` only, `refcnt` is
therefore initialized in the constructor **body** instead of the
mem-initializer, and `preborn_note()` reads word 0 first into a thread-local
that `record()` consumes on the same call — no table, no keying.

Validated end to end with the pool active (injected control, so every verdict
is the expected "THIS block"):

```
RC-DLIVE-WPRE 0x110100120 w=0xbaad0000000b8000  <-- POISON TAG
RC-DLIVE-FREEREC w-1 freed_ptr=0x110100120 (THIS block -- premature free)
    size=32 free_tid=1 age_tsc=167940 frames=4 0x104a16d14 0x109598aa8 ...
```

One trap fixed while validating: `KAME_POISON_PLAIN` (`0xBAADF00DBAADF00D`, the
word-2-and-beyond filler) also has `0xBAAD` in its top 16 bits, so it matched
the tag test and then failed to decode — reported as "ring wrapped", a
different and misleading claim.  It is now named as filler.

**For Ubuntu:** nothing new to pass — a DOUBLE-LIVE hit now carries
`RC-DLIVE-WPRE` / `RC-DLIVE-FREEREC` beside its `scope=`.  One real hit should
land in exactly one row of the table above, and that row names the allocator
path to read.
### 13.110 The hits cluster by chunk — pointing at chunk-level recycling, not per-slot

Free analysis on §13.108's five captures, no new instrumentation.

**Do multiple hits in one run share a chunk?**  Using the 256 KiB chunk mask
(`addr & ~0x3ffff`; 262 144 is the block size TSan reported in §13.47):

| run | hit addresses | same chunk | delta |
|---|---|---|---|
| 1 | `0x7fffca05cf00`, `0x7fffde46f200` | no | 324 MiB |
| 3 | `0x7fffd8065880`, `0x7fffd806f540` | **yes** | 40 128 B |
| 5 | `0x7ffff4c5b080`, `0x7ffff4c5b160` | **yes** | **224 B** |
| 2, 4 | single hit | — | — |

**Two of the three multi-hit runs have both hits in the same chunk**, and in
run 5 the two doubly-occupied addresses are **224 bytes apart** — adjacent
slots.

**Why that is informative.**  A per-slot recycling bug would place its hits
independently; the chance of two independent hits landing in the same 256 KiB
chunk out of the ~10^4 chunks a run touches is negligible, and run 5's 224-byte
separation is stronger still.  Co-located hits are what **chunk-level** reuse
looks like: a whole chunk handed back out while objects inside it are still
constructed, so several slots in it double-occupy at once.  That matches the
event kinds the pool already distinguishes — `KAME_PEV_CHUNK_RECYCLE` /
`CHUNK_RELEASE` / `DLL_DRAIN` operate on chunks, `BATCH_RETURN` on slots.

**The existing pool-event tail cannot close this.**  The anomaly dump carries
only the last 20–68 `RC-POOLEV` records, and in 4 of 5 captures **none of them
is within 1 MiB of the hit address** — the relevant chunk event has long
scrolled off.  So the correlation has to be made at the hit, not after it.

**Concrete next probe** (allocator-side, and small): at a DOUBLE-LIVE hit,
report the **chunk base** of the address and the most recent pool event
recorded *for that chunk* — a per-chunk last-event slot, not a global ring, so
it cannot scroll away.  If the answer is consistently `CHUNK_RECYCLE` (or
`DLL_DRAIN`, the owner-exit path §13.79's timing already implicates), that
names the recycle path §13.108 left open, and the fix has a specific site.

**Caveat:** three multi-hit runs, two clustered.  The 224-byte pair is hard to
get by chance, but this is a pattern in five captures, not a rate.

**Pool-active verification for the Ubuntu numbers (§13.109's standing rule).**
Checked rather than assumed, and the first attempt was wrong: calling
`kame_pool_reserved_bytes()` in a program that has not allocated returns **0**,
which looks exactly like the inactive-pool trap.  After 200 000 `new`s against
the same `libkp_forensic.so` these binaries link, it reads **33 554 432** — the
same 32 MiB the Mac reports for a pool-active run.  The Ubuntu builds link the
allocator as a **shared library** built with `-DKAMEPOOLALLOC_DYLIB`, which is
the activating configuration.  Corroborated in the captures themselves: the
logs carry `RC-POOLEV` records (the pool's own event ring) and `rc_before`
values in the forensic-poison range.  **So §13.104, §13.108 and §13.110 were
measured with the pool active** and are unaffected by §13.109's correction.

### 13.111 The free-record discriminator on Ubuntu: no poison found — with one word of it uninformative by construction

Ran §13.109's discriminator in the firing configuration.  Two failing runs,
**4 and 2 DOUBLE-LIVE hits**, and in both the probe reports words but
**`RC-DLIVE-FREEREC` never fires** — no forensic poison token was found:

```
RC-DLIVE-WPRE 0x7fffc9b6ffd0 w=0x1      RC-DLIVE-WPRE 0x7fffc4ef07b8 w=0x1
RC-DLIVE-W0   0x7fffc9b6ffd0 w=0x1      RC-DLIVE-W0   0x7fffc4ef07b8 w=0x1
RC-DLIVE-W1   0x7fffc9b6ffd0 w=0x7ffff40c0000   RC-DLIVE-W1 … w=0x7ffff4040060
```

**W0 is uninformative, and it is worth saying why before anyone reads it as
evidence.**  `atomic_countable`'s constructor is

```cpp
atomic_countable() noexcept : refcnt(1) { KAME_RC_EVT(this, OP_BORN, 1); }
```

— `refcnt` is initialised **before** the body announces BORN.  The probe runs
at BORN, so by then the incoming object has already written `1` over word 0.
`w=0x1` there is the new occupant's own refcount, not a statement about what
was in the block, and **any poison that had been at word 0 is destroyed before
the probe can see it**.

**W1 *is* informative.**  `atomic_countable` is a base class, so at BORN only
its own word has been written; the derived object's members are not yet
constructed.  W1 lies in that not-yet-written region, and in **both** hits it
holds a plain pointer-shaped value, **not the poison token**.

**So, as far as two hits can say:** the doubly-occupied block was **not freed
through the poisoning deallocate path**.  That is the outcome §13.109's
discriminator was built to separate — it argues against "freed under a live
object" and for a hand-out with **no intervening free at all**, i.e. the
allocator considering a slot available while its first occupant is live and
was never released.

**Caveats, both load-bearing.**  (1) Two hits, four probed words of which two
are uninformative — this is a direction, not a rate.  (2) The negative rests
entirely on W1; if any allocator path writes that word on free (a freelist
link, a size field), the absence of poison there would prove nothing.  Worth
confirming against `deallocate_pooled_or_free`'s actual write set before this
is leaned on — that is an allocator-side read, and the one thing that would
make this result solid rather than suggestive.

### 13.112 Bisect the other way: `-O2` baseline, licence the clone on ONE function

Every localisation so far has **subtracted** — `-fno-ipa-cp-clone` (0/167),
`noclone` ×5, `noipa` — and §5 records why that direction is weak: `noclone` on
one function moved **72 other functions'** sizes, so a disappearance cannot be
attributed to the function the attribute was on.

The **additive** direction is available, and §6's own table is what makes it
work: the minimal pair is not an `-O` level at all.

| §6 | |
|---|---|
| plain `-O2` | **0/8** |
| **`-O2 -fipa-cp-clone`** | **19/35** |

So build at `-O2`, hand the clone licence to exactly one allocator function, and
an arm that **fires** names that function *positively*, from a clean baseline
instead of a perturbed one.

**Mechanism** (`allocator_prv.h`, arms attached in `allocator.cpp`): arm
selection is an integer, because a function attribute cannot be chosen by
comparing strings — `-DKAME_CLONE_ARM=6` expands
`__attribute__((optimize("-fipa-cp-clone")))` at slot 6 only.  The default build
defines nothing and every slot expands empty, so it is byte-identical to today's.

| arm | function | why it is on the list |
|---|---|---|
| 1 | `bucket_release_chunk` | 3 clones at `-O3`; survives `noclone` ×5 (§5) |
| 2 | `find_training_zeros` | 2 clones, same |
| 3 | `batch_return_to_bitmap` (FS=false) | the only **deferred** returner; §6's cross-thread batch `cap = 1` is **0/16** |
| 4 | `batch_return_to_bitmap` (generic) | same, other overload |
| 5 | `deallocate_chunk` | returns a whole chunk |
| 6 | `claim_chunk` | publishes `back_offset` — §13.x's stale-read site |
| 7 | `orphan_chain_pop` | adoption |

Arms 3/4 and 6 map directly onto two rows of §13.109's discriminator table, so a
hit there fixes the **function and the recycle path at once**.

**Runner**: `kamepoolalloc/tests/clone_arm_bisect.sh`.  It first `nm`s an
`-O2 -fipa-cp-clone` object and prints the **measured** global clone set —
worth doing because NC7's full membership was never recorded here (§13.x asked
for it twice), so the arm list can be checked against fact rather than memory.

**Two caveats, and an arm that ignores them means nothing.**

1. **A mismatched optimisation context blocks inlining across it** — which is
   why gcc documents `optimize` as a debugging aid.  An arm therefore perturbs
   more than "one extra pass": it also pins a call boundary `-O2` would have
   inlined away.  A firing arm is evidence that *that function's codegen* is
   load-bearing; it is not proof that the clone specifically is.  Weaker than it
   looks — but still a positive localisation, which the subtractive direction
   never produced.
2. **`-O2` may refuse the clone even when licensed.**  IPA-CP's profitability
   thresholds differ by `-O` level, so the runner **verifies** each arm produced
   a `.constprop` symbol and reports an arm with none as **VACUOUS, not
   negative** — §13.61's rule applied to a compiler flag.

**Mac-side validation** (gcc cannot build kamepoolalloc on macOS, `cdb70d2cf`):
all seven arms **pass a syntax check** under clang (which parses and ignores
`optimize`), so the attribute placement is valid at every site and Ubuntu will
not lose a cycle to a build error; and the default arm-0 build is unchanged —
same reproducer, `HITS 0`, `dtor == born`, `enforced 96 555 426`.

**Order to run them in — revised by §13.110's clustering: 5, 7, 6, then 3/4,
then 1/2.**  §13.110 shows two of three multi-hit runs putting both hits in one
chunk, one pair **224 bytes apart**, which is what *chunk-level* reuse looks
like — so the chunk-scoped arms come first: `deallocate_chunk` (5) and
`orphan_chain_pop` (7) before `claim_chunk` (6), and the slot-scoped
`batch_return_to_bitmap` (3/4) after.  §13.111's negative points the same way:
no poison at the hit argues for a hand-out with **no intervening free**, and a
whole chunk being handed back out is exactly that with no per-slot free
involved.

### 13.112a Correction to §13.111: `W0` is NOT uninformative any more — §13.109 moved the store

§13.111 explains its `W0` reading with

```cpp
atomic_countable() noexcept : refcnt(1) { KAME_RC_EVT(this, OP_BORN, 1); }
```

and concludes W0 is uninformative by construction because `refcnt = 1` lands
before BORN.  **That was true, and §13.109 changed it.**  Under `KAME_RC_TRACE`
the constructor now initialises `refcnt` in the **body**, after
`preborn_note()` has read word 0 into a thread-local — precisely so that the
poison survives long enough to be seen, and precisely because §13.104's hit was
an offset-0 address where W0 is the only word that can carry it.

The captures in §13.111 show `RC-DLIVE-WPRE … w=0x1`, i.e. the pre-store word
already holding the new occupant's refcount — which cannot happen with the
§13.109 constructor, so **that binary predates the change** (only one copy of
`atomic_smart_ptr.h` exists in the tree, and HEAD carries the new ctor).  Mac
evidence that the mechanism does work, pool active:

```
RC-DLIVE-WPRE 0x110100120 w=0xbaad0000000b8000  <-- POISON TAG
RC-DLIVE-FREEREC w-1 freed_ptr=0x110100120 (THIS block) size=32 free_tid=1 …
```

That also disposes of a worry worth naming: if `atomic<Refcnt>`'s default
constructor wrote word 0 before the body, the hook would silently read zero
instead of the poison.  It does not — the token is there.

**So §13.111's conclusion should be re-taken on a rebuilt binary.**  Its
direction ("no intervening free") may well hold — §13.110 independently points
the same way — but as it stands the negative rests on W1 alone, and §13.111 is
right that W1 proves nothing until `deallocate_pooled_or_free`'s write set is
checked.  With the §13.109 constructor, W0 becomes the word the argument should
rest on, and no such audit is needed for it: the poison writer targets word 0
first (`q[0] = token`).

### 13.113 Re-measured with the §13.109 constructor: `W0` reads a LIVE refcount, not poison — the block was never freed

§13.112a is right, and my §13.111 was reasoning from the pre-§13.109
constructor: the store moved into the body, `preborn_note(this)` reads word 0
**before** `refcnt.store(1)` lands, so W0 is the pre-store content and is
informative.  §13.112a also diagnosed the binary correctly — I rebuilt and
confirmed the new hook is present (`preborn` symbol in the executable) before
re-running.

**Re-measured, firing configuration, two failing runs (7 and 3 hits):**

```
RC-DLIVE-WPRE 0x7fffd4bb4230 w=0x1     RC-DLIVE-WPRE 0x7ffff526f4e0 w=0x1
RC-DLIVE-W0   0x7fffd4bb4230 w=0x1     RC-DLIVE-W0   0x7ffff526f4e0 w=0x1
RC-DLIVE-W1   0x7fffd4bb4230 w=0x7ffff40c0000   RC-DLIVE-W1 … w=0x7fffb70f0b70
RC-DLIVE-WPRE 0x7fffd4bb4320 w=0x1
RC-DLIVE-W0   0x7fffd4bb4320 w=0x1
RC-DLIVE-W1   0x7fffd4bb4320 w=0x0
```

**W0 = 1 on every hit, and `RC-DLIVE-FREEREC` never fires.**  With the
§13.109 constructor that is a real reading: word 0 held **1 — a live
`atomic_countable::refcnt` — immediately before the incoming object's store**.
Not the forensic poison, and not a free record.

**So the doubly-handed-out block was never freed.**  It was handed out while
its previous occupant was alive at refcount 1.  That is §13.103's reading (a),
an allocator double hand-out, and it is now resting on the word §13.112a says
it should rest on rather than on W1 alone — §13.111's stated weakness is
removed.

**One divergence from the Mac worth flagging.**  arm64 saw
`WPRE = 0xbaad0000000b8000`, decoding to a full free record; every Ubuntu probe
here reads `WPRE = 1`.  A constant 1 in the word *preceding* the block is
itself consistent with the neighbouring slot holding a live object — i.e. these
hits sit inside a region of live objects, which is what §13.110's chunk
clustering already suggested.  The two machines may simply be catching
different situations (theirs a freed block, mine a live-neighbour one), but the
difference should be resolved rather than averaged: **the same probe reads
poison there and live refcounts here.**

**Standing.**  Ubuntu evidence now says: slot handed out with its previous
occupant live and unfreed, hits clustered within a chunk, on every failing run
and no clean one.  That is a coherent picture of chunk-level reuse of live
storage — and it is an allocator bookkeeping question, not an STM one.

### 13.114 The Mac/Ubuntu divergence §13.113 asks about is not a divergence — the Mac readings were the injected control

§13.113 flags that the same probe reads `WPRE = 0xbaad…` (a full free record) on
arm64 and `WPRE = 1` (a live refcount) on Ubuntu, and rightly says the
difference should be resolved rather than averaged.  It resolves immediately:

**arm64 has never produced a real DOUBLE-LIVE hit.**  Every Mac reading quoted
in §13.109 came from `KAME_RC_TRACE_DLIVE_INJECT`, which claims an address a
*second* time on the enforced path — and the enforced path requires a prior
traced death, so the injector's victim is by construction a block that was
**legitimately freed and reused**.  A freed block carries the poison, so
`WPRE = 0xbaad…` is the correct and expected reading *for the control*.  It was
presented as end-to-end validation of the decode path, which is what it is; it
is not a claim about arm64 behaviour, and §13.109's text should have said which
kind of hit produced it.

So the two machines are not disagreeing — they are exercising **the two
different branches of the discriminator**, and both branches read what they
should:

| | hit source | `WPRE` | verdict |
|---|---|---|---|
| arm64 | injected control (block was freed, then reused) | poison token, decodes to a free record | "THIS block — freed" |
| Ubuntu | real hits, failing runs | `1` — a live `refcnt` | never freed |

That is a stronger result than either alone: the probe **demonstrably
distinguishes** the freed case from the never-freed case, with the freed case
shown on a control and the never-freed case on real data.  §13.113's conclusion
therefore stands on a discriminator whose positive branch is verified, not
merely assumed.

**And `W0 = 1` says a little more than "not freed".**  For an offset-0 victim,
word 0 *is* the previous occupant's `refcnt`, so the value is that occupant's
reference count at the moment the pool handed its storage away: **1** — a single
live reference, not a large fan-out.  Combined with §13.110's clustering, the
picture is a chunk of singly-referenced live objects being recycled underneath
their owners.

**Small addition for the next capture** (`RC-DLIVE-PREV`): the previous
occupant's constructor site was already being recorded — `dl_born_` returns it
and `anomaly()` prints it in the `slot=` field — but unlabelled there it reads
as something else.  It now gets its own line, so a hit names the allocation
site that still owns the storage.  That is the one piece of provenance a hit
does not currently make obvious, and it should say which size class and which
STM path the trampled object came from.

**Next remains §13.112's arm bisect, chunk-scoped arms first (5, 7, 6).**  Both
independent Ubuntu results — chunk clustering, and never-freed storage — point
at a chunk being handed back out rather than a slot, and arms 5 and 7 are the
two chunk-level returners.

### 13.115 The previous occupants are ordinary live STM objects

§13.114's `RC-DLIVE-PREV` run in the firing configuration.  Two failing runs,
4 and 3 hits, one `PREV` record each (the previous occupant's ctor site is only
retained when its birth is still in the ring).

| capture | previous occupant's ctor site resolves to |
|---|---|
| 1 | `reset_unsafe<Packet>` ← `local_shared_ptr<Packet>(Packet*)` ← **`make_local_shared<Packet>(Packet&)`** — a `Packet` clone |
| 2 | `lsp<PacketWrapper>::swap` ← `operator=(&&)` ← **`Node::release`** (`transaction_impl.h:1575`) |

**Nothing exotic is being recycled.**  The still-live occupants whose storage
the pool gave away are an ordinary `Packet` clone and an ordinary
`PacketWrapper` produced in the node-release path — the two commonest objects
in this workload.  Combined with §13.113 (`W0 = 1`: the occupant held exactly
one live reference) and §13.110 (hits cluster inside one chunk), the Ubuntu
picture is: **a chunk of singly-referenced, live, ordinary STM objects is
handed back out underneath its owners.**

**Caveat on the second site.**  A "ctor site" resolving to `swap` ←
`operator=(&&)` is the line-table smear this section has discounted four times
(§13.75, §13.79, §13.82, §13.104); the *function* (`Node::release`) is the
trustworthy part, the exact frame is not.  The first site is a clean chain and
needs no such qualification.

**What is still missing is the recycle path itself.**  §13.110 showed the
global `RC-POOLEV` tail cannot supply it — in 4 of 5 captures no pool event was
within 1 MiB of the hit, the relevant chunk event having scrolled off.  The
fix is per-chunk rather than global: record, for each chunk, the last pool
event that touched it, and print that at the hit.  That is the one question
between the current evidence and a named allocator defect, and it is what I am
building next.

### 13.116 Reading the chunk paths: four mechanisms cleared, and the invariant that clears them selects the remaining one

While §13.115's per-chunk event probe is being built, the same question read
from the source.  §13.110/§13.113/§13.115 say: a chunk's worth of live,
singly-referenced, ordinary STM objects is handed back out, and **the victims
were never freed**.  So which pool path can hand out storage that is in use?

**Four candidates, all cleared by one invariant.**

| candidate | why it cannot do it |
|---|---|
| the two claim loops (word-grab, N-run) | one `atomicLoadAcquire` feeds BOTH `mask`/`one` and the CAS expected value, so a successful CAS proves the word went exactly from `oldv` — the claimed bits ARE `mask` (§13.107) |
| word-cache cells `m_freelist_head[1]/[2]` | touched only by claim, the lean-cold consume, and the owner-exit drain — and that drain walks **this thread's own** DLL, so it cannot race the consumer.  Under `WORDCACHE` the free side never touches the cells (the FIFO/STASH park that does is `#error`-exclusive with it) |
| release of a chunk holding undistributed mask bits | `MASK_CNT` counts **non-empty bitmap words**, not slots (`allocator_prv.h`: "MASK_CNT = count (all words non-empty)").  The word-grab leaves the word **all-ones**, so undistributed bits keep it non-empty and `MASK_CNT ≥ 1` — the chunk is unreleasable while they exist |
| the deferred cross-thread batch | same invariant: a batched slot's bit is **still set**, so its word is non-empty and its chunk is pinned for the whole accumulation window.  `push_direct` also routes each free to direct **or** hold, never both, so one free cannot produce two clears |

**The invariant is: a slot's bit stays set until its own free is applied, and
that pins its chunk.**  Every path above is built on it, which is why they are
safe — and it means the defect must be something that **breaks** it.  There are
exactly two ways:

1. **one slot's bit cleared twice** — the first clear frees the slot, a new
   object takes it, the second clear lands under the new occupant;
2. **a bit cleared for a slot that was never freed** — some *other* block's
   free clears it.

**§13.113 selects (2).**  Its `W0 = 1` says the victim was never freed, which
(1) does not produce: in (1) the victim IS a slot whose free was applied, so its
word 0 would carry the poison.  (2) is §13.x's original mis-derived
`chunk_base`: a free that resolves to the **wrong chunk** clears a bit there, so
a live object in that chunk reads as free.  Several frees mis-resolving to the
same chunk give §13.110's clustering; the victims are whatever ordinary live
objects sit in it, at whatever refcount they happen to hold — §13.115 and
§13.113 exactly.

It also explains something (1) cannot: why `cap = 1` is **0/16** and the 2 M
quarantine **0/41** while default batching fires.  Under (2) the batch is not
the defect but the *delay*: it widens the interval between the mis-resolving
free and the bit-clear, and both extremes remove the interval — `cap = 1`
applies the clear immediately, quarantine keeps the wrongly-freed slot out of
circulation until nothing depends on it.

**The reader is `resolve_chunk_from_slot`** (`allocator.cpp:3637`) — the one
function that turns a slot address into a chunk via
`rmeta->back_offset[unit_idx]`, on **every** free.  It was missing from
§13.112's arm list; it is now **arm 8, and first in the suggested order**
(`8 5 7 6 3 4 1 2`).  One honest qualification: it is `static inline`, so
licensing it also forces an out-of-line copy at `-O2` — a larger perturbation
than the other arms, so §13.112's caveat 1 applies to it more strongly, not
less.

**What would confirm (2) directly**, and it is cheap on the allocator side where
§13.115's probe is already going: at a DOUBLE-LIVE hit, report the victim's
chunk base **and** whether any recent free resolved to that chunk from a slot
address outside it.  A single such record names the defect outright.
### 13.117 The additive clone bisect, run: no single-function arm fires — and five of seven arms cannot be tested

§13.112's positive bisect run on Ubuntu.  Reproducer is `tmin_rr` at
`20 40 700`, `taskset -c 0-3`, arms **interleaved round-robin** in one job.
The runner script compiles without `-fPIC`, so its object cannot be linked into
the shared library the reproducer needs; I ran an equivalent loop that builds
each arm as a PIC `.so` with otherwise identical flags, keeping its
clone-count / VACUOUS check.

**The measured global clone set at `-O2 -fipa-cp-clone`** (§13.112 asked for
this and it had never been recorded): 29 `constprop` symbols — 17 of the 27
in the shared build are `PoolAllocator<N,true,true>::PoolAllocator` (one per
size class), plus `find_training_zeros` ×2 and `CrossDeallocBatch::flush`.

**Arm viability, checked before running any of them:**

| arm | function | clones (baseline 3) | vs baseline bytes |
|---|---|---|---|
| 1 | `bucket_release_chunk` | 5 | differs |
| 2 | `find_training_zeros` | 5 | differs |
| 3, 4 | `batch_return_to_bitmap` | 3 | differs, **no new clone** |
| 5 | `deallocate_chunk` | 3 | differs, **no new clone** |
| 6 | `claim_chunk` | 3 | **BYTE-IDENTICAL** |
| 7 | `orphan_chain_pop` | 3 | **BYTE-IDENTICAL** |

**Arms 6 and 7 are byte-identical to the baseline `.so`** — the licence expands
to nothing `-O2` will act on, so those arms test *nothing*; a 0 from them is
not a negative result.  That matters because **arm 6 (`claim_chunk`) is the one
§13.112 singled out** as mapping onto the discriminator rows.  Arms 3/4/5 add
no clone either, so on the runner's own VACUOUS criterion only **arms 1 and 2**
are real experiments.

**Results, 8 interleaved rounds:**

| arm | failures |
|---|---|
| baseline plain `-O2` | **0/8** |
| arm 1 `bucket_release_chunk` | **0/8** |
| arm 2 `find_training_zeros` | **0/8** |
| **global `-O2 -fipa-cp-clone`** | **3/8** |

**The positive control fires and the arms do not.**  Baseline 0/8 reproduces
§6's `-O2` row, and the global licence reproducing at 3/8 in the same job shows
the experiment is powered — so arms 1 and 2 returning 0/8 are real negatives
for those two functions, not a dead setup.

**What this adds.**  Licensing either function that actually gains clones under
`-O2` does not reproduce the fault, while licensing everything does.  That is
the additive counterpart of §13.53 (removing the one perfectly-correlating
clone changed nothing) and of §13.55's graded dose-response, and it points the
same way: **the effect is distributed across the clone set, not carried by one
function.**

**What it cannot say.**  Five of seven candidates were never tested — two
because the licence produced identical code, three because it produced no new
clone.  Reaching `claim_chunk` and `orphan_chain_pop` needs a mechanism that
forces cloning rather than merely permitting it (e.g. licensing at `-O3` and
subtracting elsewhere, or `-fipa-cp-clone` with per-function
`optimize("-O3")`), and until then §13.112's most interesting arm is untested
rather than refuted.

### 13.118 The arm list omits the family that carries the clone-set delta — armed it, and it still does not fire

**The observation first.**  Comparing the clone families of the two builds
whose failure rates differ (§13.117), the delta is not in any armed function:

| family | baseline `-O2` (0/8) | firing `-O2 -fipa-cp-clone` (3/8) |
|---|---|---|
| **`PoolAllocator<N,true,true>::PoolAllocator`** | 0 | **17** |
| **`PoolAllocatorBase::restamp_back_offset`** | 0 | **1** |
| `find_training_zeros` | 0 | 1 |
| `bucket_release_chunk` | 0 | 1 |
| unnamed (static/local) | 3 | 5 |

Arms 1–8 cover the three *single-clone* families.  **The 17 chunk
constructors — one per size class, and the bulk of the difference — were never
on the list**, nor was `restamp_back_offset`.  Given §13.110/§13.113 (a chunk
of live objects recycled underneath its owners), the chunk constructor is also
the mechanistically obvious candidate: it is what lays down a chunk's bitmap,
counts and `back_offset`.

**So I armed them** (slots 9 and 10, same macro mechanism).  Both are
substantial, unlike arms 3–7:

| arm | function | clones (baseline 3) |
|---|---|---|
| 9 | `PoolAllocator<N,…>::PoolAllocator` | **25** |
| 10 | `restamp_back_offset` | 6 |

Arm 9 alone reproduces **25 of the global build's 27 clones**.

**Result, 10 interleaved rounds:**

| arm | failures |
|---|---|
| baseline plain `-O2` | 0/10 |
| **arm 9** `PoolAllocator` ctor (25 clones) | **0/10** |
| **arm 10** `restamp_back_offset` | **0/10** |
| **global** `-O2 -fipa-cp-clone` (27 clones) | **4/10** |

Fisher, global vs arm 9: **p = 0.082**.

**This is the sharpest form the distributed result has taken.**  Arm 9 carries
92% of the global build's clones and fires **zero times in ten runs**, while
the global build fires four.  So it is not that some single unarmed function
was hiding the effect — reproducing almost the entire clone set is *still not
enough*.  Whatever the pass does, it needs the last two clones, or it needs
them together, or the effect is not "which functions are cloned" at all but a
whole-TU consequence of running the pass across the unit.

**Caveats.**  Ten rounds, and p = 0.082 is not significant on its own; the
result is that arm 9 shows **no hint** (0/10, not 1/10) while carrying nearly
all the clones.  And `optimize()`-based licensing is itself a codegen
perturbation — §5's warning that `noclone` on one function moved 72 others
applies in this direction too, so "25 clones" is a count, not proof that those
25 bodies are identical to the global build's.

### 13.119 The bisect, done properly: licensing EVERY cloned function reproduces the clone set and still does not fire

§13.118 could only test one function at a time, so it could not answer "does
some *subset* carry it".  Added `KAME_CLONE_MASK`, a bitmask over the same
slots (bit n-1 → arm n), plus slots 11/12 for the two families that were
cloned globally but had no arm (`CrossDeallocBatch::flush`, `global_pop_fit`).

**Two instrument errors on the way, both caught before they were reported.**
(1) My first mask block was inserted *before* the `KAME_CLONE_ARM == n` blocks,
which then redefined the macros to empty — arms 1/2 silently expanded to
nothing while appearing to be armed.  (2) Moving the block corrupted a line
(`#endif#define …`), so the mask builds **failed**, and because I had sent
`stderr` to `/dev/null` the loop read **stale `.so` files from earlier builds**
and reported plausible clone counts for binaries that did not exist.  Both were
found by preprocessing the macro and by checking the `.so` actually existed.
Sending compiler errors to `/dev/null` in a build-and-measure loop is how a
measurement silently becomes fiction.

**With the mechanism fixed**, per-function licensing composes as expected:
`0x000` → 3 clones, `0x001` → 5, `0x100` → 25, **`0x101` → 27, matching the
global build's 27**, and `0xF03` (all six families) → 33.  `0xF03` covers
**every clone family the global build has** (set difference empty).

**Result, 10 interleaved rounds:**

| arm | clones | failures |
|---|---|---|
| baseline plain `-O2` | 3 | 0/10 |
| A — chunk ctor + `restamp_back_offset` | 28 | 0/10 |
| B — the four single-clone families | 8 | 0/10 |
| **FULL — all six families** | **33** | **0/10** |
| **global `-O2 -fipa-cp-clone`** | **27** | **9/10** |

Fisher, global vs FULL: **p = 0.00003**.

**This is a clean, decisive negative, and it closes the whole line.**  A build
that clones *every function the global build clones* — 33 bodies against its 27,
superset coverage by family — **never fails in ten runs**, while the global flag
fails nine times out of ten in the same interleaved job.  So the fault is **not
carried by the clone set at all**: not by one function (§13.53, §13.118), not by
any half, and not by the union.

**What that leaves.**  `-fipa-cp-clone` as a whole-TU pass does something beyond
producing those clone bodies — it re-runs IPA-CP's propagation and the
inlining/ordering decisions across the entire unit, and `optimize()` on
individual functions does not.  The distinguishing property is therefore a
**whole-TU codegen consequence**, which is what §13.55's graded dose-response
and §13.88's wide window have been saying in other language.  "Which function"
is the wrong question; the next one has to be "what does the pass change about
the unit", e.g. a diff of the two objects restricted to functions that are *not*
clones.

### 13.120 Correction to §13.119: the clone SYMBOLS matched, the pass's DECISIONS did not — and the gap is `*_pop_fit` + `acquire_tag_ref_`

A web survey of `-fipa-cp-clone` prompted the check that undoes §13.119's
conclusion.  Two documented facts about the mechanism §13.112–§13.119 relies on:
`__attribute__((optimize(...)))` **replaces** the command-line optimisation
flags rather than adding to them, and GCC's own documentation says the
attribute *"should be used for debugging purposes only. It is not suitable in
production code."*  That is a warning that per-function licensing is not
equivalent to the flag — so I measured what the pass actually did, with
`-fdump-ipa-cp-details`, instead of counting surviving symbols.

**The two builds are not equivalent, and §13.119 measured the wrong thing:**

| | specialized nodes created | distinct functions specialized |
|---|---|---|
| global `-fipa-cp-clone` | **73** | **12** |
| FULL `KAME_CLONE_MASK=0xF03` | **46** | **8** |

§13.119 reported FULL as having *superset* coverage (33 clone symbols vs 27).
That was surviving-symbol count, which is not what the pass did — many
specializations are inlined or merged away.  By decisions, **FULL is a strict
subset of GLOBAL**, missing four functions entirely:

```
char* {anonymous}::global_pop_fit
char* {anonymous}::l1_pop_fit
char* {anonymous}::recycle_pop_fit
atomic_shared_ptr_base<...>::acquire_tag_ref_
```

**So §13.119's "decisive negative" is withdrawn.**  FULL firing 0/10 does not
show that the clone set fails to carry the fault; it shows that per-function
licensing **cannot reproduce the clone set**, because IPA-CP's decision to
specialize these four depends on whole-unit propagation from their callers,
which an attribute on the callee cannot supply.  The 0/10 is explained by the
missing four, not by the 33 that were present.

**And the missing four are the interesting ones.**  Three are the `*_pop_fit`
family, of which `global_pop_fit` is the clone §13.53 found **perfectly
correlated with the fault across all six arms** (present in both FAULT arms,
absent from all four suppressors) and then could not confirm causally — because
§13.53's test was `noclone`, i.e. subtractive, and §13.112's additive direction
was supposed to fix exactly that.  It now turns out the additive direction
**cannot reach them at all**.  The fourth,
`atomic_shared_ptr_base::acquire_tag_ref_`, is the tagged-pointer reference
acquisition in `atomic_smart_ptr.h` — the refcount machinery whose corruption
this entire section has been chasing.

**Standing.**  The arm mechanism is sound for functions the pass will specialize
from a local licence, and useless for functions whose specialization is driven
by caller context — which is the set that matters.  Reaching them needs the
pass enabled unit-wide with cloning *suppressed elsewhere* (subtractive from a
firing baseline), not licensed locally from a quiet one.

**Sources**: [GCC Common Function Attributes](https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html),
[GCC Optimize Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html),
[cython#2494 "Should not use __attribute__((optimize(…)))"](https://github.com/cython/cython/issues/2494),
[GCC bug 66616 — fipa-cp-clone ignores thunk](https://gcc.gnu.org/pipermail/gcc-bugs/2015-December/526767.html),
[GCC internals: Regular IPA passes](https://gcc.gnu.org/onlinedocs/gccint/Regular-IPA-passes.html).

### 13.121 The subtractive mechanism §13.120 asks for: `noclone` from a FIRING baseline, reaching the four caller-driven clones

§13.120's conclusion is that per-function licensing is structurally unable to
reach the four functions the pass specialises from caller context — and that
those four are the interesting set (`global_pop_fit` is §13.53's clone that
correlated perfectly with the fault across all six arms; `acquire_tag_ref_` is
the tagged-pointer refcount acquisition this whole section is about).  It also
says what to do instead: enable the pass unit-wide and **suppress cloning
elsewhere**, i.e. subtract from a build that fires rather than add to one that
is quiet.

Built that.  `KAME_NOCLONE_MASK` is the mirror of `KAME_CLONE_MASK` — a bitmask
over `noclone` attributes on the three `*_pop_fit` functions, applied on top of
a global `-O2 -fipa-cp-clone`:

```
g++ -O2 -fipa-cp-clone -DKAME_NOCLONE_MASK=0x2   # global_pop_fit only
g++ -O2 -fipa-cp-clone -DKAME_NOCLONE_MASK=0x7   # all three *_pop_fit
g++ -O2 -fipa-cp-clone -DKAME_ASP_NOCLONE        # the ASP protocol trio
```

`acquire_tag_ref_` is deliberately **not** a new slot: it already carries
`KAME_ASP_NOCLONE_ATTR` in `atomic_smart_ptr.h`, so `-DKAME_ASP_NOCLONE` is its
switch — with the caveat already written there, that it moves the other two
protocol members with it and wants the lock-add census checked first (§13.23).

**Why `noclone` reaches what `optimize()` cannot.**  IPA-CP's decision to
specialise a callee is taken with the callee's own `noclone` in hand — it
removes the licence — whereas asking for a specialisation the caller context
has to justify cannot be expressed on the callee at all.  So this direction
covers exactly the set the additive one structurally missed.  §5's objection
still stands and is handled differently: it warns against attributing a
*disappearance* to the function the attribute was on, so the runner reports the
**ipa-cp decision delta** (`-fdump-ipa-cp-details`, "Creating a specialized node
of ...") rather than surviving `.constprop` symbols — §13.120's lesson exactly,
since symbol counts said "superset" where decisions said "strict subset".

**Runner**: `kamepoolalloc/tests/noclone_mask_bisect.sh`.  Per arm it prints the
specialised-node and distinct-function counts plus the per-function histogram,
so an arm that did **not** remove its target from the list is reported as
suppressing nothing — VACUOUS, not negative, the same discipline §13.112 applied
to the additive arms and §13.117 then needed for five of seven.  It also never
sends compiler errors to `/dev/null` and refuses to proceed on a missing `.so`,
which is the §13.119 failure mode (stale artefacts read as fresh results).

**The three experiments this makes available, in order of what a result would
settle:**

| arm | if the firing build STOPS firing | if it keeps firing |
|---|---|---|
| `0x2` (`global_pop_fit`) | §13.53's perfect correlation becomes causal — the single most direct outcome available | that clone is not load-bearing alone |
| `0x7` (all three `*_pop_fit`) | the family carries it; bisect within by `0x1`/`0x2`/`0x4` | the family is not it, and `acquire_tag_ref_` is the remaining unreached candidate |
| `KAME_ASP_NOCLONE` | the refcount-protocol specialisation carries it | neither of the two reachable sets does, and the answer is a whole-unit property after all |

Note the asymmetry that makes this worth running before anything else: unlike
§13.117–§13.119's arms, **every arm here starts from a configuration that is
known to fail 9/10**, so a negative arm is a real negative — the experiment
cannot be silently unpowered.  What it can be is vacuous, and the decision-delta
report is there to catch precisely that.

**Mac-side validation**: masks `0`, `0x1`, `0x2`, `0x4`, `0x7` all pass a clang
syntax check (clang parses and ignores `noclone`), so the attribute placement is
valid at all three sites; and mask 0 leaves the default build unchanged.  The
runs need gcc, so they are Ubuntu's.
### 13.122 The gap is filled and fully accounted — and neither half of it carries the fault

§13.120 left a measured gap: the firing build makes **73** specializations
across **12** functions, the additively-licensed build **46** across **8**, and
the four unreachable functions were `l1_/global_/recycle_pop_fit` and
`acquire_tag_ref_`.  §13.120 also said the only direction that can reach them
is **subtractive from a firing baseline**.  Built that, and — the point §5's
objection turned on — **verified every suppression against
`-fdump-ipa-cp-details` instead of assuming it**.

**The gap accounts exactly:**

| | specializations |
|---|---|
| global `-fipa-cp-clone` | **73** |
| additive FULL (§13.119) | 46 |
| **difference** | **27** |
| `*_pop_fit` ×3 (measured) | **5** |
| `acquire_tag_ref_` (measured) | **22** |
| 5 + 22 | **27** ✓ |

So the whole difference between "licensed per function" and "the flag" is those
two families, and `acquire_tag_ref_` alone is **22 of the 73 specializations —
30% of everything the pass does in this unit**, in the tagged-pointer reference
acquisition that this section has spent forty sections watching corrupt.

**Both suppressions are surgical, and measured to be:**

| build | nodes | target specializations |
|---|---|---|
| firing baseline | 73 | pop_fit 5, `acquire_tag_ref_` 22 |
| `KAME_NOCLONE_MASK=7` | 68 | **pop_fit 0** |
| `-DKAME_ASP_NOCLONE` | 51 | **`acquire_tag_ref_` 0** |

73−5 = 68 and 73−22 = 51: each removes exactly its target and nothing else.
That is the check §5 said could not be made, and it can.

**Causal results — neither fires the discriminator:**

| experiment | firing arm | suppressed arm | p |
|---|---|---|---|
| `*_pop_fit` ×3 (pooled, 2 batches) | 12/22 (55%) | 12/21 (57%) | **1.000** |
| `acquire_tag_ref_` (22 nodes, 16 rounds) | 11/16 (69%) | 7/16 (44%) | **0.285** |

**So the gap is closed and empty.**  `global_pop_fit` — §13.53's clone that
correlated perfectly across all six arms — does not carry it, now tested in the
direction §13.112 was built to provide and §13.120 showed was the only one that
could reach it.  Neither does `acquire_tag_ref_`, despite being 30% of the
pass's work and sitting in the exact machinery that fails.

**What that establishes, and it is worth stating plainly.**  The difference
between the firing and non-firing builds is now **fully enumerated** — 27
specializations, in two families — and **removing either changes nothing**.
Combined with §13.119 (adding them back, by the only means available, also
changes nothing), the specialization *set* is exhausted as an explanation in
both directions.  Whatever `-fipa-cp-clone` does to make this fault appear, it
is not the set of functions it specializes.

**The remaining candidates are the pass's side effects**: it re-runs IPA-CP
propagation over the unit, which changes value ranges, alias/escape
conclusions and inline decisions for functions that are **never cloned at
all**.  §13.119's suggested next step is now the only one left standing — diff
the two objects restricted to functions that are *not* clones, which is where
an effect that survives both directions of clone-set manipulation has to live.

### 13.123 The non-clone memory-op diff §13.122 asks for

(The conflict markers `7899e2210` committed in `allocator_prv.h` and
`allocator.cpp` — both sessions built the §13.121 mask independently and the
merge went in unresolved — were fixed on the Ubuntu side in `8382836c3`, taking
the §13.121 files verbatim.  Confirmed here: `KAME_NOCLONE_MASK` ∈ {0, 2, 7}
compiles again on that tree.  Worth one line of standing practice anyway: the
allocator did not compile *at all* on the tip for several commits, and a
build-and-measure loop that does not check its object exists reads stale
artefacts as results — the same trap §13.119 documented.)

**Then the tool.**  §13.122 leaves exactly one line open: the pass's effect on
functions it never clones, "diff the two objects restricted to functions that
are not clones".  The existing runner reports *sizes* (§5: "34 function sizes
differ"), which is true and not actionable.
`kamepoolalloc/tests/nonclone_memop_diff.py` ranks by **memory-operation shape**
instead, because the fault class picks the metric: §13.113/§13.116 say a bitmap
bit was cleared for a slot whose own free never happened, i.e. some function
computed the wrong address or ran a store twice.  That shows up as an atomic RMW
gained or lost, a store moved into or out of a branch, or a load duplicated so
two reads back one CAS — not as a size delta.

Per non-clone function present in both objects it reports Δinstructions,
Δatomic, Δstore, Δload, Δbranch, sorted atomic-first, and it

- excludes `.constprop` / `.isra` / `.part` symbols (the clone set, already
  exhausted in both directions by §13.119 and §13.122);
- lists non-clone functions present in only **one** object separately — a
  function that stopped being emitted at all is a bigger change than any delta;
- demangles, and **collapses template instantiations** (`×N`): one function
  instantiated per size class otherwise produces 40 identical rows and buries
  everything else.

**Self-tested on a real object pair** (this side cannot run gcc on
kamepoolalloc, `cdb70d2cf`, so the test is clang `-O2` vs `-O3` — the parsing
and ranking are what is being checked, not the finding): 1149 non-clone
functions in both, **114 changed**, collapsed from 436 raw rows by grouping.
Top of that list is `orphan_chain_scrub` (+26 atomic, ×22 instantiations) and
`orphan_chain_pop` (+8 atomic) — which is only a demonstration that the ranking
separates memory-shape changes from noise, not a result about the fault.

Ubuntu runs it as:

```
g++ -O2                 -c ... -o A.o
g++ -O2 -fipa-cp-clone  -c ... -o B.o
kamepoolalloc/tests/nonclone_memop_diff.py A.o B.o --top 25
```

The pair to diff is §13.119's minimal pair — the two builds whose failure rates
are 0/10 and 9/10 — so every row is a candidate for the whole-unit effect that
survived both directions of clone-set manipulation.
### 13.124 Not function pointers — the indirect-call class is closed, and which `acquire_tag_ref_` it is

**The question is a good one because the pool is full of function pointers**:
every chunk header stores a `DeallocateFn` and a `SizeOfFn`
(`ALLOC_CHUNK_HEADER_FN_OFFSET` / `..._SIZEOF_FN_OFFSET`), written at
construction and called on every free.  If IPA-CP specialized those, a chunk
could carry one specialization's address and be freed through another —
which would explain chunk-level mis-recycling exactly.

**It does not.  Two checks, both negative:**

1. `deallocate_pooled_static` and `size_of_static` are **not cloned**: 52
   symbols in the firing object, 52 in the non-firing one, **zero
   `.constprop`** among them.
2. More generally, of the **24 cloned families** in the firing object,
   **none has its address taken** — no `R_X86_64_64`/`PC32` data relocation
   names any clone or its parent.  IPA-CP is leaving address-taken functions
   alone, as it is required to.

With speculative devirtualization already refuted directly
(`-fno-devirtualize-speculatively`, 28/28 vs 27/27, §12-era), **the whole
indirect-call class is closed**: the fault is not a function-pointer identity
problem.  It is an algorithmic consequence of what the specialized bodies
compute.

**Which `acquire_tag_ref_`, precisely.**  All **22** specializations are

```
atomic_shared_ptr<PoolAllocator<N,true[,false]>>::acquire_tag_ref_(Refcnt*, bool)
```

one per size class (16, 32, 48, 64, 80, 112, 144, 176, 208, 240, 256, 272,
288, 304, 320, 336, 352, 368, 1024, 4096, …).  **These are the pool's own
`atomic_shared_ptr`s — the orphan-chunk reclaim chain — not the STM's
`atomic_shared_ptr<PacketWrapper>`.**  Every one propagates the same constant:

```
- considering value 0 for param #2 weakly (caller_count: 3)
  replacing param #2 weakly with const 0
```

i.e. the `weakly` argument specialized to `false`.

**Why that is the most interesting thing found in this stretch.**  The orphan
chain is the machinery by which a chunk orphaned by an exited thread is
adopted and reused — exactly the path §13.110's chunk-clustered hits and
§13.113's "handed out with its previous occupant live and unfreed" implicate.
So the single largest block of the pass's work in this TU (22 of 73
specializations, 30%) sits on the code path the runtime evidence independently
points at, and the constant it propagates is the flag that selects between the
strong and weak acquisition protocols.

**Status of that arm — deliberately not called closed.**  §13.122 measured
11/16 vs 7/16 (p = 0.285) with all 22 suppressed.  That is a 25-point drop
which is *not* significant at n = 16, and given how well-motivated the family
is I am treating it as underpowered rather than negative; a 26-round rerun is
in progress.  Every other candidate in this section was refuted at p ≈ 1.0 or
with the arms tied — this one is the only suppression so far that moved the
rate in the right direction at all.

### 13.125 The 22 `acquire_tag_ref_` specializations are semantically INERT — `weakly` is a dead parameter for the pool's instantiation

§13.124 names the strongest remaining candidate and asks for more rounds.  Read
the source while those run, because what the specialization *is* changes what a
confirmed effect would mean.

**The function is sound under the fold.**  `acquire_tag_ref_(Refcnt*, bool
weakly)` with `weakly` folded to `false` loses only the weak block, which has
no side effect (it fails fast and returns).  What remains is a single
`load_tagged_()` per iteration feeding **both** CAS operands
(`TaggedPtr(pref + rcnt_old)` → `TaggedPtr(pref + rcnt_new)`), and
`load_tagged_()` is `m_ref.load(std::memory_order_relaxed)` — an atomic load, so
it cannot be hoisted out of the loop or CSE'd across iterations however much the
fold shrinks the body.  Same single-load discipline §13.116 verified in the
allocator's claim loops, and it holds here too.

**And for this instantiation the parameter is DEAD.**  `weakly` reaches
`acquire_tag_ref_` from four places in the header — `load_shared_()` (default
`false`), the `scoped_atomic_view` constructor (runtime argument), and
`compareAndSet_impl_<…, WEAK, …>` (template constant, instantiated `true` at
`atomic_smart_ptr.h:2510/2515/2520`).  The pool's orphan chain uses **none** of
the weak entry points: the only `atomic_shared_ptr` operations on
`m_orphan_next` / `s_orphan_chain_head()` in the whole allocator are
`compareAndSwap` (`allocator.cpp:8369`) and `compareAndSet` (`:8389`), both
`WEAK = false`, and it never constructs a `scoped_atomic_view`.  That matches the
dump's `caller_count: 3` with value 0 and no competing value.

So the largest single block of the pass's work in this TU — **22 of 73
specializations, 30 %** — **cannot change behaviour.**  It removes a branch that
was never taken, makes the returned `success` constant `true`, and lets three
call sites inline differently.  Nothing else.

**Which sharpens what §13.124's arm can mean, in both directions:**

- If the 26-round rerun **confirms** the drop, the fault is localised to a
  transformation that is *provably semantically inert* — which is the strongest
  statement available that this is a **codegen-class** problem (a miscompile, or
  a store/load the optimiser is entitled to move only because it can now prove
  something) and not a latent source race.  It would also say exactly where to
  look: not in `acquire_tag_ref_` but in its three callers — `load_shared_`, the
  `scoped_atomic_view` constructor, and `compareAndSet_impl_` — for the
  `PoolAllocator<N>` instantiations.
- If it **does not** confirm, the 11/16 vs 7/16 was the n = 16 noise it looks
  like, and the honest reading is that a semantically inert transformation
  behaved like one.

**The confound to state now rather than after the fact:** a semantically inert
change that moves a failure rate can equally be **timing or layout**
perturbation — the Heisenbug confound this whole hunt has been living with, and
`optimize()`/`noclone` attributes are exactly the kind of change that shifts code
placement.  At n = 16 the two are indistinguishable.  What would separate them is
not more rounds of the same arm but a *placement control*: suppress the same 22
and add back an equivalent amount of unrelated code motion (e.g. `noclone` on a
family with the same clone count that §13.122 already measured as inert).  Worth
setting up if the rerun comes back positive.

**Where to point the §13.123 diff if it does.**  Not at the global pair, but at
baseline vs `-DKAME_ASP_NOCLONE` — that pair isolates this one family, so the
non-clone rows are exactly the three callers' code, and the atomic/store columns
say whether a memory operation moved in them.
### 13.126 ORPHAN CHAIN ABLATION: 0/20 against 15/20 — the first suppression that eliminates the fault

**The question — `atomic_shared_ptr` or the pool — has a sharp answer because
they meet in exactly one place.**  The pool's **only** use of
`atomic_shared_ptr` is the orphan-chunk reclaim chain
(`s_orphan_chain_head()`, `m_orphan_next`; `allocator_prv.h:2473/2526`).
Nothing else in the allocator instantiates it.  And §13.124's 22
`acquire_tag_ref_` specializations are all
`atomic_shared_ptr<PoolAllocator<N,…>>` — i.e. that chain.

**So I ablated the chain.**  `KAME_ORPHAN_CHAIN` was retired long ago and the
chain is unconditional, so I added `KAME_NO_ORPHAN_CHAIN`, which skips the
`orphan_chain_push(c)` at `allocator.cpp:3388`.  Chunks are then stranded
rather than adopted — a memory cost, not a correctness change.  The chain code
still compiles and still receives all 22 specializations; only the **runtime
use** is removed.

**Result, 20 interleaved rounds, `20 40 700`, `taskset -c 0-3`:**

| arm | failures |
|---|---|
| firing baseline (`-O2 -fipa-cp-clone`) | **15/20 (75%)** |
| **`-DKAME_NO_ORPHAN_CHAIN`** | **0/20 (0%)** |

Fisher exact **p = 7.7 × 10⁻⁷**.

**This is the first ablation in the entire investigation to eliminate the
fault.**  Every other suppression tied or came back at p ≈ 1.0 — including,
finally, the `acquire_tag_ref_` arm itself, which on 22 rounds settled at
**13/22 vs 13/22, p = 1.000**, confirming §13.124's caution that its earlier
25-point drop was noise.

**It also explains the reproducer's own requirements.**  §1 records that the
minimal test needs *"a PERSISTENT tree, MANY concurrent threads, and REPEATED
thread create/exit"* — and thread exit is precisely what orphans a chunk onto
this chain.  It fits every runtime observation: §13.104's DOUBLE-LIVE (a chunk
handed out while still occupied), §13.110's per-chunk clustering, §13.113's
"previous occupant live and unfreed", and §6's finding that the fault follows
the **allocator's** compiler — the chain is compiled in the allocator's TU.

**The confound, stated because it is real.**  This ablation removes two things
at once: the `atomic_shared_ptr` chain *algorithm*, and a major source of
**chunk reuse**.  A build that strands chunks reuses far less storage, so 0/20
is consistent both with "the chain's adoption logic is wrong" and with "less
reuse means fewer opportunities".  It is a localisation, not yet a mechanism.

**The experiment that separates those**, and the obvious next one: keep the
chain and its reuse, but make adoption **verify** what it takes — e.g. refuse
to adopt a chunk whose occupancy count is non-zero, or re-check it after the
CAS.  If the fault dies with reuse intact, the adoption logic is named; if it
survives, the effect is reuse volume and the chain is merely the busiest
supplier of it.

### 13.127 The orphan chain, read and then modelled: the protocol is correct as written, and the `move` in `orphan_chain_push` is load-bearing

§13.126 localised the fault to the orphan-chunk reclaim chain by ablation
(0/20 vs 15/20).  Two things follow: read the algorithm, and — because reading
has a limit here — model it.

**Read first.  Every mechanism reading can settle is defended:**

| step | why it is not the defect |
|---|---|
| `orphan_chain_push` | **MOVEs** the self-ref into the chain-ref, preserving `refcnt` including a residual scrub pin.  The code says why a `store(1)` would be wrong |
| adopt's `BIT_OWNED` claim | single plain read feeding the CAS expected value; retries on a `MASK_CNT` change |
| adopt takes a NON-empty chunk | by design — empty chunks are released directly and never reach the chain, so a chunk with live slots is exactly what adoption is for |
| word-cache mask | drained (cells nulled AND bits returned) **before** the `BIT_OWNED` clear, unconditionally for orphaned chunks too — no double accounting |
| second releaser | for FS=true the cross-thread free's `atomicDecAndTest` return is **intentionally ignored**, so the scrub is the sole releaser; and FS=true is where `Packet`/`PacketWrapper` live |

**But one coupling survives reading, and it is the fault's shape exactly.**  The
chunk **object** (refcounted) and the chunk **storage** share a lifetime, so a
refcount reaching zero while a slot is live releases the region under live
objects — and the region is then re-carved for another size class.  That is
§13.104's DOUBLE-LIVE, §13.110's per-chunk clustering and §13.113's "previous
occupant live and unfreed", in one sentence.  And that refcount is maintained by
**three reference kinds** (chain-ref, self-ref, scrub pin) over a Treiber stack
whose scrub **unlinks** nodes and whose adopt **revives** them.  Reading does not
settle that.

**So: `kamestm/tests/tlaplus/OrphanChain.tla`.**  Split points only where a race
can occur — scrub `read → CAS`, adopt `pop → claim → move`, push `take → CAS` —
and `atomic_intrusive_dispose` **folded into the 1 → 0 decrement** rather than
modelled as its own action.  Properties: `NoDisposeWithLive` (the fault),
`NoUseAfterDispose`, `OwnedNotOnChain`, `TypeOK`.

**Results:**

| config | states | verdict |
|---|---|---|
| 2 chunks, 2 threads | 5 799 distinct, **queue 0** | **PASS (exhaustive)** |
| **3 chunks**, 2 threads | **160 699 distinct, queue 0**, depth 33 | **PASS (exhaustive)** |
| 2 chunks, **3 threads** | **60 595 distinct, queue 0**, depth 29 | **PASS (exhaustive)** |
| `BUG_STORE1` | 1 283 | `NoDisposeWithLive` **violated** |
| `BUG_SCRUB_STALE` | 2 931 | `NoDisposeWithLive` **violated** |
| `BUG_EXTRA_RELEASER` | **43** | `NoUseAfterDispose` **violated** |

All three knobs bite, so the passes are not vacuous — the §13.61 rule, applied to
a model rather than an instrument.

**And `BUG_STORE1` reproduces the code comment's own reasoning, step for step:**

```
FreshClaim(t1,c1) → Allocate → live=1 → PushTake/PushCas → on chain, refcnt=1
ScrubRead(t1,c1)                      ← residual scrub pin, refcnt=2
PopTake/AdoptClaim/AdoptMove(t2)      ← t2 adopts; hold → self-ref, refcnt=2
PushTake(t2,c1)                       ← re-orphan; store(1) CLOBBERS the pin
ScrubRelease(t1)                      ← pin drop: 1 → 0 → dispose with live=1
```

That is verbatim what `orphan_chain_push` warns about ("a `refcnt.store(1)` here
would CLOBBER that pin's count, so a later pin-drop would dispose the chunk
while it is back on the chain").  **The `move` is now machine-checked as
load-bearing**, which also means it must never be "simplified" back.

**The model caught one of my own errors first, which is the honest way to report
its value.**  The first version tracked no ownership, so it allowed a second
thread to orphan a chunk another thread was mid-adoption on — a violation that
looked like a real finding for one run.  The real code cannot do it:
`release_dll_chunks_for_thread` walks **this thread's own DLL**, so a thread can
only orphan what it owns.  Added `ownerOf` and the violation went away.  A model
that reproduces an artifact before a defect is doing its job; quoting the
artifact would not have been.

**What this does NOT say, stated plainly.**  An exhaustive pass at 3×2 and 2×3
says the *protocol* is correct at that scale, under the abstraction chosen: one
live-slot bit per chunk (`live ∈ 0..1`, i.e. "MASK_CNT zero or not"), a chain
walked one node at a time, and no modelling of the bitmap, the batch, or the
word cache.  It therefore **argues against an algorithmic defect in the chain
protocol** and, combined with §13.126's ablation, points the remaining suspicion
at the *implementation* of that protocol under `-fipa-cp-clone` — which is where
§13.122's whole-unit codegen conclusion already stood.  The two now agree: the
chain is the site, and the algorithm at that site is sound.

**The experiment §13.126 asked for is unchanged and is now better motivated:**
keep the chain and its reuse but make adoption verify what it takes.  The model
says such a check is redundant against the *algorithm*, so if adding it removes
the fault, that is direct evidence the failure is in codegen rather than
protocol — a cleaner discriminator than the ablation, which removed reuse
volume at the same time.

### 13.128 Within the chain, it is not the release path: both dispose backstops never fire, on failing runs too

§13.127 models the chain protocol as correct and names the surviving coupling —
the chunk OBJECT and chunk STORAGE share a lifetime, so a refcount reaching
zero with a live slot would release a region under live objects.  That
coupling has a last line of defence already in the code, and its comment makes
a testable claim.  `atomic_intrusive_dispose` holds two guards:

```cpp
if(p->m_flags_packed & BIT_OWNED) return;   // "never fires under the owner-ref design"
if(p->m_flags_packed & MASK_CNT)  return;   // "At runtime this never fires"
```

**Counted them** (`KAME_POOL_BACKSTOP_CENSUS`, gated — production builds contain
none of it, verified by symbol absence).  Also added a signal-handler readback,
because `atexit` does not run after `SIGSEGV`/`SIGABRT` and a crashing run is
exactly where a backstop would fire — the same gap that hid the park counters
(§13.93) and the DOUBLE-LIVE counters (§13.104), now three times.

**Result, 12 runs — 7 failing, 5 clean:**

| runs | backstop `owned` | backstop `live-slot` |
|---|---|---|
| 7 failing (`rc=139`/`134`) | **0** | **0** |
| 5 clean | never fired | never fired |

**So the release path is exonerated.**  No chunk is disposed while owned, and
none is disposed with live slots — not even on the runs that go on to crash.
The comment's claim holds under the fault, and §13.127's coupling, while real
as a design property, is **not being exercised**: the guards catch it, or it
never arises.

**That halves what is left of the chain.**  §13.126's ablation removed the
whole orphan chain and the fault went 15/20 → 0/20.  The chain has two active
halves — **scrub/dispose** (unlink a drained orphan, release its region) and
**adopt** (pop a chunk and re-own it, handing its free slots back out).  The
release half is now measured clean on failing runs, so within the chain the
remaining candidate is **adopt** — which is also the half that matches the
runtime evidence: §13.104's DOUBLE-LIVE is a slot handed *out*, §13.113 found
the previous occupant *live and unfreed*, and adopt is the only path that
hands out slots from a chunk it did not itself construct.

**Next**: instrument adopt the way dispose was just instrumented — at the
`BIT_OWNED` claim, record the chunk's occupancy and compare it against the
bitmap the adopting thread proceeds to allocate from.  A slot handed out that
the occupancy count says is still in use is the DOUBLE-LIVE event at its
source, one step upstream of where §13.104 catches it.

### 13.129 `OrphanAdopt.tla` — adopt at bitmap granularity: the invariant is `occ ⊆ bits`, and only two things break it

§13.128 exonerated the release half by measurement and left **adopt** as the
only live candidate inside the chain.  §13.127's model cannot speak to it: it
abstracted occupancy to one bit (`MASK_CNT` zero or not) and had no bitmap, so
"can adopt hand out a slot that is occupied" was outside its language.  This
model puts the bitmap in, and it answers the question §13.128 was going to
instrument for.

**Three sets per chunk, which is the whole mechanism:** `bits` (the `m_flags`
bitmap — a set bit means *not available*), `occ` (slots holding a **live**
object), `mask` (the word-cache's claimed-but-undistributed bits, parked in
`m_freelist_head[1]`).  The word-grab claims a whole word in one CAS
(`bits := ALL`) and hands out one slot, parking the rest — so `bits` legitimately
**exceeds** `occ`, and the invariant runs the other way:

> **`occ ⊆ bits`** — the bitmap never says "available" about storage in use.

Its breach **is** the DOUBLE-LIVE event, one step upstream of where §13.104
catches it.  Two supporting invariants: `mask ∩ occ = {}` (a parked bit is never
live) and `handed = {}` (no slot handed out while occupied — the fault stated
directly).

**Results (3 slots, 2 threads, exhaustive):**

| config | states | verdict |
|---|---|---|
| the code as written | 313 distinct, **queue 0** | **PASS** |
| `BUG_NO_DRAIN` — owner exit forgets to drain the mask | 379 distinct, **queue 0** | **PASS** |
| `BUG_DRAIN_KEEPS_CELLS` — returns the bits, leaves the cells | 203 | **`occ ⊆ bits` violated** |
| `BUG_WRONG_BIT` — a free clears a bit other than its own | 42 | **`occ ⊆ bits` violated** |

**`BUG_NO_DRAIN` passing is a finding, not a null.**  Forgetting to drain the
word-cache mask at owner exit is a **leak**, not a correctness bug: the parked
bits are already claimed in the bitmap, so inheriting them and handing them out
later is legitimate.  That is exactly what §13.116 argued by reading, now
machine-checked — and it removes the mask-inheritance story from the candidate
list for good.

**`BUG_DRAIN_KEEPS_CELLS` failing is the sharper half of the same point.**  The
drain must do **both** halves or **neither**: returning the bits to the bitmap
while leaving the cells populated double-counts the slots, so the same storage
goes out once from the bitmap and once from the mask.  The C++ does both (nulls
`[1]`/`[2]`, then returns each bit) — §13.116 verified that by reading, and the
model now says what the cost of getting it backwards would be.  Worth keeping as
a standing regression: this is a two-line invariant that a future
"simplification" of the drain would break silently.

**`BUG_WRONG_BIT` failing ties the model to the runtime evidence.**  §13.116
identified "a bit cleared for a slot whose own free never happened" as the one
surviving mechanism, and §13.113 measured exactly that (`W0 = 1`: the victim was
never freed).  Modelling a free that clears *some* bit rather than *its own*
breaks `occ ⊆ bits` in **42 states** — so the invariant detects the mechanism the
runtime evidence points at, which is what licenses reading the PASS as
meaningful.

**Where this leaves adopt.**  Adopt does not re-verify the bitmap: it claims
`BIT_OWNED` and allocates from whatever `m_flags` says.  The model says that is
**sound provided `occ ⊆ bits` holds on entry** — adopt itself introduces no way
to break it.  So §13.128's proposed instrument (compare occupancy against the
bitmap at the claim) is worth building not as a test of adopt's logic but as a
**detector of an invariant violated before adopt runs**: if it fires, the bit was
already wrong when the chunk was orphaned, and the culprit is on the free side —
`resolve_chunk_from_slot`, arm 8, the one the additive bisect could never reach
and §13.121's subtractive mask now can.

**Limits, as before.**  One chunk, one bitmap word, no addresses — so a
*cross-chunk* mis-derivation is modelled only by its consequence (a wrong bit
cleared), not by its cause.  That is deliberate: the cause needs addresses, and
the consequence is what any instrument can see.

### 13.130 The audit lands on one plain byte array — and a `back_offset` verifier that reproduces the crash signature from a single poked byte

§13.129 left exactly one candidate: a bit cleared for a slot whose own free
never happened, i.e. a free that mis-derived `chunk_base`.  Audited that
derivation end to end.

**What it is.**  `resolve_chunk_from_slot` (`allocator.cpp`, arm 8) turns a slot
pointer into a chunk with **one read**:

```cpp
unsigned back_off = rmeta->back_offset[unit_idx] & 0x7Fu;
unsigned base_idx = unit_idx - back_off;
char *chunk_base  = mp + base_idx * ALLOC_MIN_CHUNK_SIZE - ALLOC_CHUNK_K_MAX;
```

and `RegionMeta` declares that table as

```cpp
std::atomic<BitmapWord> claim_bitmap[BITMAP_WORDS_PER_REGION];  // atomic
std::uint8_t            back_offset[NUM_ALLOCATORS_IN_SPACE];   // PLAIN
```

**a plain `uint8_t` array, immediately beside an atomic one.**  Every free reads
it plainly; `restamp_back_offset` and `deallocate_chunk` write it plainly.  (The
comment above the reader says "a single **relaxed load** of the back-offset
table" — the code does a plain load.  Worth reconciling either way, since the
two are not the same licence.)

**Why the writer is the suspect and not the reader.**  `restamp_back_offset` is

```cpp
for(unsigned u = 0; u < chunk_units; ++u)
    rmeta->back_offset[base_unit_idx + u] = (uint8_t)u | back_off_flag;
```

a byte loop whose trip count **`-fipa-cp-clone` turns into a compile-time
constant** — §13.118 measured `restamp_back_offset` as a cloned family with 6
clones — and a constant trip count is exactly what licenses merging a byte loop
into wider stores.  A store wider than the units the chunk owns corrupts the
**next** chunk's entries, after which every free of a slot in that chunk resolves
to the wrong `chunk_base` and clears a bit there.  That is §13.116's surviving
mechanism, with a reason for the `-O3` + clone dependency attached.

**The verifier** (`KAME_POOL_VERIFY_BACKOFFSET`, gated; default builds contain
none of it).  The table carries its own invariant: within one chunk the entries
are exactly `0, 1, 2, … k-1` (bit 7 aside), so an over-write from a neighbouring
restamp is a **broken run** — detectable without knowing chunk boundaries from
any other source.  `kame_pool_check_back_offset()` walks every region against
the claim bitmap and returns an anomaly count plus the first offender.

**Deliberately outside both suspect functions.**  A check inside
`restamp_back_offset` would inhibit the very vectorisation under suspicion, and
§5's lesson (a `noclone` moved 72 other function sizes) says the same about the
reader.  So it is a separate `noinline, noipa` function that only reads.

**Validated on the Mac with the pool active, both directions:**

```
round 0: reserved=33554432  anomalies=0
round 1: reserved=67108864  anomalies=0
round 2: reserved=67108864  anomalies=0
--- positive control ---
poked unit 2 -> anomalies=1  first(unit=2 val=85 expect=1)
exit=139 (SIGSEGV)
```

Three rounds of 200 000 allocations with half freed: **zero false positives**.
One byte poked (`kame_pool_poke_back_offset`, the same gate): **caught, with the
offending unit, value and expectation**.

**And the last line is the finding.**  After that single poked byte the process
**SIGSEGVs in the free path — `rc=139`, the exact signature of the failing
runs.**  That does not prove this byte is the cause; it establishes that
corrupting it is a **sufficient** mechanism for the observed crash class, and
that the class is reachable from a one-byte perturbation of a plain,
concurrently-written table read by every free.

**For Ubuntu, two things, in order:**

1. **The asm check, which is cheap and may settle it outright.**  Disassemble
   `restamp_back_offset` and its `.constprop` clones in the firing object and
   compare each clone's store widths and offsets against its constant
   `chunk_units`.  A clone that writes more bytes than its chunk owns is the
   defect, visible statically, with no run needed:
   ```
   objdump -dC --no-show-raw-insn kp_firing.o | \
       awk '/restamp_back_offset/,/^$/' | grep -E 'mov|movd|movq|movap|movup'
   ```
2. **The verifier in the failing configuration**, called at a DOUBLE-LIVE hit
   and at exit.  A nonzero count names the region and unit; a zero across
   failing runs refutes the over-write story and sends the derivation question
   back to the reader side (a torn or stale single-byte read rather than a
   ranged over-write).

Both are narrow, and between them the plain-byte-table hypothesis is decidable.
### 13.131 Adopt does not hand out the doubly-live slots — and that leaves §13.126's confound looking like the answer

§13.129 names the invariant (`occ ⊆ bits`) and says its breach is adopt handing
out storage still in use.  §13.128 exonerated the release half, so adopt was
the last candidate inside the chain.  Tested it directly by **linking the two
events**: the allocator records every adopted chunk base
(`KAME_POOL_ADOPT_CENSUS`, gated; production builds contain none of it), and
the tracer asks, at each DOUBLE-LIVE hit, whether that slot's chunk had been
adopted.

**Control first, because a "no" is worthless if adopt never runs:** a short
clean run reports **`orphan-chain ADOPTS this run = 101`**.  Adopt happens.

**Result, two failing runs:**

| run | DOUBLE-LIVE hits | chunk-was-adopted=YES | =no |
|---|---|---|---|
| 1 | 9 | **0** | 2 |
| 2 | 8 | **0** | 2 |

**No DOUBLE-LIVE hit is in an adopted chunk.**  (Only the `WPRE` probe line
carries the query, hence 2 answers per run rather than 9 and 8; the direct-
mapped adopt table holds 8192 entries against ~101 adopts, so eviction cannot
explain the zeros.)

**Which sets up a tension I am not going to resolve by choosing the convenient
half.**  §13.126's ablation removed the chain and the fault went **15/20 → 0/20
(p = 8 × 10⁻⁷)**.  Yet:

* the release half never fires its backstops, even on failing runs (§13.128);
* the adopt half never supplies a doubly-live slot (here);
* and adopt is **rare** — ~101 per run — which makes my §13.126 confound
  ("removing the chain removes reuse volume") a *weak* explanation too: 101
  chunks is not enough reuse for its removal to suppress a 75% failure rate by
  starving opportunities.

So the chain is causally necessary (ablation) while neither of its two active
paths shows the fault at the point the model predicts.  One of three things is
true: the census misses the relevant adopts (it records the base at the
`BIT_OWNED` claim — an adopt that fails partway would not be recorded), the
chain's effect is through state it *re-arms* rather than storage it hands out
(`m_owner_id`, the DLL splice, `m_owner_dll_force_walk_ptr`), or the ablation
suppresses through a route neither §13.128 nor this section measures.

**Next**, and it is cheap: count adopts on *failing* runs too (the report is
`atexit`-only for the clean case and signal-backed for crashes — both are
wired), and record adopts that are **attempted but not claimed**.  If failing
runs show a different adopt rate, or a population of partial adopts the census
never sees, that is where the discrepancy lives.

### 13.132 §13.130's predicted wide store EXISTS in the firing binary — a 16-bit store where the source is a byte loop

§13.130 predicts that `-fipa-cp-clone` turns `restamp_back_offset`'s byte loop
trip count into a constant, licensing the compiler to merge the loop into
**wider stores**.  That is checkable in the object file, and it is there.

**The firing library carries a second, tiny body:**

| build | symbols for `restamp_back_offset` |
|---|---|
| non-firing `-O2` (`arm_base.so`) | one body, `0x169` bytes, `T` and `t` at the **same address** |
| **firing `-O2 -fipa-cp-clone`** | base `0x169` **plus a `t` local of `0x22` = 34 bytes** |

34 bytes cannot be a byte loop.  Disassembled, `.constprop.1` is:

```asm
mov    %rdi,%rax
add    $0x1000,%rdi
mov    $0x100,%edx              ; value = 0x0100  -> bytes {0x00, 0x01}
shr    $0x12,%rdi               ; >> 18  (÷ 256 KiB unit)
and    $0xfffffffffe000000,%rax ; region base (32 MiB)
and    $0x7f,%edi               ; unit index & 127
mov    %dx,0x10(%rax,%rdi,1)    ; *** 16-BIT STORE ***
ret
```

`mov %dx, …` is a **word store**: two `back_offset` bytes written at once,
`[u] = 0x00` and `[u+1] = 0x01`.  The loop is gone.  `RegionMeta` confirms the
addressing — `claim_bitmap` (16 B) then `back_offset[128]` at offset `0x10`,
matching `0x10(%rax,%rdi,1)` exactly.

**So the mechanism is no longer hypothetical: the wide store exists, only in
the arm that fires.**  Whether it is *wrong* turns on one question — the clone
stamps exactly two entries, which is correct for a **2-unit** chunk and
corrupts the next chunk's entry `[0]` with `0x01` if it is ever reached with a
**1-unit** chunk.  A slot in that next chunk would then resolve to the wrong
`chunk_base` on free, which is §13.129's "bit cleared for a slot whose own free
never happened" and §13.104's DOUBLE-LIVE.

**What is NOT yet shown**, stated plainly: that the clone is called with a
1-unit chunk.  The specialization is presumably guarded by the call site's
constant, in which case it is correct and this is a dead end.  §13.130's
whole-table verifier answers exactly that, and my harness for it is not yet
working — the self-test driver segfaults in a bare program and the env-gated
poke never triggers, both my bugs, not the verifier's.  **I have not
demonstrated the verifier fires, so I am claiming nothing from a clean table.**

**Next, and it is small**: call `kame_pool_check_back_offset()` from inside the
reproducer (where the pool is genuinely live) rather than from a standalone
driver, prove it catches `kame_pool_poke_back_offset(0,7,0x5a)`, and only then
read its verdict on a failing run.
### 13.133 §13.131's tension resolved by audit: a cross-thread poke into owner TLS, dereferenced after an unbounded window

§13.131 ends with three possibilities and says it will not pick the convenient
one.  Its third — "the chain's effect is through state it **re-arms** rather than
storage it hands out" — is the one that survives an audit, and the audit finds a
concrete use-after-free.

**The code.**  In `CrossDeallocBatch::flush`, per batched entry:

```cpp
std::atomic<bool> *p = chunk->m_owner_dll_force_walk_ptr.load(acquire);
i += chunk->batch_return_to_bitmap(&buf[i]);   // unbounded; may release a chunk
if(p) p->store(true, relaxed);                 // deref, much later
```

`p` points into the **owner thread's TLS** (`&s_tls.dll_force_walk_from_head`).
Owner exit does `m_owner_dll_force_walk_ptr.store(nullptr, release)` and then
lets its TLS die, and the comment there argues the deref is safe:

> "A freer that observes the old non-null pointer must have loaded BEFORE our
> release, in which case our TLS is still live."

**That does not follow.**  Loading before the release says nothing about the
**deref** happening before the TLS dies.  The window is `[load, deref]` and it
spans `batch_return_to_bitmap`; a freer that saw non-null has **no**
happens-before edge constraining the owner's *subsequent* teardown.  The
release-store orders the owner's prior work before a freer that reads **null** —
which is a different freer.

**Two things say this is not hypothetical.**  The teardown path already passes
`at_teardown ? nullptr` *because* the pointer "may dangle under musl's TSD-dtor
ordering", and the null-store's own comment records that this field already
produced a SEGV once (Linux 1000-thread `alloc_stress`).  **Only the teardown
path was guarded**; the general path rests on the inference above.

**And the target is recycled memory.**  The TLS block is itself
pool/malloc-allocated and freed at thread exit, so the write lands in storage
that may now hold a live object.  The value written is a one-byte `true`.

**It accounts for the observations that the chain's two halves could not:**

| observation | this mechanism |
|---|---|
| §13.131: **no** DOUBLE-LIVE hit is in an adopted chunk (0/17) | the victim is wherever the freed TLS block got reused — unrelated to the adopted chunk |
| §13.126: ablating the chain gives 0/20 | adopt is what **re-arms** this pointer to a *new* thread's TLS (`allocator.cpp:3025`); no chain, no re-arming |
| §1: the reproducer needs **repeated thread create/exit** | without exits the TLS is never freed and the window never matures |
| §13.110: hits cluster within one chunk | a ~32 KiB TLS block reused as many small slots in one chunk |
| §13.113/§13.115: ordinary live objects at refcount 1 | a stray 1-byte write, not a structural corruption |
| §13.128: release-half backstops never fire | nothing is disposed early; the damage is a write, not a lifetime error |

**The gate, which is a candidate fix as much as an experiment.**
`KAME_NO_XTHREAD_FORCEWALK_POKE` skips the cross-thread poke entirely.  The flag
it sets is documented as a hint with "one-cycle false-negative delay
acceptable", so the cost is a delayed DLL walk and nothing else.  Unlike
§13.126's ablation it removes **exactly one write** — the chain stays, adoption
stays, reuse volume stays — so a positive result is not confounded the way
§13.126 conceded its own was.  Mac: default build unchanged (`HITS 0`,
`dtor == born`, `enforced 96 823 433`), both arms compile.

**Also fixed: §13.131 broke every macOS build of the test.**  `rc_trace.cpp`
referenced `kame_pool_was_adopted` with `__attribute__((weak))`, but `weak` on an
**undefined** reference is an ELF-ism — on Mach-O it does not make the reference
optional, and neither does Darwin's `weak_import` when no linked dylib exports
the symbol at all:

```
Undefined symbols for architecture arm64:
  "_kame_pool_was_adopted", referenced from: kame_rc_trace::anomaly(...)
```

So the tracer's buildability was coupled to an allocator diagnostic flag
(`KAME_POOL_ADOPT_CENSUS`).  Now resolved by `dlsym`, which this file already
does for `kame_poison_decode`; verified both ways (links without the flag; the
symbol is exported and found with it).

**To run:** `-DKAME_NO_XTHREAD_FORCEWALK_POKE` on the allocator, interleaved
against the firing baseline in one job.  If it suppresses, the write is named and
the fix is the gate.  If it does not, the window is real but not load-bearing,
and it should be closed anyway — a documented dangling deref with a prior SEGV
attached is not something to leave on the argument that it did not happen to be
this bug.

### 13.134 A working harness for the verifier — both §13.132 symptoms are the same two traps

§13.132 confirms §13.130's predicted wide store in the firing binary (a 16-bit
`mov %dx, 0x10(%rax,%rdi,1)` where the source is a byte loop) and then reports
its verifier harness not working: "the self-test driver segfaults in a bare
program and the env-gated poke never triggers".  Both have specific causes, both
cost this side the same time, and neither is the verifier's fault.

1. **The SIGSEGV is expected, and it arrives AFTER the detection.**  Once a
   `back_offset` entry is corrupted, every later free of a slot in the affected
   chunk mis-derives `chunk_base`, so the process dies in the free path with
   `rc=139`.  That is the point — it is the failing runs' signature reproduced
   from a one-byte poke — but with block-buffered `stdout` the buffer dies with
   the process and the run looks like it printed nothing.  `setvbuf(_IONBF)`
   before anything else.
2. **Poking an UNCLAIMED unit is invisible by design.**  The verifier walks
   `claim_bitmap` and skips unclaimed units, because an unclaimed entry
   legitimately reads 0.  A poke at a fixed index hits an unclaimed unit on most
   runs and is correctly ignored — which presents exactly as "the poke never
   triggers".  Poke a unit the run has actually claimed: sweep a range and stop
   at the first index where the count rises.

**`kamepoolalloc/tests/backoffset_verify_test.cpp`** is the harness with both
handled, and it works here (pool active, arm64):

```
round 0                reserved=33554432  anomalies=0
round 1                reserved=67108864  anomalies=0
round 2                reserved=67108864  anomalies=0
--- base rate over 3 rounds: 0 anomalies ---
--- positive control ---
poked unit 2   -> anomalies=1  first(unit=2 val=85 expect=1)  CAUGHT
exit=139
```

It also **fails loudly** (`return 2`) if the poke sweep never fires, so a clean
table can never be quoted from a run whose control did not work — the §13.61 rule
built into the harness rather than left to the reader.

Build note repeated because it is the third time it has bitten: the pool must be
**active**, i.e. a shared library with `-DKAMEPOOLALLOC_DYLIB`.  Compiling
`allocator.cpp` into the executable links the pool in but leaves `new`/`delete`
on libc, `kame_pool_reserved_bytes()` reads 0, and freed blocks carry no poison
(§13.109).  The harness prints `reserved=` on every line so a pool-inactive run
is obvious at a glance.

**On §13.132's open question** — whether the 2-entry clone is ever reached with a
1-unit chunk — one observation from the disassembly worth recording, and one
reason it is probably not the bug.  The clone masks the unit index with
`and $0x7f` and then does a **2-byte** store at `0x10(%rax,%rdi,1)`; `RegionMeta`
puts `back_offset[128]` at `0x10` with `dll_next` immediately after, so a store
at index **127** would write `back_offset[127]` **and the low byte of
`dll_next`**.  But a 2-unit chunk cannot have base unit 127 (unit 128 does not
exist, so the claim could not have succeeded), and an IPA-CP clone is only
called from the sites whose constant matched — so both the 1-unit and the
index-127 cases require the specialization itself to be mis-dispatched, which is
a different and much stronger claim than a wide store.  The verifier is what
settles it either way, which is why the harness mattered more than more staring.

### 13.135 The wide store is CORRECT — `CHUNK_UNITS` is a per-instantiation constant, so the specialization is exact. One residual check names itself

§13.132 confirms the 16-bit store and leaves the right question open: is the
2-entry clone ever reached with a chunk that does not own 2 units?  The source
answers it.

```cpp
static constexpr unsigned int CHUNK_UNITS =
    (ALIGN < 256u) ? 1u : (ALIGN < 1024u) ? 2u : 4u;
static constexpr size_t CHUNK_SIZE = CHUNK_UNITS * ALLOC_MIN_CHUNK_SIZE;
```

`CHUNK_UNITS` is a **compile-time constant per instantiation**, and call site 1
passes `ALLOC::CHUNK_SIZE` — also compile-time.  So IPA-CP's specialization is
**exact**: a clone that stamps two entries can only be called from a 2-unit
instantiation (`256 ≤ ALIGN < 1024`: the instantiated 256, 272, 288, 304, 320,
336, 352, 368), where `chunk_units` genuinely is 2.  It writes precisely the two
entries that chunk owns.

The index-127 worry from §13.134 closes the same way: a 2-unit chunk's claim sets
**two contiguous** claim bits, so its base unit is ≤ 126; the clone's `and $0x7f`
is the region-offset extraction, not a clamp that could yield 127 for a 2-unit
chunk.

**So the wide store is not the defect, and §13.132's lead should be closed
rather than run.**  Recorded as a *closed* lead and not a silent drop, because
the prediction (§13.130) was right about the transformation and wrong about its
consequence — which is worth distinguishing: `-fipa-cp-clone` really does turn
that byte loop into a wider store, and that store is correct.

**The residual check, and it is the interesting one.**  Call site **2** is
different:

```cpp
restamp_back_offset(cached, actual, /*back_off_flag=*/0x80u);   // allocate_dedicated_chunk
```

`actual` is a **runtime** value — `ceil((size + K_MAX) / 256K)` units for a
dedicated (large) chunk.  The clone §13.132 disassembled hardcodes
`0x0100` = bytes `{0x00, 0x01}`, i.e. `back_off_flag == 0`, so it serves site 1.
But §13.118 measured `restamp_back_offset` with **6** clones.  So:

> Enumerate **every** `restamp_back_offset.constprop.*` body in the firing
> object and check each one's hardcoded byte count against a legal
> `CHUNK_UNITS` (1, 2 or 4) **and** its flag.  A clone carrying flag `0x80` with
> a hardcoded count is the one to look at, because site 2's size is not a
> compile-time constant — a specialization there would be IPA-CP asserting a
> constant it cannot have from that call site.

That is a static check on the object already built, no run needed:

```bash
nm -C kp_firing.so | grep restamp_back_offset      # expect several .constprop
objdump -dC --no-show-raw-insn kp_firing.so | \
    awk '/restamp_back_offset.*constprop/,/ret/'   # read each body's stores
```

**Meanwhile the live lead is §13.133**, which is a documented dangling
dereference rather than a suspicion: a cross-thread poke into owner TLS,
loaded before `batch_return_to_bitmap` and dereferenced after it, whose safety
argument ("the freer must have loaded before our release, so our TLS is still
live") does not follow — and whose field already produced one SEGV
(`alloc_stress`, 1000 threads) and is already guarded on the teardown path for
exactly this reason.  `-DKAME_NO_XTHREAD_FORCEWALK_POKE` removes that one write
and nothing else, so unlike §13.126's ablation a positive result is unconfounded,
and the write should be removed whether or not it is this bug.
### 13.136 The `back_offset` table is intact on failing runs — the wide store is real but is not corrupting it

§13.134's harness works here, and both of §13.132's harness failures were as
diagnosed: the SIGSEGV arrives **after** detection (a corrupted entry kills the
next free), and block-buffered stdout hid the report; a fixed poke index lands
on an unclaimed unit, which the verifier skips by design.

**Control, on Ubuntu, pool active** (`reserved=33554432` → `67108864`):

```
round 0/1/2                      anomalies=0
--- positive control ---
poked unit 2  -> anomalies=1  first(unit=2 val=85 expect=1)  CAUGHT
```

So the verifier is proven to fire *here*, not merely on arm64.

**Then the real question — is the table corrupt on runs that actually fail?**
Wired `kame_pool_check_back_offset()` into the reproducer **per round** (not at
exit: a failing run dies first), with `setvbuf(_IONBF)` per §13.134's note.

| runs | failures | rounds checked | `BO-CORRUPT` |
|---|---|---|---|
| 12 | **8** (`rc=139` ×7, `rc=255` ×1) | every round of every run | **0** |

**Zero corrupted entries across eight failing runs**, with the checker proven
live in the same build.

**So §13.132's wide store, though real, is not the fault.**  The 16-bit
`mov %dx,0x10(%rax,%rdi,1)` exists only in the firing build — that stands — but
it never writes an entry the whole-table invariant rejects.  The reading it
suggested (a clone stamping two units for a one-unit chunk) is therefore
**refuted**: the specialization is reached only where two units are genuinely
owned, exactly as a correct constant-propagation should.

**And that closes §13.129's remaining candidate.**  Its analysis left "a bit
cleared for a slot whose own free never happened, i.e. a free that mis-derived
`chunk_base`" as the one way to breach `occ ⊆ bits`; `chunk_base` is derived
from this table, and the table is clean under the fault.  Together with
§13.128 (release half never fires its backstops) and §13.131 (no DOUBLE-LIVE
hit is in an adopted chunk), all three named paths inside the orphan chain are
now measured clean on failing runs — while removing the chain still suppresses
the fault 15/20 → 0/20.

The discrepancy in §13.131 is therefore not a loose end but **the** finding:
the chain is necessary, and none of scrub/dispose, adopt, or back_offset
derivation is where it goes wrong.

### 13.137 Correction to §13.133: the TLS block is NOT pool storage — measured, so the mechanism I proposed does not stand (the window still does)

§13.136 clears the last of the three named chain paths, which left §13.133 as
the surviving candidate.  Before that gets run on the strength of my reasoning,
I measured the premise it rests on — and it fails.

**§13.133's mechanism required** the owner's TLS block to be freed at thread exit
and **reused as ordinary pool storage**, so the stray `p->store(true)` would land
on a live object (a `Packet`'s refcount byte, say).  I asserted that from a
memory note about the pool self-allocating its TLV block through interposed
malloc.  Measured instead, against the allocator's **own** TLS via
`PoolAllocator<ALLOC_SIZE4,true>::dll_head_tls_addr()` (not a TU-local
`thread_local`, which is a different TLV block and would not have answered the
question):

```
allocator TLS addrs: n=64 distinct=1 first=0x703007358 reserved=33554432
pool-owned: 0 / 1
allocations covering an allocator-TLS address: 0     (400 000 allocations)
```

- **`pool-owned: 0`** — the allocator's `s_tls` is not pool storage on
  macOS/arm64.  The leak fix `e6d6cd0b` moved that block to the real heap, which
  also removed its pool provenance; my memory of the pre-fix state was what I
  reasoned from.
- **0 overlaps in 400 000 allocations** — no pool block ever covered it.
- **`distinct = 1` across 64 sequential threads** — the TLS address is *identical*
  every time, so a stale pointer from thread N points at thread N+1's **live**
  TLS.  The write then sets `dll_force_walk_from_head = true` on an innocent
  thread, which is a **hint** ("one-cycle false-negative delay acceptable"):
  harmless.

**So the specific path to this fault is not supported.**  Stated plainly because
I was one message away from having the other side spend a run on it as "the live
lead" on mechanism grounds that measurement does not carry.

**What still stands, and is unaffected by the above:**

1. The window `[load, deref]` spanning `batch_return_to_bitmap` is real, and the
   comment's safety argument ("the freer must have loaded before our release, so
   our TLS is still live") is invalid as written — loading before the release
   says nothing about the deref preceding the TLS's death.
2. The field already produced one SEGV (`alloc_stress`, 1000 threads), and the
   teardown path is *already* guarded with `at_teardown ? nullptr` **because the
   pointer may dangle**.  Only the general path rests on the invalid inference.
3. It is therefore a bug to close on its own merits, and the gate closes it at
   the cost of a delayed DLL walk.

**What would revive the mechanism, and it is a Linux question I cannot settle
here.**  The dangerous case needs the TLS block **freed and reused by something
else**, which needs genuine concurrent thread churn (not the sequential
create/join above) and depends on where TLS comes from: libc heap on macOS via
`_tlv_get_addr`, the dynamic-TLS arena for a `dlopen`'d DSO on Linux.  The probe
is four lines — record `PA::dll_head_tls_addr()` per thread with threads running
**concurrently**, then check `kame_pool_malloc_usable_size()` on each address and
whether any later allocation covers one.  If Linux answers "pool-owned" or
"overlaps", §13.133's mechanism revives and the gate becomes the primary
experiment again; if it answers as macOS did, the gate is a correctness fix and
**not** this fault.

**Revised standing.**  With §13.136, all three named chain paths are clean on
failing runs and §13.133 is demoted from mechanism to unrelated-bug-plus-open-
question.  The chain remains necessary by ablation (15/20 → 0/20) with nothing
inside it accounting for that — so §13.122's whole-unit codegen reading is again
the only explanation standing, and `nonclone_memop_diff.py` on §13.119's minimal
pair is the tool that has not yet been run.  The poke gate is still worth one
job: it is cheap, and a positive result would be decisive whatever the mechanism
turns out to be.
### 13.138 The cross-thread TLS poke is a real UAF but not this fault — 10/18 vs 11/18

§13.132's audit finds a genuine use-after-free: `CrossDeallocBatch::flush`
loads `m_owner_dll_force_walk_ptr` (a pointer into the OWNER thread's TLS),
calls `batch_return_to_bitmap` (unbounded, may release a chunk), then
dereferences the pointer — with no happens-before edge on the owner's
teardown in between.  It is exactly the "state the chain re-arms rather than
storage it hands out" class §13.131 pointed at, and §13.132 gates it with
`KAME_NO_XTHREAD_FORCEWALK_POKE`.

**Causal test, 18 interleaved rounds, `20 40 700`, `taskset -c 0-3`:**

| arm | failures |
|---|---|
| poke present (baseline) | **10/18** |
| **`-DKAME_NO_XTHREAD_FORCEWALK_POKE`** | **11/18** |

**No effect.**  The gate removes the dereference entirely and the rate does not
move — so the TLS poke, while a real defect that should be fixed on its own
merits, is **not** the fault this hunt is chasing.

**A methodological correction I owe on this run.**  I first declared the batch
invalid because 18 rounds finished in ~4 minutes, when earlier `20 40 700` runs
had taken 1–2 minutes each.  I was wrong: with the machine idle a successful
run takes **~11.7 s** and a failing one **1.3–1.8 s** (measured), so 36 runs in
4 minutes is exactly right.  The earlier 1–2 min figures were runs competing
with concurrent batches.  **Run duration is not a validity check** — it tracks
machine load, which §13.51 already established varies this rate by a factor of
several, and I used it as a proxy for "did this really run" anyway.  The real
check is the one that then confirmed it: time a run directly and compare the
success and failure distributions.

**Standing.**  Four named mechanisms inside or around the orphan chain are now
measured clean on failing runs — release backstops (§13.128), adopt supplying
doubly-live slots (§13.131), `back_offset` corruption (§13.136), and now the
cross-thread TLS poke — while removing the chain still suppresses the fault
15/20 → 0/20 (§13.126).  The ablation's discriminating power is undiminished
and none of its parts has confessed.

### 13.139 §6 does not say what we have been reading it to say: the STM's refcount primitive is compiled INSIDE the allocator's TU

Four named mechanisms in and around the orphan chain are now measured clean on
failing runs — release backstops (§13.128), adopt supplying doubly-live slots
(§13.131), `back_offset` corruption (§13.136), the cross-thread TLS poke
(§13.138) — while removing the chain still suppresses the fault 15/20 → 0/20
(§13.126).  §13.138 calls that "none of its parts has confessed".  There is a
part that was never on the list.

**The structural fact.**  `atomic_smart_ptr.h` is a **header**, included by
`allocator_prv.h:525`.  So `atomic_shared_ptr<PoolAllocator<N,…>>` and its entire
tagged-pointer refcount machinery — `load_shared_`, `acquire_tag_ref_`,
`release_tag_ref_`, `compareAndSet_impl_`, `local_shared_ptr` constructors and
destructors, `atomic_intrusive_dispose` — are instantiated **inside
`allocator.cpp`'s translation unit and compiled with the allocator's flags**.
Confirmed, not assumed:

```
$ nm -C libkamepoolalloc.so | grep -o 'compareAndSet_impl_<local_shared_ptr<PoolAllocator[^>]*>'
... one per size class: 16, 112, 144, 176, 208, 240, 256, 272, ...
```

**What that does to §6.**  Its table —

| | |
|---|---|
| clang-STM + gcc-pool | 8/12 |
| gcc-STM + clang-pool | 0/12 |
| allocator `-O2` | 0/8 |

— has been read since §13.103 as "the defect is in allocator **logic**".  It does
not say that.  It says the defect is in **whatever gcc compiles in that TU**, and
that TU contains the STM's refcount primitive as well as the allocator.  A
miscompile of the primitive there follows the allocator's compiler exactly as
allocator logic would, and is **invisible in the STM's own TU** — which is also
why the STM-side audits (§13.91–§13.98) were right to come back clean while the
same code, compiled elsewhere, could still be the fault.

**And it explains the discrepancy §13.138 leaves open.**
`orphan_chain_push` is the **only** place a chunk's `refcnt` is established and
the **only** place a `local_shared_ptr<PoolAllocator>` is created.  So
`KAME_NO_ORPHAN_CHAIN` does not merely disable adopt and scrub: it removes the
**entire primitive** from the pool.  An ablation that suppresses the fault while
every *logic* path inside it measures clean is exactly what you would see if the
primitive, not the logic, were at fault.

**The arm** (`KAME_ASP_AT_O2`): wrap that one `#include` in
`#pragma GCC push_options` / `optimize("O2")` / `pop_options`, so the primitive
compiles at `-O2` while the allocator around it keeps the firing flags.  It
separates the two readings §6 cannot:

| result | conclusion |
|---|---|
| fault **dies** | the defect is the refcount primitive as compiled here; the allocator's own code is a bystander, and 30 sections of allocator-logic auditing were looking one layer off |
| fault **stays** | the primitive is not it, and the remaining candidate is the allocator's non-clone codegen (§13.122) — with `nonclone_memop_diff.py` still unrun |

**Caveats, the same ones §13.112 spells out for the function attribute.**
`#pragma GCC optimize` changes inlining across its boundary and gcc documents it
as a debugging aid, so a positive result localises "**this code's codegen
matters**", not "this line is wrong".  And verify the arm did something before
trusting either outcome — the primitive's instantiations should change shape
between the two objects; `nonclone_memop_diff.py` on the arm pair reads that
directly, and an arm whose primitive is byte-identical is vacuous, not negative
(§13.117 needed exactly this check for five of seven arms).

**Mac**: both arms compile; the default build is unchanged (`HITS 0`,
`dtor == born`, `enforced 95 938 287`).  The runs need gcc.

### 13.140 `KAME_ASP_AT_O2` as written is vacuous — corrected, the refcount primitive's own clones are not the fault either

§13.139's reframing is right and important: `atomic_smart_ptr.h` is included at
`allocator_prv.h:525`, so the STM's refcount primitive is instantiated **inside
the allocator's TU** and compiled with the allocator's flags.  §6 therefore
never localised to allocator *logic*.  The arm it adds to test that, however,
does nothing.

**Vacuity check first (§13.117's rule).**  `KAME_ASP_AT_O2` wraps the include in

```cpp
#pragma GCC optimize("O2")
```

but the TU is **already** compiled at `-O2 -fipa-cp-clone`.  The pragma restates
the level and does not cancel the pass, so the arm is a no-op: clone counts
identical (31 vs 31) and **`.text` byte-identical**.  A 0 from it would have
meant nothing.

**Corrected** to name the pass —

```cpp
#pragma GCC optimize("O2","no-ipa-cp-clone")
```

— and the arm becomes real: **31 → 29 clones**, `.text` differs.

**Causal test, 30 interleaved rounds, `20 40 700`:**

| arm | failures |
|---|---|
| firing baseline | **22/30 (73%)** |
| **primitive at `no-ipa-cp-clone`** | **26/30 (87%)** |

Fisher **p = 0.33** — no effect, and if anything slightly *more* failures.

**So the refcount primitive's own specializations are not the fault.**  §13.139
correctly identifies that `atomic_shared_ptr`'s machinery is compiled in the
allocator's TU and was never on the candidate list; measured, suppressing its
cloning changes nothing.  That is consistent with §13.124 (the 22
`acquire_tag_ref_` clones are `weakly=false`, semantically inert) and §13.126
(suppressing all 22 gave 13/22 vs 13/22).

**Running total inside the localisation.**  Five named mechanisms now measured
clean on failing runs — release backstops, adopt, `back_offset`, the TLS poke,
and the primitive's clone set — against an ablation that still takes the rate to
zero.  Every arm that *specifically* removes a candidate leaves the rate alone;
only removing the orphan chain **wholesale** suppresses.  That pattern itself is
now the most informative thing left: it says the chain's contribution is not any
single operation it performs.

### 13.141 `KAME_NO_ORPHAN_CHAIN` removes two things at once — one binary, two arms, zero codegen delta

§13.140 is right that my `KAME_ASP_AT_O2` was vacuous: `#pragma GCC
optimize("O2")` restates a level the TU already has and does not cancel a pass
given on the command line.  The vacuity check that caught it is the one §13.139's
own caveats asked for, which is the system working — but it was my arm that
needed it.  Its corrected form (`"O2","no-ipa-cp-clone"`, 31 → 29 clones) then
measured 22/30 vs 26/30, p = 0.33: the primitive's clones are not the fault
either.

**Which brings §13.140's summary to the point that matters.**  "The chain's
contribution is not any single operation it performs" has a reading that requires
no operation at all: **the compile-time ablation deletes code.**  With
`KAME_NO_ORPHAN_CHAIN` the push site becomes `(void)c;`, so
`orphan_chain_push`/`_pop`/`_scrub` and the whole
`atomic_shared_ptr<PoolAllocator>` instantiation are unreachable and eliminated —
changing IPA-CP propagation, inlining and register allocation for **everything
else in the TU**.  Measured, clang, same flags otherwise:

| build | `orphan_chain*` symbols |
|---|---|
| runtime gate (code kept) | **220** |
| `KAME_NO_ORPHAN_CHAIN` | **184** |

**36 symbols' worth of code disappears.**  So §13.126's 15/20 → 0/20 is
consistent with a **codegen** effect and not a behavioural one — which is
§13.122's whole-unit reading, and which would reconcile every clean arm with the
ablation's power at a stroke.

**`KAME_ORPHAN_CHAIN_RUNTIME_GATE` separates them.**  The chain code stays
compiled and reachable; only the *call* is skipped, decided at run time from the
environment:

```
KAME_ORPHAN_CHAIN_OFF=1 ./reproducer      # behaviour ablated
./reproducer                              # baseline
```

**One binary, two arms, zero codegen delta** — which no arm in this
investigation has achieved before.  Every previous ablation and every clone arm
changed the object; this one cannot, because it *is* the same object.

**Both halves validated on the Mac.**  Behaviourally live (§13.61), proven by the
stranding it causes rather than asserted:

```
KAME_ORPHAN_CHAIN_OFF=(unset)  reserved=234 881 024 bytes
KAME_ORPHAN_CHAIN_OFF=1        reserved=335 544 320 bytes    (+43 %)
```

40 rounds × 8 threads, each thread leaving one live slot so its chunk is
non-empty at exit and therefore orphaned.  With the chain on those chunks are
adopted; with it off they strand, and the pool grows 43 %.  The flag is read
through an `std::atomic<bool>` so the branch cannot be folded and the chain
deleted behind it; read once and cached, so the cost is one relaxed load on a
cold path.  Default build unchanged.

**What the two outcomes mean, stated before the run:**

| result | conclusion |
|---|---|
| fault **persists** with the chain behaviourally off | the chain is **not causally involved**; §13.126's ablation worked by deleting code, the fault is a whole-TU codegen consequence, and `nonclone_memop_diff.py` on §13.119's minimal pair becomes the only remaining step |
| fault **dies** | the behaviour is necessary after all — and since five named mechanisms are clean, it is an unnamed one, but now *provably behavioural*, which is a different and much better-posed search |

This is worth running ahead of everything else queued, because it is the only
experiment left that cannot be confounded by codegen.

### 13.142 Three proposals from the user, ranked — and (i) built: a tag-mask census

Three ideas, with an honest ranking before any of them is worked: **(iii) is the
most decisive, (i) the best value per unit of effort, (ii) a sharpening of a
census that already exists.**

#### (i) Does gcc ever drop the low-tag mask?  — **yes, and IPA-CP supplies the premise**

`atomic_shared_ptr` keeps a local refcount in the **low bits** of the pointer
word, so every use masks:

```cpp
(Ref*)(ref & ~(LOCAL_REF_CAPACITY - 1))     // pointer -> `and ~7`
(Refcnt)(ref & (LOCAL_REF_CAPACITY - 1))    // count   -> `and 7`
```

A compiler that concludes the value is an aligned pointer may delete `& ~7` as
**redundant** — and this is not idle: **IPA-CP is exactly what supplies the
premise**, because a specialization propagating a zero refcount makes
`(uintptr_t)pref + 0` provably 8-aligned.  Correct for that clone; wrong the
moment such a body is reached with a tagged value.  The project already knows the
mirror of this hazard (CLAUDE.md forbids `alignas(N)`/`alignof(Ref)` for the
constant, because pre-C++17 `operator new` need not honour it, "causing silent
pointer corruption and rare crashes").

What the source says: `m_ref` is declared **`uintptr_t`, not a pointer type**,
which closes the direct type-based route — so the question has to be put to the
object file, not the source.

**`kamepoolalloc/tests/tagmask_census.py`** does that.  Per body touching the tag
machinery it counts `mask_ptr` (masks with `~(CAP-1)`), `mask_cnt` (masks with
`CAP-1`) and `tagged_add` (adds of a constant `< CAP` to a pointer-shaped value,
i.e. *building* a tagged word), and flags any body that builds or consumes a
tagged value and **never masks**.  Baseline here (clang, arm64, pool library):

```
bodies touching the tag machinery: 31
totals: mask_ptr=131 mask_cnt=103 tagged_add=72
SUSPECT (builds/consumes a tagged value, never masks): 0
per body, uniformly: add=3 mask_ptr=5 mask_cnt=4
```

The uniformity is what makes it useful: with every instantiation at the same
counts, a **differential** between the firing and non-firing builds is readable
at a glance, and a body whose `mask_ptr` **drops** is the finding.  Absolute
counts prove nothing on their own — inlining moves masks between bodies — so it
is a differential tool by design and says so.

#### (ii) Do the clone count and the refcount-access count reconcile in the asm?

Worth doing, but a **raw total will not mean anything**: a clone legitimately has
fewer refcount RMWs than its parent (the `weakly=false` specializations drop a
whole branch, §13.125), so a difference is expected and a match is luck.  The
sharp form is per-path rather than per-total:

> for each clone, check that **every path from entry to return** performs the
> same multiset of refcount RMWs as the corresponding source path.

A clone with a path that reaches `ret` having done **one fewer** `lock add`/`lock
sub` on `refcnt` than the source path it specializes is a lost reference, which is
this fault.  §13.23's "22→44 lock-add census" is the same instinct applied to one
function; the generalisation is to make it path-wise and run it over every clone.
Cheaper than it sounds, because the bodies are small.

#### (iii) Hand-write the clones under two names — **the most decisive of the three**

This is the one that can do something no previous arm could.  §13.120 established
that per-function licensing **cannot reach** the four functions IPA-CP specializes
from caller context (`l1_/global_/recycle_pop_fit`, `acquire_tag_ref_`), and
§13.121's subtractive direction reaches them only by *removing* cloning.  Neither
can produce those specialized bodies at `-O2`.

**Writing them by hand can** — because you write the call sites too.  Duplicate
the function under a second name with the constant folded in, call it from the
sites that would have been specialized, and build at plain `-O2` (0/8).  Then:

| result | conclusion |
|---|---|
| fires | the clone set **is** the fault, and §13.119's negative was an artifact of the mechanism rather than evidence about the hypothesis |
| clean | the clone set is refuted a third way, by the one method that can reach all twelve specialized functions |

It also gives something no gcc-flag arm can: a **permanently reproducible** case
that does not depend on a compiler's choices, which is what a regression test
needs.  The cost is real (twelve functions, and the caller edits have to be
faithful), which is why it is ranked most decisive rather than first to run.

**Suggested order**, given what is already queued: §13.141's runtime gate first
(it is the only arm that cannot be confounded by codegen, and it is one env var),
then (i)'s census on the §13.119 pair (static, minutes), then (iii) if both come
back empty.

### 13.143 Proposal (ii), put precisely: a missing `cmpxchg` IS visible in the asm; a missing `relaxed` is NOT

The question is whether the compiler **omitted** a `cmpxchg` or a relaxed
operation.  The two halves have different answers and it matters which.

**A `cas`/`rmw` that disappears is visible, and now counted.**  GCC does not
delete an atomic read-modify-write as such — but constant propagation that kills a
branch condition removes the **whole path** containing one, which is the realistic
failure mode and is exactly what a per-body count catches.  `tagmask_census.py`
now reports, per body touching the tag machinery: `cas` (`lock cmpxchg`; arm64
`cas*` or an `ldaxr`/`stlxr` pair), `rmw` (`lock add/sub/xadd/or/and/xchg`; arm64
`ldadd*`/`swp*`) and `fence` (`mfence`/`dmb`).

Baseline (clang, arm64, pool library) — and the **uniformity is the finding
mechanism**:

```
bodies touching the tag machinery: 34
totals: mask_ptr=131 mask_cnt=103 tagged_add=72 | cas=112 rmw=231 fence=0

per body, every instantiation identical:
  add=3  mask_ptr=5  mask_cnt=4  cas=4  rmw=9  fence=0
```

Every `compareAndSet_impl_<local_shared_ptr<PoolAllocator<N,…>>, …>` has
**exactly 4 compare-exchanges and 9 other atomic RMWs**.  So on the firing build
any body that deviates is visible at a glance, and a body with `cas < 4` has lost
a compare-exchange path.  That is a one-command check on an object already built.

**A missing relaxed load or store is NOT visible, and no asm tool can make it
so.**  A relaxed load compiles to a plain `mov`/`ldr`, indistinguishable from a
non-atomic load; a relaxed store likewise.  So "the relaxed op was omitted" cannot
be observed directly — only *relatively*, as fewer loads than the source path
performs, which inlining changes anyway.  Counting `mov`s and calling the
difference a missing relaxed load would produce exactly the kind of number that
looks like evidence and is not, so the tool refuses to report it.

For that half the instrument is **runtime accounting**, and most of it exists:
the tracer's DEC ledger and the `dtor == born` identity (§13.74 closed six
previously-untraced decrement paths for exactly this reason, and §13.107's
destructor hook made the birth/death identity exact — `96 588 856 == 96 588 856`).
An omitted relaxed increment or decrement shows up there as an arithmetic
mismatch, which is a real observation, unlike an instruction count.

**So the answer to (ii) splits:** the `cmpxchg` half is answerable statically and
the tool is ready; the `relaxed` half is answerable only dynamically and that
instrument is already built and already reporting balance. What has *not* been
done is running the static half on the §13.119 minimal pair — which is minutes,
and is now second in the queue behind §13.141's runtime gate.
### 13.144 The ablation survives the codegen confound: runtime gate, one binary, 15/24 vs 0/24

§13.141 identifies a confound in §13.126 that I did not flag: the compile-time
`KAME_NO_ORPHAN_CHAIN` **deletes code**, so `orphan_chain_push/_pop/_scrub` and
the whole `atomic_shared_ptr<PoolAllocator>` instantiation become unreachable
and are eliminated — changing IPA-CP propagation, inlining and register
allocation for everything else in the TU (measured there: 220 → 184 symbols).
On that reading 15/20 → 0/20 could have been a **codegen** effect, not a
behavioural one, which would have made my §13.126 conclusion wrong in kind
rather than in degree.  I flagged the reuse-volume confound and missed this one.

**Its runtime gate settles it.**  `KAME_ORPHAN_CHAIN_RUNTIME_GATE` compiles the
chain in unconditionally and skips only the *call*, chosen from the
environment — **one binary, both arms, zero codegen delta.**

**24 interleaved rounds, `20 40 700`, `taskset -c 0-3`, same `.so`:**

| arm | failures |
|---|---|
| chain enabled (baseline) | **15/24 (63%)** |
| **`KAME_ORPHAN_CHAIN_OFF=1`** | **0/24 (0%)** |

Fisher **p = 1.4 × 10⁻⁶**.

**So the effect is behavioural, not codegen.**  Identical machine code, the
only difference being whether `orphan_chain_push` is *called*, and the fault
goes from 63% to never.  §13.126's localisation stands, now on a design that
cannot be explained by clone-set or inlining differences — which also means
§13.122's whole-unit-codegen reading, whatever else it explains, does not
explain this.

**Where that leaves the picture.**  The chain is behaviourally necessary, and
five named mechanisms inside it are individually clean on failing runs
(§13.128, §13.131, §13.136, §13.138, §13.140).  The next question is therefore
not *which operation* but *what varies with the amount of chain traffic* —
§13.141's runtime gate makes that measurable for the first time, since the
same binary can now be run with the chain on, off, or (with a small addition)
throttled.

### 13.145 Disassembly needs no Linux — real GCC 15.2 builds the allocator here. Both (i) and (ii)-static come back NO

The user's point: a static check does not need to **run**, so a local `g++`
suffices.  Correct, and it removes the Linux round-trip from every static check
in this investigation.  MacPorts has **GCC 15.2.0 — the same version Ubuntu is
using**.

**What actually blocks GCC on macOS** (three things, none a project rule; this is
the substance of `cdb70d2cf` for inspection purposes):

1. The macOS 26 SDK's `<malloc/malloc.h>` pulls in `<mach/message.h>`, which uses
   the clang-only `xnu_static_assert_struct_size` and GCC rejects outright.
   `tests/gccprobe/malloc_shim.h`, placed as `<malloc/malloc.h>` earlier on the
   include path, declares the six zone functions and the one struct member the
   code touches (`z->size`).  Layout plausible, not faithful — **inspection
   only, never run**.
2. Darwin GCC does not support `__attribute__((constructor(N)))` priorities.  A
   scratch copy rewrites `constructor(N)` → `constructor`, which changes
   initialisation ORDER and no function body being censused.
3. Nothing else.

`tests/gccprobe/build_gcc_probe.sh` does all of it and reproduces **§13.119's
minimal pair locally**: `constprop syms` **2 → 24** between `-O2` and
`-O2 -fipa-cp-clone` (Ubuntu measures ~27 in its build), object 537 104 → 549 024
bytes.  So the pair is real, not a same-file comparison.

**Answer to (i) — does GCC drop the low-tag mask?  NO.**  Whole-object census,
every body (not a name filter — at `-O2` most of the primitive is inlined into
functions whose names match nothing, so a filtered total answers a narrower
question than it appears to):

| | base `-O2` | clone `-O2 -fipa-cp-clone` |
|---|---|---|
| bodies | 523 | 526 |
| **`mask_ptr`** (`and ~7`) | **331** | **331** |
| **`mask_cnt`** (`and 7`) | **395** | **395** |
| `tagged_add` | 1307 | 1397 |
| **`cas`** | **575** | **610** |
| `rmw` | 1071 | 1110 |
| `fence` | 71 | 71 |

**Not one mask is lost** — 331 and 395 are identical across the arms even though
the clone arm has three more bodies, 90 more tagged adds and 35 more CAS.  The
premise was sound (IPA-CP propagating a zero refcount does make
`(uintptr_t)pref + 0` provably aligned, and gcc may then delete `& ~7`); it just
does not happen here.

**Answer to (ii)'s static half — was a `cmpxchg` omitted?  NO, the opposite.**
`cas` goes **575 → 610** and `rmw` **1071 → 1110**: the clone arm performs *more*
atomic operations, as duplicated paths should.  No compare-exchange disappeared.

**A predicate I had to withdraw mid-measurement.**  The census's "SUSPECT"
heuristic (builds a tagged value, never masks) is meaningful only under the name
filter.  With `--all` it fired on **268 of 523 bodies** — any function adding a
small constant to a pointer.  A number that large is not 268 findings, it is a
broken predicate, so the tool now refuses to print it under `--all` and says why
(§13.83's rule, applied to my own tool while using it).

**Caveat, and it is the real limit of this result.**  The target is the host arch:
these are **aarch64** bodies while the firing build is **x86-64**.  The
middle-end question — *was the operation kept at all* — travels across targets,
because it is decided before instruction selection; the instruction selection
does not.  So this refutes "the optimizer deleted a mask or a CAS" as a
*general* consequence of the pass, and does not exclude an x86-64-specific
selection bug.  Re-running the same two commands on Ubuntu is now trivial and
would close that gap.

**Queue after this:** §13.141's runtime gate is still the only arm that cannot be
confounded by codegen, and (iii) — hand-written clones — is still the only way to
reach the four caller-driven specializations.  What this section removes is two
hypotheses and, more usefully, the need to ask Linux for any further static
answer.
### 13.146 Chain traffic is a DOSE: the failure rate scales with how many pushes happen

§13.144 established the chain is behaviourally necessary with zero codegen
delta.  §13.141's runtime gate makes a further question askable for the first
time — not *whether* the chain runs but *how much*.  Added
`KAME_ORPHAN_CHAIN_KEEP=N` beside the gate: admit only every N-th
`orphan_chain_push`, same binary, same machine code, N chosen from the
environment.

**14 interleaved rounds, `20 40 700`, `taskset -c 0-3`, one `.so`:**

| chain traffic | failures |
|---|---|
| `KEEP=1` (every push — baseline) | **10/14 (71%)** |
| `KEEP=4` (¼ of pushes) | **4/14 (29%)** |
| `KEEP=16` | **1/14 (7%)** |
| `KEEP=64` | **0/13 (0%)** |
| `OFF` (no pushes) | **0/13 (0%)** |

**Monotone, and smooth: r = 0.95 against log(traffic).**  `KEEP=1` vs `KEEP=16`
is **p = 0.0013**; vs `KEEP=4`, p = 0.057 (the adjacent step is underpowered at
n = 14, the endpoints are not).

**This is a dose, not a trigger.**  A per-event defect — one bad push, one bad
adopt — would show a *threshold*: any traffic at all reproduces, and thinning
it merely lengthens the wait.  Instead the rate falls off with the amount of
chain activity in the same graded way §13.55 found for the clone set and
§13.88 for preemption frequency.  Three independent knobs — how many clones,
how often threads are preempted, and now how much chain traffic — all move this
fault's rate *continuously*.

**Which reframes the five clean mechanisms.**  §13.128, §13.131, §13.136,
§13.138 and §13.140 each removed one operation and found nothing, and that is
exactly what a dose-shaped fault predicts: no single operation is *the* defect,
so removing any one of them leaves the rate essentially unchanged, while
removing the traffic that drives all of them takes it to zero.  The five
negatives and the one positive are consistent, not contradictory.

**What it argues for.**  A window whose *width* is roughly fixed but whose
*frequency of exposure* scales with chain activity — i.e. the chain does not
contain the bug so much as it repeatedly creates the condition under which some
other, already-present race can be hit.  That is the shape §13.88's ~150 000
instruction window described, and it predicts the fault should also respond to
anything else that changes how often chunks change hands.

### 13.146 Proposal (ii) as an identity, not a comparison — and it closes: masks and fences conserved exactly, atomics gained

The correction: **if cloning creates N bodies, the totals must rise by what those
bodies contain.**  §13.145 read `mask_ptr 331 vs 331` as "not one mask lost",
which was the wrong test — identical totals are not evidence of health, they are a
coincidence that has to be explained.  The right test is arithmetic:

> `residual = (total delta) − (delta inside derived bodies)`

**First attempt gave a large negative residual**, i.e. apparent losses:

| op | base | clone | delta | in clones | residual |
|---|---|---|---|---|---|
| `mask_ptr` | 331 | 331 | +0 | 23 | **−23** |
| `cas` | 575 | 610 | +35 | 72 | **−37** |
| `rmw` | 1071 | 1110 | +39 | 124 | **−85** |

**Chased one of them to the bottom.**  The non-clone `bucket_release_chunk`
(clone-bisect arm 1) really does lose an atomic: base has
`casal`, `ldadd x19`, **`ldadd w1`** — the third being a **32-bit** add — and the
clone arm's non-clone body has only the first two.  `llvm-addr2line -i` on that
address:

```
std::__atomic_base<int>::fetch_add      atomic_base.h:631
  l1_base                               allocator.cpp:7689
  l1_push                               allocator.cpp:7742
```

and the clone arm contains **`l1_base() (.part.0)` carrying exactly one RMW**.
So the operation was **outlined by partial inlining, not omitted**.

**Which exposed a bug in my own tooling.**  `.part.N` is *partial inlining* and
`.isra.N` is *argument removal* — neither is an IPA-CP clone, and the parent still
exists.  Excluding them from the "non-clone" set (as `nonclone_memop_diff.py`
did) makes partial inlining read as an omission.  Fixed: only `.constprop` counts
as a clone, and the reconciliation counts all three kinds of derived body.

**With that, the identity closes** (real GCC 15.2, §13.119's minimal pair, all 523
/ 526 bodies):

| op | base | clone | delta | derived Δ | **residual** |
|---|---|---|---|---|---|
| `mask_ptr` | 331 | 331 | +0 | +0 | **0** |
| `mask_cnt` | 395 | 395 | +0 | +0 | **0** |
| `cas` | 575 | 610 | +35 | +6 | **+29** |
| `rmw` | 1071 | 1110 | +39 | +11 | **+28** |
| `fence` | 71 | 71 | +0 | +0 | **0** |

**Three of five classes reconcile to exactly zero** — which is the sign the
accounting is now right rather than fudged.  `cas` and `rmw` have a **positive**
residual: ordinary bodies *gained* ~29 compare-exchanges and ~28 other RMWs,
because each specialized call site inlines a callee that carries them.  Gaining
is not the failure mode; **losing** is, and nothing loses.

**So proposal (ii) closes as NO**, now on an identity rather than a comparison:
no mask, no fence, and no atomic RMW is dropped anywhere in the object when
`-fipa-cp-clone` is enabled.  Tools: `clone_op_reconcile.py` (the identity),
`tagmask_census.py --all` (the per-class totals), `nonclone_memop_diff.py` (which
body changed) — all three now address-keyed, since an object can carry two
symbols at one address (§13.132's `T`/`t`) and a name-keyed dict silently keeps
the wrong one.

**Errors this section had to correct, recorded because three of them were mine:**
reading identical totals as conservation (§13.145); a manual `awk` check that
swept `.constprop` bodies into a count of the parent and so "refuted" the tool
when the tool was right; and excluding `.part`/`.isra` from the non-clone set.
The user's framing — *clone count × operations must show up in the total* — is
what turned a vague comparison into a test that could close.

### 13.147 Proposal (i), the dynamic half: cloning changes neither the count nor the amount of chain work executed

§13.146 answers "does cloning grow the appropriate number of atomic ops?"
**statically** — masks and fences conserved exactly, atomics gained inside the
derived bodies, residual 0.  That census cannot see an atomic whose
*execution* a specialization elides: a propagated constant can make a branch
dead, leaving the instruction in some body while no longer running it.  So I
counted what actually runs.

**Instrument**: `KAME_CHAIN_DYNCOUNT` counts executed `orphan_chain_push` and
`orphan_chain_scrub` entries, printed at exit; both arms built from one source
with identical flags apart from `-fipa-cp-clone`.

**Successful runs only, `10 40 500`** (the arms must complete to report, and a
crashing run loses the `atexit` — the same gap as §13.93/§13.104/§13.128, now
the fourth time):

| arm | pushes | scrub visits |
|---|---|---|
| firing `-O2 -fipa-cp-clone` | 2227, 2231 | 5472, 5612 |
| non-firing `-O2` | 2154, 2159, 2155, 2201 | 5604, 5681, 5776, 5644 |

**Within ~3%, and overlapping.**  The firing arm does not execute more chain
work, nor less; cloning neither adds nor elides chain operations at run time.

**So (i) is answered in both halves and the answer is no.**  The atomic-op
count is appropriate statically (§13.146) and the executed count is
indistinguishable between arms (here).  Whatever `-fipa-cp-clone` does to make
this fault appear, it is **not** a change in how many atomic operations happen
on the chain.

**Which sharpens §13.146's dose result rather than competing with it.**  The
same amount of chain work — ~2200 pushes, ~5600 scrub visits — produces a 63-71%
failure rate in one arm and 0% in the other (§13.144), and thinning that work
scales the rate smoothly (§13.146).  Same operations, same counts, different
outcome: the difference has to be in *how* the work interleaves, not *how much*
of it there is.  That is consistent with §13.88's preemption dose-response and
with the fault surviving full serialization under rr (§13.86).

**Instrument gap, stated:** I wired counters for push and scrub only; the `pop`
and `unlink-CAS` slots exist but were never incremented, so their zeros in the
output are **not measurements**.  Anyone extending this should wire those two
before quoting them.

### 13.148 Correction to §13.147, and the right form of proposal (i): clones ADD atomics, and none is short of its parent

**The correction first.**  §13.147 concluded "cloning changes neither the count
nor the amount of chain work", which conflates two things and is wrong as
phrased.  What it measured was the number of `orphan_chain_push` /
`orphan_chain_scrub` **calls** — chain traffic, not atomic operations.  Those
being equal between arms says nothing about atomic-op counts, and "conserved"
is in any case the wrong expectation: **a clone that coexists with its parent
should make the total GROW, by exactly what the clone contains.**  The failure
mode worth looking for is a clone carrying *fewer* atomics than the parent it
was specialized from.

**Measured per family** (real GCC 15.2, `-O2 -fipa-cp-clone`, lock-prefixed
instructions + `cmpxchg`/`xchg` counted per body, clones grouped to their
parent):

* **24 cloned families**, parent body still present for **10** of them.
* **Zero families where a clone carries fewer atomics than its parent.**
* The `PoolAllocator<N>` ctors are 1 → 1 for most sizes, and **1 → 2** for
  N = 144, 288, 336 — the clone carrying *more* than the parent.
* The 14 families whose parent is gone (fully inlined) contribute their clones'
  atomics outright, including `global_pop_fit` (4) and `recycle_pop_fit` (5).

**So proposal (i) answers "yes, appropriately".**  Cloning grows the atomic-op
total by the clones' contents — roughly 29 lock-prefixed operations across the
clone bodies that would not otherwise exist — and no specialization drops an
atomic relative to the code it was derived from.  Together with §13.146's
static identity (masks and fences conserved exactly, residual 0) and §13.147's
dynamic call counts, the atomic-operation account is closed in all three
forms: statically conserved, per-clone non-decreasing, and dynamically equal.

**Process note.**  I built the object for this with `2>/dev/null` and analysed
an empty file, reporting "0 cloned families" from a build that had failed —
the exact mistake §13.119 recorded and warned about, repeated. The zero looked
like a result. Compiler stderr belongs in the log in any build-and-measure
loop, and the cheap guard is to check the artefact exists before parsing it.
### 13.149 Decomposing the chain's behaviour: push / adopt / scrub as three runtime arms in one binary

§13.144 settled that the chain's effect is **behavioural** (15/24 vs 0/24, zero
codegen delta), which retires the whole-TU-codegen reading of the ablation and
makes the behavioural question the only one left.  Five named mechanisms inside
the chain are clean — but **no half of it has ever been ablated as a behaviour.**
The gate controls `orphan_chain_push` only, so "chain on, adoption off" has not
been run.  §13.128's exoneration is by backstop counts and §13.131's by victim
location: both are statements about *mechanisms*, and a half can be necessary
while the mechanism nominated for it is innocent.

**Three acts, and they partition what the chain does:**

| env | what runs |
|---|---|
| `KAME_ORPHAN_NO_SCRUB=1` | push + adopt; orphans are never reclaimed |
| `KAME_ORPHAN_NO_ADOPT=1` | push + scrub; nothing is ever re-owned |
| both | **only `orphan_chain_push` itself** — the `refcnt` establishment (or self-ref move) and the Treiber CAS on the shared head |
| `KAME_ORPHAN_CHAIN_OFF=1` | none of it (§13.141) |
| `KAME_ORPHAN_CHAIN_KEEP=N` | dose: every N-th push (added with §13.144's run) |

The "both" row is the interesting one: if the fault needs only that, the acting
part is the **publication**, not the reuse — which no arm so far could say, and
which would explain why every reuse-side mechanism measures clean.

**Each arm is the same machine code**: the skipped call sits in the untaken
branch, the predicate is a cached `std::atomic<int>` read from the environment,
and every arm compiles under both clang and **real GCC 15.2** (verified here,
536 144 bytes).

**All five arms proven behaviourally live before use (§13.61)** — and the ladder
is coherent, which is the check that they mean what they say.  40 rounds × 8
threads, each thread leaving one live slot so its chunk is orphaned at exit:

| arm | `reserved` |
|---|---|
| baseline | 64 MiB |
| `NO_SCRUB=1` | 64 MiB — adoption is the recycler here, so losing scrub costs nothing |
| `NO_ADOPT=1` | **160 MiB** — scrub reclaims some, but nothing is re-owned |
| `CHAIN_OFF=1` | 320 MiB — nothing published, everything strands |
| `NO_ADOPT` + `NO_SCRUB` | **352 MiB** — worse than `CHAIN_OFF`, and correctly so: pushed chunks are pinned by the chain-ref, so they cannot even be released |
| `KEEP=8` | 288 MiB |

`NO_SCRUB` being indistinguishable from baseline in *memory* while
`NO_ADOPT` costs 96 MiB is itself informative: in this workload adoption does the
recycling and scrub is nearly idle — so if the fault survives `NO_ADOPT` it is
not about reuse volume at all.

**How to read the outcomes:**

| result | conclusion |
|---|---|
| dies with `NO_ADOPT`, lives with `NO_SCRUB` | adoption is behaviourally necessary despite §13.131's clean victim census — the search moves to *what adopt does* rather than *what it hands out* |
| dies with `NO_SCRUB`, lives with `NO_ADOPT` | the reclaim path is necessary despite §13.128's silent backstops |
| lives with both, dies only with `CHAIN_OFF` | **the publication itself** is the acting part: `refcnt.store(1)` / the self-ref move and the head CAS.  That is a two-statement suspect list |
| dies with both individually | the two interact, and `KEEP=N` gives the dose curve |

Run them interleaved in one job against the baseline, same `.so`.

### 13.150 The chain's own three functions are compiled IDENTICALLY in both arms — so the difference is around them, not in them

Given §13.144 (chain behaviour necessary, zero codegen delta), §13.146 (op counts
conserved statically), §13.147 (executed counts within 3 %) and §13.148 (no clone
short of its parent), the same operations run the same number of times and yet one
build fails 63–71 % and the other never.  What is left is **how** an operation is
compiled — so measure that where it matters most: the three chain functions
themselves.

Real GCC 15.2, §13.119's minimal pair, atomic positions given as instruction
index within the body (which is what sets the width of any window *inside* the
function):

| function | `-O2` | `-O2 -fipa-cp-clone` |
|---|---|---|
| `orphan_chain_push` | 112–114 insns; atomics @ 44, 72, 76, 85–88 | 113; **same positions** |
| `orphan_chain_pop` | 169; @ 37, 41, 86, 90, 105, 111 | 164; **same positions** |
| `orphan_chain_scrub` | 217; @ 35, 65, 70, 85, 118, 123, 164, 168, 189, 206 | 216; same but the last two shift by 2 (191, 207) |

22 bodies each, and the shape distribution is preserved (`pop`: 15/5/2 in both
arms).  The clone arm's only systematic change to `push` is to make every
instantiation **identical to the others** (112/114 → uniformly 113) — a
normalisation, not a widened gap.  `pop` loses 5 instructions, all *after* the
last atomic (epilogue).

**So the difference between the firing and non-firing builds is not in the
codegen of `orphan_chain_push`, `_pop` or `_scrub`.**  That is a real elimination
and it redirects the search: the chain's behaviour is necessary (§13.144) and the
chain's code is unchanged, so what differs must be code that runs **around or
concurrently with** the chain.

**And §13.122's non-clone diff already named that code.**  Run on this pair, the
bodies that gained atomics are the **allocation** paths:

```
PoolAllocator<N,true,true>::create_allocator()        atomic +4   (many N)
PoolAllocatorBase::allocate_dedicated_chunk(...)      atomic +4
PoolAllocatorBase::allocate_large_va(...)             atomic +4
PoolAllocatorBase::allocate_chunk<...>()              atomic +4
```

`create_allocator` is the function that **claims a chunk and constructs it**, and
it is the counterpart of the chain in the reuse cycle: the chain publishes and
re-owns chunks, `create_allocator` makes new ones.  A codegen change in the
claimer, racing a chain whose codegen did not change, fits every constraint
collected so far — same chain ops, same counts, chain necessary, and a difference
that only appears when both run.

**Next**, and it is two measurements rather than a hypothesis: (1) the same
positional census on `create_allocator` and `allocate_chunk` between the arms —
where did those +4 atomics land, and did anything move relative to the claim CAS;
(2) §13.149's decomposition, which says which chain act has to be present for the
fault, and therefore which pairing to look at.  Both are cheap, and (1) runs here
with no Linux round-trip (§13.145).
### 13.150 The behavioural decomposition: ADOPT is necessary, scrub is not

§13.147's three runtime arms make the decomposition possible for the first
time — the gate had only ever controlled `orphan_chain_push`, so "chain on,
adoption off" had never been run.  Its framing is the right one: a half can be
necessary while the mechanism nominated for it is innocent.

**11 interleaved rounds, one binary, `20 40 700`, `taskset -c 0-3`:**

| arm | what still runs | failures |
|---|---|---|
| `all` (baseline) | push + adopt + scrub | **10/11 (91%)** |
| **`KAME_ORPHAN_NO_ADOPT`** | push + scrub | **0/11 (0%)** |
| `KAME_ORPHAN_NO_SCRUB` | push + adopt | **8/11 (73%)** |
| `pushonly` (no adopt, no scrub) | push only | **0/11 (0%)** |
| `off` | nothing | **0/11 (0%)** |

`all` vs `noadopt`: **p = 3.4 × 10⁻⁵**.  `all` vs `noscrub`: **p = 0.59**.

**Adopt is the necessary half.**  Removing adoption alone takes a 91% failure
rate to zero, while removing scrub alone leaves it at 73% — statistically
indistinguishable from baseline.  And `pushonly` at 0/11 says the publication
itself is not sufficient: push establishes the refcount and does the Treiber
CAS on the head, and with adoption disabled that is harmless.  **It is the
re-acquisition of a published chunk that matters, not its publication.**

**This does not contradict §13.131, and the distinction is worth keeping
straight.**  §13.131 measured that no DOUBLE-LIVE hit lands in a chunk the
adopt census recorded, and concluded the adopt path "does not supply the
doubly-live slots".  That remains true as measured; what it cannot say — and
what §13.147 anticipated — is whether adoption is *necessary*, which is a
different question answered by a different experiment.  Both results stand: the
adopt path is required for the fault, and the specific slots that go doubly
live are not the ones its census records.  Reconciling those two is now the
sharpest question in the investigation, and it suggests the census is looking
at the wrong moment — recording the `BIT_OWNED` claim, when what matters may be
what the adopting thread does with the chunk *afterwards* (the DLL splice, the
owner re-arm, the first allocation out of a bitmap it did not build).

**Bounds.**  Eleven rounds; the adopt effect is far outside noise
(p = 3.4 × 10⁻⁵) but the scrub arm's 8/11 vs 10/11 is only "not different", not
"identical" — a real but small scrub contribution would not be visible at this n.

### 13.152 Walking the asm against the source, as asked — and the release-store lead it produced is refuted by the middle-end dump

The user's assessment is right and worth stating plainly: **there is no strong
suspect.**  What exists is a layer (the allocator's TU under gcc
`-fipa-cp-clone`), a necessary behaviour (the orphan chain), and a short list of
functions whose codegen changed.  With the list short, walking the assembly
against the C++ is the remaining method, so that is what this section does.

**Step 1 — where did `create_allocator`'s +4 atomics come from?**
`llvm-addr2line -i` on each atomic in `create_allocator<64u,true,true>`:

| arm | atomics and their inline chains |
|---|---|
| `-O2` | **one**: `stlr` ← `local_shared_ptr()` `atomic_smart_ptr.h:962` ← `atomic_shared_ptr()` `:1265` ← `PoolAllocator::PoolAllocator` `allocator.cpp:1303` |
| `-O2 -fipa-cp-clone` | **four**, all from `global_pop_fit` (`:7786/7790/7794/7796`) ← `recycle_pop_fit:7806` ← `large_recycle_pop:7987` — and **no `stlr`** |

So the firing arm inlines the large-recycle pop path into `create_allocator` and
outlines the `PoolAllocator` constructor.  Note **`global_pop_fit` is one of the
four caller-driven specializations §13.120 proved unreachable by per-function
licensing, and the clone §13.53 found perfectly correlated with the fault across
all six arms** — here it is, inlined into the claimer, in the firing arm only.

**Step 2 — the release-store asymmetry, which the earlier censuses could not see.**
`tagmask_census.py` counts RMWs and fences but **not release stores** (`stlr` is
neither), so §13.146's "everything conserved" never covered them.  Counted:

| | base | clone | derived Δ | residual |
|---|---|---|---|---|
| `stlr` | **52** | **36** | +0 | **−16** |
| `ldar` | 951 | 951 | +0 | 0 |
| `dmb` | 71 | 71 | — | 0 |

Sixteen release stores absent, none moved into a derived body, acquires and
barriers exactly conserved.  Concentrated in `create_allocator()` (1 → 0, many
size classes) and `allocate_chunk<PoolAllocator<N,true,true>>` (2 → 0, N =
272…368).  In `base` the run of member zero-initialisations ends with
`stlr xzr, [x19+0x120]`; in the firing arm that run is not in the body at all,
because the constructor is outlined — and the outlined copy uses plain stores.

**Step 3 — and that lead is refuted, by the one measurement that transfers.**
`stlr` is aarch64; on **x86-64 a release store is a plain `mov` and no such
instruction exists**, so this signal cannot even be observed on the firing
target.  The transferable question is what the *middle end* decided, so compare
`-fdump-tree-optimized`:

| memory order | base | clone |
|---|---|---|
| relaxed | 318 | 311 |
| **release** | **151** | **151** |
| seq_cst | 146 | 143 |

**Release stores are exactly conserved (151 = 151).**  Ten atomic stores do
disappear — seven relaxed, three seq_cst — consistent with dead-path elimination
under specialization, and **not one of them is a release.**  So no release
ordering was weakened or dropped; the `stlr` delta is instruction selection and
inlining redistributing what survives to the back end.

**Recorded because I was one step from reporting it as the finding.**  An
aarch64-only instruction count, on a machine that is not the firing target,
produced a clean-looking −16 with a plausible mechanism attached (a chunk
published without a release on its chain pointers, observable by the only reader
of those pointers — the chain, which §13.144 proved necessary).  The GIMPLE dump
took two minutes and refuted it.  **Instruction counts on a non-target
architecture can manufacture a lead; the middle-end dump is what transfers.**

**Standing after this:** the ordering question is now answered NO at both levels
that can answer it, and the asm-vs-source walk has covered the chain's three
functions (§13.150, identical) and the claimer (`create_allocator`, here).  What
the walk *did* turn up is worth keeping: **`global_pop_fit` is inlined into
`create_allocator` in the firing arm only** — the same function §13.53 found
perfectly correlated and §13.120 showed no attribute-based arm can reach.  That
is where the next reading should go, and unlike everything else in this section it
is a difference in *what code runs inside the claimer*, not in how an operation is
encoded.

### 13.152 The same inlining on x86-64 — and it lands in the half §13.150 proved necessary

§13.151's step 1 finds `global_pop_fit` inlined into `create_allocator` in the
firing arm only, with the claimer's atomics going 1 → 4.  That walk is arm64,
where the fault never reproduces; the check that matters is whether the same
thing happens on the machine where it does.  It does.

**x86-64, GCC 15.2, `-O2` vs `-O2 -fipa-cp-clone`, otherwise identical flags:**

| arm | `create_allocator<64u,true,true>` | instructions | **atomic RMWs** |
|---|---|---|---|
| non-firing `-O2` | — | 111 | **0** |
| **firing `-O2 -fipa-cp-clone`** | — | 181 | **4** |

Neither body *calls* `pop_fit`/`recycle_pop_fit`/`large_recycle_pop`, so the
firing arm's extra ~70 instructions and **four atomic read-modify-writes are
inlined into the claimer**, exactly as §13.151 measured on arm64 (1 → 4 there,
0 → 4 here — the difference in the base count being the `stlr`-vs-`lock`
counting convention, not a structural one).

**Why this is worth more than the other codegen deltas.**  `create_allocator`
is the claim path — the function reached when a thread needs a chunk and, via
`large_recycle_pop` / the orphan pop, the one that **adopts**.  §13.150 measured
adoption to be the necessary half: `KAME_ORPHAN_NO_ADOPT` takes 10/11 to 0/11
while `NO_SCRUB` leaves 8/11.  So for the first time the **codegen difference
and the behavioural necessity name the same code**: the pass inlines the
recycle path into the claimer, adding four atomics there, and the claimer's
adoption is what the fault requires.

**And it is the function that keeps recurring.**  `global_pop_fit` is one of the
four caller-driven specializations §13.120 proved unreachable by per-function
licensing, and the clone §13.53 found perfectly correlated with the fault across
all six arms — correlated, then untestable additively (§13.117), then
individually innocent when suppressed (§13.122: 12/22 vs 12/21).  Those results
stand: suppressing its *cloning* changes nothing.  What is new here is that in
the firing arm it is not a separate clone being called at all — it is **inlined
into the claimer**, which is a different object from the one those suppression
arms manipulated.

**What this does not yet show.**  That the four inlined atomics are wrong, or
that their placement differs semantically from the out-of-line version.  The
next step is a direct read of those four sites against the source they came
from — on x86-64, where comdat sections make `addr2line` offsets ambiguous, so
it needs `-ffunction-sections` or a per-section disassembly to attribute them
reliably.

### 13.153 STATUS — the Linux side, as of §13.152

Everything below was measured on the firing machine (Ubuntu, x86-64, GCC 15.2,
pool ACTIVE — `kame_pool_reserved_bytes()` = 33 554 432, verified per §13.109's
rule). Interleaved arms in one job unless stated.

**What is established, and how strongly.**

| finding | evidence | strength |
|---|---|---|
| The orphan chain is **behaviourally** necessary | runtime gate, one binary, zero codegen delta: 15/24 vs 0/24 | p = 2.4 × 10⁻⁶ |
| **Adoption** is the necessary half; publication is not | `NO_ADOPT` 0/11 vs `all` 10/11; `pushonly` 0/11 | p = 3.4 × 10⁻⁵ |
| Chain traffic is a **dose**, not a trigger | KEEP=1/4/16/64 → 10/14, 4/14, 1/14, 0/13 | r = 0.95 vs log traffic |
| The firing arm **inlines the recycle path into the claimer** | `create_allocator<64u>` 111→181 insns, 0→4 atomics | static, both arches |
| A slot is handed out while its previous occupant is **live and unfreed** | DOUBLE-LIVE on 5/5 failing runs, 0 on clean; `W0` = live refcnt, no poison | dtor-exact live-set |
| Hits cluster **within one chunk** | 2 of 3 multi-hit runs same 256 KiB chunk, one pair 224 B apart | pattern, n small |

**What is refuted, on this machine, each with a live control.**

`m_owner_id` handoff (18/20 vs 13/20, p = 0.13) · claim-side ordering + pre-CAS
metadata (20/20 vs 20/20) · `global_pop_fit`'s clone, both additively
(untestable — §13.117) and subtractively (12/22 vs 12/21) · `acquire_tag_ref_`'s
22 clones (13/22 vs 13/22) · the refcount primitive at `no-ipa-cp-clone`
(22/30 vs 26/30) · dispose backstops (0 firings on 7 failing runs) · adopt
supplying the doubly-live slots (0/17 hits in adopted chunks, census live at
101 adopts/run) · `back_offset` corruption (0 across 8 failing runs, poke
control CAUGHT) · the cross-thread TLS poke (10/18 vs 11/18) · TSan-visible
allocator races (both fixed, neither moved the rate) · ASan (suppresses the
fault entirely, 0/17).

**The one tension worth carrying forward.**  Adoption is *necessary* (§13.150)
yet no doubly-live slot lands in a chunk the adopt census records (§13.131).
Both are measured with live controls. The likeliest reconciliation is that the
census records the wrong moment — the `BIT_OWNED` claim — when what matters is
what the adopting thread does afterwards: the DLL splice, the owner re-arm, or
its first allocation out of a bitmap it did not build.

**Where the two lines now meet.**  For the first time the codegen difference and
the behavioural requirement name the same code: the pass inlines
`global_pop_fit` into `create_allocator`, and `create_allocator` is the adopter.
Every earlier suppression manipulated that function as a *separate clone*, which
is not the object the firing arm actually executes.

**Immediate next step (mine).**  Read the four inlined atomic sites in
`create_allocator` against their source. On x86-64 comdat sections make
`addr2line` offsets ambiguous — my first attempt resolved into `__tls_init` —
so this needs `-ffunction-sections` or per-section disassembly to attribute
reliably.

**Standing methodological notes** (each cost a measurement here): prove a
detector fires before trusting its zero; measure a predicate's base rate before
trusting its hit; `atexit` does not run after a fatal signal, so any counter
that matters on a failing run needs a signal-backed readback (cost four
measurements); never send compiler stderr to `/dev/null` in a build-and-measure
loop (cost two); run duration is not a validity check — it tracks machine load.

### 13.154 asm-vs-source on x86-64: the inlined `global_pop_fit` is a faithful translation, size verify intact

§13.152 located the difference (`create_allocator<64u>`: 0 → 4 atomics, the
recycle path inlined in the firing arm only) but could not attribute the sites,
because comdat sections make `addr2line` offsets ambiguous in a relocatable
object — my first attempt resolved into `__tls_init`.  Linking to a `.so` gives
unique addresses and fixes it.

**All four atomics attribute to `global_pop_fit`**, inlined through
`recycle_pop_fit:7901` → `large_recycle_pop:8082` → `allocate_chunk:2666`:

| asm | source |
|---|---|
| `lock cmpxchg %rbp,(%rdx)` | `:7881` take-CAS — `slots[idx].compare_exchange_weak(b, nullptr, acq_rel)` |
| `lock sub %rsi,…(%rip)` | `:7885` `g_lrc_bytes.fetch_sub(sz, relaxed)` |
| `lock cmpxchg %rdi,(%rdx)` | `:7889` put-back-CAS — `compare_exchange_weak(exp, b, acq_rel)` |
| `lock add %rsi,…(%rip)` | `:7891` `g_lrc_bytes.fetch_add(sz, relaxed)` |

**The safety-critical check survives, and is correct.**  The C++ verifies the
popped block is big enough (`if(sz >= need) return b;`) between the fetch_sub
and the put-back.  In the inlined firing copy:

```asm
35187:  lock sub %rsi,0x48a99(%rip)   ; g_lrc_bytes.fetch_sub(sz)      :7885
3518f:  cmp    $0x3ffff,%rsi          ; sz >= need, need folded to 0x40000
35196:  ja     350d2                  ;   -> return b   (VERIFY, taken when sz >= 256 KiB)
3519c:  mov    %rbp,%rax
3519f:  lock cmpxchg %rdi,(%rdx)      ; put back                        :7889
351a4:  jne    3521b                  ;   -> lrc_release path
351a6:  lock add %rsi,0x48a7a(%rip)   ; g_lrc_bytes.fetch_add(sz)       :7891
351ae:  add    $0x1,%ecx              ; ++kk
351b1:  cmp    %r10d,%ecx
351b4:  jne    35160                  ; loop
```

`need` is constant-propagated to `0x40000` (256 KiB — the chunk size for this
instantiation), so `sz >= 0x40000` becomes `cmp $0x3ffff / ja`.  That is the
**correct** encoding of the same predicate, not an elision.  Both CASes, both
counter updates, the verify, the `lrc_release` fall-through and the bounded
`LRC_K_MAX` loop are all present and in source order.

**So the suspicious routine is not miscompiled.**  The one function that the
codegen difference and the behavioural requirement both point at (§13.152) is a
faithful translation of its source, with its size check intact.  Together with
§13.151's refutation of the release-store lead on arm64, the "the pass broke
`global_pop_fit`" reading is now closed on both architectures.

**What that leaves.**  The inlining is real and only in the firing arm, but it
changes *where* this code runs, not *what it computes* — four atomic RMWs now
execute inside the claimer's frame rather than behind a call.  On the dose
evidence (§13.146) and the adopt requirement (§13.150), the remaining
hypothesis is not a wrong instruction but a wrong *interleaving*: the same
correct sequence, executed with a different window relative to the claim it is
now inlined into.
### 13.155 Static analysis is exhausted on these paths — every difference is inlining relocation; plus an adoption dose knob and a build fix

§13.150 makes adoption the necessary half, so the asm-vs-source walk continued
into the adopt path.  Three differences were found and **all three closed as
relocation**, with the transferable level conserved each time:

| difference found | how it closed |
|---|---|
| `create_allocator` +4 atomics, `stlr` gone (§13.152) | GIMPLE release stores **151 = 151**; the ten atomic stores that do vanish are 7 relaxed + 3 seq_cst, none release |
| `allocate_chunk_path`: 4 bodies lose a **`dmb`** and 64 instructions (344 → 280) | whole-object `dmb` **71 = 71**; the fences move into `allocate_chunk<PoolAllocator<N,false,false>>` (+2 each, N = 32/64/256/1024) — and GIMPLE thread fences are **81 = 81**, all release |
| the same 4 bodies lose **9 plain loads after the claim CAS** (36 → 27) | the *base* arm already has one instantiation at 27, and base never fails; the loads left with the outlined callee |

The third is worth a note because it is the shape the investigation has circled
since the start — the user's hypothesis A, a value cached across a
synchronisation point instead of re-read.  The firing arm has **fewer** post-CAS
re-reads (36 → 27 in four instantiations, −36 across the family), which is the
right direction for that hypothesis.  It does not survive: `-O2` already emits a
27-load form for one instantiation and `-O2` never fails, so 27 post-CAS loads is
not by itself the fault.

**So static analysis has said what it can.**  At the level that transfers between
targets — GIMPLE memory orders, fence counts, atomic-operation accounting,
executed counts (§13.147), per-clone reconciliation (§13.148) — the two builds are
**equivalent**, and every instruction-level difference resolves to inlining moving
code between bodies.  What remains is scheduling, register allocation and timing,
which no static count can adjudicate.

**Which makes the dose curve the instrument, and it was missing on the half that
matters.**  The existing throttle (`KAME_ORPHAN_CHAIN_KEEP`) thins **pushes**, and
§13.150 showed push-only is not sufficient (0/11).  Added
**`KAME_ORPHAN_ADOPT_KEEP=N`** — admit only every N-th adoption — so the curve can
be taken on adoption itself.  Verified live here (`KEEP=1` → 64 MiB reserved,
`KEEP≥2` → 128 MiB; it saturates fast in this synthetic probe, which the
reproducer will not).  What the shape would say:

- **smooth in 1/N** → a probabilistic window; each adoption carries the same small
  chance, which is §13.55's graded dose-response and §13.88's wide window;
- **threshold** → state accumulates across adoptions until something tips;
- **flat until N is huge** → one adoption suffices and the rate is limited
  elsewhere.

**Build fix, and it is the same class as the conflict markers in `7899e2210`.**
`KAME_CHAIN_CNT` (§13.147's counters) is *defined* inside the
`KAME_ORPHAN_CHAIN_RUNTIME_GATE` region but *used* unconditionally, so the
**default build of `allocator.cpp` did not compile** — "use of undeclared
identifier `KAME_CHAIN_CNT`" at both use sites.  Fixed with a fallback
`#ifndef` rather than moving the block, so the gated build keeps what it has.
Verified: the plain build, the gate, `KAME_CHAIN_DYNCOUNT`,
`KAME_NO_ORPHAN_CHAIN`, and gate+dyncount together all compile.  A diagnostic
that only compiles under its own flag breaks everyone else silently, and a
build-and-measure loop then reads stale objects as results — twice now.

### 13.156 An interleaving probe for the adopt sequence — because everything statically checkable is now correct

Where this stands after §13.154: the chain's three functions compile identically
(§13.150), release stores and fences are conserved at GIMPLE (§13.152, §13.155),
the recycle path the firing arm inlines into the claimer is a **faithful
translation with its size verify intact** (§13.154), operation counts reconcile
statically (§13.146) and dynamically (§13.147), and no clone is short of its
parent (§13.148).  **Everything statically checkable is correct** — and adoption
is still the necessary half (§13.150, 0/11 vs 10/11).

So the remaining variable is interleaving, and static censuses cannot adjudicate
it.  The way to locate an interleaving window is to **widen it at a chosen point**
and watch the rate move.

**`KAME_ADOPT_YIELD_AT=N`** delays inside the adopt sequence, `KAME_ADOPT_YIELD_US`
sets the delay (default 1 µs — and `sched_yield()` is *not* a perturbation when
cores are idle, §13.103b: 21.16 s → 21.51 s, while `usleep(1)` gave 36.07 s):

| site | position |
|---|---|
| 1 | after `orphan_chain_pop()`, **before** the `BIT_OWNED` claim CAS |
| 2 | after the claim CAS, before re-arming owner metadata |
| 3 | after the owner metadata, before the DLL splice |
| 4 | after the DLL splice **and** the self-ref move — adoption complete |

A site whose delay **raises** the rate is where a peer needs time to reach the
other side of the race; one that **suppresses** it has serialised the pair.
Either direction localises.

**All four proven live (§13.61), same binary, env only:**

```
AT=0 (off)   0.05 s
AT=1         2.70 s
AT=2/3/4     0.87 s
```

**And the asymmetry matters for reading the results**, so it is recorded rather
than smoothed over: site 1 sits before the claim and therefore fires on **every
pop attempt, including those that return null**, while 2–4 are inside
`if(claimed)` and fire only on **successful adoptions**.  Site 1's delay is
roughly 3× the dose of the others in this probe — so a larger effect at site 1 is
not by itself evidence that its position matters more.  Compare 2 against 3
against 4 for position, and use site 1 only against itself at different `US`.

**A build break I introduced and then had to fix — the same one I had just
criticised.**  §13.155 fixed `KAME_CHAIN_CNT` being *defined* inside the
runtime-gate region while *used* outside it, breaking the default build.  My first
version of this injector did exactly that: the four call sites sit outside the
gate, the declaration inside.  Fixed with a no-op `KAME_ADOPT_YIELD` fallback, and
all five flag combinations verified (plain, gate, `KAME_CHAIN_DYNCOUNT`,
`KAME_NO_ORPHAN_CHAIN`, gate + dyncount), plus a real GCC 15.2 build.  Recorded
because writing the lesson down one commit earlier did not stop me repeating it;
the guard belongs in the *call-site macro*, not in discipline.
### 13.157 Placement is not it either: forcing `global_pop_fit` out of line does not suppress the fault

§13.154 closed "the pass miscompiled `global_pop_fit`" and left the narrower
reading: the inlining changes *where* the four atomics run — inside the
claimer's frame instead of behind a call — so the suspect became the
interleaving, not the instructions.  That is directly testable by removing the
inlining while keeping the specialization.

**`KAME_NO_INLINE_POPFIT` isolates exactly that variable:**

| arm | `create_allocator<64u>` size | atomics inside | `pop_fit` calls |
|---|---|---|---|
| base (firing, inlined) | 743 | **4** | 0 |
| **`noinline`** | 747 | **1** | **1** |

Three of the four atomic RMWs move back out of the claimer's frame; everything
else — the clone set, the constants, the size verify — is unchanged.

**Result, 20 interleaved rounds, `20 40 700`:**

| arm | failures |
|---|---|
| inlined (baseline) | **8/20 (40%)** |
| noinline | **13/20 (65%)** |

Fisher **p = 0.20** — no suppression, and the arm without the inlining fails
*more*, not less.

**So the placement reading is refuted too.**  The inlining is real and
firing-arm-only (§13.152), the body it inlines is correct (§13.154), and
removing the inlining does not help.  `global_pop_fit` is now exhausted from
every angle available: cloned or not (§13.122), licensed or not (§13.117),
inlined or not (here), and correct as compiled (§13.154).

**A flag-sensitivity worth recording.**  The same two arms built *with*
`-DKAME_POISON_FORENSIC` show **0** atomics inside `create_allocator` in both —
the inlining decision disappears entirely.  I built the first version of this
experiment that way and got two identical-looking arms; the inlining that
§13.152 and §13.154 documented exists only under the flag set used there.  Any
future arm touching this function must re-check the inlining is present in its
own build before drawing a conclusion, exactly as the clone arms had to
re-check `.text` (§13.117).

**Standing.**  The chain is necessary (§13.144), adoption is the necessary half
(§13.150), traffic is a dose (§13.146) — and every mechanism nominated inside
that half has now been measured innocent.

### 13.158 The adopt-sequence yield probe: no site suppresses, and one may aggravate

Ran §13.156's interleaving probe on the firing machine.  `KAME_ADOPT_YIELD_AT=N`
delays at one point of the adopt sequence; all arms are one binary, chosen from
the environment, `sched_yield`-based per §13.103b.

**14 interleaved rounds, `20 40 700`, `taskset -c 0-3`:**

| arm | delay point | failures |
|---|---|---|
| `none` (baseline) | — | **6/14 (43%)** |
| `AT=1` | after `orphan_chain_pop`, before the claim CAS | 8/14 (57%) |
| `AT=2` | after the claim CAS, before re-arming owner metadata | 8/14 (57%) |
| `AT=3` | after owner metadata, before the DLL splice | 5/13 (38%) |
| `AT=4` | (fourth site) | **9/13 (69%)** |

Against baseline: `AT=1` p = 0.71, `AT=2` p = 0.71, `AT=3` p = 1.00,
`AT=4` p = 0.26.  **No arm suppresses**, and none reaches significance.

**What that rules out.**  If the fault were a race with a *narrow* window inside
the adopt sequence, widening that window at the right point should have made it
markedly more likely, and widening it at the wrong point should have left it
alone — a clear peak.  There is no peak.  The largest excursion (`AT=4`,
69% vs 43%) is not significant at n = 13 and would need its own run to claim.

**Read with §13.157**, which showed the same for placement (removing the
inlining moved 4 atomics out of the claimer and changed nothing, 8/20 vs 13/20),
this is the second interleaving-shaped hypothesis to come back flat.  Delaying
*within* the adopt sequence does not modulate the fault, and neither does moving
its code.

**What still stands.**  Adoption is required (§13.150, 0/11 vs 10/11) and chain
traffic is a dose (§13.146, r = 0.95).  Those two together now look less like
"a race inside adopt" and more like "adoption is the event that supplies
something the fault consumes" — the rate tracks how often adoption happens, not
how the individual adoption is scheduled.

**A caveat on this instrument specifically.**  `sched_yield` on idle cores is
close to a no-op (§13.103b measured 21.16 s → 21.51 s), and these runs are on an
idle 4-core box, so the arms may simply not be inserting much delay.  Re-running
with `KAME_ADOPT_YIELD_US` set high enough to be a real perturbation would
distinguish "no window here" from "no delay applied"; as it stands this is
weaker evidence than the flat numbers suggest.

### 13.159 The OTHER suppressor: cross-thread batch `cap = 1` is 0/16 and has never been reconciled with the chain

Asked directly, and it is a real gap in the record.  §6's table has

| experiment | result |
|---|---|
| cross-thread batch **`cap = 1`** | **0/16, 0/12** |
| batching kept, per-slot flush (no sort/merge/CAS-merge) | 5/16 — protocol innocent |

so **deferred batching is a second, independent ablation that takes the rate to
zero** — and in 150 sections it is cited only three times (the table, §13.112's
arm list, §13.116's reasoning).  Nobody has asked the obvious question: *we now
have two ablations that each give 0; are they one mechanism or two?*  That is a
strong constraint, because any single mechanism must require **both** deferred
batching and adoption.

**The obvious reconciliation is closed by the code.**  I proposed it in §13.116
and dismissed it for FS=true only; checking FS=false too, the comment is
explicit:

> FS=false `OnClearFn`: "We **DO NOT** release the chunk on the
> dec-to-0-with-`BIT_OWNED`-clear case: such a chunk is an ORPHAN on the
> `atomic_shared_ptr` chain (its chain-ref keeps it mapped), reclaimed by
> `orphan_chain_scrub` (unlink → refcnt 0 → dispose) once drained — **not freed
> here**.  The return value is intentionally ignored."

Both FS arms ignore the `atomicDecAndTest` result, so a batch flush can never
release an on-chain chunk.  **Stale comment flagged**, because a reader will
build the wrong model from it: `owner_release` still says "BIT_OWNED is now clear
so the cross-thread releaser's subsequent `atomicDecAndTest` will bring the word
to 0 and **identify itself as releaser**" — no such releaser exists any more.

**What the two suppressors need is to be run against each other, and now they
can be, in one binary.**  `cap` is already a runtime field (`kame_pool_set_
realtime_thread(2)` sets it to 1) but that call carries the rest of the §75
realtime policy, so using it as the batch knob would confound exactly this
comparison.  Added **`KAME_BATCH_CAP=N`** instead — the flush threshold alone.
With `KAME_ORPHAN_NO_ADOPT` / `KAME_ORPHAN_ADOPT_KEEP` already present, the 2×2
is:

| | batch default | `KAME_BATCH_CAP=1` |
|---|---|---|
| adopt on | baseline (fires) | §6 says 0 |
| `NO_ADOPT` | §13.150 says 0 | — |

**One binary, four cells, zero codegen delta.**  If the fault needs both, the
suspect is a *pair* of operations and not a function — which would explain why
every single-function audit has come back clean.  If either alone suppresses at
partial dose (`ADOPT_KEEP=N`, `BATCH_CAP=N`), the dose curves say which is
upstream.

**The knob reports its own liveness, because my first attempt to prove it failed.**
A cross-thread free bench gave 104 / 98 / 99 / 106 M free/s for cap unset / 1 / 8
/ 1024 — all noise.  The reason is instructive: `push_direct` reads
`m_last_coalesce_x16` and routes to **direct** rather than **hold** when
coalescing looks unprofitable, so `cap` never entered the picture and my bench
could not have shown anything.  Rather than hunt for a workload that chooses
HOLD, the batch now counts its own work and prints it:

```
CAP=unset(1024)  pushes=960  flushes=0     <- threshold never reached
CAP=1            pushes=960  flushes=320   <- knob demonstrably live
CAP=8            pushes=960  flushes=0     <- ~3 pushes per thread, never reaches 8
```

So every run states whether its cap arm did anything: `flushes == 0` under a
non-1 cap means that arm was **vacuous**, not negative.  That is §13.61 built
into the instrument instead of left to the reader — and it is what my own bench
would have needed to be trustworthy.

### 13.160 Is the batch flush an algorithmic hole?  No — but it defends against a hazard that no longer exists, and three comments still describe it

Read `flush()`, both `batch_return_to_bitmap` overloads and `batch_clear_impl`
with that question.

**The one structural hazard the algorithm has, and why it is closed.**  A flush
sorts by `(chunk, slot)` and hands each chunk's run to
`batch_return_to_bitmap`, which consumes the run and returns the count.  If a
*middle* entry of a run brought `MASK_CNT` to zero, the remaining entries would
be processed against a reclaimable chunk.  It cannot: the run is slot-ascending,
hence **word**-ascending, and `MASK_CNT` counts non-empty words — so it reaches
zero only when the last non-empty word empties, and any remaining entry implies a
non-empty word remains.  The sort order is load-bearing for that, which is worth
knowing before anyone "optimises" it away.

**No release path exists in the batch chain at all.**  Neither
`batch_return_to_bitmap` overload, nor `batch_clear_impl`, calls
`deallocate_chunk`, `bucket_release_chunk`, `owner_release`, a destructor or
`dispose` — the whole inventory is bitmap CASes plus `mask_fn` / `on_clear`, and
both `on_clear` functors say in as many words that they do **not** release an
orphan, deferring to `orphan_chain_scrub`.

**Three comments still describe the removed releaser**, and they are not
cosmetic:

| location | claim | status |
|---|---|---|
| `flush()` | "the call **may release the chunk** on last-slot return + owner-exit, after which `chunk` is a stale pointer" | cannot happen |
| `push_direct` | "batch_return releases the chunk: the placement-new destructor runs, and `c` becomes a stale pointer" | cannot happen |
| `owner_release` | "the cross-thread releaser's subsequent `atomicDecAndTest` will bring the word to 0 and **identify itself as releaser**" | no such releaser |

**The cost of that is not tidiness — the current safety argument is nowhere
written down, so readers reconstruct the removed one.**  I did it twice:
§13.116 nominated "the batch-mediated releaser" and dismissed it on the FS=true
comment; §13.157 dismissed it again on the FS=false comment.  Both times I was
arguing about a mechanism that exists in neither direction.

**And a correction to my own proposed fix, which is the interesting part.**  Since
the call cannot release the chunk, the `[load, deref]` window §13.133 found — the
force-walk pointer loaded *before* `batch_return_to_bitmap` and dereferenced
*after* — looked removable by simply loading it after the call.  **It is not.**
The flush clears bits; clearing the last one takes `MASK_CNT` to zero and makes
the chunk **eligible for reclaim by a concurrent scrub** — so a post-call
`chunk->m_owner_dll_force_walk_ptr` is a dangling *chunk* dereference instead of a
dangling *TLS* one.  The original hoist therefore has a real justification; just
not the one its comment gives.

> **The window is not fixable by reordering: load-before risks the owner's TLS,
> load-after risks the chunk.**

Which settles what to do with it.  §13.138 measured the poke as **rate-neutral**
(10/18 with it, 11/18 without), and the flag it sets is documented as a hint with
"one-cycle false-negative delay acceptable".  So the resolution is not to reorder
or to pin anything: **remove the poke**, which is what
`KAME_NO_XTHREAD_FORCEWALK_POKE` already does, at the cost of a delayed DLL walk
and nothing else.  Recommending it as the default rather than doing it, since
that is a production behaviour change and not mine to take unilaterally.

**Answer to the question, then:** the flush is not an algorithmic hole — its one
structural hazard is closed by the sort order, and the chunk-liveness invariant
(§13.116) holds throughout.  What it does carry is a defence against a removed
hazard, whose only remaining effect is a dangling dereference that cannot be
reordered away and does not need to exist.
### 13.161 The yield probe with a REAL window: site 4 (adoption complete) looked like a suppressor — see §13.162, it is not

§13.158 reported the probe flat and flagged that `sched_yield` on idle cores
might not be inserting a real delay.  That caveat was right, and the instrument
check settles it: with `KAME_ADOPT_YIELD_US=50000` a **successful** run goes
**3.5 s → 5.41 s**, i.e. ~38 adopt events × 50 ms — the delay executes, and the
default (1 µs) simply produced a window too narrow to matter.  Runtime alone
could not show this because failing runs die early; only timing *successful*
runs separates the two.

**Re-run at `KAME_ADOPT_YIELD_US=500`** (a real 500 µs per-event window, ~20 ms
total, invisible in run time).  16 interleaved rounds, `20 40 700`:

| arm | delay point | failures |
|---|---|---|
| `none` | — | **11/16 (69%)** |
| `AT=1` | after pop, before claim CAS | 9/16 (56%) |
| `AT=2` | after claim CAS, before owner re-arm | 11/16 (69%) |
| `AT=3` | after owner metadata, before DLL splice | **12/16 (75%)** |
| **`AT=4`** | **after the DLL splice and self-ref move (adoption complete)** | **6/16 (38%)** |

`AT=3` vs `AT=4`: **p = 0.073**.  `none` vs `AT=4`: p = 0.156.  **Neither
clears p < 0.05.**  (An earlier draft of this section quoted p = 0.032 for the
3-vs-4 contrast; that was computed from the round-15 snapshot, 12/14 vs 5/14,
before the run finished.  Reading a p-value off a partial batch is the same
error as §13.40's partial log, and it is corrected here.)

**This is the first non-flat result from an interleaving probe.**  By §13.156's
own reading — "a point whose delay SUPPRESSES has serialised the pair" —
delaying *after adoption completes* halves the rate, while delaying anywhere
inside the sequence does not, and `AT=3` (the last point still inside) is the
highest arm.  The contrast between the adjacent sites 3 and 4 is the
significant comparison; each against baseline is not, at n = 16.

**What it suggests.**  The exposure is not between the pop and the splice — it
is between *completing* an adoption and whatever the adopting thread does next.
Holding the thread at that boundary serialises it against a peer, which fits
§13.150 (adoption required), §13.146 (traffic is a dose) and §13.131 (the
doubly-live slots are not the ones the claim records) at once: what matters is
the adoption having happened, and the race is with the thread's *subsequent*
use of the chunk — its first allocation out of a bitmap it did not build.

**Bounds, stated.**  Sixteen rounds, and **nothing reaches significance** — the
3-vs-4 contrast is p = 0.073 and is in any case a comparison chosen after
seeing the data.  What is suggestive is the *shape*: a monotone rise across
sites 1-3 and a drop at 4, which is not what noise usually looks like, but a
shape is not a result.  That run — `none` vs
`AT=4` only, more rounds — is the obvious next measurement and is cheap.

### 13.162 §13.161 does not replicate: the yield probe is flat at every site, and the site-4 "suppressor" was noise

§13.161 ended by naming its own follow-up — "`none` vs `AT=4` only, more
rounds" — and flagged that the contrast it leaned on was chosen after seeing
the data.  That run is done, and it goes the other way.

**40 interleaved rounds, `KAME_ADOPT_YIELD_US=500`, one binary, `20 40 700`:**

| arm | failures | rate |
|---|---|---|
| `none` | 23/40 | 58% |
| `AT=4` | 25/40 | 63% |

**p = 0.82.**  Not merely non-significant — the point estimate has *reversed*
sign relative to §13.161's 11/16 vs 6/16.

**So §13.161 is withdrawn.**  Its shape (a rise across sites 1–3 and a drop at
4) was a 16-round fluctuation, exactly the failure mode its own "Bounds"
paragraph warned about, and I still let the section's title claim a
localisation.  The honest combined statement across §13.158 (1 µs), §13.161
(500 µs, n=16) and this run (500 µs, n=40) is:

> **No delay inserted anywhere in the adopt sequence changes the failure rate,
> at either a 1 µs or a 500 µs per-event window.**  The instrument is known to
> work (a 50 ms window moves a successful run 3.5 s → 5.4 s, §13.161), so this
> is a real null, not a dead probe.

That null is worth as much as a positive would have been: the exposure is not
a two-party interleaving that a 500 µs stall at any of the four adopt-sequence
points can serialise.  Combined with §13.157 (placement is not it) and the
clone bisection (§13.112–§13.148, no single clone), the search space for
"a race between two specific instructions in the adopt path" is now largely
spent, and the next section stops probing timing and asks a structural
question instead.

**Two process notes from this run, both worth keeping:**

* **A p-value read off a partial batch is not a p-value.**  §13.161's 0.032
  came from the round-15 snapshot; at round 16 the same contrast was 0.073.
  Numbers now get computed only after `DONE`.
* **`timeout` counts as a failure in every batch script I have written**
  (`[ $rc -ne 0 ] && F++`).  During this run one `none` execution livelocked —
  21 threads, **2 spinning at ~90% CPU and 19 parked in `futex_do_wait`**, for
  the full 400 s timeout, against a normal 11.7 s success / 1.5 s crash — and
  was scored as a UAF failure.  It is a distinct pathology (no SIGSEGV, no
  watchdog abort) and it is *not* the fault under investigation.  Its rate is
  unmeasured because no batch has ever recorded per-run `rc`; from here they
  do.  One event in 80 runs does not move 23/40 vs 25/40, but every earlier
  batch in §13 carries the same unquantified contamination.

### 13.163 The allocator side, measured: adoption's hazard is universal, the derivation is exact to 1.1 G checks, and three more mechanisms close on reading

§13.162 exhausted the timing probes.  This section stops perturbing and
measures allocator state instead, in three passes, plus three hypotheses that
close by reading the source.

#### (a) What does adoption actually hand over?  Always a chunk with live blocks in it

`KAME_POOL_SURVIVOR_CENSUS` records `MASK_CNT` at the claim CAS — the chunk's
**live-word count** (the alloc path increments it when a flag word goes
`0 → non-zero`, the free path decrements on `→ 0`; `allocator.cpp:1686`,
`:2211`).  Nonzero means blocks in the chunk are still live and held by
**other** threads.  20 runs, `20 40 700`:

| | |
|---|---|
| runs | 20 (18 crashed, 2 clean) |
| adoptions | 123 – 2325 per run |
| **drained (`MASK_CNT == 0`)** | **0, in every run** |
| **survivor (`MASK_CNT != 0`)** | **100%, in every run** |
| **first `allocate_pooled` out of it succeeded** | **100%, in every run** |

So every adoption re-owns a chunk that other threads still hold live blocks
in, and every adopter immediately allocates out of that chunk's bitmap.  **On
crashing and clean runs alike.**

That is by construction, not by accident: a chunk is pushed to the chain only
when its owner exits **non-empty**, and `orphan_chain_scrub` unlinks only
DRAINED orphans, so nothing else can be there to adopt.  **The consequence is
the useful part:** "adoption hands the thread a chunk with foreign live blocks
in it" cannot be what distinguishes a failing run from a passing one, because
it is what *every* adoption does on *every* run.  §13.150's finding that
adoption is necessary is therefore **not** "sometimes it picks a dangerous
chunk" — the shape is always identical, and the discriminator is elsewhere.

One observation, deliberately under-claimed: the maximum live-word count seen
at a claim tracks the adoption count at a near-constant ratio (~1 per 122–123
adoptions, across runs spanning 123 to 2325 adoptions).  A running maximum is
monotone by construction, so this is *consistent with* steady accumulation of
live words across a rotating set of ~120 chunks and is **not** by itself
evidence of a leak.  Recorded because the ratio's constancy across a 19×
range of run lengths is striking, not because it proves anything.

#### (b) Is `back_offset[]` ever mis-derived?  No — 1 144 061 498 checks, zero

§13.109 split the DOUBLE-LIVE hit into two branches and could not decide
between them from an absent poison tag.  Its second branch — *"the block was
NEVER freed; its bitmap bit was cleared by somebody ELSE's free … a
mis-derived `chunk_base` clears a bit in the wrong chunk"* — is directly
checkable, because every chunk is built as

```cpp
ALLOC *palloc = ALLOC::create(CHUNK_SIZE - ALLOC_CHUNK_HEADER,
                              addr + ALLOC_CHUNK_HEADER);   // :2676
```

so **`(char *)palloc == chunk_base + ALLOC_CHUNK_HEADER` holds by
construction** for every chunk in existence.  A wrong `base_idx` breaks it.
`KAME_POOL_RESOLVE_CHECK` tests that identity plus span containment on every
free, skipping dedicated chunks (bit 7, different header layout) and
released/in-creation chunks (`palloc <= 1`).

**First placement was wrong and the instrument said so:** it went into
`resolve_chunk_from_slot`, which this workload never calls — 20 runs reported
`ok=0`.  The hot owner-free path (`:4276`) and the cold cross-thread path
(`:4461`) each derive `chunk_base` inline and never go through the resolver.
Re-aimed at both:

| | |
|---|---|
| runs | 20 (16 crashed: 13×SIGABRT, 3×SIGSEGV; 4 clean) |
| **derivations checked** | **1 144 061 498** |
| **bad identity** | **0** |
| **bad span** | **0** |

**§13.109's second branch is refuted** — not inferred from a missing tag, but
measured, on runs that crashed, at a billion checks.

#### (c) Does the adopt claim loop's "should never happen" ever happen?  No

The claim loop discards a popped orphan that already carries `BIT_OWNED`, a
case its own comment calls *"duplicate-owned (should never happen)"* — two
threads owning one chunk, which would put two allocators on one bitmap and one
freelist: exactly the double-hand-out shape §13.104 saw.  It had never been
counted.  Counted now, together with claim-CAS retries (losses to a concurrent
cross-thread `MASK_CNT` dec):

| | |
|---|---|
| runs | 20 (14 failed: 11×SIGABRT, 2×SIGSEGV, 1×self-detected rc=255; 6 clean) |
| adoptions | **33 939** |
| **DUP-OWNED (two owners for one chunk)** | **0** |
| **claim-CAS retries** | **0** |

Zero duplicates in 34 k adoptions, so the single-push invariant holds and the
discard branch is genuinely dead code.  **Zero retries is the more interesting
number:** the claim CAS never once lost to a concurrent cross-thread
`MASK_CNT` decrement, which says no foreign free is in flight against the
chunk at the instant it is re-owned.  The claim loop's careful
MASK_CNT-preserving retry — and the comment justifying it — describe a race
that does not occur in this workload.


**Validity of the nulls in (a)–(c).**  A probe that suppresses the fault
returns nulls for the wrong reason (§13.157's noinline arm is the standing
example).  These do not suppress it — every instrumented build fails *more*
often than the uninstrumented yield build of §13.162, not less:

| build | instruments | failed |
|---|---|---|
| `yl.so` | yield only | 48/80 (60%) |
| `sv.so` | + survivor census | 18/20 (90%) |
| `rv.so` | + resolve check | 16/20 (80%) |
| `dup.so` | + dup/retry counters | 14/20 (70%) |

These arms were not interleaved against each other, so the *direction* is not
attributable to the instruments — but every null above was collected on a
sample where the fault fired in the large majority of runs, which is what the
nulls need in order to mean anything.

#### (d) Three more mechanisms, closed by reading rather than by running

Each of these is a way a bitmap bit could go free without that block being
freed — i.e. a way to reach §13.109's second branch that is *not* a
mis-derivation.  All three are closed:

1. **A stale word cache surviving into adoption.**  The allocator claims an
   entire 64-slot flag word in one CAS (`CAS oldv → ~0`) and hands out bits
   from a cache held in `m_freelist_head[1]` (remaining mask) and `[2]` (word
   base) — **chunk members, not TLS**, despite the comment at `:1655` saying
   "straight into the TLS mask".  So the cache outlives its owner thread and
   travels with the chunk into the orphan chain.  *If* the thread-exit drain
   returned those bits to the bitmap without clearing the cache, an adopter
   would hand out each of them twice.  It does not: `release_dll_chunks_for_thread`
   (`:3641`) takes the mask, **nulls both cells first**, then returns every
   undistributed bit — and it runs for every chunk in the DLL, before the
   `BIT_OWNED` clear, so orphaned chunks are drained exactly like released
   ones.  Closed.
2. **A stale per-thread pointer aimed into a recycled chunk.**  The free path
   re-aims `kame_page()->m_slots[bucket].freelist_head` to point *into* a
   chunk's `m_freelist_head[local]` cell (`:4571`).  Before releasing a chunk,
   the owner sweeps all `ALLOC_NUM_BUCKETS` of its own slots and nulls any
   that point into it (`:2968`).  The re-aim happens only on the owner-matched
   branch, so a thread only ever aims into chunks it owns — no cross-thread
   aliasing to invalidate.  Closed.
3. **The word cache's take arithmetic.**  Producer and consumer compute the
   slot address differently — producer `base + b * ALIGN` (`:1694`), consumer
   `base + ((b * bucket) << 4)` (`:5359`) — which agree only if
   `slot_size(bucket) == 16 * bucket`.  The word cache fires only for FS=true
   chunks, whose buckets all lie in 1..23, and `kBucketNewSlot[]` is
   documented and tabulated as **"Buckets 1..23: 16-step.  Slot = K*16"**
   (`allocator_prv.h:2989`).  The two agree over the cache's entire domain.
   Closed.

#### (e) Where that leaves the dichotomy — and it points away from the allocator

§13.109 framed the DOUBLE-LIVE hit as: *either* the block was freed while its
object was live (a **premature free**), *or* its bit was cleared by somebody
else (an **allocator fault**).  It could not choose, because the poison tag
was absent and its own text warned that absence is not proof — word 0 is
where `PacketWrapper` puts its refcount, so the second occupant's constructor
overwrites the tag.

The second branch now has no surviving mechanism that I can find: the
derivation is exact at a billion checks (b), the ownership invariant holds
(c), and the three remaining bit-clearing paths are closed by construction
(d).  **That shifts the weight decisively onto the first branch** — the block
*was* freed, while a live object still sat in it — and makes §13.109's absent
tag the false negative it explicitly anticipated.

**Recommendation for the Mac session.**  The allocator has now absorbed
§13.104–§13.163 and returned nulls at every structural question asked of it,
while the one thing it does unconditionally (hand a thread a chunk full of
other threads' live blocks) is invariant across passing and failing runs.  The
productive question is no longer *which pool knob* but **which reference was
dropped**: on a DOUBLE-LIVE hit, the full refcount history of the offending
address — who released it to zero, and who still held it.  The tracer already
has the ledger and the choke point (`rc_trace.cpp:799`, every destruction);
what is missing is dumping that address's history at the trip.  That is a
tracer-mechanics change, which is the Mac side's half of §13.85.

### 13.164 CORRECTION: the adopt census's key never matched, so §13.131 measured nothing — plus a self-test so it cannot recur

**Read this before using §13.131 for anything.**

`kame_pool_adopt_note` recorded the adopted chunk under

```cpp
reinterpret_cast<const char *>(oc) - ALLOC_CHUNK_HEADER      // == chunk_base
```

while `kame_pool_was_adopted` looked it up under

```cpp
(const void *)((uintptr_t)addr & ~(uintptr_t)0x3ffff)        // 256 KiB mask
```

**Those two are never equal.**  A chunk's slot region does not begin at
`chunk_base`; it begins at `chunk_base + ALLOC_CHUNK_K_MAX` — and *that* is the
256 KiB-aligned address:

```cpp
char *mempool() noexcept {                       // allocator_prv.h:1243
    return reinterpret_cast<char *>(this) + (ALLOC_CHUNK_K_MAX - ALLOC_CHUNK_HEADER);
}   // = (chunk_base + 64) + (4096 - 64) = chunk_base + 4096
```

`chunk_base = unit_boundary - K_MAX` (`allocator_prv.h:858`, `:922`), so every
user pointer masked to 256 KiB yields `chunk_base + 4096`, and the stored key
was `chunk_base`.  Off by exactly K_MAX, every time.
**`kame_pool_was_adopted()` returned false for every address ever passed to
it, in every run.**

So §13.131's result — *"no doubly-live slot lands in a census-recorded adopted
chunk"* — is **not a finding**.  It is the only answer that function could
give.  And it is the same result my §13.163(a) had to work around as "the
census records the wrong moment": the moment was fine, the *key* was wrong.
**The §13.131-vs-§13.150 tension (adoption necessary, yet never implicated in
a hit) may dissolve entirely on re-measurement.**  It is now re-measurable.

**What was fixed.**  Both censuses (adopt, and the new release tally) key on
the slot-region base and register **every** 256 KiB unit of a multi-unit
chunk, so a pointer into an upper unit answers too.

**What stops a repeat.**  A census that answers questions about addresses now
proves it can answer at all.  The adopt path holds a block pointer that
*provably* came from the chunk it just recorded — the return value of
`oc->allocate_pooled(SIZE)` — so it queries the census with it and counts
agreement.  The report carries `selftest ok=N BAD=N`.  §13.131 had no such
check; had it had one, the bug would have surfaced the first time it ran.

**Immediate consequence, already visible.**  §13.164's own first
question — has the chunk containing a doubly-live address ever been released
wholesale? — read `releases=0` under the broken key and **`releases=3`** under
the fixed one, on the very next run.

**Standing lesson, and this is the fourth time in §13 that a probe has needed
one.**  §13.155/§13.156 was a diagnostic whose call sites were outside its own
`#ifdef`; §13.163(b) was a check placed in a function the workload never
calls (`ok=0` for 20 runs); this is a lookup keyed differently from its
insert.  All three were silent — they returned plausible numbers.  A probe
that reports "nothing found" must first be made to report "found" on a case
that is true by construction; until it has, its null is not evidence.

### 13.165 With the key fixed, §13.131 reverses — and the double hand-out happens inside ONE chunk incarnation

§13.164 fixed the census key and made every answer carry a receipt.  Re-asking
the question §13.131 believed it had answered, on the tracer build with
`KAME_RC_TRACE_DLIVE=2`, one DOUBLE-LIVE hit per run:

| run | rc | `chunk-was-adopted` | keying self-test | releases Δ | constructions Δ |
|---|---|---|---|---|---|
| 1 | 134 | **YES** | ok=612 BAD=0 | 0 (0→0) | 0 (1→1) |
| 2 | 134 | no | ok=490 BAD=0 | 0 (2→2) | 0 (4→4) |
| 3 | 134 | no | ok=977 BAD=0 | 0 (3→3) | 0 (5→5) |
| 4 | 134 | **YES** | ok=122 BAD=0 | 0 (0→0) | 0 (1→1) |

Δ = the counter at the second birth minus its value recorded at the previous
occupant's birth, stamped into the live-set entry when that occupant was born.

**§13.131 is reversed.**  Doubly-live slots *do* land in adopted chunks — 2 of
4 here — and the self-test says the census could answer (hundreds of
successful self-queries, zero failures, on every run).  §13.131's "no" was the
only answer a mis-keyed lookup could return.  **The §13.131-vs-§13.150 tension
is gone:** there was never a contradiction, only a broken key.

That adoption is not universal *for the faulting chunk* (2 of 4) while §13.150
showed adoption is behaviourally *necessary* is not a new contradiction — it
says adoption's role is to keep the recycle machine turning, which is also
what §13.146's dose-response says.  It is not "the faulting chunk must itself
have been adopted".

**The load-bearing column is the last one.**  In every usable observation the
chunk was neither released nor re-constructed between the two births: **the
same chunk incarnation held both occupants.**  (See the correction below: 2 of
these 4 rows are usable, not 4.)  Constructions are counted, not just
releases, precisely because the §22 warm path recycles a cached chunk with its
units still claimed and never calls `deallocate_chunk` — a release tally
cannot see that reuse, and `construct_chunk_at` is the one point every reuse
route passes through.

So **whole-chunk recycle under live objects is ruled out** — the last
mechanism §13.163(d) left open.  Combined with §13.163(b) (derivation exact to
1.1 G checks) and §13.163(c) (no duplicate ownership in 34 k adoptions), what
remains is narrow and follows as a consequence:

> A slot inside a single live chunk became re-allocatable while its occupant
> was still a constructed object.

There are exactly two ways for that: parked on the owner freelist
(`freelist_push`, bit stays set) or returned to the bitmap
(`batch_return_to_bitmap`).  An audit of all seven `freelist_push` call sites
confirms every one is guarded by
`m_owner_id == page_owner_id && page_owner_id != 0`, and owner ids come from a
monotonic `s_owner_id_next.fetch_add(1)` held in a per-thread GD TLV, so they
are never recycled and the non-atomic list really is owner-only.  §13.166's
instrument records which of the two paths freed the address, which thread, and
whether that free happened **after** the previous occupant was born.

**CORRECTION, found while building §13.166's instrument and applied here
rather than left standing.**  `dl_born_` claims a live-set slot on **four**
branches, and the birth-time counters were stamped on only **two** of them —
the two that re-claim an already-known address.  The branches that claim a
*fresh* slot (`cur == 0`, and the steal path) left `relcnt`/`concnt` at their
initial 0.  So `at_prev_birth = 0` is ambiguous: it can mean "the counter was
genuinely zero" or "never recorded".

Applying that to the table above: **runs 2 and 3 are sound** — their
`at_prev_birth` values are 2 and 3, which only a real stamp can produce, and
both deltas are 0.  **Runs 1 and 4 are ambiguous** and must not be counted:
both read 0→0, which is exactly what an unrecorded slot looks like.

So the incarnation claim rests on **2 clean observations, not 4**.  It is
still the only direction any observation points, and §13.163(d)'s reasoning is
independent of it, but "4/4" was wrong and is withdrawn.  All four branches
now stamp the counters, and an unrecorded prev-birth value prints
`[verdict withheld]` instead of a verdict — the same disease as §13.164's
keying bug (a missing write that reads as a legitimate value), caught this
time by looking at the raw fields rather than the verdict string.

**Bounds, stated.**  n = 4, of which 2 usable.  The batch was **capped deliberately, not
completed**: per-run cost had risen to ~6 min because runs kept entering the
livelock mode of §13.162 (3 threads spinning, 38 parked, killed at the 400 s
timeout), and it was blocking the decisive measurement.  The
incarnation result is the claim worth carrying forward, at n = 2; the 2/4 on
adoption is a ratio from four samples and should not be quoted as one.  A zero Δ can also be
produced by a hash collision evicting the table entry (the census is
direct-mapped), which the self-test does not cover — with ~120 distinct chunk
bases in 8192 slots that is unlikely, but it is not excluded.

### 13.166 The double-free hypothesis, built and refuted: 0/40 in the shipping configuration, and the 37% that looked like it was a false positive

§13.165 left one shape: a slot inside a single live chunk became re-allocatable
while its occupant was still constructed.  A slot parked on the owner freelist
keeps its bitmap bit SET, so the list is the only record that it is free —
which means pushing an address **already on the list** puts it there twice, and
two pops hand one block to two live objects.  That is exactly §13.104's
DOUBLE-LIVE, with the first occupant's destructor never running.  So I built a
detector for it: one bit per slot address, set by push, cleared by pop and by
`batch_return_to_bitmap`.

**It works, and it fires.**  But the answer is the opposite of the hypothesis.

#### The detector had to be wrong twice first

* **13 295 208 "double frees"** on the first build.  That absurd magnitude is
  the finding: the hot allocation path pops the freelist **inline at seven
  sites** rather than through `freelist_pop()`, so the bit was never cleared.
  Hooking every pop site, the second push path (`allocator_prv.h:2963`) and the
  chunk-claim pre-fill dropped it to 0–1 per run.  **A new counter's first
  duty is a magnitude sanity-check**; had this one come back plausible-looking
  instead of absurd, I would have published it.
* **`construct_chunk_at`'s pre-fill marks every slot of a chunk as on-list**,
  so warm-reusing a chunk whose slots were still marked re-sets bits with no
  free having happened.  That is a false positive aimed squarely at this
  claim, and it is now counted in a separate bucket rather than assumed away.

It also carries a **positive control** that runs at load — push/push must count
once, push/pop/push must not, counter restored afterwards — and prints
`PASS` beside every number.  After §13.155, §13.163(b) and §13.164, no probe in
this investigation gets to report a zero without first proving it can report a
one.

#### An intermediate result that was WRONG, and how the separation caught it

Before the pre-fill bucket existed, the detector reported **a double free in 10
of 27 failing runs (37%)**, and I came within one write-up of publishing that
as the mechanism.  With pre-fill counted apart, the numbers are:

| build | metric | rate |
|---|---|---|
| conflated (old) | "double frees" | **10/27** |
| separated (new) | double frees only | **0/40** |
| separated (new) | pre-fill re-sets only | **4/25** |

10/27 vs 4/25 → **p = 0.12: statistically the same population.**
10/27 vs 0/40 → **p = 0.00003: a different one.**

So the 37% was **pre-fill re-sets misattributed to double frees** — the exact
false positive the separation was built for, confirmed quantitatively rather
than assumed.  The claim is withdrawn.

#### The controlled measurement

One binary, runtime gate, arms interleaved within each round, 15 rounds:

| arm | failures | runs with a double free | pre-fill re-sets |
|---|---|---|---|
| `base` | **15/15** | **0/15** | 0 in 14/15 |
| `KAME_ORPHAN_NO_ADOPT=1` | **0/15** | **2/15** | 2–7 in 12/15 |

An uninterleaved 25-run base batch on the same binary agrees: **0/25** double
frees (22 failures, 3 clean), 4 runs with pre-fill re-sets.

The suppressor replicates perfectly: **15/15 vs 0/15, p = 1.3 × 10⁻⁸** — the
cleanest reproduction of §13.150 so far, on one binary with zero codegen delta.

And **the double frees are in the arm that does not fail.**  Every failing run
in this A/B had zero; the only two runs that double-freed both completed
cleanly.  A double free is therefore **neither necessary nor sufficient** for
the fault:

> In the shipping configuration the allocator does **not** double-free at all:
> **0 in 40 base runs**, with the detector's positive control passing on every
> one.  The only double frees seen anywhere are 2/15 in the `NO_ADOPT` arm —
> and that arm is an **ablation** that deliberately strands orphaned chunks
> instead of re-owning them, so it is not evidence of a defect in the real
> allocator either.

I had earlier written that a genuine double free existed and was worth fixing
on its own merits.  **That is retracted**: it rested on the conflated counter
above.  What survives is the negative — the double-free path is not the
mechanism, and in the configuration that ships it does not occur.

(The elevated pre-fill re-sets under `NO_ADOPT` are expected, not anomalous:
with adoption gated off, orphaned chunks are stranded and fresh chunks are
constructed more often, and a stranded chunk's slots keep their on-list bits.)

#### A pre-registered prediction of mine failed, and it is worth more than the result

Before the last-free data landed I wrote down (scratchpad, timestamped): since
`kame_slot_link_()` writes the freelist link at **word 0** when
`KAME_POISON_FORENSIC` is off, a `freelist_push` would leave a *pool pointer*
in word 0, whereas §13.113 measured "W0 reads a LIVE refcount".  I therefore
predicted the last free would be `batch_return_to_bitmap`.

Both recorded hits said **`freelist_push`**.  My first reading was that one of
the two instruments must be wrong.  **That was too strong, and the same
captures settle it**, because they carry the W-probe and the free record for
the *same* hit:

```
RC-DLIVE-LASTFREE path=freelist_push (owner park) tid=76 seq=11481462 prev_birth_seq=0
RC-DLIVE-ADOPTED  0x74620d026d60 chunk-was-adopted=YES [selftest ok=122 BAD=0 TRUSTWORTHY]
RC-DLIVE-WPRE     0x74620d026d60 w=0x2          <- a refcount, not a pool pointer
RC-DLIVE-W1       0x74620d026d60 w=0x7461e24c3de0
```

`WPRE = 0x2` is the word as it stood *before* the second occupant's
`refcnt = 1` landed, and it is a small integer — the previous occupant's live
refcount — not the freelist link a `freelist_push` writes there.  Both
instruments are right, and they are consistent under one reading: **the
recorded free predates the previous occupant's birth.**  The slot was pushed
and popped, then O1 was born into it and wrote its own refcount over the link,
and then O2 was born on top of O1 **with no free in between**.

That is not a reconciliation I get to assert, because the very field that
would test it — `prev_birth_seq` — read 0 on both hits, the §13.165 bug where
`dl_born_` stamped its counters on only two of four slot-claim branches.  Both
verdicts ("FREED AFTER…") were therefore vacuous, and are withdrawn.  With all
four branches stamped, a hit carrying **both** a present free record and a real
`prev_birth_seq` decides it:

* free-seq **after** the previous birth → an ordinary free landed on a live
  object (a premature free, the STM side);
* free-seq **before** it → the allocator handed out a slot that nothing freed
  since its occupant was born, which sends the search back into the pool.

That measurement is queued.  (The lookup is lossy — a 2^20 table under ~10⁸
frees per run evicts fast, so most hits report "no record"; a *present* record
is address-verified and is the true last free of that address, while an absent
one says nothing.)

### 13.167 The block is freed WHILE ITS OCCUPANT IS LIVE — 6/6, through both free paths

§13.166 withdrew its own ordering verdicts: `prev_birth_seq` read 0 because
`dl_born_` stamped its counters on only two of four slot-claim branches, so
"FREED AFTER" was `seq > 0` and meant nothing.  With all four branches stamped
(§13.165's correction) the comparison is real.  9 runs, all of which failed,
6 DOUBLE-LIVE hits carrying **both** a present free record and a genuine
birth stamp:

| hit | path | free seq | prev-birth seq | thread |
|---|---|---|---|---|
| 1 | `freelist_push` | 32 360 502 | 330 638 | 240 |
| 2 | `freelist_push` | 11 565 208 | 3 681 296 | 63 |
| 3 | `batch_return_to_bitmap` | 5 396 460 | 1 232 268 | 22 |
| 4 | `freelist_push` | 4 592 197 | 1 199 031 | 11 |
| 5 | `freelist_push` | 4 760 359 | 1 107 851 | 17 |
| 6 | `freelist_push` | 16 655 167 | 580 775 | 92 |

**6 of 6: freed AFTER the occupant was born.**  Zero "predates", zero "no
record" (the 2^20 table retains what the earlier 2^16 one evicted), six
distinct threads, and — the part that matters — **both** free paths behave
identically.  This is not a quirk of one code path: whatever issues the free
reaches the allocator through the ordinary owner-side route *and* through the
cross-thread batch.

The ordering is therefore:

> O1 is constructed at slot A → an ordinary free returns A to the pool → O2 is
> constructed at A, while O1 has never been destroyed.

That is §13.109's **first** branch — the block *was* freed, under a live
object — which is exactly where §13.163(e) predicted the weight would fall
once the second branch ran out of mechanisms.  Combined with the allocator's
unbroken run of nulls (derivation exact to 1.14 G checks §13.163(b); no
duplicate ownership in 34 k adoptions §13.163(c); no whole-chunk recycle
§13.165; no double free in 40 base runs §13.166), the reading is:

> **The pool faithfully recycles blocks it is told to free.  Something is
> telling it to free live objects.**

**The load-bearing assumption, named.**  All of this rests on §13.107's claim
that the live-set's destruction feed is complete — that a `DL_LIVE` slot means
the destructor genuinely never ran.  If some destruction path bypassed that
choke point, O1 would in fact be dead, the free would be legitimate, and every
DOUBLE-LIVE since §13.104 would be a false positive.  The evidence for
completeness is §13.104's base rate: **0 hits in ~21 M enforced checks on runs
that succeeded**.  An incomplete feed would leak false positives into clean
runs too, and it does not.  That is strong but indirect, and it is the single
assumption this whole line now hangs on — worth an independent check by the
Mac session (§13.85's half), because if it fails, §13.104–§13.167 fall with it.

**Next, already instrumented.**  Detect the premature free *at the free*
rather than inferring it at the next birth: the tracer exports
`kame_rc_dl_islive()` and the allocator asks it on every free — a legitimate
`delete` runs the destructor, and so the death hook, before `operator delete`,
so a legitimate free must see 0.  That fires at ~10^8 frees per run instead of
one rare hit, and captures a raw `backtrace()` of the first offender
(never `backtrace_symbols`, which mallocs and would re-enter the allocator
from inside a free).  It carries a positive control: at a DOUBLE-LIVE hit the
previous occupant is live *by definition*, so the report prints
`islive(obj)` and it must read 1.
