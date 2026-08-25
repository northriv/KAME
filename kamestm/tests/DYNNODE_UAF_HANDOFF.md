# `transaction_dynamic_node_test`: the fault is a `Packet` use-after-free

Handoff for the TLA+ work.  Written 2026-08-24 from the x86-64 / g++ 15.2 /
Ubuntu 26.04 `PREEMPT_RT` session (2 cores).  Continues
`origin/claude/great-turing-Ufao2` and the header of `soak_dynnode.sh`, which
narrowed the fault to ILP32 + the pool allocator and ended with "no backtrace
yet" on LP64.

**There is now an LP64 reproducer that fires most runs, and the fault has been
caught in the act.  It is not an allocator defect and not a miscompile.**

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
