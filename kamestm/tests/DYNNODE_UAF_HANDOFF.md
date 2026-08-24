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
