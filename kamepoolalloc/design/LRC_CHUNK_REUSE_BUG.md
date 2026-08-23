# The LRC_CHUNK recycle-cache identity bug

Status as of 2026-08-23.  Two sessions in parallel: an Ubuntu 26.04 RT box
(2-core, x86-64, g++ 15.2) and a cloud container (4-core, x86-64 + ILP32 via
`-m32`, g++ 13.3 / clang 18).  This is a running record, not a fix.

The symptom is heap corruption reached through `transaction_dynamic_node_test`,
in every mode: SIGSEGV, SIGABRT, silent hangs, an `objcnt != 0` live-object
leak, an uncaught `STM lookup failed` (`std::domain_error` → `terminate`), and
crashes *inside* `operator new` — the last says the heap's own structures are
already damaged, so this is broad damage, not one stale pointer.

## Established

Numbers worth standing behind.  Each line is one variable against a control
run in the same sweep.

| | |
|---|---|
| Pool-only | 0/6 without the pool (RT); 0/42 on i486 and 0/44 on i586 (cloud) |
| Address reuse is the variable | delaying reuse by 2 M allocations: 0/41 vs 13/41 |
| Only the ALLOCATOR's codegen matters | clang-STM + gcc-pool 8/12; gcc-STM + clang-pool 0/12 |
| One flag, both directions | `-O3 -fno-ipa-cp-clone` 0/67; `-O2 -fipa-cp-clone` 19/35; `-O3` 12/36 |
| The clone SET, not perturbation | 3 ways of removing the clones: 0/91 combined.  5 ways of perturbing without removing them (`-falign-*`, `-fno-tree-vectorize`, `-fno-ipa-sra`, `-fno-inline-functions`, `inline-unit-growth=10`): all still fail, including one that reshuffles 334 functions and fails 7/16 |
| No single clone is responsible | and `noclone` cannot isolate one — suppressing a single clone changes 72 other functions |
| Not an ordering weakness | `seq_cst` on every atomic in the allocator: 4/8, ABOVE the control.  A placebo fence: 7/8 |
| Reproduces on ILP32 too | i586 25%, i486 12.5% with `tmin3 20 8 2500`, where the compact-state layout is NOT in use — so `KAME_STM_COMPACT_STATE` is not involved |

## Ruled out

Recorded so nobody re-runs them.

* **Stamp wraparound.**  Shrinking `STAMP_US_BITS` 24 → 8 (a 0.128 ms
  half-window, ~65000x tighter than compact mode's 8.3 s) still passes.
* **`KAME_STM_COMPACT_STATE`.**  i586 has it OFF and fails *more* than i486.
* **`LOCAL_REF_CAPACITY`.**  arm64/M3, ~2400 runs across nine configurations
  including OVERRIDE=4 and 2 and forced uint32 bitmap: zero failures.
* **Link form** (static 4/30 vs shared 6/30), **TLS model** (initial-exec
  13/80 vs default 21/80 — shifts the rate, not the cause), **core count**
  (4/2/1 cores: 0/48, 1/55, 0/73).
* **fast_vector element leaks.**  A purpose-built probe over
  `fast_vector<shared_ptr<Counted>,16>` — the shipped test uses `weak_ptr` at
  inline capacity 4 and cannot see this — clean over 25 lifetime scenarios on
  x86-64 and i486.
* **`Linkage::m_tx_commit_count`.**  Was a genuine data race (TSan: 282-404
  reports, including write/write, disproving its "single writer" comment) and
  is now a relaxed `diag_counter_t` atomic.  Pre- and post-fix run
  simultaneously under one load: 0/134 and 0/134.  Fixed, unrelated.
* **The 32-bit radix bound check.**  `radix_lookup_slow` does drop it on ILP32
  (`kBoundShift` 48 >= 32) but the indices cannot escape: `region_idx =
  up >> 25` caps at 127, so l1 is always 0 and l2 always < 2048.
* **Static reading of all four IPA-CP clones** (`l1_pop_fit`,
  `global_pop_fit`, `recycle_pop_fit`, `CrossDeallocBatch::flush`): the
  `sz >= need` verification survives specialisation (as `cmpl $262143`), the
  manual pre-call caching in `flush` and `push_direct` is honoured (the field
  load precedes the call in both builds), `global_pop_fit`'s sharing is all
  atomic with the metadata read after CAS ownership, and `l1_pop_fit`'s plain
  slot access is legitimate because L1 is `ALLOC_TLS`.
* **Mechanical binary scans**: TLS access model, atomic instruction counts,
  instruction mix.  None separate failing from clean builds.

## Why no sanitiser sees it

* TSan, compact mode, pool on and off: races on exactly one variable
  (`m_tx_commit_count`, now fixed), never in the payload path,
  `allocator.cpp` never in a race frame.
* valgrind memcheck, 8 threads, pool off, `--keep-stacktraces=alloc-and-free`,
  `--freelist-vol=500M`: 4/4 runs complete, zero invalid accesses.
* UBSan on i586, 82 runs: exactly one report, every run — the deliberate
  header type-pun at `allocator.cpp:4292`.  Real UB, worth rewriting as a
  byte-offset store, but not this bug: `offsetof(m_owner_id)` is 64 and
  `ALLOC_CHUNK_HEADER` is 64 on both widths, so the store lands at +128
  either way.  What matters is the negative: no misalignment, no over-width
  shift, no signed overflow.

The reason is structural, and it is the hypothesis below.

## Leading hypothesis: a parked chunk keeps its identity

`bucket_release_chunk` has two exits:

```c++
if(large_recycle_push(chunk_base, chunk_size, LRC_CHUNK))
    return;                     // units stay claimed
deallocate_chunk(chunk_base, chunk_size);   // real release: bitmap clear + madvise
```

|  | parked in the LRC | really released |
|---|---|---|
| claim bits | **still set** | cleared |
| `chunk_header` | **intact** — `DEDICATED_SIZE` is written *before* the push, because `lrc_block_size(LRC_CHUNK)` reads it back | destroyed |
| radix entry | **still `KAME_RADIX_POOL`** | removed |
| physical pages | kept warm | `madvise`d |

Re-acquisition does not re-claim anything — the bits are already set:

```c++
if(char *cached = large_recycle_pop(CHUNK_SIZE, LRC_CHUNK)) {
    restamp_back_offset(cached, CHUNK_SIZE, /*back_off_flag=*/0u);
    return construct_chunk_at(cached);      // placement-new for the NEW owner
}
```

So the contract that breaks is:

> A parked chunk is *freed* yet its identity stays *live*.  A stale pointer
> into it does not fault and does not read as freed — it resolves cleanly
> through the radix to whatever now occupies the slot.

`LRC_MMAP` does not have this problem: its blocks go back through
`large_va_raw_unmap` → `munmap`, so the VA disappears and a stale pointer
faults.  `radix_remove` has no callers at all; nothing on the park path
touches the radix.

That accounts for every negative above.  Nothing is unmapped, so valgrind is
silent.  Nothing is unsynchronised and no atomic is missing, so TSan and the
instruction scans are silent.  It also accounts for the two positives that
matter: quarantining reuse fixes it, and it cannot happen without the pool.

The invariant to model:

```
∀ chunk c, ∀ pointer p into c:
    resolvable(p) through the radix  ⟹  owner(c) == the owner that issued p
```

The park → re-acquire transition violates it.

## Open

Whether the IPA-CP clones are a miscompile, or whether their boundary
placement widens a race that exists in every build.  Neither session's methods
separate those, and both stories need the same defect underneath.  Treat
`-O3` vs `-O3 -fno-ipa-cp-clone` as a reliable on/off switch for experiments
(65% vs 0/91) rather than as a lead.

## Reproducer

`tmin3.cpp`, 148 lines from the original 365.  `./tmin3 100 16 1250`, ~25 s
per run at 50-65% on LP64; `./tmin3 20 8 2500` gives 25% on i586.  Build
against the pool as a shared library and vary only its flags:

```sh
c++ -DKAMEPOOLALLOC_DYLIB -I kamestm/tests -I kamestm -I kamepoolalloc \
    -O3 -g -DNDEBUG -std=gnu++17 -include kamestm/tests/support_standalone.h \
    tmin3.cpp kamestm/tests/support_standalone.cpp kamestm/threadlocal.cpp \
    -o tmin3 -Wl,-rpath,<pooldir> <pooldir>/libkamepoolalloc.so -ldl
```

Ablation-verified as necessary: a persistent 3-level tree, many threads,
repeated thread create/exit, the `gn2` multi-node Tx, the `trans(*gn3)` pair,
the private p1/p2 churn, and the p2-into-gn2 hard link.  Unnecessary: `gn4`,
the swap, the main-thread churn, the per-round tree rebuild.

Two cautions for whoever picks this up, both learned the hard way here.  A
silent control discriminates nothing — the rate wobbles with load, and arms
run at different times are not comparable, so interleave them in one sweep.
And clone-set equality is not codegen equality: `noclone` on the five
functions produces the same `.constprop` set as `-fno-ipa-cp-clone` yet a
`.text` 27 KB larger.
