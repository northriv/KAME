# Thread lists and fit lists (`5fc34e052`) — checks for the M5 Ultra and Linux

`5fc34e052` (master `04d904ccf` with its bench commit) takes every walk over
chunks off the allocation path.  The M3 results are in its commit message and
in `GAME_FRAME_BENCH.md`; the bounds in `RT_READINESS.md`.  This note lists
what the MacBook Air could not settle and how to run it elsewhere.  Append
each run below as a section, with provenance (`bench_ab.sh` writes it).

The v1.2.1 allocator to compare against is master `45f1d07f9`'s
`kamepoolalloc/` (no tag was cut).

## What changed

- Owner frees go onto the freeing thread's list for the slot's bucket
  (`KameTlsPage::m_slots[b]`), whichever chunk holds the slot; mallocs pop
  it.  A miss means the thread holds no free slot of the bucket, and goes to
  bitmap room.  The per-bucket shortcut into one chunk's freelist, §fl-scan
  (`FL_UNTRACKED`, `fl_scan`, `fl_retarget`) and §walk are gone.  The bucket
  comes from the borrow prefix, `m_base_bucket`, or `m_sizes`, which now
  records `(N << 8) | bucket` in the full tier.  Thread exit hands the lists
  back to their chunks' freelists (`tl_return_to_chunk`) for the exit walk.
- Fit lists (§fit): bitmap room a variable-size chunk keeps when it cannot
  serve a request, filed by the largest tier size it still fits; at most
  `FIT_TRIES` (4) chunks tried per request.
- An FS=true miss claims a whole bitmap word in one CAS and chains its slots
  (≤ 63 stores) for the thread's list.  The word cache's mask, the FIFO and
  the STASH experiments are removed; `m_sizes` is no longer in a union with
  the ring's counters, so the chunk object is 4 B smaller on ILP32 (LP64:
  320 B either way, the bitmap still at +384).

## What to look at

1. **ILP32.**  The union removal changes the 32-bit layout, and the claim's
   chain loop runs on a 32-bit `FUINT` in the `KAME_FORCE_UINT32_BITMAP`
   fallback (`__builtin_ctzll` on the word widened to 64 bits).
2. **No-DCAS.**  The new code adds no 64-bit atomic; phase 2 of
   `tools/audit/check_no_dcas.sh` (i486 objects export no `__atomic_*_8`)
   runs only where gcc-multilib is installed.
3. **Thread exit.**  The lists' hand-back is new: the thread-exit soaks.
4. **Zen 2.**  The word cache avoided per-slot stores because of Zen 2's
   store-port plateau (the reason `8a4d26229` gave when it made the word
   cache the default); the claim now writes one link per slot.  On the M3 the SPSC ring went from 28 (v1.2.1) to 50 M ops/s;
   whether Zen 2 keeps that is open.
5. **Tails.**  The Air throttles after ~30 minutes of benchmarks; frame
   tails were noisy for every arm, mimalloc included.
6. **Memory.**  Owner-freed slots stay on the thread's lists until it exits,
   as they stayed on chunk freelists before; peak RSS was 0.88x v1.2.1's on
   `bench_frame`, still ~2x mimalloc's.

## Linux

Six trees as in `POOL_FL_AVAIL_LINUX_CHECK.md` §1 (LP64, i586, i486;
release and asserts on), at master `04d904ccf`:

- `tools/audit/check_no_dcas.sh`, with gcc-multilib (3 phases).
- ctest in all six.  `transaction_wait_budget_test` misses its p99.99 bound
  under `ctest -j` on a small host; run it alone before counting it.
- Soaks, scored by exit status: `alloc_tsd_exclusivity_test`,
  `alloc_tsd_exclusivity_var_test` and `alloc_tsd_exclusivity_mid_test`
  2000 runs each on LP64, 500 on i486 and i586; `alloc_fit_test` 300;
  the `alloc_thread_exit_*` tests 200 each.

## M5 Ultra

Build the v1.2.1 arm from the archive, then run the A/B from this checkout:

    mkdir -p ~/bench-ab/arms/v121
    git -C ~/KAME archive 45f1d07f9 kamepoolalloc | tar -x -C ~/bench-ab/arms/v121
    cd ~/bench-ab/arms/v121/kamepoolalloc && cmake -B out -DCMAKE_BUILD_TYPE=Release && cmake --build out -j
    cd ~/KAME && kamepoolalloc/tests/bench/bench_ab.sh --tags tree \
        --arm v121=$HOME/bench-ab/arms/v121/kamepoolalloc/out/libkamepoolalloc.dylib \
        --arm mi=/path/to/libmimalloc.dylib \
        --only "loop xthread xmalloc larson rptest xlat frame frameRT" \
        --threads "4 8" --reps 5

Then the micro benches (`tests/bench/micro/README.md`) with `./ab.py . 5
v121=... tree=...`, and STM `transaction_payload_integrity_3level_mixed_test`
from two test trees, one per allocator, interleaved, with exactly four
arguments: `3 128 128 10`, `3 128 128 0`, `3 8 128 10`.  Keep the frame loop
within the performance cores (`--threads` ≤ 10 there).

## Ohtaka (Zen 2, like PS5 / Xbox Series)

The same `bench_ab.sh` set as the M5's, under the Ohtaka rules in
`CLAUDE.md` (`srun --exclusive` on `i8cpu`, one job per measurement, median
of 5, provenance): `--threads "4 16 64" --xlat-cpu 0,1`, and the frame loop
pinned to one NUMA node (`numactl -N 0`), as a console has none.  Item 4
above is the one to watch: ring and `xmalloc-test` 64 B against v1.2.1.
