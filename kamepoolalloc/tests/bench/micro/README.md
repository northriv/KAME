# Micro benches for the allocation miss paths

Single-file programs behind the thread-list / fit-list work
(`design/GAME_FRAME_BENCH.md`, `design/RT_READINESS.md`).  Each runs under
whatever malloc is preloaded; with kamepoolalloc preloaded, replace1 and
heavy also print its live chunk count.

| file | what |
|---|---|
| `replace1.c` | larson's steady state on one thread: LIVE blocks of random size in [MIN, MAX), each iteration frees a random one and allocates a replacement.  `replace1 LIVE ITERS MIN MAX` → `rate=…M ops/s` |
| `heavy.c` | allocation ahead of frees: the heap grows, and only every EVERY-th allocation is followed by a free of a random old block.  `heavy LIVE0 ALLOCS EVERY MIN MAX` → `ns_per_alloc=` |
| `ring.c` | SPSC ring: a producer mallocs SIZE-byte blocks into a ring of S slots, a consumer frees them, every free cross-thread.  `ring SIZE S MILLION_OPS` → `rate=…M` |
| `freelat.c` | the latency of every free while one thread frees MB megabytes of SIZE-byte blocks, in allocation order or (`SHUFFLE=1`) a random order; `RT=1` frees in realtime mode.  `freelat SIZE MB` → mean, p99.99, max, frees over 1 / 10 / 100 µs |
| `ab.py` | interleaved A/B of preloaded libraries over the configurations above: `ab.py BINDIR REPS NAME=LIB …` |
| `leafpc.py`, `chase.c` | macOS: histogram of leaf PCs inside the library from `xctrace export` of a Time Profiler trace; chase.c calibrates the attribution (a sample's address minus 1 is the PC) |
| `ccagg.py` | macOS: cycle-weighted top-down metrics of the main thread from an xctrace CPU Counters trace |

Build here (`kame_pool.h` is three levels up):

    cc -O2 -I../../.. -o replace1 replace1.c
    cc -O2 -o heavy heavy.c
    cc -O2 -o ring ring.c -lpthread
    cc -O2 -o freelat freelat.c

Linux needs `-ldl` for replace1 and heavy on older glibc.  Then e.g.
`./ab.py . 5 old=/path/to/old/libkamepoolalloc.so new=/path/to/new/libkamepoolalloc.so`.
