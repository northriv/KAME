#!/usr/bin/env python3
# Interleaved A/B of malloc libraries on the micro benches: every repetition
# runs each configuration under every arm, the arm order rotating, and the
# median per arm is printed.  The library is preloaded per run
# (DYLD_INSERT_LIBRARIES on macOS, LD_PRELOAD elsewhere).
# usage: ab.py BINDIR REPS NAME=LIB [NAME=LIB ...]
#   BINDIR holds the built replace1 / heavy / ring (see README.md).
import os, re, statistics, subprocess, sys
bindir, reps = sys.argv[1], int(sys.argv[2])
arms = [tuple(a.split('=', 1)) for a in sys.argv[3:]]
preload = 'DYLD_INSERT_LIBRARIES' if sys.platform == 'darwin' else 'LD_PRELOAD'
CFG = [('replace1', '5000 20000000 400 1000'), ('replace1', '5000 20000000 8 1000'),
       ('replace1', '100000 20000000 16 4096'), ('replace1', '5000 20000000 1500 3000'),
       ('replace1', '5000 20000000 16 368'),
       ('heavy', '100000 1000000 4 400 1000'), ('heavy', '100000 1000000 32 400 1000'),
       ('heavy', '100000 1000000 32 32 512'),
       ('ring', '64 4 20'), ('ring', '64 32 20'), ('ring', '64 128 20')]
for b, a in CFG:
    res = {n: [] for n, _ in arms}
    for r in range(reps):
        for n, lib in arms[r % len(arms):] + arms[:r % len(arms)]:
            env = dict(os.environ, **{preload: lib})
            out = subprocess.run([os.path.join(bindir, b)] + a.split(), env=env,
                                 capture_output=True, text=True).stdout
            m = re.search(r'ns_per_alloc=([0-9.]+)' if b == 'heavy' else r'rate=([0-9.]+)M', out)
            if m: res[n].append(float(m.group(1)))
    unit = 'ns/alloc' if b == 'heavy' else 'M ops/s'
    print(f'{b} {a}  [{unit}]  ' + '  '.join(
        f'{n} {statistics.median(res[n]):.1f}' if res[n] else f'{n} -' for n, _ in arms), flush=True)
