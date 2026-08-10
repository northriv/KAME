#!/usr/bin/env python3
"""Find the slow commit inside a `perf script --call-trace` dump.

Two streaming passes, so a multi-GB trace does not have to fit in memory.

WHAT THE FIRST VERSION OF THIS GOT WRONG, because both mistakes are easy and
both produce confident nonsense:

  * It ranked raw timestamp gaps between consecutive same-tid LINES.  A PT
    snapshot is a RING BUFFER and `perf script` emits it per thread/buffer, so
    file order is not time order.  The top hit was a "145 ms stall" on a thread
    committing at 173 kHz — that is the ring's wrap point — and below it sat
    multi-millisecond "stalls" whose two lines were a million lines apart, i.e.
    buffer switches.  Real gaps never got near the top-40.  A gap is only
    meaningful when the two lines are ADJACENT IN THE FILE for that thread, and
    that is now required.
  * It reported the DENSEST windows as "where the work is".  Backwards.  Call
    density on a busy thread is near constant (measured: 1786-1855 calls per
    10 us, a 4 % spread), because a commit that merely takes longer emits
    proportionally more calls.  A STALL is the opposite — the thread retires no
    branches, so its window goes SPARSE.  Look for the thin windows.

So the discriminator is: sparse window (or an adjacent-line gap) => the thread
was resident but retiring no USER-SPACE branches.  That is NOT yet "memory
stall": PT recorded with -e intel_pt//u traces user space only, so KERNEL time
looks exactly the same.  The window histogram decides — syscall stubs
(__read_nocancel, __close_nocancel, sched_yield ...) mean the kernel, and on
the first real trace that is what they were: the livelock probe's uncached
hardware_concurrency() doing an openat/read/close of
/sys/devices/system/cpu/online per tick.  No sparse window => it was
executing user code, and the histogram says what.

Two artifact classes to expect in section 1, both seen on the first real
trace: a gap population that is ONE value repeated (2004.67 us x8 here) is a
timed wait — the thread parked on a futex with that timeout, off-CPU, which PT
does not trace; and the single largest gap ending at the trace's newest
timestamp is the SNAPSHOT BOUNDARY, where the fragment written at the SIGUSR2
moment abuts the older ring body (145 ms here, on a thread committing at
173 kHz — obviously not a stall).
"""
import re, sys, collections, heapq, os

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pt.txt"
LINE = re.compile(r'^\s*(\S+)\s+(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+\((.*?)\)\s*\t(.*)$')
#! A gap is only a stall if the two samples are ADJACENT in the file for this
#! thread.  Anything further apart is the decoder moving between ring buffers.
MAX_LINE_SPAN = 64
#! ...and even then it need not be a stall.  Off-CPU time is not traced, so a
#! voluntary park shows up as a gap; and the decoder's own packet cadence can
#! manufacture one.  A gap POPULATION that is all one value, to the TSC
#! quantum, is not a physical stall distribution — hence the histogram.
BUCKET_HZ = 1e5          # 10 us buckets

print(f"# {PATH}  {os.path.getsize(PATH)/1e6:.1f} MB")

last = {}
gaps = []
gap_hist = collections.Counter()
gap_sum = collections.Counter()
buckets = collections.Counter()
nlines = 0
with open(PATH, errors="replace") as f:
    for i, line in enumerate(f):
        m = LINE.match(line)
        if not m: continue
        nlines += 1
        tid = int(m.group(2)); t = float(m.group(4))
        buckets[(tid, int(t * BUCKET_HZ))] += 1
        p = last.get(tid)
        if p is not None and (i - p[1]) <= MAX_LINE_SPAN:
            d = (t - p[0]) * 1e6
            if 1.0 <= d <= 1e6:           # below 1 us is the TSC quantum
                gap_hist[(tid, round(d, 2))] += 1
                gap_sum[tid] += d
                heapq.heappush(gaps, (d, tid, p[1], i, t))
                if len(gaps) > 40: heapq.heappop(gaps)
        last[tid] = (t, i)

print(f"# {nlines:,} parsed lines, threads: " + ", ".join(f"{t}" for t in sorted(last)))
if not nlines: sys.exit("no lines parsed — is this a --call-trace dump?")

busiest = collections.Counter()
for (tid, _b), c in buckets.items(): busiest[tid] += c
TID = busiest.most_common(1)[0][0]
mine = {b: c for (tid, b), c in buckets.items() if tid == TID}
lo, hi = min(mine), max(mine)
occupied = len(mine)
mean = sum(mine.values()) / occupied
print(f"# busiest thread = tid {TID}: {busiest[TID]:,} calls in {occupied:,} "
      f"of {hi-lo+1:,} 10 us windows, mean {mean:.0f} calls/window")
print(f"# (span {(hi-lo+1)/BUCKET_HZ*1e3:.1f} ms wall, "
      f"{occupied/BUCKET_HZ*1e3:.1f} ms actually covered — a ring buffer need "
      f"not be contiguous)\n")

print("=== 1. adjacent-line gaps on tid", TID, "(ring artifacts excluded) ===")
mine_gaps = [(d, n) for (tid, d), n in gap_hist.items() if tid == TID]
ngap = sum(n for _d, n in mine_gaps)
print(f"    {ngap} gaps totalling {gap_sum[TID]/1000:.1f} ms, against "
      f"{occupied/BUCKET_HZ*1e3:.1f} ms of traced execution")
if ngap:
    print("    by size — ONE value repeated is a decoder or off-CPU artifact, "
          "a spread is physical:")
    for d, n in sorted(mine_gaps, key=lambda x: -x[1])[:8]:
        print(f"      {n:6d} x {d:10.2f} us   ({n*d/1000:8.1f} ms total)")
for d, n in sorted(mine_gaps, key=lambda x: -x[0])[:3]:
    if n >= 3 and d >= 1000.0:
        print(f"    NOTE: {n} identical {d:.2f} us gaps = a timed wait "
              f"(futex timeout), off-CPU, not a stall")
        break
t_new = max(t for t, _l in last.values())
top = sorted(gaps, reverse=True)[:8]
for d, tid, lb, la, t in top:
    tag = "  <= SNAPSHOT BOUNDARY (ends at the newest fragment), not a stall" \
          if (t_new - t) < 0.002 else ""
    print(f"    largest: {d:9.2f} us  tid {tid}  at {t:.9f}  lines {lb}-{la}{tag}")

print(f"\n=== 2. STALLS by sparse window: thinnest 10 us windows on tid {TID} ===")
print(f"    (mean {mean:.0f}; a window far below it is time the thread was "
      f"resident and not retiring)")
#! A window at the START or END of a traced fragment is partially filled by
#! construction and says nothing.  Every one of the 15 thinnest windows in the
#! first real run had a zero neighbour — i.e. they were ALL fragment edges and
#! the list was pure structure.  Interior windows only.
interior = [(b, c) for b, c in mine.items()
            if mine.get(b - 1, 0) and mine.get(b + 1, 0)]
edges = len(mine) - len(interior)
print(f"    {len(interior):,} interior windows ({edges:,} fragment edges "
      f"excluded — those are partial by construction, not stalls)")
thin = sorted(interior, key=lambda kv: kv[1])[:15]
if not thin: print("    no interior windows at all")
for b, c in thin:
    nb = [mine.get(b + k, 0) for k in (-1, 1)]
    print(f"  {c:6d} calls at {b/BUCKET_HZ:.5f}   ({c/mean*100:5.1f} % of mean;"
          f" neighbours {nb[0]}, {nb[1]})")
if thin:
    lo_c = thin[0][1]
    print(f"    => thinnest interior window is {lo_c/mean*100:.0f} % of mean."
          + ("  Nothing stall-shaped; the thread retired user code "
             "throughout." if lo_c > 0.5 * mean else
             "  The thread was resident but retiring no USER branches "
             "there — kernel time or a memory stall; the histogram below "
             "decides (syscall stubs mean kernel)."))

print(f"\n=== 3. WORK: densest windows, for contrast ===")
for b, c in sorted(mine.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {c:6d} calls at {b/BUCKET_HZ:.5f}   ({c/mean*100:5.1f} % of mean)")

# --- pass 2 ---------------------------------------------------------------
regions = []
if top:
    d, tid, lb, la, t = top[0]
    regions.append((f"the {d:.1f} us adjacent-line gap", max(0, lb-400), la+400,
                    tid, None))
if thin:
    b, c = thin[0]
    regions.append((f"the thinnest INTERIOR window ({c} calls, "
                    f"{c/mean*100:.0f} % of mean)", None, None, TID, b))
hists = {n: collections.Counter() for n, _a, _z, _t, _b in regions}
if regions:
    with open(PATH, errors="replace") as f:
        for i, line in enumerate(f):
            m = LINE.match(line)
            if not m: continue
            tid = int(m.group(2))
            fn = m.group(6).strip().split("(")[0]
            for (n, a, z, want_tid, want_b) in regions:
                if tid != want_tid: continue
                if a is not None:
                    if a <= i <= z: hists[n][fn] += 1
                elif int(float(m.group(4)) * BUCKET_HZ) == want_b:
                    hists[n][fn] += 1
for n, _a, _z, _t, _b in regions:
    tot = sum(hists[n].values())
    print(f"\n=== 4. calls in {n}  (n={tot}) ===")
    for fn, c in hists[n].most_common(20):
        print(f"  {c:6d}  {100*c/tot:5.1f} %  {fn[:110]}")
