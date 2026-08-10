#!/usr/bin/env python3
"""Find the slow commit inside a `perf script --call-trace` dump.

Two streaming passes, so a multi-GB trace does not have to fit in memory.

  pass 1  per-thread timestamp gaps (a gap = the CPU retired no branch, i.e.
          a STALL) and 10 us bucket densities (dense = it was doing work)
  pass 2  re-read only the interesting line ranges and histogram the calls

A 30 us commit shows up as exactly one of the two, and which one it is
answers the question: a gap means memory/coherence, density means code.
"""
import re, sys, collections, heapq, os

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pt.txt"
LINE = re.compile(r'^\s*(\S+)\s+(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+\((.*?)\)\s*\t(.*)$')
GAP_US = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

print(f"# {PATH}  {os.path.getsize(PATH)/1e6:.1f} MB")

last = {}                       # tid -> (t, lineno)
gaps = []                       # (gap_us, tid, lineno_before, lineno_after, t)
buckets = collections.Counter() # (tid, 10us bucket) -> count
nlines = 0
with open(PATH, errors="replace") as f:
    for i, line in enumerate(f):
        m = LINE.match(line)
        if not m: continue
        nlines += 1
        tid = int(m.group(2)); t = float(m.group(4))
        buckets[(tid, int(t * 1e5))] += 1
        p = last.get(tid)
        if p is not None:
            d = (t - p[0]) * 1e6
            if d >= GAP_US:
                heapq.heappush(gaps, (d, tid, p[1], i, t))
                if len(gaps) > 40: heapq.heappop(gaps)
        last[tid] = (t, i)

print(f"# {nlines:,} parsed lines, threads: "
      + ", ".join(f"tid {t}" for t in sorted(last)))
if not nlines:
    sys.exit("no lines parsed — is this a --call-trace dump?")

busiest = collections.Counter()
for (tid, _b), c in buckets.items(): busiest[tid] += c
TID = busiest.most_common(1)[0][0]
print(f"# busiest thread = tid {TID} ({busiest[TID]:,} calls)\n")

print("=== 1. STALLS: largest timestamp gaps (no branch retired) ===")
top = sorted(gaps, reverse=True)[:15]
if not top:
    print(f"  none over {GAP_US} us")
for d, tid, lb, la, t in top:
    print(f"  {d:9.2f} us  tid {tid}  at {t:.9f}  lines {lb}-{la}")

print("\n=== 2. WORK: densest 10 us windows on tid", TID, "===")
dens = sorted(((c, b) for (tid, b), c in buckets.items() if tid == TID),
              reverse=True)[:10]
for c, b in dens:
    print(f"  {c:6d} calls in 10 us at {b/1e5:.5f}")

# --- pass 2: histogram the regions of interest -----------------------------
regions = []
if top:
    d, tid, lb, la, t = top[0]
    regions.append((f"around the {d:.1f} us STALL (tid {tid})",
                    max(0, lb - 400), la + 400, tid))
if dens:
    c, b = dens[0]
    regions.append((f"the densest 10 us window ({c} calls)", None, None, TID))
    dense_bucket = b
else:
    dense_bucket = None

want_lines = set()
for _n, a, z, _t in regions:
    if a is not None: want_lines.update(range(a, z + 1))

hists = {n: collections.Counter() for n, _a, _z, _t in regions}
with open(PATH, errors="replace") as f:
    for i, line in enumerate(f):
        m = LINE.match(line)
        if not m: continue
        fn = m.group(6).strip()
        fn = fn.split("(")[0]
        tid = int(m.group(2))
        for (n, a, z, want_tid) in regions:
            if tid != want_tid:
                continue          # the gap is one THREAD's; other tids are noise
            if a is not None:
                if a <= i <= z: hists[n][fn] += 1
            elif dense_bucket is not None:
                if int(float(m.group(4)) * 1e5) == dense_bucket:
                    hists[n][fn] += 1

for n, _a, _z, _t in regions:
    print(f"\n=== 3. calls {n} ===")
    tot = sum(hists[n].values())
    for fn, c in hists[n].most_common(20):
        print(f"  {c:6d}  {100*c/tot:5.1f} %  {fn[:110]}")
