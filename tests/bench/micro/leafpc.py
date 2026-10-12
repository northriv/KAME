#!/usr/bin/env python3
# Histogram of leaf PCs inside libkamepoolalloc from an xctrace time-profile export.
# usage: leafpc.py export.xml dylib  -> prints per-function totals and top PCs (dylib-relative)
import sys, re, subprocess, collections
import xml.etree.ElementTree as ET
xmlf, dylib = sys.argv[1], sys.argv[2]
frames, bts, bins = {}, {}, {}
leaf = collections.Counter()
for ev, el in ET.iterparse(xmlf, events=('end',)):
    if el.tag == 'binary' and el.get('id'):
        bins[el.get('id')] = (el.get('name'), int(el.get('load-addr'), 16))
    elif el.tag == 'frame' and el.get('id'):
        b = el.find('binary')
        bid = None
        if b is not None: bid = b.get('id') or b.get('ref')
        frames[el.get('id')] = (int(el.get('addr'), 16), bid)
    elif el.tag == 'backtrace' and el.get('id'):
        fs = []
        for f in el.findall('frame'):
            fs.append(f.get('id') or f.get('ref'))
        bts[el.get('id')] = fs
    elif el.tag == 'row':
        b = el.find('backtrace')
        if b is None: el.clear(); continue
        fs = bts.get(b.get('id') or b.get('ref'), [])
        if fs:
            leaf[fs[0]] += 1
        el.clear()
kp = [ (n, la) for (n, la) in bins.values() if n.startswith('libkamepoolalloc') ]
la = kp[0][1]
pcs = collections.Counter()
other = collections.Counter()
for fid, n in leaf.items():
    addr, bid = frames[fid]
    nm = bins.get(bid, ('?', 0))[0] if bid else '?'
    if nm.startswith('libkamepoolalloc') or (bid is None and la <= addr < la + 0x200000):
        pcs[addr - la] += n
    else:
        other[nm] += n
total = sum(leaf.values())
# symbolize with nm (sorted text symbols)
syms = []
for line in subprocess.run(['nm', '-n', dylib], capture_output=True, text=True).stdout.splitlines():
    p = line.split()
    if len(p) >= 3 and p[1] in 'tT':
        syms.append((int(p[0], 16), p[2]))
import bisect
addrs = [a for a, _ in syms]
def sym(off):
    i = bisect.bisect_right(addrs, off) - 1
    return (syms[i][1], off - syms[i][0]) if i >= 0 else ('?', off)
byfn = collections.Counter()
for off, n in pcs.items():
    byfn[sym(off)[0]] += n
dem = lambda s: subprocess.run(['c++filt', s], capture_output=True, text=True).stdout.strip()
print(f'total samples {total}; in kp {sum(pcs.values())}; other {dict(other.most_common(5))}')
for fn, n in byfn.most_common(8):
    print(f'{n:6d} {100*n/total:5.1f}%  {dem(fn)[:110]}')
print('top PCs:')
for off, n in pcs.most_common(40):
    s, d = sym(off)
    print(f'{n:6d}  0x{off:x}  {dem(s)[:60]}+0x{d:x}')
