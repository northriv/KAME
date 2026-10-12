#!/usr/bin/env python3
# Aggregate xctrace CPU Counters (MetricAggregationForThread) for the main thread:
# cycle-weighted mean of each top-down metric over the 10 ms (non-precise) rows.
import sys, re, subprocess, collections
import xml.etree.ElementTree as ET
xml = subprocess.run(['xcrun', 'xctrace', 'export', '--input', sys.argv[1], '--xpath',
      '/trace-toc/run[@number="1"]/data/table[@schema="MetricAggregationForThread"]'],
      capture_output=True, text=True).stdout
root = ET.fromstring(xml)
ids = {}
def val(el):
    if el is None: return None
    r = el.get('ref')
    if r: return ids[r]
    v = (el.text or '').strip() if len(el) == 0 else el.get('fmt')
    if el.get('id'): ids[el.get('id')] = v
    # register children too
    for c in el.iter():
        if c is not el and c.get('id') and c.get('id') not in ids:
            ids[c.get('id')] = (c.text or '').strip() if len(c) == 0 else c.get('fmt')
    return v
bystart = collections.defaultdict(dict)
for row in root.iter('row'):
    cols = list(row)
    t = val(cols[0]); d = val(cols[1]); th = val(cols[2]); pr = val(cols[3])
    vi = val(cols[4]); vd = val(cols[5]); name = val(cols[6]); prec = val(cols[7])
    if prec not in ('0', 'No'): continue
    if 'replace1' not in str(th) and 'Main Thread' not in str(th): continue
    bystart[t][name] = (int(vi) if vi not in (None, '') else 0, float(vd) if vd not in (None, '') else 0.0)
tot = collections.Counter(); cyc = 0
for t, m in bystart.items():
    c = m.get('cycle', (0, 0))[0]
    if not c: continue
    cyc += c
    for k, (i, f) in m.items():
        if k != 'cycle': tot[k] += f * c
print(f'cycles {cyc/1e9:.2f} G  ' + '  '.join(f'{k} {tot[k]/cyc:.3f}' for k in sorted(tot)))
