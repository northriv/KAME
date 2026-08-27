#!/usr/bin/env python3
"""§13.142  Tag-mask census: does any compiled body use the tagged pointer
without masking the tag off?

The question (user's proposal i).  `atomic_shared_ptr` keeps a local refcount in
the LOW BITS of the pointer word, and every use has to mask:

    (Ref*)(ref & ~(LOCAL_REF_CAPACITY - 1))     // pointer   -> `and ~7`
    (Refcnt)(ref & (LOCAL_REF_CAPACITY - 1))    // the count -> `and 7`

A compiler that concludes the value is an aligned pointer may drop `& ~7` as
redundant -- and IPA-CP is exactly what supplies the premise, since a
specialization that propagates a zero refcount makes `(uintptr_t)pref + 0`
provably 8-aligned.  Correct *for that clone*; wrong if such a body is ever
reached with a tagged value.  `m_ref` is declared `uintptr_t`, not a pointer
type, which closes the direct type-based route -- so this asks the object file
rather than the source.

What it reports, per function that touches the tag machinery:
  * `mask_ptr`   -- masks with ~(CAP-1): the pointer extraction
  * `mask_cnt`   -- masks with (CAP-1):  the count extraction
  * `tagged_add` -- adds a small constant to a pointer-shaped value (building a
                    tagged value: `(uintptr_t)pref + rcnt`)
A body with `tagged_add > 0` and `mask_ptr == 0` is the shape to look at: it
constructs or consumes a tagged value and never masks.

Usage:
    tagmask_census.py <obj-or-so> [more...] [--objdump CMD] [--cap 8]

Compare the firing and non-firing builds: a function whose `mask_ptr` count
DROPS between them is the finding.  Absolute counts alone prove nothing --
inlining moves masks between bodies, which is why this is a differential tool.
"""
import re, subprocess, sys, collections

def bodies(path, objdump):
    out = subprocess.run([objdump, "-d", "-C", "--no-show-raw-insn", path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("objdump failed on %s:\n%s" % (path, out.stderr[:300]))
    fns, cur = {}, None
    hdr = re.compile(r'^[0-9a-f]+\s+<(.+)>:\s*$')
    ins = re.compile(r'^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$')
    for line in out.stdout.splitlines():
        m = hdr.match(line)
        if m: cur = m.group(1); fns[cur] = []; continue
        m = ins.match(line)
        if m and cur is not None: fns[cur].append((m.group(1), m.group(2)))
    return fns

def census(insns, cap):
    lo = cap - 1                       # 7
    hi_hex = "%x" % ((1 << 64) - cap)  # fffffffffffffff8
    c = collections.Counter()
    for mnem, ops in insns:
        o = ops.lower()
        if mnem.startswith(("and", "bic")):
            if hi_hex in o or ("$0x%x" % (0x100000000 - cap)) in o \
               or re.search(r'#-?%d\b' % cap, o):
                c['mask_ptr'] += 1
            elif re.search(r'[$#]0x?%x\b' % lo, o) or re.search(r'#%d\b' % lo, o):
                c['mask_cnt'] += 1
        # building a tagged value: add/lea of a small immediate < cap
        if mnem.startswith(("add", "lea", "sub")):
            m = re.search(r'[$#](0x[0-9a-f]+|\d+)', o)
            if m:
                v = int(m.group(1), 16) if m.group(1).startswith('0x') else int(m.group(1))
                if 0 < v < cap: c['tagged_add'] += 1
    return c

def main():
    objdump, cap, args, skip = "objdump", 8, [], False
    for i, a in enumerate(sys.argv[1:], 1):
        if skip: skip = False; continue
        if a == '--objdump': objdump = sys.argv[i+1]; skip = True
        elif a == '--cap': cap = int(sys.argv[i+1]); skip = True
        elif a.startswith('--'): pass
        else: args.append(a)
    if not args: sys.exit(__doc__)
    interesting = re.compile(r'atomic_shared_ptr|local_shared_ptr|tag_ref|load_shared|'
                             r'scoped_atomic_view|compareAndSet|compareAndSwap')
    for path in args:
        fns = bodies(path, objdump)
        rows = []
        tot = collections.Counter()
        for n, insns in fns.items():
            if not interesting.search(n): continue
            c = census(insns, cap)
            if not c: continue
            tot.update(c)
            rows.append((c['tagged_add'], c['mask_ptr'], c['mask_cnt'], n))
        rows.sort(reverse=True)
        print("== %s ==  bodies touching the tag machinery: %d" % (path, len(rows)))
        print("   totals: mask_ptr=%d mask_cnt=%d tagged_add=%d"
              % (tot['mask_ptr'], tot['mask_cnt'], tot['tagged_add']))
        susp = [r for r in rows if r[0] > 0 and r[1] == 0]
        print("   SUSPECT (builds/consumes a tagged value, never masks): %d" % len(susp))
        for add, mp, mc, n in susp[:10]:
            print("     %-70s add=%d mask_ptr=0 mask_cnt=%d" % (n[:70], add, mc))
        for add, mp, mc, n in rows[:8]:
            print("   %-70s add=%d mask_ptr=%d mask_cnt=%d" % (n[:70], add, mp, mc))
        print()

main()
