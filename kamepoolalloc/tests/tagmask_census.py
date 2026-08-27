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
  * `cas`        -- compare-exchange (`lock cmpxchg`; arm64 `cas*` or an
                    `ldaxr`/`stlxr` pair)
  * `rmw`        -- other atomic read-modify-writes (`lock add/sub/xadd/or/and/
                    xchg`; arm64 `ldadd*`/`swp*`)
  * `fence`      -- `mfence` / `dmb`

**On "was a cmpxchg or a relaxed op omitted?" -- the two halves differ.**

A `cas`/`rmw` that disappears IS visible here.  GCC does not delete an atomic RMW
as such, but constant propagation that kills a branch condition removes the whole
PATH containing one -- which is the realistic failure mode, and it is countable:
a clone with fewer `cas` than every source path through it requires has lost an
atomic operation.

A **relaxed load or store is NOT visible**, and no asm tool can make it so: a
relaxed load compiles to a plain `mov`/`ldr`, indistinguishable from a
non-atomic load.  So "the relaxed op was omitted" cannot be observed directly --
only relatively, as *fewer loads than the source path performs*, which inlining
also changes.  For that half the runtime accounting is the real instrument (the
tracer's DEC ledger and `dtor == born`, §13.74/§13.107), not the disassembly.
This is stated because counting `mov`s and calling the difference a missing
relaxed load would be exactly the kind of number that looks like evidence.
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
        if mnem.startswith("lock"):
            t = o.split()[0] if o else ""
            if "cmpxchg" in mnem or "cmpxchg" in t: c['cas'] += 1
            else: c['rmw'] += 1
        elif mnem in ("cas", "casa", "casal", "casl", "casb", "cash") \
             or mnem.startswith(("casal", "casa", "casl")):
            c['cas'] += 1
        elif mnem.startswith(("ldaxr", "ldxr")):
            c['cas'] += 1            # half of an LL/SC pair; counted once
        elif mnem.startswith(("ldadd", "ldclr", "ldset", "ldeor", "swp", "stadd")):
            c['rmw'] += 1
        elif mnem.startswith(("mfence", "dmb", "dsb")):
            c['fence'] += 1
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
            rows.append((c['tagged_add'], c['mask_ptr'], c['mask_cnt'],
                         c['cas'], c['rmw'], c['fence'], n))
        rows.sort(reverse=True)
        print("== %s ==  bodies touching the tag machinery: %d" % (path, len(rows)))
        print("   totals: mask_ptr=%d mask_cnt=%d tagged_add=%d | cas=%d rmw=%d fence=%d"
              % (tot['mask_ptr'], tot['mask_cnt'], tot['tagged_add'],
                 tot['cas'], tot['rmw'], tot['fence']))
        susp = [r for r in rows if r[0] > 0 and r[1] == 0]
        print("   SUSPECT (builds/consumes a tagged value, never masks): %d" % len(susp))
        for add, mp, mc, cas, rmw, fen, n in susp[:10]:
            print("     %-58s add=%d mask_ptr=0 mask_cnt=%d" % (n[:58], add, mc))
        print("   %-58s %4s %4s %4s %4s %4s %4s"
              % ("body", "add", "mskP", "mskC", "cas", "rmw", "fnc"))
        for add, mp, mc, cas, rmw, fen, n in rows[:10]:
            print("   %-58s %4d %4d %4d %4d %4d %4d"
                  % (n[:58], add, mp, mc, cas, rmw, fen))
        print()

main()
