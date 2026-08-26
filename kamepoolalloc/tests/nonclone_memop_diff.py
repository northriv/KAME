#!/usr/bin/env python3
"""§13.123  Diff two allocator objects restricted to functions that are NOT clones,
ranked by change in MEMORY-OPERATION shape rather than by size.

Why this and not the existing asm diff.  §13.119 and §13.122 exhausted the
specialization SET in both directions: adding every reachable clone does not make
the fault appear, and removing the two families that account for the entire gap
does not make it disappear.  What is left is the pass's effect on functions it
never clones -- re-run IPA-CP propagation changes value ranges, alias/escape
conclusions and inline decisions unit-wide.  The earlier runner reported "34
function sizes differ", which is true and not actionable.

The fault class narrows the ranking.  §13.113/§13.116: a bitmap bit was cleared
for a slot whose own free never happened, i.e. some free computed the wrong
address or ran twice.  That is a change in the *memory operations* a function
performs -- an atomic RMW added or lost, a store hoisted out of or into a branch,
a load duplicated so two reads back one CAS.  So rank by deltas in atomic ops,
then stores, then loads; a function that grew 40 bytes of scheduling noise
without changing its memory shape ranks last, and one that lost a `lock` prefix
ranks first even if its size is identical.

Usage:
    nonclone_memop_diff.py <A.o|A.so> <B.o|B.so> [--top N] [--objdump CMD]

A is the reference (e.g. plain -O2), B the firing build (-O2 -fipa-cp-clone).
Works on x86-64 and arm64 output; mnemonic classes cover both.
"""
import re, subprocess, sys, collections

def disassemble(path, objdump):
    out = subprocess.run([objdump, "-d", "-C", "--no-show-raw-insn", path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("objdump failed on %s:\n%s" % (path, out.stderr[:400]))
    funcs, cur = {}, None
    # GNU objdump: "0000000000001234 <name>:";  llvm-objdump: "0000... <name>:"
    hdr = re.compile(r'^[0-9a-f]+\s+<(.+)>:\s*$')
    insn = re.compile(r'^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$')
    for line in out.stdout.splitlines():
        m = hdr.match(line)
        if m:
            cur = m.group(1); funcs[cur] = []
            continue
        m = insn.match(line)
        if m and cur is not None:
            funcs[cur].append((m.group(1), m.group(2)))
    return funcs

# Mnemonic classes.  Deliberately coarse: the question is "did the memory shape
# change", not "which instruction".
ATOMIC = re.compile(r'^(lock|xchg|cmpxchg|xadd|ldxr|ldaxr|stxr|stlxr|cas|casa|casal|casl|swp|swpa|swpal|ldadd|ldadda|ldaddal|stadd|dmb|dsb|mfence|lfence|sfence)', re.I)
STORE  = re.compile(r'^(mov[a-z]*|st[a-z0-9]*|push)$', re.I)
LOAD   = re.compile(r'^(mov[a-z]*|ld[a-z0-9]*|pop)$', re.I)
BRANCH = re.compile(r'^(j[a-z]+|b|b\.[a-z]+|bl|blr|cb[nz]+|tb[nz]+|call|ret)', re.I)

def shape(insns):
    c = collections.Counter()
    for mnem, ops in insns:
        c['n'] += 1
        if ATOMIC.match(mnem): c['atomic'] += 1
        if BRANCH.match(mnem): c['branch'] += 1
        # memory direction: on x86 `mov` is both; use the operand form.
        if STORE.match(mnem) and ('(' in ops.split(',')[-1] or '[' in ops.split(',')[-1]):
            c['store'] += 1
        elif LOAD.match(mnem) and ('(' in ops or '[' in ops):
            c['load'] += 1
    return c

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) < 2: sys.exit(__doc__)
    top = 20; objdump = "objdump"
    for i, a in enumerate(sys.argv):
        if a == '--top': top = int(sys.argv[i+1])
        if a == '--objdump': objdump = sys.argv[i+1]
    A, B = disassemble(args[0], objdump), disassemble(args[1], objdump)

    clones = {n for n in set(A) | set(B) if '.constprop' in n or '.isra' in n
                                        or '.part' in n}
    common = [n for n in (set(A) & set(B)) - clones]
    onlyB  = sorted(n for n in set(B) - set(A) - clones)
    onlyA  = sorted(n for n in set(A) - set(B) - clones)

    rows = []
    for n in common:
        sa, sb = shape(A[n]), shape(B[n])
        d = {k: sb[k] - sa[k] for k in ('n','atomic','store','load','branch')}
        if any(d.values()):
            rows.append((abs(d['atomic'])*1000 + abs(d['store'])*100
                         + abs(d['load'])*10 + abs(d['branch']), n, sa, sb, d))
    rows.sort(reverse=True)

    # Collapse template instantiations.  One function instantiated per size
    # class produces 40 identical rows and buries every other finding, so group
    # by (name with template args elided, delta signature) and show the count.
    def key(n):
        return re.sub(r'<[^<>]*>', '<>', re.sub(r'\b\d{2,4}u?\b', 'N', n))
    grouped = {}
    for score, n, sa, sb, d in rows:
        k = (key(n), tuple(sorted(d.items())))
        g = grouped.setdefault(k, [score, n, sa, sb, d, 0])
        g[5] += 1
    rows = sorted(grouped.values(), reverse=True)

    print("clones excluded: %d   non-clone functions in both: %d   changed: %d"
          % (len(clones), len(common), len(rows)))
    print("non-clone functions only in B: %d%s" % (len(onlyB),
          ("  e.g. " + ", ".join(onlyB[:4])) if onlyB else ""))
    print("non-clone functions only in A: %d%s" % (len(onlyA),
          ("  e.g. " + ", ".join(onlyA[:4])) if onlyA else ""))
    print()
    print("%-52s %4s %6s %7s %6s %5s %6s" %
          ("function (non-clone, changed; xN = instantiations)", "xN",
           "insns", "atomic", "store", "load", "branch"))
    for score, n, sa, sb, d, cnt in rows[:top]:
        nm = n if len(n) <= 50 else n[:47] + "..."
        print("%-52s %4d %+6d %+7d %+7d %+6d %+6d" %
              (nm, cnt, d['n'], d['atomic'], d['store'], d['load'], d['branch']))
    print()
    print("Read the atomic column first: a non-clone function that gained or lost")
    print("an atomic RMW between these builds is the shape §13.113/§13.116 predict")
    print("(a bit cleared for a slot whose free never happened).  A large `insns`")
    print("delta with all memory columns at 0 is scheduling noise -- deprioritise it.")

main()
