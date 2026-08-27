#!/usr/bin/env python3
"""§13.146  Clone/operation reconciliation -- what proposal (ii) actually asks.

Not "did an operation vanish from one body" but an ARITHMETIC IDENTITY: if
cloning creates new bodies, the object's operation totals must rise by what those
bodies contain.  Identical totals are therefore NOT evidence of health -- they are
a coincidence that needs explaining, which is the mistake §13.145 made in reading
`mask_ptr 331 vs 331` as "nothing lost".

    residual = (total delta) - (delta inside DERIVED bodies)

Derived means `.constprop` (IPA-CP clones), `.part` (partial inlining) and
`.isra` (argument removal): all three move operations out of a parent, and
counting only `.constprop` makes partial inlining read as an omission -- the false
lead this tool was written to kill.  Residual 0 = conserved.  NEGATIVE = ordinary
bodies lost that many operations, which is the failure mode worth chasing.

Usage: clone_op_reconcile.py <base.o> <clone.o>
"""
import sys, collections
src = open("kamepoolalloc/tests/tagmask_census.py").read().replace("\nmain()", "\n")
ns = {}; exec(compile(src, "tc", "exec"), ns)
bodies, census = ns['bodies'], ns['census']
A = bodies(sys.argv[1], "objdump"); B = bodies(sys.argv[2], "objdump")
CAP = 8
def tot(fns, pred=lambda n: True):
    c = collections.Counter()
    for n, ins in fns.items():
        if pred(n): c.update(census(ins, CAP))
    return c
# derived bodies of ANY kind: constprop (clones), part (partial inlining),
# isra (argument removal).  All three take ops out of their parent.
derived = lambda n: ('.constprop' in n) or ('.part' in n) or ('.isra' in n)
tA, tB = tot(A), tot(B)
dA, dB = tot(A, derived), tot(B, derived)
print("derived bodies: base=%d clone=%d"
      % (sum(1 for n in A if derived(n)), sum(1 for n in B if derived(n))))
print("%-10s %8s %8s %8s %10s %10s" % ("op","base","clone","delta","derived Δ","residual"))
for k in ('mask_ptr','mask_cnt','cas','rmw','fence'):
    d = tB[k]-tA[k]; dd = dB[k]-dA[k]
    print("%-10s %8d %8d %+8d %+10d %+10d" % (k, tA[k], tB[k], d, dd, d-dd))
