import re, itertools, sys

# σ-alphabet for 2-level (Parent=root → {C1,C2}), matching the fields the four
# STRUCTURAL safety invariants read. P.hasPriority ≡ TRUE (root).
#   mP            : Parent.packet.missing            ∈ {0,1}
#   per child c   : cstate ∈ {'Pri','Bun'}           (well-formed: hp ⟺ bb=Null)
#                   subc  ∈ {'N','PNM','PM'}          (P.packet.sub[c]:
#                                                       Null / present-not-missing / present-missing)
# A σ is the tuple (mP, c1, sub1, c2, sub2).

CSTATES = ['Pri','Bun']         # priority / bundled
SUBS    = ['N','PNM','PM']      # P.sub[c]: Null / present¬missing / present+missing

def wellformed():
    for mP in (0,1):
        for c1 in CSTATES:
            for s1 in SUBS:
                for c2 in CSTATES:
                    for s2 in SUBS:
                        yield (mP,c1,s1,c2,s2)

# --- the four structural invariants, as predicates on σ (P.hasPriority=TRUE) ---
def snapshot_consistency(s):
    mP,c1,s1,c2,s2 = s
    # (P.priority ∧ ¬mP) ⇒ ∀c: sub[c] ≠ Null
    if not mP:
        return s1!='N' and s2!='N'
    return True
def no_priority_loss(s):
    # ∀c: hp ∨ bb≠Null. Well-formed: Pri→hp; Bun→bb=Parent≠Null. Always holds.
    return True
def bundle_ref_consistency(s):
    # ∀c: (¬hp ∧ bb=Parent) ⇒ P.hasPriority(=TRUE). Vacuous/always true here.
    return True
def missing_propagation(s):
    mP,c1,s1,c2,s2 = s
    # (P.priority ∧ ¬mP) ⇒ ∀c: sub[c]≠Null ⇒ ¬sub[c].missing
    if not mP:
        return s1!='PM' and s2!='PM'
    return True

def in_B(s):
    return (snapshot_consistency(s) and no_priority_loss(s)
            and bundle_ref_consistency(s) and missing_propagation(s))

# --- candidate extra invariant from memo §7.2 ---
def stale_parent_excluded(s):
    mP,c1,s1,c2,s2 = s
    # ∀c: (c holds own priority ∧ P advertises a sub copy for c) ⇒ P.missing
    for cstate,sc in ((c1,s1),(c2,s2)):
        if cstate=='Pri' and sc!='N':
            if not mP:
                return False
    return True

B  = [s for s in wellformed() if in_B(s)]
Bp = [s for s in B if stale_parent_excluded(s)]

print(f"well-formed σ-alphabet size: {sum(1 for _ in wellformed())}")
print(f"|B|  (4 structural invariants)            = {len(B)}")
print(f"|B'| (B ∧ StaleParentExcluded)            = {len(Bp)}")
print(f"B ∖ B' (invariant-satisfying but excluded by StaleParentExcluded) = {len(B)-len(Bp)}:")
for s in B:
    if s not in Bp:
        print(f"    {s}")

# --- representation well-formedness the wrapper encoding maintains (local, level-uniform) ---
def wf_sub_not_missing(s):      # WF1: P's sub copies are leaf packets → never missing (2-level)
    _,_,s1,_,s2 = s
    return s1!='PM' and s2!='PM'
def wf_bundled_has_copy(s):     # WF2: a bundled child ⟺ P holds its present copy
    _,c1,s1,c2,s2 = s
    for c,sc in ((c1,s1),(c2,s2)):
        if c=='Bun' and sc!='PNM': return False
    return True

def count(preds):
    return [s for s in wellformed() if all(p(s) for p in preds)]

inv4 = [snapshot_consistency,no_priority_loss,bundle_ref_consistency,missing_propagation]
sets = {
 "B0 = 4 safety invariants"                          : count(inv4),
 "B1 = +WF1 (sub never missing)"                     : count(inv4+[wf_sub_not_missing]),
 "B2 = +WF2 (bundled ⟺ present copy)"                : count(inv4+[wf_sub_not_missing,wf_bundled_has_copy]),
 "B3 = +StaleParentExcluded"                         : count(inv4+[wf_sub_not_missing,wf_bundled_has_copy,stale_parent_excluded]),
}
print()
A = {(0,'Bun','PNM','Bun','PNM'),(1,'Bun','PNM','Bun','PNM'),(1,'Bun','PNM','Pri','PNM'),
     (1,'Pri','N','Pri','N'),(1,'Pri','PNM','Bun','PNM'),(1,'Pri','PNM','Pri','PNM')}
for name,S in sets.items():
    Sset=set(S)
    print(f"{name:42s} |.|={len(S):3d}   ⊇A? {A<=Sset}   gap(.∖A)={len(Sset-A)}")
print()
print("B3 ∖ A (still permitted but unreachable):")
B3=set(count(inv4+[wf_sub_not_missing,wf_bundled_has_copy,stale_parent_excluded]))
for s in sorted(B3-A): print("   ",s)

# --- candidate 5th conjunct: sub-presence is set atomically in Phase2 (both or neither) ---
def sub_presence_uniform(s):
    _,_,s1,_,s2 = s
    return (s1!='N') == (s2!='N')

A = {(0,'Bun','PNM','Bun','PNM'),(1,'Bun','PNM','Bun','PNM'),(1,'Bun','PNM','Pri','PNM'),
     (1,'Pri','N','Pri','N'),(1,'Pri','PNM','Bun','PNM'),(1,'Pri','PNM','Pri','PNM')}

B4 = set(count(inv4+[wf_sub_not_missing,wf_bundled_has_copy,stale_parent_excluded,sub_presence_uniform]))
print()
print(f"B4 = B3 + SubPresenceUniform : |B4| = {len(B4)}")
print(f"   B4 == A ?  {B4 == A}")
print(f"   B4 ⊇ A ?   {A <= B4}    B4 ∖ A = {sorted(B4 - A)}    A ∖ B4 = {sorted(A - B4)}")
