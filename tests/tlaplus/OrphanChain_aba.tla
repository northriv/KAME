(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanChain_atomicshared.tla.
 ***************************************************************************)
---------------------------- MODULE OrphanChain_aba ----------------------------
(*
 * The orphan chain with its CAS sites split the way allocator.cpp splits them,
 * and with any number of concurrent consumers.
 *
 * OrphanChain_adopt.tla models AdoptPop as ONE atomic step (head' = nxt[head])
 * and admits one adopter and one scrubber at a time (pop_ref / scrub_pin are
 * single variables).  The code is three steps -- load head, load
 * head->m_orphan_next, CAS head -- and runs in every thread that reaches
 * allocate_chunk_path.  Between the second and third step another thread can
 * pop the same chunk, adopt it, exit, and push it back, so the head holds the
 * same pointer again with a different successor.  A pointer-only CAS then
 * succeeds with the stale successor.  Observed on an M5 Ultra (2026-10-08):
 * a chunk adopted by two threads and linked twice, and a cycle
 * E -> S -> C -> E that orphan_chain_scrub walks forever.
 *
 * Serial = TRUE models the fix: atomic_shared_ptr bumps a serial in the word on
 * every write that stores a pointer (tag-only updates leave it alone), and a
 * local_shared_ptr / scoped view loaded from the word keeps the serial it saw,
 * so a CAS expecting it fails if the word was written since -- even when the
 * pointer is the same again.  Only "written since my load?" matters to a CAS,
 * so each loaded value carries the word it came from (src) and a dirty flag set
 * by any later write to that word: an unbounded, non-wrapping serial without
 * the serial values in the state.  A CAS whose expected value came from a
 * different word fails in this model; the code avoids that case by reloading
 * cur from the predecessor's word after a scrub unlink (ScReload).
 * Serial = FALSE is the shipped pointer-only CAS.
 *
 * Out of scope here (covered by OrphanChain_adopt / _atomicshared): refcounts,
 * pins and disposal.  Unlinked dead chunks simply stay unreachable; no address
 * is reused, which is what the pins guarantee.
 *
 * Expected (run_orphan_chain.sh, 2 threads x 3 chunks): Serial = FALSE
 * violates Inv_NoStrandedLive at depth 13 -- the stale-null-next case seen on
 * the M5 Ultra; Serial = TRUE is clean (4.05M distinct states, depth 57).
 * OrphanChain_aba_serial_3t_mc.cfg is the same check with 3 threads, a long
 * run kept out of the regression script: clean, 1,500,210,294 distinct
 * states, depth 73, 49 min on an M5 Ultra (-Xmx24g, 44 GB off-heap
 * fingerprints: -XX:MaxDirectMemorySize=44g -fpmem 0.9 and
 * -Dtlc2.tool.fp.FPSet.impl=tlc2.tool.fp.OffHeapDiskFPSet).  At that size
 * TLC's own estimate of a fingerprint collision having hidden a state is
 * 0.2 (0.55 optimistic) for that run (-fp 59).  A second run with -fp 7 is
 * also clean and finds exactly the same 1,500,210,294 distinct states
 * (estimate 0.1): a collision that hid states would have changed the count
 * under a different fingerprint polynomial.
 *)

EXTENDS Naturals, FiniteSets

CONSTANTS Nodes, Threads, NIL, NONE, HEADREF, Serial, MaxPush

ASSUME NIL \notin Nodes /\ NONE \notin Threads /\ HEADREF \notin Nodes
ASSUME Serial \in BOOLEAN

VARIABLES
    head,      \* the chain head's pointer
    nx,        \* [Nodes -> Nodes \cup {NIL}]  each chunk's m_orphan_next
    own,       \* [Nodes -> Threads \cup {NONE}]  BIT_OWNED + owner
    filled,    \* [Nodes -> BOOLEAN]  MASK_CNT != 0
    pc,        \* per-thread program counter
    cur,       \* [Threads -> ptr]  pop: loaded head / scrub: loaded cur / push: loaded head
    csrc,      \* [Threads -> word id]  the word cur was loaded from
    cdirty,    \* [Threads -> BOOLEAN]  that word has been written since
    nv,        \* [Threads -> ptr]  loaded cur->m_orphan_next
    nsrc,      \* [Threads -> word id]
    ndirty,    \* [Threads -> BOOLEAN]
    pred,      \* [Threads -> Nodes \cup {HEADREF}]  scrub predecessor (its word)
    pushing,   \* [Threads -> Nodes \cup {NIL}]  chunk being pushed at owner exit
    npush,     \* pushes so far (bound)
    dupClaim   \* a pop claimed a chunk somebody already owns

vars == <<head, nx, own, filled, pc, cur, csrc, cdirty, nv, nsrc, ndirty,
          pred, pushing, npush, dupClaim>>

Words == Nodes \cup {HEADREF}           \* HEADREF = the head; n = nx[n]
Ptr(w) == IF w = HEADREF THEN head ELSE nx[w]

\* Effect of writing the words in WS on every reader's saved value.
Dirt(src, dirty, WS) == [t \in Threads |-> dirty[t] \/ src[t] \in WS]
MarkWritten(WS) ==
    /\ cdirty' = Dirt(csrc, cdirty, WS)
    /\ ndirty' = Dirt(nsrc, ndirty, WS)

\* A CAS on word w expecting thread t's cur.
Match(w, t) == Ptr(w) = cur[t] /\ (Serial => csrc[t] = w /\ ~cdirty[t])

PCs == {"idle", "popNext", "popCas", "pushStore", "pushCas",
        "scRead", "scCheck", "scCas", "scReload"}

TypeOK ==
    /\ head \in Nodes \cup {NIL}
    /\ nx \in [Nodes -> Nodes \cup {NIL}]
    /\ own \in [Nodes -> Threads \cup {NONE}]
    /\ filled \in [Nodes -> BOOLEAN]
    /\ pc \in [Threads -> PCs]
    /\ pushing \in [Threads -> Nodes \cup {NIL}]

RECURSIVE Walk(_, _)
Walk(p, k) == IF p = NIL \/ k = 0 THEN {} ELSE {p} \cup Walk(nx[p], k - 1)
Reach == Walk(head, Cardinality(Nodes) + 1)

RECURSIVE Hop(_, _)
Hop(p, k) == IF p = NIL \/ k = 0 THEN p ELSE Hop(nx[p], k - 1)

\* Initial state: n1 -> n2 on the chain, n3 owned by the first thread.
Init ==
    LET n1 == CHOOSE n \in Nodes : TRUE
        n2 == CHOOSE n \in Nodes \ {n1} : TRUE
        n3 == CHOOSE n \in Nodes \ {n1, n2} : TRUE
        t1 == CHOOSE t \in Threads : TRUE
    IN /\ head = n1
       /\ nx = [n \in Nodes |-> IF n = n1 THEN n2 ELSE NIL]
       /\ own = [n \in Nodes |-> IF n = n3 THEN t1 ELSE NONE]
       /\ filled = [n \in Nodes |-> TRUE]
       /\ pc = [t \in Threads |-> "idle"]
       /\ cur = [t \in Threads |-> NIL]
       /\ csrc = [t \in Threads |-> NIL]
       /\ cdirty = [t \in Threads |-> FALSE]
       /\ nv = [t \in Threads |-> NIL]
       /\ nsrc = [t \in Threads |-> NIL]
       /\ ndirty = [t \in Threads |-> FALSE]
       /\ pred = [t \in Threads |-> HEADREF]
       /\ pushing = [t \in Threads |-> NIL]
       /\ npush = 0
       /\ dupClaim = FALSE

\* Thread t loads word w into cur.
LoadCur(t, w) ==
    /\ cur' = [cur EXCEPT ![t] = Ptr(w)]
    /\ csrc' = [csrc EXCEPT ![t] = w]

----------------------------------------------------------------------------
\* orphan_chain_pop: old(head); nxt(old->m_orphan_next); head.CAS(old, nxt)

PopLoad(t) ==
    /\ pc[t] = "idle" /\ head # NIL
    /\ LoadCur(t, HEADREF)
    /\ cdirty' = [cdirty EXCEPT ![t] = FALSE]
    /\ pc' = [pc EXCEPT ![t] = "popNext"]
    /\ UNCHANGED <<head, nx, own, filled, nv, nsrc, ndirty, pred, pushing, npush, dupClaim>>

PopNext(t) ==
    /\ pc[t] = "popNext"
    /\ nv' = [nv EXCEPT ![t] = nx[cur[t]]]
    /\ nsrc' = [nsrc EXCEPT ![t] = cur[t]]
    /\ ndirty' = [ndirty EXCEPT ![t] = FALSE]
    /\ pc' = [pc EXCEPT ![t] = "popCas"]
    /\ UNCHANGED <<head, nx, own, filled, cur, csrc, cdirty, pred, pushing, npush, dupClaim>>

\* Success: head moves to the loaded successor, the popped chunk's own link is
\* cleared (a store), and the chunk is claimed (BIT_OWNED).  Failure: the code's
\* compareAndSwap reloads `old` and loops -- modelled as starting over.
PopCas(t) ==
    /\ pc[t] = "popCas"
    /\ LET e == cur[t] IN
       IF Match(HEADREF, t)
       THEN /\ head' = nv[t]
            /\ nx' = [nx EXCEPT ![e] = NIL]
            /\ MarkWritten({HEADREF, e})
            /\ IF own[e] # NONE
               THEN /\ dupClaim' = TRUE /\ UNCHANGED own
               ELSE /\ own' = [own EXCEPT ![e] = t] /\ UNCHANGED dupClaim
       ELSE UNCHANGED <<head, nx, own, dupClaim, cdirty, ndirty>>
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<filled, cur, csrc, nv, nsrc, pred, pushing, npush>>

----------------------------------------------------------------------------
\* Owner exit (release_dll_chunks_for_thread): a non-empty chunk drops
\* BIT_OWNED and is pushed -- old(head); c->m_orphan_next = old; head.CAS(old, c).

PushStart(t, n) ==
    /\ pc[t] = "idle" /\ own[n] = t /\ filled[n] /\ npush < MaxPush
    /\ own' = [own EXCEPT ![n] = NONE]
    /\ pushing' = [pushing EXCEPT ![t] = n]
    /\ LoadCur(t, HEADREF)
    /\ cdirty' = [cdirty EXCEPT ![t] = FALSE]
    /\ npush' = npush + 1
    /\ pc' = [pc EXCEPT ![t] = "pushStore"]
    /\ UNCHANGED <<head, nx, filled, nv, nsrc, ndirty, pred, dupClaim>>

PushStore(t) ==
    /\ pc[t] = "pushStore"
    /\ nx' = [nx EXCEPT ![pushing[t]] = cur[t]]
    /\ MarkWritten({pushing[t]})
    /\ pc' = [pc EXCEPT ![t] = "pushCas"]
    /\ UNCHANGED <<head, own, filled, cur, csrc, nv, nsrc, pred, pushing, npush, dupClaim>>

PushCas(t) ==
    /\ pc[t] = "pushCas"
    /\ IF Match(HEADREF, t)
       THEN /\ head' = pushing[t]
            /\ MarkWritten({HEADREF})
            /\ pushing' = [pushing EXCEPT ![t] = NIL]
            /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ UNCHANGED <<cur, csrc>>
       ELSE /\ LoadCur(t, HEADREF)                      \* compareAndSwap reloads old
            /\ cdirty' = [cdirty EXCEPT ![t] = FALSE]
            /\ pc' = [pc EXCEPT ![t] = "pushStore"]
            /\ UNCHANGED <<head, pushing, ndirty>>
    /\ UNCHANGED <<nx, own, filled, nv, nsrc, pred, npush, dupClaim>>

----------------------------------------------------------------------------
\* Slot traffic.  An owner allocates and frees freely; an orphan only drains
\* (cross-thread frees), it never refills.

OwnerToggle(n) ==
    /\ own[n] # NONE
    /\ filled' = [filled EXCEPT ![n] = ~filled[n]]
    /\ UNCHANGED <<head, nx, own, pc, cur, csrc, cdirty, nv, nsrc, ndirty,
                   pred, pushing, npush, dupClaim>>

OrphanDrain(n) ==
    /\ own[n] = NONE /\ filled[n]
    /\ filled' = [filled EXCEPT ![n] = FALSE]
    /\ UNCHANGED <<head, nx, own, pc, cur, csrc, cdirty, nv, nsrc, ndirty,
                   pred, pushing, npush, dupClaim>>

----------------------------------------------------------------------------
\* orphan_chain_scrub: walk with pred/cur; unlink a dead (empty) chunk with a
\* CAS on pred's word (or the head); a lost CAS restarts from the head; after
\* an unlink, cur is reloaded from pred's word.

ScStart(t) ==
    /\ pc[t] = "idle"
    /\ LoadCur(t, HEADREF)
    /\ cdirty' = [cdirty EXCEPT ![t] = FALSE]
    /\ pred' = [pred EXCEPT ![t] = HEADREF]
    /\ pc' = [pc EXCEPT ![t] = "scRead"]
    /\ UNCHANGED <<head, nx, own, filled, nv, nsrc, ndirty, pushing, npush, dupClaim>>

ScRead(t) ==
    /\ pc[t] = "scRead"
    /\ IF cur[t] = NIL
       THEN /\ pc' = [pc EXCEPT ![t] = "idle"] /\ UNCHANGED <<nv, nsrc, ndirty>>
       ELSE /\ nv' = [nv EXCEPT ![t] = nx[cur[t]]]
            /\ nsrc' = [nsrc EXCEPT ![t] = cur[t]]
            /\ ndirty' = [ndirty EXCEPT ![t] = FALSE]
            /\ pc' = [pc EXCEPT ![t] = "scCheck"]
    /\ UNCHANGED <<head, nx, own, filled, cur, csrc, cdirty, pred, pushing, npush, dupClaim>>

ScCheck(t) ==
    /\ pc[t] = "scCheck"
    /\ IF filled[cur[t]]
       THEN /\ pred' = [pred EXCEPT ![t] = cur[t]]       \* live: keep, advance
            /\ cur' = [cur EXCEPT ![t] = nv[t]]
            /\ csrc' = [csrc EXCEPT ![t] = nsrc[t]]
            /\ cdirty' = [cdirty EXCEPT ![t] = ndirty[t]]
            /\ pc' = [pc EXCEPT ![t] = "scRead"]
       ELSE /\ pc' = [pc EXCEPT ![t] = "scCas"]          \* dead: unlink next
            /\ UNCHANGED <<cur, csrc, cdirty, pred>>
    /\ UNCHANGED <<head, nx, own, filled, nv, nsrc, ndirty, pushing, npush, dupClaim>>

ScCas(t) ==
    /\ pc[t] = "scCas"
    /\ LET w == pred[t] IN
       IF Match(w, t)
       THEN /\ IF w = HEADREF
               THEN head' = nv[t] /\ UNCHANGED nx
               ELSE nx' = [nx EXCEPT ![w] = nv[t]] /\ UNCHANGED head
            /\ MarkWritten({w})
            /\ pc' = [pc EXCEPT ![t] = "scReload"]
            /\ UNCHANGED <<cur, csrc, pred>>
       ELSE /\ LoadCur(t, HEADREF)                       \* lost: restart from head
            /\ cdirty' = [cdirty EXCEPT ![t] = FALSE]
            /\ pred' = [pred EXCEPT ![t] = HEADREF]
            /\ pc' = [pc EXCEPT ![t] = "scRead"]
            /\ UNCHANGED <<head, nx, ndirty>>
    /\ UNCHANGED <<own, filled, nv, nsrc, pushing, npush, dupClaim>>

ScReload(t) ==
    /\ pc[t] = "scReload"
    /\ LoadCur(t, pred[t])
    /\ cdirty' = [cdirty EXCEPT ![t] = FALSE]
    /\ pc' = [pc EXCEPT ![t] = "scRead"]
    /\ UNCHANGED <<head, nx, own, filled, nv, nsrc, ndirty, pred, pushing, npush, dupClaim>>

----------------------------------------------------------------------------

Next ==
    \/ \E t \in Threads :
        \/ PopLoad(t) \/ PopNext(t) \/ PopCas(t)
        \/ \E n \in Nodes : PushStart(t, n)
        \/ PushStore(t) \/ PushCas(t)
        \/ ScStart(t) \/ ScRead(t) \/ ScCheck(t) \/ ScCas(t) \/ ScReload(t)
    \/ \E n \in Nodes : OwnerToggle(n) \/ OrphanDrain(n)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
\* Safety.

\* A pop never claims a chunk that is already owned.
Inv_NoDupClaim == ~dupClaim

\* An owned chunk is never reachable from the head.
Inv_OwnedOffChain == \A n \in Nodes : own[n] # NONE => n \notin Reach

\* The chain ends.
Inv_Acyclic == Hop(head, Cardinality(Nodes) + 1) = NIL

\* A live orphan (not owned, not mid-push, still has slots) stays reachable:
\* losing it strands its free slots, the leak the chain exists to prevent.
Inv_NoStrandedLive ==
    \A n \in Nodes :
        (own[n] = NONE /\ filled[n] /\ \A t \in Threads : pushing[t] # n)
            => n \in Reach

=============================================================================
