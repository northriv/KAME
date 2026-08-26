---------------------------- MODULE OrphanChain ----------------------------
(***************************************************************************)
(* kamepoolalloc's orphan-chunk reclaim chain -- the pool's ONLY use of     *)
(* atomic_shared_ptr, and the one ablation in the whole UAF investigation   *)
(* that ELIMINATED the fault (§13.126: 0/20 against 15/20, p = 7.7e-7).     *)
(*                                                                         *)
(* Why a model and not more reading.  §13.116 and §13.125 cleared every     *)
(* mechanism that source reading can settle: the claim loops are            *)
(* single-load, the word-cache cells are owner-confined and drained before  *)
(* the orphan push, MASK_CNT pins a chunk while any bit is set, and         *)
(* orphan_chain_push MOVES the self-ref (a store(1) would clobber a         *)
(* residual scrub pin -- the code says so).  What reading cannot settle is  *)
(* the interaction of THREE reference kinds (chain-ref, self-ref, scrub     *)
(* pin) with a Treiber stack whose scrub UNLINKS nodes and whose adopt      *)
(* REVIVES them.  That is what this models.                                *)
(*                                                                         *)
(* The property that matters is not "the refcount is right" but its         *)
(* consequence: the chunk OBJECT and the chunk STORAGE share a lifetime, so *)
(* a refcount reaching zero while a slot is live releases the region under  *)
(* live objects -- and the region is then handed to another size class.     *)
(* That is exactly §13.104's DOUBLE-LIVE, §13.110's per-chunk clustering    *)
(* and §13.113's "previous occupant live and unfreed".                      *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Chunks,          \* finite set of chunk ids
    Threads,         \* finite set of thread ids
    MaxLive,         \* per-chunk live-slot bound (1 suffices: 0 vs >0 is what matters)
    MaxRef,          \* refcount bound; a step that would exceed it is disabled
                     \* (preconditions, not a StateConstraint -- the latter
                     \*  silently disables actions and has already produced one
                     \*  bogus "exhaustive PASS" in this investigation)
    BUG_STORE1,      \* push does refcnt := 1 instead of MOVEing the self-ref
    BUG_SCRUB_STALE, \* scrub unlinks on its (possibly stale) earlier read
    BUG_EXTRA_RELEASER, \* the cross-thread last-slot freer also releases
    NIL              \* model value standing for the null pointer

VARIABLES
    head,       \* Chunks \cup {NIL}
    nxt,        \* [Chunks -> Chunks \cup {NIL}]
    onChain,    \* [Chunks -> BOOLEAN]  (reachable from head)
    owned,      \* [Chunks -> BOOLEAN]  (BIT_OWNED)
    selfref,    \* [Chunks -> BOOLEAN]  (m_owner_self_ref holds itself)
    refcnt,     \* [Chunks -> Nat]      as the C++ maintains it
    live,       \* [Chunks -> Nat]      live slots  (MASK_CNT /= 0 <=> live > 0)
    disposed,   \* [Chunks -> BOOLEAN]  region released
    pin,        \* [Threads -> Chunks \cup {NIL}]   a thread's local_shared_ptr
    ownerOf,    \* [Chunks -> Threads \cup {NIL}]  which thread's DLL holds c
    pc,         \* [Threads -> STRING]
    obs         \* [Threads -> Nat]  what a scrubber last READ for live[pin]

vars == <<head, nxt, onChain, owned, selfref, refcnt, live, disposed,
          pin, ownerOf, pc, obs>>

(*************************** the derived truth ****************************)
(* The reference count the protocol OUGHT to have: one per chain link that  *)
(* names c (the head, or a predecessor's next), one for a live self-ref,    *)
(* one per thread holding c.                                                *)
ChainRefs(c) == (IF head = c THEN 1 ELSE 0)
              + Cardinality({d \in Chunks : nxt[d] = c /\ onChain[d]})
SelfRefs(c)  == IF selfref[c] THEN 1 ELSE 0
PinRefs(c)   == Cardinality({t \in Threads : pin[t] = c})
Derived(c)   == ChainRefs(c) + SelfRefs(c) + PinRefs(c)

TypeOK ==
    /\ head \in Chunks \cup {NIL}
    /\ nxt \in [Chunks -> Chunks \cup {NIL}]
    /\ onChain \in [Chunks -> BOOLEAN]
    /\ owned \in [Chunks -> BOOLEAN]
    /\ selfref \in [Chunks -> BOOLEAN]
    /\ refcnt \in [Chunks -> 0..MaxRef]
    /\ live \in [Chunks -> 0..MaxLive]
    /\ disposed \in [Chunks -> BOOLEAN]
    /\ pin \in [Threads -> Chunks \cup {NIL}]
    /\ ownerOf \in [Chunks -> Threads \cup {NIL}]
    /\ obs \in [Threads -> 0..MaxLive]

Init ==
    /\ head = NIL
    /\ nxt = [c \in Chunks |-> NIL]
    /\ onChain = [c \in Chunks |-> FALSE]
    /\ owned = [c \in Chunks |-> FALSE]
    /\ ownerOf = [c \in Chunks |-> NIL]
    /\ selfref = [c \in Chunks |-> FALSE]
    /\ refcnt = [c \in Chunks |-> 0]
    /\ live = [c \in Chunks |-> 0]
    /\ disposed = [c \in Chunks |-> FALSE]
    /\ pin = [t \in Threads |-> NIL]
    /\ pc = [t \in Threads |-> "idle"]
    /\ obs = [t \in Threads |-> 0]

(**************************** owner-side steps ****************************)
\* create_allocator: a never-refcounted, unowned chunk is handed to a thread.
FreshClaim(t, c) ==
    /\ pc[t] = "idle"
    /\ ~owned[c] /\ ~onChain[c] /\ ~disposed[c]
    /\ refcnt[c] = 0 /\ live[c] = 0 /\ ~selfref[c]
    /\ \A u \in Threads : pin[u] /= c
    /\ owned' = [owned EXCEPT ![c] = TRUE]
    /\ ownerOf' = [ownerOf EXCEPT ![c] = t]
    /\ UNCHANGED <<head, nxt, onChain, selfref, refcnt, live, disposed, pin, pc, obs>>

\* Allocate a slot from an owned chunk.  Only the owner allocates, which is
\* the invariant the scrub's "orphans never refill" comment rests on.
Allocate(t, c) ==
    /\ pc[t] = "idle"
    /\ ownerOf[c] = t          \* only the owner allocates from a chunk
    /\ owned[c] /\ ~disposed[c] /\ live[c] < MaxLive
    /\ live' = [live EXCEPT ![c] = @ + 1]
    /\ UNCHANGED <<head, nxt, onChain, owned, selfref, refcnt, disposed, pin, ownerOf, pc, obs>>

\* A cross-thread free of one live slot.  Legal whether or not c is owned.
CrossFree(t, c) ==
    /\ pc[t] = "idle"
    /\ ~disposed[c] /\ live[c] > 0
    /\ live' = [live EXCEPT ![c] = @ - 1]
    /\ IF BUG_EXTRA_RELEASER /\ live[c] = 1 /\ ~owned[c]
         THEN \* the FS=false "last-slot returner releases" path, wrongly
              \* reached for a chunk that is still on the chain
              /\ disposed' = [disposed EXCEPT ![c] = TRUE]
         ELSE /\ disposed' = disposed
    /\ UNCHANGED <<head, nxt, onChain, owned, selfref, refcnt, pin, ownerOf, pc, obs>>

(**************************** owner exit: push ****************************)
\* Empty chunks are released directly and never reach the chain, so the push
\* precondition is live > 0 -- the chain-only no-release invariant.
PushTake(t, c) ==
    /\ pc[t] = "idle"
    \* release_dll_chunks_for_thread walks THIS thread's DLL, so a thread can
    \* only orphan a chunk it owns.  Omitting this let a second thread orphan a
    \* chunk mid-adoption -- the model's first violation, and an artifact.
    /\ ownerOf[c] = t
    /\ owned[c] /\ ~onChain[c] /\ ~disposed[c] /\ live[c] > 0
    /\ ownerOf' = [ownerOf EXCEPT ![c] = NIL]
    /\ owned' = [owned EXCEPT ![c] = FALSE]
    /\ pin' = [pin EXCEPT ![t] = c]
    /\ IF selfref[c]
         THEN \* re-orphaning: MOVE self-ref -> the hold.  refcnt unchanged,
              \* which is what preserves a residual scrub pin's count.
              /\ selfref' = [selfref EXCEPT ![c] = FALSE]
              /\ refcnt' = IF BUG_STORE1
                             THEN [refcnt EXCEPT ![c] = 1]   \* clobbers the pin
                             ELSE refcnt
         ELSE \* first orphaning: establish 1 and adopt
              /\ selfref' = selfref
              /\ refcnt' = [refcnt EXCEPT ![c] = 1]
    /\ pc' = [pc EXCEPT ![t] = "push_cas"]
    /\ UNCHANGED <<head, nxt, onChain, live, disposed, obs>>

PushCas(t) ==
    /\ pc[t] = "push_cas"
    /\ LET c == pin[t] IN
       /\ c /= NIL
       /\ refcnt[c] < MaxRef
       /\ nxt' = [nxt EXCEPT ![c] = head]
       /\ head' = c
       /\ onChain' = [onChain EXCEPT ![c] = TRUE]
       \* hold released into the chain-ref: +1 for the chain, -1 for the hold
       /\ refcnt' = refcnt
       /\ pin' = [pin EXCEPT ![t] = NIL]
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<owned, selfref, live, disposed, ownerOf, obs>>

(*************************** adopt: pop + claim ***************************)
PopTake(t) ==
    /\ pc[t] = "idle"
    /\ head /= NIL
    /\ LET c == head IN
       /\ ~disposed[c]
       /\ refcnt[c] < MaxRef
       /\ head' = nxt[c]
       /\ onChain' = [onChain EXCEPT ![c] = FALSE]
       /\ nxt' = [nxt EXCEPT ![c] = NIL]
       \* chain-ref becomes the caller's hold: net refcnt unchanged
       /\ refcnt' = refcnt
       /\ pin' = [pin EXCEPT ![t] = c]
    /\ pc' = [pc EXCEPT ![t] = "adopt_claim"]
    /\ UNCHANGED <<owned, selfref, live, disposed, ownerOf, obs>>

AdoptClaim(t) ==
    /\ pc[t] = "adopt_claim"
    /\ LET c == pin[t] IN
       /\ c /= NIL
       /\ IF owned[c]
            THEN \* duplicate-owned: discard (the defensive break in the C++)
                 /\ owned' = owned /\ ownerOf' = ownerOf
                 /\ pc' = [pc EXCEPT ![t] = "adopt_drop"]
            ELSE /\ owned' = [owned EXCEPT ![c] = TRUE]
                 /\ ownerOf' = [ownerOf EXCEPT ![c] = t]
                 /\ pc' = [pc EXCEPT ![t] = "adopt_move"]
    /\ UNCHANGED <<head, nxt, onChain, selfref, refcnt, live, disposed, pin, obs>>

\* MOVE the hold into the chunk's self-ref: refcnt unchanged, hold released.
AdoptMove(t) ==
    /\ pc[t] = "adopt_move"
    /\ LET c == pin[t] IN
       /\ c /= NIL
       /\ selfref' = [selfref EXCEPT ![c] = TRUE]
       /\ refcnt' = refcnt
       /\ pin' = [pin EXCEPT ![t] = NIL]
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<head, nxt, onChain, owned, live, disposed, ownerOf, obs>>

AdoptDrop(t) ==
    /\ pc[t] = "adopt_drop"
    /\ LET c == pin[t] IN
       /\ c /= NIL
       /\ refcnt' = [refcnt EXCEPT ![c] = IF @ > 0 THEN @ - 1 ELSE 0]
       /\ disposed' = [disposed EXCEPT ![c] = IF refcnt[c] = 1 THEN TRUE ELSE @]
       /\ pin' = [pin EXCEPT ![t] = NIL]
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<head, nxt, onChain, owned, selfref, live, ownerOf, obs>>

(********************************* scrub **********************************)
\* Pin a chain node and READ its live count -- the two are separate steps,
\* which is the whole point: the C++ reads m_flags_packed and then CASes.
ScrubRead(t, c) ==
    /\ pc[t] = "idle"
    /\ onChain[c] /\ ~disposed[c] /\ refcnt[c] < MaxRef
    /\ pin' = [pin EXCEPT ![t] = c]
    /\ refcnt' = [refcnt EXCEPT ![c] = @ + 1]      \* the scrub pin
    /\ obs' = [obs EXCEPT ![t] = live[c]]
    /\ pc' = [pc EXCEPT ![t] = "scrub_cas"]
    /\ UNCHANGED <<head, nxt, onChain, owned, selfref, live, disposed, ownerOf>>

ScrubCas(t) ==
    /\ pc[t] = "scrub_cas"
    /\ LET c == pin[t] IN
       /\ c /= NIL
       /\ \* unlink only a node the scrub believes empty.  Without the bug the
          \* decision re-reads live[c]; with it, the stale `obs` is trusted.
          IF BUG_SCRUB_STALE THEN obs[t] = 0 ELSE live[c] = 0
       /\ onChain[c]
       /\ \* CAS-unlink from head or from the predecessor that still names c
          \/ /\ head = c
             /\ head' = nxt[c]
             /\ nxt' = nxt
          \/ /\ head /= c
             /\ \E p \in Chunks : nxt[p] = c /\ onChain[p]
                /\ nxt' = [nxt EXCEPT ![p] = nxt[c]]
             /\ head' = head
       /\ onChain' = [onChain EXCEPT ![c] = FALSE]
       /\ refcnt' = [refcnt EXCEPT ![c] = IF @ > 0 THEN @ - 1 ELSE 0]  \* chain-ref gone
       /\ disposed' = [disposed EXCEPT ![c] = IF refcnt[c] = 1 THEN TRUE ELSE @]
    /\ pc' = [pc EXCEPT ![t] = "scrub_release"]
    /\ UNCHANGED <<owned, selfref, live, pin, ownerOf, obs>>

\* The scrub keeps walking; releasing its pin is a separate step, and this is
\* where the C++ comment says refcnt hits 0 -> atomic_intrusive_dispose.
ScrubRelease(t) ==
    /\ pc[t] \in {"scrub_cas", "scrub_release"}
    /\ LET c == pin[t] IN
       /\ c /= NIL
       /\ refcnt' = [refcnt EXCEPT ![c] = IF @ > 0 THEN @ - 1 ELSE 0]
       /\ disposed' = [disposed EXCEPT ![c] = IF refcnt[c] = 1 THEN TRUE ELSE @]
       /\ pin' = [pin EXCEPT ![t] = NIL]
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<head, nxt, onChain, owned, selfref, live, ownerOf, obs>>

(**************************************************************************)
(* Dispose is NOT a free-standing action.  atomic_intrusive_dispose fires   *)
(* on the 1 -> 0 transition of refcnt, i.e. inside the decrement that       *)
(* releases the last reference -- so it is folded into every decrementing   *)
(* step above.  Modelling it separately let a fresh, owned, never-chained   *)
(* chunk (refcnt 0 because it was never refcounted) be "disposed", which is *)
(* not a thing the code can do and produced a spurious violation on the     *)
(* first run.                                                              *)
(**************************************************************************)

Next ==
    \/ \E t \in Threads, c \in Chunks : FreshClaim(t, c) \/ Allocate(t, c)
                                     \/ CrossFree(t, c)
                                     \/ PushTake(t, c) \/ ScrubRead(t, c)
    \/ \E t \in Threads : PushCas(t) \/ PopTake(t) \/ AdoptClaim(t)
                       \/ AdoptMove(t) \/ AdoptDrop(t)
                       \/ ScrubCas(t) \/ ScrubRelease(t)

Spec == Init /\ [][Next]_vars

(******************************* properties *******************************)
\* THE property.  Object and storage share a lifetime, so disposing a chunk
\* with a live slot releases the region under live objects -- the fault.
NoDisposeWithLive == \A c \in Chunks : disposed[c] => live[c] = 0

\* Nothing may still name a disposed chunk.
NoUseAfterDispose ==
    \A c \in Chunks : disposed[c] =>
        /\ ~onChain[c] /\ ~owned[c] /\ ~selfref[c] /\ head /= c
        /\ \A t \in Threads : pin[t] /= c

\* The refcount the code maintains must equal the references that exist.
RefcntAgrees ==
    \A c \in Chunks : (~disposed[c] /\ pc \in [Threads -> {"idle"}])
                        => refcnt[c] = Derived(c)

\* An owned chunk is off the chain: what makes "orphans never refill" true.
OwnedNotOnChain == \A c \in Chunks : owned[c] => ~onChain[c]

Inv == TypeOK /\ NoDisposeWithLive /\ NoUseAfterDispose /\ OwnedNotOnChain
=============================================================================
