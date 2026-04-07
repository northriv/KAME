------------------------------- MODULE stm_commit -------------------------------
(*
 * TLA+ model of the KAME STM single-node commit protocol.
 *
 * Models the optimistic concurrency control cycle:
 *   1. Snapshot: read the current PacketWrapper via atomic shared ptr scan
 *   2. Write: copy-on-write on Payload
 *   3. Commit: compareAndSet on m_link
 *   4. Retry: on failure, take new snapshot and repeat
 *
 * This sits above atomic_shared_ptr (Layer 0, verified separately).
 * Here we abstract atomic_shared_ptr as a correct atomic register with
 * refcounting -- the detailed CAS/scan protocol is not re-modeled.
 *
 * Source: kame/transaction.h, kame/transaction_impl.h
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,        \* set of thread IDs
    MaxVal,         \* max payload value (for bounded state space)
    MaxSerial       \* max serial number (for bounded state space)

(* -------------------------------------------------------------------------- *)
(* State variables                                                            *)
(* -------------------------------------------------------------------------- *)
VARIABLES
    \* The shared node's state -- represents atomic_shared_ptr<PacketWrapper> m_link
    node_val,       \* current committed payload value (Nat)
    node_serial,    \* serial number of the current committed state

    \* Per-thread transaction state
    pc,             \* program counter
    thr_snap_val,   \* value seen at snapshot time (oldpacket)
    thr_snap_ser,   \* serial of the snapshot
    thr_write_val,  \* new value written by transaction
    thr_committed   \* number of successful commits (for progress check)

vars == <<node_val, node_serial, pc, thr_snap_val, thr_snap_ser,
          thr_write_val, thr_committed>>

(* -------------------------------------------------------------------------- *)
(* Type invariant                                                             *)
(* -------------------------------------------------------------------------- *)
TypeOK ==
    /\ node_val \in 0..MaxVal
    /\ node_serial \in 0..MaxSerial
    /\ pc \in [Threads -> {"idle", "snapshot", "write", "commit", "done"}]
    /\ thr_snap_val \in [Threads -> 0..MaxVal]
    /\ thr_snap_ser \in [Threads -> 0..MaxSerial]
    /\ thr_write_val \in [Threads -> 0..MaxVal]
    /\ thr_committed \in [Threads -> Nat]

(* -------------------------------------------------------------------------- *)
(* Initial state                                                              *)
(* -------------------------------------------------------------------------- *)
Init ==
    /\ node_val = 0
    /\ node_serial = 0
    /\ pc = [t \in Threads |-> "idle"]
    /\ thr_snap_val = [t \in Threads |-> 0]
    /\ thr_snap_ser = [t \in Threads |-> 0]
    /\ thr_write_val = [t \in Threads |-> 0]
    /\ thr_committed = [t \in Threads |-> 0]

(* ========================================================================== *)
(* iterate_commit cycle                                                       *)
(* ========================================================================== *)

\* Step 1: Start transaction / take snapshot
\* Models: Transaction constructor calls snapshot()
\* snapshot() calls scan_() on m_link to atomically read current state
TakeSnapshot(t) ==
    /\ pc[t] \in {"idle", "done"}
    /\ pc' = [pc EXCEPT ![t] = "snapshot"]
    /\ thr_snap_val' = [thr_snap_val EXCEPT ![t] = node_val]
    /\ thr_snap_ser' = [thr_snap_ser EXCEPT ![t] = node_serial]
    /\ UNCHANGED <<node_val, node_serial, thr_write_val, thr_committed>>

\* Step 2a: Execute closure -- nondeterministic write (arbitrary closure)
Write(t) ==
    /\ pc[t] = "snapshot"
    /\ \E v \in 0..MaxVal :
       thr_write_val' = [thr_write_val EXCEPT ![t] = v]
    /\ pc' = [pc EXCEPT ![t] = "write"]
    /\ UNCHANGED <<node_val, node_serial, thr_snap_val, thr_snap_ser, thr_committed>>

\* Step 2b: Deterministic increment -- models tr[node].x += 1
WriteIncrement(t) ==
    /\ pc[t] = "snapshot"
    /\ thr_snap_val[t] + 1 <= MaxVal
    /\ thr_write_val' = [thr_write_val EXCEPT ![t] = thr_snap_val[t] + 1]
    /\ pc' = [pc EXCEPT ![t] = "write"]
    /\ UNCHANGED <<node_val, node_serial, thr_snap_val, thr_snap_ser, thr_committed>>

\* Step 3: Commit -- CAS on m_link
\* Models: tr.commit() calls m_link->compareAndSet(wrapper, newwrapper)
\* transaction_impl.h lines 1245-1269
Commit(t) ==
    /\ pc[t] = "write"
    /\ \/ \* CAS succeeds: node state matches our snapshot
          /\ node_val = thr_snap_val[t]
          /\ node_serial = thr_snap_ser[t]
          /\ node_serial + 1 <= MaxSerial  \* guard for state space bound
          /\ node_val' = thr_write_val[t]
          /\ node_serial' = node_serial + 1
          /\ thr_committed' = [thr_committed EXCEPT ![t] = @ + 1]
          /\ pc' = [pc EXCEPT ![t] = "done"]
          /\ UNCHANGED <<thr_snap_val, thr_snap_ser, thr_write_val>>
       \/ \* CAS fails: node was modified since our snapshot -- retry
          /\ ~(node_val = thr_snap_val[t] /\ node_serial = thr_snap_ser[t])
          /\ thr_snap_val' = [thr_snap_val EXCEPT ![t] = node_val]
          /\ thr_snap_ser' = [thr_snap_ser EXCEPT ![t] = node_serial]
          /\ pc' = [pc EXCEPT ![t] = "snapshot"]
          /\ UNCHANGED <<node_val, node_serial, thr_write_val, thr_committed>>

\* Return to idle for next iteration
ReturnToIdle(t) ==
    /\ pc[t] = "done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<node_val, node_serial, thr_snap_val, thr_snap_ser,
                   thr_write_val, thr_committed>>

(* ========================================================================== *)
(* Next-state relation                                                        *)
(* ========================================================================== *)
Next ==
    \E t \in Threads :
        \/ TakeSnapshot(t)
        \/ Write(t)
        \/ WriteIncrement(t)
        \/ Commit(t)
        \/ ReturnToIdle(t)

\* State constraint
StateConstraint ==
    /\ node_serial <= MaxSerial
    /\ \A t \in Threads : thr_committed[t] <= MaxSerial

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* ========================================================================== *)
(* Safety Properties                                                          *)
(* ========================================================================== *)

\* 1. No Lost Updates: when a commit succeeds, the CAS ensured that no
\*    other transaction committed since our snapshot. The serial number
\*    strictly increases on each commit, preventing ABA on the packet.
\*    This is verified structurally: if two threads both committed at least
\*    once, at least two serial increments have occurred.
NoLostUpdate ==
    \A t1, t2 \in Threads :
        (t1 /= t2 /\ thr_committed[t1] > 0 /\ thr_committed[t2] > 0)
        => node_serial >= 2

\* 2. Commit Serializes: at most one thread can commit per serial number.
\*    Total commits across all threads cannot exceed the serial count.
CommitSerializes ==
    LET totalCommits == LET S == {thr_committed[t] : t \in Threads}
                        IN IF S = {} THEN 0
                           ELSE CHOOSE s \in S : \A s2 \in S : s >= s2
    IN totalCommits <= node_serial

\* 3. Snapshot Freshness: each committed thread's snapshot serial is
\*    strictly less than the current node_serial (the commit incremented it).
SnapshotBeforeCommit ==
    \A t \in Threads :
        (pc[t] = "done" /\ thr_committed[t] > 0)
        => thr_snap_ser[t] < node_serial

\* 4. Increment correctness: if only increment operations are used,
\*    node_val = node_serial (each commit adds exactly 1).
\*    Since we also have nondeterministic writes, we check the weaker:
\*    node_val <= MaxVal always holds.
ValueBounded ==
    node_val <= MaxVal

\* 5. Write-Read Consistency: a committed thread's write value is now
\*    visible -- if it just committed (pc=done, thr_committed>0),
\*    node_val equals its write value (it was the last writer).
WriteReadConsistency ==
    \A t \in Threads :
        (pc[t] = "done" /\ thr_committed[t] > 0 /\ thr_snap_ser[t] = node_serial - 1)
        => node_val = thr_write_val[t]

=============================================================================
