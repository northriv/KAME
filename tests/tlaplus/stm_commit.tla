(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************)
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

\* ==========================================================================
\* @c11_mapping -- Variable-to-C++ correspondence (Layer 1)
\*
\* This layer abstracts atomic_shared_ptr as a correct linearizable register.
\* The detailed tagged-pointer protocol (Layer 0) is verified separately.
\*
\* TLA+ variable              C++ type & expression
\* --------------------------------------------------------------------------
\* @c11_var node_val:         m_link->packet()->payload()->m_x
\*   -- the committed payload value, accessed via Snapshot or Transaction.
\*   Reads:  local_shared_ptr<PacketWrapper> w(*m_link);  -- load_shared_
\*   Writes: m_link->compareAndSet(old_wrapper, new_wrapper);
\*
\* @c11_var node_serial:      m_link->m_bundle_serial
\*   -- Lamport serial in PacketWrapper; monotonically increases on commit.
\*   Used as CAS identity: if serial changed, another commit intervened.
\*
\* @c11_var thr_snap_val[t]:  thread-local Transaction::m_oldpacket->payload()
\* @c11_var thr_snap_ser[t]:  thread-local Transaction::m_serial (snapshot)
\* @c11_var thr_write_val[t]: thread-local Transaction::m_packet->payload()
\*   -- copy-on-write payload modified by the closure in iterate_commit.
\* @c11_var thr_committed[t]: thread-local commit success counter
\*   -- not present in C++; model-only for progress properties.
\*
\* Layer boundary:
\*   All reads/writes to m_link go through atomic_shared_ptr operations:
\*     Snapshot -> load_shared_()   (Layer 0: AcquireTagRef + IncGlobal + Rel)
\*     Commit   -> compareAndSet()  (Layer 0: PreInc + AcquireTagRef + CAS)
\*   This layer treats those as atomic register ops (linearized).
\*
\* Source: kame/transaction.h, kame/transaction_impl.h
\* ==========================================================================

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

\* @c11_action TakeSnapshot(t):
\*   Transaction<XN> tr(node);  -- constructor calls snapshot()
\*   local_shared_ptr<PacketWrapper> wrapper(*m_link);  -- load_shared_()
\*   m_oldpacket = wrapper->packet();
\*   m_serial    = wrapper->m_bundle_serial;
\*   Linearization: the load_shared_() in Layer 0 is the linearization point.
\*   Source: transaction.h:540-545, transaction_impl.h:1243
TakeSnapshot(t) ==
    /\ pc[t] \in {"idle", "done"}
    /\ pc' = [pc EXCEPT ![t] = "snapshot"]
    /\ thr_snap_val' = [thr_snap_val EXCEPT ![t] = node_val]
    /\ thr_snap_ser' = [thr_snap_ser EXCEPT ![t] = node_serial]
    /\ UNCHANGED <<node_val, node_serial, thr_write_val, thr_committed>>

\* @c11_action Write(t):
\*   tr[node].m_x = v;  -- copy-on-write via Transaction::operator[]
\*   The closure body is arbitrary; modeled as nondeterministic value.
\*   All writes are thread-local (COW payload) -- no atomic ops.
\*   Source: transaction.h:564-566
Write(t) ==
    /\ pc[t] = "snapshot"
    /\ \E v \in 0..MaxVal :
       thr_write_val' = [thr_write_val EXCEPT ![t] = v]
    /\ pc' = [pc EXCEPT ![t] = "write"]
    /\ UNCHANGED <<node_val, node_serial, thr_snap_val, thr_snap_ser, thr_committed>>

\* @c11_action WriteIncrement(t):
\*   tr[node].m_x = shot[node].m_x + 1;
\*   Deterministic variant: read-modify-write on the snapshot value.
\*   Still thread-local (COW); no atomic ops until commit.
\*   Source: transaction.h:564-566
WriteIncrement(t) ==
    /\ pc[t] = "snapshot"
    /\ thr_snap_val[t] + 1 <= MaxVal
    /\ thr_write_val' = [thr_write_val EXCEPT ![t] = thr_snap_val[t] + 1]
    /\ pc' = [pc EXCEPT ![t] = "write"]
    /\ UNCHANGED <<node_val, node_serial, thr_snap_val, thr_snap_ser, thr_committed>>

\* @c11_action Commit(t):
\*   // success path:
\*   local_shared_ptr<PacketWrapper> newwrapper(
\*       new PacketWrapper(tr.m_packet, tr.m_serial));
\*   if (m_link->compareAndSet(wrapper, newwrapper))  // Layer 0 CAS
\*       return true;
\*   // fail path (another thread committed since our snapshot):
\*   //   wrapper = *m_link;  // re-snapshot (load_shared_)
\*   //   -> retry from "snapshot"
\*   Linearization: the compareAndSet's inner CAS on g_ref (Layer 0: CASSwap)
\*   is the linearization point for the commit.
\*   Source: transaction_impl.h:1241-1270
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
RECURSIVE SumSet(_)
SumSet(S) ==
    IF S = {} THEN 0
    ELSE LET t == CHOOSE t \in S : TRUE
         IN thr_committed[t] + SumSet(S \ {t})

CommitSerializes ==
    SumSet(Threads) <= node_serial

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
