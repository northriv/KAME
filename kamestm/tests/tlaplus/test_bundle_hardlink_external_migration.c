/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
/*
 * C11 test generated mechanically from
 * BundleUnbundle_hardlink_external_migration.tla.
 *
 * Hard-link external-parent model WITH cross-tree migration.  4 nodes:
 *
 *       GN1 (bundle root)         P1 (external root, not in GN1's subtree)
 *        |                         |
 *       GN2 (under GN1)            +-- P2 (P2's packet initially in P1.sub[P2])
 *        |
 *       P2 (hard-linked: also under GN2 via subnode list; sub[P2] starts Null)
 *
 * Unlike the predecessor _external.tla model (which could only ever leave
 * GN1 missing=TRUE), Bundle(GN1) here performs a CROSS-TREE PULL as part of
 * its protocol:
 *
 *   BundlePhase1     read all 4 wrappers; detect P2's current home.
 *   BundlePullP1     CAS-clear P1.sub[P2], read the packet into local.p2Pkt.
 *   BundleCASP2      CAS P2.bundledBy from P1 to GN2.
 *   BundleUpdateGN1  CAS GN1 so GN2.sub[P2] holds the pulled packet (still
 *                    missing=TRUE).
 *   BundlePhase4     finalize -- clear GN1 missing, gated on subtree
 *                    integrity (subOK).
 *   BundlePhase5     publish: bundleDone.
 *
 * Each *Fail companion action restarts the pipeline at BundlePhase1.
 *
 * Concurrent peer race: P1RaceBundle re-bundles P1's tree, potentially
 * racing the cross-tree pull and forcing a retry.  Per the literal TLA+
 * (line 281-291) the refresh wrapper is value-equal to the current P1
 * wrapper unless P1.packet.missing=TRUE, so the action is enabled only when
 * P1 is currently missing -- mirrored exactly below.
 *
 * The TLA+ model is a per-thread SEQUENTIAL state machine driven by an
 * explicit `pc` (program counter); each Next-disjunct is one atomic step.
 * There are no MODE_COARSE/FINE/SUPERFINE atomicity knobs because the
 * source model has none.  As in test_bundle_hardlink_external.c the shared
 * `linkage` array is serialized under a single coarse mutex so that, with
 * NUM_THREADS>1, each thread's action executes atomically w.r.t. peers --
 * matching the TLA+ interleaving semantics.
 *
 * Terminal/safety invariants (post-join):
 *   SnapshotConsistency   -- GN1 published priority+~missing leaves no
 *                            unreachable Null grandchild slot.
 *   HardlinkExclusive     -- P2's packet exists in at most one parent's
 *                            sub[] (GN1's tree XOR P1's tree).
 *   BundleRefConsistency  -- every bundled chain reaches a priority node.
 *   EventuallyConsistent  -- final GN1 is priority and either missing OR
 *                            P2 reachable in-tree.
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sched.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- Compile-time knobs (consistent with the reference) --- */
#ifndef NUM_THREADS
#define NUM_THREADS 2
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0
#endif

/* MAX_COMMITS: number of bundle pipelines each thread runs.  In this model
 * each thread runs BundleStart..BundlePhase5 once to reach bundleDone
 * (mirrors the TLA+ ~bundleDone[t] guard); a budget > 1 re-arms and replays
 * the whole pipeline for stress.  Default 1 for the bounded unit test. */
#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 1
#  endif
#endif

/* ============================================================================
 * Node IDs (TLA+ CONSTANTS GN1, GN2, P2, P1, Null).
 * ============================================================================ */
#define NODE_GN1   0
#define NODE_GN2   1
#define NODE_P2    2
#define NODE_P1    3
#define NUM_NODES  4
#define NODE_NULL  0xFFu   /* TLA+ Null (no parent / no node) */

static const char *node_name(uint8_t n) {
    switch (n) {
        case NODE_GN1:  return "GN1";
        case NODE_GN2:  return "GN2";
        case NODE_P2:   return "P2";
        case NODE_P1:   return "P1";
        case NODE_NULL: return "Null";
        default:        return "?";
    }
}

/* ============================================================================
 * Packet.  TLA+ MakePacket(node, sub, miss).  We flatten the only
 * sub-structure the spec ever inspects:
 *   - GN1.sub[GN2]            (a packet, or Null)
 *   - GN1.sub[GN2].sub[P2]    (a packet, or Null)
 *   - P1.sub[P2]              (a packet, or Null)
 *   - GN2.sub[P2]             (a packet, or Null)  [when GN2 has priority]
 * `present` distinguishes a real Packet from TLA+ Null.
 * ============================================================================ */
typedef struct {
    bool    present;        /* false == TLA+ Null */
    uint8_t node;           /* which node this packet belongs to */
    bool    missing;        /* TLA+ packet.missing */
    bool    has_gn2;        /* GN2 in DOMAIN sub (GN1 packets) */
    bool    gn2_present;    /* sub[GN2] /= Null */
    bool    has_p2;         /* P2 in DOMAIN sub (GN2 / P1 / GN1 packets) */
    bool    p2_present;     /* sub[P2] /= Null  (for GN1: sub[GN2].sub[P2]) */
} Packet;

#define NULL_PACKET ((Packet){0})

/* ============================================================================
 * Wrapper.  TLA+ PriorityWrapper(packet) / BundledRefWrapper(parent).
 * ============================================================================ */
typedef struct {
    bool    has_priority;
    uint8_t bundled_by;     /* NODE_NULL when priority */
    Packet  packet;         /* valid iff has_priority (else NULL_PACKET) */
} Wrapper;

static inline Wrapper priority_wrapper(Packet p) {
    return (Wrapper){ .has_priority = true, .bundled_by = NODE_NULL, .packet = p };
}
static inline Wrapper bundled_ref_wrapper(uint8_t parent) {
    return (Wrapper){ .has_priority = false, .bundled_by = parent,
                      .packet = NULL_PACKET };
}

/* Structural equality -- TLA+ uses value equality for CAS guards
 * `linkage[X] = oldW`.  Compare every observable field. */
static bool pkt_eq(const Packet *a, const Packet *b) {
    if (a->present != b->present) return false;
    if (!a->present) return true;            /* both Null */
    return a->node == b->node
        && a->missing == b->missing
        && a->has_gn2 == b->has_gn2
        && a->gn2_present == b->gn2_present
        && a->has_p2 == b->has_p2
        && a->p2_present == b->p2_present;
}
static bool wrapper_eq(const Wrapper *a, const Wrapper *b) {
    if (a->has_priority != b->has_priority) return false;
    if (a->bundled_by != b->bundled_by) return false;
    return pkt_eq(&a->packet, &b->packet);
}

/* ============================================================================
 * Shared state.  TLA+ `linkage` is [Nodes -> Wrapper].  Sequential model:
 * all transitions on `linkage` run under one mutex; each critical section
 * is exactly one TLA+ atomic action.
 * ============================================================================ */
static Wrapper          linkage[NUM_NODES];
static pthread_mutex_t  linkage_mtx = PTHREAD_MUTEX_INITIALIZER;
static _Atomic(bool)    g_stop;
static _Atomic(uint64_t) total_bundles;   /* completed bundle pipelines */

#define LK_LOCK()   pthread_mutex_lock(&linkage_mtx)
#define LK_UNLOCK() pthread_mutex_unlock(&linkage_mtx)

/* ============================================================================
 * Packet builders mirroring the TLA+ MakePacket / *SubInit / *Wrapper.
 * ============================================================================ */

/* P2Packet == MakePacket(P2, <<>>, FALSE): a leaf packet, empty sub. */
static inline Packet make_p2_packet(void) {
    return (Packet){ .present = true, .node = NODE_P2, .missing = false,
                     .has_gn2 = false, .gn2_present = false,
                     .has_p2 = false, .p2_present = false };
}
/* GN1's packet: sub = [GN2 |-> gn2pkt(sub=[P2|->p2slot])].  We flatten
 * GN1.sub[GN2].sub[P2] presence into the GN1 packet's p2_present field. */
static inline Packet make_gn1_pkt(bool missing, bool gn2_present, bool gn2_p2_present) {
    return (Packet){ .present = true, .node = NODE_GN1, .missing = missing,
                     .has_gn2 = true, .gn2_present = gn2_present,
                     .has_p2 = gn2_present ? true : false,
                     .p2_present = gn2_present ? gn2_p2_present : false };
}
/* P1's packet: sub = [P2 |-> p2slot]. */
static inline Packet make_p1_pkt(bool missing, bool p2_present) {
    return (Packet){ .present = true, .node = NODE_P1, .missing = missing,
                     .has_gn2 = false, .gn2_present = false,
                     .has_p2 = true, .p2_present = p2_present };
}

/* ============================================================================
 * ReachableFromGN1(rootPkt) -- TLA+ lines 115-120.
 *   rootPkt.node = GN1 /\ GN2 in DOMAIN sub /\ sub[GN2] /= Null
 *   /\ P2 in DOMAIN sub[GN2].sub /\ sub[GN2].sub[P2] /= Null
 * In our flattened encoding sub[GN2].sub[P2] /= Null is gn1Pkt.p2_present
 * (only meaningful when gn2_present).
 * ============================================================================ */
static bool reachable_from_gn1(const Packet *root) {
    return root->present
        && root->node == NODE_GN1
        && root->has_gn2
        && root->gn2_present
        && root->has_p2
        && root->p2_present;
}

/* ============================================================================
 * Per-thread program-counter state.  Mirrors TLA+ pc[t]/local[t].
 * ============================================================================ */
typedef enum {
    PC_IDLE = 0,
    PC_PHASE1,        /* bundle_phase1     */
    PC_PULL_P1,       /* bundle_pull_p1    */
    PC_CAS_P2,        /* bundle_cas_p2     */
    PC_UPDATE_GN1,    /* bundle_update_gn1 */
    PC_PHASE4,        /* bundle_phase4     */
    PC_PHASE5,        /* bundle_phase5     */
} PC;

/* InitLocal (TLA+ lines 90-97). */
typedef struct {
    Wrapper gn1Wrapper;     /* local[t].gn1Wrapper */
    Wrapper gn2Wrapper;     /* local[t].gn2Wrapper */
    Wrapper p2Wrapper;      /* local[t].p2Wrapper  */
    Wrapper p1Wrapper;      /* local[t].p1Wrapper  */
    Packet  p2Pkt;          /* local[t].p2Pkt -- pulled from P1.sub[P2] */
} Local;

static inline void init_local(Local *l) {
    memset(l, 0, sizeof(*l));
    Wrapper nullw = (Wrapper){ .has_priority=false, .bundled_by=NODE_NULL,
                               .packet=NULL_PACKET };
    l->gn1Wrapper = nullw;
    l->gn2Wrapper = nullw;
    l->p2Wrapper  = nullw;
    l->p1Wrapper  = nullw;
    l->p2Pkt      = NULL_PACKET;
}

typedef struct {
    uint32_t tid;
    PC       pc;
    Local    local;
} ThreadCtx;

/* ============================================================================
 * Safety invariants (checked on the live linkage).
 * ============================================================================ */

/* SnapshotConsistency -- TLA+ lines 340-348.
 *   (gn1.hasPriority /\ ~gn1Pkt.missing) =>
 *      \A c in DOMAIN sub: sub[c] /= Null =>
 *          \A gc in DOMAIN sub[c].sub: sub[c].sub[gc] = Null =>
 *              ReachableFromGN1(gn1Pkt)
 * Only (c,gc) candidate is (GN2,P2): the Null grandchild slot is
 * gn2_present && !p2_present. */
static bool snapshot_consistency(void) {
    Wrapper gn1w = linkage[NODE_GN1];
    if (!(gn1w.has_priority && !gn1w.packet.missing)) return true;
    const Packet *p = &gn1w.packet;
    if (p->has_gn2 && p->gn2_present) {
        if (p->has_p2 && !p->p2_present) {
            if (!reachable_from_gn1(p)) return false;
        }
    }
    return true;
}

/* HardlinkExclusive -- TLA+ lines 351-359.
 *   inGN2 == gn1.hasPriority /\ gn1.sub[GN2] /= Null /\ gn1.sub[GN2].sub[P2] /= Null
 *   inP1  == p1.hasPriority  /\ p1.sub[P2] /= Null
 *   ~(inGN2 /\ inP1) */
static bool hardlink_exclusive(void) {
    Wrapper gn1w = linkage[NODE_GN1];
    Wrapper p1w  = linkage[NODE_P1];
    bool inGN2 = gn1w.has_priority
              && gn1w.packet.has_gn2 && gn1w.packet.gn2_present
              && gn1w.packet.has_p2 && gn1w.packet.p2_present;
    bool inP1  = p1w.has_priority
              && p1w.packet.has_p2 && p1w.packet.p2_present;
    return !(inGN2 && inP1);
}

/* ReachesPriority(n, depth) -- TLA+ lines 364-369 (RECURSIVE). */
static bool reaches_priority(uint8_t n, int depth) {
    if (depth == 0) return true;
    if (n >= NUM_NODES) return false;
    if (linkage[n].has_priority) return true;
    uint8_t bb = linkage[n].bundled_by;
    if (bb < NUM_NODES) return reaches_priority(bb, depth - 1);
    return false;
}

/* BundleRefConsistency -- TLA+ lines 371-373.
 *   \A n in {GN2,P2}: ~linkage[n].hasPriority => ReachesPriority(n.bundledBy, 3) */
static bool bundle_ref_consistency(void) {
    const uint8_t nodes[2] = { NODE_GN2, NODE_P2 };
    for (int i = 0; i < 2; i++) {
        uint8_t n = nodes[i];
        if (linkage[n].has_priority) continue;
        if (!reaches_priority(linkage[n].bundled_by, 3)) return false;
    }
    return true;
}

/* ============================================================================
 * TLA+ actions.  Each performs ONE atomic step under LK_LOCK and returns
 * true if it fired so the worker loop advances pc the way Next would.
 * The *Fail companions are folded into the success actions' ELSE branch
 * (restart at PHASE1), exactly as in the predecessor port.
 * ============================================================================ */

/* Reset local to InitLocal with op="bundle" (TLA+ *Fail / BundleStart). */
static inline void restart_to_phase1(ThreadCtx *ctx) {
    init_local(&ctx->local);
    ctx->pc = PC_PHASE1;
}

/* BundleStart(t) -- TLA+ lines 125-131. pc idle -> phase1, reset local. */
static bool bundle_start(ThreadCtx *ctx) {
    init_local(&ctx->local);
    ctx->pc = PC_PHASE1;
    return true;
}

/* BundlePhase1(t) -- TLA+ lines 134-150.  Read all 4 wrappers; branch on
 * P2's current home. */
static bool bundle_phase1(ThreadCtx *ctx) {
    LK_LOCK();
    Wrapper gn1w = linkage[NODE_GN1];
    Wrapper gn2w = linkage[NODE_GN2];
    Wrapper p2w  = linkage[NODE_P2];
    Wrapper p1w  = linkage[NODE_P1];
    if (!gn1w.has_priority) { LK_UNLOCK(); return false; }  /* guard */

    Local *l = &ctx->local;
    l->gn1Wrapper = gn1w;
    l->gn2Wrapper = gn2w;
    l->p2Wrapper  = p2w;
    l->p1Wrapper  = p1w;

    /* pc' = IF p2w.bundledBy = P1 THEN bundle_pull_p1 ELSE bundle_phase4 */
    if (!p2w.has_priority && p2w.bundled_by == NODE_P1)
        ctx->pc = PC_PULL_P1;
    else
        ctx->pc = PC_PHASE4;   /* P2 already in GN2 -- finalize */
    LK_UNLOCK();
    return true;
}

/* BundlePullP1(t)/BundlePullP1Fail(t) -- TLA+ lines 155-185.  CAS-clear
 * P1.sub[P2] and read the pulled packet into local.p2Pkt.  Requires P1 has
 * priority, ~missing, holds P2, and matches our snapshot. */
static bool bundle_pull_p1(ThreadCtx *ctx) {
    LK_LOCK();
    Local *l = &ctx->local;
    Wrapper p1w = linkage[NODE_P1];

    bool ok = p1w.has_priority
           && !p1w.packet.missing
           && (p1w.packet.has_p2 && p1w.packet.p2_present)   /* sub[P2] /= Null */
           && wrapper_eq(&p1w, &l->p1Wrapper);               /* still our snapshot */
    if (ok) {
        /* pulled = p1w.packet.sub[P2]  (== P2Packet leaf) */
        Packet pulled = make_p2_packet();
        /* newP1: sub = [P2 |-> Null], missing=FALSE */
        Packet newP1Pkt = make_p1_pkt(/*missing*/false, /*p2_present*/false);
        linkage[NODE_P1] = priority_wrapper(newP1Pkt);
        l->p2Pkt = pulled;
        ctx->pc = PC_CAS_P2;
    } else {
        /* BundlePullP1Fail: P1 changed under us OR P2 no longer at P1. */
        restart_to_phase1(ctx);
    }
    LK_UNLOCK();
    return true;
}

/* BundleCASP2(t)/BundleCASP2Fail(t) -- TLA+ lines 188-206.  CAS P2.bundledBy
 * from P1 to GN2. */
static bool bundle_cas_p2(ThreadCtx *ctx) {
    LK_LOCK();
    Local *l = &ctx->local;
    Wrapper p2w = linkage[NODE_P2];

    bool ok = wrapper_eq(&p2w, &l->p2Wrapper)
           && !p2w.has_priority && p2w.bundled_by == NODE_P1;
    if (ok) {
        Wrapper newW = bundled_ref_wrapper(NODE_GN2);
        linkage[NODE_P2] = newW;
        l->p2Wrapper = newW;
        ctx->pc = PC_UPDATE_GN1;
    } else {
        /* BundleCASP2Fail: p2w /= our snapshot. */
        restart_to_phase1(ctx);
    }
    LK_UNLOCK();
    return true;
}

/* BundleUpdateGN1(t)/BundleUpdateGN1Fail(t) -- TLA+ lines 209-235.  CAS GN1
 * so GN2.sub[P2] holds the pulled packet.  Keep GN1 missing=TRUE. */
static bool bundle_update_gn1(ThreadCtx *ctx) {
    LK_LOCK();
    Local *l = &ctx->local;
    Wrapper gn1w = linkage[NODE_GN1];

    bool ok = wrapper_eq(&gn1w, &l->gn1Wrapper) && l->p2Pkt.present;  /* p2Pkt /= Null */
    if (ok) {
        /* newGN1: sub[GN2].sub[P2] = pulled packet; missing=TRUE. */
        Packet newGN1Pkt = make_gn1_pkt(/*missing*/true,
                                        /*gn2_present*/true,
                                        /*gn2_p2_present*/true);
        Wrapper newW = priority_wrapper(newGN1Pkt);
        linkage[NODE_GN1] = newW;
        l->gn1Wrapper = newW;
        ctx->pc = PC_PHASE4;
    } else {
        /* BundleUpdateGN1Fail: gn1w /= our snapshot. */
        restart_to_phase1(ctx);
    }
    LK_UNLOCK();
    return true;
}

/* BundlePhase4(t)/BundlePhase4Fail(t) -- TLA+ lines 238-263.  Finalize:
 * clear GN1.missing, gated on subtree integrity (subOK).
 *
 *   subOK == \A c in DOMAIN sub: sub[c] /= Null =>
 *               (\A gc in DOMAIN sub[c].sub:
 *                   sub[c].sub[gc] /= Null \/ ReachableFromGN1(oldPkt))
 *   targetMissing == ~subOK
 */
static bool bundle_phase4(ThreadCtx *ctx) {
    LK_LOCK();
    Local *l = &ctx->local;
    Wrapper gn1w = linkage[NODE_GN1];

    if (!wrapper_eq(&gn1w, &l->gn1Wrapper)) {
        /* BundlePhase4Fail: gn1w /= our snapshot. */
        restart_to_phase1(ctx);
        LK_UNLOCK();
        return true;
    }

    const Packet *old = &l->gn1Wrapper.packet;
    bool subOK = true;
    if (old->has_gn2 && old->gn2_present) {           /* sub[GN2] /= Null */
        if (old->has_p2) {                            /* gc = P2 in DOMAIN */
            bool gcNonNull = old->p2_present;         /* sub[GN2].sub[P2] /= Null */
            if (!(gcNonNull || reachable_from_gn1(old))) subOK = false;
        }
    }
    bool targetMissing = !subOK;

    /* finalPkt = MakePacket(GN1, oldW.packet.sub, targetMissing) */
    Packet finalPkt = *old;
    finalPkt.missing = targetMissing;
    Wrapper finalW = priority_wrapper(finalPkt);

    linkage[NODE_GN1] = finalW;
    l->gn1Wrapper = finalW;
    ctx->pc = PC_PHASE5;
    LK_UNLOCK();
    return true;
}

/* BundlePhase5(t) -- TLA+ lines 265-270.  Publish: bundleDone. */
static bool bundle_phase5(ThreadCtx *ctx) {
    init_local(&ctx->local);
    ctx->pc = PC_IDLE;
    atomic_fetch_add_explicit(&total_bundles, 1u, memory_order_relaxed);
    return true;
}

/* P1RaceBundle(t) -- TLA+ lines 281-291.  Peer re-bundle of P1's tree.
 * refresh == PriorityWrapper(MakePacket(P1, p1w.packet.sub, FALSE)); enabled
 * iff p1w.hasPriority AND refresh /= p1w.  Since refresh rebuilds the same
 * sub with missing=FALSE, it differs from p1w only when p1w.packet.missing
 * is TRUE -- so this fires only while P1 is currently missing. */
static bool p1_race_bundle(void) {
    LK_LOCK();
    Wrapper p1w = linkage[NODE_P1];
    if (!p1w.has_priority) { LK_UNLOCK(); return false; }

    Packet refreshPkt = make_p1_pkt(/*missing*/false,
                                    /*p2_present*/p1w.packet.p2_present);
    Wrapper refresh = priority_wrapper(refreshPkt);
    if (wrapper_eq(&refresh, &p1w)) { LK_UNLOCK(); return false; }  /* refresh = p1w */

    linkage[NODE_P1] = refresh;
    LK_UNLOCK();
    return true;
}

/* ============================================================================
 * Worker.  Drives one bundle pipeline per "commit", mirroring the TLA+ Next
 * disjunction.  P1RaceBundle is attempted opportunistically before each
 * pipeline (enabled only while P1 is currently missing -- effectively never
 * in the reachable state space, faithful to the literal spec semantics).
 *
 * A step budget bounds the inner loop so a wedged phase (should be
 * impossible) cannot livelock the unit test.
 * ============================================================================ */
static void *worker(void *arg) {
    ThreadCtx ctx;
    ctx.tid = ((ThreadCtx*)arg)->tid;
    ctx.pc  = PC_IDLE;
    init_local(&ctx.local);

    for (uint32_t commit = 0; commit < (uint32_t)MAX_COMMITS; commit++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;

        /* Opportunistic peer race (out-of-band). */
        p1_race_bundle();

        ctx.pc = PC_IDLE;
        bool done = false;
        for (uint64_t step = 0; step < (1u << 20) && !done; step++) {
            switch (ctx.pc) {
                case PC_IDLE:       bundle_start(&ctx);                 break;
                case PC_PHASE1:     if (!bundle_phase1(&ctx))  sched_yield(); break;
                case PC_PULL_P1:    bundle_pull_p1(&ctx);              break;
                case PC_CAS_P2:     bundle_cas_p2(&ctx);               break;
                case PC_UPDATE_GN1: bundle_update_gn1(&ctx);           break;
                case PC_PHASE4:     bundle_phase4(&ctx);               break;
                case PC_PHASE5:     bundle_phase5(&ctx); done = true;  break;
            }
        }
        assert(done && "bundle pipeline failed to terminate (livelock?)");
    }
    return NULL;
}

/* ============================================================================
 * Post-join invariant check + terminal assertion.
 * ============================================================================ */
static void check_invariants(void) {
    assert(bundle_ref_consistency());
    assert(snapshot_consistency());
    assert(hardlink_exclusive());
    /* GN1 is the bundle root: always priority. */
    assert(linkage[NODE_GN1].has_priority);
}

int main(void) {
    /* Init (TLA+ lines 99-110):
     *   GN1 = PriorityWrapper(GN1, sub=[GN2 |-> GN2pkt(sub=[P2|->Null])], TRUE)
     *   GN2 = BundledRefWrapper(GN1)
     *   P1  = PriorityWrapper(P1, sub=[P2 |-> P2Packet], FALSE)
     *   P2  = BundledRefWrapper(P1)     (initially in P1's tree)
     */
    linkage[NODE_GN1] = priority_wrapper(
        make_gn1_pkt(/*missing*/true, /*gn2_present*/true, /*gn2_p2_present*/false));
    linkage[NODE_GN2] = bundled_ref_wrapper(NODE_GN1);
    linkage[NODE_P1]  = priority_wrapper(
        make_p1_pkt(/*missing*/false, /*p2_present*/true));
    linkage[NODE_P2]  = bundled_ref_wrapper(NODE_P1);

    atomic_store(&g_stop, false);
    atomic_store(&total_bundles, 0u);

    /* Sanity: Init satisfies the safety invariants.  (SnapshotConsistency
     * is vacuous since GN1 starts missing=TRUE.) */
    assert(snapshot_consistency());
    assert(hardlink_exclusive());
    assert(bundle_ref_consistency());

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid = (uint32_t)(i + 1);
        ctxs[i].pc  = PC_IDLE;
        init_local(&ctxs[i].local);
        pthread_create(&threads[i], NULL, worker, &ctxs[i]);
    }

#if STRESS_SECONDS > 0
    struct timespec ts = { .tv_sec = STRESS_SECONDS, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    atomic_store_explicit(&g_stop, true, memory_order_release);
#endif

    for (int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);

    check_invariants();

    /* Terminal invariant -- TLA+ EventuallyConsistent (lines 382-385):
     *   GN1 priority /\ (GN1.missing \/ ReachableFromGN1(GN1.packet))
     * After the cross-tree migration the success path re-homes P2 into GN2,
     * so the consistent terminal state has GN1 ~missing AND P2 reachable. */
    Wrapper gn1 = linkage[NODE_GN1];
    assert(gn1.has_priority);
    assert((gn1.packet.missing || reachable_from_gn1(&gn1.packet))
           && "EventuallyConsistent: GN1 neither missing nor P2-reachable");
    if (gn1.has_priority && !gn1.packet.missing) {
        assert(reachable_from_gn1(&gn1.packet)
               && "GN1 published ~missing but P2 hard-link unreachable in GN1");
    }

    uint64_t bundles = atomic_load(&total_bundles);

#if STRESS_SECONDS > 0
    Wrapper p1 = linkage[NODE_P1];
    Wrapper p2 = linkage[NODE_P2];
    printf("[hardlink_external_migration stress %ds threads=%d] bundles=%llu\n",
           STRESS_SECONDS, NUM_THREADS, (unsigned long long)bundles);
    printf("  GN1: prio=%d missing=%d gn2_present=%d p2_present=%d\n",
           gn1.has_priority, gn1.packet.missing,
           gn1.packet.gn2_present, gn1.packet.p2_present);
    printf("  GN2: prio=%d bundled_by=%s\n",
           linkage[NODE_GN2].has_priority, node_name(linkage[NODE_GN2].bundled_by));
    printf("  P1 : prio=%d p2_present=%d   P2: prio=%d bundled_by=%s\n",
           p1.has_priority, p1.packet.p2_present,
           p2.has_priority, node_name(p2.bundled_by));
#else
    /* Bounded unit: every thread completed exactly MAX_COMMITS pipelines. */
    uint64_t expected = (uint64_t)MAX_COMMITS * (uint64_t)NUM_THREADS;
    assert(bundles == expected);
    (void)node_name; /* used only in stress diag */
    printf("[hardlink_external_migration unit threads=%d commits=%d] bundles=%llu OK\n",
           NUM_THREADS, (int)MAX_COMMITS, (unsigned long long)bundles);
#endif

    return 0;
}
