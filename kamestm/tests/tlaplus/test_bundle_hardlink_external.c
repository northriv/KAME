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
 * C11 test generated mechanically from BundleUnbundle_hardlink_external.tla.
 *
 * Hard-link with EXTERNAL parent.  4 nodes:
 *
 *       GN1 (bundle root)         P1 (external root, not in GN1's subtree)
 *        |                         |
 *       GN2 (under GN1)            +-- P2 (P2's packet may live in P1.sub[P2])
 *        |
 *       P2 (hard-linked: parent2 = GN2 -- inside GN1; parent1 = P1 -- outside)
 *
 * This is the production dyn_node_test scenario: p2 (= P2) is hard-linked
 * between gn2 (= GN2, inside gn1's subtree) and a worker-local p1 (= P1,
 * OUTSIDE gn1's subtree).
 *
 * IMPORTANT: the TLA+ model is SINGLE-THREADED SEQUENTIAL -- no true
 * concurrency, no insert/release.  The bug is purely structural:
 * bundle(GN1) cannot finalize without losing the hard-link to P2.
 * We therefore mirror the spec as a per-thread sequential state machine
 * driven by an explicit `pc` (program counter), exactly like the TLA+
 * Next disjunction.  There are no MODE_COARSE/FINE/SUPERFINE atomicity
 * knobs because the source model has none.
 *
 * The harness STYLE (self-contained C11, _Atomic shared state, pthread
 * workers, post-join check_invariants + terminal assert) follows
 * test_bundle_2level_LLfree.c.  Because the model is sequential the
 * shared `linkage` array is serialized under a single coarse mutex so
 * that, with NUM_THREADS>1, each thread's bundle pipeline executes its
 * actions atomically w.r.t. peers -- matching the TLA+ interleaving
 * semantics where each action is one atomic step.
 *
 * Terminal/safety invariant (post-join):
 *   SnapshotConsistency -- mirrors Packet::checkConsistensy at
 *   transaction_impl.h:870-871.  When GN1 is published priority + ~missing,
 *   every Null grandchild slot inside GN1's tree (here GN2.sub[P2]) must be
 *   reachable via reverseLookup anchored at GN1's published root packet.
 *   With P2 living at P1.sub[P2] (external), the ONLY way to satisfy this
 *   in the minimal model is for GN1 to stay missing (Phase4 override denied).
 *   The Phase4 reachability gate (subOK) is what enforces this; we assert
 *   the invariant holds for the final published GN1 state of every thread.
 *   We also assert BundleRefConsistency: GN2 bundled => its bundler has
 *   priority.
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

/* MAX_COMMITS: number of bundle pipelines each thread runs.  In this
 * sequential model each thread runs BundleStart..BundlePhase5 exactly
 * once to reach bundleDone (mirrors the TLA+ ~bundleDone[t] guard); a
 * budget > 1 simply replays the whole pipeline (re-arming bundleDone)
 * for stress.  Default 1 for the bounded unit test. */
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
 * Packet.  TLA+ MakePacket(node, sub, miss) where `sub` is a map
 * child-node -> child-packet-or-Null.  We do NOT need the full recursive
 * map: the only sub-structure the spec ever inspects is
 *   - GN1.sub[GN2]            (a packet, or Null)
 *   - GN1.sub[GN2].sub[P2]    (a packet, or Null)
 *   - P1.sub[P2]              (a packet, or Null)
 *   - GN2.sub[P2]             (a packet, or Null)  [when GN2 has priority]
 * so we encode each packet as a flat record carrying the slots the spec
 * reads.  `present` distinguishes a real Packet from TLA+ Null.
 * ============================================================================ */
typedef struct {
    bool    present;        /* false == TLA+ Null */
    uint8_t node;           /* which node this packet belongs to */
    bool    missing;        /* TLA+ packet.missing */
    /* sub[] entries.  For GN1: gn2 slot.  For GN2/P1: p2 slot.
     * A `*_present` flag false means that sub-slot's value is TLA+ Null
     * (the slot exists in DOMAIN but maps to Null). */
    bool    has_gn2;        /* GN2 in DOMAIN sub (GN1 packets) */
    bool    gn2_present;    /* sub[GN2] /= Null */
    bool    has_p2;         /* P2 in DOMAIN sub (GN2 / P1 packets) */
    bool    p2_present;     /* sub[P2] /= Null */
} Packet;

#define NULL_PACKET ((Packet){0})

/* ============================================================================
 * Wrapper.  TLA+ PriorityWrapper(packet) / BundledRefWrapper(parent).
 *   PriorityWrapper:  {packet, hasPriority=TRUE,  bundledBy=Null}
 *   BundledRefWrapper:{packet=Null, hasPriority=FALSE, bundledBy=parent}
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

/* Structural equality of wrappers -- TLA+ uses value equality for the
 * CAS guards `linkage[X] = oldW`.  We compare every observable field. */
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
 * Shared state.  TLA+ `linkage` is [Nodes -> Wrapper].  The model is
 * sequential, so all transitions on `linkage` run under one mutex; each
 * critical section corresponds to exactly one TLA+ atomic action.
 * ============================================================================ */
static Wrapper          linkage[NUM_NODES];
static pthread_mutex_t  linkage_mtx = PTHREAD_MUTEX_INITIALIZER;
static _Atomic(bool)    g_stop;
static _Atomic(uint64_t) total_bundles;   /* completed bundle pipelines */

#define LK_LOCK()   pthread_mutex_lock(&linkage_mtx)
#define LK_UNLOCK() pthread_mutex_unlock(&linkage_mtx)

/* ============================================================================
 * Packet builders mirroring the TLA+ MakePacket / GN1SubInit / etc.
 * ============================================================================ */

/* GN2's bundled packet: sub = [P2 |-> Null]  (hard-link: P2 slot stays Null) */
static inline Packet make_gn2_pkt_null_p2(bool missing) {
    return (Packet){ .present = true, .node = NODE_GN2, .missing = missing,
                     .has_gn2 = false, .gn2_present = false,
                     .has_p2 = true, .p2_present = false };
}
/* GN2's valid packet: sub = [P2 |-> P2Packet] */
static inline Packet make_gn2_pkt_valid_p2(bool missing) {
    return (Packet){ .present = true, .node = NODE_GN2, .missing = missing,
                     .has_gn2 = false, .gn2_present = false,
                     .has_p2 = true, .p2_present = true };
}
/* GN1's packet: sub = [GN2 |-> gn2pkt].  gn2_p2_present mirrors whether
 * GN1.sub[GN2].sub[P2] is populated (it is carried inside the GN2 sub-pkt;
 * we flatten it as the GN1 packet's p2_present field for spec reads). */
static inline Packet make_gn1_pkt(bool missing, bool gn2_present, bool gn2_p2_present) {
    return (Packet){ .present = true, .node = NODE_GN1, .missing = missing,
                     .has_gn2 = true, .gn2_present = gn2_present,
                     .has_p2 = gn2_present ? true : false,
                     .p2_present = gn2_present ? gn2_p2_present : false };
}
/* P1's packet: sub = [P2 |-> p2slot] */
static inline Packet make_p1_pkt(bool missing, bool p2_present) {
    return (Packet){ .present = true, .node = NODE_P1, .missing = missing,
                     .has_gn2 = false, .gn2_present = false,
                     .has_p2 = true, .p2_present = p2_present };
}

/* ============================================================================
 * ReachableFromGN1(rootPkt) -- TLA+ lines 126-133.
 *   P2 reachable from GN1 iff GN1.sub[GN2] populated, not Null, AND
 *   GN1.sub[GN2].sub[P2] populated and not Null.
 * In our flattened encoding GN1.sub[GN2].sub[P2] != Null is gn1Pkt.p2_present
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
    PC_PHASE1,
    PC_PHASE2,
    PC_PHASE3,
    PC_PHASE4,
    PC_PHASE5,
} PC;

typedef struct {
    Wrapper wrapper;        /* local[t].wrapper -- GN1's wrapper snapshot */
    Wrapper gn2Wrapper;     /* local[t].gn2Wrapper */
    Wrapper p2Wrapper;      /* local[t].p2Wrapper */
    Wrapper p1Wrapper;      /* local[t].p1Wrapper */
    Packet  gn2SubPkt;      /* local[t].gn2SubPkt */
    Packet  p2SubPkt;       /* local[t].p2SubPkt */
} Local;

static inline void init_local(Local *l) {
    memset(l, 0, sizeof(*l));
    l->wrapper    = (Wrapper){ .has_priority=false, .bundled_by=NODE_NULL,
                               .packet=NULL_PACKET };
    l->gn2Wrapper = l->wrapper;
    l->p2Wrapper  = l->wrapper;
    l->p1Wrapper  = l->wrapper;
    l->gn2SubPkt  = NULL_PACKET;
    l->p2SubPkt   = NULL_PACKET;
}

typedef struct {
    uint32_t tid;
    PC       pc;
    Local    local;
} ThreadCtx;

/* ============================================================================
 * SnapshotConsistency -- TLA+ lines 343-352.  Checked on the live linkage.
 *   (gn1.hasPriority /\ ~gn1Pkt.missing) =>
 *      \A c in DOMAIN sub: sub[c] /= Null =>
 *          \A gc in DOMAIN sub[c].sub: sub[c].sub[gc] = Null =>
 *              ReachableFromGN1(gn1Pkt)
 * Our only (c,gc) candidate is (GN2,P2): the Null grandchild slot is
 * GN1.sub[GN2].sub[P2] == Null, i.e. gn2_present && !p2_present.
 * ============================================================================ */
static bool snapshot_consistency(void) {
    Wrapper gn1w = linkage[NODE_GN1];
    if (!(gn1w.has_priority && !gn1w.packet.missing)) return true;
    const Packet *p = &gn1w.packet;
    /* c = GN2 */
    if (p->has_gn2 && p->gn2_present) {
        /* gc = P2 : sub[GN2].sub[P2] == Null ? */
        if (p->has_p2 && !p->p2_present) {
            /* Null grandchild slot -- must be reachable elsewhere in GN1. */
            if (!reachable_from_gn1(p)) return false;
        }
    }
    return true;
}

/* BundleRefConsistency -- TLA+ lines 355-357.
 *   ~gn2.hasPriority => linkage[gn2.bundledBy].hasPriority */
static bool bundle_ref_consistency(void) {
    Wrapper gn2w = linkage[NODE_GN2];
    if (gn2w.has_priority) return true;
    if (gn2w.bundled_by >= NUM_NODES) return false;
    return linkage[gn2w.bundled_by].has_priority;
}

/* ============================================================================
 * TLA+ actions.  Each function performs ONE atomic step under LK_LOCK and
 * returns true if it fired (enabling condition met) so the worker loop can
 * advance the pc the way the TLA+ Next disjunction would.
 * ============================================================================ */

/* ExternalMigration(t) -- TLA+ lines 146-176.
 * Out-of-band move of P2 from GN2 (in GN1's subtree) to P1 (external).
 * Enabled iff: ~bundleDone, linkage[P2].bundledBy = GN2, GN1 priority,
 * ~GN1.missing, and GN1.sub[GN2].sub[P2] /= Null.
 *
 * NOTE: the "fix attempt" baked into the spec sets GN1.missing=TRUE
 * atomically with the migration so SnapshotConsistency's ~missing guard
 * holds afterward. */
static bool external_migration(void) {
    LK_LOCK();
    Wrapper gn1w = linkage[NODE_GN1];
    Wrapper p2w  = linkage[NODE_P2];
    bool enabled =
        p2w.bundled_by == NODE_GN2 && !p2w.has_priority
        && gn1w.has_priority
        && !gn1w.packet.missing
        /* p2pkt = GN1.sub[GN2].sub[P2] /= Null */
        && gn1w.packet.has_gn2 && gn1w.packet.gn2_present
        && gn1w.packet.has_p2 && gn1w.packet.p2_present;
    if (!enabled) { LK_UNLOCK(); return false; }

    /* newGN1Pkt: sub[GN2] populated, sub[GN2].sub[P2] = Null, missing=TRUE */
    Packet newGN1Pkt = make_gn1_pkt(/*missing*/true,
                                    /*gn2_present*/true,
                                    /*gn2_p2_present*/false);
    /* newP1Pkt: sub[P2] now holds P2's packet (p2pkt), missing=FALSE */
    Packet newP1Pkt = make_p1_pkt(/*missing*/false, /*p2_present*/true);

    linkage[NODE_GN1] = priority_wrapper(newGN1Pkt);
    linkage[NODE_P1]  = priority_wrapper(newP1Pkt);
    linkage[NODE_P2]  = bundled_ref_wrapper(NODE_P1);
    LK_UNLOCK();
    return true;
}

/* BundleStart(t) -- TLA+ lines 181-187. pc idle -> phase1, reset local. */
static bool bundle_start(ThreadCtx *ctx) {
    /* pc[t] = idle /\ ~bundleDone[t] guaranteed by caller. */
    init_local(&ctx->local);
    ctx->pc = PC_PHASE1;
    return true;
}

/* BundlePhase1(t) -- TLA+ lines 193-219.  Collect wrappers + sub-packets. */
static bool bundle_phase1(ThreadCtx *ctx) {
    LK_LOCK();
    Wrapper gn1w = linkage[NODE_GN1];
    Wrapper gn2w = linkage[NODE_GN2];
    Wrapper p2w  = linkage[NODE_P2];
    Wrapper p1w  = linkage[NODE_P1];
    if (!gn1w.has_priority) { LK_UNLOCK(); return false; }  /* guard */

    Local *l = &ctx->local;
    l->wrapper    = gn1w;
    l->gn2Wrapper = gn2w;
    l->p2Wrapper  = p2w;
    l->p1Wrapper  = p1w;

    /* gn2SubPkt = IF gn2w.hasPriority THEN gn2w.packet ELSE gn1w.packet.sub[GN2] */
    if (gn2w.has_priority) {
        l->gn2SubPkt = gn2w.packet;
    } else {
        /* gn1w.packet.sub[GN2] : present iff gn2_present */
        if (gn1w.packet.has_gn2 && gn1w.packet.gn2_present) {
            /* reconstruct the GN2 sub-packet from GN1's flattened view:
             * its sub[P2] slot mirrors gn1w.packet.p2_present. */
            l->gn2SubPkt = gn1w.packet.p2_present
                ? make_gn2_pkt_valid_p2(false)
                : make_gn2_pkt_null_p2(false);
        } else {
            l->gn2SubPkt = NULL_PACKET;
        }
    }

    /* p2SubPkt =
     *   IF p2w.hasPriority THEN p2w.packet
     *   ELSE IF p2w.bundledBy = P1 /\ p1w.hasPriority THEN p1w.packet.sub[P2]
     *   ELSE IF p2w.bundledBy = GN2 /\ ~gn2w.hasPriority
     *           /\ gn1w.packet.sub[GN2] /= Null
     *        THEN gn1w.packet.sub[GN2].sub[P2]
     *   ELSE Null */
    if (p2w.has_priority) {
        l->p2SubPkt = p2w.packet;
    } else if (p2w.bundled_by == NODE_P1 && p1w.has_priority) {
        /* p1w.packet.sub[P2] : present iff p2_present */
        l->p2SubPkt = (p1w.packet.has_p2 && p1w.packet.p2_present)
            ? make_gn2_pkt_valid_p2(false) /* P2Packet shape: a leaf packet */
            : NULL_PACKET;
        /* P2Packet in the spec is MakePacket(P2,<<>>,FALSE) -- a leaf with
         * empty sub.  We only ever test it for /= Null, so any present
         * packet suffices; mark its node correctly below. */
        if (l->p2SubPkt.present) {
            l->p2SubPkt.node = NODE_P2;
            l->p2SubPkt.has_p2 = false; l->p2SubPkt.p2_present = false;
        }
    } else if (p2w.bundled_by == NODE_GN2 && !gn2w.has_priority
               && gn1w.packet.has_gn2 && gn1w.packet.gn2_present) {
        /* gn1w.packet.sub[GN2].sub[P2] : present iff p2_present */
        if (gn1w.packet.has_p2 && gn1w.packet.p2_present) {
            l->p2SubPkt = (Packet){ .present=true, .node=NODE_P2,
                                    .missing=false };
        } else {
            l->p2SubPkt = NULL_PACKET;
        }
    } else {
        l->p2SubPkt = NULL_PACKET;
    }

    ctx->pc = PC_PHASE2;
    LK_UNLOCK();
    return true;
}

/* BundlePhase2(t) -- TLA+ lines 225-245.  CAS GN1 to missing=TRUE state.
 * GN1.sub[GN2] holds GN2's bundled packet; GN2.sub[P2] stays Null. */
static bool bundle_phase2(ThreadCtx *ctx) {
    LK_LOCK();
    Local *l = &ctx->local;
    /* newGN2Pkt sub=[P2|->Null]; newGN1Pkt = MakePacket(GN1,[GN2|->newGN2Pkt],TRUE) */
    Packet newPkt = make_gn1_pkt(/*missing*/true,
                                 /*gn2_present*/true,
                                 /*gn2_p2_present*/false);
    Wrapper newW = priority_wrapper(newPkt);

    if (wrapper_eq(&linkage[NODE_GN1], &l->wrapper)) {
        linkage[NODE_GN1] = newW;
        l->wrapper = newW;
        ctx->pc = PC_PHASE3;
    } else {
        /* CAS failed -> restart from phase1 (TLA+ ELSE branch). */
        init_local(l);
        ctx->pc = PC_PHASE1;
    }
    LK_UNLOCK();
    return true;
}

/* BundlePhase3(t) -- TLA+ lines 252-264.  CAS GN2 -> BundledRef(GN1)
 * (or skip if already at GN1).  P2 is SKIPPED (skip-Null fix): its packet
 * does not move into GN2 (stays at P1.sub[P2]; GN2.sub[P2] = Null). */
static bool bundle_phase3(ThreadCtx *ctx) {
    LK_LOCK();
    Local *l = &ctx->local;
    if (wrapper_eq(&linkage[NODE_GN2], &l->gn2Wrapper)) {
        linkage[NODE_GN2] = bundled_ref_wrapper(NODE_GN1);
        l->gn2Wrapper = bundled_ref_wrapper(NODE_GN1);
        ctx->pc = PC_PHASE4;
        LK_UNLOCK();
        return true;
    }
    if (!linkage[NODE_GN2].has_priority
        && linkage[NODE_GN2].bundled_by == NODE_GN1) {
        ctx->pc = PC_PHASE4;
        LK_UNLOCK();
        return true;
    }
    /* Neither disjunct enabled -- not reachable in this model, but be
     * defensive: stay in phase3 so WF eventually retries (the gn2Wrapper
     * branch fires once linkage[GN2] matches the snapshot). */
    LK_UNLOCK();
    return false;
}

/* BundlePhase4(t) -- TLA+ lines 281-301.  is_bundle_root override, GATED
 * on the Phase4 reachability gate (subOK).  Only clears GN1.missing if
 * every Null grandchild slot is reachable within GN1's tree.
 *
 *   subOK == \A c in DOMAIN sub: sub[c] /= Null =>
 *               (\A gc in DOMAIN sub[c].sub:
 *                   sub[c].sub[gc] /= Null \/ ReachableFromGN1(oldPkt))
 *   targetMissing == ~subOK
 */
static bool bundle_phase4(ThreadCtx *ctx) {
    LK_LOCK();
    Local *l = &ctx->local;
    const Packet *old = &l->wrapper.packet;

    /* Evaluate subOK over the single (c=GN2, gc=P2) candidate. */
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

    if (wrapper_eq(&linkage[NODE_GN1], &l->wrapper)) {
        linkage[NODE_GN1] = finalW;
        l->wrapper = finalW;
        ctx->pc = PC_PHASE5;
    } else {
        init_local(l);
        ctx->pc = PC_PHASE1;
    }
    LK_UNLOCK();
    return true;
}

/* BundlePhase5(t) -- TLA+ lines 304-309.  Publish: done. */
static bool bundle_phase5(ThreadCtx *ctx) {
    init_local(&ctx->local);
    ctx->pc = PC_IDLE;
    atomic_fetch_add_explicit(&total_bundles, 1u, memory_order_relaxed);
    return true;
}

/* ============================================================================
 * Worker.  Drives one bundle pipeline per "commit", mirroring the TLA+
 * Next disjunction.  ExternalMigration is attempted opportunistically
 * before each pipeline (it is enabled only while P2 is still in-tree and
 * GN1 is priority+~missing), modelling the out-of-band race.
 *
 * Per the spec each thread runs ~bundleDone-gated; we re-arm for the next
 * commit by resetting pc to idle.  An iteration-step counter bounds the
 * inner loop so a wedged phase (should be impossible) cannot livelock the
 * unit test.
 * ============================================================================ */
static void *worker(void *arg) {
    ThreadCtx ctx;
    ctx.tid = ((ThreadCtx*)arg)->tid;
    ctx.pc  = PC_IDLE;
    init_local(&ctx.local);

    for (uint32_t commit = 0; commit < (uint32_t)MAX_COMMITS; commit++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;

        /* Opportunistically fire ExternalMigration (out-of-band).  Only
         * the first thread to find it enabled succeeds; others no-op. */
        external_migration();

        ctx.pc = PC_IDLE;
        /* Drive the bundle pipeline to completion.  Each phase is one
         * atomic TLA+ step.  Step budget bounds any defensive retry. */
        bool done = false;
        for (uint64_t step = 0; step < 1u << 20 && !done; step++) {
            switch (ctx.pc) {
                case PC_IDLE:   bundle_start(&ctx);            break;
                case PC_PHASE1: bundle_phase1(&ctx);           break;
                case PC_PHASE2: bundle_phase2(&ctx);           break;
                case PC_PHASE3: if (!bundle_phase3(&ctx)) {
                                    /* yield; peer must advance linkage */
                                    sched_yield();
                                }                              break;
                case PC_PHASE4: bundle_phase4(&ctx);           break;
                case PC_PHASE5: bundle_phase5(&ctx); done = true; break;
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
    /* BundleRefConsistency: GN2 bundled => bundler has priority. */
    assert(bundle_ref_consistency());

    /* SnapshotConsistency: the terminal safety invariant.  This is the
     * checkConsistensy(GN1) mirror -- GN1 published priority+~missing must
     * not leave an unreachable Null grandchild slot. */
    assert(snapshot_consistency());

    /* GN1 is the bundle root: always priority. */
    assert(linkage[NODE_GN1].has_priority);
}

int main(void) {
    /* Init (TLA+ lines 111-119):
     *   GN1 = PriorityWrapper(GN1, sub=[GN2 |-> GN2pkt(sub=[P2|->P2pkt])], FALSE)
     *   GN2 = BundledRefWrapper(GN1)
     *   P1  = PriorityWrapper(P1, sub=[P2 |-> Null], FALSE)
     *   P2  = BundledRefWrapper(GN2)        (initially in-tree under GN2)
     */
    linkage[NODE_GN1] = priority_wrapper(
        make_gn1_pkt(/*missing*/false, /*gn2_present*/true, /*gn2_p2_present*/true));
    linkage[NODE_GN2] = bundled_ref_wrapper(NODE_GN1);
    linkage[NODE_P1]  = priority_wrapper(
        make_p1_pkt(/*missing*/false, /*p2_present*/false));
    linkage[NODE_P2]  = bundled_ref_wrapper(NODE_GN2);

    atomic_store(&g_stop, false);
    atomic_store(&total_bundles, 0u);

    /* Sanity: Init satisfies both invariants. */
    assert(snapshot_consistency());
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

    /* Terminal invariant check: the published state must be consistent. */
    check_invariants();

    /* Terminal structural assertion mirroring the spec's conclusion:
     * after all bundles, if GN1 ended priority + ~missing then GN2.sub[P2]
     * (the hard-link slot) must be reachable within GN1's tree.  Given the
     * external migration / hard-link, the only consistent outcome is
     * EITHER GN1.missing stays TRUE (override denied by the Phase4 gate)
     * OR GN1.sub[GN2].sub[P2] is populated (P2 re-homed in-tree). */
    Wrapper gn1 = linkage[NODE_GN1];
    if (gn1.has_priority && !gn1.packet.missing) {
        /* override fired -> must be reachable */
        assert(reachable_from_gn1(&gn1.packet)
               && "GN1 published ~missing but P2 hard-link unreachable in GN1");
    }

    uint64_t bundles = atomic_load(&total_bundles);

#if STRESS_SECONDS > 0
    Wrapper p1 = linkage[NODE_P1];
    Wrapper p2 = linkage[NODE_P2];
    printf("[hardlink_external stress %ds threads=%d] bundles=%llu\n",
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
    /* Bounded unit: every thread must have completed exactly MAX_COMMITS
     * bundle pipelines (no livelock, full termination). */
    uint64_t expected = (uint64_t)MAX_COMMITS * (uint64_t)NUM_THREADS;
    assert(bundles == expected);
    (void)node_name; /* used only in stress diag */
    printf("[hardlink_external unit threads=%d commits=%d] bundles=%llu OK\n",
           NUM_THREADS, (int)MAX_COMMITS, (unsigned long long)bundles);
#endif

    return 0;
}
