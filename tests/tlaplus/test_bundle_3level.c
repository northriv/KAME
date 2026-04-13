/*
 * C11 test generated mechanically from BundleUnbundle.tla (Layer 2, 3-level)
 *
 * Tree structure:
 *   Grand --+-- Parent --+-- Child1
 *                         +-- Child2
 *
 * Models:
 *   - Recursive bundling: snapshot(Grand) bundles Parent, which bundles Children
 *   - Multi-level unbundle: commit(Child) when bundled 2 levels deep
 *   - snapshotSupernode() walking up through bundledBy chain
 *
 * TLA+ variable mapping:
 *   Same packing approach as 2-level test.
 *   Grand's sub-packets hold Parent's wrapper.
 *   Parent's sub-packets hold Child1/Child2's wrappers.
 *   We encode the nested structure with separate atomic words per node.
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>

/* --- Node IDs --- */
#define NODE_GRAND   0
#define NODE_PARENT  1
#define NODE_CHILD1  2
#define NODE_CHILD2  3
#define NUM_NODES    4
#define NULL_NODE    0xFF

/* --- Modular serial arithmetic --- */
#define MAX_SERIAL  16  /* must be even */
#define MAX_PAYLOAD 4

static inline bool mod_gt(uint8_t a, uint8_t b) {
    uint8_t diff = (a - b + MAX_SERIAL) % MAX_SERIAL;
    return diff > 0 && diff < MAX_SERIAL / 2;
}

static inline uint8_t gen_serial(uint8_t thread_ser, uint8_t last_ser) {
    uint8_t base = mod_gt(last_ser, thread_ser) ? last_ser : thread_ser;
    return (base + 1) % MAX_SERIAL;
}

/* --- ParentOf / ChildrenOf --- */
static int parent_of(int n) {
    if (n == NODE_CHILD1 || n == NODE_CHILD2) return NODE_PARENT;
    if (n == NODE_PARENT) return NODE_GRAND;
    return NULL_NODE;
}

/* --- Wrapper per node --- */
/*
 * Each node's linkage is an atomic 64-bit word packing:
 *   hasPriority, serial, bundledBy, missing, payload,
 *   sub-packet payloads for its children
 *
 * Grand's children: {Parent}
 * Parent's children: {Child1, Child2}
 * Children: no sub-packets (leaf)
 *
 * Encoding (same as 2-level but with different sub-packet layout per node):
 *   bits 63:    hasPriority
 *   bits 62-56: serial (7 bits)
 *   bits 55-48: bundledBy node id
 *   bits 47:    missing
 *   bits 46-40: payload (7 bits)
 *   For Grand:  bits 39-32: sub_parent_payload (W_NULL_SUB if absent)
 *               bits 31:    sub_parent_missing
 *               bits 30-24: sub_parent_sub_child1 (nested)
 *               bits 23-17: sub_parent_sub_child2 (nested)
 *   For Parent: bits 39-32: sub_child1_payload
 *               bits 31-24: sub_child2_payload
 *   For leaves: no sub-packets
 *
 * This nested encoding is complex. For clarity and to avoid errors,
 * we use a simpler representation: each node has its own atomic word
 * for its direct state, and we use a separate "sub-packet store" for
 * the bundle's nested sub-packet data.
 */

/* Simplified approach: each node's wrapper is an atomic struct-like word
 * encoding just its own level's data. Bundle state is tracked by whether
 * nodes are priority or bundled-ref. The actual sub-packet data lives
 * in the parent's wrapper encoding. */

#define W_NULL_SUB 0xFF

typedef struct {
    bool     has_priority;
    uint8_t  serial;
    uint8_t  bundled_by;    /* NULL_NODE if priority */
    bool     missing;
    uint8_t  payload;
    /* Sub-packet payloads (only meaningful if has_priority) */
    /* For Grand: sub[0]=Parent's payload */
    /* For Parent: sub[0]=Child1's payload, sub[1]=Child2's payload */
    uint8_t  sub[2];
    /* Nested: for Grand, we also need Parent's sub-packets */
    uint8_t  sub_nested[2]; /* Grand only: Parent's child1/child2 payloads */
    bool     sub_missing;   /* sub-packet's missing flag */
} Wrapper;

/* Pack to 64 bits */
static uint64_t wrapper_pack(Wrapper w) {
    uint64_t v = 0;
    v |= ((uint64_t)w.has_priority) << 63;
    v |= ((uint64_t)(w.serial & 0x7F)) << 56;
    v |= ((uint64_t)w.bundled_by) << 48;
    v |= ((uint64_t)w.missing) << 47;
    v |= ((uint64_t)(w.payload & 0x7F)) << 40;
    v |= ((uint64_t)w.sub[0]) << 32;
    v |= ((uint64_t)w.sub[1]) << 24;
    v |= ((uint64_t)w.sub_nested[0]) << 16;
    v |= ((uint64_t)w.sub_nested[1]) << 8;
    v |= ((uint64_t)w.sub_missing);
    return v;
}

static Wrapper wrapper_unpack(uint64_t v) {
    Wrapper w;
    w.has_priority = (v >> 63) & 1;
    w.serial       = (v >> 56) & 0x7F;
    w.bundled_by   = (v >> 48) & 0xFF;
    w.missing      = (v >> 47) & 1;
    w.payload      = (v >> 40) & 0x7F;
    w.sub[0]       = (v >> 32) & 0xFF;
    w.sub[1]       = (v >> 24) & 0xFF;
    w.sub_nested[0]= (v >> 16) & 0xFF;
    w.sub_nested[1]= (v >> 8)  & 0xFF;
    w.sub_missing  = v & 1;
    return w;
}

/* --- Shared state --- */
_Atomic(uint64_t) linkage[NUM_NODES];

/* Per-node commit counter (model-only, for QuiescentCheck) */
_Atomic(uint32_t) commit_count[NUM_NODES];

static Wrapper load_wrapper(int node) {
    return wrapper_unpack(atomic_load_explicit(&linkage[node], memory_order_acquire));
}

static bool cas_wrapper(int node, Wrapper *expected, Wrapper desired) {
    uint64_t exp_val = wrapper_pack(*expected);
    uint64_t des_val = wrapper_pack(desired);
    bool ok = atomic_compare_exchange_strong_explicit(&linkage[node],
            &exp_val, des_val, memory_order_acq_rel, memory_order_relaxed);
    if (!ok) *expected = wrapper_unpack(exp_val);
    return ok;
}

static Wrapper make_priority(uint8_t payload, uint8_t serial,
                              uint8_t s0, uint8_t s1, bool missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .missing = missing, .payload = payload,
        .sub = {s0, s1}, .sub_nested = {W_NULL_SUB, W_NULL_SUB},
        .sub_missing = false
    };
}

static Wrapper make_bundled_ref(uint8_t parent_node, uint8_t serial) {
    return (Wrapper){
        .has_priority = false, .serial = serial, .bundled_by = parent_node,
        .missing = false, .payload = 0,
        .sub = {W_NULL_SUB, W_NULL_SUB}, .sub_nested = {W_NULL_SUB, W_NULL_SUB},
        .sub_missing = false
    };
}

/* ========================================================================== */
/* Snapshot Grand (with recursive bundle)                                     */
/* ========================================================================== */

/*
 * snapshot(Grand): SnapRead → SnapCheck → Bundle 4-phase → SnapDone
 * May recursively bundle Parent's children.
 */
static uint8_t snapshot_grand(uint8_t *thread_ser) {
    for (;;) {
        /* SnapCheck: read Grand */
        Wrapper gw = load_wrapper(NODE_GRAND);

        if (gw.has_priority && !gw.missing) {
            /* Fast path: Grand has all sub-packets, complete */
            assert(gw.sub[0] != W_NULL_SUB);  /* Parent sub present */
            return gw.payload;
        }

        if (gw.has_priority && gw.missing) {
            /* Bundle Grand's child (Parent) */
            uint8_t bundle_ser = gen_serial(*thread_ser, gw.serial);
            *thread_ser = bundle_ser;

            /* BundlePhase1: collect Parent's state */
            Wrapper pw = load_wrapper(NODE_PARENT);
            uint8_t parent_payload = W_NULL_SUB;

            if (pw.has_priority) {
                if (!pw.missing) {
                    parent_payload = pw.payload;
                } else {
                    /* Parent is missing → need to bundle Parent first.
                     * For simplicity, we first bundle Parent's children,
                     * then retry Grand's bundle. */

                    /* Bundle Parent's children */
                    uint8_t p_ser = gen_serial(*thread_ser, pw.serial);
                    *thread_ser = p_ser;

                    Wrapper c1w = load_wrapper(NODE_CHILD1);
                    Wrapper c2w = load_wrapper(NODE_CHILD2);
                    uint8_t sp1 = W_NULL_SUB, sp2 = W_NULL_SUB;

                    if (c1w.has_priority) sp1 = c1w.payload;
                    else if (c1w.bundled_by == NODE_PARENT) sp1 = pw.sub[0];

                    if (c2w.has_priority) sp2 = c2w.payload;
                    else if (c2w.bundled_by == NODE_PARENT) sp2 = pw.sub[1];

                    if (sp1 == W_NULL_SUB || sp2 == W_NULL_SUB) continue;

                    /* CAS Parent with subs (still missing=TRUE) */
                    Wrapper new_pw = make_priority(pw.payload, p_ser, sp1, sp2, true);
                    Wrapper exp_pw = pw;
                    if (!cas_wrapper(NODE_PARENT, &exp_pw, new_pw)) continue;

                    /* CAS children to bundled-ref */
                    Wrapper exp_c1 = c1w, exp_c2 = c2w;
                    if (!cas_wrapper(NODE_CHILD1, &exp_c1,
                                     make_bundled_ref(NODE_PARENT, p_ser)) ||
                        !cas_wrapper(NODE_CHILD2, &exp_c2,
                                     make_bundled_ref(NODE_PARENT, p_ser))) {
                        continue;
                    }

                    /* Finalize Parent: missing=FALSE */
                    Wrapper final_pw = make_priority(pw.payload, p_ser, sp1, sp2, false);
                    Wrapper exp_pw2 = new_pw;
                    if (!cas_wrapper(NODE_PARENT, &exp_pw2, final_pw)) continue;

                    /* Now retry Grand's bundle with bundled Parent */
                    continue;
                }
            } else if (pw.bundled_by == NODE_GRAND) {
                /* Parent already bundled by Grand, use existing sub */
                parent_payload = gw.sub[0];
            } else {
                continue;  /* can't collect */
            }

            if (parent_payload == W_NULL_SUB) continue;

            /* BundlePhase2: CAS Grand with Parent's packet (still missing=TRUE) */
            Wrapper new_gw = make_priority(gw.payload, bundle_ser,
                                            parent_payload, W_NULL_SUB, true);
            /* Store Parent's sub-packets in Grand's nested slots */
            Wrapper pw_now = load_wrapper(NODE_PARENT);
            if (pw_now.has_priority && !pw_now.missing) {
                new_gw.sub_nested[0] = pw_now.sub[0];
                new_gw.sub_nested[1] = pw_now.sub[1];
            }
            Wrapper exp_gw = gw;
            if (!cas_wrapper(NODE_GRAND, &exp_gw, new_gw)) continue;

            /* BundlePhase3: CAS Parent to bundled-ref pointing to Grand */
            Wrapper exp_pw3 = pw_now;
            if (!cas_wrapper(NODE_PARENT, &exp_pw3,
                             make_bundled_ref(NODE_GRAND, bundle_ser))) {
                continue;
            }

            /* BundlePhase4: finalize Grand, missing=FALSE */
            Wrapper final_gw = new_gw;
            final_gw.missing = false;
            Wrapper exp_gw2 = new_gw;
            if (!cas_wrapper(NODE_GRAND, &exp_gw2, final_gw)) continue;

            assert(final_gw.sub[0] != W_NULL_SUB);
            return final_gw.payload;
        }

        /* Grand is bundled (shouldn't happen for root) → retry */
        continue;
    }
}

/* ========================================================================== */
/* Commit on a child node (with 2-level unbundle)                             */
/* ========================================================================== */

static bool commit_child(int child_node, uint8_t *thread_ser) {
    for (;;) {
        /* CommitRead */
        Wrapper cw = load_wrapper(child_node);

        if (cw.has_priority) {
            /* CommitTryCAS: direct commit */
            uint8_t new_payload = (cw.payload + 1) % MAX_PAYLOAD;
            uint8_t ser = gen_serial(*thread_ser, cw.serial);
            Wrapper new_cw = make_priority(new_payload, ser,
                                            cw.sub[0], cw.sub[1], cw.missing);
            Wrapper expected = cw;
            if (cas_wrapper(child_node, &expected, new_cw)) {
                *thread_ser = ser;
                atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
                return true;
            }
            if (expected.has_priority && expected.payload == cw.payload) {
                continue;  /* single-node optimization: retry with adopted children */
            }
            if (expected.has_priority) {
                return false;  /* true conflict */
            }
            cw = expected;  /* got bundled → fall through */
        }

        /* UnbundleWalk: walk up bundledBy chain */
        int parent = cw.bundled_by;
        if (parent == NULL_NODE) continue;

        Wrapper parent_w = load_wrapper(parent);

        if (parent_w.has_priority) {
            /* 1-level unbundle: UnbundleCASAncestor → UnbundleCASChild */
            /* Save parentWrapper (local[t].parentWrapper = parentW) */
            Wrapper saved_parent_w = parent_w;
            int sub_idx = (child_node == NODE_CHILD1) ? 0 : 1;
            if (saved_parent_w.sub[sub_idx] == W_NULL_SUB) { continue; }
            uint8_t old_payload = saved_parent_w.sub[sub_idx];
            uint8_t new_payload = (old_payload + 1) % MAX_PAYLOAD;

            /* UnbundleCASAncestor: CAS parent using saved parentWrapper */
            uint8_t ser = gen_serial(*thread_ser, saved_parent_w.serial);
            Wrapper new_pw = saved_parent_w;
            new_pw.serial = ser;
            new_pw.sub[sub_idx] = W_NULL_SUB;
            new_pw.missing = true;
            Wrapper exp_pw = saved_parent_w;
            if (!cas_wrapper(parent, &exp_pw, new_pw)) continue;
            *thread_ser = ser;

            /* UnbundleCASChild: restore to priority */
            uint8_t child_ser = gen_serial(*thread_ser, cw.serial);
            Wrapper new_child = make_priority(new_payload, child_ser,
                                               W_NULL_SUB, W_NULL_SUB, false);
            Wrapper exp_child = cw;
            if (cas_wrapper(child_node, &exp_child, new_child)) {
                *thread_ser = child_ser;
                atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
                return true;
            }
            continue;  /* retry */
        }

        /* Parent also bundled → 2-level unbundle (UnbundleCASGP) */
        int gp = parent_w.bundled_by;
        if (gp == NULL_NODE) { continue; }

        Wrapper gp_w = load_wrapper(gp);
        if (!gp_w.has_priority) continue;

        /* Save gpWrapper (local[t].gpWrapper = gpW) */
        Wrapper saved_gp_w = gp_w;

        /* Extract child's payload from GP's nested sub-packets */
        int sub_idx = (child_node == NODE_CHILD1) ? 0 : 1;
        uint8_t old_payload;
        if (gp == NODE_GRAND && parent == NODE_PARENT) {
            if (saved_gp_w.sub[0] == W_NULL_SUB) continue;
            old_payload = saved_gp_w.sub_nested[sub_idx];
            if (old_payload == W_NULL_SUB) continue;
        } else {
            continue;
        }

        uint8_t new_payload = (old_payload + 1) % MAX_PAYLOAD;

        /* UnbundleCASGP: CAS GP using saved gpWrapper */
        uint8_t ser = gen_serial(*thread_ser, saved_gp_w.serial);
        Wrapper new_gp = saved_gp_w;
        new_gp.serial = ser;
        new_gp.sub_nested[sub_idx] = W_NULL_SUB;
        new_gp.missing = true;
        Wrapper exp_gp = saved_gp_w;
        if (!cas_wrapper(gp, &exp_gp, new_gp)) continue;
        *thread_ser = ser;

        /* UnbundleRestoreParent: restore Parent to priority */
        Wrapper pw_now = load_wrapper(NODE_PARENT);
        if (!pw_now.has_priority && pw_now.bundled_by == NODE_GRAND) {
            /* Extract Parent's packet from GP and restore */
            Wrapper gp_now = load_wrapper(NODE_GRAND);
            if (gp_now.has_priority && gp_now.sub[0] != W_NULL_SUB) {
                uint8_t p_ser = gen_serial(*thread_ser, pw_now.serial);
                Wrapper restored_pw = make_priority(gp_now.sub[0], p_ser,
                    gp_now.sub_nested[0], gp_now.sub_nested[1], true);
                restored_pw.sub[sub_idx] = W_NULL_SUB;  /* child being unbundled */
                Wrapper exp_pw = pw_now;
                if (!cas_wrapper(NODE_PARENT, &exp_pw, restored_pw)) continue;
                *thread_ser = p_ser;
            } else {
                continue;
            }
        }
        /* else: parent already has priority (another thread restored it) */

        /* UnbundleCASChild: restore child to priority */
        uint8_t child_ser = gen_serial(*thread_ser, cw.serial);
        Wrapper new_child = make_priority(new_payload, child_ser,
                                           W_NULL_SUB, W_NULL_SUB, false);
        Wrapper exp_child = cw;
        if (cas_wrapper(child_node, &exp_child, new_child)) {
            *thread_ser = child_ser;
            atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
            return true;
        }
        /* retry */
    }
}

/* ========================================================================== */
/* Test threads                                                               */
/* ========================================================================== */

/* Thread 0: snapshot Grand (exercises recursive bundle) */
static void *thread_snapshot(void *arg) {
    (void)arg;
    uint8_t my_serial = 0;
    uint8_t result = snapshot_grand(&my_serial);
    (void)result;
    return NULL;
}

/* Thread 1: commit on Child1 (exercises unbundle including 2-level) */
static void *thread_commit(void *arg) {
    (void)arg;
    uint8_t my_serial = 0;
    bool ok = commit_child(NODE_CHILD1, &my_serial);
    (void)ok;
    return NULL;
}

int main(void) {
    /* Init:
     *   Grand:  priority, missing=TRUE (no sub-packets)
     *   Parent: priority, missing=TRUE (no sub-packets)
     *   Child1, Child2: priority, missing=FALSE (leaf nodes)
     */
    Wrapper init_grand = make_priority(0, 0, W_NULL_SUB, W_NULL_SUB, true);
    Wrapper init_parent = make_priority(0, 0, W_NULL_SUB, W_NULL_SUB, true);
    Wrapper init_child1 = make_priority(0, 0, W_NULL_SUB, W_NULL_SUB, false);
    Wrapper init_child2 = make_priority(0, 0, W_NULL_SUB, W_NULL_SUB, false);

    /* Grand's sub[0] is for Parent, sub[1] unused */
    /* Parent's sub[0] is for Child1, sub[1] for Child2 */

    atomic_store(&linkage[NODE_GRAND],  wrapper_pack(init_grand));
    atomic_store(&linkage[NODE_PARENT], wrapper_pack(init_parent));
    atomic_store(&linkage[NODE_CHILD1], wrapper_pack(init_child1));
    atomic_store(&linkage[NODE_CHILD2], wrapper_pack(init_child2));
    for (int i = 0; i < NUM_NODES; i++)
        atomic_store(&commit_count[i], 0);

    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread_snapshot, NULL);
    pthread_create(&t1, NULL, thread_commit, NULL);

    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    /* --- Post-join safety invariants --- */

    /* GrandAlwaysPriority */
    Wrapper gw = load_wrapper(NODE_GRAND);
    assert(gw.has_priority);

    /* SnapshotConsistency for inner nodes */
    if (gw.has_priority && !gw.missing) {
        assert(gw.sub[0] != W_NULL_SUB);
    }
    Wrapper pw = load_wrapper(NODE_PARENT);
    if (pw.has_priority && !pw.missing) {
        assert(pw.sub[0] != W_NULL_SUB);
        assert(pw.sub[1] != W_NULL_SUB);
    }

    /* NoPriorityLoss */
    for (int n = NODE_PARENT; n <= NODE_CHILD2; n++) {
        Wrapper w = load_wrapper(n);
        assert(w.has_priority || w.bundled_by != NULL_NODE);
    }

    /* BundledByCorrect: bundledBy points to structural parent */
    for (int n = NODE_PARENT; n <= NODE_CHILD2; n++) {
        Wrapper w = load_wrapper(n);
        if (!w.has_priority) {
            assert(w.bundled_by == (uint8_t)parent_of(n));
        }
    }

    /* BundleChainValid: if bundled, bundledBy target has priority or is itself bundled */
    for (int n = NODE_PARENT; n <= NODE_CHILD2; n++) {
        Wrapper w = load_wrapper(n);
        if (!w.has_priority && w.bundled_by != NULL_NODE) {
            Wrapper p = load_wrapper(w.bundled_by);
            assert(p.has_priority || p.bundled_by != NULL_NODE);
        }
    }

    /* QuiescentCheck: all threads idle → non-Grand hasPriority nodes have
     * payload == commit_count[node] % MAX_PAYLOAD */
    for (int n = NODE_PARENT; n <= NODE_CHILD2; n++) {
        Wrapper w = load_wrapper(n);
        if (w.has_priority) {
            uint32_t cc = atomic_load(&commit_count[n]);
            assert(w.payload == cc % MAX_PAYLOAD);
        }
    }

    return 0;
}
