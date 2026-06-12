/*
 * transaction_lookup_memo_test.cpp
 *
 * Correctness test for the 1-entry lookup memo in Snapshot::at() /
 * Transaction::operator[] (Snapshot<XN>::LookupMemo).
 *
 * Covers:
 *  1. repeated Snapshot reads (memo hit) return the same payload as fresh
 *     lookups, alternating between nodes (memo replacement),
 *  2. repeated Transaction subscripts return the transaction's clone,
 *     and writes through memo hits survive commit,
 *  3. release(tr, child) invalidates the memo — a subsequent tr[*child]
 *     throws NodeNotFoundError instead of silently hitting a detached
 *     packet,
 *  4. insert(tr, child, true) invalidates the memo and the new child is
 *     accessible,
 *  5. projection Snapshot(node, shot) does not inherit a memo that would
 *     mask NodeNotFoundError for nodes outside the projected subtree,
 *  6. a single Snapshot instance read from many threads concurrently
 *     (tear-safety of the relaxed memo pair).
 */

#include "support_standalone.h"

#include <stdint.h>
#include <thread>
#include <vector>

#include "transaction.h"
#include "transaction_impl.h"

#include "xthread.cpp"

atomic<int> objcnt = 0;
atomic<long> total = 0;

class LongNode : public Transactional::Node<LongNode> {
public:
    LongNode() { ++objcnt; }
    virtual ~LongNode() { --objcnt; }

    struct Payload : public Transactional::Node<LongNode>::Payload {
        Payload() : Transactional::Node<LongNode>::Payload(), m_x(0) {}
        long m_x;
    };
};

using Snapshot = Transactional::Snapshot<LongNode>;
using Transaction = Transactional::Transaction<LongNode>;

static int s_failures = 0;
#define VERIFY(cond) do { if( !(cond)) { \
    fprintf(stderr, "FAIL at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    ++s_failures; } } while(0)

int main() {
    {
        // --- 0. local_weak_ptr::same_control_block primitive -----------
        // (Underpins reverseLookupWithHint's promotion-free fast path.)
        {
            struct W { int v = 0; };
            local_shared_ptr<W> p(make_local_shared<W>());
            local_shared_ptr<W> q(make_local_shared<W>());
            local_weak_ptr<W> wp(p);
            VERIFY(wp.same_control_block(p));        // same CB
            local_shared_ptr<W> p2(p);               // shares p's CB
            VERIFY(wp.same_control_block(p2));        // identity, not handle eq
            VERIFY( !wp.same_control_block(q));       // distinct CB
            local_shared_ptr<W> empty;
            VERIFY( !wp.same_control_block(empty));    // live weak vs null
            local_weak_ptr<W> empty_wp;
            VERIFY( !empty_wp.same_control_block(p));  // null weak vs live
            VERIFY(empty_wp.same_control_block(empty)); // null == null
        }

        shared_ptr<LongNode> root(LongNode::create<LongNode>());
        shared_ptr<LongNode> a(LongNode::create<LongNode>());
        shared_ptr<LongNode> b(LongNode::create<LongNode>());
        shared_ptr<LongNode> outside(LongNode::create<LongNode>());
        root->insert(a);
        root->insert(b);

        // --- 1. Snapshot repeated/alternating reads --------------------
        root->iterate_commit([&](Transaction &tr) {
            tr[ *a].m_x = 11;
            tr[ *b].m_x = 22;
        });
        {
            Snapshot shot( *root);
            // first lookups populate the memo, repeats must agree
            VERIFY(shot[ *a].m_x == 11);
            VERIFY(shot[ *a].m_x == 11);   // memo hit
            VERIFY(shot[ *b].m_x == 22);   // memo replaced
            VERIFY(shot[ *a].m_x == 11);   // replaced back
            VERIFY( &shot[ *a] == &shot[ *a]); // identical payload object
            // copy carries a usable memo
            Snapshot shot2(shot);
            VERIFY(shot2[ *a].m_x == 11);
            VERIFY(shot2[ *b].m_x == 22);
        }

        // --- 2. Transaction repeated subscripts + commit ----------------
        root->iterate_commit([&](Transaction &tr) {
            tr[ *a].m_x = 1;
            tr[ *a].m_x += 2;            // memo hit (serial match)
            tr[ *a].m_x += 4;            // memo hit
            VERIFY( &tr[ *a] == &tr[ *a]);  // same clone
            tr[ *b].m_x = 100;           // memo replaced
            tr[ *a].m_x += 8;            // full lookup again, same clone
            // const view through the Snapshot base must see the clone
            const Snapshot &shot(tr);
            VERIFY(shot[ *a].m_x == 15);
        });
        {
            Snapshot shot( *root);
            VERIFY(shot[ *a].m_x == 15);
            VERIFY(shot[ *b].m_x == 100);
        }

        // --- 3. release(tr) invalidates the memo ------------------------
        root->iterate_commit_if([&](Transaction &tr) -> bool {
            tr[ *b].m_x = 777;          // memo := {b, ...}
            if( !root->release(tr, b))
                return false;
            bool thrown = false;
            try {
                tr[ *b].m_x = 888;      // must throw, not hit stale memo
            } catch (Transactional::Node<LongNode>::NodeNotFoundError &) {
                thrown = true;
            }
            VERIFY(thrown);
            return true;
        });

        // --- 4. insert(tr, child, true) invalidates + new child usable --
        root->iterate_commit_if([&](Transaction &tr) -> bool {
            tr[ *a].m_x = 1000;         // memo := {a, ...}
            if( !root->insert(tr, b, true))
                return false;
            tr[ *b].m_x = 2000;         // online-inserted child
            VERIFY(tr[ *a].m_x == 1000); // a's clone still addressed correctly
            return true;
        });
        {
            Snapshot shot( *root);
            VERIFY(shot[ *a].m_x == 1000);
            VERIFY(shot[ *b].m_x == 2000);
        }

        // --- 5. projection must not inherit a masking memo --------------
        {
            Snapshot shot( *root);
            VERIFY(shot[ *a].m_x == 1000);  // memo := {a, ...}
            Snapshot shot_b( *b, shot);     // project to b's subtree
            bool thrown = false;
            try {
                (void)shot_b.at( *a);       // a is outside b's subtree
            } catch (Transactional::Node<LongNode>::NodeNotFoundError &) {
                thrown = true;
            }
            VERIFY(thrown);
        }

        // --- 6. one Snapshot instance, many concurrent readers ----------
        root->iterate_commit([&](Transaction &tr) {
            tr[ *a].m_x = 0x0a0a;
            tr[ *b].m_x = 0x0b0b;
        });
        {
            Snapshot shot( *root);
            atomic<int> mismatches = 0;
            std::vector<std::thread> threads;
            for(int t = 0; t < 8; ++t) {
                threads.emplace_back([&shot, &a, &b, &mismatches, t]() {
                    for(int i = 0; i < 200000; ++i) {
                        // interleave so peers keep replacing the memo
                        if((i + t) % 2) {
                            if(shot[ *a].m_x != 0x0a0a) ++mismatches;
                        }
                        else {
                            if(shot[ *b].m_x != 0x0b0b) ++mismatches;
                        }
                    }
                });
            }
            for(auto &th: threads) th.join();
            VERIFY(mismatches == 0);
        }

        // --- 7. multi-slot: const view must see the clone, not a stale
        // committed payload left in another slot (set() uniqueness).
        root->iterate_commit([&](Transaction &tr) {
            const Snapshot &shot(tr);
            long before = shot[ *a].m_x;     // caches the COMMITTED payload
            tr[ *a].m_x = before + 5000;     // clone; set() must overwrite, not append
            VERIFY(shot[ *a].m_x == before + 5000);  // stale-slot regression
        });

        // --- 8. multi-slot: rotations within and beyond capacity ---------
        {
            shared_ptr<LongNode> more[6];
            for(int i = 0; i < 6; ++i) {
                more[i] = shared_ptr<LongNode>(LongNode::create<LongNode>());
                root->insert(more[i]);
            }
            root->iterate_commit([&](Transaction &tr) {
                for(int rep = 0; rep < 3; ++rep)      // 6 nodes > SLOTS: evictions
                    for(int i = 0; i < 6; ++i)
                        tr[ *more[i]].m_x = 10 * (i + 1) + rep;
            });
            Snapshot shot( *root);
            for(int rep = 0; rep < 3; ++rep)          // reads through evictions
                for(int i = 0; i < 6; ++i)
                    VERIFY(shot[ *more[i]].m_x == 10 * (i + 1) + 2);
            // 4-node rotation (== SLOTS) must stay coherent across repeats
            for(int rep = 0; rep < 8; ++rep)
                for(int i = 0; i < 4; ++i)
                    VERIFY(shot[ *more[i]].m_x == 10 * (i + 1) + 2);
            for(int i = 0; i < 6; ++i)
                root->release(more[i]);
        }

        (void)outside;
    }

    if(s_failures) {
        fprintf(stderr, "FAILED (%d)\n", s_failures);
        return -1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
