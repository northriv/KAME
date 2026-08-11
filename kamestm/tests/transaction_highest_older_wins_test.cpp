/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/
//! Pins the HIGHEST-vs-HIGHEST older-wins contract (per user, 2026-08-11):
//! between two HIGHEST transactions the loser must DEFER to a live
//! privileged peer — the same fair-mode predicate a NORMAL loser sleeps on —
//! not keep firing CAS while the stamp comparison decides nothing.  The tier
//! contract (never park) is kept by spinning instead of sleeping, so what
//! this test asserts is the *blocking*, and its prompt release.
//!
//! Four arms, mirroring NORMAL-vs-NORMAL exactly:
//!   1. foreign HIGHEST *Reserved* stamp   -> a HIGHEST commit is HELD, and
//!      completes promptly once the stamp clears (the teeth).
//!   2. foreign HIGHEST *plain* tag        -> does NOT hold a HIGHEST commit
//!      (plain tags never sleep-block a NORMAL either).
//!   3. foreign *non-HIGHEST* Reserved     -> does NOT hold a HIGHEST commit
//!      (cross-tier behaviour unchanged: HIGHEST does not park behind a
//!      foreign-tier privilege; Rule 0 handles the stuck case).
//!   4. the NORMAL mirror of arm 1         -> a NORMAL commit is held by a
//!      HIGHEST Reserved stamp too (criterion: a HIGHEST tag/privilege reads
//!      to NORMAL like an older one).
//!
//! White-box (-fno-access-control), like transaction_priv_strip_test: a
//! privilege claim is probe-gated and cannot be manufactured
//! deterministically through the public API, so the Reserved stamp is
//! planted on the Linkage directly.

#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

class MyNode : public Transactional::Node<MyNode> {
public:
    struct Payload : public Transactional::Node<MyNode>::Payload {
        long m_x = 0;
    };
};
typedef Transactional::Transaction<MyNode> Tr;
using NC = Transactional::Node<MyNode>::NegotiationCounter;

//! A synthetic stamp \a age_us in the past, held by foreign \a tid.
static NC::cnt_t fake_stamp(uint16_t tid, int64_t age_us, bool highest,
                            bool reserved) {
    NC::cnt_t st = ((NC::cnt_t)(NC::now_us() - age_us) & NC::STAMP_US_MASK)
        | ((NC::cnt_t)tid << NC::STAMP_TID_SHIFT);
    if(highest) st = NC::with_highest_flag(st);
    if(reserved)
        st = NC::with_kind(st, Transactional::detail::StampKind::Reserved);
    return st;
}

//! Run one arm: plant \a blocker on the node's linkage, start a committer at
//! \a pr, and report whether it was still blocked after \a hold_ms.  The
//! blocker is then cleared and the committer must finish within 2 s.
//! Returns {held, released}.
struct ArmResult { bool held; bool released; };
static ArmResult run_arm(Transactional::Priority pr, NC::cnt_t blocker,
                         int hold_ms) {
    shared_ptr<MyNode> node(MyNode::create<MyNode>());
    shared_ptr<MyNode> leaf(MyNode::create<MyNode>());
    node->insert(leaf);   // multi-nodal => commits negotiate + tag

    auto &slot = node->m_link->m_transaction_started_time;
    slot.store(blocker);

    std::atomic<bool> committed{false};
    std::thread t([&]{
        Transactional::ScopedPriority p(pr);
        node->iterate_commit([&](Tr &tr){ tr[ *node].m_x++; });
        committed.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    const bool held = !committed.load();

    slot.store((NC::cnt_t)0);    // holder "commits": stamp clears
    bool released = false;
    for(int i = 0; i < 2000; ++i) {
        if(committed.load()) { released = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if(released) t.join(); else t.detach();  // leak the thread over hanging
    return {held, released};
}

int main() {
    int failures = 0;
    auto check = [&](const char *what, bool ok) {
        std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
        if( !ok) ++failures;
    };
    const uint16_t foreign_tid = 0x7ee1;   // != any live ProcessCounter id
    using P = Transactional::Priority;

    {   // 1. HIGHEST behind a live privileged HIGHEST peer: held, then freed.
        auto r = run_arm(P::HIGHEST,
                         fake_stamp(foreign_tid, 1000, true, true), 80);
        check("HIGHEST defers to a HIGHEST Reserved stamp", r.held);
        check("...and proceeds the moment it clears", r.released);
    }
    {   // 2. plain HIGHEST tag: no blocking (mirror of NORMAL vs plain tag).
        auto r = run_arm(P::HIGHEST,
                         fake_stamp(foreign_tid, 1000, true, false), 80);
        check("a plain HIGHEST tag does not hold a HIGHEST commit",
              !r.held && r.released);
    }
    {   // 3. foreign-tier privilege: HIGHEST passes through (tier contract).
        auto r = run_arm(P::HIGHEST,
                         fake_stamp(foreign_tid, 1000, false, true), 80);
        check("HIGHEST does not park behind a non-HIGHEST Reserved",
              !r.held && r.released);
    }
    {   // 4. NORMAL mirror: held by the same HIGHEST Reserved stamp.
        auto r = run_arm(P::NORMAL,
                         fake_stamp(foreign_tid, 1000, true, true), 80);
        check("NORMAL is held by a HIGHEST Reserved stamp", r.held);
        check("...and proceeds the moment it clears", r.released);
    }

    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
