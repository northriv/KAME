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
//! Regression test for tag_as_contender's Rule 0: a HIGHEST tagger strips a
//! foreign non-HIGHEST Reserved (privilege) stamp it has been stuck behind for
//! KAME_STM_PREEMPT_WINDOW_US — and never strips anything else.
//!
//! Why this exists as a white-box test: the strip fired **zero** times in every
//! benchmark scenario (grand -t 8 -P 1, five interleaved reps), which is the
//! intended production behaviour — a healthy privilege holder commits within
//! microseconds and must never be stripped, because while it holds, fair-mode
//! silences the other NORMAL contenders and thins the HIGHEST's opposition
//! (stripping on sight was measured NET NEGATIVE: aggregate −4.6 %, HIGHEST
//! p99.9 1.5 → 2.6 µs).  But "fired zero times, cost nothing" proves only half
//! the design; this test manufactures the other half — the genuinely stuck
//! holder the rule insures against — which cannot be produced deterministically
//! through the public API (privilege claims are gated behind the livelock
//! probe).  So it plants a synthetic Reserved stamp + side word directly on the
//! Linkage.  Built with -fno-access-control (see CMakeLists.txt): Node's
//! Linkage and NegotiationCounter are private by design, and this test is
//! deliberately white-box.
//!
//! The four cases mirror the decision exactly:
//!   A. non-HIGHEST holder, valid side word, stuck past the window  → stripped
//!   B. holder marked HIGHEST in the side word                      → untouched
//!   C. side word tid does not match the stamp (unknown class)      → untouched
//!   D. non-HIGHEST holder but patience not yet elapsed             → untouched

#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
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
using Linkage = Transactional::Node<MyNode>::Linkage;

static uint64_t strips() {
    return Transactional::detail::g_priv_strips.load();
}

//! A synthetic Reserved stamp \a age_us in the past, held by \a tid.
static NC::cnt_t fake_reserved(uint16_t tid, int64_t age_us) {
    NC::cnt_t st = ((NC::cnt_t)(NC::now_us() - age_us) & NC::STAMP_US_MASK)
        | ((NC::cnt_t)tid << NC::STAMP_TID_SHIFT);
    return NC::with_kind(st, Transactional::detail::StampKind::Reserved);
}

//! Runs one multi-nodal HIGHEST transaction with \a retries forced retries of
//! \a pause_us each — every retry re-tags the linkage, so Rule 0 is evaluated
//! against whatever stamp is planted there.
static void run_highest_tx(const shared_ptr<MyNode> &root,
                           int retries, int pause_us) {
    Transactional::ScopedPriority hp(Transactional::Priority::HIGHEST);
    int n = 0;
    root->iterate_commit_if([&](Tr &tr) -> bool {
        tr[ *root].m_x++;
        if(++n > retries)
            return true;
        if(pause_us)
            std::this_thread::sleep_for(std::chrono::microseconds(pause_us));
        return false;
    });
}

int main() {
    int failures = 0;
    shared_ptr<MyNode> root(MyNode::create<MyNode>());
    shared_ptr<MyNode> child(MyNode::create<MyNode>());
    root->insert(child);    // multi-nodal: operator++ tags root's linkage
    auto &lk = *root->m_link;

    const uint16_t kFakeTid = 0x3ffb;   // nonzero, far from real ids
    // Aged well past every burst/window constant so only Rule 0 (not the
    // age rules) can be the reason anything changes.
    const int64_t kAge = 5'000;
    // Comfortably past KAME_STM_PREEMPT_WINDOW_US (100) in total, in slices
    // small enough that several Rule-0 evaluations land beyond the window.
    const int kRetries = 8, kPauseUs = 40;

    struct Case {
        const char *name;
        uint32_t side_word;      // planted m_priv_owner_prio
        int retries, pause_us;
        bool want_strip;
    } cases[] = {
        { "A stuck non-HIGHEST holder -> stripped",
          (uint32_t)kFakeTid,                               kRetries, kPauseUs, true  },
        { "B holder marked HIGHEST -> untouched",
          (uint32_t)kFakeTid | Linkage::PRIV_OWNER_HIGHEST, kRetries, kPauseUs, false },
        { "C side-word tid mismatch -> untouched",
          (uint32_t)(kFakeTid + 1),                         kRetries, kPauseUs, false },
        { "D patience not elapsed -> untouched",
          (uint32_t)kFakeTid,                               1,        0,        false },
    };
    for(const auto &c : cases) {
        const NC::cnt_t fake = fake_reserved(kFakeTid, kAge);
        lk.m_transaction_started_time.store(fake);
        lk.m_priv_owner_prio.store(c.side_word);
        uint64_t s0 = strips();
        run_highest_tx(root, c.retries, c.pause_us);
        uint64_t ds = strips() - s0;
        NC::cnt_t after = lk.m_transaction_started_time.load();
        bool stripped = (ds != 0);
        // The planted stamp must be gone iff stripped.  (When stripped — or
        // after our own commit's drop_tags — the slot holds our stamp or 0,
        // never the fake.)
        bool fake_gone = (after != fake);
        std::printf("  %-44s strips +%llu, stamp %s\n", c.name,
                    (unsigned long long)ds, fake_gone ? "replaced" : "intact");
        if(stripped != c.want_strip || fake_gone != c.want_strip) {
            std::printf("    FAIL (want %s)\n",
                        c.want_strip ? "stripped" : "untouched");
            ++failures;
        }
        // Clean up: a leftover foreign Reserved stamp would block every
        // NORMAL contender on this linkage via fair-mode, forever (it is not
        // lowprio, so it never expires).
        lk.m_transaction_started_time.store(0);
        lk.m_priv_owner_prio.store(0);
    }

    // The node must still be fully usable at NORMAL after the cleanup.
    root->iterate_commit([&](Tr &tr){ tr[ *root].m_x++; });
    Transactional::Snapshot<MyNode> shot( *root);
    std::printf("  node still commits at NORMAL (m_x=%ld)\n",
                (long)shot[ *root].m_x);

    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
