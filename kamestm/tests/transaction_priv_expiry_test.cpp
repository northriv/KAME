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
//! Pins the privilege-expiry rules on the negotiation predicates directly —
//! the deterministic half of the 2026-07-30 field-crash fix
//! (transaction_priv_pin_test is the behavioural half).
//!
//! The field abort: a NORMAL analysis transaction claimed privilege, could
//! not win (fresh first-attempt commits never negotiate, and a budget-expired
//! peer bypasses fair-mode — privilege cannot stop either), and its Reserved
//! stamp **never expired**, so every thread that did negotiate was pinned
//! until the 3 x 5 s HANG watchdog called abort().  "NORMAL / HIGHEST priv
//! never expires — measurement / driver-critical Tx must not be disrupted"
//! was the design; the field falsified it: what must not happen is a holder
//! that blocks others without a wall-clock bound.
//!
//! New rules, checked here against BOTH agreeing consumers
//! (`fair_mode_blocks_me` and `i_am_privileged_now` — they must never
//! diverge, or per-Linkage stamps go stuck):
//!   * lowprio Reserved: expires after ~51 ms (unchanged);
//!   * NORMAL Reserved: now expires on the same bound — the holder loses its
//!     shield but nothing else (it is NOT thrown at, unlike the revocable
//!     tiers; it just negotiates like everyone again);
//!   * HIGHEST Reserved: exempt, but only when the side word
//!     (Linkage::m_priv_owner_prio) confirms the holder — an unknown or
//!     mismatched holder class expires, because wrongly expiring costs a
//!     healthy holder a few tens of ms of shield while wrongly NOT expiring
//!     reproduces the 15 s abort.
//!
//! White-box (-fno-access-control): privilege claims are probe-gated and
//! cannot be planted deterministically through the public API.

#include "support_standalone.h"
#include "transaction.h"
#include "transaction_impl.h"
#include <cstdio>

class MyNode : public Transactional::Node<MyNode> {
public:
    struct Payload : public Transactional::Node<MyNode>::Payload {
        long m_x = 0;
    };
};
using NC = Transactional::Node<MyNode>::NegotiationCounter;
using Linkage = Transactional::Node<MyNode>::Linkage;

//! A Reserved stamp by \a tid, \a age_us in the past, lowprio bit per \a low.
static NC::cnt_t reserved(uint16_t tid, int64_t age_us, bool low) {
    NC::cnt_t st = ((NC::cnt_t)(NC::now_us() - age_us) & NC::STAMP_US_MASK)
        | ((NC::cnt_t)tid << NC::STAMP_TID_SHIFT);
    if(low) st |= NC::STAMP_LOWPRIO_MASK;
    return NC::with_kind(st, Transactional::detail::StampKind::Reserved);
}

int main() {
    int failures = 0;
    shared_ptr<MyNode> node(MyNode::create<MyNode>());
    auto &lk = *node->m_link;
    const uint16_t kHolder = 0x3ffb, kPeer = 0x3ffc;
    // Ages: far past / far below the ~51 ms bound, so scheduling jitter
    // cannot flip an arm.
    const int64_t kOld = 500'000, kFresh = 1'000;
    // A peer stamp for fair_mode_blocks_me's self-check (tid must differ).
    const NC::cnt_t peer_stamp = reserved(kPeer, kFresh, false)
        ^ (NC::cnt_t)0;   // any non-holder stamp works; kind irrelevant

    struct Case {
        const char *name;
        int64_t age_us;
        bool lowprio;
        uint32_t side_word;
        bool want_blocks;    // fair_mode_blocks_me(peer) — and the holder's
                             // i_am_privileged_now must agree exactly.
    } cases[] = {
        { "fresh NORMAL holder        -> shields", kFresh, false,
          (uint32_t)kHolder,                                true  },
        { "aged NORMAL holder         -> expires", kOld,   false,
          (uint32_t)kHolder,                                false },
        { "aged HIGHEST (side word)   -> shields", kOld,   false,
          (uint32_t)kHolder | Linkage::PRIV_OWNER_HIGHEST,  true  },
        { "aged, unknown holder class -> expires", kOld,   false,
          (uint32_t)(kHolder + 7),                          false },
        { "aged lowprio holder        -> expires", kOld,   true,
          (uint32_t)kHolder,                                false },
        { "fresh lowprio holder       -> shields", kFresh, true,
          (uint32_t)kHolder,                                true  },
    };
    for(const auto &c : cases) {
        const NC::cnt_t stamp = reserved(kHolder, c.age_us, c.lowprio);
        lk.m_transaction_started_time.store(stamp);
        lk.m_priv_owner_prio.store(c.side_word);
        bool blocks = NC::fair_mode_blocks_me(peer_stamp, &lk);
        bool mine   = NC::i_am_privileged_now(
            reserved(kHolder, 0, c.lowprio), &lk);
        std::printf("  %-42s blocks=%d holder_sees=%d (want %d)\n",
                    c.name, (int)blocks, (int)mine, (int)c.want_blocks);
        if(blocks != c.want_blocks || mine != c.want_blocks) {
            std::printf("    FAIL%s\n", (blocks != mine)
                ? " (CONSUMERS DIVERGE - per-Linkage stamps can go stuck)"
                : "");
            ++failures;
        }
        lk.m_transaction_started_time.store(0);
        lk.m_priv_owner_prio.store(0);
    }
    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
