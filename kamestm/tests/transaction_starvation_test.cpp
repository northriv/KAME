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
//! Regression test for the starvation bound on revocable priorities —
//! `KAME_STM_LOWPRIO_STARVE_MS` and `Transactional::StarvationTimeoutError`.
//!
//! The rule under test: a priority whose privilege can be REVOKED must have a
//! way to fail.  `stamp_is_expired_lowprio` revokes privilege from LOWEST /
//! UI_DEFERRABLE / SCRIPTING after 51 ms; without a failure path those same
//! priorities can retry forever, and the only existing exit is the negotiation
//! HANG watchdog aborting the whole process after 3 x 5 s.
//!
//! **Why this drives the loop directly instead of manufacturing contention.**
//! Real starvation was tried first and does not make a deterministic test.  It
//! reproduces readily in `transaction_latency_bench -m grand -L 2` (and at -L 4
//! and -L 8, but NOT at -L 1: a lone lowprio thread gets through — lowprio
//! threads starve each *other*, being excluded from the owner-skip lease and the
//! jittered gate).  As a ctest it was flaky in both directions: a two-level tree
//! never starved where the bench's three-level one did, and once the starved
//! peers caught their own exceptions and restarted, the victim stopped starving
//! too.  Contention dynamics are what the bench is for; a regression test should
//! pin the mechanism.
//!
//! `iterate_commit_if` retries unconditionally when the closure returns false,
//! so a single thread can age one transaction past the bound with no contention
//! at all.  That exercises exactly what was added — the check at the loop top,
//! the retry-count gate, and the lowprio bit read off the transaction's own
//! stamp — identically on every run.
//!
//! The negative cases matter as much as the positive one: a bound that reached
//! NORMAL would silently lose driver records.

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

//! Ages one transaction past the bound by skipping the commit, and reports
//! whether StarvationTimeoutError came out.
static bool age_one_transaction(Transactional::Priority pr, int retries) {
    Transactional::ScopedPriority guard(pr);
    shared_ptr<MyNode> node(MyNode::create<MyNode>());
    const auto per_retry = std::chrono::microseconds(
        (KAME_STM_LOWPRIO_STARVE_MS * 1000) / (retries > 1 ? retries - 1 : 1));
    int n = 0;
    try {
        node->iterate_commit_if([&](Tr &tr) -> bool {
            tr[ *node].m_x++;
            if(++n >= retries)
                return true;            // let it commit and end the loop
            std::this_thread::sleep_for(per_retry);
            return false;               // skip: iterate_commit_if retries
        });
    }
    catch (Transactional::StarvationTimeoutError &) {
        return true;
    }
    return false;
}

int main() {
#if KAME_STM_LOWPRIO_STARVE_MS <= 0
    std::printf("KAME_STM_LOWPRIO_STARVE_MS=0 - bound compiled out, skipping\n");
    return 0;
#else
    std::printf("starvation test: bound %d ms, min retries %d\n",
                (int)KAME_STM_LOWPRIO_STARVE_MS,
                (int)KAME_STM_LOWPRIO_STARVE_MIN_RETRIES);
    const int kRetries = (int)KAME_STM_LOWPRIO_STARVE_MIN_RETRIES + 8;
    int failures = 0;

    struct Case { Transactional::Priority pr; const char *name; bool want; };
    static const Case cases[] = {
        { Transactional::Priority::SCRIPTING,     "SCRIPTING",     true  },
        { Transactional::Priority::UI_DEFERRABLE, "UI_DEFERRABLE", true  },
        { Transactional::Priority::LOWEST,        "LOWEST",        true  },
        { Transactional::Priority::NORMAL,        "NORMAL",        false },
        { Transactional::Priority::HIGHEST,       "HIGHEST",       false },
    };
    for(const auto &c : cases) {
        bool got = age_one_transaction(c.pr, kRetries);
        std::printf("  %-14s aged past bound : %-10s (want %s)\n",
                    c.name, got ? "thrown" : "not thrown",
                    c.want ? "thrown" : "not thrown");
        if(got != c.want) {
            std::printf("    FAIL: %s\n", c.want
                ? "a revocable priority retried without an exit - check that "
                  "throw_if_starved_ is reached from iterate_commit_if, and that "
                  "the lowprio bit reaches the stamp "
                  "(lowprio_mask_for_current_priority; KAME_STM_COMPACT_STATE "
                  "seals it)"
                : "the bound reached a priority whose privilege never expires");
            ++failures;
        }
    }

    bool early = age_one_transaction(Transactional::Priority::SCRIPTING, 1);
    std::printf("  %-14s below gate      : %-10s (want not thrown)\n",
                "SCRIPTING", early ? "THROWN" : "not thrown");
    if(early) {
        std::printf("    FAIL: threw before "
                    "KAME_STM_LOWPRIO_STARVE_MIN_RETRIES retries\n");
        ++failures;
    }

    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
#endif
}
