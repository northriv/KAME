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
//! Regression test for the debug-only detector that reports `msecsleep()` called
//! while a Transaction is alive — see the hook's doc block in `xtime.h`.
//!
//! Sleeping inside a transaction keeps it open for the whole sleep, so every
//! thread negotiating against it waits; inside an `iterate_commit` closure it
//! re-sleeps on each CAS retry; and it exceeds any `ScopedWaitBudget` outright.
//! `tools/audit/check_stm_closures.py` already flags a literal `msecsleep(` inside
//! a closure, so this exists for what a source scan cannot see: a sleep several
//! call levels below the closure, and a sleep anywhere else in a transaction's
//! lifetime.
//!
//! **Why this test is built with `-UNDEBUG`.** The detector is debug-only, and the
//! test tree is configured Release. Without forcing NDEBUG off for this target the
//! test would compile to nothing and pass while checking nothing — which is what a
//! first attempt at verifying this actually did.
//!
//! **Why it counts through the library instead of substituting its own hook.** The
//! gate that matters — `Transactional::isInTransaction()` — is evaluated *inside*
//! `Transactional::warnIfInTransaction`, not in `msecsleep`, because `xtime`
//! deliberately knows nothing about transactions. A test that installed a counting
//! hook of its own would therefore be testing its own gate. So it reads
//! `detail::s_in_tx_reports`, which the shared reporter bumps at the point it
//! decides to report — the same reporter `XInterface::lock` reaches through
//! `gWarnIfInTransaction`.

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
typedef Transactional::Transaction<MyNode> Tr;

//! One call site, called twice.  A `for(i < 2)` loop around the transaction gets
//! UNROLLED at -O3 into two distinct return addresses, so it measured two reports
//! and looked like a deduplication failure; the detector was right and the test
//! was wrong.  `noinline` pins the site.
__attribute__((noinline))
static void sleep_in_one_transaction(const shared_ptr<MyNode> &node) {
    node->iterate_commit([&](Tr &tr){ tr[ *node].m_x++; msecsleep(1); });
}

static int reports() {
#ifdef NDEBUG
    return 0;
#else
    return Transactional::detail::s_in_tx_reports.load();
#endif
}

int main() {
#ifdef NDEBUG
    std::printf("NDEBUG is defined - the detector is compiled out.\n"
                "This target is supposed to be built with -UNDEBUG; see "
                "kamestm/tests/CMakeLists.txt.\nFAILED\n");
    return 1;
#else
    int failures = 0;
    shared_ptr<MyNode> node(MyNode::create<MyNode>());

    auto expect = [&](const char *what, int before, int want_delta) {
        int got = reports() - before;
        std::printf("  %-46s reports +%d (want +%d)%s\n",
                    what, got, want_delta, got == want_delta ? "" : "   FAIL");
        if(got != want_delta) ++failures;
    };

    // No transaction alive: the reporter must return at its first line.  This is
    // the assertion that pins the isInTransaction() gate — everything else would pass
    // with the gate deleted.
    int b = reports();
    msecsleep(1);
    expect("sleep with no transaction alive", b, 0);

    // Inside a transaction: two distinct call sites, so two reports.  (Same-site
    // repetition is deduplicated; that is the next case.)
    b = reports();
    node->iterate_commit([&](Tr &tr){
        tr[ *node].m_x++;
        msecsleep(1);
        msecsleep(1);
    });
    expect("sleep inside a transaction, two call sites", b, 2);

    // Deduplication by caller address: one site, two transactions, one report.
    b = reports();
    sleep_in_one_transaction(node);
    sleep_in_one_transaction(node);
    expect("same call site in two transactions", b, 1);

    // The suppression kamestm uses for its own two legitimate in-transaction
    // sleeps (the out-of-memory backoff, and lazy TSC calibration).
    b = reports();
    node->iterate_commit([&](Tr &tr){
        tr[ *node].m_x++;
        ScopedSleepInTransactionOK ok;
        msecsleep(1);
    });
    expect("sleep under ScopedSleepInTransactionOK", b, 0);

    // ...and that the suppression is scoped, not sticky.
    b = reports();
    node->iterate_commit([&](Tr &tr){
        tr[ *node].m_x++;
        { ScopedSleepInTransactionOK ok; msecsleep(1); }
        msecsleep(1);
    });
    expect("sleep after the guard leaves scope", b, 1);

    // A Snapshot is not a Transaction: it blocks nothing, so holding one across a
    // sleep is not the defect this detects.  Pinned so that a future change to
    // s_tx_nest's bookkeeping cannot silently widen the detector into noise.
    b = reports();
    {
        Transactional::Snapshot<MyNode> shot( *node);
        (void)shot[ *node].m_x;
        msecsleep(1);
    }
    expect("sleep while only a Snapshot is alive", b, 0);

    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
#endif
}
