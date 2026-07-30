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
//! Pins the re-anchoring invariant of
//! `Transaction::newTransactionUsingSnapshotFor`: across the snapshot-base
//! reassignment, the negotiation bookkeeping — `m_started_time` (the stamp
//! identity), `m_tagged_linkages` (the ledger of stamps planted by
//! `operator++` and by the supernode bundling), `m_registered_privileged` —
//! must survive, or the planted stamps become ownerless ghosts every
//! negotiator waits behind (`drop_tags_n_privilege` clears only what the
//! ledger lists under the current identity).
//!
//! History (2026-07-31): this function was suspected as the field-freeze
//! ghost source (「transactionUsingSnapshotがまずい？」).  Running THIS test
//! against the pre-change code refuted that: the bookkeeping survived by
//! accident — `shot_this` is copy-chained from `*this` via `shot_super`, so
//! the default member-wise `operator=` happened to write the same values
//! back.  The change makes the preservation explicit and this test pins it,
//! so no future edit to either constructor on that copy chain can silently
//! sever the ledger from the planted stamps.  Its one caller is the
//! secondary-driver analysis (`secondarydriverinterface.h`), once per
//! analyzed record.
//!
//! White-box (-fno-access-control), like the other stamp tests: the ghost is
//! asserted directly on the linkage slots.

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
using NC = Transactional::Node<MyNode>::NegotiationCounter;

int main() {
    int failures = 0;
    // supernode(root) -> dev -> leaf : the secondary-driver shape, where the
    // transaction lives on `dev` and re-anchors through `root`.
    shared_ptr<MyNode> root(MyNode::create<MyNode>());
    shared_ptr<MyNode> dev(MyNode::create<MyNode>());
    shared_ptr<MyNode> leaf(MyNode::create<MyNode>());
    root->insert(dev);
    dev->insert(leaf);          // dev is multi-nodal => operator++ tags it

    auto &dev_slot = dev->m_link->m_transaction_started_time;

    {
        Tr tr( *dev);
        tr[ *dev].m_x++;
        const auto id0 = tr.m_started_time;

        // Simulate the field state at the moment of re-anchoring: the ++
        // inside newTransactionUsingSnapshotFor plants dev's tag; here we
        // additionally simulate a probe escalation that upgraded it to
        // Reserved during the supernode bundling.
        Transactional::Snapshot<MyNode> shot_super =
            tr.newTransactionUsingSnapshotFor( *root);
        (void)shot_super;

        const bool id_kept = (tr.m_started_time == id0);
        const bool ledger_kept = !tr.m_tagged_linkages.empty();
        std::printf("  identity preserved across re-anchor : %s\n",
                    id_kept ? "yes" : "NO");
        std::printf("  tag ledger preserved                : %s (%u entries)\n",
                    ledger_kept ? "yes" : "NO",
                    (unsigned)tr.m_tagged_linkages.size());
        if( !id_kept || !ledger_kept) {
            std::printf("    FAIL: Snapshot::operator= wiped the negotiation "
                        "bookkeeping — planted stamps are now ownerless.\n");
            ++failures;
        }

        // Escalation variant: mark privileged and upgrade dev's stamp to
        // Reserved with the (preserved) identity, exactly what the probe does.
        tr.m_registered_privileged = true;
        dev_slot.store(NC::with_kind(tr.m_started_time,
                                     Transactional::detail::StampKind::Reserved));
        tr[ *leaf].m_x++;
        if( !tr.commit()) {
            std::printf("  (commit disturbed — retrying once)\n");
            if( !(++tr).commit()) { std::printf("    FAIL: commit\n"); ++failures; }
        }
    }   // ~Transaction: drop_tags_n_privilege with the preserved ledger

    const auto after = dev_slot.load();
    std::printf("  dev linkage stamp after Tx end      : %s\n",
                after == 0 ? "cleared" : "GHOST LEFT BEHIND");
    if(after != 0) {
        std::printf("    FAIL: an ownerless stamp (kind=%d) survived the "
                    "transaction — every negotiator on this linkage would now "
                    "wait for a ghost (the 2026-07-31 freeze).\n",
                    (int)((after >> NC::STAMP_KIND_SHIFT) & NC::STAMP_KIND_MASK));
        ++failures;
    }

    // The node must remain fully usable.
    dev->iterate_commit([&](Tr &tr){ tr[ *dev].m_x++; });
    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
