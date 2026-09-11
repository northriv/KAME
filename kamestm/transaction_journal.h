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
#ifndef TRANSACTION_JOURNAL_H_
#define TRANSACTION_JOURNAL_H_

#include "atomic_queue.h"
#include <cstdint>
#include <type_traits>

namespace Transactional {

//! Ordered, bounded capture of committed changes: the mechanism behind a
//! provenance journal, with no knowledge of what a change means.
//!
//! A snapshot of this framework is O(1), which makes something otherwise
//! expensive nearly free: recording what the whole tree looked like at an
//! instant.  Keeping a base snapshot and the ordered stream of what changed
//! after it lets any intermediate state be reconstructed later — not only the
//! endpoints that a saved settings file gives you.
//!
//! What belongs HERE is only what does not depend on meaning: the bounded
//! ring, the accounting of what could not be kept, and the ordering rule.
//! Node names, paths, value formatting, who made the change and the file it
//! all ends up in belong to the application, which is also where the records
//! are captured from — there is deliberately no hook in the commit path.
//! Subscribing a listener is the switch: with nothing subscribed a talker
//! creates no message at all, so a journal that is off costs exactly nothing.
//!
//! \tparam Record whatever the application needs per change, trivially
//!         copyable and self-contained (an index into its own tables, say).
//! \tparam SIZE   ring capacity, a power of two.  Bounded on purpose: a
//!         journal that grows without limit fails a long run at its end,
//!         which is the worst moment.
//!
//! \sa Snapshot, Node
template <class Record, unsigned int SIZE>
class Journal {
public:
    static_assert(std::is_trivially_copyable<Record>::value,
        "Journal records are copied into a claimed slot: nothing may allocate "
        "or run a destructor there.");

    struct Entry {
        //! Serial of the transaction that committed the change.  Carries the
        //! Lamport counter and the committing thread's id, so ordering and
        //! attribution both come from it at no cost.
        int64_t serial;
        Record record;
    };

    //! Offers one change.  Never blocks, never allocates: safe to call from
    //! wherever a commit happens, real-time paths included.
    //!
    //! \return false when the ring was full, in which case the change is NOT
    //!         recorded and the loss is counted.  Losing records is allowed;
    //!         losing them silently is not, which is what takeDropped() is
    //!         for — a journal that cannot say where it is incomplete is
    //!         worse than no journal at all.
    bool capture(int64_t serial, const Record &record) noexcept {
        Entry e;
        e.serial = serial;
        e.record = record;
        if(m_ring.push(e))
            return true;
        m_dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    //! Hands every entry waiting to \a sink, oldest first.  Call from an
    //! ordinary thread: draining is needed to reclaim room, never for the
    //! correctness of capture().
    //!
    //! Entries arrive in publication order, which is the order in which
    //! producers claimed their slots.  That is NOT the transaction order when
    //! several threads commit at once: sort by serial with isOlder() to get
    //! that, and see its note on what "order" can mean here.
    //! \return how many entries were handed over.
    template <class F>
    unsigned int drain(F &&sink) {
        unsigned int n = 0;
        Entry e;
        while(m_ring.pop(e)) {
            sink(const_cast<const Entry &>(e));
            ++n;
        }
        return n;
    }

    //! Number of changes lost since the last call, and clears the count.
    //! Call it around drain() to place a gap where it belongs: the loss lies
    //! between the entries drained before it and those drained after.
    uintptr_t takeDropped() noexcept {
        return m_dropped.exchange(0, std::memory_order_relaxed);
    }
    uintptr_t dropped() const noexcept {
        return m_dropped.load(std::memory_order_relaxed);
    }

    //! Does \a a come before \a b?
    //!
    //! Subtraction reinterpreted as signed, as everywhere else serials are
    //! compared here, so the counter may wrap.  It orders what is causally
    //! ordered; two commits with no causal relation between them get an
    //! arbitrary but consistent order, which is all a Lamport clock offers
    //! and all a journal needs to be replayable.
    static bool isOlder(int64_t a, int64_t b) noexcept {
        return (int64_t)((uint64_t)a - (uint64_t)b) < 0;
    }

    constexpr unsigned int capacity() const noexcept {return SIZE;}
private:
    atomic_bounded_ring<Entry, SIZE> m_ring;
    atomic<uintptr_t> m_dropped {0};
};

} //namespace Transactional

#endif /*TRANSACTION_JOURNAL_H_*/
