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
#ifndef TRANSACTION_H
#define TRANSACTION_H

#include "support.h"
#include "threadlocal.h"
#include "atomic_smart_ptr.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>
#include "atomic.h"
#include "xtime.h"
#include "transaction_signal.h"

// L1 dcache line size by ABI. Used to prevent false sharing on hot
// per-thread / per-shard counters. ABI-driven, not host-detected:
//   x86_64, ARM Cortex-A: 64
//   Apple Silicon (M1+):  128
//   IBM POWER9/10:        128
//   Fujitsu A64FX (SVE):  256 (sector cache)
// std::hardware_destructive_interference_size is ABI-fragile across
// libc++/libstdc++ versions and emits a warning under GCC, so a fixed
// macro is preferred.
#ifndef KAME_CACHE_LINE
  #if defined(__APPLE__) && defined(__aarch64__)
    #define KAME_CACHE_LINE 128
  #elif defined(__powerpc64__) || defined(__POWERPC__)
    #define KAME_CACHE_LINE 128
  #elif defined(__aarch64__) && (defined(__FUJITSU) || defined(__CLANG_FUJITSU))
    #define KAME_CACHE_LINE 256
  #else
    #define KAME_CACHE_LINE 64
  #endif
#endif

namespace Transactional {

//! \page stmintro Brief introduction of software transactional memory using the class Node
//!  Tree-structure objects, consisting of Node and derived classes, work as
//! software transactional memory (STM) by accessing through Transaction or Snapshot class.\n
//!  STM is a recent trend in a many-processor era for realizing scalable concurrent computing.
//! As opposed to pessimistic mutual exclusions (mutex, semaphore, spin lock, and so on),
//! STM takes an optimistic approach for writing,
//! and each thread does not wait for other threads. Instead, commitment of transactional writing
//! could sometimes fail and the operation will be restarted. The benefits of this optimistic approach are
//! scalability, composability, and freeness of deadlocks.\n
//!  The class Transaction supports transactional accesses of data,
//! which were implemented as Node::Payload or T::Payload in derived classes T, and also handles
//! multiple insertion (hard-link)/removal (unlink) of objects to the tree.
//! Transaction / Snapshot for subtree can be operated at any node at any time by lock-free means.
//! Of course, transactions for separate subtrees do not disturb each other.
//! Since this STM is implemented based on the object model (i.e. not of address/log-based model),
//! accesses can be performed without huge additional costs.
//! Snapshot always holds consistency of the contents of Node::Payload including those for the subnodes,
//! and can be taken typically in O(1) time.
//! During a transaction, unavoidable cost is to copy-on-write Payload of the nodes referenced by
//! Transaction::operator[].\n
//! The smart pointer atomic_shared_ptr, which adds a support for lock-free atomic update on shared_ptr,
//! is a key material in this STM to realize snapshot reading and commitment of a transaction.\n
//! \n
//! Example 1 for snapshot reading: reading two variables in a snapshot.\n
//! \code { Snapshot<NodeA> shot1(node1);
//! 	double x = shot1[node1].m_x;
//! 	double y = shot1[node1].m_y;
//! }
//! \endcode\n
//! Example 2 for simple transactional writing: adding one atomically\n
//! \code node1.iterate_commit([](Transaction<NodeA> &tr1(node1)) {
//! 	tr1[node1].m_x = tr1[node1].m_x + 1;
//! });
//! \endcode\n
//! Example 3 for adding a child node.\n
//! \code node1.iterate_commit_if([](Transaction<NodeA> &tr1(node1)) {
//! 	return tr1.insert(node2));
//! });
//! \endcode \n
//! More real examples are shown in the test codes: transaction_test.cpp,
//! transaction_dynamic_node_test.cpp, transaction_negotiation_test.cpp.\n
//! \sa Node, Snapshot, Transaction.
//! \sa atomic_shared_ptr.

template <class XN>
class Snapshot;
template <class XN>
class Transaction;
template <class XN>
class ScopedNegotiateLinkage;

enum class Priority {NORMAL = 0, LOWEST, UI_DEFERRABLE, HIGHEST};
DECLSPEC_KAME void setCurrentPriorityMode(Priority pr);
DECLSPEC_KAME Priority getCurrentPriorityMode();

namespace detail {
    //! Per-thread nesting depth of Transaction/Snapshot scopes. The
    //! per-thread runner counter (RunnerCounterEntry below) picks up +1
    //! only on the 0 → 1 transition so nested transactions do not
    //! inflate the runner count. Namespace-scope (not template static)
    //! because nesting is a per-thread property and to work around an
    //! Apple clang / arm64 TLS-wrapper-emission bug for template
    //! static `thread_local`.
    inline thread_local int s_tx_nest = 0;
    //! Per-thread nesting depth of ReleaseOneCount (sleeping) scopes.
    inline thread_local int s_sleep_nest = 0;
    using cnt_t = int64_t;
    inline cnt_t pack_stamp(cnt_t us, uint16_t tid) noexcept {
        return (us & ((cnt_t{1} << 48) - 1)) | (cnt_t{tid} << 48);
    }
    inline cnt_t stamp_us(cnt_t x) noexcept {
        // Low 48 bits; steady-clock µs is always positive so a
        // plain mask is correct (no sign-extension needed).
        return x & ((cnt_t{1} << 48) - 1);
    }
    inline uint16_t stamp_tid(cnt_t x) noexcept {
        return (uint16_t)((uint64_t)x >> 48);
    }

    //! Cacheline-padded "I am currently running a Tx" counter, owned
    //! by exactly one thread. Heap-allocated per thread so increments
    //! are TLS-affine (no cacheline ping-pong even across NUMA nodes).
    //! Replaces the previous single
    //! `alignas(64) atomic<unsigned> s_running` whose ping-pong on
    //! every Tx entry/exit was the K=0 disjoint NUMA-scaling ceiling
    //! on 128c EPYC (≈8.3 M Tx/s × 2 atomic RMWs/Tx ≈ 16.6 M ops/s ≈
    //! cross-socket cacheline bandwidth).
    struct alignas(KAME_CACHE_LINE) RunnerCounterEntry {
        std::atomic<uint64_t> v{0};
        char _pad[KAME_CACHE_LINE - sizeof(std::atomic<uint64_t>)];
    };
    using RunnerCounterVec = std::vector<std::weak_ptr<RunnerCounterEntry>>;

    //! TLS holder of this thread's RunnerCounterEntry. The shared_ptr
    //! is the sole strong reference; the global s_runner_counters
    //! vector only holds weak_ptrs that expire on thread exit. The raw
    //! pointer cache below makes the hot path (`AcquireOneCount` ctor)
    //! a single TLS load + relaxed fetch_add — no shared_ptr ref ops.
    inline thread_local std::shared_ptr<RunnerCounterEntry>
        tls_runner_counter_holder;
    inline thread_local RunnerCounterEntry*
        tls_runner_counter_ptr = nullptr;

    //! Global "currently registered runner counters" vector. COW: any
    //! thread's first registration publishes a new vector via
    //! compareAndSwap, pruning expired entries (threads that have
    //! exited) in the same step. Defined in transaction_impl.h.
    DECLSPEC_KAME extern atomic_shared_ptr<RunnerCounterVec>
        s_runner_counters;

    //! Allocate + register this thread's counter on first call;
    //! return the cached raw pointer thereafter. Defined in
    //! transaction_impl.h.
    DECLSPEC_KAME RunnerCounterEntry& runner_counter_register();

    inline RunnerCounterEntry& my_runner_counter() {
        auto *p = tls_runner_counter_ptr;
        if(p) return *p;
        return runner_counter_register();
    }

    //! Sum across all registered threads. Vector traversal is
    //! contiguous-prefetch-friendly. Per entry one
    //! `weak_ptr::lock()` + relaxed load. Called only from
    //! `negotiate_internal` (gate / lottery / wake decisions) — never
    //! on the K=0 disjoint hot path.
    inline unsigned int num_threads_running() noexcept {
        local_shared_ptr<RunnerCounterVec> snap(s_runner_counters);
        if( !snap) return 0;
        uint64_t s = 0;
        for(auto &w : *snap)
            if(auto sp = w.lock())
                s += sp->v.load(std::memory_order_relaxed);
        return (unsigned)s;
    }
} // namespace detail

// Portable 64-bit popcount. Visible in transaction.h so inline member
// functions (e.g. negotiate()) can use it. GCC/Clang/MSVC intrinsics.
static inline int popcount_u64(uint64_t x) noexcept {
#ifdef _MSC_VER
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

//! \brief This is a base class of nodes which carries data sets for itself (Payload) and for subnodes.\n
//! See \ref stmintro for basic ideas of this STM and code examples.
//!
//! \tparam XN a class type used in the smart pointers of NodeList. \a XN must be a derived class of Node<XN> itself.
//! \sa Snapshot, Transaction.
//! \sa XNode.
template <class XN>
class DECLSPEC_KAME Node {
    template <class> friend class ScopedNegotiateLinkage;
public:
    template <class T, typename... Args>
    static T *create(Args&&... args);

    virtual ~Node();

    using NodeNotFoundError = std::domain_error;

    //! Adds a hard link to \a var.
    //! The subnode \a var will be storaged in the list of shared_ptr<XN>, NodeList.
    //! \param[in] online_after_insertion If true, \a var can be accessed through \a tr (not appropriate for a shared object).
    //! \return True if succeeded.
    //! \sa release(), releaseAll(), swap().
    bool insert(Transaction<XN> &tr, const shared_ptr<XN> &var, bool online_after_insertion = false);
    //! Adds a hard link to \a var.
    //! The subnode \a var will be storaged in the list of shared_ptr<XN>, NodeList.
    //! \sa release(), releaseAll(), swap().
    void insert(const shared_ptr<XN> &var);
    //! Removes a hard link to \a var from the list, NodeList.
    //! \return True if succeeded.
    //! \sa insert(), releaseAll(), swap().
    bool release(Transaction<XN> &tr, const shared_ptr<XN> &var);
    //! Removes a hard link to \a var from the list, NodeList.
    //! \sa insert(), releaseAll(), swap().
    void release(const shared_ptr<XN> &var);
    //! Removes all links to the subnodes.
    //! \sa insert(), release(), swap().
    void releaseAll();
    //! Swaps orders in the subnode list.
    //! \return True if succeeded.
    //! \sa insert(), release(), releaseAll().
    bool swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y);
    //! Swaps orders in the subnode list.
    //! \sa insert(), release(), releaseAll().
    void swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y);

    //! Finds the parent node in \a shot.
    XN *upperNode(Snapshot<XN> &shot);

    //! Iterates a transaction covering the node and children.
    //! \param Closure Typical: [=](Transaction<Node1> &tr){ somecode...}
    template <typename Closure>
    Snapshot<XN> iterate_commit(Closure&&);
    //! Iterates a transaction covering the node and children. Skips the iteration when the closure returns false.
    //! \param Closure Typical: [=](Transaction<Node1> &tr){ somecode...; return ret; }
    template <typename Closure>
    Snapshot<XN> iterate_commit_if(Closure&&);
    //! Iterates a transaction covering the node and children, as long as the closure returns true.
    //! \param Closure Typical: [=](Transaction<Node1> &tr){ somecode...; return ret; }
    template <typename Closure>
    void iterate_commit_while(Closure&&);

    //! Data holder and accessor for the node.
    //! Derive Node<XN>::Payload as (\a subclass)::Payload.
    //! The instances have to be capable of copy-construction and be safe to be shared reading.
    struct DECLSPEC_KAME Payload : public atomic_countable {
        Payload() noexcept : m_node(nullptr), m_serial(-1), m_tr(nullptr) {}
        virtual ~Payload() = default;

        //! Points to the corresponding node.
        XN &node() noexcept {return *m_node;}
        //! Points to the corresponding node.
        const XN &node() const noexcept {return *m_node;}
        int64_t serial() const noexcept {return this->m_serial;}
        Transaction<XN> &tr() noexcept { return *this->m_tr;}

        virtual void catchEvent(const shared_ptr<XN>&, int) {}
        virtual void releaseEvent(const shared_ptr<XN>&, int) {}
        virtual void moveEvent(unsigned int /*src_idx*/, unsigned int /*dst_idx*/) {}
        virtual void listChangeEvent() {}
    private:
        friend class Node;
        friend class Transaction<XN>;
        using rettype_clone = local_shared_ptr<Payload>;
        virtual rettype_clone clone(Transaction<XN> &tr, int64_t serial) = 0;

        XN *m_node;
        //! Serial number of the transaction.
        int64_t m_serial;
        Transaction<XN> *m_tr;
    };

    void print_() const;

    using NodeList = fast_vector<shared_ptr<XN>>;
    using iterator = typename NodeList::iterator;
    using const_iterator = typename NodeList::const_iterator;

    Node(const Node &) = delete; //non-copyable.
    Node &operator=(const Node &) = delete; //non-copyable.
private:
    struct Packet;

    struct DECLSPEC_KAME PacketList : public fast_vector<local_shared_ptr<Packet> > {
        shared_ptr<NodeList> m_subnodes;
        PacketList() : fast_vector<local_shared_ptr<Packet>>(), m_serial(SerialGenerator::gen()) {}
        ~PacketList() {this->clear();} //destroys payloads prior to nodes.
        //! Serial number of the transaction.
        int64_t m_serial;
    };

    template <class P>
    struct DECLSPEC_KAME PayloadWrapper : public P::Payload {
        virtual typename PayloadWrapper::rettype_clone clone(Transaction<XN> &tr, int64_t serial) {
//            auto p = allocate_local_shared<PayloadWrapper>(this->m_node->m_allocatorPayload, *this);
            auto p = make_local_shared<PayloadWrapper>( *this);
            p->m_tr = &tr;
            p->m_serial = serial;
            return p;
        }
        PayloadWrapper() = delete;
        PayloadWrapper& operator=(const PayloadWrapper &x) = delete;
        PayloadWrapper(XN &node) noexcept : P::Payload(){ this->m_node = &node;}
        PayloadWrapper(const PayloadWrapper &x) = default;
    private:
    };
    struct PacketWrapper;
    struct Linkage;
    //! A package containing \a Payload, sub-Packets, and a list of subnodes.\n
    //! Not-"missing" packet is always up-to-date and can be a snapshot of the subtree,
    //! and packets possessed by the sub-instances may be out-of-date.\n
    //! "missing" indicates that the package lacks any Packet for subnodes, or
    //! any content may be out-of-date.\n
    struct DECLSPEC_KAME Packet : public atomic_countable {
        Packet() noexcept : m_missing(false) {}
        int size() const noexcept {return subpackets() ? subpackets()->size() : 0;}
        local_shared_ptr<Payload> &payload() noexcept { return m_payload;}
        const local_shared_ptr<Payload> &payload() const noexcept { return m_payload;}
        shared_ptr<NodeList> &subnodes() noexcept { return subpackets()->m_subnodes;}
        shared_ptr<PacketList> &subpackets() noexcept { return m_subpackets;}
        const shared_ptr<NodeList> &subnodes() const noexcept { return subpackets()->m_subnodes;}
        const shared_ptr<PacketList> &subpackets() const noexcept { return m_subpackets;}

        //! Points to the corresponding node.
        Node &node() noexcept {return payload()->node();}
        //! Points to the corresponding node.
        const Node &node() const noexcept {return payload()->node();}

        //! \return false if the packet contains the up-to-date subpackets for all the subnodes.
        bool missing() const noexcept { return m_missing;}

        //! For debugging.
        void print_() const;
        //! For debugging.
        bool checkConsistensy(const local_shared_ptr<Packet> &rootpacket) const;

        local_shared_ptr<Payload> m_payload;
        shared_ptr<PacketList> m_subpackets;
        bool m_missing;
    };

    struct DECLSPEC_KAME NegotiationCounter {
        using cnt_t = detail::cnt_t;
        //! Monotonic µs counter. Uses steady_clock (not wall-clock
        //! gettimeofday) so the µs since program start fit comfortably
        //! in 48 bits — the lower half of the tid-packed stamp type
        //! defined below. Boot-to-now-in-µs is at most a few days on
        //! any realistic machine, well under 2^48 µs (~9 years).
        static cnt_t now_us() noexcept {
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        //! Packed transaction stamp: [ tid : 16 bits | us : 48 bits ].
        //! Raw µs timestamps collide at ~1 MHz (same CPU can issue two
        //! Transactions in the same µs), which breaks the "older wins"
        //! comparison in tag_as_contender() and makes tag-ownership
        //! detection in the livelock probe ambiguous. Packing the
        //! ProcessCounter::id (low 16 bits) into the upper 16 bits makes
        //! every stamp unique per-thread.
        //!
        //! Accessors are in Node<XN>::Linkage (see stamp_us / stamp_tid
        //! / pack_stamp / now_us_tagged below) so negotiate_internal
        //! and the other hot paths go through the same helpers — at
        //! -O2/-O3 the mask is folded into the surrounding instruction
        //! stream at zero extra cost on top of the atomic load itself.
        static inline cnt_t pack_stamp(cnt_t us, uint16_t tid) noexcept {
            return detail::pack_stamp(us, tid);
        }
        static inline cnt_t stamp_us(cnt_t x) noexcept {
            return detail::stamp_us(x);
        }
        static inline uint16_t stamp_tid(cnt_t x) noexcept {
            return detail::stamp_tid(x);
        }
        //! `now_us()` with the current thread's ProcessCounter::id
        //! packed into the upper 16 bits. Used at Transaction /
        //! Snapshot construction to stamp m_started_time.
        static inline cnt_t now_us_tagged() noexcept {
            return pack_stamp(now_us(),
                (uint16_t)(ProcessCounter::id() & 0xFFFFu));
        }

        //=====================================================================
        // Fair-mode escape API. State + accessors are owned by
        // NegotiationCounter; bodies live out-of-class in transaction_impl.h
        // (template member definitions). The thread_local LivelockProbe
        // remains namespace-scope (Transactional::detail) due to an Apple
        // clang / arm64 bug with template static `thread_local`.
        //=====================================================================
        //! Globally registered "privileged TID+stamp" for the fair-mode
        //! escape. Set by `try_register_privileged_tidstamp`, cleared by
        //! `release_privileged_tidstamp`. C++17 inline static so it has
        //! exactly one instance per `Node<XN>` instantiation (KAME only
        //! instantiates `Node<XNode>`, so effectively one global).
        alignas(KAME_CACHE_LINE) static inline std::atomic<cnt_t>
            s_privileged_tidstamp{0};

        static int64_t min_privilege_age_us(Priority pr) noexcept;
        static bool    try_register_privileged_tidstamp(Priority pr,
                                                        cnt_t tidstamp) noexcept;
        static void    release_privileged_tidstamp(cnt_t my_tidstamp) noexcept;
        static bool    fair_mode_blocks_me(cnt_t tidstamp) noexcept;

        //! Per-priority livelock-probe parameters (retry threshold + label).
        struct PriorityProbeInfo {
            int retry_threshold;
            const char *name;
        };
        static PriorityProbeInfo priority_probe_info(Priority pr) noexcept;

        //! Livelock probe tick. Body references `Transactional::detail::
        //! tls_livelock_probe` (TLS, kept namespace-scope for portability).
        //! Returns true iff this tick concluded `verdict=LIVELOCK`.
        static bool livelock_probe_tx_tick(const void *linkage,
                                           uint32_t my_tx_retries,
                                           uint64_t tx_commit_count,
                                           int tags_owned,
                                           int tags_total,
                                           int64_t tx_age_us,
                                           Priority prio) noexcept;

        //=====================================================================
        // Sleep-slot infrastructure for negotiate_sleep / notify_n_contenders.
        //=====================================================================
        //! Cache-line aligned so adjacent slots don't false-share when
        //! two notifier threads on different cores touch neighbouring
        //! indices. KAME_CACHE_LINE matches the ABI L1d line size
        //! (64/128/256 by arch); the previous fixed `alignas(128)` was
        //! correct on x86 and Apple Silicon but oversized for x86 and
        //! undersized for A64FX.
        struct alignas(KAME_CACHE_LINE) NegotiateSleepSlot {
            std::mutex mtx;
            std::condition_variable cv;
            bool notified = false;
        };
        static constexpr int NEGOTIATE_SLEEP_SLOTS = 512;
        static inline NegotiateSleepSlot s_sleep_slots[NEGOTIATE_SLEEP_SLOTS]{};

        static void negotiate_sleep(int ms_timeout) noexcept;

        template<int WORDS>
        static void notify_n_contenders(const uint64_t (&tid_bitset)[WORDS],
                                        int n) noexcept;
        template<int WORDS>
        static void try_notify_n_contenders(const uint64_t (&tid_bitset)[WORDS],
                                            int n) noexcept;

        //! Sum of per-thread "in Tx" counters. See
        //! detail::num_threads_running for the design rationale. Hot
        //! path increments via AcquireOneCount; this read is only used
        //! inside `negotiate_internal`.
        static unsigned int numThreadsRunning() noexcept {
            return detail::num_threads_running();
        }

        //! RAII acquire: bumps this thread's per-thread counter iff
        //! this is the outermost Transaction/Snapshot on the thread
        //! AND we are not inside a ReleaseOneCount (sleeping) scope.
        //! Nested Transactions share the same running-slot — the
        //! thread is one runner regardless of depth.
        struct DECLSPEC_KAME AcquireOneCount {
            AcquireOneCount() {
                if(++detail::s_tx_nest == 1 && detail::s_sleep_nest == 0)
                    detail::my_runner_counter().v
                        .fetch_add(1, std::memory_order_relaxed);
            }
            ~AcquireOneCount() {
                if(--detail::s_tx_nest == 0 && detail::s_sleep_nest == 0)
                    detail::my_runner_counter().v
                        .fetch_sub(1, std::memory_order_relaxed);
            }
        };
        //! RAII yield of the running slot for the duration of a sleep
        //! inside negotiate_internal. Pairs with the TLS nest counters
        //! so nested sleep scopes don't double-decrement.
        struct DECLSPEC_KAME ReleaseOneCount {
            ReleaseOneCount() {
                if(++detail::s_sleep_nest == 1 && detail::s_tx_nest > 0)
                    detail::my_runner_counter().v
                        .fetch_sub(1, std::memory_order_relaxed);
            }
            ~ReleaseOneCount() {
                if(--detail::s_sleep_nest == 0 && detail::s_tx_nest > 0)
                    detail::my_runner_counter().v
                        .fetch_add(1, std::memory_order_relaxed);
            }
        };
    };

    //! Generates a serial number assigned to bundling and transaction.\n
    //! During a transaction, a serial is used for determining whether Payload or PacketList should be cloned.\n
    //! During bundle(), it is used to prevent infinite loops due to unbundle()-ing itself.
    struct DECLSPEC_KAME SerialGenerator {
        enum : int64_t {SERIAL_NULL = 0};
        struct cnt_t {
            cnt_t() noexcept {
                m_var = (int64_t)ProcessCounter::id(); // thread ID in lower 16 bits
            }
            operator int64_t() const noexcept {return m_var;}
            //48bit counter in upper bits + 16bit thread ID in lower 16 bits.
            cnt_t &operator++(int) noexcept {
                m_var = int64_t(uint64_t(m_var) + uint64_t(int64_t(1) << 16));
                return *this;
            }
            int64_t m_var;
        };
        static int64_t current() noexcept {
            return *stl_serial;
        }
        //! Generates a new serial, optionally advancing the counter past \a last_serial.
        //! When \a last_serial is provided, the counter is advanced to at least
        //! last_serial's counter + 1 (Lamport clock), ensuring temporal ordering.
        static int64_t gen(int64_t last_serial = SERIAL_NULL) noexcept {
            auto &v = *stl_serial;
            int64_t last_counter = last_serial & ~int64_t(0xFFFF);
            if(int64_t(uint64_t(last_counter) - uint64_t(v.m_var & ~int64_t(0xFFFF))) > 0)
                v.m_var = last_counter | (v.m_var & int64_t(0xFFFF));
            v++;
            return v;
        }
        static XThreadLocal<cnt_t> stl_serial;
    };
    //! A class wrapping Packet and providing indice and links for lookup.\n
    //! If packet() is absent, a super node should have the up-to-date Packet.\n
    //! If hasPriority() is not set, Packet in a super node may be latest.
    struct DECLSPEC_KAME PacketWrapper : public atomic_countable {
        PacketWrapper(const local_shared_ptr<Packet> &x, int64_t bundle_serial) noexcept;
        //! creates a wrapper not containing a packet but pointing to the upper node.
        //! \param[in] bp \a m_link of the upper node.
        //! \param[in] reverse_index The index for this node in the list of the upper node.
        PacketWrapper(const shared_ptr<Linkage> &bp, int reverse_index, int64_t bundle_serial) noexcept;
        PacketWrapper(const PacketWrapper &x, int64_t bundle_serial) noexcept;
        bool hasPriority() const noexcept { return m_reverse_index == (int)PACKET_STATE::PACKET_HAS_PRIORITY; }
        const local_shared_ptr<Packet> &packet() const noexcept {return m_packet;}
        local_shared_ptr<Packet> &packet() noexcept {return m_packet;}

        //! Points to the upper node that should have the up-to-date Packet when this lacks priority.
        shared_ptr<Linkage> bundledBy() const noexcept {return m_bundledBy.lock();}
        //! The index for this node in the list of the upper node.
        int reverseIndex() const noexcept {return m_reverse_index;}
        void setReverseIndex(int i) noexcept {m_reverse_index = i;}

        void print_() const;
        weak_ptr<Linkage> const m_bundledBy;
        local_shared_ptr<Packet> m_packet;
        int m_reverse_index;
        int64_t m_bundle_serial;
        enum class PACKET_STATE : int { PACKET_HAS_PRIORITY = -1};

        PacketWrapper(const PacketWrapper &) = delete;
    };
    //! Contention-observation bitset size (512 TIDs = 8 * uint64_t).
    //! Used by negotiate() to accumulate distinct ProcessCounter::id values
    //! extracted from PacketWrapper::m_bundle_serial (low 16 bits) during a
    //! transaction's lifetime. Lives inside Transaction; Snapshot-only paths
    //! use a stack-local array.
    static constexpr int TID_BITSET_WORDS = 8;
    using TidBitset = uint64_t[TID_BITSET_WORDS];

#ifndef KAME_LEASE_NS_BASE
#define KAME_LEASE_NS_BASE 10000    // initial 10 µs
#endif

    struct DECLSPEC_KAME Linkage : public atomic_shared_ptr<PacketWrapper> {
        Linkage() noexcept : atomic_shared_ptr<PacketWrapper>(),
            m_transaction_started_time(0),
            m_priority_state(packPriority(0, KAME_LEASE_NS_BASE / 1000, 0)) {}
        ~Linkage() {this->reset(); } //Packet should be freed before memory pools.
        atomic<typename NegotiationCounter::cnt_t> m_transaction_started_time;

        //! Non-atomic Transaction-commit counter — bumped in
        //! `Transaction<XN>::finalizeCommitment()` (only the
        //! iterate_commit winner executes that, one writer per tx, no
        //! concurrent writes on a given Linkage, so the plain ++ is
        //! race-free). Counts **successful Transactions** rooted at
        //! this linkage (not individual CAS ops inside a single tx).
        //! Readers — the optional livelock probe — load via the same
        //! atomic_shared_ptr CAS acquire they already used to reach
        //! this Linkage.
        mutable uint64_t m_tx_commit_count = 0;

        // Implicit commit-lease. within a certain window after the
        // current wrapper was installed, the committing TID holds a soft lease —
        // a subsequent negotiate() call from the same TID skips the msec-sleep
        // path so it can chain a follow-up commit attempt immediately. Lease
        // auto-expires by wall-clock; no explicit release. Override via
        // -DKAMEE_PRIORITY_LEASE_DISABLE
#ifndef KAME_PRIORITY_LEASE_DISABLE
#define KAME_PRIORITY_LEASE
#endif
        //! Packed per-Linkage priority/lease state. One 64-bit atomic holds
        //! three fields that are always read together on the fast path:
        //!
        //!   bits  0..15 :  tid       — last committer TID (lower 16 bits of
        //!                              ProcessCounter::id()).
        //!   bits 16..31 :  lease_us  — adaptive lease window in µs,
        //!                              clamped to [KAME_LEASE_NS_MIN/1000,
        //!                              KAME_LEASE_NS_MAX/1000] (default
        //!                              1..500 µs, fits in 16 bits).
        //!   bits 32..63 :  start_us  — install time (µs since counter
        //!                              epoch). 32 bits wraps at ~71 min;
        //!                              age diffs use unsigned modular
        //!                              subtraction so wrap is transparent
        //!                              for windows < 35 min (our lease is
        //!                              ≤ 500 µs).
        //!
        //! Packing keeps the three fields on one cache line and makes the
        //! negotiate() fast path a single atomic load + shifts instead of
        //! three separate atomic loads (one per former field). Writes are
        //! relaxed-store or CAS via the pack/unpack helpers below.
        atomic<uint64_t> m_priority_state;

        struct PriorityState {
            uint16_t tid;
            uint16_t lease_us;
            uint32_t start_us;
        };
        static inline uint64_t packPriority(uint16_t tid, uint16_t lease_us,
                                            uint32_t start_us) noexcept {
            return ((uint64_t)start_us << 32)
                 | ((uint64_t)lease_us << 16)
                 | (uint64_t)tid;
        }
        static inline PriorityState unpackPriority(uint64_t raw) noexcept {
            return PriorityState{
                (uint16_t)(raw & 0xFFFFu),
                (uint16_t)((raw >> 16) & 0xFFFFu),
                (uint32_t)(raw >> 32)
            };
        }
        //! Fast-path load of the packed priority/lease state. Relaxed by
        //! default: m_priority_state is self-contained (consumers use the
        //! unpacked fields directly and don't infer order on any other
        //! memory from this load), and on every caller's hot path a
        //! wrapper-CAS (acq_rel) has already executed, so the thread
        //! naturally observes post-CAS state without needing another
        //! acquire fence. Callers that specifically need acquire can pass
        //! std::memory_order_acquire.
        PriorityState loadPriority(std::memory_order mo = std::memory_order_relaxed)
            const noexcept {
            return unpackPriority(m_priority_state.load(mo));
        }
        //! Store a new packed state (relaxed; lease adaptation is advisory).
        void storePriority(PriorityState ps,
                           std::memory_order mo = std::memory_order_relaxed) noexcept {
            m_priority_state.store(packPriority(ps.tid, ps.lease_us, ps.start_us), mo);
        }
        //! CAS on the packed state. Succeeds only if the whole 64-bit word
        //! hasn't changed since \p expected was read. On failure, refreshes
        //! \p expected to the current observed value.
        bool casPriority(PriorityState &expected, PriorityState desired) noexcept {
            uint64_t e = packPriority(expected.tid, expected.lease_us, expected.start_us);
            uint64_t d = packPriority(desired.tid,  desired.lease_us,  desired.start_us);
            if(m_priority_state.compare_set_strong(e, d)) return true;
            expected = unpackPriority(m_priority_state.load(std::memory_order_acquire));
            return false;
        }

        //! Puts a wait so that the slowest thread gains a chance to finish its transaction, if needed.
        //! \a tid_bitset accumulates observed committer TIDs (via the Linkage's
        //! m_priority_tid atomic) over the lifetime of the calling transaction.
        //! negotiate_internal() uses its popcount C to scale the backoff jitter
        //! range as √C; negotiate() uses C to drive the adaptive-lease length.
        void negotiate(Snapshot<XN> &snap,
                       float mult_wait) noexcept {
#if defined(KAME_STM_DISABLE_BACKOFF) && KAME_STM_DISABLE_BACKOFF
            (void)snap; (void)mult_wait;
            return;
#else
            // Fast path for the common "no collision" case. An extra
            // atomic_shared_ptr load plus an out-of-line call inside
            // negotiate_internal is measurable in low-contention workloads,
            // so we short-circuit when the collision marker is clear. The
            // relaxed load here is the same one negotiate_internal would
            // do first, so we pay no extra in the collision case either.
            if( !m_transaction_started_time.load(std::memory_order_relaxed)) [[likely]]
                return;
            negotiate_internal(snap, mult_wait);
#endif
        }
        void tags_successful_cas(typename NegotiationCounter::cnt_t started_time = {}) noexcept {
#if defined(KAME_STM_DISABLE_BACKOFF) && KAME_STM_DISABLE_BACKOFF
            (void)started_time;
            return;
#else
            auto ps = loadPriority();
            // Build the desired tuple. Keep ps.start_us when the owner TID
            // hasn't changed — a same-owner re-commit doesn't count as a
            // fresh lease start, so we avoid differing start_us values
            // making `desired != ps` spuriously true on every call. lease_us
            // is always preserved (owner change doesn't reset the Linkage's
            // contention profile).
            uint16_t my_tid = (uint16_t)(ProcessCounter::id() & 0xFFFFu);
            PriorityState desired{ my_tid, ps.lease_us, ps.start_us };
            if(my_tid != ps.tid) {
                // started_time is a tid-packed stamp; unpack the µs
                // component before feeding into the 32-bit µs field.
                desired.start_us = started_time
                    ? Node<XN>::NegotiationCounter::stamp_us(started_time)
                    : Node<XN>::NegotiationCounter::now_us();
            }
            // Only store when something actually changed. The 64-bit
            // packed compare is one instruction; skipping the store on
            // identical state avoids cache-line ping-pong on same-owner
            // recommits.
            //
            // Relaxed order suffices: m_priority_state is self-contained
            // (consumers use ps.tid / ps.lease_us / ps.start_us on their
            // own; they don't infer ordering on any other memory from this
            // store). The companion wrapper CAS on m_link carries its own
            // release semantics and the wrapper's TID lives in the serial
            // low bits, so no cross-atomic happens-before is needed here.
            if(packPriority(desired.tid, desired.lease_us, desired.start_us)
               != packPriority(ps.tid, ps.lease_us, ps.start_us)) {
                storePriority(desired, std::memory_order_relaxed);
            }
#endif
        }
        //! Unified retry-loop backoff helper. Called once per retry (retry > 0)
        //! at the top of each CAS-retry loop (commit/snapshot/bundle/bundle-child).
        //! Always issues retry_pause (CPU-side pause spins, count ∝ retry);
        //! at retry ≥ NEGOTIATE_THRESHOLD also engages sleep-based negotiate()
        //! so the contender gets priority escalation that pure spin cannot give.
        //! GDB-sampling on 32-thread 3level stress showed ~52% of time in
        //! bundle()'s outer retry, which previously had only retry_pause and
        //! no negotiate between Phase 3 failure and the next Phase 1 attempt.
        void negotiate_after_retry_pause(int retry,
                                         Snapshot<XN> &snap,
                                         float mult_wait) noexcept;
        void negotiate_internal(Snapshot<XN> &snap,
                                float mult_wait) noexcept;

    };

    friend class Snapshot<XN>;
    friend class Transaction<XN>;

    void snapshot(Snapshot<XN> &target, bool multi_nodal) const;
    void snapshot(Transaction<XN> &target, bool multi_nodal) const {
        m_link->negotiate(target, 4.0f);
        snapshot(static_cast<Snapshot<XN> &>(target), multi_nodal);
        target.m_oldpacket = target.m_packet;
    }
    enum class SnapshotStatus {SUCCESS = 0, DISTURBED = 1,
        VOID_PACKET = 2, NODE_MISSING = 4,
        COLLIDED = 8, NODE_MISSING_AND_COLLIDED = 12};
    struct CASInfo {
        CASInfo(const shared_ptr<Linkage> &b, const local_shared_ptr<PacketWrapper> &o,
            const local_shared_ptr<PacketWrapper> &n) : linkage(b), old_wrapper(o), new_wrapper(n) {}
        shared_ptr<Linkage> linkage;
        local_shared_ptr<PacketWrapper> old_wrapper, new_wrapper;
    };
    using CASInfoList = fast_vector<CASInfo, 32>;

    //! Result of walkUpChainImpl() / ascendOneLevel().
    //! ascendOneLevel fills parent_linkage, parent_wrapper, child_wrapper, reverse_index.
    //! walkUpChainImpl adds find_status, status, is_root_level, parent_packet.
    struct WalkUpResult {
        SnapshotStatus find_status;  //!< result of findChildSlot (or early-return status)
        SnapshotStatus status;       //!< status after convertRecursiveStatus (before find)
        bool is_root_level;          //!< true if this parent is the chain root
        shared_ptr<Linkage> parent_linkage;    //!< m_link of the parent node (= bundledBy)
        local_shared_ptr<PacketWrapper> parent_wrapper;  //!< snapshot of parent's wrapper (= original shot_upper)
        local_shared_ptr<PacketWrapper> child_wrapper;   //!< wrapper saved before ascending (for staleness check)
        int reverse_index;
        local_shared_ptr<Packet> *parent_packet;  //!< parent's packet containing child slot
    };

    //! Ascend one level: read bundledBy, update root_wrapper to parent, save snapshot.
    //! On success, find_status == SUCCESS and parent/child fields are filled.
    //! On failure, find_status == DISTURBED or NODE_MISSING.
    static inline WalkUpResult ascendOneLevel(
        const shared_ptr<Linkage> &child_linkage,
        local_shared_ptr<PacketWrapper> &root_wrapper);

    //! Convert recursive status and determine the upper packet.
    //! Sets is_root_level = true if this parent level is the root.
    static inline SnapshotStatus convertRecursiveStatus(
        SnapshotStatus recursive_status,
        WalkUpResult &r,
        local_shared_ptr<PacketWrapper> &root_wrapper,
        local_shared_ptr<Packet> *&parent_packet);

    //! Find child's sub-packet slot in parent's packet by scanning from reverse_index hint.
    static inline SnapshotStatus findChildSlot(
        const shared_ptr<Linkage> &child_linkage,
        local_shared_ptr<Packet> *parent_packet,
        local_shared_ptr<Packet> **child_subpacket_out,
        int &reverse_index,
        SnapshotStatus current_status);

    //! Common chain-walk: Steps A→B→C→D→E (ascend, recurse, convert, staleness, findChildSlot).
    //! Recurser performs the recursive call at Step B.
    template <class Recurser>
    static inline WalkUpResult walkUpChainImpl(
        const shared_ptr<Linkage> &child_linkage,
        local_shared_ptr<PacketWrapper> &root_wrapper,
        local_shared_ptr<Packet> **child_subpacket_out,
        Recurser &&recurse);

    //! Recursively walks up the bundledBy chain to locate a child's sub-packet.
    //! Used by snapshot() (FOR_BUNDLE path).
    static inline SnapshotStatus walkUpChain(
        const shared_ptr<Linkage> &child_linkage,
        local_shared_ptr<PacketWrapper> &root_wrapper,
        local_shared_ptr<Packet> **child_subpacket_out);

    //! Walk up the chain and build CAS info list for unbundling.
    //! Used only by unbundle().
    static inline SnapshotStatus snapshotForUnbundle(
        const shared_ptr<Linkage> &child_linkage,
        local_shared_ptr<PacketWrapper> &root_wrapper,
        local_shared_ptr<Packet> **child_subpacket_out,
        int64_t serial, CASInfoList *cas_infos);

    //! Updates a packet to \a tr.m_packet if the current packet is unchanged (== \a tr.m_oldpacket).
    //! If this node has been bundled at the super node, unbundle() will be called.
    //! \sa Transaction<XN>::commit().
    bool commit(Transaction<XN> &tr);
//	bool commit_at_super(Transaction<XN> &tr);

    enum class BundledStatus {SUCCESS, DISTURBED};
    //! Bundles all the subpackets so that the whole packet can be treated atomically.
    //! Namely this function takes a snapshot.
    //! All the subpackets held by \a m_link at the subnodes will be
    //! cleared and each PacketWrapper::bundledBy() will point to its upper node.
    //! \sa snapshot().
    BundledStatus bundle(local_shared_ptr<PacketWrapper> &target,
        Snapshot<XN> &snap,
        int64_t bundle_serial, bool is_bundle_root);
    BundledStatus bundle_subpacket(local_shared_ptr<PacketWrapper> *superwrapper, const shared_ptr<Node> &subnode,
        local_shared_ptr<PacketWrapper> &subwrapper, local_shared_ptr<Packet> &subpacket_new,
        Snapshot<XN> &snap,
        int64_t bundle_serial);
    enum class UnbundledStatus {W_NEW_SUBVALUE, SUBVALUE_HAS_CHANGED, COLLIDED, DISTURBED};
    //! Unbundles a subpacket to \a sublinkage from a snapshot.
    //! it performs unbundling for all the super nodes.
    //! The super nodes will lose priorities against their lower nodes.
    //! \param[in] bundle_serial If not zero, consistency/collision wil be checked.
    //! \param[in] bundled_ref The current value of \a sublinkage and should not contain \a packet().
    //! \param[in] oldsubpacket If not zero, the packet will be compared with the packet inside the super packet.
    //! \param[in,out] newsubwrapper If \a oldsubpacket and \a newsubwrapper are not zero, \a newsubwrapper will be a new value.
    //! If \a oldsubpacket is zero, unloaded value  of \a sublinkage will be substituted to \a newsubwrapper.
    //! \param[in] snap Snapshot that chains started_time, tid_bitset and
    //! the tagged-linkage list through the commit/negotiate/bundle path.
    static UnbundledStatus unbundle(const int64_t *bundle_serial, Snapshot<XN> &snap,
        const shared_ptr<Linkage> &sublinkage, const local_shared_ptr<PacketWrapper> &bundled_ref,
        const local_shared_ptr<Packet> *oldsubpacket = NULL,
        local_shared_ptr<PacketWrapper> *newsubwrapper = NULL,
        local_shared_ptr<PacketWrapper> *superwrapper = NULL);
    //! The point where the packet is held.
    shared_ptr<Linkage> m_link;

    //! finds the packet for this node in the (un)bundled \a packet.
    //! \param[in,out] superpacket The bundled packet.
    //! \param[in] copy_branch If ture, new packets and packet lists will be copy-created for writing.
    //! \param[in] tr_serial The serial number associated with the transaction.
    inline local_shared_ptr<Packet> *reverseLookup(local_shared_ptr<Packet> &superpacket,
        bool copy_branch, int64_t tr_serial, bool set_missing, XN** uppernode);
    local_shared_ptr<Packet> &reverseLookup(local_shared_ptr<Packet> &superpacket,
        bool copy_branch, int64_t tr_serial = 0, bool set_missing = false);
    const local_shared_ptr<Packet> &reverseLookup(const local_shared_ptr<Packet> &superpacket) const;
    inline static local_shared_ptr<Packet> *reverseLookupWithHint(shared_ptr<Linkage > &linkage,
        local_shared_ptr<Packet> &superpacket, bool copy_branch, int64_t tr_serial, bool set_missing,
        local_shared_ptr<Packet> *upperpacket, int *index);
    //! finds this node and a corresponding packet in the (un)bundled \a packet.
    inline local_shared_ptr<Packet> *forwardLookup(local_shared_ptr<Packet> &superpacket,
        bool copy_branch, int64_t tr_serial, bool set_missing,
        local_shared_ptr<Packet> *upperpacket, int *index) const;
//	static void fetchSubpackets(std::deque<local_shared_ptr<PacketWrapper> >  &subwrappers,
//		const local_shared_ptr<Packet> &packet);
    static void eraseSerials(local_shared_ptr<Packet> &packet, int64_t serial,
                             Snapshot<XN> &snap);
protected:
    //! Use \a create().
    Node();
private:
    using FuncPayloadCreator = Payload *(*)(XN &);
    static XThreadLocal<FuncPayloadCreator> stl_funcPayloadCreator;
    void lookupFailure() const;
    local_shared_ptr<typename Node<XN>::Packet>*lookupFromChild(local_shared_ptr<Packet> &superpacket,
        bool copy_branch, int64_t tr_serial, bool set_missing, XN **uppernode);
    static void print_recoverable_error(const char*);
};

template <class XN>
template <class T, typename... Args>
T *Node<XN>::create(Args&&... args) {
    *T::stl_funcPayloadCreator = [](XN &node)->Payload *{ return new PayloadWrapper<T>(node);};
    return new T(std::forward<Args>(args)...);
}

//! \brief This class takes a snapshot for a subtree.\n
//! See \ref stmintro for basic ideas of this STM and code examples.
//! \sa Node, Transaction, SingleSnapshot, SingleTransaction.
template <class XN>
class DECLSPEC_KAME Snapshot {
    template <class> friend class ScopedNegotiateLinkage;
public:
    // Defaulted copy/move handle all members including the C-array
    // m_tid_bitset (implicit memcpy). Transaction<XN>-accepting ctors
    // delegate to the base copy/move path via a static_cast to the
    // Snapshot subobject so they pick up the same defaulted logic
    // without hand-written std::copy boilerplate.
    Snapshot(const Snapshot&x) noexcept = default;
    Snapshot(Snapshot&&x) noexcept = default;
    Snapshot& operator=(const Snapshot&x) noexcept = default;
    Snapshot(Node<XN> &node, const Snapshot &x) noexcept : Snapshot(x) {
        // View the same snapshot through \a node: copy all base fields,
        // then redirect m_packet to the node-local view. m_serial /
        // m_started_time / m_tid_bitset stay identical to x.
        m_packet = node.reverseLookup(x.m_packet);
    }
    explicit Snapshot(const Transaction<XN>&x) noexcept
        : Snapshot(static_cast<const Snapshot&>(x)) {}
    explicit Snapshot(Transaction<XN>&&x) noexcept
        : Snapshot(static_cast<Snapshot&&>(x)) {}
    explicit Snapshot(const Node<XN>&node, bool multi_nodal = true) {
        m_started_time = Node<XN>::NegotiationCounter::now_us_tagged();
        typename Node<XN>::NegotiationCounter::AcquireOneCount oneup{};
        node.snapshot( *this, multi_nodal);
        drop_tags_n_privilege();
    }

    //! \return Payload instance for \a node, which should be included in the snapshot.
    template <class T>
    const typename T::Payload &operator[](const shared_ptr<T> &node) const noexcept {
        return operator[](const_cast<const T&>( *node));
    }
    //! \return Payload instance for \a node, which should be included in the snapshot.
    template <class T>
    const typename T::Payload &operator[](const T &node) const noexcept {
        return this->at(node);
    }
    //! may raise NodeNotFoundError;
    template <class T>
    const typename T::Payload &at(const T &node) const {
        const local_shared_ptr<typename Node<XN>::Packet> &packet(node.reverseLookup(m_packet));
        const local_shared_ptr<typename Node<XN>::Payload> &payload(packet->payload());
        return *static_cast<const typename T::Payload*>(payload.get());
    }
    //! # of child nodes.
    int size() const noexcept {return m_packet->size();}
    //! The list of child nodes.
    const shared_ptr<const typename Node<XN>::NodeList> list() const noexcept {
        if( !size())
            return shared_ptr<typename Node<XN>::NodeList>();
        return m_packet->subnodes();
    }
    //! # of child nodes owned by \a node.
    int size(const shared_ptr<Node<XN> > &node) const noexcept {
        return node->reverseLookup(m_packet)->size();
    }
    //! The list of child nodes owned by \a node.
    shared_ptr<const typename Node<XN>::NodeList> list(const shared_ptr<Node<XN> > &node) const noexcept {
        auto const &packet(node->reverseLookup(m_packet));
        if( !packet->size() )
            return shared_ptr<typename Node<XN>::NodeList>();
        return packet->subnodes();
    }
    //! Whether \a lower is a child of this or not.
    bool isUpperOf(const XN &lower) const noexcept {
        const shared_ptr<const typename Node<XN>::NodeList> lx(list());
        if( !lx)
            return false;
        for(auto &&x: *lx) {
            if(x.get() == &lower)
                return true;
        }
        return false;
    }

    void print() {
        m_packet->print_();
    }

    //! Stores an event immediately from \a talker with \a arg.
    template <typename T, typename...Args>
    void talk(T &talker, Args&&...arg) const { talker.talk( *this, std::forward<Args>(arg)...); }
    //! Returns true if this snapshot is older than \a other.
    //! Uses unsigned subtraction reinterpreted as signed to handle counter wrap-around,
    //! correct as long as the true counter difference is much less than 2^47.
    bool isOlderThan(const Snapshot &other) const noexcept {
        return int64_t(uint64_t(m_serial) - uint64_t(other.m_serial)) < 0;
    }
    bool operator==(const Snapshot &other) const noexcept { return m_serial == other.m_serial; }

    //! Register \a link as a contender for this snapshot's started_time.
    //! Overwrites the linkage's m_transaction_started_time with ours iff
    //! the linkage is currently untagged, or tagged by a younger thread
    //! ("oldest-wins"). Also pushes the linkage onto m_tagged_linkages so
    //! subsequent passes can walk a single list for cleanup.
    //!
    //! Semantics match the inline block that pre-RAII code had inside
    //! Transaction::operator++; this is just the extraction into a
    //! Snapshot-level helper so snapshot()/bundle() can adopt it too in
    //! later refactor passes.
    void tag_as_contender(const std::shared_ptr<typename Node<XN>::Linkage> &link) noexcept {
        // CAS-loop variant (Option A). Atomically claim the linkage's
        // priority slot iff the slot is empty OR the current tagger is
        // YOUNGER than us (compare on stamp_us only — the tid packed in
        // the upper 16 bits would otherwise dominate). Without the CAS,
        // a non-atomic load-compare-store can let a younger Tx overwrite
        // an older Tx's stamp (last-writer-wins races), which then leads
        // the younger Tx's livelock probe to incorrectly see
        // tags_owned == tags_total and claim the privileged slot
        // ("exception" case observed in N=128 0-commits stalls; see
        // [ll-debug] dump in ChangeLog / paper §X.).
        using NC = typename Node<XN>::NegotiationCounter;
        auto &slot = link->m_transaction_started_time;
        // ----- Option B (relaxed store + acquire verify) -----
        // Linkage stamp may transiently reflect a younger Tx (eventually
        // corrected by other Txs' retries); the probe-side false-positive
        // is still avoided because this Tx simply doesn't track the
        // linkage in its list when its store didn't survive.
        const auto my_us = NC::stamp_us(m_started_time);
        auto cur = slot.load(std::memory_order_relaxed);
        if(!cur || NC::stamp_us(cur) > my_us) {
            slot.store(m_started_time, std::memory_order_release);
            if(slot.load(std::memory_order_acquire) != m_started_time) [[unlikely]]
                return;  // overwritten — don't add to list
        }

        // (Option A — CAS-loop variant — is preserved in git history;
        //  bench at 100ms preempt floor showed Option B beats A by
        //  +16% median / +20% mean; A is removed from this hot path.)

        // Dedup: ++tr re-tags the same primary-node linkage on every
        // retry, which otherwise piles duplicate shared_ptr entries
        // onto m_tagged_linkages.
        for(auto &&l: m_tagged_linkages)
            if(l.get() == link.get())
                return; //duplicated.
        m_tagged_linkages.push_back(link);
    }

    //! Walks m_tagged_linkages and clears each linkage's tag only if it
    //! still equals m_started_time ("mine-only" compare-and-clear). Idempotent
    //! w.r.t. duplicate entries — repeated tags from the same snapshot across
    //! retries land in the list multiple times; the second pass sees tag=0
    //! and skips. Runs from ~Transaction(),
    //! ~SingleTransaction() or else gets cleanup for
    //! free.
    //!
    //! A stale tag (value != m_started_time, e.g. somebody younger
    //! overwrote ours) is left alone — the other thread will clear it
    //! from its own tranaction destructor.
    void drop_tags_n_privilege() noexcept {
        for(auto &sp : m_tagged_linkages) {
            if(sp->m_transaction_started_time == m_started_time)
                sp->m_transaction_started_time = 0;
        }
        // If we held the fair-mode privilege, release it on commit so the
        // global slot is available for the next stuck Tx. Reset the local
        // flag too, otherwise ~Transaction() would attempt a redundant
        // (and now stale) clear.
        if (this->m_registered_privileged) {
            Node<XN>::NegotiationCounter::release_privileged_tidstamp(this->m_started_time);
            this->m_registered_privileged = false;
        }
    }

protected:
    friend class Node<XN>;
    //! The snapshot.
    local_shared_ptr<typename Node<XN>::Packet> m_packet;
    int64_t m_serial;
    //! When this snapshot attempt started. Negotiation / backoff uses this
    //! (vs. the peer linkage's m_transaction_started_time) to decide who is
    //! "older" and therefore earns priority to commit first.
    //!
    //! Promoted from Transaction<XN> so snapshot() tree-walk paths and
    //! negotiate() helpers can be rewritten to take a single Snapshot&
    //! parameter instead of threading a separate started_time argument.
    //!
    //! Set once per object by either the standalone-Snapshot ctor
    //! (Snapshot(const Node&)) or the Transaction ctor — never reset by
    //! retries. The µs component (via NegotiationCounter::stamp_us)
    //! is what the livelock probe reports as `tx_age_us`; that label
    //! covers both Snapshot age and Transaction age depending on which
    //! ctor stamped this field.
    typename Node<XN>::NegotiationCounter::cnt_t m_started_time = {};
    //! Per-attempt TID observation bitset for the adaptive-lease path.
    //! Also promoted from Transaction<XN>. The standalone-Snapshot ctor
    //! fills this in tree-walk; Transaction<XN> inherits and reuses.
    typename Node<XN>::TidBitset m_tid_bitset = {};
    //! Retry count consumed by the livelock probe.
    //!
    //! Bumped from **two** distinct sites:
    //!   1. `Transaction::operator++()` — once per outer iterate_commit retry
    //!      (Tx-layer retry; covers Tx that never bundle).
    //!   2. `Node::snapshot()` retry loop — once per pure-Snapshot bundle retry
    //!      (Snapshot/bundle-layer retry; can fire without any Tx).
    //!
    //! The `m_tx_` prefix is therefore historical (predates the snapshot-side
    //! increment): the field aggregates both Tx-retry and bundle-retry counts.
    //! The name is retained for log-format / ABI continuity (the probe prints
    //! it as `my_tx_retries=…`); no single short prefix captures both sources
    //! without lying at one site or the other.
    //!
    //! Stays 0 for a pure Snapshot that needs no retry, and for a fresh
    //! Transaction (default-initialized). Consumed by the optional livelock
    //! probe to separate "CAS retries piling up (normal under contention)"
    //! from "Tx/bundle retries piling up while the linkage's CAS commits
    //! continue" — livelock at the Tx-or-bundle layer.
    uint32_t m_tx_retry_count = 0;
    //! Per-Tx ownership flag for the global `s_privileged_tidstamp` slot.
    //! Set true in negotiate_internal's probe path when THIS Transaction
    //! successfully CAS-claimed the slot (i.e. won the 0→my_tid race).
    //! Cleared by the registering Tx itself in finalizeCommitment
    //! (success path) and Transaction::~Transaction() (abort path).
    //! Required for nesting safety: an inner Tx on the same thread sees
    //! s_privileged_tidstamp == my_tid + stamp and does not CAS again, so its
    //! m_registered_privileged stays false and its dtor will not steal
    //! the outer's privilege.
    bool m_registered_privileged = false;
    //! Linkages whose m_transaction_started_time this attempt has tagged
    //! (or intends to tag). Held as shared_ptr to keep the Linkage alive
    //! until clear_tags() runs; otherwise dynamic-node release could leave
    //! a dangling reference in the cleanup loop.
    //!
    //! Unused in the first Snapshot-base refactor — the field only
    //! provides the storage scaffold so a later pass can migrate
    //! eager-tag / eager-clear logic out of tr++/Negotiator RAII without
    //! further restructuring. Inline-first-16 (fast_vector) keeps
    //! low-contention workloads zero-alloc.
    fast_vector<std::shared_ptr<typename Node<XN>::Linkage>, 16> m_tagged_linkages;

    Snapshot() = default;
};
//! \brief Snapshot class which does not care of contents (Payload) for subnodes.\n
//! See \ref stmintro for basic ideas of this STM and code examples.
//! \sa Node, Snapshot, Transaction, SingleTransaction.
template <class XN, typename T>
class DECLSPEC_KAME SingleSnapshot : protected Snapshot<XN> {
public:
    explicit SingleSnapshot(const T &node) : Snapshot<XN>(node, false) {}
    SingleSnapshot(SingleSnapshot&&x) noexcept = default;

    //! \return a pointer to Payload for \a node.
    const typename T::Payload *operator->() const {
        return static_cast<const typename T::Payload *>(this->m_packet->payload().get());
    }
    //! \return reference to Payload for \a node.
    const typename T::Payload &operator*() const {
        return *operator->();
    }
protected:
};

//! Assert that the given Snapshot/Transaction is NOT currently the
//! fair-mode privileged Tx. Use at any CAS-fail / loop-fail site to
//! catch livelock-free invariant violations: a privileged Tx must
//! make forward progress, so failing a CAS or re-iterating a spin
//! loop while privileged means some other thread bypassed the
//! fair-mode yield (= a bug in the negotiate / tag_as_contender
//! coverage). Default 0 (production); enable with
//! `-DKAME_STM_ASSERT_PRIVILEGE=1` for debug builds.
#ifndef KAME_STM_ASSERT_PRIVILEGE
#define KAME_STM_ASSERT_PRIVILEGE 0
#endif

// ScopedNegotiateLinkage<XN> definition lives in transaction_impl.h
// (only used by impl-side retry loops). Forward-declared near the top
// of this file; friend-declared in Node<XN> and Snapshot<XN>.

//! \brief A class supporting transactional writing for a subtree.\n
//! See \ref stmintro for basic ideas of this STM and code examples.\n
//!
//! Transaction inherits features of Snapshot.
//! Do something like the following to avoid a copy-on-write due to Transaction::operator[]():
//! \code
//! const Snapshot<NodeA> &shot(transaction_A);
//! double x = shot[chidnode].m_x; //reading a value of m_x stored in transaction_A.
//! \endcode
//! \sa Node, Snapshot, SingleSnapshot, SingleTransaction.
template <class XN>
class DECLSPEC_KAME Transaction : public Snapshot<XN> {
public:
    // Inherited bookkeeping now lives on Snapshot<XN> (base). Pull the
    // names into scope so member-function bodies don't need this->
    // qualification for every use.
    using Snapshot<XN>::m_started_time;
    using Snapshot<XN>::m_tid_bitset;
    using Snapshot<XN>::m_packet;
    using Snapshot<XN>::m_serial;
    using Snapshot<XN>::m_tagged_linkages;

    //! Be sure for the persistence of the \a node.
    //! \param[in] multi_nodal If false, the snapshot and following commitment are not aware of the contents of the child nodes.
    explicit Transaction(Node<XN>&node, bool multi_nodal = true) :
        Snapshot<XN>(), m_oldpacket(), m_multi_nodal(multi_nodal) {
        m_started_time = Node<XN>::NegotiationCounter::now_us_tagged();
        m_oneup = std::make_unique<typename Node<XN>::NegotiationCounter::AcquireOneCount>();
        node.snapshot( *this, multi_nodal);
        assert( &m_packet->node() == &node);
        assert( &m_oldpacket->node() == &node);
    }
    //! \param[in] x The snapshot containing the old value of \a node.
    //! \param[in] multi_nodal If false, the snapshot and following commitment are not aware of the contents of the child nodes.
    explicit Transaction(const Snapshot<XN> &x, bool multi_nodal = true) noexcept : Snapshot<XN>(x),
        m_oldpacket(m_packet), m_multi_nodal(multi_nodal) {
        m_started_time = Node<XN>::NegotiationCounter::now_us_tagged();
        m_oneup = std::make_unique<typename Node<XN>::NegotiationCounter::AcquireOneCount>();
    }
    Transaction(Transaction&&x) = default;
    //! Releases any tagged linkages (clearing m_transaction_started_time
    //! when it still equals ours) so later contenders don't race against
    //! a stale owner. Always drops tags; the primary-node tag is always
    //! in the list on multi-nodal retries. Also clears the global
    //! `s_privileged_tidstamp` if THIS Transaction registered it (covers the
    //! abort path so a fair-mode escape doesn't outlive its issuing Tx).
    ~Transaction() {
        Snapshot<XN>::drop_tags_n_privilege();
    }

    //! \return Copy-constructed Payload instance for \a node, which will be included in the commitment.
    template <class T>
    typename T::Payload &operator[](const shared_ptr<T> &node) {
        return operator[]( *node);
    }
    //! \return Copy-constructed Payload instance for \a node, which will be included in the commitment.
    template <class T>
    typename T::Payload &operator[](T &node) {
        assert(isMultiNodal() || ( &node == &this->m_packet->node()));
        auto &payload(
            node.reverseLookup(this->m_packet, true, this->m_serial)->payload());
        if(payload->m_serial != this->m_serial) {
            payload = payload->clone( *this, this->m_serial);
            auto &p( *static_cast<typename T::Payload *>(payload.get()));
            return p;
        }
        auto &p( *static_cast<typename T::Payload *>(payload.get()));
        return p;
    }
    bool isMultiNodal() const noexcept {return m_multi_nodal;}

    //! Reserves an event, to be emitted from \a talker with \a arg after the transaction is committed.
    template <typename T, typename ...Args>
    void mark(T &talker, Args&&...args) {
        if(auto m = talker.createMessage(this->m_serial, std::forward<Args>(args)...)) {
            m_messages.emplace_back(std::move(m));
        }
    }
    //! Cancels reserved events made toward \a x.
    //! \return # of unmarked events.
    int unmark(const shared_ptr<Listener> &x) {
        int canceled = 0;
        for(auto &&msg: m_messages)
            canceled += msg->unmark(x);
        return canceled;
    }

    bool isModified() const noexcept {
        return (this->m_packet != this->m_oldpacket);
    }
    //! \return true if succeeded.
    bool commit() {
        Node<XN> &node(this->m_packet->node());
        if( !isModified() || node.commit( *this)) {
            finalizeCommitment(node);
            return true;
        }
        return false;
    }
    //! Combination of commit() and operator++().
    bool commitOrNext() {
        if(commit())
            return true;
        ++( *this);
        return false;
    }
    //! Prepares for a next transaction after taking a snapshot for \a supernode.
    //! \return a snapshot for \a supernode.
    Snapshot<XN> newTransactionUsingSnapshotFor(Node<XN> &supernode) {
        Snapshot<XN> shot( *this); //for node persistence.
        Node<XN> &node(this->m_packet->node());
        this->operator++();
        supernode.snapshot( *this, true);
        Snapshot<XN> shot_super( *this);
        Snapshot<XN> shot_this(node, shot_super);
        this->Snapshot<XN>::operator=(shot_this);
        this->m_oldpacket = this->m_packet;
        return shot_super;
    }
    Transaction(const Transaction &tr) = delete; //non-copyable.
    Transaction& operator=(const Transaction &tr) = delete; //non-copyable.
private:
    friend class Node<XN>;
//	bool commitAt(Node<XN> &supernode) {
//		if(supernode.commit_at_super( *this)) {
//			finalizeCommitment(this->m_packet->node());
//			return true;
//		}
//		return false;
//	}
    //! Takes another snapshot and prepares for a next transaction.
    Transaction &operator++() {
        // Tx-layer retry counter feeding the livelock probe. Counts outer
        // iterate_commit iterations. The same field (m_tx_retry_count on
        // the Snapshot base, zero-initialised by default) is also bumped
        // from Node::snapshot()'s retry loop — see the doc-block at the
        // field declaration. The probe consumes the aggregated value.
        ++this->m_tx_retry_count;
        Node<XN> &node(this->m_packet->node());
        if(isMultiNodal())
            this->tag_as_contender(node.m_link);
        // Preserve m_tid_bitset across retry cycles: contention evidence
        // from the previous attempt is directly relevant to the next one
        // (same linkages, same contenders). Clearing would reset C=1 on
        // every ++tr, losing adaptive jitter benefit.
        m_messages.clear();
        // m_oneup is still held from ctor; nested-safe AcquireOneCount
        // means we must not re-acquire on every retry.
        this->m_packet->node().snapshot( *this, m_multi_nodal);
        return *this;
    }

    void finalizeCommitment(Node<XN> &node);

    local_shared_ptr<typename Node<XN>::Packet> m_oldpacket;
    const bool m_multi_nodal;
    // m_started_time, m_tid_bitset (and m_tagged_linkages list scaffold)
    // live on Snapshot<XN> (the base class). Transaction retains only its
    // Transaction-specific members below.
    using MessageList = fast_vector<shared_ptr<Message_<Snapshot<XN>>>, 16>;
    MessageList m_messages;
    std::unique_ptr<typename Node<XN>::NegotiationCounter::AcquireOneCount> m_oneup;
};

//! \brief Transaction which does not care of contents (Payload) of subnodes.\n
//! See \ref stmintro for basic ideas of this STM and code examples.
//! \sa Node, Transaction, Snapshot, SingleSnapshot.
template <class XN, typename T>
class DECLSPEC_KAME SingleTransaction : public Transaction<XN> {
public:
    explicit SingleTransaction(T &node) : Transaction<XN>(node, false) {}

    //! \return Copy-constructed Payload instance for \a node, which will be included in the commitment.
    typename T::Payload &operator*() noexcept {
        return ( *this)[static_cast<T &>(this->m_packet->node())];
    }
    //! \return Copy-constructed Payload instance for \a node, which will be included in the commitment.
    typename T::Payload *operator->() noexcept {
        return &( **this);
    }
protected:
};

template <class XN>
void Transaction<XN>::finalizeCommitment(Node<XN> &node) {
    // Bump the per-root-linkage Transaction-commit counter consumed by
    // the optional livelock probe. Single writer (the iterate_commit
    // winner) per tx, so the non-atomic ++ is race-free.
    ++node.m_link->m_tx_commit_count;
    //Clears the time stamp linked to this object and privilage.
    // Drop all contender tags (including TAG_ON_DISTURB child tags) before
    // zeroing m_started_time; drop_tags_n_privilege() matches on the current value.
    this->drop_tags_n_privilege();
    m_started_time = 0;
    m_oneup.reset();

    m_oldpacket.reset();
    //Messaging.
    for(auto &&msg: m_messages)
        msg->talk( *this);
    m_messages.clear();
}

// Helper: strict-retry escalation arbiter.
//  at_threshold — is this iteration eligible for escalation?
//  my_time      — the Transaction's m_started_time.
//  escalated    — [in/out] true once this thread holds the watermark.
//  saved_pr     — [out] previous priority to restore on success.
// Returns true if this call flipped priority up (caller must restore
// on commit). No-op when threshold == 0 (paper-ablation row).
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
inline void strict_escalate_if_oldest(bool at_threshold, int64_t my_time,
                                      bool &escalated, Priority &saved_pr) {
    if(!at_threshold || escalated) return;
    // Min-CAS: update watermark if my_time is smaller.
    int64_t prev = g_strict_watermark.load(std::memory_order_relaxed);
    while(my_time < prev &&
          !g_strict_watermark.compare_exchange_weak(prev, my_time,
              std::memory_order_relaxed)) {}
    // Escalate only if we actually own the watermark.
    if(g_strict_watermark.load(std::memory_order_relaxed) == my_time) {
        saved_pr = getCurrentPriorityMode();
        setCurrentPriorityMode(Priority::HIGHEST);
        escalated = true;
    }
}
inline void strict_release(bool escalated, int64_t my_time, Priority saved_pr) {
    if(!escalated) return;
    setCurrentPriorityMode(saved_pr);
    int64_t expected = my_time;
    // CAS-release the watermark (strong so we don't leak it on spurious
    // failure; no-op if some other thread already replaced our value).
    g_strict_watermark.compare_exchange_strong(expected,
        (int64_t)0x7fffffffffffffffLL, std::memory_order_relaxed);
}
#endif

template <class XN>
template <typename Closure>
Snapshot<XN> Node<XN>::iterate_commit_if(Closure &&closure) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
    int n = 0; Priority saved_pr = Priority::NORMAL; bool escalated = false;
#endif
    for(Transaction<XN> tr( *this);;++tr) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
        ++n;
        strict_escalate_if_oldest(n >= KAME_STM_STRICT_RETRY_THRESHOLD,
                                  tr.m_started_time, escalated, saved_pr);
#endif
        try {
            if( !closure(tr))
                continue; //skipping.
            if(tr.commit()) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
                strict_release(escalated, tr.m_started_time, saved_pr);
#endif
                return std::move(tr);
            }
        }
        catch (const std::bad_alloc &e) {
            Node<XN>::print_recoverable_error(e.what());
        }
    }
}
template <class XN>
template <typename Closure>
Snapshot<XN> Node<XN>::iterate_commit(Closure &&closure) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
    int n = 0; Priority saved_pr = Priority::NORMAL; bool escalated = false;
#endif
    for(Transaction<XN> tr( *this);;++tr) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
        ++n;
        strict_escalate_if_oldest(n >= KAME_STM_STRICT_RETRY_THRESHOLD,
                                  tr.m_started_time, escalated, saved_pr);
#endif
          try {
              closure(tr);
              if(tr.commit()) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
                  strict_release(escalated, tr.m_started_time, saved_pr);
#endif
                  return std::move(tr);
              }
          }
          catch (const std::bad_alloc &e) {
              Node<XN>::print_recoverable_error(e.what());
          }
    }
}
template <class XN>
template <typename Closure>
void Node<XN>::iterate_commit_while(Closure &&closure) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
    int n = 0; Priority saved_pr = Priority::NORMAL; bool escalated = false;
#endif
    for(Transaction<XN> tr( *this);;++tr) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
        ++n;
        strict_escalate_if_oldest(n >= KAME_STM_STRICT_RETRY_THRESHOLD,
                                  tr.m_started_time, escalated, saved_pr);
#endif
        try {
            if( !closure(tr)) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
                strict_release(escalated, tr.m_started_time, saved_pr);
#endif
                 return;
            }
            if(tr.commit()) {
#if KAME_STM_STRICT_RETRY_THRESHOLD > 0
                strict_release(escalated, tr.m_started_time, saved_pr);
#endif
                return;
            }
        }
        catch (const std::bad_alloc &e) {
            Node<XN>::print_recoverable_error(e.what());
        }
    }
}

} //namespace Transactional

#endif /*TRANSACTION_H*/
