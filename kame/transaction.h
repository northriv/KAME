/***************************************************************************
        Copyright (C) 2002-2015 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef TRANSACTION_H
#define TRANSACTION_H

#include "support.h"
#include "threadlocal.h"
#include "atomic_smart_ptr.h"
#include <vector>
#include "atomic.h"
#include "allocator.h"
#include "xtime.h"

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

//! \brief This is a base class of nodes which carries data sets for itself (Payload) and for subnodes.\n
//! See \ref stmintro for basic ideas of this STM and code examples.
//!
//! \tparam XN a class type used in the smart pointers of NodeList. \a XN must be a derived class of Node<XN> itself.
//! \sa Snapshot, Transaction.
//! \sa XNode.
template <class XN>
class DECLSPEC_KAME Node {
public:
    template <class T, typename... Args>
    static T *create(Args&&... args);

    virtual ~Node();

    typedef std::domain_error NodeNotFoundError;

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
    inline Snapshot<XN> iterate_commit(Closure);
    //! Iterates a transaction covering the node and children. Skips the iteration when the closure returns false.
    //! \param Closure Typical: [=](Transaction<Node1> &tr){ somecode...; return ret; }
    template <typename Closure>
    inline Snapshot<XN> iterate_commit_if(Closure);
    //! Iterates a transaction covering the node and children, as long as the closure returns true.
    //! \param Closure Typical: [=](Transaction<Node1> &tr){ somecode...; return ret; }
    template <typename Closure>
    inline void iterate_commit_while(Closure);

    //! Data holder and accessor for the node.
    //! Derive Node<XN>::Payload as (\a subclass)::Payload.
    //! The instances have to be capable of copy-construction and be safe to be shared reading.
    struct Payload : public atomic_countable {
        Payload() noexcept : m_serial(-1), m_tr(nullptr) {}
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

    typedef fast_vector<shared_ptr<XN> > NodeList;
    typedef typename NodeList::iterator iterator;
    typedef typename NodeList::const_iterator const_iterator;

    Node(const Node &) = delete; //non-copyable.
    Node &operator=(const Node &) = delete; //non-copyable.
private:
    struct Packet;

    struct PacketList;
    struct PacketList : public fast_vector<local_shared_ptr<Packet> > {
        shared_ptr<NodeList> m_subnodes;
        PacketList() : fast_vector<local_shared_ptr<Packet> >(), m_serial(SerialGenerator::gen()) {}
        ~PacketList() {this->clear();} //destroys payloads prior to nodes.
        //! Serial number of the transaction.
        int64_t m_serial;
    };

    template <class P>
    struct PayloadWrapper : public P::Payload {
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
        PayloadWrapper(const PayloadWrapper &x) : P::Payload(x) {}
    private:
    };

    struct PacketWrapper;
    struct Linkage;
    //! A package containing \a Payload, sub-Packets, and a list of subnodes.\n
    //! Not-"missing" packet is always up-to-date and can be a snapshot of the subtree,
    //! and packets possessed by the sub-instances may be out-of-date.\n
    //! "missing" indicates that the package lacks any Packet for subnodes, or
    //! any content may be out-of-date.\n
    struct Packet : public atomic_countable {
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

        enum {SERIAL_NULL = 0};
        local_shared_ptr<Payload> m_payload;
        shared_ptr<PacketList> m_subpackets;
        bool m_missing;
    };

    struct NegotiationCounter {
        using cnt_t = int64_t;
        static cnt_t now() noexcept {
            auto tm = XTime::now();
            return (cnt_t)tm.sec() * 1000000uL + tm.usec();
        }
    private:
    };
    struct ProcessCounter {
        using cnt_t = uint16_t;
        ProcessCounter();
        operator cnt_t() const noexcept {return m_var;}
    private:
        cnt_t m_var;
        static atomic<cnt_t> s_count;
    };
    //! Holds the current porcess(thread) internal ID. Taking non-zero value.
    static XThreadLocal<ProcessCounter> stl_processID;
    //! Generates a serial number assigned to bundling and transaction.\n
    //! During a transaction, a serial is used for determining whether Payload or PacketList should be cloned.\n
    //! During bundle(), it is used to prevent infinite loops due to unbundle()-ing itself.
    struct SerialGenerator {
        struct cnt_t {
            cnt_t() {
                m_var = (int64_t)*stl_processID * 1000000000000uLL;
            }
            operator int64_t() const noexcept {return m_var;}
            //16bit process ID + 48bit counter.
            cnt_t &operator++(int) noexcept {
                m_var = ((m_var + 1) & 0xffffffffffffuLL)
                | ((m_var) & 0xffff000000000000uLL);
                return *this;
            }
            int64_t m_var;
        };
        static int64_t current() noexcept {
            return *stl_serial;
        }
        static int64_t gen() noexcept {
            auto &v = *stl_serial;
            v++;
            return v;
        }
        static XThreadLocal<cnt_t> stl_serial;
    };
    //! A class wrapping Packet and providing indice and links for lookup.\n
    //! If packet() is absent, a super node should have the up-to-date Packet.\n
    //! If hasPriority() is not set, Packet in a super node may be latest.
    struct PacketWrapper : public atomic_countable {
        PacketWrapper(const local_shared_ptr<Packet> &x, int64_t bundle_serial) noexcept;
        //! creates a wrapper not containing a packet but pointing to the upper node.
        //! \param[in] bp \a m_link of the upper node.
        //! \param[in] reverse_index The index for this node in the list of the upper node.
        PacketWrapper(const shared_ptr<Linkage> &bp, int reverse_index, int64_t bundle_serial) noexcept;
        PacketWrapper(const PacketWrapper &x, int64_t bundle_serial) noexcept;
        bool hasPriority() const noexcept { return m_ridx == PACKET_HAS_PRIORITY; }
        const local_shared_ptr<Packet> &packet() const noexcept {return m_packet;}
        local_shared_ptr<Packet> &packet() noexcept {return m_packet;}

        //! Points to the upper node that should have the up-to-date Packet when this lacks priority.
        shared_ptr<Linkage> bundledBy() const noexcept {return m_bundledBy.lock();}
        //! The index for this node in the list of the upper node.
        int reverseIndex() const noexcept {return m_ridx;}
        void setReverseIndex(int i) noexcept {m_ridx = i;}

        void print_() const;
        weak_ptr<Linkage> const m_bundledBy;
        local_shared_ptr<Packet> m_packet;
        int m_ridx;
        int64_t m_bundle_serial;
        enum PACKET_STATE { PACKET_HAS_PRIORITY = -1};

        PacketWrapper(const PacketWrapper &) = delete;
    };
    struct Linkage : public atomic_shared_ptr<PacketWrapper> {
        Linkage() noexcept : atomic_shared_ptr<PacketWrapper>(), m_transaction_started_time(0) {}
        ~Linkage() {this->reset(); } //Packet should be freed before memory pools.
        atomic<typename NegotiationCounter::cnt_t> m_transaction_started_time;
        //! Puts a wait so that the slowest thread gains a chance to finish its transaction, if needed.
        void negotiate(typename NegotiationCounter::cnt_t &started_time, float mult_wait = 6.0f) noexcept {
            auto transaction_started_time = m_transaction_started_time;
            if( !transaction_started_time)
                return; //collision has not been detected.
            negotiate_internal(started_time, mult_wait);
        }
        void negotiate_internal(typename NegotiationCounter::cnt_t &started_time, float mult_wait) noexcept;
        MemoryPool m_mempoolPayload;
        MemoryPool m_mempoolPacket;
        MemoryPool m_mempoolPacketList;
        MemoryPool m_mempoolPacketWrapper;
        NegotiationCounter m_negotiationCounter;
    };

    friend class Snapshot<XN>;
    friend class Transaction<XN>;

    void snapshot(Snapshot<XN> &target, bool multi_nodal, typename NegotiationCounter::cnt_t started_time) const;
    void snapshot(Transaction<XN> &target, bool multi_nodal) const {
        m_link->negotiate(target.m_started_time, 4.0f);
        snapshot(static_cast<Snapshot<XN> &>(target), multi_nodal, target.m_started_time);
        target.m_oldpacket = target.m_packet;
    }
    enum SnapshotStatus {SNAPSHOT_SUCCESS = 0, SNAPSHOT_DISTURBED = 1,
        SNAPSHOT_VOID_PACKET = 2, SNAPSHOT_NODE_MISSING = 4,
        SNAPSHOT_COLLIDED = 8, SNAPSHOT_NODE_MISSING_AND_COLLIDED = 12};
    struct CASInfo {
        CASInfo() = default;
        CASInfo(const shared_ptr<Linkage> &b, const local_shared_ptr<PacketWrapper> &o,
            const local_shared_ptr<PacketWrapper> &n) : linkage(b), old_wrapper(o), new_wrapper(n) {}
        shared_ptr<Linkage> linkage;
        local_shared_ptr<PacketWrapper> old_wrapper, new_wrapper;
    };
    typedef fast_vector<CASInfo, 16> CASInfoList;
    enum SnapshotMode {SNAPSHOT_FOR_UNBUNDLE, SNAPSHOT_FOR_BUNDLE};
    static inline SnapshotStatus snapshotSupernode(const shared_ptr<Linkage> &linkage, shared_ptr<Linkage> &linkage_super,
        local_shared_ptr<PacketWrapper> &shot, local_shared_ptr<Packet> **subpacket,
        SnapshotMode mode,
        int64_t serial = Packet::SERIAL_NULL, CASInfoList *cas_infos = nullptr);

    //! Updates a packet to \a tr.m_packet if the current packet is unchanged (== \a tr.m_oldpacket).
    //! If this node has been bundled at the super node, unbundle() will be called.
    //! \sa Transaction<XN>::commit().
    bool commit(Transaction<XN> &tr);
//	bool commit_at_super(Transaction<XN> &tr);

    enum BundledStatus {BUNDLE_SUCCESS, BUNDLE_DISTURBED};
    //! Bundles all the subpackets so that the whole packet can be treated atomically.
    //! Namely this function takes a snapshot.
    //! All the subpackets held by \a m_link at the subnodes will be
    //! cleared and each PacketWrapper::bundledBy() will point to its upper node.
    //! \sa snapshot().
    BundledStatus bundle(local_shared_ptr<PacketWrapper> &target,
        typename NegotiationCounter::cnt_t &started_time, int64_t bundle_serial, bool is_bundle_root);
    BundledStatus bundle_subpacket(local_shared_ptr<PacketWrapper> *superwrapper, const shared_ptr<Node> &subnode,
        local_shared_ptr<PacketWrapper> &subwrapper, local_shared_ptr<Packet> &subpacket_new,
        typename NegotiationCounter::cnt_t &started_time, int64_t bundle_serial);
    enum UnbundledStatus {UNBUNDLE_W_NEW_SUBVALUE,
        UNBUNDLE_SUBVALUE_HAS_CHANGED,
        UNBUNDLE_COLLIDED, UNBUNDLE_DISTURBED};
    //! Unbundles a subpacket to \a sublinkage from a snapshot.
    //! it performs unbundling for all the super nodes.
    //! The super nodes will lose priorities against their lower nodes.
    //! \param[in] bundle_serial If not zero, consistency/collision wil be checked.
    //! \param[in] null_linkage The current value of \a sublinkage and should not contain \a packet().
    //! \param[in] oldsubpacket If not zero, the packet will be compared with the packet inside the super packet.
    //! \param[in,out] newsubwrapper If \a oldsubpacket and \a newsubwrapper are not zero, \a newsubwrapper will be a new value.
    //! If \a oldsubpacket is zero, unloaded value  of \a sublinkage will be substituted to \a newsubwrapper.
    static UnbundledStatus unbundle(const int64_t *bundle_serial, typename NegotiationCounter::cnt_t &time_started,
        const shared_ptr<Linkage> &sublinkage, const local_shared_ptr<PacketWrapper> &null_linkage,
        const local_shared_ptr<Packet> *oldsubpacket = NULL,
        local_shared_ptr<PacketWrapper> *newsubwrapper = NULL,
        local_shared_ptr<PacketWrapper> *superwrapper = NULL);
    //! The point where the packet is held.
    shared_ptr<Linkage> m_link;
    //! Allocators for memory pools in the Linkage.
    allocator<Payload> m_allocatorPayload;
    allocator<Packet> m_allocatorPacket;
    allocator<PacketList> m_allocatorPacketList;
    allocator<PacketWrapper> m_allocatorPacketWrapper;

    //! finds the packet for this node in the (un)bundled \a packet.
    //! \param[in,out] superpacket The bundled packet.
    //! \param[in] copy_branch If ture, new packets and packet lists will be copy-created for writing.
    //! \param[in] tr_serial The serial number associated with the transaction.
    inline local_shared_ptr<Packet> *reverseLookup(local_shared_ptr<Packet> &superpacket,
        bool copy_branch, int64_t tr_serial, bool set_missing, XN** uppernode);
    inline local_shared_ptr<Packet> &reverseLookup(local_shared_ptr<Packet> &superpacket,
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
    static void eraseSerials(local_shared_ptr<Packet> &packet, int64_t serial);
protected:
    //! Use \a create().
    Node();
private:
    using FuncPayloadCreator = Payload *(*)(XN &);
    static XThreadLocal<FuncPayloadCreator> stl_funcPayloadCreator;
    void lookupFailure() const;
    local_shared_ptr<typename Node<XN>::Packet>*lookupFromChild(local_shared_ptr<Packet> &superpacket,
        bool copy_branch, int64_t tr_serial, bool set_missing, XN **uppernode);
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
class Snapshot {
public:
    Snapshot(const Snapshot&x) noexcept = default;
    Snapshot(Snapshot&&x) noexcept = default;
    Snapshot(Node<XN> &node, const Snapshot &x) noexcept : m_packet(node.reverseLookup(x.m_packet)), m_serial(x.m_serial) {}
    explicit Snapshot(const Transaction<XN>&x) noexcept : m_packet(x.m_packet), m_serial(x.m_serial) {}
    explicit Snapshot(Transaction<XN>&&x) noexcept : m_packet(std::move(x.m_packet)), m_serial(x.m_serial) {}
    Snapshot& operator=(const Snapshot&x) noexcept = default;
    explicit Snapshot(const Node<XN>&node, bool multi_nodal = true) {
        node.snapshot( *this, multi_nodal, Node<XN>::NegotiationCounter::now());
    }

    //! \return Payload instance for \a node, which should be included in the snapshot.
    template <class T>
    const typename T::Payload &operator[](const shared_ptr<T> &node) const noexcept {
        return operator[](const_cast<const T&>( *node));
    }
    //! \return Payload instance for \a node, which should be included in the snapshot.
    template <class T>
    const typename T::Payload &operator[](const T &node) const noexcept {
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
    template <typename T, typename tArgRef>
    void talk(T &talker, tArgRef arg) const { talker.talk( *this, arg); }
protected:
    friend class Node<XN>;
    //! The snapshot.
    local_shared_ptr<typename Node<XN>::Packet> m_packet;
    int64_t m_serial;
    Snapshot() = default;
};
//! \brief Snapshot class which does not care of contents (Payload) for subnodes.\n
//! See \ref stmintro for basic ideas of this STM and code examples.
//! \sa Node, Snapshot, Transaction, SingleTransaction.
template <class XN, typename T>
class SingleSnapshot : protected Snapshot<XN> {
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

}
#include "transaction_signal.h"
namespace Transactional {

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
class Transaction : public Snapshot<XN> {
public:
    //! Be sure for the persistence of the \a node.
    //! \param[in] multi_nodal If false, the snapshot and following commitment are not aware of the contents of the child nodes.
    explicit Transaction(Node<XN>&node, bool multi_nodal = true) :
        Snapshot<XN>(), m_oldpacket(), m_multi_nodal(multi_nodal) {
        m_started_time = Node<XN>::NegotiationCounter::now();
        node.snapshot( *this, multi_nodal);
        assert( &this->m_packet->node() == &node);
        assert( &this->m_oldpacket->node() == &node);
    }
    //! \param[in] x The snapshot containing the old value of \a node.
    //! \param[in] multi_nodal If false, the snapshot and following commitment are not aware of the contents of the child nodes.
    explicit Transaction(const Snapshot<XN> &x, bool multi_nodal = true) noexcept : Snapshot<XN>(x),
        m_oldpacket(this->m_packet), m_multi_nodal(multi_nodal) {
        m_started_time = Node<XN>::NegotiationCounter::now();
    }
    ~Transaction() {
        //Do not leave the time stamp.
        if(m_started_time && this->m_packet) { //not moved
            Node<XN> &node(this->m_packet->node());
            if(node.m_link->m_transaction_started_time == m_started_time) {
                node.m_link->m_transaction_started_time = 0;
            }
        }
    }
    Transaction(Transaction&&x) noexcept = default;

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
    template <typename T, typename tArgRef>
    void mark(T &talker, tArgRef arg) {
        if(auto m = talker.createMessage(this->m_serial, arg)) {
            if( !m_messages)
                m_messages.reset(new MessageList);
            m_messages->push_back(m);
        }
    }
    //! Cancels reserved events made toward \a x.
    //! \return # of unmarked events.
    int unmark(const shared_ptr<XListener> &x) {
        int canceled = 0;
        if(m_messages)
            for(auto &&msg: *m_messages)
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
        Node<XN> &node(this->m_packet->node());
        if(isMultiNodal()) {
            auto time(node.m_link->m_transaction_started_time);
            if( !time || (time > m_started_time))
                node.m_link->m_transaction_started_time = m_started_time;
        }
        m_messages.reset();
        this->m_packet->node().snapshot( *this, m_multi_nodal);
        return *this;
    }

    void finalizeCommitment(Node<XN> &node);

    local_shared_ptr<typename Node<XN>::Packet> m_oldpacket;
    const bool m_multi_nodal;
    typename Node<XN>::NegotiationCounter::cnt_t m_started_time;
//    typedef fast_vector<shared_ptr<Message_<XN> >, 4> MessageList;
    typedef std::vector<shared_ptr<Message_<XN> >> MessageList;
    unique_ptr<MessageList> m_messages;
};

//! \brief Transaction which does not care of contents (Payload) of subnodes.\n
//! See \ref stmintro for basic ideas of this STM and code examples.
//! \sa Node, Transaction, Snapshot, SingleSnapshot.
template <class XN, typename T>
class SingleTransaction : public Transaction<XN> {
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
    //Clears the time stamp linked to this object.
    if(node.m_link->m_transaction_started_time == m_started_time) {
        node.m_link->m_transaction_started_time = 0;
    }
    m_started_time = 0;

    m_oldpacket.reset();
    //Messaging.
    if(m_messages) {
        for(auto &&msg: *m_messages)
            msg->talk( *this);
        m_messages.reset();
    }
}

template <class XN>
template <typename Closure>
inline Snapshot<XN> Node<XN>::iterate_commit(Closure closure) {
    for(Transaction<XN> tr( *this);;++tr) {
        closure(tr);
        if(tr.commit())
            return std::move(tr);
    }
}
template <class XN>
template <typename Closure>
inline Snapshot<XN> Node<XN>::iterate_commit_if(Closure closure) {
    //std::is_integral<std::result_of<Closure>>::type
    for(Transaction<XN> tr( *this);;++tr) {
        if( !closure(tr))
            continue; //skipping.
        if(tr.commit())
            return std::move(tr);
    }
}
template <class XN>
template <typename Closure>
inline void Node<XN>::iterate_commit_while(Closure closure) {
    for(Transaction<XN> tr( *this);;++tr) {
        if( !closure(tr))
             return;
        if(tr.commit())
            return;
    }
}

template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookup(local_shared_ptr<Packet> &superpacket,
    bool copy_branch, int64_t tr_serial, bool set_missing, XN **uppernode) {
    local_shared_ptr<Packet> *foundpacket;
    if( &superpacket->node() == this)
        foundpacket = &superpacket;
    else
        foundpacket = lookupFromChild(superpacket,
            copy_branch, tr_serial, set_missing, uppernode);

    if(copy_branch && (( *foundpacket)->payload()->m_serial != tr_serial)) {
        foundpacket->reset(new Packet( **foundpacket));
//        *foundpacket = allocate_local_shared<Packet>(
//                (*foundpacket)->node().m_allocatorPacket, **foundpacket);
    }
    return foundpacket;
}

template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>&
Node<XN>::reverseLookup(local_shared_ptr<Packet> &superpacket,
    bool copy_branch, int64_t tr_serial, bool set_missing) {
    local_shared_ptr<Packet> *foundpacket = reverseLookup(superpacket, copy_branch, tr_serial, set_missing, 0);
    if( !foundpacket)
        lookupFailure();
    return *foundpacket;
}

template <class XN>
inline const local_shared_ptr<typename Node<XN>::Packet> &
Node<XN>::reverseLookup(const local_shared_ptr<Packet> &superpacket) const {
    local_shared_ptr<Packet> *foundpacket = const_cast<Node*>(this)->reverseLookup(
        const_cast<local_shared_ptr<Packet> &>(superpacket), false, 0, false, 0);
    if( !foundpacket)
        lookupFailure();
    return *foundpacket;
}


} //namespace Transactional

#endif /*TRANSACTION_H*/
