/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

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

//! Example 1\n
//! shared_ptr<Subscriber> ss1 = monitor1->monitorData();\n
//! sleep(1);\n
//! if(Snapshot<NodeA> shot1(monitor1)) { // checking consistency (i.e. requiring at least one transaction).\n
//! double x = shot1[node1]; //implicit conversion defined in Node1::Payload.\n
//! double y = shot1[node1].y(); }\n
//!\n
//! Example 2\n
//! double x = *node1; // for an immediate access.\n
//!\n
//! Example 3\n
//! for(;;) { Transaction<NodeA> tr1(monitor1);\n
//! tr1[node1] = tr1[node1] * 2.0;\n
//! if(tr1.commit()) break;}\n
//! \n
//! Example 5\n
//! //Obsolete, for backward compatibility.\n
//! monitor1.readLock(); // or use lock(), a snapshot will be copied to TLS.\n
//! double x = (*node1).y();\n
//! monitor1.readUnock(); // or use unlock()\n
//! \n
//! Example 6\n
//! //Obsolete, for backward compatibility.\n
//! monitor1.writeLock(); // a transaction will be prepared in TLS.\n
//! (*node1).value(1.0);\n
//! monitor1.writeUnock(); // commit the transaction.\n

//! Watch point for transactional memory access.\n
//! The list of the pointers to data is atomically read/written.
/*!
 * Criteria during the transaction:\n
 * 	a) When a conflicting writing to the same node is detected, the \a commit() will fail.
 * 	b) Writing to listed nodes not by the this context but by another thread will be included in the final result.
 */
//! A node which carries data sets for itself and subnodes.
//! Transactional accesses will be made on the top branch in the tree.
//! \sa Snapshot, Transaction, XNode

#include <vector>
#include "atomic.h"

namespace Transactional {

template <class XN>
class Snapshot;
template <class XN>
class Transaction;

template <class XN>
class Node {
public:
	template <class T>
	static T *create();
	template <class T, typename A1>
	static T *create(A1 a1);
	template <class T, typename A1, typename A2>
	static T *create(A1 a1, A2 a2);
	template <class T, typename A1, typename A2, typename A3>
	static T *create(A1 a1, A2 a2, A3 a3);
	template <class T, typename A1, typename A2, typename A3, typename A4>
	static T *create(A1 a1, A2 a2, A3 a3, A4 a4);
	template <class T, typename A1, typename A2, typename A3, typename A4, typename A5>
	static T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5);
	template <class T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
	static T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6);
	template <class T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
	static T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7);

	virtual ~Node();

	class Packet;

	struct PacketList;
	struct NodeList : public std::vector<shared_ptr<XN> > {
		NodeList() : std::vector<shared_ptr<XN> >() {}
	private:
	};
	struct PacketList : public std::vector<local_shared_ptr<Packet> > {
		shared_ptr<NodeList> m_subnodes;
		PacketList() : std::vector<local_shared_ptr<Packet> >(), m_serial(-1) {}
		//! Serial number of the transaction.
		int64_t m_serial;
	};

	typedef typename NodeList::iterator iterator;
	typedef typename NodeList::const_iterator const_iterator;

	struct PayloadWrapperBase : public atomic_countable {
		PayloadWrapperBase(Node &node) : m_node(&node), m_serial(-1) {}
		virtual ~PayloadWrapperBase() {}
		virtual PayloadWrapperBase *clone() = 0;
		//! points to the node.
		Node &node() {return *m_node;}
		//! points to the node.
		const Node &node() const {return *m_node;}
		Node * const m_node;
		//! Serial number of the transaction.
		int64_t m_serial;
	};
	template <class P>
	struct PayloadWrapper : public PayloadWrapperBase, public P {
		virtual PayloadWrapper *clone() { return new PayloadWrapper(*this);}
		static PayloadWrapperBase *funcPayloadCreator(Node &node) { return new PayloadWrapper<P>(node); }
	private:
		PayloadWrapper();
		PayloadWrapper(Node &node) : PayloadWrapperBase(node), P() {}
		PayloadWrapper(const PayloadWrapper &x) : PayloadWrapperBase(x), P(x) {}
		PayloadWrapper& operator=(const PayloadWrapper &x); //non-copyable
	};
	//! Data holder.
	struct Payload {
		Payload() {}
		virtual ~Payload() {}
	};

	struct Packet : public atomic_countable {
		Packet();
		int size() const {return subpackets() ? subpackets()->size() : 0;}
		local_shared_ptr<PayloadWrapperBase> &payload() {return m_payload;}
		const local_shared_ptr<PayloadWrapperBase> &payload() const {return m_payload;}
		shared_ptr<NodeList> &subnodes() {return subpackets()->m_subnodes;}
		shared_ptr<PacketList> &subpackets() {return m_subpackets;}
		const shared_ptr<NodeList> &subnodes() const {return subpackets()->m_subnodes;}
		const shared_ptr<PacketList> &subpackets() const {return m_subpackets;}

		//! points to the node.
		Node &node() {return payload()->node();}
		//! points to the node.
		const Node &node() const {return payload()->node();}

		void _print() const;

		local_shared_ptr<PayloadWrapperBase> m_payload;
		shared_ptr<PacketList> m_subpackets;
		bool m_hasCollision;
		static atomic<int64_t> s_serial;
	};
	struct BranchPoint;
	struct PacketWrapper : public atomic_countable {
		PacketWrapper(const local_shared_ptr<Packet> &x, bool bundled);
		PacketWrapper(const shared_ptr<BranchPoint> &bp, int reverse_index);
		 ~PacketWrapper() {}
		//! \return If true, the content is a snapshot, and is up-to-date for the watchpoint.\n
		//! The subnodes must not have their own payloads.
		//! If false, the content may be out-of-date and ones should fetch those on subnodes.
		//! \sa isHere().
		bool isBundled() const {return packet() && (m_state & PACKET_BUNDLE_STATE) == PACKET_BUNDLED;}
		void setBundled(bool x) {m_state = (m_state & ~PACKET_BUNDLE_STATE) |
			(x ? PACKET_BUNDLED : PACKET_UNBUNDLED);
		}
		const local_shared_ptr<Packet> &packet() const {return m_packet;}
		local_shared_ptr<Packet> &packet() {return m_packet;}

		shared_ptr<BranchPoint> branchpoint() const {return m_branchpoint.lock();}
		int reverseIndex() const {return m_state;}
		void setReverseIndex(int i) {m_state = i;}

		void _print() const;

		weak_ptr<BranchPoint> const m_branchpoint;
		local_shared_ptr<Packet> m_packet;
		int m_state; //!< is also used for reverseIndex().
		enum STATE {
			PACKET_BUNDLE_STATE = 0xf,
			PACKET_UNBUNDLED = 0x1, PACKET_BUNDLED = 0x2
		};
	};
	struct BranchPoint : public atomic_shared_ptr<PacketWrapper> {
		BranchPoint() : atomic_shared_ptr<PacketWrapper>(), m_bundle_serial(-1) {}
		atomic<int64_t> m_bundle_serial;
	};

	bool insert(Transaction<XN> &tr, const shared_ptr<XN> &var);
	void insert(const shared_ptr<XN> &var);
	bool release(Transaction<XN> &tr, const shared_ptr<XN> &var);
	void release(const shared_ptr<XN> &var);
	void releaseAll();
	bool swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y);
	void swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y);
private:
	friend class Snapshot<XN>;
	friend class Transaction<XN>;
	void snapshot(Snapshot<XN> &target, bool multi_nodal) const;
	void snapshot(Transaction<XN> &target, bool multi_nodal) const {
		snapshot(static_cast<Snapshot<XN> &>(target), multi_nodal);
		target.m_oldpacket = target.m_packet;
	}
	enum SnapshotStatus {SNAPSHOT_SUCCESS, SNAPSHOT_DISTURBED, SNAPSHOT_STRUCTURE_HAS_CHANGED};
	static SnapshotStatus snapshotFromSuper(shared_ptr<BranchPoint > &branchpoint,
		local_shared_ptr<PacketWrapper> &shot, local_shared_ptr<Packet> **subpacket,
		shared_ptr<BranchPoint > *branchpoint_2nd = NULL);
	bool commit(Transaction<XN> &tr, bool new_bundle_state = true);

	enum BundledStatus {BUNDLE_SUCCESS, BUNDLE_DISTURBED};
	BundledStatus bundle(local_shared_ptr<PacketWrapper> &target, const int64_t *bundle_serial = NULL);
	enum UnbundledStatus {UNBUNDLE_W_NEW_SUBVALUE, UNBUNDLE_W_NEW_VALUES,
		UNBUNDLE_SUBVALUE_HAS_CHANGED, UNBUNDLE_COLLIDED,
		UNBUNDLE_SUCCESS, UNBUNDLE_PARTIALLY, UNBUNDLE_DISTURBED};
	static UnbundledStatus unbundle(const int64_t *bundle_serial,
		BranchPoint &branchpoint,
		BranchPoint &subbranchpoint, const local_shared_ptr<PacketWrapper> &nullwrapper,
		const local_shared_ptr<Packet> *oldsubpacket = NULL, local_shared_ptr<PacketWrapper> *newsubwrapper = NULL,
		const local_shared_ptr<Packet> *oldsuperpacket = NULL, const local_shared_ptr<PacketWrapper> *newsuperwrapper = NULL,
		bool new_sub_bunlde_state = true);
	shared_ptr<BranchPoint> m_wrapper;

	struct Cache : public atomic_countable {
		weak_ptr<PacketList> subpackets;
		int index;
	};
	atomic_shared_ptr<Cache> m_packet_cache;
	mutable atomic<int> m_transaction_count;

	//! finds the packet for this node in the (un)bundled \a packet.
	//! \arg packet The bundled packet.
	//! \arg copy_branch If ture, new packets and packet lists will be copy-created for writing.
	//! \arg tr_serial The serial number associated with the transaction.
	local_shared_ptr<Packet> &reverseLookup(local_shared_ptr<Packet> &packet,
		bool copy_branch, int tr_serial = 0);
	const local_shared_ptr<Packet> &reverseLookup(
		const local_shared_ptr<Packet> &packet) const {
		return const_cast<Node*>(this)->reverseLookup(
			const_cast<local_shared_ptr<Packet> &>(packet), false);
	}
	static local_shared_ptr<Packet> *reverseLookupWithHint(shared_ptr<BranchPoint > &branchpoint,
		local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial, Cache *cache);
	//! finds this node and a corresponding packet in the (un)bundled \a packet.
	local_shared_ptr<Packet> *forwardLookup(local_shared_ptr<Packet> &packet,
		bool copy_branch, int tr_serial, Cache *cache) const;
protected:
	//! Use \a create().
	Node();
private:
	Node(const Node &); //non-copyable.
	Node &operator=(const Node &); //non-copyable.
	typedef PayloadWrapperBase *(*FuncPayloadCreator)(Node &);
	static XThreadLocal<FuncPayloadCreator> stl_funcPayloadCreator;
	void _print() const;
};

template <class XN>
template <class T>
T *Node<XN>::create() {
	*T::stl_funcPayloadCreator = &PayloadWrapper<typename T::Payload>::funcPayloadCreator;
	return new T();
}
template <class XN>
template <class T, typename A1>
T *Node<XN>::create(A1 a1) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<typename T::Payload>::funcPayloadCreator;
	return new T(a1);
}
template <class XN>
template <class T, typename A1, typename A2>
T *Node<XN>::create(A1 a1, A2 a2) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<typename T::Payload>::funcPayloadCreator;
	return new T(a1, a2);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<typename T::Payload>::funcPayloadCreator;
	return new T(a1, a2, a3);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3, typename A4>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3, A4 a4) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<typename T::Payload>::funcPayloadCreator;
	return new T(a1, a2, a3, a4);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3, typename A4, typename A5>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<typename T::Payload>::funcPayloadCreator;
	return new T(a1, a2, a3, a4, a5);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<typename T::Payload>::funcPayloadCreator;
	return new T(a1, a2, a3, a4, a5, a6);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<typename T::Payload>::funcPayloadCreator;
	return new T(a1, a2, a3, a4, a5, a6, a7);
}

//! This class takes a snapshot for a monitored data set.
template <class XN>
class Snapshot {
public:
	Snapshot(const Snapshot&x) : m_packet(x.m_packet), m_bundled(x.m_bundled) {
	}
	Snapshot(const Transaction<XN>&x);
	explicit Snapshot(const Node<XN>&node, bool multi_nodal = true) {
		node.snapshot(*this, multi_nodal);
		++node.m_transaction_count;
	}
	virtual ~Snapshot() {
		--this->m_packet->node().m_transaction_count;
	}

	template <class T>
	const typename T::Payload &operator[](const shared_ptr<T> &node) const {
		return operator[](const_cast<const T&>(*node));
	}
	template <class T>
	const typename T::Payload &operator[](const T &node) const {
		const local_shared_ptr<typename Node<XN>::Packet> &packet(node.reverseLookup(m_packet));
		const local_shared_ptr<typename Node<XN>::PayloadWrapperBase> &payload(packet->payload());
		typedef typename Node<XN>::template PayloadWrapper<typename T::Payload> Payload;
		const typename T::Payload *payload_t(static_cast<const Payload*>(payload.get()));
		return *payload_t;
	}
	int size() const {return m_packet->size();}
	const shared_ptr<const typename Node<XN>::NodeList> list() const {
		if( ! size())
			return shared_ptr<typename Node<XN>::NodeList>();
		return m_packet->subnodes();
	}
	int size(const shared_ptr<Node<XN> > &node) const {
		return node->reverseLookup(m_packet)->size();
	}
	shared_ptr<const typename Node<XN>::NodeList> list(const shared_ptr<Node<XN> > &node) const {
		local_shared_ptr<typename Node<XN>::Packet> const &packet(node->reverseLookup(m_packet));
		if( ! packet->size() )
			return shared_ptr<typename Node<XN>::NodeList>();
		return packet->subnodes();
	}
	void print() {
		m_packet->_print();
	}
	bool isBundled() const {return m_bundled;}
protected:
	friend class Node<XN>;
	//! The snapshot.
	local_shared_ptr<typename Node<XN>::Packet> m_packet;
	bool m_bundled;
	Snapshot() : m_packet() {}
};

template <class XN, typename T>
class SingleSnapshot : protected Snapshot<XN> {
public:
	explicit SingleSnapshot(const T&node) : Snapshot<XN>(node, false) {}
	virtual ~SingleSnapshot() {}

	const typename T::Payload *operator->() const {
		typedef typename Node<XN>::template PayloadWrapper<typename T::Payload> Payload;
		return &static_cast<const typename T::Payload&>(
			*static_cast<const Payload *>(this->m_packet->payload().get()));
	}
	template <class X>
	operator X() const {
		typedef typename Node<XN>::template PayloadWrapper<typename T::Payload> Payload;
		return (X)static_cast<const typename T::Payload&>(
			*static_cast<const Payload *>(this->m_packet->payload().get()));
	}
protected:
};
//! Transactional writing for a monitored data set.
//! The revision will be committed implicitly on leaving the scope.
template <class XN>
class Transaction : public Snapshot<XN> {
public:
	//! Be sure to the persistence of the \a node.
	explicit Transaction(Node<XN>&node, bool multi_nodal = true) :
		Snapshot<XN>(), m_oldpacket(), m_trial_count(0), m_multi_nodal(multi_nodal) {
		for(;;) {
			m_serial = Node<XN>::Packet::s_serial;
			if(Node<XN>::Packet::s_serial.compareAndSet(m_serial, m_serial + 1))
				break;
		}
		m_serial++;
		node.snapshot(*this, multi_nodal);
		ASSERT(&this->m_packet->node() == &node);
		ASSERT(&this->m_oldpacket->node() == &node);
	}
	virtual ~Transaction() {
	}
	//! Explicitly commits.
	bool commit() {
		++m_trial_count;
		if( ! isModified())
			return true;
		return this->m_packet->node().commit(*this);
	}
	//! Explicitly commits.
	bool commitOrNext() {
		if(commit())
			return true;
		++(*this);
		return false;
	}

	bool isModified() const {
		return (this->m_packet != this->m_oldpacket);
	}

	Transaction &operator++() {
		for(;;) {
			m_serial = Node<XN>::Packet::s_serial;
			if(Node<XN>::Packet::s_serial.compareAndSet(m_serial, m_serial + 1))
				break;
		}
		m_serial++;
		this->m_packet->node().snapshot(*this, m_multi_nodal);
		return *this;
	}

	template <class T>
	typename T::Payload &operator[](const shared_ptr<T> &node) {
		return operator[](*node);
	}
	template <class T>
	typename T::Payload &operator[](T &node) {
		local_shared_ptr<typename Node<XN>::PayloadWrapperBase> &payload(
			node.reverseLookup(this->m_packet, true, this->m_serial)->payload());
		typedef typename Node<XN>::template PayloadWrapper<typename T::Payload> Payload;
		Payload *payload_t(static_cast<Payload*>(payload.get()));
		if(payload->m_serial != this->m_serial) {
			payload_t = payload_t->clone();
			payload.reset(payload_t);
			payload_t->m_serial = this->m_serial;
			return *payload_t;
		}
		ASSERT(payload_t);
		return *payload_t;
	}
	bool isMultiNodal() const {return m_multi_nodal;}
private:
	Transaction(const Transaction &tr); //non-copyable.
	Transaction& operator=(const Transaction &tr); //non-copyable.
	friend class Node<XN>;
	local_shared_ptr<typename Node<XN>::Packet> m_oldpacket;
	int m_trial_count;
	int64_t m_serial;
	const bool m_multi_nodal;
};

template <class XN>
class SingleTransaction : public Transaction<XN> {
public:
	explicit SingleTransaction(Node<XN>&node) : Transaction<XN>(node, false) {}
	virtual ~SingleTransaction() {}
protected:
};

template <class XN>
inline Snapshot<XN>::Snapshot(const Transaction<XN>&x) : m_packet(x.m_packet) {}

} //namespace Transactional

#endif /*TRANSACTION_H*/
