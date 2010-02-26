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
#include <vector>
#include <deque>
#include "atomic.h"
#include "xtime.h"

//! \desc
//! Lock-free software transactional memory for treed objects. \n
//! Transactional accesses for data (implemented in Payload and derived classes), and\n
//! (multiple) insertion/removal of objects to the tree can be performed atomically. \n
//! \n
//! Example 1\n
//! { Snapshot<NodeA> shot1(node1); \n
//! double x = shot1[node1]; //implicit conversion defined in NodeA::Payload.\n
//! double y = shot1[node1].y(); }\n
//!\n
//! Example 2\n
//! for(Transaction<NodeA> tr1(node1);; ++tr1) { \n
//! tr1[node1] = tr1[node1] * 2.0;\n
//! if(tr1.commit()) break;}\n
//! \n
//! Example 3\n
//! for(Transaction<NodeA> tr1(node1);; ++tr1) { \n
//! if(tr1.insert(node2)) break;}\n
//! { Snapshot<NodeA> shot1(node1); \n
//! double x = shot1[node1]; \n
//! double y = shot1[node2]; }\n
//! Other examples can be seen in transaction_test.cpp\n

namespace Transactional {

template <class XN>
class Snapshot;
template <class XN>
class Transaction;

//! A node which carries data sets for itself and for subnodes.\n
//! The data is held by \a *this or packed by the "bundled" data in the super node.\n
//! If the packed data (packet) is tagged as "unbundled" (not up-to-date),
//! readers must check data held by the subnodes.
//! \sa Snapshot, Transaction, XNode
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

	bool insert(Transaction<XN> &tr, const shared_ptr<XN> &var, bool online_after_insertion = false);
	void insert(const shared_ptr<XN> &var);
	bool release(Transaction<XN> &tr, const shared_ptr<XN> &var);
	void release(const shared_ptr<XN> &var);
	void releaseAll();
	bool swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y);
	void swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y);

	XN *upperNode(Snapshot<XN> &shot);

	//! Data and accessor linked to the node.
	//! Re-implement members in its subclasses.
	struct Payload : public atomic_countable {
		Payload() : m_node(0), m_serial(-1), m_tr(0) {}
		virtual ~Payload() {}
		//! points to the node.
		XN &node() {return *m_node;}
		//! points to the node.
		const XN &node() const {return *m_node;}
		int64_t serial() const {return this->m_serial;}
		Transaction<XN> &tr() { return *this->m_tr;}
		virtual Payload *clone(Transaction<XN> &tr, int64_t serial) = 0;

		virtual void catchEvent(const shared_ptr<XN>&, int) {}
		virtual void releaseEvent(const shared_ptr<XN>&, int) {}
		virtual void moveEvent(unsigned int src_idx, unsigned int dst_idx) {}
		virtual void listChangeEvent() {}

		XN *m_node;
		//! Serial number of the transaction.
		int64_t m_serial;
		Transaction<XN> *m_tr;
	};

	void _print() const;

	struct NodeList : public std::vector<shared_ptr<XN> > {
		NodeList() : std::vector<shared_ptr<XN> >() {}
	private:
	};
	typedef typename NodeList::iterator iterator;
	typedef typename NodeList::const_iterator const_iterator;

private:
	class Packet;

	struct PacketList;
	struct PacketList : public std::vector<local_shared_ptr<Packet> > {
		shared_ptr<NodeList> m_subnodes;
		PacketList() : std::vector<local_shared_ptr<Packet> >(), m_serial(Packet::SERIAL_INIT) {}
		//! Serial number of the transaction.
		int64_t m_serial;
	};

	template <class P>
	struct PayloadWrapper : public P::Payload {
		virtual PayloadWrapper *clone(Transaction<XN> &tr, int64_t serial) {
			PayloadWrapper *p = new PayloadWrapper(*this);
			p->m_tr = &tr;
			p->m_serial = serial;
			return p;
		}
		static Payload *funcPayloadCreator(XN &node) {
			Payload *p = new PayloadWrapper(node);
			return p;
		}
	private:
		PayloadWrapper();
		PayloadWrapper(XN &node) : P::Payload() {this->m_node = &node;}
		PayloadWrapper(const PayloadWrapper &x) : P::Payload(x) {}
		PayloadWrapper& operator=(const PayloadWrapper &x); //non-copyable
	};
	class PacketWrapper;
	struct BranchPoint;
	//! A package containing \a Payload, subpackages, and a list of subnodes.
	struct Packet : public atomic_countable {
		Packet();
		int size() const {return subpackets() ? subpackets()->size() : 0;}
		local_shared_ptr<Payload> &payload() { return m_payload;}
		const local_shared_ptr<Payload> &payload() const { return m_payload;}
		shared_ptr<NodeList> &subnodes() { return subpackets()->m_subnodes;}
		shared_ptr<PacketList> &subpackets() { return m_subpackets;}
		const shared_ptr<NodeList> &subnodes() const { return subpackets()->m_subnodes;}
		const shared_ptr<PacketList> &subpackets() const { return m_subpackets;}

		//! points to the linked node.
		Node &node() {return payload()->node();}
		//! points to the linked node.
		const Node &node() const {return payload()->node();}

		void _print() const;
		bool missing() const { return m_missing;}

		bool checkConsistensy(const local_shared_ptr<Packet> &rootpacket) const;

		//! Generates a serial number for bundling or transaction.
		static int64_t newSerial() {
			for(;;) {
				int64_t oldserial;
				int64_t newserial;
				oldserial = s_serial;
				newserial = oldserial + 1;
				if(newserial == SERIAL_NULL) newserial++;
				if(s_serial.compareAndSet(oldserial, newserial))
					return newserial;
			}
		}
		enum {SERIAL_NULL = 0, SERIAL_FIRST = 1, SERIAL_INIT = -1};
		local_shared_ptr<Payload> m_payload;
		shared_ptr<PacketList> m_subpackets;
		static atomic<int64_t> s_serial;
		//! indicates whether the bundle contains the up-to-date subpackets or not.
		bool m_missing;
	};
	struct PacketWrapper : public atomic_countable {
		PacketWrapper(const local_shared_ptr<Packet> &x);
		//! creates a wrapper not containing a packet but pointing to the upper node.
		//! \arg bp \a m_wrapper of the upper node.
		//! \arg reverse_index The index for this node in the list of the upper node.
		PacketWrapper(const shared_ptr<BranchPoint> &bp, int reverse_index);
		 ~PacketWrapper() {}
		bool hasPriority() const { return m_ridx == PACKET_HAS_PRIORITY; }
		const local_shared_ptr<Packet> &packet() const {return m_packet;}
		local_shared_ptr<Packet> &packet() {return m_packet;}

		shared_ptr<BranchPoint> branchpoint() const {return m_branchpoint.lock();}
		//! The index for this node in the list of the upper node.
		int reverseIndex() const {return m_ridx;}
		void setReverseIndex(int i) {m_ridx = i;}

		void _print() const;
		//! If a packet is absent at this node, it points to \a m_wrapper of the upper node.
		weak_ptr<BranchPoint> const m_branchpoint;
		local_shared_ptr<Packet> m_packet;
		int m_ridx;
		enum PACKET_STATE { PACKET_HAS_PRIORITY = -1};
	};
	struct BranchPoint : public atomic_shared_ptr<PacketWrapper> {
		BranchPoint() : atomic_shared_ptr<PacketWrapper>(), m_bundle_serial(Packet::SERIAL_INIT) {}
		atomic<int64_t> m_bundle_serial;
		atomic<uint64_t> m_transaction_started_time;
		inline void negotiate(uint64_t &started_time);
	};

	friend class Snapshot<XN>;
	friend class Transaction<XN>;
	void snapshot(Snapshot<XN> &target, bool multi_nodal,
		uint64_t &started_time) const;
	void snapshot(Transaction<XN> &target, bool multi_nodal) const {
		snapshot(static_cast<Snapshot<XN> &>(target), multi_nodal, target.m_started_time);
		target.m_oldpacket = target.m_packet;
	}
	enum SnapshotStatus {SNAPSHOT_SUCCESS, SNAPSHOT_DISTURBED,
		SNAPSHOT_NODE_MISSING, SNAPSHOT_VOID_PACKET, SNAPSHOT_COLLIDED};
	struct CASInfo {
		CASInfo(const shared_ptr<BranchPoint> &b, const local_shared_ptr<PacketWrapper> &o,
			const local_shared_ptr<PacketWrapper> &n) : branchpoint(b), old_wrapper(o), new_wrapper(n) {}
		shared_ptr<BranchPoint> branchpoint;
		local_shared_ptr<PacketWrapper> old_wrapper, new_wrapper;
	};
	static inline SnapshotStatus snapshotSupernode(const shared_ptr<BranchPoint> &branchpoint,
		local_shared_ptr<PacketWrapper> &shot, local_shared_ptr<Packet> **subpacket,
		bool make_unbundled_branch = false,
		int serial = Packet::SERIAL_NULL, std::deque<CASInfo> *cas_infos = 0);
	bool commit(Transaction<XN> &tr);

	enum BundledStatus {BUNDLE_SUCCESS, BUNDLE_DISTURBED};
	BundledStatus bundle(local_shared_ptr<PacketWrapper> &target,
		uint64_t &started_time, int64_t bundle_serial, bool is_bundle_root);
	BundledStatus bundle_subpacket(const shared_ptr<Node> &subnode,
		local_shared_ptr<PacketWrapper> &subwrapper, local_shared_ptr<Packet> &subpacket_new,
		uint64_t &started_time, int64_t bundle_serial);
	enum UnbundledStatus {UNBUNDLE_W_NEW_SUBVALUE,
		UNBUNDLE_SUBVALUE_HAS_CHANGED, UNBUNDLE_SUBVALUE_NOT_FOUND,
		UNBUNDLE_COLLIDED, UNBUNDLE_DISTURBED};
	//! Unloads a subpacket to \a subbranchpoint.
	//! it performs unbundling for all the super nodes.
	//! \arg bundle_serial If not zero, consistency/collision wil be checked.
	//! \arg nullwrapper The current value of \a subbranchpoint and should not contain \a packet().
	//! \arg oldsubpacket If not zero, the packet will be compared with the packet inside the super packet.
	//! \arg newsubwrapper If \a oldsubpacket and \a newsubwrapper are not zero, \a newsubwrapper will be a new value.
	//! If \a oldsubpacket is zero, unloaded value  of \a subbranchpoint will be substituted to \a newsubwrapper.
	static UnbundledStatus unbundle(const int64_t *bundle_serial, uint64_t &time_started,
		const shared_ptr<BranchPoint> &subbranchpoint, const local_shared_ptr<PacketWrapper> &nullwrapper,
		const local_shared_ptr<Packet> *oldsubpacket = NULL,
		local_shared_ptr<PacketWrapper> *newsubwrapper = NULL);
	//! The point where the packet is held.
	shared_ptr<BranchPoint> m_wrapper;

	//! finds the packet for this node in the (un)bundled \a packet.
	//! \arg superpacket The bundled packet.
	//! \arg copy_branch If ture, new packets and packet lists will be copy-created for writing.
	//! \arg tr_serial The serial number associated with the transaction.
	inline local_shared_ptr<Packet> *reverseLookup(local_shared_ptr<Packet> &superpacket,
		bool copy_branch, int tr_serial, bool set_missing, XN** uppernode);
	local_shared_ptr<Packet> &reverseLookup(local_shared_ptr<Packet> &superpacket,
		bool copy_branch, int tr_serial = 0, bool set_missing = false);
	const local_shared_ptr<Packet> &reverseLookup(const local_shared_ptr<Packet> &superpacket) const;
	inline static local_shared_ptr<Packet> *reverseLookupWithHint(shared_ptr<BranchPoint > &branchpoint,
		local_shared_ptr<Packet> &superpacket, bool copy_branch, int tr_serial, bool set_missing,
		local_shared_ptr<Packet> *upperpacket, int *index);
	//! finds this node and a corresponding packet in the (un)bundled \a packet.
	inline local_shared_ptr<Packet> *forwardLookup(local_shared_ptr<Packet> &superpacket,
		bool copy_branch, int tr_serial, bool set_missing,
		local_shared_ptr<Packet> *upperpacket, int *index) const;
	static void eraseBundleSerials(const local_shared_ptr<Packet> &packet);
	static void eraseTransactionSerials(local_shared_ptr<Packet> &packet, int64_t tr_serial);
	bool hasAnyBundleSerial(const local_shared_ptr<Packet> &packet);
	static void fetchSubpackets(std::deque<local_shared_ptr<PacketWrapper> >  &subwrappers,
		const local_shared_ptr<Packet> &packet);
protected:
	//! Use \a create().
	Node();
private:
	Node(const Node &); //non-copyable.
	Node &operator=(const Node &); //non-copyable.
	typedef Payload *(*FuncPayloadCreator)(XN &);
	static XThreadLocal<FuncPayloadCreator> stl_funcPayloadCreator;
};

template <class XN>
template <class T>
T *Node<XN>::create() {
	*T::stl_funcPayloadCreator = &PayloadWrapper<T>::funcPayloadCreator;
	return new T();
}
template <class XN>
template <class T, typename A1>
T *Node<XN>::create(A1 a1) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<T>::funcPayloadCreator;
	return new T(a1);
}
template <class XN>
template <class T, typename A1, typename A2>
T *Node<XN>::create(A1 a1, A2 a2) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<T>::funcPayloadCreator;
	return new T(a1, a2);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<T>::funcPayloadCreator;
	return new T(a1, a2, a3);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3, typename A4>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3, A4 a4) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<T>::funcPayloadCreator;
	return new T(a1, a2, a3, a4);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3, typename A4, typename A5>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<T>::funcPayloadCreator;
	return new T(a1, a2, a3, a4, a5);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<T>::funcPayloadCreator;
	return new T(a1, a2, a3, a4, a5, a6);
}
template <class XN>
template <class T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
T *Node<XN>::create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7) {
	*T::stl_funcPayloadCreator = &PayloadWrapper<T>::funcPayloadCreator;
	return new T(a1, a2, a3, a4, a5, a6, a7);
}

//! This class takes a snapshot for a monitored data set.
template <class XN>
class Snapshot {
public:
	Snapshot(const Snapshot&x) : m_packet(x.m_packet), m_serial(x.m_serial) {
	}
	Snapshot(const Transaction<XN>&x);
	explicit Snapshot(const Node<XN>&node, bool multi_nodal = true) {
		m_serial = Node<XN>::Packet::newSerial();
		XTime time(XTime::now());
		uint64_t ms = (uint64_t)time.sec() * 1000u + time.usec() / 1000u;
		node.snapshot(*this, multi_nodal, ms);
	}
	explicit Snapshot(const local_shared_ptr<typename Node<XN>::Packet> &packet) :
		m_packet(packet) {
		m_serial = Node<XN>::Packet::newSerial();
	}
	virtual ~Snapshot() {}

	template <class T>
	const typename T::Payload &operator[](const shared_ptr<T> &node) const {
		return operator[](const_cast<const T&>(*node));
	}
	template <class T>
	const typename T::Payload &operator[](const T &node) const {
		const local_shared_ptr<typename Node<XN>::Packet> &packet(node.reverseLookup(m_packet));
		const local_shared_ptr<typename Node<XN>::Payload> &payload(packet->payload());
		return *static_cast<const typename T::Payload*>(payload.get());
	}
	int size() const {return m_packet->size();}
	const shared_ptr<const typename Node<XN>::NodeList> list() const {
		if( !size())
			return shared_ptr<typename Node<XN>::NodeList>();
		return m_packet->subnodes();
	}
	int size(const shared_ptr<Node<XN> > &node) const {
		return node->reverseLookup(m_packet)->size();
	}
	shared_ptr<const typename Node<XN>::NodeList> list(const shared_ptr<Node<XN> > &node) const {
		local_shared_ptr<typename Node<XN>::Packet> const &packet(node->reverseLookup(m_packet));
		if( !packet->size() )
			return shared_ptr<typename Node<XN>::NodeList>();
		return packet->subnodes();
	}
	void print() {
		m_packet->_print();
	}

	template <typename T, typename tArgRef>
	void talk(T &talker, tArgRef arg) const { talker.talk(*this, arg); }
protected:
	friend class Node<XN>;
	//! The snapshot.
	local_shared_ptr<typename Node<XN>::Packet> m_packet;
	int64_t m_serial;
	Snapshot() : m_packet() {
		m_serial = Node<XN>::Packet::newSerial();
	}
};

template <class XN, typename T>
class SingleSnapshot : protected Snapshot<XN> {
public:
	explicit SingleSnapshot(const T&node) : Snapshot<XN>(node, false) {}
	virtual ~SingleSnapshot() {}

	const typename T::Payload *operator->() const {
		return static_cast<const typename T::Payload *>(this->m_packet->payload().get());
	}
	template <class X>
	operator X() const {
		return (X)( *operator->());
	}
protected:
};

}
#include "transaction_signal.h"
namespace Transactional {

//! Transactional writing for a monitored data set.
//! The revision will be committed implicitly on leaving the scope.
template <class XN>
class Transaction : public Snapshot<XN> {
public:
	//! Be sure to the persistence of the \a node.
	explicit Transaction(Node<XN>&node, bool multi_nodal = true) :
		Snapshot<XN>(), m_oldpacket(), m_multi_nodal(multi_nodal) {
		XTime time(XTime::now());
		m_started_time = (uint64_t)time.sec() * 1000u + time.usec() / 1000u;
		node.snapshot(*this, multi_nodal);
		ASSERT(&this->m_packet->node() == &node);
		ASSERT(&this->m_oldpacket->node() == &node);
	}
	explicit Transaction(const Snapshot<XN> &x, bool multi_nodal = true) : Snapshot<XN>(x),
		m_oldpacket(this->m_packet), m_multi_nodal(multi_nodal) {
		XTime time(XTime::now());
		m_started_time = (uint64_t)time.sec() * 1000u + time.usec() / 1000u;
		this->m_serial = Node<XN>::Packet::newSerial();
	}
//	Transaction(const Transaction &x) : Snapshot<XN>(x),
//		m_oldpacket(x.m_oldpacket), m_serial(x.m_serial), m_multi_nodal(x.m_multi_nodal),
//		m_started_time(x.m_started_time), m_messages() {}
	virtual ~Transaction() {
		//Do not leave the time stamp.
		if(m_started_time) {
			Node<XN> &node(this->m_packet->node());
			if(node.m_wrapper->m_transaction_started_time >= m_started_time) {
				node.m_wrapper->m_transaction_started_time = 0;
			}
		}
	}
	//! Explicitly commits.
	bool commit() {
		Node<XN> &node(this->m_packet->node());
		if( !isModified() || node.commit(*this)) {
			finalizeCommitment();
			return true;
		}
		return false;
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
		Node<XN> &node(this->m_packet->node());
		if(isMultiNodal()) {
			uint64_t time(node.m_wrapper->m_transaction_started_time);
			if( !time || (time > m_started_time))
				node.m_wrapper->m_transaction_started_time = m_started_time;
		}
		m_messages.clear();
		this->m_serial = Node<XN>::Packet::newSerial();
		this->m_packet->node().snapshot(*this, m_multi_nodal);
		return *this;
	}

	template <class T>
	typename T::Payload &operator[](const shared_ptr<T> &node) {
		return operator[](*node);
	}
	template <class T>
	typename T::Payload &operator[](T &node) {
		local_shared_ptr<typename Node<XN>::Payload> &payload(
			node.reverseLookup(this->m_packet, true, this->m_serial)->payload());
		if(payload->m_serial != this->m_serial) {
			payload.reset(payload->clone(*this, this->m_serial));
			typename T::Payload &p( *static_cast<typename T::Payload *>(payload.get()));
			return p;
		}
		typename T::Payload &p( *static_cast<typename T::Payload *>(payload.get()));
		return p;
	}
	bool isMultiNodal() const {return m_multi_nodal;}

	template <typename T, typename tArgRef>
	void mark(T &talker, tArgRef arg) {
		_Message<XN> *m = talker.message(arg);
		if(m)
			m_messages.push_back(shared_ptr<_Message<XN> >(m));
	}
private:
	Transaction(const Transaction &tr); //non-copyable.
	Transaction& operator=(const Transaction &tr); //non-copyable.
	friend class Node<XN>;
	void finalizeCommitment() {
		//Clears the time stamp linked to this object.
		Node<XN> &node(this->m_packet->node());
		if(node.m_wrapper->m_transaction_started_time >= m_started_time) {
			node.m_wrapper->m_transaction_started_time = 0;
		}
		m_started_time = 0;

		//Messaging.
		if(m_messages.size()) {
			for(typename MessageList::iterator it = m_messages.begin(); it != m_messages.end(); ++it) {
				(*it)->talk(*this);
			}
		}
		m_messages.clear();
	}

	local_shared_ptr<typename Node<XN>::Packet> m_oldpacket;
	const bool m_multi_nodal;
	uint64_t m_started_time;
	typedef std::deque<shared_ptr<_Message<XN> > > MessageList;
	MessageList m_messages;
};

template <class XN, typename T>
class SingleTransaction : public Transaction<XN> {
public:
	explicit SingleTransaction(T&node) : Transaction<XN>(node, false) {}
	virtual ~SingleTransaction() {}

	typename T::Payload &operator*() {
		return (*this)[static_cast<T&>(this->m_packet->node())];
	}
	typename T::Payload *operator->() {
		return &(**this);
	}
	template <class X>
	operator X() const {
		return (X)( *static_cast<const typename T::Payload *>(this->m_packet->payload().get()));
	}
protected:
};

template <class XN>
inline Snapshot<XN>::Snapshot(const Transaction<XN>&x) : m_packet(x.m_packet), m_serial(x.m_serial) {}

} //namespace Transactional

#endif /*TRANSACTION_H*/
