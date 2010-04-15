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

namespace Transactional {

//! \desc
//! Lock-free software transactional memory for treed objects. \n
//! The Transaction supports transactional accesses of data (implemented as Payload or derived classes), and\n
//! multiple insertion (hard-link)/removal (unlink) of objects to the tree. \n
//! Transaction/snapshot of subtree can be taken at any node at any time.\n
//! \n
//! Example 1 for snapshot reading.\n
//! { Snapshot<NodeA> shot1(node1); \n
//! double x = shot1[node1]; //implicit conversion defined in NodeA::Payload.\n
//! double y = shot1[node1].y();\n}\n
//!\n
//! Example 2 for simple writing.\n
//! for(Transaction<NodeA> tr1(node1);; ++tr1) { \n
//! tr1[node1] = tr1[node1] * 2.0;\n
//! if(tr1.commit()) break;\n}\n
//! \n
//! Example 3 for adding a child node.\n
//! for(Transaction<NodeA> tr1(node1);; ++tr1) { \n
//! if(tr1.insert(node2)) break;\n
//! if(tr1.commit()) break;\n}\n
//! \n
//! Other examples are shown in the test codes: transaction_test.cpp, transaction_dynamic_node_test.cpp,
//! transaction_negotiation_test.cpp.\n

template <class XN>
class Snapshot;
template <class XN>
class Transaction;

//! A basis of nodes which carries data sets for itself and for subnodes.\n
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

	typedef std::domain_error NodeNotFoundError;

	//! adds hard link to \a var.
	//! \arg online_after_insertion If true, \a var can be accessed through \a tr (not appropriate for a shared object).
	bool insert(Transaction<XN> &tr, const shared_ptr<XN> &var, bool online_after_insertion = false);
	void insert(const shared_ptr<XN> &var);
	bool release(Transaction<XN> &tr, const shared_ptr<XN> &var);
	void release(const shared_ptr<XN> &var);
	void releaseAll();
	//! swaps orders in the child list.
	bool swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y);
	void swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y);

	//! finds a parent in \a shot.
	XN *upperNode(Snapshot<XN> &shot);

	//! Data holder and accessor for the node.
	//! Re-implement members in its subclasses.
	//! The instances have to be capable of copy-construction and be safe to be shared reading.
	struct Payload : public atomic_countable {
		Payload() : m_node(0), m_serial(-1), m_tr(0) {}
		virtual ~Payload() {}

		//! Points to the corresponding node.
		XN &node() {return *m_node;}
		//! Points to the corresponding node.
		const XN &node() const {return *m_node;}
		int64_t serial() const {return this->m_serial;}
		Transaction<XN> &tr() { return *this->m_tr;}

		virtual void catchEvent(const shared_ptr<XN>&, int) {}
		virtual void releaseEvent(const shared_ptr<XN>&, int) {}
		virtual void moveEvent(unsigned int src_idx, unsigned int dst_idx) {}
		virtual void listChangeEvent() {}
	private:
		friend class Node;
		friend class Transaction<XN>;
		virtual Payload *clone(Transaction<XN> &tr, int64_t serial) = 0;

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
			PayloadWrapper *p = new PayloadWrapper( *this);
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
	struct Linkage;
	//! A package containing \a Payload, sub-Packets, and a list of subnodes.\n
	//! Not-"missing" packet is always up-to-date and can be a snapshot of the subtree,
	//! and packets possessed by the sub-instances may be out-of-date.\n
	//! "missing" indicates that the package lacks any Packet for subnodes, or
	//! any content may be out-of-date.\n
	struct Packet : public atomic_countable {
		Packet();
		int size() const {return subpackets() ? subpackets()->size() : 0;}
		local_shared_ptr<Payload> &payload() { return m_payload;}
		const local_shared_ptr<Payload> &payload() const { return m_payload;}
		shared_ptr<NodeList> &subnodes() { return subpackets()->m_subnodes;}
		shared_ptr<PacketList> &subpackets() { return m_subpackets;}
		const shared_ptr<NodeList> &subnodes() const { return subpackets()->m_subnodes;}
		const shared_ptr<PacketList> &subpackets() const { return m_subpackets;}

		//! Points to the corresponding node.
		Node &node() {return payload()->node();}
		//! Points to the corresponding node.
		const Node &node() const {return payload()->node();}

		void _print() const;
		bool missing() const { return m_missing;}

		bool checkConsistensy(const local_shared_ptr<Packet> &rootpacket) const;

		//! Generates a serial number assigned to bundling and transaction.\n
		//! During a transaction, a serial is used for determining whether Payload or PacketList should be cloned.\n
		//! During bundle(), it is used to prevent infinite loops due to unbundle()-ing itself.
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
	//! A class wrapping Packet and providing indice and links for lookup.\n
	//! If packet() is absent, a super node should have the up-to-date Packet.\n
	//! If hasPriority() is not set, Packet is a super node may be latest.
	struct PacketWrapper : public atomic_countable {
		PacketWrapper(const local_shared_ptr<Packet> &x, int64_t bundle_serial);
		//! creates a wrapper not containing a packet but pointing to the upper node.
		//! \arg bp \a m_link of the upper node.
		//! \arg reverse_index The index for this node in the list of the upper node.
		PacketWrapper(const shared_ptr<Linkage> &bp, int reverse_index, int64_t bundle_serial);
		PacketWrapper(const PacketWrapper &x, int64_t bundle_serial);
		 ~PacketWrapper() {}
		bool hasPriority() const { return m_ridx == PACKET_HAS_PRIORITY; }
		const local_shared_ptr<Packet> &packet() const {return m_packet;}
		local_shared_ptr<Packet> &packet() {return m_packet;}

		//! Points to the upper node that should have the up-to-date Packet when this lacks priority.
		shared_ptr<Linkage> linkedBy() const {return m_linkedBy.lock();}
		//! The index for this node in the list of the upper node.
		int reverseIndex() const {return m_ridx;}
		void setReverseIndex(int i) {m_ridx = i;}

		void _print() const;
		weak_ptr<Linkage> const m_linkedBy;
		local_shared_ptr<Packet> m_packet;
		int m_ridx;
		int64_t m_bundle_serial;
		enum PACKET_STATE { PACKET_HAS_PRIORITY = -1};
	private:
		PacketWrapper(const PacketWrapper &);
	};
	struct Linkage : public atomic_shared_ptr<PacketWrapper> {
		Linkage() : atomic_shared_ptr<PacketWrapper>(), m_transaction_started_time(0) {}
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
	enum SnapshotStatus {SNAPSHOT_SUCCESS = 0, SNAPSHOT_DISTURBED = 1,
		SNAPSHOT_VOID_PACKET = 2, SNAPSHOT_NODE_MISSING = 4,
		SNAPSHOT_COLLIDED = 8, SNAPSHOT_NODE_MISSING_AND_COLLIDED = 12};
	struct CASInfo {
		CASInfo(const shared_ptr<Linkage> &b, const local_shared_ptr<PacketWrapper> &o,
			const local_shared_ptr<PacketWrapper> &n) : linkage(b), old_wrapper(o), new_wrapper(n) {}
		shared_ptr<Linkage> linkage;
		local_shared_ptr<PacketWrapper> old_wrapper, new_wrapper;
	};
	enum SnapshotMode {SNAPSHOT_FOR_UNBUNDLE, SNAPSHOT_FOR_BUNDLE};
	static inline SnapshotStatus snapshotSupernode(const shared_ptr<Linkage> &linkage,
		local_shared_ptr<PacketWrapper> &shot, local_shared_ptr<Packet> **subpacket,
		SnapshotMode mode,
		int64_t serial = Packet::SERIAL_NULL, std::deque<CASInfo> *cas_infos = 0);

	bool commit(Transaction<XN> &tr);
//	bool commit_at_super(Transaction<XN> &tr);

	enum BundledStatus {BUNDLE_SUCCESS, BUNDLE_DISTURBED};
	//! Takes a snapshot.
	BundledStatus bundle(local_shared_ptr<PacketWrapper> &target,
		uint64_t &started_time, int64_t bundle_serial, bool is_bundle_root);
	BundledStatus bundle_subpacket(local_shared_ptr<PacketWrapper> *superwrapper, const shared_ptr<Node> &subnode,
		local_shared_ptr<PacketWrapper> &subwrapper, local_shared_ptr<Packet> &subpacket_new,
		uint64_t &started_time, int64_t bundle_serial);
	enum UnbundledStatus {UNBUNDLE_W_NEW_SUBVALUE,
		UNBUNDLE_SUBVALUE_HAS_CHANGED,
		UNBUNDLE_COLLIDED, UNBUNDLE_DISTURBED};
	//! Unloads a subpacket to \a sublinkage from a snapshot.
	//! it performs unbundling for all the super nodes.
	//! \arg bundle_serial If not zero, consistency/collision wil be checked.
	//! \arg null_linkage The current value of \a sublinkage and should not contain \a packet().
	//! \arg oldsubpacket If not zero, the packet will be compared with the packet inside the super packet.
	//! \arg newsubwrapper If \a oldsubpacket and \a newsubwrapper are not zero, \a newsubwrapper will be a new value.
	//! If \a oldsubpacket is zero, unloaded value  of \a sublinkage will be substituted to \a newsubwrapper.
	static UnbundledStatus unbundle(const int64_t *bundle_serial, uint64_t &time_started,
		const shared_ptr<Linkage> &sublinkage, const local_shared_ptr<PacketWrapper> &null_linkage,
		const local_shared_ptr<Packet> *oldsubpacket = NULL,
		local_shared_ptr<PacketWrapper> *newsubwrapper = NULL,
		local_shared_ptr<PacketWrapper> *superwrapper = NULL);
	//! The point where the packet is held.
	shared_ptr<Linkage> m_link;

	//! finds the packet for this node in the (un)bundled \a packet.
	//! \arg superpacket The bundled packet.
	//! \arg copy_branch If ture, new packets and packet lists will be copy-created for writing.
	//! \arg tr_serial The serial number associated with the transaction.
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
	static void fetchSubpackets(std::deque<local_shared_ptr<PacketWrapper> >  &subwrappers,
		const local_shared_ptr<Packet> &packet);
	static void eraseSerials(local_shared_ptr<Packet> &packet, int64_t serial);
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

//! This class takes a snapshot for a subtree.
template <class XN>
class Snapshot {
public:
	Snapshot(const Snapshot&x) : m_packet(x.m_packet), m_serial(x.m_serial) {
	}
	Snapshot(Node<XN> &node, const Snapshot &x) : m_packet(node.reverseLookup(x.m_packet)), m_serial(x.m_serial) {
	}
	Snapshot(const Transaction<XN>&x);
	explicit Snapshot(const Node<XN>&node, bool multi_nodal = true) {
		XTime time(XTime::now());
		uint64_t ms = (uint64_t)time.sec() * 1000u + time.usec() / 1000u;
		node.snapshot( *this, multi_nodal, ms);
	}
	virtual ~Snapshot() {}

	//! \return A Payload instance for \a node, which should be included in the snapshot.
	template <class T>
	const typename T::Payload &operator[](const shared_ptr<T> &node) const {
		return operator[](const_cast<const T&>( *node));
	}
	//! \return A Payload instance for \a node, which should be included in the snapshot.
	template <class T>
	const typename T::Payload &operator[](const T &node) const {
		const local_shared_ptr<typename Node<XN>::Packet> &packet(node.reverseLookup(m_packet));
		const local_shared_ptr<typename Node<XN>::Payload> &payload(packet->payload());
		return *static_cast<const typename T::Payload*>(payload.get());
	}
	//! # of child nodes.
	int size() const {return m_packet->size();}
	//! The list of child nodes.
	const shared_ptr<const typename Node<XN>::NodeList> list() const {
		if( !size())
			return shared_ptr<typename Node<XN>::NodeList>();
		return m_packet->subnodes();
	}
	//! # of child nodes owned by \a node.
	int size(const shared_ptr<Node<XN> > &node) const {
		return node->reverseLookup(m_packet)->size();
	}
	//! The list of child nodes owned by \a node.
	shared_ptr<const typename Node<XN>::NodeList> list(const shared_ptr<Node<XN> > &node) const {
		local_shared_ptr<typename Node<XN>::Packet> const &packet(node->reverseLookup(m_packet));
		if( !packet->size() )
			return shared_ptr<typename Node<XN>::NodeList>();
		return packet->subnodes();
	}
	//! Whether \a lower is a child of this or not.
	bool isUpperOf(const XN &lower) const {
		const shared_ptr<const typename Node<XN>::NodeList> _list(list());
		if( !_list)
			return false;
		for(typename Node<XN>::NodeList::const_iterator it = _list->begin(); it != _list->end(); ++it) {
			if(it->get() == &lower)
				return true;
		}
		return false;
	}

	void print() {
		m_packet->_print();
	}

	//! Sends an event from \a talker with \a arg.
	template <typename T, typename tArgRef>
	void talk(T &talker, tArgRef arg) const { talker.talk( *this, arg); }
protected:
	friend class Node<XN>;
	//! The snapshot.
	local_shared_ptr<typename Node<XN>::Packet> m_packet;
	int64_t m_serial;
	Snapshot() : m_packet() {}
};

template <class XN, typename T>
class SingleSnapshot : protected Snapshot<XN> {
public:
	explicit SingleSnapshot(const T &node) : Snapshot<XN>(node, false) {}
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

//! Transactional writing for a subtree.
template <class XN>
class Transaction : public Snapshot<XN> {
public:
	//! Be sure for the persistence of the \a node.
	//! \arg multi_nodal If false, the snapshot and following commitment are not aware of the contents of the child nodes.
	explicit Transaction(Node<XN>&node, bool multi_nodal = true) :
		Snapshot<XN>(), m_oldpacket(), m_multi_nodal(multi_nodal) {
		XTime time(XTime::now());
		m_started_time = (uint64_t)time.sec() * 1000u + time.usec() / 1000u;
		node.snapshot( *this, multi_nodal);
		ASSERT( &this->m_packet->node() == &node);
		ASSERT( &this->m_oldpacket->node() == &node);
	}
	//! \arg x The snapshot containing the old value of \a node.
	//! \arg multi_nodal If false, the snapshot and following commitment are not aware of the contents of the child nodes.
	explicit Transaction(const Snapshot<XN> &x, bool multi_nodal = true) : Snapshot<XN>(x),
		m_oldpacket(this->m_packet), m_multi_nodal(multi_nodal) {
		XTime time(XTime::now());
		m_started_time = (uint64_t)time.sec() * 1000u + time.usec() / 1000u;
	}
	//! Prepares a transaction for a subtree beneath \a node.
	//! \arg x The snapshot containing the old value of \a node.
	Transaction(Node<XN> &node, const Snapshot<XN> &x) :
		Snapshot<XN>(node, x),
		m_oldpacket(this->m_packet), m_multi_nodal(true) {
		XTime time(XTime::now());
		m_started_time = (uint64_t)time.sec() * 1000u + time.usec() / 1000u;
	}
	virtual ~Transaction() {
		//Do not leave the time stamp.
		if(m_started_time) {
			Node<XN> &node(this->m_packet->node());
			if(node.m_link->m_transaction_started_time >= m_started_time) {
				node.m_link->m_transaction_started_time = 0;
			}
		}
	}
	//! \return True if succeeded.
	bool commit() {
		Node<XN> &node(this->m_packet->node());
		if( !isModified() || node.commit( *this)) {
			finalizeCommitment(node);
			return true;
		}
		return false;
	}
//	bool commitAt(Node<XN> &supernode) {
//		if(supernode.commit_at_super( *this)) {
//			finalizeCommitment(this->m_packet->node());
//			return true;
//		}
//		return false;
//	}
	//! Combination of commit() and operator++().
	bool commitOrNext() {
		if(commit())
			return true;
		++( *this);
		return false;
	}
	bool isModified() const {
		return (this->m_packet != this->m_oldpacket);
	}
	//! Takes another snapshot and prepares for a next transaction.
	Transaction &operator++() {
		Node<XN> &node(this->m_packet->node());
		if(isMultiNodal()) {
			uint64_t time(node.m_link->m_transaction_started_time);
			if( !time || (time > m_started_time))
				node.m_link->m_transaction_started_time = m_started_time;
		}
		m_messages.clear();
		this->m_packet->node().snapshot( *this, m_multi_nodal);
		return *this;
	}

	//! \return A copy-constructed Payload instance for \a node, which should be included in the transaction.
	template <class T>
	typename T::Payload &operator[](const shared_ptr<T> &node) {
		return operator[]( *node);
	}
	//! \return A copy-constructed Payload instance for \a node, which should be included in the transaction.
	template <class T>
	typename T::Payload &operator[](T &node) {
		ASSERT(isMultiNodal() || ( &node == &this->m_packet->node()));
		local_shared_ptr<typename Node<XN>::Payload> &payload(
			node.reverseLookup(this->m_packet, true, this->m_serial)->payload());
		if(payload->m_serial != this->m_serial) {
			payload.reset(payload->clone( *this, this->m_serial));
			typename T::Payload &p( *static_cast<typename T::Payload *>(payload.get()));
			return p;
		}
		typename T::Payload &p( *static_cast<typename T::Payload *>(payload.get()));
		return p;
	}
	bool isMultiNodal() const {return m_multi_nodal;}

	//! Reserves an event, to be emitted from \a talker with \a arg.
	template <typename T, typename tArgRef>
	void mark(T &talker, tArgRef arg) {
		_Message<XN> *m = talker.createMessage(arg);
		if(m)
			m_messages.push_back(shared_ptr<_Message<XN> >(m));
	}
	//! Cancels events made toward \a x.
	void unmark(const shared_ptr<XListener> &x) {
		for(typename MessageList::iterator it = m_messages.begin(); it != m_messages.end(); ++it)
			( *it)->unmark(x);
	}
private:
	Transaction(const Transaction &tr); //non-copyable.
	Transaction& operator=(const Transaction &tr); //non-copyable.
	friend class Node<XN>;
	void finalizeCommitment(Node<XN> &node) {
		//Clears the time stamp linked to this object.
		if(node.m_link->m_transaction_started_time >= m_started_time) {
			node.m_link->m_transaction_started_time = 0;
		}
		m_started_time = 0;

		m_oldpacket.reset();
		//Messaging.
		if(m_messages.size()) {
			for(typename MessageList::iterator it = m_messages.begin(); it != m_messages.end(); ++it) {
				( *it)->talk( *this);
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
	explicit SingleTransaction(T &node) : Transaction<XN>(node, false) {}
	virtual ~SingleTransaction() {}

	typename T::Payload &operator*() {
		return ( *this)[static_cast<T &>(this->m_packet->node())];
	}
	typename T::Payload *operator->() {
		return &( **this);
	}
	template <class X>
	operator X() const {
		return (X)( *static_cast<const typename T::Payload *>(this->m_packet->payload().get()));
	}
protected:
};

template <class XN>
inline Snapshot<XN>::Snapshot(const Transaction<XN>&x) :
m_packet(x.m_packet), m_serial(x.m_serial) {}

} //namespace Transactional

#endif /*TRANSACTION_H*/
