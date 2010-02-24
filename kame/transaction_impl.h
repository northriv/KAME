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
#include "transaction.h"
#include <vector>

#ifdef TRANSACTIONAL_STRICT_ASSERT
	#undef STRICT_ASSERT
	#define STRICT_ASSERT(expr) ASSERT(expr)
	#define STRICT_TEST(expr) expr
#else
	#define STRICT_ASSERT(expr)
	#define STRICT_TEST(expr)
#endif

namespace Transactional {

template <class XN>
XThreadLocal<typename Node<XN>::FuncPayloadCreator> Node<XN>::stl_funcPayloadCreator;

template <class XN>
atomic<int64_t> Node<XN>::Packet::s_serial = 0;

template <class XN>
Node<XN>::Packet::Packet() : m_missing(false) {}

template <class XN>
void
Node<XN>::Packet::_print() const {
	printf("Packet: ");
	printf("node:%llx, ", (unsigned long long)(uintptr_t)&node());
	printf("branchpoint:%llx, ", (unsigned long long)(uintptr_t)node().m_wrapper.get());
	if(missing())
		printf("missing, ");
	if(size()) {
		printf("%d subnodes : [ \n", (int)size());
		for(int i = 0; i < size(); i++) {
			if(subpackets()->at(i)) {
				subpackets()->at(i)->_print();
				printf("; ");
			}
			else {
				printf("node:%llx w/o packet; ", (unsigned long long)(uintptr_t)subnodes()->at(i).get());
			}
		}
		printf("]\n");
	}
	printf(";");
}

template <class XN>
bool
Node<XN>::Packet::checkConsistensy(const local_shared_ptr<Packet> &rootpacket) const {
	try {
		if(size()) {
			if( !(payload()->m_serial - subpackets()->m_serial < 0x7fffffffffffffffLL))
				throw __LINE__;
		}
		for(int i = 0; i < size(); i++) {
			if( !subpackets()->at(i)) {
				if( !rootpacket->missing()) {
					if( !subnodes()->at(i)->reverseLookup(
						const_cast<local_shared_ptr<Packet>&>(rootpacket), false, 0, false, 0))
						throw __LINE__;
				}
			}
			else {
				if(subpackets()->at(i)->size())
					if( !(subpackets()->m_serial - subpackets()->at(i)->subpackets()->m_serial < 0x7fffffffffffffffLL))
						throw __LINE__;
				if(subpackets()->at(i)->missing() && (rootpacket.get() != this)) {
					if( !missing())
						throw __LINE__;
				}
				if( !subpackets()->at(i)->checkConsistensy(
					subpackets()->at(i)->missing() ? rootpacket : subpackets()->at(i)))
					return false;
			}
		}
	}
	catch (int &line) {
		fprintf(stderr, "Line %d, losing consistensy on node %llx:\n", line, (unsigned long long)&node());
		rootpacket->_print();
		throw *this;
	}
	return true;
}

template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const local_shared_ptr<Packet> &x) :
	m_branchpoint(), m_packet(x), m_ridx(PACKET_HAS_PRIORITY) {
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const shared_ptr<BranchPoint > &bp, int reverse_index) :
	m_branchpoint(bp), m_packet(), m_ridx() {
	setReverseIndex(reverse_index);
}

template <class XN>
void
Node<XN>::PacketWrapper::_print() const {
	printf("PacketWrapper: ");
	if( !hasPriority()) {
		printf("referred to :%llx, ", (unsigned long long)(uintptr_t)branchpoint().get());
	}
	if(packet()) {
		packet()->_print();
	}
	else {
		printf("absent, ");
	}
	printf("\n");
}

template <class XN>
inline void
Node<XN>::BranchPoint::negotiate(uint64_t &started_time) {
	int64_t transaction_started_time = m_transaction_started_time;
	if(transaction_started_time) {
		int ms = ((int64_t)started_time - transaction_started_time);
		if(ms > 0) {
			XTime t0 = XTime::now();
			t0 += ms * 1e-3;
			while(t0 > XTime::now()) {
//				usleep(std::min(ms * 1000 / 50, 1000));
				msecsleep(1);
				if( !m_transaction_started_time)
					break;
			}
			started_time -= XTime::now().diff_msec(t0) + ms;
		}
	}
}

template <class XN>
Node<XN>::Node() : m_wrapper(new BranchPoint()), m_transaction_count(0) {
	local_shared_ptr<Packet> packet(new Packet());
	m_wrapper->reset(new PacketWrapper(packet));
	//Use create() for this hack.
	packet->m_payload.reset(( *stl_funcPayloadCreator)(static_cast<XN&>( *this)));
	*stl_funcPayloadCreator = NULL;
}
template <class XN>
Node<XN>::~Node() {
	releaseAll();
}
template <class XN>
void
Node<XN>::_print() const {
	local_shared_ptr<PacketWrapper> packet( *m_wrapper);
	printf("Local packet: ");
	packet->_print();
}

template <class XN>
void
Node<XN>::insert(const shared_ptr<XN> &var) {
	for(Transaction<XN> tr( *this);; ++tr) {
		if( !insert(tr, var))
			continue;
		if(tr.commit())
			break;
	}
}
template <class XN>
bool
Node<XN>::insert(Transaction<XN> &tr, const shared_ptr<XN> &var, bool online_after_insertion) {
	local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
	packet->subpackets().reset(packet->size() ? (new PacketList( *packet->subpackets())) : (new PacketList));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->m_missing = true;
	packet->subnodes().reset(packet->size() ? (new NodeList( *packet->subnodes())) : (new NodeList));
	packet->subpackets()->resize(packet->size() + 1);
	ASSERT(packet->subnodes());
	ASSERT(std::find(packet->subnodes()->begin(), packet->subnodes()->end(), var) == packet->subnodes()->end());
	packet->subnodes()->push_back(var);
	ASSERT(packet->subpackets()->size() == packet->subnodes()->size());

	for(;;) {
		local_shared_ptr<Packet> subpacket_new;
		local_shared_ptr<PacketWrapper> subwrapper;
		BundledStatus status = bundle_subpacket(var, subwrapper, subpacket_new, tr.m_started_time, tr.m_serial);
		if(status != BUNDLE_SUCCESS) {
			continue;
		}
		if( !subpacket_new)
			//Inserted twice inside the package.
			break;

		bool has_failed = false;
		//Marks for writing at subnode.
		local_shared_ptr<Packet> newpacket(tr.m_packet);
		tr.m_packet.reset(new Packet( *tr.m_oldpacket));
		if( !tr.m_packet->node().commit(tr)) {
			has_failed = true;
		}
		tr.m_oldpacket = tr.m_packet;
		tr.m_packet = newpacket;

		local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(m_wrapper, packet->size() - 1));
		if(online_after_insertion) {
			packet->subpackets()->back() = subpacket_new;
		}
		else {
			newwrapper->packet() = subpacket_new;
		}
		if(has_failed)
			return false;
		if( !var->m_wrapper->compareAndSet(subwrapper, newwrapper)) {
			tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
			return false;
		}
		break;
	}
	tr[ *this].catchEvent(var, packet->size() - 1);
	tr[ *this].listChangeEvent();
	STRICT_ASSERT(tr.m_packet->checkConsistensy(tr.m_packet));
	return true;
//		printf("i");
}
template <class XN>
void
Node<XN>::release(const shared_ptr<XN> &var) {
	for(Transaction<XN> tr( *this);; ++tr) {
		if( !release(tr, var))
			continue;
		if(tr.commit())
			break;
	}
}

template <class XN>
void
Node<XN>::eraseBundleSerials(const local_shared_ptr<Packet> &packet) {
	packet->node().m_wrapper->m_bundle_serial = Packet::SERIAL_NULL;
	for(int i = 0; i < packet->size(); ++i) {
		const local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
		if(subpacket)
			eraseBundleSerials(subpacket);
	}
}
//template <class XN>
//bool
//Node<XN>::hasAnyBundleSerial(const local_shared_ptr<Packet> &packet) {
//	if(m_wrapper->m_bundle_serial != Packet::SERIAL_NULL)
//		return true;
//	for(int i = 0; i < packet->size(); ++i) {
//		const local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
//		if(subpacket)
//			if(packet->subnodes()->at(i)->hasAnyBundleSerial(subpacket))
//				return true;
//	}
//	return false;
//}
template <class XN>
bool
Node<XN>::release(Transaction<XN> &tr, const shared_ptr<XN> &var) {
	local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
	ASSERT(packet->size());
	packet->subpackets().reset(new PacketList( *packet->subpackets()));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->subnodes().reset(new NodeList( *packet->subnodes()));
	unsigned int idx = 0;
	int old_idx = -1;
	local_shared_ptr<PacketWrapper> nullsubwrapper, newsubwrapper;
	typename NodeList::iterator nit = packet->subnodes()->begin();
	for(typename PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
		ASSERT(nit != packet->subnodes()->end());
		if(nit->get() == &*var) {
			if( *pit) {
				if( !( *pit)->size()) {
					pit->reset(new Packet( **pit));
				}
				nullsubwrapper = *var->m_wrapper;
				if(nullsubwrapper->hasPriority() || (nullsubwrapper->branchpoint() != m_wrapper)) {
					tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
					return false;
				}
				newsubwrapper.reset(new PacketWrapper(m_wrapper, idx));
				newsubwrapper->packet() = *pit;
			}
			pit = packet->subpackets()->erase(pit);
			nit = packet->subnodes()->erase(nit);
			old_idx = idx;
		}
		else {
			++nit;
			++pit;
			++idx;
		}
	}

	if( !packet->subpackets()->size()) {
		packet->subpackets().reset();
		packet->m_missing = false;
	}
	tr[ *this].releaseEvent(var, old_idx);
	tr[ *this].listChangeEvent();

	if( !nullsubwrapper) {
		//Packet of the released node is held by the other point inside the tr.m_packet.
		return true;
	}
	if(tr.m_packet->size()) {
		tr.m_packet->m_missing = true;
	}

	//Marks as missing.
	local_shared_ptr<Packet> newpacket(tr.m_packet);
	tr.m_packet = tr.m_oldpacket;
	local_shared_ptr<Packet> *p = var->reverseLookup(tr.m_packet, true, tr.m_serial, true, 0);
	if( !p)
		tr.m_packet.reset(new Packet( *tr.m_packet));
	if( !tr.m_packet->node().commit(tr)) {
		tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
		tr.m_packet = newpacket;
		return false;
	}
	tr.m_oldpacket = tr.m_packet;
	tr.m_packet = newpacket;

	m_wrapper->m_bundle_serial = Packet::SERIAL_NULL;
	eraseBundleSerials(newsubwrapper->packet());

//	if(var->hasAnyBundleSerial(newsubwrapper->packet()) || (m_wrapper->m_bundle_serial != Packet::SERIAL_NULL)) {
//		//Being bundled by another thread.
//		tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
//		return false;
//	}

	//Sets temporary packet on the released node.
	if( !var->m_wrapper->compareAndSet(nullsubwrapper, newsubwrapper)) {
		tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
		return false;
	}
//		printf("r");
	STRICT_ASSERT(tr.m_packet->checkConsistensy(tr.m_packet));
	return true;
}
template <class XN>
void
Node<XN>::releaseAll() {
	for(Transaction<XN> tr( *this);; ++tr) {
		bool failed = false;
		while(tr.size()) {
			shared_ptr<const NodeList> list(tr.list());
			if( !release(tr, list->front())) {
				failed = true;
				break;
			}
		}
		if( !failed && tr.commit())
			break;
	}
//	for(;;) {
//		Transaction<XN> tr( *this);
//		if( !tr.size())
//			break;
//		shared_ptr<const NodeList> list(tr.list());
//		if( !release(tr, list->front()))
//			continue;
//		if( !tr.commit())
//			continue;
//		if( !tr.size())
//			break;
//	}
}
template <class XN>
void
Node<XN>::swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
	for(Transaction<XN> tr( *this);; ++tr) {
		if( !swap(tr, x, y))
			continue;
		if(tr.commit())
			break;
	}
}
template <class XN>
bool
Node<XN>::swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
	local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
	packet->subpackets().reset(packet->size() ? (new PacketList( *packet->subpackets())) : (new PacketList));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->m_missing = true;
	packet->subnodes().reset(packet->size() ? (new NodeList( *packet->subnodes())) : (new NodeList));
	unsigned int idx = 0;
	int x_idx = -1, y_idx = -1;
	for(typename NodeList::iterator nit = packet->subnodes()->begin(); nit != packet->subnodes()->end(); ++nit) {
		if( *nit == x)
			x_idx = idx;
		if( *nit == y)
			y_idx = idx;
		++idx;
	}
	ASSERT(x_idx >= 0);
	ASSERT(y_idx >= 0);
	local_shared_ptr<Packet> px = packet->subpackets()->at(x_idx);
	local_shared_ptr<Packet> py = packet->subpackets()->at(y_idx);
	packet->subpackets()->at(x_idx) = py;
	packet->subpackets()->at(y_idx) = px;
	packet->subnodes()->at(x_idx) = y;
	packet->subnodes()->at(y_idx) = x;
	tr[ *this].moveEvent(x_idx, y_idx);
	tr[ *this].listChangeEvent();
	STRICT_ASSERT(tr.m_packet->checkConsistensy(tr.m_packet));
	return true;
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookupWithHint(shared_ptr<BranchPoint> &branchpoint,
	local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial, bool set_missing,
	local_shared_ptr<Packet> *superpacket, int *index) {
	if( !packet->size())
		return NULL;
	local_shared_ptr<PacketWrapper> wrapper( *branchpoint);
	if(wrapper->hasPriority())
		return NULL;
	shared_ptr<BranchPoint> branchpoint_super(wrapper->branchpoint());
	if( !branchpoint_super)
		return NULL;
	local_shared_ptr<Packet> *foundpacket;
	if(branchpoint_super == packet->node().m_wrapper)
		foundpacket = &packet;
	else {
		foundpacket = reverseLookupWithHint(branchpoint_super,
			packet, copy_branch, tr_serial, set_missing, NULL, NULL);
		if( !foundpacket)
			return NULL;
	}
	int ridx = wrapper->reverseIndex();
	if( !( *foundpacket)->size() || (ridx >= ( *foundpacket)->size()))
		return NULL;
	if(copy_branch) {
		if(( *foundpacket)->subpackets()->m_serial != tr_serial) {
			foundpacket->reset(new Packet( **foundpacket));
			( *foundpacket)->subpackets().reset(new PacketList( *( *foundpacket)->subpackets()));
			( *foundpacket)->m_missing = ( *foundpacket)->m_missing || set_missing;
			( *foundpacket)->subpackets()->m_serial = tr_serial;
		}
	}
	local_shared_ptr<Packet> &p(( *foundpacket)->subpackets()->at(ridx));
	if( !p || (p->node().m_wrapper != branchpoint)) {
		return NULL;
	}
	if(superpacket) {
		*superpacket = *foundpacket;
		*index = ridx;
	}
	return &p;
}

template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::forwardLookup(local_shared_ptr<Packet> &packet,
	bool copy_branch, int tr_serial, bool set_missing,
	local_shared_ptr<Packet> *superpacket, int *index) const {
	ASSERT(packet);
	if( !packet->subpackets())
		return NULL;
	if(copy_branch) {
		if(packet->subpackets()->m_serial != tr_serial) {
			packet.reset(new Packet( *packet));
			packet->subpackets().reset(new PacketList( *packet->subpackets()));
			packet->subpackets()->m_serial = tr_serial;
			packet->m_missing = packet->m_missing || set_missing;
		}
	}
	for(unsigned int i = 0; i < packet->subnodes()->size(); i++) {
		if(packet->subnodes()->at(i).get() == this) {
			local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
			if(subpacket) {
				*superpacket = packet;
				*index = i;
				return &subpacket;
			}
		}
	}
	for(unsigned int i = 0; i < packet->subnodes()->size(); i++) {
		local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
		if(subpacket) {
			if(local_shared_ptr<Packet> *p =
				forwardLookup(subpacket, copy_branch, tr_serial, set_missing, superpacket, index)) {
				return p;
			}
		}
	}
	return NULL;
}

template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookup(local_shared_ptr<Packet> &packet,
	bool copy_branch, int tr_serial, bool set_missing, XN **supernode) {
	local_shared_ptr<Packet> *foundpacket;
	if( &packet->node() == this) {
		foundpacket = &packet;
	}
	else {
		local_shared_ptr<Packet> superpacket;
		int index;
		foundpacket = reverseLookupWithHint(m_wrapper, packet,
			copy_branch, tr_serial, set_missing, &superpacket, &index);
		if(foundpacket) {
//				printf("$");
		}
		else {
//				printf("!");
			foundpacket = forwardLookup(packet, copy_branch, tr_serial, set_missing,
				&superpacket, &index);
			if( !foundpacket)
				return 0;
		}
		if(supernode)
			*supernode = static_cast<XN*>(&superpacket->node());
		ASSERT( &( *foundpacket)->node() == this);
	}
	if(copy_branch && (( *foundpacket)->payload()->m_serial != tr_serial)) {
		foundpacket->reset(new Packet( **foundpacket));
	}
//						printf("#");
	return foundpacket;
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet>&
Node<XN>::reverseLookup(local_shared_ptr<Packet> &packet,
	bool copy_branch, int tr_serial, bool set_missing) {
	local_shared_ptr<Packet> *foundpacket = reverseLookup(packet, copy_branch, tr_serial, set_missing, 0);
	ASSERT(foundpacket);
	return *foundpacket;
}

template <class XN>
const local_shared_ptr<typename Node<XN>::Packet> &
Node<XN>::reverseLookup(const local_shared_ptr<Packet> &packet) const {
	local_shared_ptr<Packet> *foundpacket = const_cast<Node*>(this)->reverseLookup(
		const_cast<local_shared_ptr<Packet> &>(packet), false, 0, false, 0);
	ASSERT(foundpacket);
	return *foundpacket;
}

template <class XN>
XN *
Node<XN>::superNode(Snapshot<XN> &shot) {
	XN *supernode = 0;
	reverseLookup(shot.m_packet, false, 0, false, &supernode);
	return supernode;
}

template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::snapshotSupernode(const shared_ptr<BranchPoint > &branchpoint,
	local_shared_ptr<PacketWrapper> &shot, local_shared_ptr<Packet> **subpacket,
	shared_ptr<BranchPoint > *branchpoint_super_returned, bool copy_branch,
	int serial, bool set_missing) {
	local_shared_ptr<PacketWrapper> oldwrapper(shot);
	ASSERT( !shot->hasPriority());
	shared_ptr<BranchPoint > branchpoint_super(shot->branchpoint());
	if( !branchpoint_super) {
		//Checking if it is up-to-date.
		if( *branchpoint == oldwrapper) {
			//Supernode has been destroyed.
			return SNAPSHOT_NODE_MISSING;
		}
		return SNAPSHOT_DISTURBED;
	}
	int ridx = shot->reverseIndex();
	shot = *branchpoint_super;
	local_shared_ptr<Packet> *foundpacket = 0;
	if(shot->packet()) {
		foundpacket = &shot->packet();
		if(branchpoint_super_returned)
			*branchpoint_super_returned = branchpoint;
	}
	SnapshotStatus status = SNAPSHOT_SUCCESS;
	if( !shot->hasPriority()) {
		SnapshotStatus status = snapshotSupernode(branchpoint_super, shot, &foundpacket,
			branchpoint_super_returned, copy_branch, serial, set_missing);
		switch(status) {
		case SNAPSHOT_DISTURBED:
		default:
			return status;
		case SNAPSHOT_VOID_PACKET:
		case SNAPSHOT_NODE_MISSING:
			status = SNAPSHOT_SUCCESS;
			break;
		case SNAPSHOT_SUCCESS:
			break;
		case SNAPSHOT_COLLIDED:
			break;
		}
	}
	//Checking if it is up-to-date.
	if( *branchpoint == oldwrapper) {
		ASSERT(foundpacket);
		int size = ( *foundpacket)->size();
		int i = ridx;
		for(int cnt = 0; cnt < size; ++cnt) {
			if(i >= size)
				i = 0;
			local_shared_ptr<Packet> &p(( *foundpacket)->subpackets()->at(i));
			if(( *foundpacket)->subnodes()->at(i)->m_wrapper == branchpoint) {
				if( !p)
					return SNAPSHOT_VOID_PACKET;
				//Requested node is found.
				if((serial != Packet::SERIAL_NULL) && (branchpoint_super->m_bundle_serial == serial)) {
					//The node has been already bundled in the same snapshot.
			//		printf("C");
					return SNAPSHOT_COLLIDED;
				}
				*subpacket = &p; //Bundled packet or unbundled packet w/o local packet.
				if(copy_branch) {
					if(( *foundpacket)->subpackets()->m_serial != serial) {
						foundpacket->reset(new Packet( **foundpacket));
						( *foundpacket)->subpackets().reset(new PacketList( *( *foundpacket)->subpackets()));
						( *foundpacket)->subpackets()->m_serial = serial;
						( *foundpacket)->m_missing =
							( *foundpacket)->m_missing || set_missing;
					}
				}
				ASSERT((status == SNAPSHOT_COLLIDED) || (status == SNAPSHOT_SUCCESS));
				return status;
			}
			//The index might be modified by swap().
			++i;
		}
		return SNAPSHOT_NODE_MISSING;
	}
	return SNAPSHOT_DISTURBED;
}

template <class XN>
void
Node<XN>::snapshot(Snapshot<XN> &snapshot, bool multi_nodal, uint64_t &started_time) const {
	local_shared_ptr<PacketWrapper> target;
	for(;;) {
		target = *m_wrapper;
		if(target->hasPriority()) {
			if( !multi_nodal)
				break;
			if( !target->packet()->missing()) {
				STRICT_ASSERT(target->packet()->checkConsistensy(target->packet()));
				break;
			}
		}
		else {
			// Taking a snapshot inside the super packet.
			shared_ptr<BranchPoint > branchpoint(m_wrapper);
			local_shared_ptr<PacketWrapper> superwrapper(target);
			local_shared_ptr<Packet> *foundpacket;
			SnapshotStatus status = snapshotSupernode(branchpoint, superwrapper, &foundpacket);
			switch(status) {
			case SNAPSHOT_SUCCESS: {
					if( !( *foundpacket)->missing() || !multi_nodal) {
						snapshot.m_packet = *foundpacket;
						STRICT_ASSERT(snapshot.m_packet->checkConsistensy(snapshot.m_packet));
						return;
					}
					// The packet is imperfect, and then re-bundling the subpackets.
					shared_ptr<BranchPoint > branchpoint_super(target->branchpoint());
					if( !branchpoint_super)
						continue;
					unbundle(NULL, started_time, *branchpoint_super, *m_wrapper, target);
					continue;
				}
			case SNAPSHOT_DISTURBED:
			default:
				continue;
			case SNAPSHOT_NODE_MISSING: {
					local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(target->packet()));
					if( !m_wrapper->compareAndSet(target, newwrapper))
						continue;
		//			printf("n\n");
					snapshot.m_packet = target->packet();
					STRICT_ASSERT(snapshot.m_packet->checkConsistensy(snapshot.m_packet));
					return;
				}
			case SNAPSHOT_VOID_PACKET:
				//Just after the node was inserted.
				superwrapper->packet()->node().bundle(superwrapper, started_time, snapshot.m_serial, true);
				continue;
			}
		}
		BundledStatus status = const_cast<Node *>(this)->bundle(target, started_time, snapshot.m_serial, true);
		if(status == BUNDLE_SUCCESS) {
			ASSERT( !target->packet()->missing());
			STRICT_ASSERT(target->packet()->checkConsistensy(target->packet()));
			break;
		}
	}
	snapshot.m_packet = target->packet();
}

template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle_subpacket(const shared_ptr<Node> &subnode,
	local_shared_ptr<PacketWrapper> &subwrapper, local_shared_ptr<Packet> &subpacket_new,
	uint64_t &started_time, int64_t bundle_serial) {

	subwrapper = *subnode->m_wrapper;
	if( !subwrapper->hasPriority()) {
		shared_ptr<BranchPoint > branchpoint(subwrapper->branchpoint());
		bool need_for_unbundle = false;
		bool detect_collision = false;
		if(branchpoint == m_wrapper) {
			if(subpacket_new) {
				if(subpacket_new->missing())
					need_for_unbundle = true;
				else
					return BUNDLE_SUCCESS;
			}
			else {
				if(subwrapper->packet()) {
					//Re-inserted.
				}
				else
					return BUNDLE_DISTURBED;
			}
		}
		else {
			if( !branchpoint) {
				//Supernode has been destroyed.
				ASSERT(subwrapper->packet());
			}
			else {
				//bundled by another node.
				need_for_unbundle = true;
				detect_collision = true;
			}
		}
		if(need_for_unbundle) {
			local_shared_ptr<PacketWrapper> superwrapper( *branchpoint);
			local_shared_ptr<PacketWrapper> subwrapper_new;
			UnbundledStatus status = unbundle(detect_collision ? &bundle_serial : NULL, started_time,
				*branchpoint, *subnode->m_wrapper, subwrapper, NULL, &subwrapper_new);
			switch(status) {
			case UNBUNDLE_W_NEW_SUBVALUE:
				subwrapper = subwrapper_new;
				break;
			case UNBUNDLE_SUBVALUE_NOT_FOUND:
				if(subwrapper->packet()) {
					//The node has been released from the supernode.
					break;
				}
			case UNBUNDLE_COLLIDED:
				//The subpacket has already been included in the snapshot.
				subpacket_new.reset();
				return BUNDLE_SUCCESS;
			case UNBUNDLE_SUBVALUE_HAS_CHANGED:
			default:
				return BUNDLE_DISTURBED;
			}
		}
	}
	if(subwrapper->packet()->size() && subwrapper->packet()->missing() ) {
		BundledStatus status = subnode->bundle(subwrapper, started_time, bundle_serial, false);
		switch(status) {
		case BUNDLE_SUCCESS:
//						ASSERT(subwrapper->isBundled());
			break;
		case BUNDLE_DISTURBED:
		default:
			return BUNDLE_DISTURBED;
		}
	}
	subpacket_new = subwrapper->packet();
	return BUNDLE_SUCCESS;
}

template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle(local_shared_ptr<PacketWrapper> &target,
	uint64_t &started_time, int64_t bundle_serial, bool is_bundle_root) {
	ASSERT(target->packet());
	ASSERT(target->packet()->size());
	ASSERT(target->packet()->missing());
	local_shared_ptr<Packet> packet(new Packet( *target->packet()));

	m_wrapper->m_bundle_serial = bundle_serial;

	local_shared_ptr<PacketWrapper> oldwrapper(target);
	target.reset(new PacketWrapper(packet));
	//copying all sub-packets from nodes to the new packet.
	packet->subpackets().reset(new PacketList( *packet->subpackets()));
	shared_ptr<PacketList> &subpackets(packet->subpackets());
	packet->m_missing = false;
	shared_ptr<NodeList> &subnodes(packet->subnodes());
	std::vector<local_shared_ptr<PacketWrapper> > subwrappers_org(subpackets->size());
	for(unsigned int i = 0; i < subpackets->size(); ++i) {
		shared_ptr<Node> child(subnodes->at(i));
		local_shared_ptr<Packet> &subpacket_new(subpackets->at(i));
		for(;;) {
			local_shared_ptr<PacketWrapper> subwrapper;
			BundledStatus status = bundle_subpacket(child, subwrapper, subpacket_new, started_time, bundle_serial);
			if(status != BUNDLE_SUCCESS) {
				if(target == *m_wrapper)
					continue;
				else
					return BUNDLE_DISTURBED;
			}
			subwrappers_org[i] = subwrapper;
			if(subpacket_new) {
				if(subpacket_new->missing()) {
					packet->m_missing = true;
				}
				ASSERT(&subpacket_new->node() == child.get());
			}
			else
				packet->m_missing = true;
			break;
		}
	}
	if(is_bundle_root)
		packet->m_missing = false;
	bool missing = packet->missing();
	packet->m_missing = true;

	//First checkpoint.
	if( !m_wrapper->compareAndSet(oldwrapper, target)) {
		return BUNDLE_DISTURBED;
	}
	//clearing all packets on sub-nodes if not modified.
	for(unsigned int i = 0; i < subnodes->size(); i++) {
		shared_ptr<Node> child(subnodes->at(i));
		local_shared_ptr<PacketWrapper> nullwrapper;
		if(subpackets->at(i))
			nullwrapper.reset(new PacketWrapper(m_wrapper, i));
		else
			nullwrapper.reset(new PacketWrapper( *subwrappers_org[i]));
		//Second checkpoint, the written bundle is valid or not.
		if( !child->m_wrapper->compareAndSet(subwrappers_org[i], nullwrapper)) {
			return BUNDLE_DISTURBED;
		}
	}
	oldwrapper = target;
	target.reset(new PacketWrapper(packet));
	if( !missing) {
		packet.reset(new Packet( *packet));
		target->packet() = packet;
		packet->m_missing = missing;
	}

	//Finally, tagging as bundled.
	if( !m_wrapper->compareAndSet(oldwrapper, target))
		return BUNDLE_DISTURBED;

	STRICT_ASSERT(packet->checkConsistensy(packet));
	return BUNDLE_SUCCESS;
}

template <class XN>
void
Node<XN>::fetchSubpackets(std::deque<local_shared_ptr<PacketWrapper> > &subwrappers,
	const local_shared_ptr<Packet> &packet) {
	for(int i = 0; i < packet->size(); ++i) {
		const local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
		subwrappers.push_back( *packet->subnodes()->at(i)->m_wrapper);
		if(subpacket)
			fetchSubpackets(subwrappers, subpacket);
	}
}

template <class XN>
bool
Node<XN>::commit(Transaction<XN> &tr) {

	m_wrapper->negotiate(tr.m_started_time);

	local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(tr.m_packet));
	for(int retry = 0;; ++retry) {
		local_shared_ptr<PacketWrapper> wrapper( *m_wrapper);
		if(wrapper->hasPriority()) {
			//Committing directly to the node.
			if(wrapper->packet() != tr.m_oldpacket) {
				if( !tr.isMultiNodal() && (wrapper->packet()->payload() == tr.m_oldpacket->payload())) {
					//Single-node mode, the payload in the snapshot is unchanged.
					tr.m_packet->subpackets() = wrapper->packet()->subpackets();
				}
				else
					return false;
			}
//			STRICT_TEST(std::deque<local_shared_ptr<PacketWrapper> > subwrappers);
//			STRICT_TEST(fetchSubpackets(subwrappers, wrapper->packet()));
			STRICT_ASSERT(tr.m_packet->checkConsistensy(tr.m_packet));
			if(m_wrapper->compareAndSet(wrapper, newwrapper)) {
//				STRICT_TEST(if(wrapper->isBundled())
//					for(typename std::deque<local_shared_ptr<PacketWrapper> >::const_iterator
//					it = subwrappers.begin(); it != subwrappers.end(); ++it)
//					ASSERT( !( *it)->hasPriority()));
				return true;
			}
			continue;
		}
		//Unbundling this node from the super packet.
		shared_ptr<BranchPoint > branchpoint_super(wrapper->branchpoint());
		if( !branchpoint_super)
			continue; //Supernode has been destroyed.
		UnbundledStatus status = unbundle(NULL, tr.m_started_time, *branchpoint_super, *m_wrapper, wrapper,
			tr.isMultiNodal() ? &tr.m_oldpacket : NULL, tr.isMultiNodal() ? &newwrapper : NULL);
		switch(status) {
		case UNBUNDLE_W_NEW_SUBVALUE:
			if(tr.isMultiNodal())
				return true;
			continue;
		case UNBUNDLE_SUBVALUE_HAS_CHANGED:
			return false;
		case UNBUNDLE_SUBVALUE_NOT_FOUND:
		case UNBUNDLE_PARTIALLY:
		case UNBUNDLE_DISTURBED:
		default:
			continue;
		}
	}
}

template <class XN>
typename Node<XN>::UnbundledStatus
Node<XN>::unbundle(const int64_t *bundle_serial, uint64_t &time_started,
	BranchPoint &branchpoint,
	BranchPoint &subbranchpoint, const local_shared_ptr<PacketWrapper> &nullwrapper,
	const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<PacketWrapper> *newsubwrapper) {
	ASSERT( !nullwrapper->hasPriority());

	local_shared_ptr<PacketWrapper> wrapper(branchpoint);
	local_shared_ptr<PacketWrapper> copied;
//	printf("u");
	UnbundledStatus status = UNBUNDLE_W_NEW_SUBVALUE;
	if( !wrapper->hasPriority()) {
//		// Taking a snapshot inside the super packet.
//		shared_ptr<BranchPoint > branchpoint_super(m_wrapper);
//		local_shared_ptr<PacketWrapper> superwrapper(target);
//		local_shared_ptr<Packet> *foundpacket;
//		SnapshotStatus status = snapshotFromSuper(branchpoint_super, superwrapper, &foundpacket);
//		switch(status) {
//		case SNAPSHOT_SUCCESS: {
//				local_shared_ptr<Packet> newsuperpacket(new Packet(superwrapper->packet()));
//				foundpacket = &( *foundpacket)->node().reverseLookup(
//					superwrapper->packet(), true, Packet::newSerial(), true);
//				if( !( *foundpacket)->missing() || !multi_nodal) {
//					snapshot.m_packet = *foundpacket;
//					snapshot.m_bundled = true;
//					STRICT_ASSERT(snapshot.m_packet->checkConsistensy(snapshot.m_packet));
//					return;
//				}
//				// The packet is imperfect, and then re-bundling the subpackets.
//				shared_ptr<BranchPoint > branchpoint_super(target->branchpoint());
//				if( !branchpoint_super)
//					continue;
//				unbundle(NULL, started_time, *branchpoint_super, *m_wrapper, target);
//				continue;
//			}
//		case SNAPSHOT_STRUCTURE_HAS_CHANGED:
//		case SNAPSHOT_DISTURBED:
//			continue;
//		case SNAPSHOT_NOT_FOUND: {
//				local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(target->packet(), true));
//				if( !m_wrapper->compareAndSet(target, newwrapper))
//					continue;
//	//			printf("n\n");
//				snapshot.m_packet = target->packet();
//				snapshot.m_bundled = true;
//				STRICT_ASSERT(snapshot.m_packet->checkConsistensy(snapshot.m_packet));
//				return;
//			}
//		case SNAPSHOT_VOID_PACKET:
//			//Just after the node was inserted.
//			superwrapper->packet()->node().bundle(superwrapper, started_time, snapshot.m_serial, true);
//			continue;
//		}
//

		//Unbundle all supernodes.
		shared_ptr<BranchPoint > branchpoint_super(wrapper->branchpoint());
		local_shared_ptr<PacketWrapper> wrapper_super( *branchpoint_super);
		if(branchpoint_super)
			status = unbundle(bundle_serial, time_started, *branchpoint_super, branchpoint, wrapper, NULL, &copied);
		else
			status = UNBUNDLE_SUBVALUE_NOT_FOUND;

		switch(status) {
		case UNBUNDLE_W_NEW_SUBVALUE:
			ASSERT(copied);
			wrapper = copied;
			break;
		case UNBUNDLE_PARTIALLY:
			return UNBUNDLE_PARTIALLY;
		case UNBUNDLE_COLLIDED:
			break;
		case UNBUNDLE_SUBVALUE_NOT_FOUND:
			if(wrapper == branchpoint) {
				//The node has been released from the supernode.
				if( !wrapper->packet())
					return UNBUNDLE_SUBVALUE_NOT_FOUND;
				break;
			}
		default:
			return UNBUNDLE_DISTURBED;
		}
	}

	local_shared_ptr<Packet> subpacket;
//	typename NodeList::iterator nit = copied->packet()->subnodes()->begin();
//	PacketList &subpackets( *copied->packet()->subpackets());
//	for(typename PacketList::iterator pit = subpackets.begin(); pit != subpackets.end();) {
//		if(( *nit)->m_wrapper.get() == &subbranchpoint) {
//			subpacket = *pit;
//			break;
//		}
//		++pit;
//		++nit;
//	}
	if(wrapper->packet()) {
		int size = wrapper->packet()->size();
		if( !size) {
			if(wrapper == branchpoint)
				return UNBUNDLE_SUBVALUE_NOT_FOUND;
			else
				return UNBUNDLE_DISTURBED;
		}
		PacketList &subpackets( *wrapper->packet()->subpackets());
		int i = nullwrapper->reverseIndex();
		for(int cnt = 0; cnt < size; ++cnt) {
			if(i >= size)
				i = 0;
			local_shared_ptr<Packet> &p(subpackets.at(i));
			if( p && (p->node().m_wrapper.get() == &subbranchpoint)) {
				subpacket = p; //Bundled packet or unbundled packet w/o local packet.
				break;
			}
			//The index might be modified by swap().
			++i;
		}
		if( !subpacket) {
			if(wrapper == branchpoint)
				return UNBUNDLE_SUBVALUE_NOT_FOUND;
			else
				return UNBUNDLE_DISTURBED;
		}
	}
	else {
		ASSERT(status == UNBUNDLE_COLLIDED);
	}
	if(status == UNBUNDLE_COLLIDED)
		return UNBUNDLE_COLLIDED;

	if(bundle_serial && (branchpoint.m_bundle_serial == *bundle_serial)) {
		//The node has been already bundled in the same snapshot.
//		printf("C");
		return UNBUNDLE_COLLIDED;
	}
	branchpoint.negotiate(time_started);

	if(status != UNBUNDLE_SUBVALUE_NOT_FOUND) {
		//Tagging superpacket as missing.
		copied.reset(new PacketWrapper(wrapper->packet()));
		copied->packet().reset(new Packet( *copied->packet()));
		copied->packet()->m_missing = true;
		if( !branchpoint.compareAndSet(wrapper, copied)) {
			return UNBUNDLE_DISTURBED;
		}
	}

	local_shared_ptr<PacketWrapper> newsubwrapper_copied;
	if(oldsubpacket) {
		newsubwrapper_copied = *newsubwrapper;
		if(subpacket != *oldsubpacket) {
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		}
	}
	else {
		newsubwrapper_copied.reset(new PacketWrapper(subpacket));
	}
	STRICT_ASSERT(newsubwrapper_copied->packet()->checkConsistensy(newsubwrapper_copied->packet()));

	if( !subbranchpoint.compareAndSet(nullwrapper, newsubwrapper_copied)) {
		if( !local_shared_ptr<PacketWrapper>(subbranchpoint)->packet())
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		return UNBUNDLE_PARTIALLY;
	}
	if(newsubwrapper)
		*newsubwrapper = newsubwrapper_copied;
	return UNBUNDLE_W_NEW_SUBVALUE;
}

} //namespace Transactional

