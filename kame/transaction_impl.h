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

namespace Transactional {

template <class XN>
XThreadLocal<typename Node<XN>::FuncPayloadCreator> Node<XN>::stl_funcPayloadCreator;

template <class XN>
atomic<int64_t> Node<XN>::Packet::s_serial = 0;

template <class XN>
Node<XN>::Packet::Packet() : m_hasCollision(false) {}

template <class XN>
void
Node<XN>::Packet::_print() const {
	printf("Packet: ");
	printf("Node:%llx, ", (unsigned long long)(uintptr_t)&node());
	printf("Branchpoint:%llx, ", (unsigned long long)(uintptr_t)node().m_wrapper.get());
	if(m_hasCollision)
		printf("w/ collision, ");
	if(size()) {
		printf("%d subnodes : [ \n", (int)size());
		for(unsigned int i = 0; i < size(); i++) {
			if(subpackets()->at(i)) {
				subpackets()->at(i)->_print();
			}
			else {
				printf("Node:%llx, [void] ", (unsigned long long)(uintptr_t)subnodes()->at(i).get());
			}
		}
		printf("]\n");
	}
	printf(";");
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const local_shared_ptr<Packet> &x, bool bundled) :
	m_branchpoint(), m_packet(x), m_state(0) {
	ASSERT( !bundled || !x->m_hasCollision);
	setBundled(bundled);
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const shared_ptr<BranchPoint > &bp, int reverse_index) :
	m_branchpoint(bp), m_packet(), m_state() {
	setReverseIndex(reverse_index);
}

template <class XN>
void
Node<XN>::PacketWrapper::_print() const {
	printf("PacketWrapper: ");
	if( !packet()) {
		printf("Not here, ");
		printf("Bundled by:%llx, ", (unsigned long long)(uintptr_t)branchpoint().get());
	}
	else {
		if(isBundled())
			printf("Bundled, ");
		packet()->_print();
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
Node<XN>::Node() : m_wrapper(new BranchPoint()), m_packet_cache(new Cache), m_transaction_count(0) {
	local_shared_ptr<Packet> packet(new Packet());
	m_wrapper->reset(new PacketWrapper(packet, true));
	//Use create() for this hack.
	packet->m_payload.reset(( *stl_funcPayloadCreator)(static_cast<XN&>( *this)));
	*stl_funcPayloadCreator = NULL;

	if(this == (Node*)0x1)
		_print(); //Do not strip out the debug function, _print() please.
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
	for(;;) {
		Transaction<XN> tr( *this);
		if(insert(tr, var))
			break;
	}
}
template <class XN>
bool
Node<XN>::insert(Transaction<XN> &tr, const shared_ptr<XN> &var) {
	local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
	tr.m_packet->m_hasCollision = false;
	packet->subpackets().reset(packet->size() ? (new PacketList( *packet->subpackets())) : (new PacketList));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->subnodes().reset(packet->size() ? (new NodeList( *packet->subnodes())) : (new NodeList));
	packet->subpackets()->resize(packet->size() + 1);
	ASSERT(packet->subnodes());
	ASSERT(std::find(packet->subnodes()->begin(), packet->subnodes()->end(), var) == packet->subnodes()->end());
	packet->subnodes()->push_back(var);
	ASSERT(packet->subpackets()->size() == packet->subnodes()->size());
	tr[ *this].catchEvent(var, packet->size() - 1);
	tr[ *this].listChangeEvent();
//		printf("i");
	return tr.commit(false);
}
template <class XN>
void
Node<XN>::release(const shared_ptr<XN> &var) {
	for(;;) {
		Transaction<XN> tr(*this);
		if(release(tr, var))
			break;
	}
}
template <class XN>
bool
Node<XN>::release(Transaction<XN> &tr, const shared_ptr<XN> &var) {
	local_shared_ptr<Packet> oldsubpacket(
		var->reverseLookup(tr.m_oldpacket));
	local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
	tr.m_packet->m_hasCollision = false;
	packet->subpackets().reset(packet->size() ? (new PacketList( *packet->subpackets())) : (new PacketList));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->subnodes().reset(packet->size() ? (new NodeList( *packet->subnodes())) : (new NodeList));
	local_shared_ptr<PacketWrapper> newsubwrapper;
	unsigned int idx = 0;
	int old_idx = -1;
	typename NodeList::iterator nit = packet->subnodes()->begin();
	for(typename PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
		ASSERT(nit != packet->subnodes()->end());
		if(nit->get() == &*var) {
			if( *pit) {
				if( !( *pit)->size()) {
					pit->reset(new Packet( **pit));
				}
				newsubwrapper.reset(new PacketWrapper( *pit, !( *pit)->m_hasCollision));
			}
			pit = packet->subpackets()->erase(pit);
			nit = packet->subnodes()->erase(nit);
			old_idx = idx;
		}
		else {
			if( *pit) {
				if(( *pit)->size()) {
					if(( *pit)->m_hasCollision)
						packet->m_hasCollision = true;
				}
			}
			else
				packet->m_hasCollision = true;
			++nit;
			++pit;
			++idx;
		}
	}

	if( !packet->size()) {
		packet->subpackets().reset();
	}
	tr[ *this].releaseEvent(var, old_idx);
	tr[ *this].listChangeEvent();
	if( !newsubwrapper) {
		return tr.commit( !packet->m_hasCollision);
	}
	local_shared_ptr<PacketWrapper> nullwrapper( *var->m_wrapper);
	if(nullwrapper->packet())
		return false;
//		printf("r");
	local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(packet, !packet->m_hasCollision));
	UnbundledStatus ret = unbundle(NULL, tr.m_started_time, *m_wrapper, *var->m_wrapper,
		nullwrapper, &oldsubpacket, &newsubwrapper, &tr.m_oldpacket, &newwrapper);
	if(ret == UNBUNDLE_W_NEW_VALUES) {
//			printf("%d", (int)packet->size());
		tr.finalizeCommitment();
		return true;
	}
	return false;
}
template <class XN>
void
Node<XN>::releaseAll() {
	for(;;) {
		Transaction<XN> tr( *this);
		if( !tr.size())
			break;
		shared_ptr<const NodeList> list(tr.list());
		release(tr, list->front());
	}
}
template <class XN>
void
Node<XN>::swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
	for(;;) {
		Transaction<XN> tr( *this);
		if(swap(tr, x, y))
			break;
	}
}
template <class XN>
bool
Node<XN>::swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
	local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
	tr.m_packet->m_hasCollision = false;
	packet->subpackets().reset(packet->size() ? (new PacketList( *packet->subpackets())) : (new PacketList));
	packet->subpackets()->m_serial = tr.m_serial;
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
	return tr.commit(false);
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookupWithHint(shared_ptr<BranchPoint> &branchpoint,
	local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial, bool has_collision, Cache *cache) {
	ASSERT(packet->size());
	local_shared_ptr<PacketWrapper> wrapper( *branchpoint);
	if(wrapper->packet())
		return NULL;
	shared_ptr<BranchPoint> branchpoint_super(wrapper->branchpoint());
	if( !branchpoint_super)
		return NULL;
	local_shared_ptr<Packet> *foundpacket;
	if(branchpoint_super == packet->node().m_wrapper)
		foundpacket = &packet;
	else {
		foundpacket = reverseLookupWithHint(branchpoint_super, packet, copy_branch, tr_serial, has_collision, NULL);
		if( !foundpacket)
			return NULL;
	}
	int ridx = wrapper->reverseIndex();
	if( !( *foundpacket)->size() || (ridx >= ( *foundpacket)->size()))
		return NULL;
	if(copy_branch) {
		if(( *foundpacket)->subpackets()->m_serial != tr_serial) {
			foundpacket->reset(new Packet( **foundpacket));
			( *foundpacket)->m_hasCollision = has_collision;
			( *foundpacket)->subpackets().reset(new PacketList( *( *foundpacket)->subpackets()));
			( *foundpacket)->subpackets()->m_serial = tr_serial;
		}
	}
	local_shared_ptr<Packet> &p(( *foundpacket)->subpackets()->at(ridx));
	if( !p || (p->node().m_wrapper != branchpoint)) {
		return NULL;
	}
	if(cache) {
		cache->subpackets = ( *foundpacket)->subpackets();
		cache->index = ridx;
	}
	return &p;
}
template <class XN>
local_shared_ptr<typename Node<XN>::Packet>&
Node<XN>::reverseLookup(local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial, bool has_collision) {
	local_shared_ptr<Packet> *foundpacket;
	if( &packet->node() == this) {
		foundpacket = &packet;
	}
	else {
		local_shared_ptr<Cache> cached(m_packet_cache);
		shared_ptr<PacketList> subpackets_cached(cached->subpackets.lock());
		if(subpackets_cached && ( !copy_branch || (packet->subpackets()->m_serial == tr_serial)) &&
			(subpackets_cached->m_serial == packet->subpackets()->m_serial)) {
				foundpacket = &subpackets_cached->at(cached->index);
//				printf("%%");
		}
		else {
			local_shared_ptr<Cache> newcache(new Cache);
			foundpacket = reverseLookupWithHint(m_wrapper, packet, copy_branch, tr_serial, has_collision, newcache.get());
			if(foundpacket) {
//				printf("$");
			}
			else {
//				printf("!");
				foundpacket = forwardLookup(packet, copy_branch, tr_serial, has_collision, newcache.get());
				ASSERT(foundpacket);
			}
			m_packet_cache = newcache;
		}
		ASSERT( &( *foundpacket)->node() == this);
	}
	if(copy_branch && (( *foundpacket)->payload()->m_serial != tr_serial)) {
		foundpacket->reset(new Packet( **foundpacket));
		if(( *foundpacket)->size())
			( *foundpacket)->m_hasCollision = has_collision;
	}
//						printf("#");
	return *foundpacket;
}
template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::forwardLookup(local_shared_ptr<Packet> &packet,
	bool copy_branch, int tr_serial, bool has_collision, Cache *cache) const {
	ASSERT(packet);
	if( !packet->subpackets())
		return NULL;
	if(copy_branch) {
		if(packet->subpackets()->m_serial != tr_serial) {
			packet.reset(new Packet( *packet));
			packet->subpackets().reset(new PacketList( *packet->subpackets()));
			packet->subpackets()->m_serial = tr_serial;
			packet->m_hasCollision = has_collision;
		}
	}
	for(unsigned int i = 0; i < packet->subnodes()->size(); i++) {
		if(packet->subnodes()->at(i).get() == this) {
			local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
			if(subpacket) {
				if(cache) {
					cache->subpackets = packet->subpackets();
					cache->index = i;
				}
				return &subpacket;
			}
		}
	}
	for(unsigned int i = 0; i < packet->subnodes()->size(); i++) {
		local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
		if(subpacket) {
			if(local_shared_ptr<Packet> *p = forwardLookup(subpacket, copy_branch, tr_serial, has_collision, cache)) {
				return p;
			}
		}
	}
	return NULL;
}

template <class XN>
void
Node<XN>::snapshot(Snapshot<XN> &snapshot, bool multi_nodal, uint64_t &started_time) const {
	local_shared_ptr<PacketWrapper> target;
	for(;;) {
		target = *m_wrapper;
		if(target->isBundled()) {
			ASSERT( !target->packet()->m_hasCollision);
			break;
		}
		if( !target->packet()) {
			// Taking a snapshot inside the super packet.
			shared_ptr<BranchPoint > branchpoint(m_wrapper);
			local_shared_ptr<Packet> *foundpacket;
			SnapshotStatus status = snapshotFromSuper(branchpoint, target, &foundpacket);
			if(status != SNAPSHOT_SUCCESS)
				continue;
			if( !( *foundpacket)->m_hasCollision || !multi_nodal) {
				snapshot.m_packet = *foundpacket;
				snapshot.m_bundled = !( *foundpacket)->m_hasCollision;
				return;
			}
			// The packet is imperfect, and then re-bundling the subpackets.
			target = *m_wrapper;
			if( target->packet())
				continue;
			shared_ptr<BranchPoint > branchpoint_super(target->branchpoint());
			if( !branchpoint_super)
				continue;
			unbundle(NULL, started_time, *branchpoint_super, *m_wrapper, target);
			continue;
		}
		if( !multi_nodal)
			break;
		BundledStatus status = const_cast<Node *>(this)->bundle(target, started_time);
		if(status == BUNDLE_SUCCESS) {
			ASSERT( !target->packet()->m_hasCollision);
			ASSERT( target->isBundled() );
			break;
		}
	}
	snapshot.m_packet = target->packet();
	snapshot.m_bundled = target->isBundled();
}

template <class XN>
typename Node<XN>::SnapshotStatus
Node<XN>::snapshotFromSuper(shared_ptr<BranchPoint > &branchpoint,
	local_shared_ptr<PacketWrapper> &shot, local_shared_ptr<Packet> **subpacket,
	shared_ptr<BranchPoint > *branchpoint_2nd) {
	local_shared_ptr<PacketWrapper> oldwrapper(shot);
	ASSERT( !shot->packet());
	shared_ptr<BranchPoint > branchpoint_super(shot->branchpoint());
	if( !branchpoint_super)
		return SNAPSHOT_STRUCTURE_HAS_CHANGED; //Supernode has been destroyed.
	int ridx = shot->reverseIndex();
	shot = *branchpoint_super;
	local_shared_ptr<Packet> *foundpacket;
	if(shot->packet()) {
		foundpacket = &shot->packet();
		if(branchpoint_2nd)
			*branchpoint_2nd = branchpoint;
	}
	else {
		 SnapshotStatus status = snapshotFromSuper(branchpoint_super, shot, &foundpacket, branchpoint_2nd);
		if(status != SNAPSHOT_SUCCESS)
			return status;
	}
	//Checking if it is up-to-date.
	if( *branchpoint == oldwrapper) {
		ASSERT( ( *foundpacket)->size());
		//The index might be modified by swap().
		for(int i = ridx; ;) {
			if(i >= ( *foundpacket)->size())
				i = 0;
			local_shared_ptr<Packet> &p(( *foundpacket)->subpackets()->at(i));
			if( p && (p->node().m_wrapper == branchpoint)) {
				*subpacket = &p; //Bundled packet or unbundled packet w/o local packet.
				branchpoint = branchpoint_super;
				return SNAPSHOT_SUCCESS;
			}
			++i;
		}
	}
	return SNAPSHOT_DISTURBED;
}


template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle(local_shared_ptr<PacketWrapper> &target, uint64_t &started_time,
	const int64_t *bundle_serial) {
	ASSERT( !target->isBundled() && target->packet());
	ASSERT(target->packet()->size());
	local_shared_ptr<Packet> packet(new Packet( *target->packet()));
	packet->m_hasCollision = false;

	bool is_bundle_root = !bundle_serial;
	int64_t _serial;
	if(is_bundle_root) {
		for(;;) {
			_serial = Packet::s_serial;
			if(Packet::s_serial.compareAndSet(_serial, _serial + 1))
				break;
		}
		_serial++;
		bundle_serial = &_serial;
	}
	m_wrapper->m_bundle_serial = *bundle_serial;

	local_shared_ptr<PacketWrapper> oldwrapper(target);
	target.reset(new PacketWrapper(packet, false));
	//copying all sub-packets from nodes to the new packet.
	packet->subpackets().reset(new PacketList( *packet->subpackets()));
	packet->subpackets()->m_serial = *bundle_serial;
	shared_ptr<PacketList> &subpackets(packet->subpackets());
	shared_ptr<NodeList> &subnodes(packet->subnodes());
	std::vector<local_shared_ptr<PacketWrapper> > subwrappers_org(subpackets->size());
	for(unsigned int i = 0; i < subpackets->size(); ++i) {
		shared_ptr<Node> child(subnodes->at(i));
		local_shared_ptr<Packet> &subpacket_new(subpackets->at(i));
		for(;;) {
			local_shared_ptr<PacketWrapper> subwrapper( *child->m_wrapper);
			if(subwrapper->packet()) {
				if( !subwrapper->isBundled()) {
					BundledStatus status = child->bundle(subwrapper, started_time, bundle_serial);
					switch(status) {
					case BUNDLE_SUCCESS:
//						ASSERT(subwrapper->isBundled());
						break;
					case BUNDLE_DISTURBED:
						if(target == *m_wrapper)
							continue;
					default:
						return BUNDLE_DISTURBED;
					}
				}
				subpacket_new = subwrapper->packet();
			}
			else {
				shared_ptr<BranchPoint > branchpoint(subwrapper->branchpoint());
				if( !branchpoint)
					return BUNDLE_DISTURBED; //Supernode has been destroyed.
				if(branchpoint != m_wrapper) {
					//bundled by another node.
					local_shared_ptr<PacketWrapper> subwrapper_new;
					UnbundledStatus status = unbundle(bundle_serial, started_time,
						*branchpoint, *child->m_wrapper, subwrapper, NULL, &subwrapper_new);
					switch(status) {
					case UNBUNDLE_COLLIDED:
						//The subpacket has already been included in the snapshot.
						subpacket_new.reset();
						packet->m_hasCollision = true;
						break;
					case UNBUNDLE_W_NEW_SUBVALUE:
					case UNBUNDLE_W_NEW_VALUES:
						subwrapper = subwrapper_new;
						subpacket_new = subwrapper_new->packet();
						ASSERT(subwrapper->packet());
						break;
					default:
						if(target == *m_wrapper)
							continue;
						else
							return BUNDLE_DISTURBED;
					}
				}
				else {
					if( !subpacket_new) {
		//				printf("?");
						ASSERT(target != *m_wrapper);
						//m_wrapper has changed, bundled by the other thread.
						return BUNDLE_DISTURBED;
					}
				}
			}
			subwrappers_org[i] = subwrapper;
			if(subpacket_new) {
				if(subpacket_new->m_hasCollision) {
					packet->m_hasCollision = true;
				}
				ASSERT(&subpacket_new->node() == child.get());
			}
			break;
		}
	}
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
	if(is_bundle_root)
		packet->m_hasCollision = false;
	target.reset(new PacketWrapper(packet, !packet->m_hasCollision));
	//Finally, tagging as bundled.
	if( !m_wrapper->compareAndSet(oldwrapper, target))
		return BUNDLE_DISTURBED;
	return BUNDLE_SUCCESS;
}

template <class XN>
bool
Node<XN>::commit(Transaction<XN> &tr, bool new_bundle_state) {

	m_wrapper->negotiate(tr.m_started_time);

	if( !tr.isBundled()) {
		ASSERT( !tr.isMultiNodal());
		new_bundle_state = false;
	}
	local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(tr.m_packet, new_bundle_state));
	ASSERT( tr.m_packet->size() || newwrapper->isBundled());
	for(int retry = 0;; ++retry) {
		local_shared_ptr<PacketWrapper> wrapper( *m_wrapper);
		if(wrapper->packet()) {
			//Committing directly to the node.
			if(wrapper->packet() != tr.m_oldpacket) {
				if( !tr.isMultiNodal() && (wrapper->packet()->payload() == tr.m_oldpacket->payload())) {
					//Single-node mode, the payload in the snapshot is unchanged.
					tr.m_packet->subpackets() = wrapper->packet()->subpackets();
					if(wrapper->packet()->m_hasCollision)
						newwrapper->setBundled(false);
				}
				else
					return false;
			}
			if( !wrapper->isBundled()) {
				if( !tr.isMultiNodal())
					newwrapper->setBundled(false);
				else
					return false;
			}
			if(m_wrapper->compareAndSet(wrapper, newwrapper)) {
				return true;
			}
			continue;
		}
		if(new_bundle_state || tr.m_packet->m_hasCollision) {
			//Committing to the super node at which the snapshot was taken.
			shared_ptr<BranchPoint > branchpoint(m_wrapper);
			shared_ptr<BranchPoint > branchpoint_2nd;
			local_shared_ptr<Packet> *packet;
			int index_at_top;
			SnapshotStatus status = snapshotFromSuper(branchpoint, wrapper, &packet, &branchpoint_2nd);
			switch(status) {
			case SNAPSHOT_SUCCESS:
				break;
			case SNAPSHOT_DISTURBED:
			case SNAPSHOT_STRUCTURE_HAS_CHANGED:
				continue;
			default:
				return false;
			}
			//The super packet has to be bundled.
			if( !wrapper->isBundled() ||
				((wrapper->packet().use_count() < 3) && !m_transaction_count)) {
				//Unbundling the packet if it is partially unbundled or
				//is not actively held by other threads.
				local_shared_ptr<PacketWrapper> wrapper_2nd( *branchpoint_2nd);
				if(wrapper_2nd->packet())
					continue;
				UnbundledStatus status = unbundle(NULL, tr.m_started_time, *branchpoint,
					*branchpoint_2nd, wrapper_2nd, NULL, &wrapper);
				continue;
			}
			if( *packet != tr.m_oldpacket) {
				if( !tr.isMultiNodal() && (( *packet)->payload() == tr.m_oldpacket->payload()) &&
					(tr.m_packet->m_hasCollision == ( *packet)->m_hasCollision)) {
					//Single-node mode, the payload in the snapshot is unchanged.
					tr.m_packet->subpackets() = ( *packet)->subpackets();
				}
				else
					return false;
			}
			local_shared_ptr<PacketWrapper> newsuper(new PacketWrapper( *wrapper));
			reverseLookup(newsuper->packet(), true, tr.m_serial) = tr.m_packet;

			branchpoint->negotiate(tr.m_started_time);
			if(branchpoint->compareAndSet(wrapper, newsuper)) {
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
		case UNBUNDLE_W_NEW_VALUES:
			if(tr.isMultiNodal())
				return true;
		case UNBUNDLE_SUCCESS:
		case UNBUNDLE_PARTIALLY:
			continue;
		case UNBUNDLE_SUBVALUE_HAS_CHANGED:
			return false;
		case UNBUNDLE_DISTURBED:
		default:
			break;
		}
	}
}

template <class XN>
typename Node<XN>::UnbundledStatus
Node<XN>::unbundle(const int64_t *bundle_serial, uint64_t &time_started,
	BranchPoint &branchpoint,
	BranchPoint &subbranchpoint, const local_shared_ptr<PacketWrapper> &nullwrapper,
	const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<PacketWrapper> *newsubwrapper,
	const local_shared_ptr<Packet> *oldsuperpacket, const local_shared_ptr<PacketWrapper> *newsuperwrapper,
	bool new_sub_bunlde_state) {
	ASSERT( !nullwrapper->packet());

	branchpoint.negotiate(time_started);

	if(bundle_serial && (branchpoint.m_bundle_serial == *bundle_serial)) {
		//The node has been already bundled in the same snapshot.
//		printf("C");
		return UNBUNDLE_COLLIDED;
	}

	local_shared_ptr<PacketWrapper> wrapper(branchpoint);
	local_shared_ptr<PacketWrapper> copied;
//	printf("u");
	if( !wrapper->packet()) {
		//Unbundle all supernodes.
		shared_ptr<BranchPoint > branchpoint_super(wrapper->branchpoint());
		if( !branchpoint_super)
			return UNBUNDLE_DISTURBED; //Supernode has been destroyed.
		if(oldsuperpacket) {
			copied.reset(new PacketWrapper(( *oldsuperpacket), false));
		}
		UnbundledStatus status = unbundle(bundle_serial, time_started, *branchpoint_super, branchpoint, wrapper,
			oldsuperpacket ? &( *oldsuperpacket) : NULL, &copied, NULL, NULL, false);
		switch(status) {
		case UNBUNDLE_W_NEW_VALUES:
		case UNBUNDLE_W_NEW_SUBVALUE:
			break;
		case UNBUNDLE_SUCCESS:
		case UNBUNDLE_PARTIALLY:
			return UNBUNDLE_PARTIALLY;
		case UNBUNDLE_COLLIDED:
			return UNBUNDLE_COLLIDED;
		default:
			return UNBUNDLE_DISTURBED;
		}
		ASSERT(copied);
	}
	else {
		if( !wrapper->packet()->size())
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		if(oldsuperpacket)
			if( !wrapper->isBundled() || (wrapper->packet() != *oldsuperpacket))
				return UNBUNDLE_DISTURBED;
		//Tagging as unbundled.
		copied.reset(new PacketWrapper(wrapper->packet(), false));
		if( ! branchpoint.compareAndSet(wrapper, copied)) {
			return UNBUNDLE_DISTURBED;
		}
	}

	if( ! copied->packet()->size())
		return UNBUNDLE_SUBVALUE_HAS_CHANGED;
	local_shared_ptr<Packet> subpacket;
	typename NodeList::iterator nit = copied->packet()->subnodes()->begin();
	PacketList &subpackets( *copied->packet()->subpackets());
	for(typename PacketList::iterator pit = subpackets.begin(); pit != subpackets.end();) {
		if(( *nit)->m_wrapper.get() == &subbranchpoint) {
			subpacket = *pit;
			break;
		}
		++pit;
		++nit;
	}
	if( ! subpacket)
		return UNBUNDLE_SUBVALUE_HAS_CHANGED;

	local_shared_ptr<PacketWrapper> newsubwrapper_copied;
	if(oldsubpacket) {
		newsubwrapper_copied = *newsubwrapper;
		ASSERT( !newsubwrapper_copied->isBundled() || !newsubwrapper_copied->packet()->m_hasCollision);
		if(subpacket != *oldsubpacket) {
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		}
	}
	else {
		if( !subpacket)
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		newsubwrapper_copied.reset(new PacketWrapper(subpacket,
			!subpacket->size() || (new_sub_bunlde_state && !subpacket->m_hasCollision)));
	}
	ASSERT(newsubwrapper_copied->isBundled() || newsubwrapper_copied->packet()->size());

	if( !subbranchpoint.compareAndSet(nullwrapper, newsubwrapper_copied)) {
		if( !local_shared_ptr<PacketWrapper>(subbranchpoint)->packet())
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		return UNBUNDLE_SUCCESS;
	}
	if(newsubwrapper)
		*newsubwrapper = newsubwrapper_copied;
	local_shared_ptr<PacketWrapper> copied2;
	if(newsuperwrapper) {
		copied2 = *newsuperwrapper;
	}
	else {
//		//Erasing out-of-date subpackets on the unbundled superpacket.
//		copied2.reset(new PacketWrapper(*copied));
//		copied2->packet().reset(new Packet(*copied2->packet()));
//		copied2->packet()->subpackets().reset(new PacketList(*copied2->packet()->subpackets()));
//		PacketList &subpackets(*copied2->packet()->subpackets());
//		typename NodeList::iterator nit = copied2->packet()->subnodes()->begin();
//		for(typename PacketList::iterator pit = subpackets.begin(); pit != subpackets.end();) {
//			if(*pit) {
//				local_shared_ptr<PacketWrapper> subwrapper(*(*nit)->m_wrapper);
//				if(subwrapper->packet()) {
//					//Touch (*nit)->m_wrapper once before erasing.
//					if((*nit)->m_wrapper.get() == &subbranchpoint)
//						pit->reset();
//					else {
//						if( !bundle_serial && (subwrapper->packet() == *pit)) {
//							local_shared_ptr<PacketWrapper> newsubw(new PacketWrapper( *subwrapper));
//							if((*nit)->m_wrapper->compareAndSet(subwrapper, newsubw)) {
//								pit->reset();
//							}
//						}
//					}
//				}
//			}
//			++pit;
//			++nit;
//		}
		return UNBUNDLE_W_NEW_SUBVALUE;
	}
	ASSERT(copied2->isBundled() || copied2->packet()->size());
	if(branchpoint.compareAndSet(copied, copied2))
		return UNBUNDLE_W_NEW_VALUES;
	else
		return UNBUNDLE_W_NEW_SUBVALUE;
}

} //namespace Transactional

