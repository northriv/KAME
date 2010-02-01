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
	setCommitBundled(true);
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
		if(isCommitBundled())
			printf("CommitBundled, ");
		packet()->_print();
	}
	printf("\n");
}

template <class XN>
Node<XN>::Node() : m_wrapper(new BranchPoint()), m_packet_cache(new Cache) {
	local_shared_ptr<Packet> packet(new Packet());
	m_wrapper->reset(new PacketWrapper(packet, true));
	//Use create() for this hack.
	packet->m_payload.reset((*stl_funcPayloadCreator)(*this));
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
	local_shared_ptr<PacketWrapper> packet(*m_wrapper);
	printf("Local packet: ");
	packet->_print();
}

template <class XN>
void
Node<XN>::insert(const shared_ptr<XN> &var) {
	for(;;) {
		Transaction<XN> tr(*this);
		if(insert(tr, var))
			break;
	}
}
template <class XN>
bool
Node<XN>::insert(Transaction<XN> &tr, const shared_ptr<XN> &var) {
	local_shared_ptr<Packet> &packet(tr.m_packet);
	packet.reset(new Packet(*packet));
	packet->subpackets().reset(packet->size() ? (new PacketList(*packet->subpackets())) : (new PacketList));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->subnodes().reset(packet->size() ? (new NodeList(*packet->subnodes())) : (new NodeList));
	packet->subpackets()->resize(packet->size() + 1);
	ASSERT(packet->subnodes());
	ASSERT(std::find(packet->subnodes()->begin(), packet->subnodes()->end(), var) == packet->subnodes()->end());
	packet->subnodes()->push_back(var);
	ASSERT(packet->subpackets()->size() == packet->subnodes()->size());
//		printf("i");
	if(commit(tr, false)) {
		return true;
	}
	return false;
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
	local_shared_ptr<Packet> &packet(tr.m_packet);
	local_shared_ptr<Packet> oldsubpacket(
		var->reverseLookup(packet));
	packet.reset(new Packet(*packet));
	packet->subpackets().reset(packet->size() ? (new PacketList(*packet->subpackets())) : (new PacketList));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->subnodes().reset(packet->size() ? (new NodeList(*packet->subnodes())) : (new NodeList));
	local_shared_ptr<PacketWrapper> newsubwrapper;
	packet->m_hasCollision = false;
	unsigned int idx = 0;
	typename NodeList::iterator nit = packet->subnodes()->begin();
	for(typename PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
		ASSERT(nit != packet->subnodes()->end());
		if(nit->get() == &*var) {
			if(*pit) {
				if( !(*pit)->size()) {
					pit->reset(new Packet(**pit));
				}
				newsubwrapper.reset(new PacketWrapper(*pit, !(*pit)->m_hasCollision));
			}
			pit = packet->subpackets()->erase(pit);
			nit = packet->subnodes()->erase(nit);
		}
		else {
			if(*pit) {
				if((*pit)->size()) {
					if((*pit)->m_hasCollision)
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
	if( !newsubwrapper) {
		return commit(tr, !packet->m_hasCollision);
	}
	local_shared_ptr<PacketWrapper> nullwrapper( *var->m_wrapper);
	if(nullwrapper->packet())
		return false;
//		printf("r");
	local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(packet, !packet->m_hasCollision));
	UnbundledStatus ret = unbundle(NULL, *m_wrapper, *var->m_wrapper,
		nullwrapper, &oldsubpacket, &newsubwrapper, &tr.m_oldpacket, &newwrapper);
	if(ret == UNBUNDLE_W_NEW_VALUES) {
//			printf("%d", (int)packet->size());
		return true;
	}
	return false;
}
template <class XN>
void
Node<XN>::releaseAll() {
	for(;;) {
		Transaction<XN> tr(*this);
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
		Transaction<XN> tr(*this);
		if(swap(tr, x, y))
			break;
	}
}
template <class XN>
bool
Node<XN>::swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
	local_shared_ptr<Packet> &packet(tr.m_packet);
	packet.reset(new Packet(*packet));
	packet->subpackets().reset(packet->size() ? (new PacketList(*packet->subpackets())) : (new PacketList));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->subnodes().reset(packet->size() ? (new NodeList(*packet->subnodes())) : (new NodeList));
	unsigned int idx = 0;
	int x_idx = -1, y_idx = -1;
	for(typename NodeList::iterator nit = packet->subnodes()->begin(); nit != packet->subnodes()->end(); ++nit) {
		if(*nit == x)
			x_idx = idx;
		if(*nit == y)
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
	if(commit(tr, false)) {
		return true;
	}
	return false;
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookupWithHint(shared_ptr<BranchPoint> &branchpoint,
	local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial, Cache *cache) {
	ASSERT(packet->size());
	local_shared_ptr<PacketWrapper> wrapper(*branchpoint);
	if(wrapper->packet())
		return NULL;
	shared_ptr<BranchPoint> branchpoint_super(wrapper->branchpoint());
	if( !branchpoint_super)
		return NULL;
	local_shared_ptr<Packet> *foundpacket;
	if(branchpoint_super == packet->node().m_wrapper)
		foundpacket = &packet;
	else {
		foundpacket = reverseLookupWithHint(branchpoint_super, packet, copy_branch, tr_serial, NULL);
		if( !foundpacket)
			return NULL;
	}
	int ridx = wrapper->reverseIndex();
	if( !(*foundpacket)->size() || (ridx >= (*foundpacket)->size()))
		return NULL;
	if(copy_branch) {
		if((*foundpacket)->subpackets()->m_serial != tr_serial) {
			foundpacket->reset(new Packet(**foundpacket));
			(*foundpacket)->subpackets().reset(new PacketList(*(*foundpacket)->subpackets()));
			(*foundpacket)->subpackets()->m_serial = tr_serial;
		}
	}
	local_shared_ptr<Packet> &p((*foundpacket)->subpackets()->at(ridx));
	if( !p || (p->node().m_wrapper != branchpoint)) {
		return NULL;
	}
	if(cache) {
		cache->subpackets = (*foundpacket)->subpackets();
		cache->index = ridx;
	}
	return &p;
}
template <class XN>
local_shared_ptr<typename Node<XN>::Packet>&
Node<XN>::reverseLookup(local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial) {
	local_shared_ptr<Packet> *foundpacket;
	if(&packet->node() == this) {
		foundpacket = &packet;
	}
	else {
		local_shared_ptr<Cache> cached(m_packet_cache);
		shared_ptr<PacketList> subpackets_cached(cached->subpackets.lock());
		if(subpackets_cached && (!copy_branch || (packet->subpackets()->m_serial == tr_serial)) &&
			(subpackets_cached->m_serial == packet->subpackets()->m_serial)) {
				foundpacket = &subpackets_cached->at(cached->index);
//				printf("%%");
		}
		else {
			local_shared_ptr<Cache> newcache(new Cache);
			foundpacket = reverseLookupWithHint(m_wrapper, packet, copy_branch, tr_serial, newcache.get());
			if(foundpacket) {
//				printf("$");
			}
			else {
//				printf("!");
				foundpacket = forwardLookup(packet, copy_branch, tr_serial, newcache.get());
				ASSERT(foundpacket);
			}
			m_packet_cache = newcache;
		}
		ASSERT(&(*foundpacket)->node() == this);
	}
	if(copy_branch) {
		foundpacket->reset(new Packet(**foundpacket));
	}
//						printf("#");
	return *foundpacket;
}
template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::forwardLookup(local_shared_ptr<Packet> &packet,
	bool copy_branch, int tr_serial, Cache *cache) const {
	ASSERT(packet);
	if( !packet->subpackets())
		return NULL;
	if(copy_branch) {
		if(packet->subpackets()->m_serial != tr_serial) {
			packet.reset(new Packet(*packet));
			packet->subpackets().reset(new PacketList(*packet->subpackets()));
			packet->subpackets()->m_serial = tr_serial;
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
			if(local_shared_ptr<Packet> *p = forwardLookup(subpacket, copy_branch, tr_serial, cache)) {
				return p;
			}
		}
	}
	return NULL;
}

template <class XN>
void
Node<XN>::snapshot(Snapshot<XN> &snapshot, bool multi_nodal, Transaction<XN> *tr) const {
	local_shared_ptr<PacketWrapper> target;
	for(;;) {
		target = *m_wrapper;
		if(target->isBundled()) {
			ASSERT( !target->packet()->m_hasCollision);
			break;
		}
		if( !target->packet()) {
			// Taking a snapshot at the super node.
			shared_ptr<BranchPoint > branchpoint(m_wrapper);
			local_shared_ptr<Packet> *foundpacket = snapshotFromSuper(branchpoint, target);
			if( !foundpacket)
				continue;
			if( !(*foundpacket)->m_hasCollision || !multi_nodal) {
				snapshot.m_packet = *foundpacket;
				snapshot.m_bundled = !(*foundpacket)->m_hasCollision;
				if(tr) {
					tr->m_oldpacket = tr->m_packet;
					if(target->isBundled() && target->isCommitBundled()) {
						tr->m_packet_at_branchpoint = target->packet();
						tr->m_branchpoint = branchpoint;
					}
					else {
						tr->m_packet_at_branchpoint.reset();
						tr->m_branchpoint.reset();
					}
				}
				return;
			}
			// The packet is imperfect, and then re-bundling the subpackets.
			target = *m_wrapper;
			if( target->packet())
				continue;
			shared_ptr<BranchPoint > branchpoint_super(target->branchpoint());
			if( !branchpoint_super)
				continue;
			unbundle(NULL, *branchpoint_super, *m_wrapper, target);
			continue;
		}
		if( !multi_nodal)
			break;
		BundledStatus status = const_cast<Node*>(this)->bundle(target);
		if(status == BUNDLE_SUCCESS) {
			ASSERT( !target->packet()->m_hasCollision);
			ASSERT( target->isBundled() );
			break;
		}
	}
	snapshot.m_packet = target->packet();
	snapshot.m_bundled = target->isBundled();
	if(tr) {
		tr->m_packet_at_branchpoint.reset();
		tr->m_branchpoint.reset();
		tr->m_oldpacket = tr->m_packet;
	}
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet> *
Node<XN>::snapshotFromSuper(shared_ptr<BranchPoint > &branchpoint,
	local_shared_ptr<PacketWrapper> &target) {
	local_shared_ptr<PacketWrapper> oldwrapper(target);
	ASSERT( !target->packet());
	shared_ptr<BranchPoint > branchpoint_super(target->branchpoint());
	if( !branchpoint_super)
		return NULL; //Supernode has been destroyed.
	int ridx = target->reverseIndex();
	BranchPoint &branchpoint_this(*branchpoint);
	branchpoint = branchpoint_super;
	target = *branchpoint;
	local_shared_ptr<Packet> *foundpacket;
	if(target->packet()) {
		foundpacket = &target->packet();
	}
	else {
		foundpacket = snapshotFromSuper(branchpoint, target);
		if( !foundpacket)
			return NULL;
	}
	//Checking if it is up-to-date.
	if(branchpoint_this == oldwrapper) {
		ASSERT( (*foundpacket)->size());
		//The index might be modified by swap().
		for(int i = ridx; ;) {
			if(i >= (*foundpacket)->size())
				i = 0;
			local_shared_ptr<Packet> &p((*foundpacket)->subpackets()->at(i));
			if( p && (p->node().m_wrapper.get() == &branchpoint_this)) {
				return &p; //Bundled packet or unbundled packet w/o local packet.
			}
			++i;
		}
	}
	return NULL;
}


template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle(local_shared_ptr<PacketWrapper> &target, const int64_t *bundle_serial) {
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
			local_shared_ptr<PacketWrapper> subwrapper(*child->m_wrapper);
			if(subwrapper->packet()) {
				if( !subwrapper->isBundled()) {
					BundledStatus status = child->bundle(subwrapper, bundle_serial);
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
					UnbundledStatus status = unbundle(bundle_serial,
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
			nullwrapper.reset(new PacketWrapper(*subwrappers_org[i]));
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
	if( !tr.isBundled()) {
		ASSERT( !tr.isMultiNodal());
		new_bundle_state = false;
	}
	local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(tr.m_packet, new_bundle_state));
	ASSERT( tr.m_packet->size() || newwrapper->isBundled());
	bool unbundled = false;
	for(int retry = 0;; ++retry) {
		local_shared_ptr<PacketWrapper> wrapper( *m_wrapper);
		if(wrapper->packet()) {
			if(wrapper->packet() != tr.m_oldpacket) {
				if( !tr.isMultiNodal() && (wrapper->packet()->payload() == tr.m_oldpacket->payload())) {
					newwrapper->packet()->subpackets() = wrapper->packet()->subpackets();
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
			if(m_wrapper->compareAndSet(wrapper, newwrapper))
				return true;
			continue;
		}
		if( !unbundled && new_bundle_state && tr.m_branchpoint) {
			//Commiting to the super node at which the snapshot was taken.
			local_shared_ptr<PacketWrapper> wrapper_super = *tr.m_branchpoint;
			if(wrapper_super->isBundled()) {
				if((wrapper_super->packet() == tr.m_packet_at_branchpoint) ||
					((wrapper_super->isCommitBundled() &&
						(reverseLookup(wrapper_super->packet()) == tr.m_oldpacket)))) {
					ASSERT( !wrapper_super->packet()->m_hasCollision);
					local_shared_ptr<PacketWrapper> newwrapper_super(
						new PacketWrapper(wrapper_super->packet(), true));
					newwrapper_super->setCommitBundled(false);
					reverseLookup(newwrapper_super->packet(), true, tr.m_serial) = tr.m_packet;
					if(tr.m_branchpoint->compareAndSet(wrapper_super, newwrapper_super))
						return true;
					continue;
				}
			}
		}
		//Unbundling this node.
		shared_ptr<BranchPoint > branchpoint_super(wrapper->branchpoint());
		if( !branchpoint_super)
			continue; //Supernode has been destroyed.
		UnbundledStatus status = unbundle(NULL, *branchpoint_super, *m_wrapper, wrapper,
			tr.isMultiNodal() ? &tr.m_oldpacket : NULL, tr.isMultiNodal() ? &newwrapper : NULL);
		switch(status) {
		case UNBUNDLE_W_NEW_SUBVALUE:
		case UNBUNDLE_W_NEW_VALUES:
			if(tr.isMultiNodal())
				return true;
		case UNBUNDLE_SUCCESS:
		case UNBUNDLE_PARTIALLY:
			unbundled = true;
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
Node<XN>::unbundle(const int64_t *bundle_serial,
	BranchPoint &branchpoint,
	BranchPoint &subbranchpoint, const local_shared_ptr<PacketWrapper> &nullwrapper,
	const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<PacketWrapper> *newsubwrapper,
	const local_shared_ptr<Packet> *oldsuperpacket, const local_shared_ptr<PacketWrapper> *newsuperwrapper,
	bool new_sub_bunlde_state) {
	ASSERT( ! nullwrapper->packet());
	local_shared_ptr<PacketWrapper> wrapper(branchpoint);
	if(bundle_serial && (branchpoint.m_bundle_serial == *bundle_serial)) {
		//The node has been already bundled in the same snapshot.
//		printf("C");
		return UNBUNDLE_COLLIDED;
	}
	local_shared_ptr<PacketWrapper> copied;
//	printf("u");
	if( ! wrapper->packet()) {
		//Unbundle all supernodes.
		shared_ptr<BranchPoint > branchpoint_super(wrapper->branchpoint());
		if( ! branchpoint_super)
			return UNBUNDLE_DISTURBED; //Supernode has been destroyed.
		if(oldsuperpacket) {
			copied.reset(new PacketWrapper((*oldsuperpacket), false));
		}
		UnbundledStatus status = unbundle(bundle_serial, *branchpoint_super, branchpoint, wrapper,
			oldsuperpacket ? &(*oldsuperpacket) : NULL, &copied, NULL, NULL, false);
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
		if( ! wrapper->packet()->size())
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		if(newsuperwrapper)
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
	PacketList &subpackets(*copied->packet()->subpackets());
	for(typename PacketList::iterator pit = subpackets.begin(); pit != subpackets.end();) {
		if((*nit)->m_wrapper.get() == &subbranchpoint) {
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
		if( ! subpacket)
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		newsubwrapper_copied.reset(new PacketWrapper(subpacket,
			!subpacket->size() || (new_sub_bunlde_state && !subpacket->m_hasCollision)));
	}
	ASSERT(newsubwrapper_copied->isBundled() || newsubwrapper_copied->packet()->size());

	if( ! subbranchpoint.compareAndSet(nullwrapper, newsubwrapper_copied)) {
		if( ! local_shared_ptr<PacketWrapper>(subbranchpoint)->packet())
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
		//Erasing out-of-date subpackets on the unbundled superpacket.
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

