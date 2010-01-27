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
atomic<int> Transaction<XN>::s_serial = 0;

template <class XN>
Node<XN>::Packet::Packet() : m_serial(-1) {}

template <class XN>
void
Node<XN>::Packet::print() const {
	printf("Packet: ");
	printf("Node:%llx, ", (uintptr_t)&node());
	if(size()) {
		printf("%d subnodes : [ ", size());
		for(unsigned int i = 0; i < size(); i++) {
			if(subpackets()->at(i)) {
				subpackets()->at(i)->print();
			}
		}
		printf("]");
	}
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const local_shared_ptr<Packet> &x, bool bundled) :
	m_branchpoint(), m_packet(x), m_state(0) {
		setBundled(bundled);
		setCommitBundled(true);
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const shared_ptr<atomic_shared_ptr<PacketWrapper> > &bp) :
	m_branchpoint(bp), m_packet(), m_state(0) {}

template <class XN>
void
Node<XN>::PacketWrapper::print() const {
	printf("PacketWrapper: ");
	if( ! packet()) {
		printf("Not here, ");
		printf("Bundler:%llx, ", (uintptr_t)branchpoint().get());
	}
	else {
		if(isBundled())
			printf("Bundled, ");
		packet()->print();
	}
	printf("\n");
}

template <class XN>
Node<XN>::Node() : m_packet(new atomic_shared_ptr<PacketWrapper>(new PacketWrapper())) {
	local_shared_ptr<Packet> packet(new Packet());
	local_shared_ptr<PacketWrapper>(*m_packet)->packet() = packet;
	//Use create() for this hack.
	packet->m_payload.reset((*stl_funcPayloadCreator)(*this));
	*stl_funcPayloadCreator = NULL;
}
template <class XN>
Node<XN>::~Node() {
	releaseAll();
}
template <class XN>
void
Node<XN>::recreateNodeTree(local_shared_ptr<Packet> &packet) {
	unsigned int idx = 0;
	packet.reset(new Packet(*packet));
	packet->subpackets().reset(packet->size() ? (new PacketList(*packet->subpackets())) : (new PacketList));
	packet->subnodes().reset(packet->size() ? (new NodeList(*packet->subnodes())) : (new NodeList));
	for(typename PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
		if((*pit)->size()) {
			pit->reset(new Packet(**pit));
			(*pit)->subpackets().reset(new PacketList(*(*pit)->subpackets()));
			(*pit)->subnodes().reset(new NodeList(*(*pit)->subnodes()));
			(*pit)->subnodes()->m_superNodeList = packet->subnodes();
			ASSERT((*pit)->subnodes()->m_index == idx);
			recreateNodeTree(*pit);
		}
		++pit;
		++idx;
	}
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
	recreateNodeTree(packet);
	packet->subpackets()->resize(packet->size() + 1);
	ASSERT(packet->subnodes());
	packet->subnodes()->push_back(var);
	ASSERT(packet->subpackets()->size() == packet->subnodes()->size());
//		printf("i");
	if(commit(tr, false)) {
		local_shared_ptr<LookupHint> hint(new LookupHint);
		hint->m_index = packet->size() - 1;
		hint->m_superNodeList = packet->subnodes();
		var->m_lookupHint = hint;
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
	recreateNodeTree(packet);
	local_shared_ptr<PacketWrapper> newsubwrapper(new PacketWrapper());
	newsubwrapper->setBundled(true);
	unsigned int idx = 0;
	typename NodeList::iterator nit = packet->subnodes()->begin();
	for(typename PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
		ASSERT(nit != packet->subnodes()->end());
		if(nit->get() == &*var) {
			if((*pit)->size()) {
				(*pit)->subnodes()->m_superNodeList.reset();
			}
			else {
				pit->reset(new Packet(**pit));
			}
			newsubwrapper->packet() = *pit;
			pit = packet->subpackets()->erase(pit);
			nit = packet->subnodes()->erase(nit);
		}
		else {
			if((*pit)->size()) {
				(*pit)->subnodes()->m_index = idx;
			}
			++nit;
			++pit;
			++idx;
		}
	}
	ASSERT(newsubwrapper->packet());

	if( ! packet->size()) {
		packet->subpackets().reset();
	}
	local_shared_ptr<PacketWrapper> nullwrapper( *var->m_packet);
	if(nullwrapper->packet())
		return false;
//		printf("r");
	local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(packet, true));
	UnbundledStatus ret = unbundle(NULL, *m_packet, *var->m_packet,
		nullwrapper, &oldsubpacket, &newsubwrapper, &tr.m_oldpacket, &newwrapper);
	if(ret == UNBUNDLE_W_NEW_VALUES) {
//			printf("%d", (int)packet->size());
		var->m_lookupHint.reset();
		return true;
	}
	return false;
}
template <class XN>
void
Node<XN>::releaseAll() {
	for(;;) {
		Transaction<XN> tr(*this);
		if( ! tr.size())
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
	recreateNodeTree(packet);
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
	if(px->size()) {
		px->subnodes()->m_index = y_idx;
		ASSERT(px->subnodes()->m_superNodeList.lock() == packet->subnodes());
	}
	if(py->size()) {
		py->subnodes()->m_index = x_idx;
		ASSERT(py->subnodes()->m_superNodeList.lock() == packet->subnodes());
	}
	if(commit(tr)) {
		{
			local_shared_ptr<LookupHint> hint(new LookupHint);
			hint->m_index = y_idx;
			hint->m_superNodeList = packet->subnodes();
			x->m_lookupHint = hint;
		}
		{
			local_shared_ptr<LookupHint> hint(new LookupHint);
			hint->m_index = x_idx;
			hint->m_superNodeList = packet->subnodes();
			y->m_lookupHint = hint;
		}
		return true;
	}
	return false;
}

template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::NodeList::reverseLookup(local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial) {
	local_shared_ptr<Packet> *foundpacket;
	if(packet->subnodes().get() == this) {
		foundpacket = &packet;
	}
	else {
		shared_ptr<NodeList> superlist = m_superNodeList.lock();
		if( ! superlist)
			return NULL;
		foundpacket =
			superlist->reverseLookup(packet, copy_branch, tr_serial);
		if( ! foundpacket)
			return NULL;
		if((*foundpacket)->size() <= m_index)
			return NULL;
		foundpacket = &(*foundpacket)->subpackets()->at(m_index);
		if((*foundpacket)->subnodes().get() != this)
			return NULL;
	}
	if(copy_branch) {
		if((*foundpacket)->subpackets()->m_serial != tr_serial) {
			if((*foundpacket)->m_serial != tr_serial) {
				foundpacket->reset(new Packet(**foundpacket));
				(*foundpacket)->m_serial = tr_serial;
			}
			(*foundpacket)->subpackets().reset(new PacketList(*(*foundpacket)->subpackets()));
			(*foundpacket)->subpackets()->m_serial = tr_serial;
		}
//		ASSERT((*foundpacket)->m_serial == tr_serial);
	}
	return foundpacket;
}
template <class XN>
local_shared_ptr<typename Node<XN>::Packet>&
Node<XN>::reverseLookup(local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial) const {
	local_shared_ptr<Packet> *foundpacket;
	if(&packet->node() == this) {
		foundpacket = &packet;
	}
	else {
		ASSERT(packet->size());
		local_shared_ptr<LookupHint> hint(m_lookupHint);
		for(;;) {
			if(hint) {
				shared_ptr<NodeList> supernodelist = hint->m_superNodeList.lock();
				if(supernodelist &&
					((hint->m_index < supernodelist->size()) &&
						(supernodelist->at(hint->m_index).get() == this))) {
					local_shared_ptr<Packet>* superpacket = supernodelist->reverseLookup(packet, copy_branch, tr_serial);
					if(superpacket &&
						((*superpacket)->size() > hint->m_index) ) {
						foundpacket = &(*superpacket)->subpackets()->at(hint->m_index);
						if(&(*foundpacket)->node() == this) {
							break;
						}
					}
				}
			}
		//		printf("!");
			foundpacket = forwardLookup(packet, hint);
			ASSERT(foundpacket);
			ASSERT(&(*foundpacket)->node() == this);
			m_lookupHint = hint;
			break;
		}
	}

	if(copy_branch && ((*foundpacket)->m_serial != tr_serial)) {
		foundpacket->reset(new Packet(**foundpacket));
		(*foundpacket)->m_serial = tr_serial;
	}
//						printf("#");
	return *foundpacket;
}
template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::forwardLookup(const local_shared_ptr<Packet> &packet, local_shared_ptr<LookupHint> &hint) const {
	ASSERT(packet);
	if( ! packet->subpackets())
		return NULL;
	for(unsigned int i = 0; i < packet->subnodes()->size(); i++) {
		if(packet->subnodes()->at(i).get() == this) {
			hint.reset(new LookupHint);
			hint->m_index = i;
			hint->m_superNodeList = packet->subnodes();
			return &packet->subpackets()->at(i);
		}
	}
	for(unsigned int i = 0; i < packet->subnodes()->size(); i++) {
		const local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
		 // Checking if the branch (including the finding packet for the node) is up-to-date.
		if(subpacket) {
			if(local_shared_ptr<Packet> *p = forwardLookup(subpacket, hint)) {
				return p;
			}
		}
	}
	return NULL;
}
template <class XN>
void
Node<XN>::snapshot(Snapshot<XN> &snapshot) const {
	local_shared_ptr<PacketWrapper> target;
	for(;;) {
		target = *m_packet;
		if(target->isBundled()) {
			break;
		}
		if( ! target->packet()) {
			shared_ptr<atomic_shared_ptr<PacketWrapper> > branchpoint(m_packet);
			if(trySnapshotSuper(branchpoint, target)) {
				if( ! target->packet()->size())
					continue;
				snapshot.m_packet = reverseLookup(target->packet());
				return;
			}
			continue;
		}
		BundledStatus status = const_cast<Node*>(this)->bundle(target, target);
		if(status == BUNDLE_SUCCESS) {
			break;
		}
	}
	snapshot.m_packet = target->packet();
}

template <class XN>
void
Node<XN>::snapshot(Transaction<XN> &tr) const {
	local_shared_ptr<PacketWrapper> target;
	for(;;) {
		target = *m_packet;
		if(target->isBundled()) {
			break;
		}
		if( ! target->packet()) {
			shared_ptr<atomic_shared_ptr<PacketWrapper> > branchpoint(m_packet);
			if(trySnapshotSuper(branchpoint, target)) {
				if( ! target->packet()->size())
					continue;
				tr.m_packet = reverseLookup(target->packet());
				tr.m_oldpacket = tr.m_packet;
				if(target->isBundled() && target->isCommitBundled()) {
					tr.m_packet_at_branchpoint = target->packet();
					tr.m_branchpoint = branchpoint;
				}
				else {
					tr.m_packet_at_branchpoint.reset();
					tr.m_branchpoint.reset();
				}
				return;
			}
			continue;
		}
		BundledStatus status = const_cast<Node*>(this)->bundle(target, target);
		if(status == BUNDLE_SUCCESS) {
			break;
		}
	}
	tr.m_packet = target->packet();
	tr.m_packet_at_branchpoint.reset();
	tr.m_branchpoint.reset();
	tr.m_oldpacket = tr.m_packet;
}

template <class XN>
bool
Node<XN>::trySnapshotSuper(shared_ptr<atomic_shared_ptr<PacketWrapper> > &branchpoint,
	local_shared_ptr<PacketWrapper> &target) {
	local_shared_ptr<PacketWrapper> oldpacket(target);
	ASSERT( ! target->packet());
	atomic_shared_ptr<PacketWrapper> &branchpoint_this(*branchpoint);
	shared_ptr<atomic_shared_ptr<PacketWrapper> > branchpoint_super(target->branchpoint());
	if( ! branchpoint_super)
		return false; //Supernode has been destroyed.
	branchpoint = branchpoint_super;
	target = *branchpoint;
	if(target->isBundled())
		return true;
	if( ! target->packet()) {
		if( ! trySnapshotSuper(branchpoint, target))
			return false;
	}
	ASSERT(target->packet()->size());
	if(branchpoint_this == oldpacket) {
		ASSERT( ! oldpacket->packet());
		return true; //Up-to-date unbundled packet w/o local packet.
	}
	return false;
}

template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle(const local_shared_ptr<PacketWrapper> &rootwrapper, local_shared_ptr<PacketWrapper> &target) {
	ASSERT( ! target->isBundled() && target->packet());
	ASSERT(target->packet()->size());
	local_shared_ptr<Packet> packet(new Packet( *target->packet()));
	local_shared_ptr<PacketWrapper> oldwrapper(target);
	target.reset(new PacketWrapper(packet, false));
	//copying all sub-packets from nodes to the new packet.
	packet->subpackets().reset(new PacketList( *packet->subpackets()));
	shared_ptr<PacketList> &packets(packet->subpackets());
	shared_ptr<NodeList> &subnodes(packet->subnodes());
	std::vector<local_shared_ptr<PacketWrapper> > subwrappers_org(packets->size());
	for(unsigned int i = 0; i < packets->size(); ++i) {
		shared_ptr<Node> child(subnodes->at(i));
		local_shared_ptr<Packet> &subpacket_new(packets->at(i));
		for(;;) {
			local_shared_ptr<PacketWrapper> subwrapper(*child->m_packet);
			if(subwrapper->packet()) {
				if( ! subwrapper->isBundled()) {
					BundledStatus status = child->bundle((oldwrapper == rootwrapper) ? target : rootwrapper, subwrapper);
					if((status == BUNDLE_DISTURBED) &&
						(target == *m_packet))
							continue;
					if(status != BUNDLE_SUCCESS)
						return BUNDLE_DISTURBED;
				}
				subpacket_new = subwrapper->packet();
				ASSERT(subwrapper->isBundled());
			}
			else {
				shared_ptr<atomic_shared_ptr<PacketWrapper> > branchpoint(subwrapper->branchpoint());
				if( ! branchpoint)
					return BUNDLE_DISTURBED; //Supernode has been destroyed.
				if(branchpoint != m_packet) {
					//bundled by another node.
					local_shared_ptr<PacketWrapper> subwrapper_new;
					UnbundledStatus status = unbundle(rootwrapper->packet()->node().m_packet.get(),
						*branchpoint, *child->m_packet, subwrapper, NULL, &subwrapper_new);
					if(status == UNBUNDLE_COLLIDED) {
						//The subpacket has already been included in the snapshot.
						subpacket_new = child->reverseLookup(rootwrapper->packet());
						fprintf(stderr, "Collision\n");
					}
					else {
						if((status != UNBUNDLE_W_NEW_SUBVALUE) && (status != UNBUNDLE_W_NEW_VALUES)) {
							if(target == *m_packet)
								continue;
							else
								return BUNDLE_DISTURBED;
						}
						subwrapper = subwrapper_new;
						subpacket_new = subwrapper_new->packet();
					}
					ASSERT(subwrapper->packet());
				}
			}
			if( ! subpacket_new) {
//				printf("?");
				ASSERT(target != *m_packet);
				//m_packet has changed, bundled by the other thread.
				return BUNDLE_DISTURBED;
			}
			subwrappers_org[i] = subwrapper;
			if(subpacket_new->size()) {
				if((subpacket_new->subnodes()->m_superNodeList.lock() != subnodes) ||
					(subpacket_new->subnodes()->m_index != i)) {
					subpacket_new.reset(new Packet(*subpacket_new));
					subpacket_new->subpackets().reset(new PacketList(*subpacket_new->subpackets()));
					subpacket_new->subnodes().reset(new NodeList(*subpacket_new->subnodes()));
					subpacket_new->subnodes()->m_superNodeList = subnodes;
					subpacket_new->subnodes()->m_index = i;
				}
			}
			ASSERT(&subpacket_new->node() == child.get());
			break;
		}
		ASSERT(subpacket_new);
		if(subpacket_new->size()) {
			ASSERT(subpacket_new->subnodes()->m_superNodeList.lock() == subnodes);
			ASSERT(subpacket_new->subnodes()->m_index == i);
		}
	}
	//First checkpoint.
	if( ! m_packet->compareAndSet(oldwrapper, target)) {
		return BUNDLE_DISTURBED;
	}
	//clearing all packets on sub-nodes if not modified.
	for(unsigned int i = 0; i < subnodes->size(); i++) {
		shared_ptr<Node> child(subnodes->at(i));
		local_shared_ptr<PacketWrapper> nullwrapper(new PacketWrapper(m_packet));
		//Second checkpoint, the written bundle is valid or not.
		if( ! child->m_packet->compareAndSet(subwrappers_org[i], nullwrapper)) {
			return BUNDLE_DISTURBED;
		}
	}
	oldwrapper = target;
	target.reset(new PacketWrapper(packet, true));
	//Finally, tagging as bundled.
	if( ! m_packet->compareAndSet(oldwrapper, target))
		return BUNDLE_DISTURBED;
	return BUNDLE_SUCCESS;
}

template <class XN>
bool
Node<XN>::commit(Transaction<XN> &tr, bool new_bundle_state) {
	local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(tr.m_packet, new_bundle_state));
	bool unbundled = false;
	for(int retry = 0;; ++retry) {
		local_shared_ptr<PacketWrapper> wrapper(*m_packet);
		if(wrapper->packet()) {
			if(wrapper->packet() != tr.m_oldpacket)
				return false;
			if( !wrapper->isBundled())
				return false;
			if(m_packet->compareAndSet(wrapper, newwrapper))
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
		shared_ptr<atomic_shared_ptr<PacketWrapper> > branchpoint_super(wrapper->branchpoint());
		if( ! branchpoint_super)
			continue; //Supernode has been destroyed.
		UnbundledStatus status = unbundle(NULL, *branchpoint_super, *m_packet, wrapper,
			&tr.m_oldpacket, &newwrapper);
		switch(status) {
		case UNBUNDLE_SUCCESS:
		case UNBUNDLE_PARTIALLY:
			unbundled = true;
			continue;
		case UNBUNDLE_W_NEW_SUBVALUE:
		case UNBUNDLE_W_NEW_VALUES:
			return true;
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
Node<XN>::unbundle(const atomic_shared_ptr<PacketWrapper> *bundleroot, atomic_shared_ptr<PacketWrapper> &branchpoint,
	atomic_shared_ptr<PacketWrapper> &subbranchpoint, const local_shared_ptr<PacketWrapper> &nullwrapper,
	const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<PacketWrapper> *newsubwrapper,
	const local_shared_ptr<Packet> *oldsuperpacket, const local_shared_ptr<PacketWrapper> *newsuperwrapper) {
	ASSERT( ! nullwrapper->packet());
	local_shared_ptr<PacketWrapper> wrapper(branchpoint);
	local_shared_ptr<PacketWrapper> copied;
//	printf("u");
	if( ! wrapper->packet()) {
		//Unbundle all supernodes.
		shared_ptr<atomic_shared_ptr<PacketWrapper> > branchpoint_super(wrapper->branchpoint());
		if( ! branchpoint_super)
			return UNBUNDLE_DISTURBED; //Supernode has been destroyed.
		if(bundleroot == branchpoint_super.get()) {
			fprintf(stderr, "Collision\n");
			return UNBUNDLE_COLLIDED;
		}
		if(oldsuperpacket) {
			copied.reset(new PacketWrapper((*oldsuperpacket), false));
		}
		UnbundledStatus status = unbundle(bundleroot, *branchpoint_super, branchpoint, wrapper,
			oldsuperpacket ? &(*oldsuperpacket) : NULL, &copied);
		switch(status) {
		case UNBUNDLE_W_NEW_SUBVALUE:
		case UNBUNDLE_W_NEW_VALUES:
//			break;
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
		if((*nit)->m_packet.get() == &subbranchpoint) {
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
		if(subpacket != *oldsubpacket) {
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		}
	}
	else {
		if( ! subpacket)
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		newsubwrapper_copied.reset(new PacketWrapper(subpacket, true));
	}

	if( ! subbranchpoint.compareAndSet(nullwrapper, newsubwrapper_copied)) {
		if( ! local_shared_ptr<PacketWrapper>(subbranchpoint)->packet())
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		return UNBUNDLE_SUCCESS;
	}
	*newsubwrapper = newsubwrapper_copied;
	local_shared_ptr<PacketWrapper> copied2;
	if(newsuperwrapper) {
		copied2 = *newsuperwrapper;
	}
	else {
		//Erasing out-of-date subpackets on the unbundled superpacket.
		copied2.reset(new PacketWrapper(*copied));
		copied2->packet().reset(new Packet(*copied2->packet()));
		copied2->packet()->subpackets().reset(new PacketList(*copied2->packet()->subpackets()));
		PacketList &subpackets(*copied2->packet()->subpackets());
		typename NodeList::iterator nit = copied2->packet()->subnodes()->begin();
		for(typename PacketList::iterator pit = subpackets.begin(); pit != subpackets.end();) {
			if(*pit) {
				local_shared_ptr<PacketWrapper> subwrapper(*(*nit)->m_packet);
				if(subwrapper->packet()) {
					//Touch (*nit)->m_packet once before erasing.
					if((*nit)->m_packet.get() == &subbranchpoint)
						pit->reset();
					else {
						if(subwrapper->packet() == *pit) {
							local_shared_ptr<PacketWrapper> newsubw(new PacketWrapper( *subwrapper));
							if((*nit)->m_packet->compareAndSet(subwrapper, newsubw)) {
								pit->reset();
							}
						}
					}
				}
			}
			++pit;
			++nit;
		}
	}
	if(branchpoint.compareAndSet(copied, copied2))
		return UNBUNDLE_W_NEW_VALUES;
	else
		return UNBUNDLE_W_NEW_SUBVALUE;
}

} //namespace Transactional

