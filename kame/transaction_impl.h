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
Node<XN>::NullPacket::NullPacket(const shared_ptr<atomic_shared_ptr<PacketBase> > &branchpoint) :
	PacketBase(), m_branchpoint(branchpoint) {
	this->m_state = Node<XN>::Packet::PACKET_NOT_HERE;
}
template <class XN>
Node<XN>::Packet::Packet() :
	PacketBase(), m_serial(-1) {
	this->m_state = Node<XN>::Packet::PACKET_BUNDLED;
}

template <class XN>
void
Node<XN>::PacketBase::print() const {
	printf("Packet: ");
	if( ! isHere()) {
		printf("Not here, ");
		printf("Bundler:%llx, ", (uintptr_t)reinterpret_cast<const NullPacket*>(this)->branchpoint().get());
	}
	else {
		const Packet &packet = reinterpret_cast<const Packet&>(*this);
		printf("Node:%llx, ", (uintptr_t)&packet.node());

		if(packet.isBundled())
			printf("Bundled, ");
		if(packet.size()) {
			printf("%d subnodes : [ ", packet.size());
			for(unsigned int i = 0; i < packet.size(); i++) {
				if(packet.subpackets()->at(i)) {
					packet.subpackets()->at(i)->print();
				}
			}
			printf("]");
		}
	}
	printf("\n");
}

template <class XN>
Node<XN>::Node() : m_packet(new atomic_shared_ptr<PacketBase>()) {
	Packet *packet = new Packet();
	m_packet->reset(packet);
	//Use create() for this hack.
	packet->m_payload.reset((*stl_funcPayloadCreator)(*this));
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
		Snapshot<XN> shot(*this);
		if(insert(shot, var))
			break;
	}
}
template <class XN>
bool
Node<XN>::insert(const Snapshot<XN> &snapshot, const shared_ptr<XN> &var) {
	local_shared_ptr<Packet> packet(snapshot.m_packet);
	recreateNodeTree(packet);
	packet->subpackets()->resize(packet->size() + 1);
	ASSERT(packet->subnodes());
	packet->subnodes()->push_back(var);
	ASSERT(packet->subpackets()->size() == packet->subnodes()->size());
	packet->setBundled(false);
//		printf("i");
	if(commit(snapshot.m_packet, packet)) {
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
		Snapshot<XN> shot(*this);
		if(release(shot, var))
			break;
	}
}
template <class XN>
bool
Node<XN>::release(const Snapshot<XN> &snapshot, const shared_ptr<XN> &var) {
	local_shared_ptr<Packet> packet(snapshot.m_packet);
	local_shared_ptr<Packet> oldsubpacket(
		var->reverseLookup(packet));
	recreateNodeTree(packet);
	local_shared_ptr<Packet> newsubpacket;

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
			newsubpacket = *pit;
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
	ASSERT(newsubpacket);

	if( ! packet->size()) {
		packet->subpackets().reset();
		ASSERT(packet->isBundled());
	}
	else {
		packet->setBundled(false);
	}
	local_shared_ptr<Packet> nullpacket(*var->m_packet);
	if(nullpacket->isHere())
		return false;
//		printf("r");
	UnbundledStatus ret = unbundle(*m_packet, *var->m_packet,
		nullpacket, &oldsubpacket, &newsubpacket, &snapshot.m_packet, &packet);
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
		Snapshot<XN> shot(*this);
		if( ! shot.size())
			break;
		shared_ptr<const NodeList> list(shot.list());
		release(shot, list->front());
	}
}
template <class XN>
void
Node<XN>::swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
	for(;;) {
		Snapshot<XN> shot(*this);
		if(swap(shot, x, y))
			break;
	}
}
template <class XN>
bool
Node<XN>::swap(const Snapshot<XN> &snapshot, const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
	local_shared_ptr<Packet> packet(snapshot.m_packet);
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
	if(commit(snapshot.m_packet, packet)) {
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
		ASSERT((*foundpacket)->isBundled());
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
		ASSERT((*foundpacket)->m_serial == tr_serial);
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
		for(int i = 0;; ++i) {
			ASSERT(i < 2);
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
			bool ret = forwardLookup(packet, hint);
			ASSERT(ret);
			m_lookupHint = hint;
		}
	}

	if(copy_branch && ((*foundpacket)->m_serial != tr_serial)) {
		foundpacket->reset(new Packet(**foundpacket));
		(*foundpacket)->m_serial = tr_serial;
	}
	ASSERT((*foundpacket)->isBundled());
//						printf("#");
	return *foundpacket;
}
template <class XN>
bool
Node<XN>::forwardLookup(const local_shared_ptr<Packet> &packet, local_shared_ptr<LookupHint> &hint) const {
	ASSERT(packet);
	if( ! packet->subpackets())
		return false;
	for(unsigned int i = 0; i < packet->subnodes()->size(); i++) {
		if(packet->subnodes()->at(i).get() == this) {
			hint.reset(new LookupHint);
			hint->m_index = i;
			hint->m_superNodeList = packet->subnodes();
			return true;
		}
	}
	for(unsigned int i = 0; i < packet->subnodes()->size(); i++) {
		const local_shared_ptr<Packet> &subpacket(packet->subpackets()->at(i));
		 // Checking if the branch (including the finding packet for the node) is up-to-date.
		if(subpacket && subpacket->isBundled()) {
			if(forwardLookup(subpacket, hint)) {
				return true;
			}
		}
	}
	return false;
}
template <class XN>
void
Node<XN>::snapshot(local_shared_ptr<Packet> &target) const {
	for(;;) {
		local_shared_ptr<PacketBase> packet = *m_packet;
		if(packet->isBundled()) {
			target = reinterpret_cast<local_shared_ptr<Packet> &>(packet);
			return;
		}
		if( ! packet->isHere()) {
			shared_ptr<atomic_shared_ptr<PacketBase> > branchpoint(m_packet);
			if(trySnapshotSuper(*branchpoint, packet)) {
				target = reinterpret_cast<local_shared_ptr<Packet> &>(packet);
				if( ! target->size())
					continue;
				target = const_cast<Node*>(this)->reverseLookup(target);
				ASSERT(target->isBundled());
				return;
			}
			continue;
		}
		target = reinterpret_cast<local_shared_ptr<Packet> &>(packet);
		BundledStatus status = const_cast<Node*>(this)->bundle(target);
		if(status == BUNDLE_SUCCESS)
			return;
	}
}
template <class XN>
inline bool
Node<XN>::trySnapshotSuper(atomic_shared_ptr<PacketBase> &branchpoint, local_shared_ptr<PacketBase> &target) {
	local_shared_ptr<PacketBase> oldpacket(target);
	ASSERT( ! target->isHere());
	shared_ptr<atomic_shared_ptr<PacketBase> > branchpoint_super(
		reinterpret_cast<local_shared_ptr<NullPacket> &>(target)->branchpoint());
	if( ! branchpoint_super)
		return false; //Supernode has been destroyed.
	target = *branchpoint_super;
	if(target->isBundled())
		return true;
	if( ! target->isHere()) {
		if( ! trySnapshotSuper(*branchpoint_super, target))
			return false;
	}
	ASSERT(reinterpret_cast<local_shared_ptr<Packet> &>(target)->size());
	if(branchpoint == oldpacket) {
		ASSERT( ! oldpacket->isHere());
		return true; //Up-to-date unbundled packet w/o local packet.
	}
	return false;
}

template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle(local_shared_ptr<Packet> &target) {
	ASSERT( ! target->isBundled() && target->isHere());
	ASSERT(target->size());
	local_shared_ptr<Packet> prebundled(new Packet(*target));
	//copying all sub-packets from nodes to the new packet.
	prebundled->subpackets().reset(new PacketList(*prebundled->subpackets()));
	shared_ptr<PacketList> &packets(prebundled->subpackets());
	std::vector<local_shared_ptr<PacketBase> > subpacket_org(target->size());
	for(unsigned int i = 0; i < packets->size(); ++i) {
		shared_ptr<Node> child(prebundled->subnodes()->at(i));
		local_shared_ptr<Packet> &subpacket_new(packets->at(i));
		for(;;) {
			local_shared_ptr<PacketBase> packetonnode(*child->m_packet);
			if(packetonnode->isHere()) {
				local_shared_ptr<Packet> &cp(reinterpret_cast<local_shared_ptr<Packet> &>(packetonnode));
				if( ! cp->isBundled()) {
					BundledStatus status = child->bundle(cp);
					if((status == BUNDLE_DISTURBED) &&
						(target == *m_packet))
							continue;
					if(status != BUNDLE_SUCCESS)
						return BUNDLE_DISTURBED;
				}
				subpacket_new = cp;
				ASSERT(cp->isBundled());
			}
			else {
				local_shared_ptr<NullPacket> &cp(reinterpret_cast<local_shared_ptr<NullPacket> &>(packetonnode));
				shared_ptr<atomic_shared_ptr<PacketBase> > branchpoint(cp->branchpoint());
				if( ! branchpoint)
					return BUNDLE_DISTURBED; //Supernode has been destroyed.
				if(branchpoint != m_packet) {
					//bundled by another node.
					UnbundledStatus status = unbundle(*branchpoint, *child->m_packet, cp, NULL, &subpacket_new);
					if((status == UNBUNDLE_SUCCESS) || (status == UNBUNDLE_DISTURBED))
						if(target == *m_packet)
							continue;
					if((status != UNBUNDLE_W_NEW_SUBVALUE) && (status != UNBUNDLE_W_NEW_VALUES))
						return BUNDLE_DISTURBED;
					packetonnode = subpacket_new;
					ASSERT(subpacket_new);
				}
			}
			if( ! subpacket_new) {
//				printf("?");
				ASSERT(target != *m_packet);
				//m_packet has changed, bundled by the other thread.
				return BUNDLE_DISTURBED;
			}
			subpacket_org[i] = packetonnode;
			if(subpacket_new->size()) {
				if((subpacket_new->subnodes()->m_superNodeList.lock() != prebundled->subnodes()) ||
					(subpacket_new->subnodes()->m_index != i)) {
					subpacket_new.reset(new Packet(*subpacket_new));
					subpacket_new->subpackets().reset(new PacketList(*subpacket_new->subpackets()));
					subpacket_new->subnodes().reset(new NodeList(*subpacket_new->subnodes()));
					subpacket_new->subnodes()->m_superNodeList = prebundled->subnodes();
					subpacket_new->subnodes()->m_index = i;
				}
			}
			ASSERT(&subpacket_new->node() == child.get());
			break;
		}
		ASSERT(subpacket_new);
		ASSERT(subpacket_new->isBundled());
		if(subpacket_new->size()) {
			ASSERT(subpacket_new->subnodes()->m_superNodeList.lock() == prebundled->subnodes());
			ASSERT(subpacket_new->subnodes()->m_index == i);
		}
	}
	//First checkpoint.
	if( ! m_packet->compareAndSet(target, prebundled)) {
		return BUNDLE_DISTURBED;
	}
	//clearing all packets on sub-nodes if not modified.
	for(unsigned int i = 0; i < prebundled->size(); i++) {
		shared_ptr<Node> child(prebundled->subnodes()->at(i));
		local_shared_ptr<PacketBase> nullpacket(new NullPacket(m_packet));
		//Second checkpoint, the written bundle is valid or not.
		if( ! child->m_packet->compareAndSet(subpacket_org[i], nullpacket)) {
			return BUNDLE_DISTURBED;
		}
	}
	target.reset(new Packet(*prebundled));
	target->setBundled(true);
	//Finally, tagging as bundled.
	if( ! m_packet->compareAndSet(prebundled, target))
		return BUNDLE_DISTURBED;
	return BUNDLE_SUCCESS;
}

template <class XN>
bool
Node<XN>::commit(const local_shared_ptr<Packet> &oldpacket, local_shared_ptr<Packet> &newpacket) {
	for(;;) {
		local_shared_ptr<PacketBase> packet(*m_packet);
		if(packet->isHere()) {
			if(packet != oldpacket)
				return false;
			if(m_packet->compareAndSet(oldpacket, newpacket))
				return true;
			continue;
		}
		local_shared_ptr<NullPacket> & nullpacket(reinterpret_cast<local_shared_ptr<NullPacket> &>(packet));
		shared_ptr<atomic_shared_ptr<PacketBase> > branchpoint_super(nullpacket->branchpoint());
		if( ! branchpoint_super)
			continue; //Supernode has been destroyed.
		UnbundledStatus status = unbundle(*branchpoint_super, *m_packet, nullpacket, &oldpacket, &newpacket);
		switch(status) {
		case UNBUNDLE_W_NEW_SUBVALUE:
		case UNBUNDLE_W_NEW_VALUES:
			return true;
		case UNBUNDLE_SUBVALUE_HAS_CHANGED:
			return false;
		case UNBUNDLE_SUCCESS:
		case UNBUNDLE_DISTURBED:
		default:
			continue;
		}
	}
}

template <class XN>
typename Node<XN>::UnbundledStatus
Node<XN>::unbundle(atomic_shared_ptr<PacketBase> &branchpoint,
	atomic_shared_ptr<PacketBase> &subbranchpoint, const local_shared_ptr<NullPacket> &nullpacket,
	const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<Packet> *newsubpacket,
	const local_shared_ptr<Packet> *oldsuperpacket, const local_shared_ptr<Packet> *newsuperpacket) {
	ASSERT( ! nullpacket->isHere());
	local_shared_ptr<PacketBase> packet(branchpoint);
	local_shared_ptr<Packet> copied;
//	printf("u");
	if( ! packet->isHere()) {
		//Unbundle all supernodes.
		if(oldsuperpacket) {
			copied.reset(new Packet(**oldsuperpacket));
			copied->setBundled(false);
		}
		local_shared_ptr<NullPacket> &p(reinterpret_cast<local_shared_ptr<NullPacket> &>(packet));
		shared_ptr<atomic_shared_ptr<PacketBase> > branchpoint_super(p->branchpoint());
		if( ! branchpoint_super)
			return UNBUNDLE_DISTURBED; //Supernode has been destroyed.
		UnbundledStatus ret = unbundle(*branchpoint_super, branchpoint, p,
			oldsuperpacket ? oldsuperpacket : NULL, &copied);
		if((ret != UNBUNDLE_W_NEW_SUBVALUE) || (ret != UNBUNDLE_W_NEW_VALUES))
			return UNBUNDLE_DISTURBED;
		ASSERT(copied);
	}
	else {
		local_shared_ptr<Packet> &p(reinterpret_cast<local_shared_ptr<Packet> &>(packet));
		if( ! p->size())
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		copied.reset(new Packet(*p));
		copied->setBundled(false);
		if(newsuperpacket)
			if(p != *oldsuperpacket)
				return UNBUNDLE_DISTURBED;
		//Tagging as unbundled.
		if( ! branchpoint.compareAndSet(packet, copied)) {
			return UNBUNDLE_DISTURBED;
		}
	}

	if( ! copied->size())
		return UNBUNDLE_SUBVALUE_HAS_CHANGED;
	local_shared_ptr<Packet> subpacket;
	typename NodeList::iterator nit = copied->subnodes()->begin();
	for(typename PacketList::iterator pit = copied->subpackets()->begin(); pit != copied->subpackets()->end();) {
		if((*nit)->m_packet.get() == &subbranchpoint) {
			subpacket = *pit;
			break;
		}
		++pit;
		++nit;
	}
	if( ! subpacket)
		return UNBUNDLE_SUBVALUE_HAS_CHANGED;

	local_shared_ptr<Packet> newsubpacket_copied;
	if(oldsubpacket) {
		newsubpacket_copied = *newsubpacket;
		if(subpacket != *oldsubpacket) {
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		}
	}
	else {
		if( ! subpacket)
			return UNBUNDLE_DISTURBED;
		newsubpacket_copied = subpacket;
		if(newsubpacket_copied->size()) {
			newsubpacket_copied.reset(new Packet(*newsubpacket_copied));
//			newsubpacket_copied->setBundled(false);
		}
	}

	if( ! subbranchpoint.compareAndSet(nullpacket, newsubpacket_copied)) {
		if( ! local_shared_ptr<Packet>(subbranchpoint)->isHere())
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		return UNBUNDLE_SUCCESS;
	}
	*newsubpacket = newsubpacket_copied;
	local_shared_ptr<Packet> copied2;
	if(newsuperpacket) {
		copied2 = *newsuperpacket;
	}
	else {
		//Erasing out-of-date subpackets on the unbundled superpacket.
		copied2.reset(new Packet(*copied));
		copied2->subpackets().reset(new PacketList(*copied2->subpackets()));
		typename NodeList::iterator nit = copied2->subnodes()->begin();
		for(typename PacketList::iterator pit = copied2->subpackets()->begin(); pit != copied2->subpackets()->end();) {
			if(*pit) {
				local_shared_ptr<Packet> subpacket(*(*nit)->m_packet);
				if(subpacket->isHere()) {
					//Touch (*nit)->m_packet once before erasing.
					if(((*nit)->m_packet.get() == &subbranchpoint) ||
						(*nit)->m_packet->compareAndSet(*pit, local_shared_ptr<Packet>(new Packet(**pit)))) {
						pit->reset();
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

