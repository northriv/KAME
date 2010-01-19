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

atomic<int> Transaction::s_serial = 0;

Node::Packet::Packet(Node *supernode) : m_state(PACKET_BUNDLED), m_payload(),
	m_supernode(supernode), m_serial(-1) {
}
Node::Packet::~Packet() {

}
void
Node::Packet::print() {
	printf("Packet: ");
	printf("Super:%llx, ", (uintptr_t)&supernode());
	if( ! isHere())
		printf("Not here, ");
	else {
		printf("Node:%llx, ", (uintptr_t)&node());

		if(isBundled())
			printf("Bundled, ");
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
	printf("\n");
}

Node::Node() : m_packet(new Packet(NULL)) {
	initPayload(new Payload(*this));
}
Node::~Node() {

}
void
Node::initPayload(Payload *payload) {
	local_shared_ptr<Packet>(m_packet)->payload().reset(payload);
}
void
Node::recreateNodeTree(local_shared_ptr<Packet> &packet) {
	unsigned int idx = 0;
	packet.reset(new Packet(*packet));
	packet->subpackets().reset(packet->size() ? (new PacketList(*packet->subpackets())) : (new PacketList));
	packet->subnodes().reset(packet->size() ? (new NodeList(*packet->subnodes())) : (new NodeList));
	for(PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
		if((*pit)->size()) {
			pit->reset(new Packet(**pit));
		}
		if((*pit)->size()) {
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
void
Node::insert(const shared_ptr<Node> &var) {
	for(;;) {
		local_shared_ptr<Packet> oldpacket;
		snapshot(oldpacket);
		local_shared_ptr<Packet> packet(oldpacket);
		recreateNodeTree(packet);
		packet->subpackets()->resize(packet->size() + 1);
		ASSERT(packet->subnodes());
		packet->subnodes()->push_back(var);
		ASSERT(packet->subpackets()->size() == packet->subnodes()->size());
		packet->setBundled(false);
		if(commit(oldpacket, packet)) {
			local_shared_ptr<LookupHint> hint(new LookupHint);
			hint->m_index = packet->size() - 1;
			hint->m_superNodeList = packet->subnodes();
			var->m_lookupHint = hint;
			break;
		}
	}
}
void
Node::release(const shared_ptr<Node> &var) {
	for(;;) {
		local_shared_ptr<Packet> oldpacket;
		snapshot(oldpacket);
		local_shared_ptr<Packet> packet(oldpacket);
		local_shared_ptr<Node::Packet> oldsubpacket(
			var->reverseLookup(packet));
		recreateNodeTree(packet);
		local_shared_ptr<Node::Packet> newsubpacket;

		NodeList::iterator nit = packet->subnodes()->begin();
		for(PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
			if(nit->get() == &*var) {
				if((*pit)->size()) {
					(*pit)->subnodes()->m_superNodeList.reset();
				}
				else {
					pit->reset(new Packet(**pit));
				}
				newsubpacket = *pit;
				(*pit)->m_supernode = NULL;
				pit = packet->subpackets()->erase(pit);
				nit = packet->subnodes()->erase(nit);
			}
			else {
				++nit;
				++pit;
			}
		}
		ASSERT(newsubpacket);
		ASSERT( ! newsubpacket->m_supernode);

		if( ! packet->size()) {
			packet->subpackets().reset();
		}
		else {
			packet->setBundled(false);
		}
		local_shared_ptr<Packet> nullpacket(var->m_packet);
		if(nullpacket->isHere())
			continue;
		Node::UnbundledStatus ret = unbundle(*var, nullpacket, &oldsubpacket, &newsubpacket, &oldpacket, &packet);
		if(ret == UNBUNDLE_W_NEW_VALUES) {
			var->m_lookupHint.reset();
			break;
		}
	}
}
local_shared_ptr<Node::Packet>*
Node::NodeList::reverseLookup(local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial) {
	local_shared_ptr<Node::Packet> *foundpacket;
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
		foundpacket = &(*foundpacket)->subpackets()->at(m_index);
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
local_shared_ptr<Node::Packet>&
Node::reverseLookup(local_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial) const {
	ASSERT(packet->size());
	local_shared_ptr<LookupHint> hint(m_lookupHint);
	for(int i = 0;; ++i) {
		ASSERT(i < 2);
		if(hint) {
			shared_ptr<NodeList> supernodelist = hint->m_superNodeList.lock();
			if(supernodelist &&
				((hint->m_index < supernodelist->size()) &&
					(supernodelist->at(hint->m_index).get() == this))) {
				local_shared_ptr<Node::Packet>* superpacket = supernodelist->reverseLookup(packet, copy_branch, tr_serial);
				if(superpacket) {
					local_shared_ptr<Node::Packet> &foundpacket((*superpacket)->subpackets()->at(hint->m_index));
					if(copy_branch && (foundpacket->m_serial != tr_serial)) {
						foundpacket.reset(new Packet(*foundpacket));
						foundpacket->m_serial = tr_serial;
					}
					ASSERT(foundpacket->isBundled());
					ASSERT(&foundpacket->node() == this);
					return foundpacket;
				}
			}
		}
		printf("!");
		bool ret = forwardLookup(packet, hint);
		ASSERT(ret);
		m_lookupHint = hint;
	}
}
bool
Node::forwardLookup(const local_shared_ptr<Packet> &packet, local_shared_ptr<LookupHint> &hint) const {
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
		if(forwardLookup(packet->subpackets()->at(i), hint)) {
			return true;
		}
	}
	return false;
}

void
Node::snapshot(local_shared_ptr<Packet> &target) const {
	for(;;) {
		target = m_packet;
		if(target->isBundled())
			return;
		if( ! target->isHere()) {
			if(trySnapshotSuper(target)) {
				target = const_cast<Node*>(this)->reverseLookup(target);
				ASSERT(target->isBundled());
				return;
			}
			continue;
		}
		if(const_cast<Node*>(this)->bundle(target))
			return;
	}
}
bool
Node::trySnapshotSuper(local_shared_ptr<Packet> &target) const {
	local_shared_ptr<Packet> oldpacket(target);
	ASSERT( ! target->isHere());
	Node& supernode(target->supernode());
	target = supernode.m_packet;
	if(target->isBundled())
		return true;
	if( ! target->isHere()) {
		if( ! supernode.trySnapshotSuper(target))
			return false;
	}
	ASSERT(target->size());
	if(m_packet == oldpacket) {
		ASSERT( ! oldpacket->isHere());
		return true; //Up-to-date unbundled packet w/o local packet.
	}
	return false;
}

bool
Node::bundle(local_shared_ptr<Packet> &target) {
	ASSERT( ! target->isBundled() && target->isHere());
	ASSERT(target->size());
	local_shared_ptr<Packet> prebundled(new Packet(*target));
	//copying all sub-packets from nodes to the new packet.
	shared_ptr<PacketList> packets(new PacketList(*prebundled->subpackets()));
	prebundled->subpackets() = packets;
	std::vector<local_shared_ptr<Packet> > packets_onnode(target->size());
	for(unsigned int i = 0; i < prebundled->size(); ++i) {
		shared_ptr<Node> child(prebundled->subnodes()->at(i));
		for(;;) {
			local_shared_ptr<Packet> packetonnode(child->m_packet);
			if( ! packetonnode->isHere() && ! packets->at(i)) {
				//m_packet has changed.
				return false;
			}
			packets_onnode[i] = packetonnode;
			if(packetonnode->isHere()) {
				if( ! packetonnode->isBundled()) {
					if( ! child->bundle(packetonnode))
						continue;
				}
				packets->at(i) = packetonnode;
			}
			if(packets->at(i)->m_supernode != this) {
				packets->at(i).reset(new Packet(*packets->at(i)));
				packets->at(i)->m_supernode = this;
			}
			if(packets->at(i)->size()) {
				if((packets->at(i)->subnodes()->m_superNodeList.lock() != prebundled->subnodes()) ||
					(packets->at(i)->subnodes()->m_index != i)) {
					packets->at(i).reset(new Packet(*packets->at(i)));
					packets->at(i)->subpackets().reset(new PacketList(*packets->at(i)->subpackets()));
					packets->at(i)->subnodes().reset(new NodeList(*packets->at(i)->subnodes()));
					packets->at(i)->subnodes()->m_superNodeList = prebundled->subnodes();
					packets->at(i)->subnodes()->m_index = i;
				}
			}
			ASSERT(&packets->at(i)->node() == child.get());
			break;
		}
		ASSERT(packets->at(i));
		ASSERT(packets->at(i)->isBundled());
		if(packets->at(i)->size()) {
			ASSERT(packets->at(i)->subnodes()->m_superNodeList.lock() == prebundled->subnodes());
			ASSERT(packets->at(i)->subnodes()->m_index == i);
		}
	}
	//First checkpoint.
	if( ! m_packet.compareAndSet(target, prebundled)) {
		return false;
	}
	//clearing all packets on sub-nodes if not modified.
	for(unsigned int i = 0; i < prebundled->size(); i++) {
		shared_ptr<Node> child(prebundled->subnodes()->at(i));
		local_shared_ptr<Packet> nullpacket(new NullPacket(this));
		//Second checkpoint, the written bundle is valid or not.
		if( ! child->m_packet.compareAndSet(packets_onnode[i], nullpacket)) {
			return false;
		}
	}
	target.reset(new Packet(*prebundled));
	target->setBundled(true);
	target->subpackets() = packets;
	//Finally, tagging as bundled.
	return m_packet.compareAndSet(prebundled, target);
}

bool
Node::commit(const local_shared_ptr<Packet> &oldpacket, local_shared_ptr<Packet> &newpacket) {
	for(;;) {
		local_shared_ptr<Packet> packet(m_packet);
		if(packet->isHere()) {
			if(packet != oldpacket)
				return false;
			if(m_packet.compareAndSet(oldpacket, newpacket))
				return true;
			continue;
		}
		Node &supernode(newpacket->supernode());
		UnbundledStatus ret = supernode.unbundle(*this, packet, &oldpacket, &newpacket);
		switch(ret) {
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

Node::UnbundledStatus
Node::unbundle(Node &subnode, const local_shared_ptr<Packet> &nullpacket,
	const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<Packet> *newsubpacket,
	const local_shared_ptr<Packet> *oldsuperpacket, const local_shared_ptr<Packet> *newsuperpacket) {
	ASSERT( ! nullpacket->isHere());
	local_shared_ptr<Packet> packet(m_packet);
	local_shared_ptr<Packet> copied;
	if( ! packet->isHere()) {
		//Unbundle all supernodes.
		Node *supernode = packet->m_supernode;
		if(oldsuperpacket) {
			copied.reset(new Packet(**oldsuperpacket));
			copied->setBundled(false);
		}
		UnbundledStatus ret = supernode->unbundle(*this, packet,
			oldsuperpacket ? oldsuperpacket : NULL, &copied);
		if((ret != UNBUNDLE_W_NEW_SUBVALUE) || (ret != UNBUNDLE_W_NEW_VALUES))
			return UNBUNDLE_DISTURBED;
	}
	else {
		copied.reset(new Packet(*packet));
		copied->setBundled(false);
		if(newsuperpacket)
			if(packet != *oldsuperpacket)
				return UNBUNDLE_DISTURBED;
		//Tagging as unbundled.
		if( ! m_packet.compareAndSet(packet, copied)) {
			return UNBUNDLE_DISTURBED;
		}
	}

	local_shared_ptr<Packet> subpacket;
	NodeList::iterator nit = packet->subnodes()->begin();
	for(PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
		if(nit->get() == &subnode) {
			subpacket = *pit;
		}
		++pit;
		++nit;
	}

	if(oldsubpacket) {
		if(subpacket != *oldsubpacket) {
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		}
	}
	else {
		if( ! subpacket)
			return UNBUNDLE_DISTURBED;
		*newsubpacket = subpacket;
		newsubpacket->reset(new Packet(**newsubpacket));
		(*newsubpacket)->setBundled(false);
	}

	if( ! subnode.m_packet.compareAndSet(nullpacket, *newsubpacket)) {
		if( ! local_shared_ptr<Packet>(subnode.m_packet)->isHere())
			return UNBUNDLE_SUBVALUE_HAS_CHANGED;
		return UNBUNDLE_SUCCESS;
	}
	if(newsuperpacket) {
		packet = *newsuperpacket;
	}
	else {
		//Erasing out-of-date subpackets on the unbundled superpacket.
		packet.reset(new Packet(*copied));
		packet->subpackets().reset(new PacketList(*packet->subpackets()));
		NodeList::iterator nit = packet->subnodes()->begin();
		for(PacketList::iterator pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
			if(*pit) {
				if(local_shared_ptr<Packet>((*nit)->m_packet)->isHere())
					pit->reset();
			}
			++pit;
			++nit;
		}
	}
	if(m_packet.compareAndSet(copied, packet))
		return UNBUNDLE_W_NEW_VALUES;
	else
		return UNBUNDLE_W_NEW_SUBVALUE;
}

