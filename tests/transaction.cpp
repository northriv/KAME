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

Node::Packet::Packet(Node *bundler) : m_state(PACKET_BUNDLED), m_payload(),
	m_bundler(bundler), m_serial(-1) {
}
Node::Packet::~Packet() {

}
void
Node::Packet::print() {
	printf("Packet: ");
	printf("Bundler:%llx, ", (uintptr_t)&bundler());
	if(!isHere())
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
	m_packet->payload().reset(new Payload(*this));
}
Node::~Node() {

}
void
Node::insert(const shared_ptr<Node> &var) {
	for(;;) {
		atomic_shared_ptr<Packet> oldpacket;
		snapshot(oldpacket);
		atomic_shared_ptr<Packet> packet(new Packet(*oldpacket));
		packet->subpackets().reset(packet->size() ? (new PacketList(*packet->subpackets())) : (new PacketList));
		packet->subnodes().reset(packet->size() ? (new NodeList(*packet->subnodes())) : (new NodeList));
		packet->subpackets()->resize(packet->size() + 1);
		ASSERT(packet->subnodes());
		packet->subnodes()->push_back(var);
		ASSERT(packet->subpackets()->size() == packet->subnodes()->size());

		packet->setBundled(false);
		if(commit(oldpacket, packet)) {
			atomic_shared_ptr<LookupHint> hint(new LookupHint);
			hint->m_index = packet->size() - 1;
			hint->m_superNodeList = packet->subnodes();
			var->m_lookupHint = hint;
			break;
		}
	}
}
void
Node::NodeList::reverseLookup(atomic_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial) {
	if(packet->subnodes().get() != this) {
		ASSERT(m_superNodeList);
		m_superNodeList->reverseLookup(packet, copy_branch, tr_serial);
		packet = packet->subpackets()->at(m_index);
		ASSERT(packet->isBundled());
	}
	if(copy_branch) {
		if(packet->subpackets()->m_serial != tr_serial) {
			if(packet->m_serial != tr_serial) {
				packet.reset(new Packet(*packet));
				packet->m_serial = tr_serial;
			}
			packet->subpackets().reset(new PacketList(*packet->subpackets()));
			packet->subpackets()->m_serial = tr_serial;
		}
		ASSERT(packet->m_serial == tr_serial);
	}
}
void
Node::reverseLookup(atomic_shared_ptr<Packet> &packet, bool copy_branch, int tr_serial) {
	ASSERT(packet->size());
	atomic_shared_ptr<LookupHint> hint(m_lookupHint);
	for(int i = 0;; ++i) {
		ASSERT(i < 2);
		if(hint) {
			shared_ptr<NodeList> supernodelist = hint->m_superNodeList.lock();
			if(supernodelist &&
				((hint->m_index < supernodelist->size()) &&
					(supernodelist->at(hint->m_index).get() == this))) {
				supernodelist->reverseLookup(packet, copy_branch, tr_serial);
				packet = packet->subpackets()->at(hint->m_index);
				ASSERT(packet);
				ASSERT(packet->isBundled());
				ASSERT(&packet->node() == this);
				if(copy_branch && (packet->m_serial != tr_serial)) {
					packet.reset(new Packet(*packet));
					packet->m_serial = tr_serial;
				}
				return;
			}
		}
		printf("!");
		bool ret = forwardLookup(packet, hint);
		ASSERT(ret);
		m_lookupHint = hint;
	}
}
bool
Node::forwardLookup(const atomic_shared_ptr<Packet> &packet, atomic_shared_ptr<LookupHint> &hint) const {
	if(!packet->subpackets())
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
Node::snapshot(atomic_shared_ptr<Packet> &target) const {
	for(;;) {
		target = m_packet;
		if(target->isBundled())
			return;
		if(!target->isHere()) {
			if(trySnapshotSuper(target)) {
				const_cast<Node*>(this)->reverseLookup(target);
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
Node::trySnapshotSuper(atomic_shared_ptr<Packet> &target) const {
	atomic_shared_ptr<Packet> oldpacket(target);
	ASSERT(!target->isHere());
	Node& bundler(target->bundler());
	target = bundler.m_packet;
	if(target->isBundled())
		return true;
	if(!target->isHere()) {
		return bundler.trySnapshotSuper(target);
	}
	ASSERT(target->size());
	if(m_packet == oldpacket)
			return true; //Unbundled packet w/o local packet.
	return false;
}
bool
Node::commit(const atomic_shared_ptr<Packet> &oldpacket, atomic_shared_ptr<Packet> &newpacket) {
	atomic_shared_ptr<Packet> packet(m_packet);
	if(!packet->isHere()) {
		Node &bundler(newpacket->bundler());
		return bundler.unbundle(*this, &oldpacket, &newpacket);
	}
	return newpacket.compareAndSet(oldpacket, m_packet);
}

bool
Node::bundle(atomic_shared_ptr<Packet> &target) {
	ASSERT(!target->isBundled() && target->isHere());
	ASSERT(target->size());
	atomic_shared_ptr<Packet> prebundled(new Packet(*target));
	//copying all sub-packets from nodes to the new packet.
	shared_ptr<PacketList> packets(new PacketList(*prebundled->subpackets()));
	prebundled->subpackets() = packets;
	std::vector<atomic_shared_ptr<Packet> > packets_onnode(target->size());
	for(unsigned int i = 0; i < prebundled->size(); i++) {
		shared_ptr<Node> child(prebundled->subnodes()->at(i));
		for(;;) {
			atomic_shared_ptr<Packet> packetonnode(child->m_packet);
			if(!packetonnode->isHere() && !packets->at(i)) {
				//m_packet has changed.
				return false;
			}
			packets_onnode[i] = packetonnode;
			if(packetonnode->isHere()) {
				if(!packetonnode->isBundled()) {
					if(!child->bundle(packetonnode))
						continue;
				}
				if(packetonnode->m_bundler != this) {
					packetonnode.reset(new Packet(*packetonnode));
					packetonnode->m_bundler = this;
					if(packetonnode->size()) {
						packetonnode->subpackets().reset(new PacketList(*packetonnode->subpackets()));
						packetonnode->subnodes().reset(new NodeList(*packetonnode->subnodes()));
						packetonnode->subnodes()->m_superNodeList = prebundled->subnodes().get();
						packetonnode->subnodes()->m_index = i;
					}
				}
				packets->at(i) = packetonnode;
				ASSERT(&packetonnode->node() == child.get());
			}
			break;
		}
	}
	//First checkpoint.
	if(!prebundled.compareAndSet(target, m_packet)) {
		return false;
	}
	//clearing all packets on sub-nodes if not modified.
	for(unsigned int i = 0; i < prebundled->size(); i++) {
		shared_ptr<Node> child(prebundled->subnodes()->at(i));
		atomic_shared_ptr<Packet> nullpacket(new NullPacket(this));
		//Second checkpoint, the written bundle is valid or not.
		if(!nullpacket.compareAndSwap(packets_onnode[i], child->m_packet)) {
			return false;
		}
	}
	target.reset(new Packet(*prebundled));
	target->setBundled(true);
	target->subpackets() = packets;
	//Finally, tagging as bundled.
	return target.compareAndSet(prebundled, m_packet);
}
bool
Node::unbundle(Node &subnode, const atomic_shared_ptr<Packet> *oldsubpacket, atomic_shared_ptr<Packet> *newsubpacket) {
	for(;;) {
		atomic_shared_ptr<Packet> nullpacket(subnode.m_packet);
		if(nullpacket->isHere()) {
			//Already unbundled.
			return false;
		}
		atomic_shared_ptr<Packet> packet(m_packet);
		//Unbundle all supernodes.
		if(!packet->isHere()) {
			Node *bundler = packet->m_bundler;
			bundler->unbundle(*this);
			packet = m_packet;
			if(!packet->isHere())
				continue;
		}
		unsigned int idx;
		if(newsubpacket) {
			for(unsigned int i = 0; ; i++) {
				ASSERT(i < packet->subnodes()->size());
				shared_ptr<Node> child(packet->subnodes()->at(i));
				if(child.get() == &subnode) {
					if(packet->subpackets()->at(i) != *oldsubpacket)
						return false;
					idx = i;
					break;
				}
			}
		}
		atomic_shared_ptr<Packet> copied(new Packet(*packet));
		copied->setBundled(false);
		//Tagging as unbundled.
		if(!copied.compareAndSet(packet, m_packet)) {
			continue;
		}
		if(!newsubpacket)
			return false;
		if(!newsubpacket->compareAndSet(nullpacket, subnode.m_packet))
			return false;
		//Erasing the old packet on the useless old bundle.
		packet.reset(new Packet(*copied));
		packet->subpackets().reset(new PacketList(*packet->subpackets()));
		packet->subpackets()->at(idx).reset();
		packet.compareAndSwap(copied, m_packet);
		return true;
	}
}

