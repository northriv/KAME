/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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

#ifdef TRANSACTIONAL_STRICT_assert
	#undef STRICT_assert
	#define STRICT_assert(expr) assert(expr)
	#define STRICT_TEST(expr) expr
#else
	#define STRICT_assert(expr)
	#define STRICT_TEST(expr)
#endif

namespace Transactional {

STRICT_TEST(static atomic<int64_t> s_serial_abandoned = -2);

template <class XN>
XThreadLocal<typename Node<XN>::FuncPayloadCreator> Node<XN>::stl_funcPayloadCreator;

template <class XN>
atomic<int64_t> Node<XN>::Packet::s_serial = Node<XN>::Packet::SERIAL_FIRST;

template <class XN>
Node<XN>::Packet::Packet() : m_missing(false) {}

template <class XN>
void
Node<XN>::Packet::print_() const {
	printf("Packet: ");
	printf("%s@%p, ", typeid(*this).name(), &node());
	printf("BP@%p, ", node().m_link.get());
	if(missing())
		printf("missing, ");
	if(size()) {
		printf("%d subnodes : [ \n", (int)size());
		for(int i = 0; i < size(); i++) {
			if(subpackets()->at(i)) {
				subpackets()->at(i)->print_();
				printf("; ");
			}
			else {
				printf("%s@%p, w/o packet, ", typeid(*this).name(), subnodes()->at(i).get());
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
		fprintf(stderr, "Line %d, losing consistensy on node %p:\n", line, &node());
		rootpacket->print_();
		throw *this;
	}
	return true;
}

template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const local_shared_ptr<Packet> &x, int64_t bundle_serial) :
	m_bundledBy(), m_packet(x), m_ridx(PACKET_HAS_PRIORITY), m_bundle_serial(bundle_serial) {
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const shared_ptr<Linkage > &bp, int reverse_index,
	int64_t bundle_serial) :
	m_bundledBy(bp), m_packet(), m_ridx(), m_bundle_serial(bundle_serial) {
	setReverseIndex(reverse_index);
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const PacketWrapper &x, int64_t bundle_serial) :
	m_bundledBy(x.m_bundledBy), m_packet(x.m_packet),
	m_ridx(x.m_ridx), m_bundle_serial(bundle_serial) {}

template <class XN>
void
Node<XN>::PacketWrapper::print_() const {
	printf("PacketWrapper: ");
	if( !hasPriority()) {
		printf("referred to BP@%p, ", bundledBy().get());
	}
	printf("serial:%lld, ", (long long)m_bundle_serial);
	if(packet()) {
		packet()->print_();
	}
	else {
		printf("absent, ");
	}
	printf("\n");
}

template <class XN>
inline void
Node<XN>::Linkage::negotiate(uint64_t &started_time) {
	int64_t transaction_started_time = m_transaction_started_time;
	if(transaction_started_time) {
		int ms = ((int64_t)started_time - transaction_started_time);
		if(ms > 0) {
            XTime t0 = XTime::now();
            if(ms > 2000) {
                if(ms > 4000)
					fprintf(stderr, "Nested transaction?, ");
                fprintf(stderr, "Negotiating, %f sec. requested, limited to 2 sec. ", ms*1e-3);
				fprintf(stderr, "for BP@%p\n", this);
                ms = 2000;
			}
            t0 += ms * 1e-3;
            while(t0 > XTime::now()) {
//				usleep(1000);
                msecsleep(1);
                if( !m_transaction_started_time || (((int64_t)started_time <= (int64_t)m_transaction_started_time)))
					break;
            }
//			started_time -= XTime::now().diff_msec(t0) + ms;
		}
	}
}

template <class XN>
Node<XN>::Node() : m_link(std::make_shared<Linkage>()) {
	local_shared_ptr<Packet> packet(new Packet());
	m_link->reset(new PacketWrapper(packet, Packet::SERIAL_INIT));
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
Node<XN>::print_() const {
	local_shared_ptr<PacketWrapper> packet( *m_link);
//	printf("Node:%p, ", this);
//	printf("BP:%p, ", m_link.get());
//	printf(" packet: ");
	packet->print_();
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
    packet->subpackets() = packet->size() ? std::make_shared<PacketList>( *packet->subpackets()) : std::make_shared<PacketList>();
	packet->subpackets()->m_serial = tr.m_serial;
	packet->m_missing = true;
    packet->subnodes() = packet->size() ? std::make_shared<NodeList>( *packet->subnodes()) : std::make_shared<NodeList>();
	packet->subpackets()->resize(packet->size() + 1);
	assert(packet->subnodes());
	assert(std::find(packet->subnodes()->begin(), packet->subnodes()->end(), var) == packet->subnodes()->end());
	packet->subnodes()->push_back(var);
	assert(packet->subpackets()->size() == packet->subnodes()->size());

	if(online_after_insertion) {
		bool has_failed = false;
		//Tags serial.
		local_shared_ptr<Packet> newpacket(tr.m_packet);
		tr.m_packet.reset(new Packet( *tr.m_oldpacket));
		if( !tr.m_packet->node().commit(tr)) {
			printf("*\n");
			has_failed = true;
		}
		tr.m_oldpacket = tr.m_packet;
		tr.m_packet = newpacket;
		for(;;) {
			local_shared_ptr<Packet> subpacket_new;
			local_shared_ptr<PacketWrapper> subwrapper;
			subwrapper = *var->m_link;
			BundledStatus status = bundle_subpacket(0, var, subwrapper, subpacket_new,
				tr.m_started_time, tr.m_serial);
			if(status != BUNDLE_SUCCESS) {
				continue;
			}
			if( !subpacket_new)
				//Inserted twice inside the package.
				break;

			//Marks for writing at subnode.
			tr.m_packet.reset(new Packet( *tr.m_oldpacket));
			if( !tr.m_packet->node().commit(tr)) {
				printf("&\n");
				has_failed = true;
			}
			tr.m_oldpacket = tr.m_packet;
			tr.m_packet = newpacket;

			local_shared_ptr<PacketWrapper> newwrapper(
				new PacketWrapper(m_link, packet->size() - 1, tr.m_serial));
			newwrapper->packet() = subpacket_new;
			packet->subpackets()->back() = subpacket_new;
			if(has_failed)
				return false;
			if( !var->m_link->compareAndSet(subwrapper, newwrapper)) {
				tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
				return false;
			}
			break;
		}
	}
	tr[ *this].catchEvent(var, packet->size() - 1);
	tr[ *this].listChangeEvent();
	STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
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
Node<XN>::eraseSerials(local_shared_ptr<Packet> &packet, int64_t serial) {
	if(packet->size() && packet->subpackets()->m_serial == serial)
		packet->subpackets()->m_serial = Packet::SERIAL_NULL;
	if(packet->payload()->m_serial == serial)
		packet->payload()->m_serial = Packet::SERIAL_NULL;

	for(;;) {
		local_shared_ptr<PacketWrapper> wrapper( *packet->node().m_link);
		if(wrapper->m_bundle_serial != serial)
			break;
		local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper( *wrapper, Packet::SERIAL_NULL));
		if(packet->node().m_link->compareAndSet(wrapper, newwrapper))
			break;
	}
	for(int i = 0; i < packet->size(); ++i) {
		local_shared_ptr<Packet> &subpacket(( *packet->subpackets())[i]);
		if(subpacket)
			eraseSerials(subpacket, serial);
	}
}

template <class XN>
bool
Node<XN>::release(Transaction<XN> &tr, const shared_ptr<XN> &var) {
	local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
	assert(packet->size());
	packet->subpackets().reset(new PacketList( *packet->subpackets()));
	packet->subpackets()->m_serial = tr.m_serial;
	packet->subnodes().reset(new NodeList( *packet->subnodes()));
	unsigned int idx = 0;
	int old_idx = -1;
	local_shared_ptr<PacketWrapper> nullsubwrapper, newsubwrapper;
	auto nit = packet->subnodes()->begin();
	for(auto pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
		assert(nit != packet->subnodes()->end());
		if(nit->get() == &*var) {
			if( *pit) {
				nullsubwrapper = *var->m_link;
				if(nullsubwrapper->hasPriority()) {
					if(nullsubwrapper->packet() != *pit) {
						tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
						return false;
					}
				}
				else {
					shared_ptr<Linkage> bp(nullsubwrapper->bundledBy());
					if((bp && (bp != m_link)) ||
						( !bp && (nullsubwrapper->packet() != *pit))) {
						tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
						return false;
					}
				}
				newsubwrapper.reset(new PacketWrapper(m_link, idx, Packet::SERIAL_NULL));
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
	if(old_idx < 0)
		throw NodeNotFoundError("Lookup failure.");

	if( !packet->subpackets()->size()) {
		packet->subpackets().reset();
		packet->m_missing = false;
	}
	if(tr.m_packet->size()) {
		tr.m_packet->m_missing = true;
	}

	tr[ *this].releaseEvent(var, old_idx);
	tr[ *this].listChangeEvent();

	if( !newsubwrapper) {
		//Packet of the released node is held by the other point inside the tr.m_packet.
		return true;
	}

	eraseSerials(packet, tr.m_serial);

	local_shared_ptr<Packet> newpacket(tr.m_packet);
	tr.m_packet = tr.m_oldpacket;
	tr.m_packet.reset(new Packet( *tr.m_packet));
	if( !tr.m_packet->node().commit(tr)) {
		tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
		tr.m_packet = newpacket;
		return false;
	}
	tr.m_oldpacket = tr.m_packet;
	tr.m_packet = newpacket;

	//Unload the packet of the released node.
	if( !var->m_link->compareAndSet(nullsubwrapper, newsubwrapper)) {
		tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
		return false;
	}
//		printf("r");
	STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
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
	for(auto nit = packet->subnodes()->begin(); nit != packet->subnodes()->end(); ++nit) {
		if( *nit == x)
			x_idx = idx;
		if( *nit == y)
			y_idx = idx;
		++idx;
	}
	if((x_idx < 0) || (y_idx < 0))
		throw NodeNotFoundError("Lookup failure.");
	local_shared_ptr<Packet> px = packet->subpackets()->at(x_idx);
	local_shared_ptr<Packet> py = packet->subpackets()->at(y_idx);
	packet->subpackets()->at(x_idx) = py;
	packet->subpackets()->at(y_idx) = px;
	packet->subnodes()->at(x_idx) = y;
	packet->subnodes()->at(y_idx) = x;
	tr[ *this].moveEvent(x_idx, y_idx);
	tr[ *this].listChangeEvent();
	STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
	return true;
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookupWithHint(shared_ptr<Linkage> &linkage,
	local_shared_ptr<Packet> &superpacket, bool copy_branch, int64_t tr_serial, bool set_missing,
	local_shared_ptr<Packet> *upperpacket, int *index) {
	if( !superpacket->size())
		return NULL;
	local_shared_ptr<PacketWrapper> wrapper( *linkage);
	if(wrapper->hasPriority())
		return NULL;
	shared_ptr<Linkage> linkage_upper(wrapper->bundledBy());
	if( !linkage_upper)
		return NULL;
	local_shared_ptr<Packet> *foundpacket;
	if(linkage_upper == superpacket->node().m_link)
		foundpacket = &superpacket;
	else {
		foundpacket = reverseLookupWithHint(linkage_upper,
			superpacket, copy_branch, tr_serial, set_missing, NULL, NULL);
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
	if( !p || (p->node().m_link != linkage)) {
		return NULL;
	}
	if(upperpacket) {
		*upperpacket = *foundpacket;
		*index = ridx;
	}
	return &p;
}

template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::forwardLookup(local_shared_ptr<Packet> &superpacket,
	bool copy_branch, int64_t tr_serial, bool set_missing,
	local_shared_ptr<Packet> *upperpacket, int *index) const {
	assert(superpacket);
	if( !superpacket->subpackets())
		return NULL;
	if(copy_branch) {
		if(superpacket->subpackets()->m_serial != tr_serial) {
			superpacket.reset(new Packet( *superpacket));
			superpacket->subpackets().reset(new PacketList( *superpacket->subpackets()));
			superpacket->subpackets()->m_serial = tr_serial;
			superpacket->m_missing = superpacket->m_missing || set_missing;
		}
	}
	for(unsigned int i = 0; i < superpacket->subnodes()->size(); i++) {
		if(( *superpacket->subnodes())[i].get() == this) {
			local_shared_ptr<Packet> &subpacket(( *superpacket->subpackets())[i]);
			if(subpacket) {
				*upperpacket = superpacket;
				*index = i;
				return &subpacket;
			}
		}
	}
	for(unsigned int i = 0; i < superpacket->subnodes()->size(); i++) {
		local_shared_ptr<Packet> &subpacket(( *superpacket->subpackets())[i]);
		if(subpacket) {
			if(local_shared_ptr<Packet> *p =
				forwardLookup(subpacket, copy_branch, tr_serial, set_missing, upperpacket, index)) {
				return p;
			}
		}
	}
	return NULL;
}

template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookup(local_shared_ptr<Packet> &superpacket,
	bool copy_branch, int64_t tr_serial, bool set_missing, XN **uppernode) {
	local_shared_ptr<Packet> *foundpacket;
	if( &superpacket->node() == this) {
		foundpacket = &superpacket;
	}
	else {
		local_shared_ptr<Packet> upperpacket;
		int index;
		foundpacket = reverseLookupWithHint(m_link, superpacket,
			copy_branch, tr_serial, set_missing, &upperpacket, &index);
		if(foundpacket) {
//				printf("$");
		}
		else {
//				printf("!");
			foundpacket = forwardLookup(superpacket, copy_branch, tr_serial, set_missing,
				&upperpacket, &index);
			if( !foundpacket)
				return 0;
		}
		if(uppernode)
			*uppernode = static_cast<XN*>(&upperpacket->node());
		assert( &( *foundpacket)->node() == this);
	}
	if(copy_branch && (( *foundpacket)->payload()->m_serial != tr_serial)) {
		foundpacket->reset(new Packet( **foundpacket));
	}
//						printf("#");
	return foundpacket;
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet>&
Node<XN>::reverseLookup(local_shared_ptr<Packet> &superpacket,
	bool copy_branch, int64_t tr_serial, bool set_missing) {
	local_shared_ptr<Packet> *foundpacket = reverseLookup(superpacket, copy_branch, tr_serial, set_missing, 0);
	if( !foundpacket) {
		fprintf(stderr, "Node not found during a lookup.\n");
		throw NodeNotFoundError("Lookup failure.");
	}
	return *foundpacket;
}

template <class XN>
const local_shared_ptr<typename Node<XN>::Packet> &
Node<XN>::reverseLookup(const local_shared_ptr<Packet> &superpacket) const {
	local_shared_ptr<Packet> *foundpacket = const_cast<Node*>(this)->reverseLookup(
		const_cast<local_shared_ptr<Packet> &>(superpacket), false, 0, false, 0);
	if( !foundpacket) {
		fprintf(stderr, "Node not found during a lookup.\n");
		throw NodeNotFoundError("Lookup failure.");
	}
	return *foundpacket;
}

template <class XN>
XN *
Node<XN>::upperNode(Snapshot<XN> &shot) {
	XN *uppernode = 0;
	reverseLookup(shot.m_packet, false, 0, false, &uppernode);
	return uppernode;
}

template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::snapshotSupernode(const shared_ptr<Linkage > &linkage,
	local_shared_ptr<PacketWrapper> &shot, local_shared_ptr<Packet> **subpacket,
	SnapshotMode mode, int64_t serial, std::deque<CASInfo> *cas_infos) {
	local_shared_ptr<PacketWrapper> oldwrapper(shot);
	assert( !shot->hasPriority());
	shared_ptr<Linkage > linkage_upper(shot->bundledBy());
	if( !linkage_upper) {
		if( *linkage == oldwrapper)
			//Supernode has been destroyed.
			return SNAPSHOT_NODE_MISSING;
		return SNAPSHOT_DISTURBED;
	}
	int reverse_index = shot->reverseIndex();

	shot = *linkage_upper;
	local_shared_ptr<PacketWrapper> shot_upper(shot);
	SnapshotStatus status = SNAPSHOT_NODE_MISSING;
	local_shared_ptr<Packet> *upperpacket;
	if( !shot_upper->hasPriority()) {
		status = snapshotSupernode(linkage_upper, shot, &upperpacket,
			mode, serial, cas_infos);
	}
	switch(status) {
	case SNAPSHOT_DISTURBED:
	default:
		return status;
	case SNAPSHOT_VOID_PACKET:
	case SNAPSHOT_NODE_MISSING:
		shot = shot_upper;
		upperpacket = &shot->packet();
		status = SNAPSHOT_SUCCESS;
		break;
	case SNAPSHOT_NODE_MISSING_AND_COLLIDED:
		shot = shot_upper;
		upperpacket = &shot->packet();
		status = SNAPSHOT_COLLIDED;
		break;
	case SNAPSHOT_COLLIDED:
	case SNAPSHOT_SUCCESS:
		break;
	}
	//Checking if it is up-to-date.
	if( *linkage != oldwrapper)
			return SNAPSHOT_DISTURBED;

	assert( *upperpacket);
	int size = ( *upperpacket)->size();
	int i = reverse_index;
	for(int cnt = 0;; ++cnt) {
		if(cnt >= size) {
			if(status == SNAPSHOT_COLLIDED)
				return SNAPSHOT_NODE_MISSING;
			status = SNAPSHOT_NODE_MISSING;
			break;
		}
		if(i >= size)
			i = 0;
		if(( *( *upperpacket)->subnodes())[i]->m_link == linkage) {
			//Requested node is found.
			*subpacket = &( *( *upperpacket)->subpackets())[i];
			reverse_index = i;
			if( !**subpacket) {
				if(mode == SNAPSHOT_FOR_UNBUNDLE) {
					cas_infos->clear();
				}
//				printf("V\n");
				assert(( *upperpacket)->missing());
				return SNAPSHOT_VOID_PACKET;
			}
			break;
		}
		//The index might be modified by swap().
		++i;
	}

	assert( !shot_upper->packet() || (shot_upper->packet()->node().m_link == linkage_upper));
	assert(( *upperpacket)->node().m_link == linkage_upper);
	if(mode == SNAPSHOT_FOR_UNBUNDLE) {
		if(status == SNAPSHOT_COLLIDED) {
			return SNAPSHOT_COLLIDED;
		}
		if((serial != Packet::SERIAL_NULL) && (shot_upper->m_bundle_serial == serial)) {
			//The node has been already bundled in the same snapshot.
			if(status == SNAPSHOT_NODE_MISSING)
				return SNAPSHOT_NODE_MISSING;
			return SNAPSHOT_COLLIDED;
		}
		local_shared_ptr<Packet> *p(upperpacket);
		local_shared_ptr<PacketWrapper> newwrapper;
		if(shot == shot_upper) {
			newwrapper.reset(
				new PacketWrapper( *shot_upper, shot_upper->m_bundle_serial));
		}
		else {
			assert(cas_infos->size());
//			if(shot->packet()->missing()) {
				newwrapper.reset(
					new PacketWrapper( *p, shot->m_bundle_serial));
//			}
		}
		if(newwrapper) {
            cas_infos->emplace_back(linkage_upper, shot_upper, newwrapper);
			p = &newwrapper->packet();
		}
		if(size) {
			p->reset(new Packet( **p));
//			( *p)->subpackets().reset(new PacketList( *( *p)->subpackets()));
			( *p)->m_missing = true;
//			if(status == SNAPSHOT_SUCCESS)
//				*subpacket = &( *p)->subpackets()->at(reverse_index);
		}
		if((status == SNAPSHOT_NODE_MISSING) && (serial != Packet::SERIAL_NULL) &&
			(( !oldwrapper->hasPriority()) && (oldwrapper->m_bundle_serial == serial))) {
			printf("!");
			return SNAPSHOT_NODE_MISSING_AND_COLLIDED;
		}
	}
	return status;
}

template <class XN>
void
Node<XN>::snapshot(Snapshot<XN> &snapshot, bool multi_nodal, uint64_t &started_time) const {
	local_shared_ptr<PacketWrapper> target;
	for(;;) {
		snapshot.m_serial = Node<XN>::Packet::newSerial();
		target = *m_link;
		if(target->hasPriority()) {
			if( !multi_nodal)
				break;
			if( !target->packet()->missing()) {
				STRICT_assert(target->packet()->checkConsistensy(target->packet()));
				break;
			}
		}
		else {
			// Taking a snapshot inside the super packet.
			shared_ptr<Linkage > linkage(m_link);
			local_shared_ptr<PacketWrapper> superwrapper(target);
			local_shared_ptr<Packet> *foundpacket;
			SnapshotStatus status = snapshotSupernode(linkage, superwrapper, &foundpacket, SNAPSHOT_FOR_BUNDLE);
			switch(status) {
			case SNAPSHOT_SUCCESS: {
					if( !( *foundpacket)->missing() || !multi_nodal) {
						snapshot.m_packet = *foundpacket;
						STRICT_assert(snapshot.m_packet->checkConsistensy(snapshot.m_packet));
						return;
					}
					// The packet is imperfect, and then re-bundling the subpackets.
					UnbundledStatus status = unbundle(NULL, started_time, m_link, target);
					switch(status) {
					case UNBUNDLE_W_NEW_SUBVALUE:
					case UNBUNDLE_COLLIDED:
					case UNBUNDLE_SUBVALUE_HAS_CHANGED:
					default:
						break;
					}
					continue;
				}
			case SNAPSHOT_DISTURBED:
			default:
				continue;
			case SNAPSHOT_NODE_MISSING:
			case SNAPSHOT_VOID_PACKET:
				//The packet has been released.
				if( !target->packet()->missing() || !multi_nodal) {
					snapshot.m_packet = target->packet();
					return;
				}
				break;
			}
		}
		BundledStatus status = const_cast<Node *>(this)->bundle(
			target, started_time, snapshot.m_serial, true);
		switch (status) {
		case BUNDLE_SUCCESS:
			assert( !target->packet()->missing());
			STRICT_assert(target->packet()->checkConsistensy(target->packet()));
			break;
		default:
			continue;
		}
	}
	snapshot.m_packet = target->packet();
}

template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle_subpacket(local_shared_ptr<PacketWrapper> *superwrapper,
	const shared_ptr<Node> &subnode,
	local_shared_ptr<PacketWrapper> &subwrapper, local_shared_ptr<Packet> &subpacket_new,
	uint64_t &started_time, int64_t bundle_serial) {

	if( !subwrapper->hasPriority()) {
		shared_ptr<Linkage > linkage(subwrapper->bundledBy());
		bool need_for_unbundle = false;
		bool detect_collision = false;
		if(linkage == m_link) {
			if(subpacket_new) {
				if(subpacket_new->missing()) {
					need_for_unbundle = true;
				}
				else
					return BUNDLE_SUCCESS;
			}
			else {
				if(subwrapper->packet()) {
					//Re-inserted.
//					need_for_unbundle = true;
				}
				else
					return BUNDLE_DISTURBED;
			}
		}
		else {
			need_for_unbundle = true;
			detect_collision = true;
		}
		if(need_for_unbundle) {
			local_shared_ptr<PacketWrapper> subwrapper_new;
			UnbundledStatus status = unbundle(detect_collision ? &bundle_serial : NULL, started_time,
				subnode->m_link, subwrapper, NULL, &subwrapper_new, superwrapper);
			switch(status) {
			case UNBUNDLE_W_NEW_SUBVALUE:
				subwrapper = subwrapper_new;
				break;
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
	if(subwrapper->packet()->missing()) {
		assert(subwrapper->packet()->size());
		BundledStatus status = subnode->bundle(subwrapper, started_time, bundle_serial, false);
		switch(status) {
		case BUNDLE_SUCCESS:
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
Node<XN>::bundle(local_shared_ptr<PacketWrapper> &oldsuperwrapper,
	uint64_t &started_time, int64_t bundle_serial, bool is_bundle_root) {

	assert(oldsuperwrapper->packet());
	assert(oldsuperwrapper->packet()->size());
	assert(oldsuperwrapper->packet()->missing());

	Node &supernode(oldsuperwrapper->packet()->node());

	if( !oldsuperwrapper->hasPriority() ||
		(oldsuperwrapper->m_bundle_serial != bundle_serial)) {
		//Tags serial.
		local_shared_ptr<PacketWrapper> superwrapper(
			new PacketWrapper(oldsuperwrapper->packet(), bundle_serial));
		if( !supernode.m_link->compareAndSet(oldsuperwrapper, superwrapper)) {
			return BUNDLE_DISTURBED;
		}
		oldsuperwrapper = superwrapper;
	}

	std::vector<local_shared_ptr<PacketWrapper> > subwrappers_org(oldsuperwrapper->packet()->subpackets()->size());

	for(;;) {
		local_shared_ptr<PacketWrapper> superwrapper(
			new PacketWrapper( *oldsuperwrapper, bundle_serial));
		local_shared_ptr<Packet> &newpacket(
			reverseLookup(superwrapper->packet(), true, Packet::newSerial()));
		assert(newpacket->size());
		assert(newpacket->missing());

		STRICT_assert(s_serial_abandoned != newpacket->subpackets()->m_serial);

		//copying all sub-packets from nodes to the new packet.
		newpacket->subpackets().reset(new PacketList( *newpacket->subpackets()));
		shared_ptr<PacketList> &subpackets(newpacket->subpackets());
		shared_ptr<NodeList> &subnodes(newpacket->subnodes());

		bool missing = false;
		for(unsigned int i = 0; i < subpackets->size(); ++i) {
			shared_ptr<Node> child(( *subnodes)[i]);
			local_shared_ptr<Packet> &subpacket_new(( *subpackets)[i]);
			for(;;) {
				local_shared_ptr<PacketWrapper> subwrapper;
				subwrapper = *child->m_link;
				if(subwrapper == subwrappers_org[i])
					break;
				BundledStatus status = bundle_subpacket( &oldsuperwrapper,
					child, subwrapper, subpacket_new, started_time, bundle_serial);
				switch(status) {
				case BUNDLE_SUCCESS:
					break;
				case BUNDLE_DISTURBED:
				default:
					if(oldsuperwrapper == *supernode.m_link)
						continue;
					return status;
				}
				subwrappers_org[i] = subwrapper;
				if(subpacket_new) {
					if(subpacket_new->missing()) {
						missing = true;
					}
					assert(&subpacket_new->node() == child.get());
				}
				else
					missing = true;
				break;
			}
		}
		if(is_bundle_root) {
			assert( &supernode == this);
			missing = false;
		}
		newpacket->m_missing = true;

		supernode.m_link->negotiate(started_time);
		//First checkpoint.
		if( !supernode.m_link->compareAndSet(oldsuperwrapper, superwrapper)) {
//			superwrapper = *supernode.m_link;
//			if(superwrapper->m_bundle_serial != bundle_serial)
				return BUNDLE_DISTURBED;
//			oldsuperwrapper = superwrapper;
//			continue;
		}
		oldsuperwrapper = superwrapper;

		//clearing all packets on sub-nodes if not modified.
		bool changed_during_bundling = false;
		for(unsigned int i = 0; i < subnodes->size(); i++) {
			shared_ptr<Node> child(( *subnodes)[i]);
			local_shared_ptr<PacketWrapper> null_linkage;
			if(( *subpackets)[i])
				null_linkage.reset(new PacketWrapper(m_link, i, bundle_serial));
			else
				null_linkage.reset(new PacketWrapper( *subwrappers_org[i], bundle_serial));

			assert( !null_linkage->hasPriority());
			//Second checkpoint, the written bundle is valid or not.
			if( !child->m_link->compareAndSet(subwrappers_org[i], null_linkage)) {
				if(local_shared_ptr<PacketWrapper>( *child->m_link)->m_bundle_serial != bundle_serial)
					return BUNDLE_DISTURBED;
				if(oldsuperwrapper != *supernode.m_link)
					return BUNDLE_DISTURBED;
				changed_during_bundling = true;
				break;
			}
		}
		if(changed_during_bundling)
			continue;

		superwrapper.reset(new PacketWrapper( *superwrapper, bundle_serial));
		if( !missing) {
			local_shared_ptr<Packet> &newpacket(
				reverseLookup(superwrapper->packet(), true, Packet::newSerial()));
			newpacket->m_missing = false;
			STRICT_assert(newpacket->checkConsistensy(newpacket));
		}

		if( !supernode.m_link->compareAndSet(oldsuperwrapper, superwrapper))
			return BUNDLE_DISTURBED;
		oldsuperwrapper = superwrapper;

		break;
	}
	return BUNDLE_SUCCESS;
}

//template <class XN>
//void
//Node<XN>::fetchSubpackets(std::deque<local_shared_ptr<PacketWrapper> > &subwrappers,
//	const local_shared_ptr<Packet> &packet) {
//	for(int i = 0; i < packet->size(); ++i) {
//		const local_shared_ptr<Packet> &subpacket(( *packet->subpackets())[i]);
//		subwrappers.push_back( *( *packet->subnodes())[i]->m_link);
//		if(subpacket)
//			fetchSubpackets(subwrappers, subpacket);
//	}
//}
//template <class XN>
//bool
//Node<XN>::commit_at_super(Transaction<XN> &tr) {
//	Node &node(tr.m_packet->node());
//	for(Transaction<XN> tr_super( *this);; ++tr_super) {
//		local_shared_ptr<Packet> *packet
//			= node.reverseLookup(tr_super.m_packet, false, Packet::SERIAL_NULL, false, 0);
//		if( !packet)
//			return false; //Released.
//		if( *packet != tr.m_oldpacket) {
//			if( !tr.isMultiNodal() && (( *packet)->payload() == tr.m_oldpacket->payload())) {
//				//Single-node mode, the payload in the snapshot is unchanged.
//				tr.m_packet->subpackets() = ( *packet)->subpackets();
//				tr.m_packet->m_missing = ( *packet)->missing();
//			}
//			else {
//				return false;
//			}
//		}
//		node.reverseLookup(tr_super.m_packet, true, tr_super.m_serial, tr.m_packet->missing())
//			= tr.m_packet;
//		if(tr_super.commit()) {
//			tr.m_packet = tr_super.m_packet;
//			return true;
//		}
//	}
//}
template <class XN>
bool
Node<XN>::commit(Transaction<XN> &tr) {
	assert(tr.m_oldpacket != tr.m_packet);
	assert(tr.isMultiNodal() || tr.m_packet->subpackets() == tr.m_oldpacket->subpackets());
	assert(this == &tr.m_packet->node());

	m_link->negotiate(tr.m_started_time);

	local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(tr.m_packet, tr.m_serial));
	for(int retry = 0;; ++retry) {
		local_shared_ptr<PacketWrapper> wrapper( *m_link);
		if(wrapper->hasPriority()) {
			//Committing directly to the node.
			if(wrapper->packet() != tr.m_oldpacket) {
				if( !tr.isMultiNodal() && (wrapper->packet()->payload() == tr.m_oldpacket->payload())) {
					//Single-node mode, the payload in the snapshot is unchanged.
					tr.m_packet->subpackets() = wrapper->packet()->subpackets();
					tr.m_packet->m_missing = wrapper->packet()->missing();
				}
				else {
					STRICT_TEST(s_serial_abandoned = tr.m_serial);
//					fprintf(stderr, "F");
					return false;
				}
			}
//			STRICT_TEST(std::deque<local_shared_ptr<PacketWrapper> > subwrappers);
//			STRICT_TEST(fetchSubpackets(subwrappers, wrapper->packet()));
			STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
			if(m_link->compareAndSet(wrapper, newwrapper)) {
//				STRICT_TEST(if(wrapper->isBundled())
//					for(typename std::deque<local_shared_ptr<PacketWrapper> >::const_iterator
//					it = subwrappers.begin(); it != subwrappers.end(); ++it)
//					assert( !( *it)->hasPriority()));
				return true;
			}
			continue;
		}
		//Unbundling this node from the super packet.
		UnbundledStatus status = unbundle(NULL, tr.m_started_time, m_link, wrapper,
			tr.isMultiNodal() ? &tr.m_oldpacket : NULL, tr.isMultiNodal() ? &newwrapper : NULL);
		switch(status) {
		case UNBUNDLE_W_NEW_SUBVALUE:
			if(tr.isMultiNodal())
				return true;
			continue;
		case UNBUNDLE_SUBVALUE_HAS_CHANGED: {
				STRICT_TEST(s_serial_abandoned = tr.m_serial);
//				fprintf(stderr, "F");
				return false;
			}
		case UNBUNDLE_DISTURBED:
		default:
			continue;
		}
	}
}

template <class XN>
typename Node<XN>::UnbundledStatus
Node<XN>::unbundle(const int64_t *bundle_serial, uint64_t &time_started,
	const shared_ptr<Linkage> &sublinkage, const local_shared_ptr<PacketWrapper> &null_linkage,
	const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<PacketWrapper> *newsubwrapper_returned,
	local_shared_ptr<PacketWrapper> *oldsuperwrapper) {

	assert( !null_linkage->hasPriority());

// Taking a snapshot inside the super packet.
	local_shared_ptr<PacketWrapper> superwrapper(null_linkage);
	local_shared_ptr<Packet> *newsubpacket;
	std::deque<CASInfo> cas_infos;
	SnapshotStatus status = snapshotSupernode(sublinkage, superwrapper, &newsubpacket,
		SNAPSHOT_FOR_UNBUNDLE, bundle_serial ? *bundle_serial : Packet::SERIAL_NULL, &cas_infos);
	switch(status) {
	case SNAPSHOT_SUCCESS:
		break;
	case SNAPSHOT_DISTURBED:
		return UNBUNDLE_DISTURBED;
	case SNAPSHOT_VOID_PACKET:
	case SNAPSHOT_NODE_MISSING:
		newsubpacket = const_cast<local_shared_ptr<Packet> *>( &null_linkage->packet());
		assert(newsubpacket);
		break;
	case SNAPSHOT_NODE_MISSING_AND_COLLIDED:
		newsubpacket = const_cast<local_shared_ptr<Packet> *>( &null_linkage->packet());
		assert(newsubpacket);
		status = SNAPSHOT_COLLIDED;
		break;
	case SNAPSHOT_COLLIDED:
		break;
	}

	if(oldsubpacket && ( *newsubpacket != *oldsubpacket))
		return UNBUNDLE_SUBVALUE_HAS_CHANGED;

	for(auto it = cas_infos.begin(); it != cas_infos.end(); ++it) {
		it->linkage->negotiate(time_started);
		if( !it->linkage->compareAndSet(it->old_wrapper, it->new_wrapper))
			return UNBUNDLE_DISTURBED;
		if(oldsuperwrapper) {
			if( ( *oldsuperwrapper)->packet()->node().m_link == it->linkage) {
				if( *oldsuperwrapper != it->old_wrapper)
					return UNBUNDLE_DISTURBED;
//				printf("1\n");
				*oldsuperwrapper = it->new_wrapper;
			}
		}
	}
	if(status == SNAPSHOT_COLLIDED)
		return UNBUNDLE_COLLIDED;

	local_shared_ptr<PacketWrapper> newsubwrapper;
	if(oldsubpacket)
		newsubwrapper = *newsubwrapper_returned;
	else
		newsubwrapper.reset(new PacketWrapper( *newsubpacket, Packet::SERIAL_NULL));

	if( !sublinkage->compareAndSet(null_linkage, newsubwrapper))
		return UNBUNDLE_SUBVALUE_HAS_CHANGED;

	if(newsubwrapper_returned)
		*newsubwrapper_returned = newsubwrapper;

//	if(oldsuperwrapper) {
//		if( &( *oldsuperwrapper)->packet()->node().m_link != &cas_infos.front().linkage)
//			return UNBUNDLE_DISTURBED;
//		if( *oldsuperwrapper != cas_infos.front().old_wrapper)
//			return UNBUNDLE_DISTURBED;
//		printf("1\n");
//		*oldsuperwrapper = cas_infos.front().new_wrapper;
//	}

	return UNBUNDLE_W_NEW_SUBVALUE;
}

} //namespace Transactional

