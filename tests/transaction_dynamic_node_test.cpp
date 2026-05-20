/*
 * transaction_dynamic_node_test.cpp
 *
 *  Minimal reproducer for the hard-link consistency bug in the STM framework.
 *
 *  Two worker threads each repeatedly create a hard link.  Each thread owns a
 *  static node p1 (always a child of global gn1) and a dynamic node p2 that
 *  is atomically inserted into both p1 and global gn2 — giving p2 two parents
 *  both reachable from gn1 (gn1→p1→p2 and gn1→gn2→p2).  This triggers the
 *  COLLIDED bundle phase and exercises the allSubReachable fix for the "losing
 *  consistensy" abort at transaction.h line 871.
 *
 *  The tree structure maps to the TLA+ 4-node hardlink model
 *  (BundleUnbundle_hardlink_4node) where liveness is verified:
 *    gn1 = Root,  gn2 = B (static child of Root),
 *    p1  = A (static child of Root while active),
 *    p2  = C (dynamically inserted into A and B as a hard link).
 *  p1 is inserted into gn1 non-transactionally before the loop and released
 *  after, so it is always visible from gn1's transaction packet; p2's
 *  double-parent insertion and release are both done in a single transaction.
 */

#include "support_standalone.h"

#include <thread>

#include "transaction.h"

class LongNode;
typedef Transactional::Transaction<LongNode> Transaction;

class LongNode : public Transactional::Node<LongNode> {
public:
	LongNode() : Transactional::Node<LongNode>() {}
	virtual ~LongNode() {}

	struct Payload : public Transactional::Node<LongNode>::Payload {};
};

#include "transaction_impl.h"
template class Transactional::Node<LongNode>;

shared_ptr<LongNode> gn1, gn2;

void
start_routine(void) {
	shared_ptr<LongNode> p1(LongNode::create<LongNode>());
	shared_ptr<LongNode> p2(LongNode::create<LongNode>());
	// p1 is a static child of gn1 for the duration of this thread's work,
	// so it is always reachable from gn1's transaction packet.
	gn1->insert(p1);
	for(int i = 0; i < 500; i++) {
		// Atomically create the hard link: p2 gets two parents (p1 and gn2),
		// both reachable from gn1.  Single transaction matches the TLA+
		// 4-node model where liveness is proved for this structure.
		gn1->iterate_commit_if([=](Transaction &tr1)->bool{
			if( !gn2->insert(tr1, p2))
				return false;
			if( !p1->insert(tr1, p2))  // second parent: hard link
				return false;
			return true;
		});
		// Atomically tear down the hard link.
		gn1->iterate_commit_if([=](Transaction &tr1)->bool{
			if( !gn2->release(tr1, p2))
				return false;
			if( !p1->release(tr1, p2))
				return false;
			return true;
		});
	}
	gn1->release(p1);
}

#define NUM_THREADS 2

int
main(void) {
	for(int k = 0; k < 20; k++) {
		gn1.reset(LongNode::create<LongNode>());
		gn2.reset(LongNode::create<LongNode>());

		gn1->insert(gn2);

		std::thread threads[NUM_THREADS];

		for(int i = 0; i < NUM_THREADS; i++) {
			std::thread th(&start_routine);
			threads[i].swap(th);
		}
		for(int i = 0; i < NUM_THREADS; i++) {
			threads[i].join();
		}
		gn1.reset();
		gn2.reset();
	}
	printf("succeeded\n");
	return 0;
}
