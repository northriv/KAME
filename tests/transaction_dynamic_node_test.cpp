/*
 * transaction_dynamic_node_test.cpp
 *
 *  Minimal reproducer for a hard-link consistency bug in the STM framework.
 *
 *  Two worker threads each repeatedly create a hard link: worker-local p2 is
 *  made a child of both worker-local p1 and global gn2, while p1 itself is
 *  inserted into global gn1 (which already owns gn2). This gives p2 two
 *  parents both reachable from gn1, triggering the bug detected by
 *  TRANSACTIONAL_STRICT_assert as "losing consistensy on node ..." at
 *  transaction.h line 871.
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
	for(int i = 0; i < 500; i++) {
		// p2 becomes a child of p1 (non-transactional)
		p1->insert(p2);
		// hard link: p2 also becomes a child of gn2; p1 inserted into gn1,
		// so p2 now has two parents (p1 and gn2) both reachable from gn1.
		gn1->iterate_commit_if([=](Transaction &tr1)->bool{
			if( !gn2->insert(tr1, p2))
				return false;
			if( !gn1->insert(tr1, p1))
				return false;
			return true;
		});
		p1->release(p2);
		gn1->iterate_commit_if([=](Transaction &tr1)->bool{
			if( !gn2->release(tr1, p2))
				return false;
			if( !gn1->release(tr1, p1))
				return false;
			return true;
		});
	}
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
