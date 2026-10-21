// kame_pool_trim() must give back the cross-thread frees the calling thread
// still holds in its batch (allocator.cpp, CrossDeallocBatch).  A free of a
// slot of up to 240 B in another thread's chunk waits there, its bit still
// set, until the batch fills (CAP = 1024), one of the freeing thread's own
// chunks fills, or that thread exits -- every such free below 64 B, and from
// 64 B the ones the batch holds to measure how they coalesce.  Meanwhile the
// chunk cannot empty.
//
// Per size: a worker allocates N < CAP blocks and waits; this thread frees
// them all and calls kame_pool_trim(); the worker exits.  Its chunk is empty
// by then, so its exit releases it, and chunks_live must be back where it was
// before the worker started.  The trim used to leave the batch alone, and the
// chunk stayed claimed, orphaned, after the worker's exit.  The same without
// the trim must leave the chunk claimed -- else the frees were not held, and
// the first check proved nothing.
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include "kame_pool.h"

namespace {
constexpr int N = 1000;   // under the batch's CAP, so no free flushes it
void *g_blocks[N];
std::mutex g_mtx;
std::condition_variable g_cv;
int g_stage;

long long chunks_live() {
	kame_pool_stats_t st;
	std::memset(&st, 0, sizeof st);
	st.version = 1;
	kame_pool_get_stats(&st);
	return static_cast<long long>(st.chunks_live);
}

void set_stage(int s) {
	{ std::lock_guard<std::mutex> lk(g_mtx); g_stage = s; }
	g_cv.notify_all();
}
void wait_stage(int s) {
	std::unique_lock<std::mutex> lk(g_mtx);
	g_cv.wait(lk, [s] { return g_stage == s; });
}

// Chunks the worker's exit left claimed.
long long run(std::size_t sz, bool trim) {
	const long long c0 = chunks_live();
	g_stage = 0;
	std::thread w([sz] {
		for(int i = 0; i < N; ++i) {
			g_blocks[i] = kame_pool_malloc(sz);
			std::memset(g_blocks[i], i & 0xFF, sz);
		}
		set_stage(1);
		wait_stage(2);
	});
	wait_stage(1);
	for(int i = 0; i < N; ++i)
		kame_pool_free(g_blocks[i]);
	if(trim)
		kame_pool_trim();
	set_stage(2);
	w.join();
	return chunks_live() - c0;
}
} // namespace

int main() {
	// This thread's first pool and thread allocations, in a size of its own:
	// a chunk the worker's frees keep claimed is orphaned, and a later worker
	// allocating its size would take it over rather than add one.
	run(128, true);
	const std::size_t sizes[] = {16, 32, 48, 64, 200};
	long long kept[5], held[5];
	for(int i = 0; i < 5; ++i)
		kept[i] = run(sizes[i], true);
	for(int i = 0; i < 5; ++i) {
		held[i] = run(sizes[i], false);
		kame_pool_trim();
	}
	int rc = 0;
	for(int i = 0; i < 5; ++i) {
		std::printf("%zu B: %lld chunks kept after the trim, %lld without it\n",
		            sizes[i], kept[i], held[i]);
		if(kept[i] > 0) {
			std::printf("  FAIL: the trim left the batched frees held\n");
			rc = 1;
		}
		else if(held[i] <= 0) {
			std::printf("  FAIL: no free was held without the trim either: proves nothing\n");
			rc = 1;
		}
	}
	std::puts(rc ? "FAIL" : "PASS");
	return rc;
}
