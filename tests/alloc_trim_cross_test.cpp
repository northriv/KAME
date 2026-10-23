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
//
// The worker's blocks must lie in a chunk it claims itself.  A new thread
// first takes over an orphaned chunk of its size with room in it, and one
// that other live slots keep claimed never empties: both counts then come out
// 0 and prove nothing.  So it was on Windows at 16 and 32 B, and with gcc on
// Linux at 32 B, the runtime's own small blocks pinning such a chunk.  The
// worker fills whatever chunk it took over until chunks_live rises, and this
// thread frees the fillers after counting.
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include "kame_pool.h"

namespace {
constexpr int N = 1000;   // under the batch's CAP, so no free flushes it
void *g_blocks[N];
constexpr int MAX_FILL = 1 << 16;   // more slots than a 16 B chunk has
void *g_fill[MAX_FILL];
int g_nfill;
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

// Chunks the worker's exit left claimed; `claimed` says whether the worker
// had a chunk of its own to leave.
long long run(std::size_t sz, bool trim, bool &claimed) {
	const long long c0 = chunks_live();
	g_stage = 0;
	g_nfill = 0;
	std::thread w([sz] {
		const long long a0 = chunks_live();
		g_blocks[0] = nullptr;
		while(g_nfill < MAX_FILL) {
			void *p = kame_pool_malloc(sz);
			if(chunks_live() > a0) {
				g_blocks[0] = p;   // the first block of a chunk of its own
				break;
			}
			g_fill[g_nfill++] = p;
		}
		if(g_blocks[0]) {
			std::memset(g_blocks[0], 0, sz);
			for(int i = 1; i < N; ++i) {
				g_blocks[i] = kame_pool_malloc(sz);
				std::memset(g_blocks[i], i & 0xFF, sz);
			}
		}
		set_stage(1);
		wait_stage(2);
	});
	wait_stage(1);
	claimed = g_blocks[0] != nullptr;
	if(claimed)
		for(int i = 0; i < N; ++i)
			kame_pool_free(g_blocks[i]);
	if(trim)
		kame_pool_trim();
	set_stage(2);
	w.join();
	const long long kept = chunks_live() - c0;
	for(int i = 0; i < g_nfill; ++i)
		kame_pool_free(g_fill[i]);
	kame_pool_trim();
	// A chunk the run left orphaned is empty now, but is released only when
	// another thread passes by, and that must not happen inside the next run.
	for(long long c = chunks_live(), prev = c + 1; c < prev; prev = c, c = chunks_live())
		std::thread([sz] { kame_pool_free(kame_pool_malloc(sz)); }).join();
	return kept;
}
} // namespace

int main() {
	// This thread's first pool and thread allocations, in a size of its own:
	// a chunk the worker's frees keep claimed is orphaned, and a later worker
	// allocating its size would take it over rather than add one.
	bool claimed[5], claimed_held[5];
	run(128, true, claimed[0]);
	const std::size_t sizes[] = {16, 32, 48, 64, 200};
	long long kept[5], held[5];
	for(int i = 0; i < 5; ++i)
		kept[i] = run(sizes[i], true, claimed[i]);
	for(int i = 0; i < 5; ++i)
		held[i] = run(sizes[i], false, claimed_held[i]);
	int rc = 0;
	for(int i = 0; i < 5; ++i) {
		std::printf("%zu B: %lld chunks kept after the trim, %lld without it\n",
		            sizes[i], kept[i], held[i]);
		if( !claimed[i] || !claimed_held[i]) {
			std::printf("  FAIL: the worker claimed no chunk of its own\n");
			rc = 1;
		}
		else if(kept[i] > 0) {
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
