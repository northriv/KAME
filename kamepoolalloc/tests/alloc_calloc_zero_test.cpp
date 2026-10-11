// calloc must return zeros in every size tier: on memory it maps fresh, and
// on a block that an earlier owner wrote and freed.
//
// kame_calloc memsets only memory that may hold an earlier owner's bytes.
// Since 9bb57a41f a fresh §19 mapping (an anonymous mmap, already zero) skips
// the memset -- allocate_large_va reports one through its `zeroed`
// out-parameter -- while everything else keeps it: a large block recycled
// through the warm cache, every bucket slot and dedicated chunk, and all of
// Windows, whose spans come from _aligned_malloc.  The case that must not
// regress is the recycled one.  For each size:
//
//   1. calloc and check (the first large-tier calloc maps fresh: the skip);
//   2. free it, malloc the size again (the cache hands the block back), fill
//      it with 0xA5 and free it;
//   3. calloc again and check every byte.
//
// In the cached large tier step 3 must get step 2's block back -- asserted,
// as alloc_huge_test asserts warm reuse, so the memset branch is known to
// have run.  With one thread that is deterministic: a free takes the first
// empty cache slot from the thread's start index, a pop the first block that
// fits, and step 2's malloc took the first that fit.  The global cache's
// lazy drain sweeps one slot per 10 ms from the smallest class up, far from
// the 40 / 100 MiB slots within this test.  Elsewhere reuse is reported only.
//
// The tiers (ALLOC_MAX_BUCKETED_SIZE = 32 KiB; a chunk's payload is
// ALLOC_MAX_CHUNK_SIZE - ALLOC_CHUNK_K_MAX = 4 MiB - 4096; LRC_HI = 256 MiB):
//
//   bucket      up to 32 KiB            pool slot          always memset
//   dedicated   up to 4 MiB - 4096      dedicated chunk    always memset
//   large       mapping up to 256 MiB   §19 span, cached   memset if recycled
//   huge        mapping over 256 MiB    §19 span, never    no memset
//                                       cached
//
// Through kame_pool_calloc / kame_pool_malloc / kame_pool_free on every
// platform, and through plain calloc / malloc / free where both reach the
// pool (Linux; the macOS dylib) -- probed with kame_pool_malloc_usable_size,
// which is 0 for a pointer the pool does not own.  kame_pool_trim() before
// the huge case empties the caches, so each pass meets the large tier fresh.
//
// Licensed under Apache-2.0 OR GPL-2.0-or-later, as the rest of the tree.

#include "kame_pool.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_fails = 0;
#define CHECK(c, ...) do { if( !(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * KiB;

struct Case {
	std::size_t size;
	const char *tier;
	bool cached_large;  // reuse asserted
	bool huge;          // never cached; a mapping refused is not a failure
};

const Case CASES[] = {
	{8, "bucket", false, false},
	{64, "bucket", false, false},
	{368, "bucket", false, false},             // the largest FS=true class
	{369, "bucket", false, false},             // the smallest FS=false class
	{1000, "bucket", false, false},
	{4096, "bucket", false, false},
	{32 * KiB, "bucket", false, false},        // the largest bucket
	{32 * KiB + 1, "dedicated", false, false},
	{300 * KiB, "dedicated", false, false},
	{MiB + 17, "dedicated", false, false},
	{4 * MiB - 4096, "dedicated", false, false},  // the largest chunk payload
	{4 * MiB - 4095, "large", true, false},       // the smallest §19 span
	{5 * MiB + 123, "large", true, false},
	{40 * MiB, "large", true, false},  // past the per-thread cache: the global one
	{100 * MiB, "large", true, false},
	{257 * MiB, "huge", false, true},
};

struct Api {
	const char *name;
	void *(*calloc_fn)(std::size_t, std::size_t);
	void *(*malloc_fn)(std::size_t);
	void (*free_fn)(void *);
};

void *pool_calloc(std::size_t n, std::size_t s) { return kame_pool_calloc(n, s); }
void *pool_malloc(std::size_t n) { return kame_pool_malloc(n); }
void pool_free(void *p) { kame_pool_free(p); }
void *libc_calloc(std::size_t n, std::size_t s) { return std::calloc(n, s); }
void *libc_malloc(std::size_t n) { return std::malloc(n); }
void libc_free(void *p) { std::free(p); }

// The first non-zero byte of [p, p + n), or n.
std::size_t first_nonzero(const unsigned char *p, std::size_t n) {
	std::size_t i = 0;
	for(; i + 4096 <= n; i += 4096) {
		std::uint64_t acc = 0;
		for(std::size_t j = 0; j < 4096; j += 8) {
			std::uint64_t w;
			std::memcpy(&w, p + i + j, 8);
			acc |= w;
		}
		if(acc) break;
	}
	for(; i < n; ++i)
		if(p[i]) return i;
	return n;
}

// Steps 1-3 for one size.
void run_case(const Api &api, const Case &c) {
	const std::size_t n = c.size;
	// Release the cached blocks before the huge one: the peak stays near one
	// huge block, and the next pass starts from fresh mappings again.
	if(c.huge)
		kame_pool_trim();
	unsigned char *p = static_cast<unsigned char *>(api.calloc_fn(1, n));
	CHECK(p, "%s: calloc(1, %zu) returned NULL", api.name, n);
	if( !p) return;
	std::size_t at = first_nonzero(p, n);
	CHECK(at == n, "%s: calloc(1, %zu) (%s, first): byte %zu is 0x%02x",
	      api.name, n, c.tier, at, at < n ? p[at] : 0u);
	api.free_fn(p);

	unsigned char *d = static_cast<unsigned char *>(api.malloc_fn(n));
	CHECK(d, "%s: malloc(%zu) returned NULL", api.name, n);
	if( !d) return;
	std::memset(d, 0xA5, n);
	const std::uintptr_t dirty = reinterpret_cast<std::uintptr_t>(d);
	api.free_fn(d);

	// n / 8 elements of 8 B where that is exact, so the multiply runs too.
	unsigned char *q = static_cast<unsigned char *>(
	    n % 8 == 0 ? api.calloc_fn(n / 8, 8) : api.calloc_fn(n, 1));
	CHECK(q, "%s: calloc of %zu B returned NULL", api.name, n);
	if( !q) return;
	const bool reused = reinterpret_cast<std::uintptr_t>(q) == dirty;
	const std::size_t usable = kame_pool_malloc_usable_size(q);
	if(c.huge) {
		if(usable < n)
			std::printf("  [info] %s: %zu B came from the system allocator "
			            "(the mapping was refused)\n", api.name, n);
	} else {
		CHECK(usable >= n, "%s: calloc of %zu B (%s) is not the pool's "
		      "(usable size %zu)", api.name, n, c.tier, usable);
	}
	at = first_nonzero(q, n);
	CHECK(at == n, "%s: calloc of %zu B (%s) after a 0xA5-filled block was "
	      "freed%s: byte %zu is 0x%02x", api.name, n, c.tier,
	      reused ? " (the same block)" : "", at, at < n ? q[at] : 0u);
	if(c.cached_large)
		CHECK(reused, "%s: %zu B: the calloc did not get the freed block back "
		      "(%p, then %p), so the recycled branch did not run", api.name, n,
		      reinterpret_cast<void *>(dirty), static_cast<void *>(q));
	std::printf("  [%s] %-4s %-9s %10zu B  %s\n", at == n ? "ok" : "BAD",
	            api.name, c.tier, n,
	            c.huge ? "uncached tier" : reused ? "reused" : "not reused");
	api.free_fn(q);
}

} // namespace

int main() {
	const Api pool = {"pool", pool_calloc, pool_malloc, pool_free};
	for(const Case &c : CASES)
		run_case(pool, c);

	// Plain calloc / malloc / free, where both reach the pool.
	void *m = std::malloc(64);
	void *z = std::calloc(8, 8);
	const bool plain = m && z && kame_pool_malloc_usable_size(m) >= 64
	                   && kame_pool_malloc_usable_size(z) >= 64;
	std::free(m);
	std::free(z);
	if(plain) {
		const Api libc = {"libc", libc_calloc, libc_malloc, libc_free};
		for(const Case &c : CASES)
			run_case(libc, c);
	} else {
		std::printf("  [info] plain malloc / calloc do not both reach the "
		            "pool here; only the kame_pool_* API was checked\n");
	}

	std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
	return g_fails ? 1 : 0;
}
