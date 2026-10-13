// The room a variable-size (FS=false borrow) chunk keeps when it can no
// longer serve a request -- runs too short for it -- goes on its owner's
// fit lists (allocator.cpp, §fit), filed by the largest of the tier's sizes
// it still fits, and a later miss of a size that fits takes it from there.
// Two things can go wrong, and this checks both:
//
//   * the slot a fit hands out must hold its bucket's largest request.  A
//     class is an N from the SIZE slow_allocate passes for the bucket, and
//     for buckets 6..16 that is bucket * 16, one unit more with the borrow
//     prefix than the bucket's table slot -- the carve_room that came before
//     the fit lists sized from the table, its slots overran the next prefix
//     and bench_frame crashed on it.  Read back through
//     kame_pool_malloc_usable_size (the SIZE the slot prefix records).
//   * a chunk must leave the fit lists before it is released.  Round 2 has
//     another thread free every block, which empties the chunks in their
//     bitmaps (the owner's own frees go to its thread lists and never empty
//     a chunk); the owner's next misses release empty neighbours while some
//     of them are filed, then take from the fit lists.
//
// Per tier, fill chunks with blocks that leave a run at the end of every
// bitmap word, so every chunk that stops being the pin is filed for the
// largest of the tier's sizes that run fits; then allocate that size at its
// largest request.  Every block is painted and checked before it is freed,
// in a shuffled order.  A word is a uintptr_t, 64 units on LP64 and 32 on
// ILP32, so the ALIGN 32 filler goes by the width:
//
//   ALIGN 32: filler 160 B (bucket 10, N = 6: ten a word, 4 units left) on LP64,
//             192 B (bucket 12, N = 7: four a word, 4 units left) on ILP32;
//             probe 96 B (bucket 6, N = 4), the tier's smallest
//   ALIGN 64: filler 696 B (bucket 28, N = 11: five a word on LP64 with 9
//             units left, two on ILP32 with 10); probe 568 B (bucket 27, N = 9)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>
#include "kame_pool.h"

namespace {
std::uint64_t rng = 88172645463325252ull;
std::uint64_t rnd() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

struct Blk { unsigned char *p; std::size_t n; unsigned char tag; };

void paint(const Blk &b) { std::memset(b.p, b.tag, b.n); }
bool intact(const Blk &b) {
	for(std::size_t i = 0; i < b.n; ++i)
		if(b.p[i] != b.tag) return false;
	return true;
}

struct Round { unsigned short_slots = 0, inside = 0, broken = 0; bool failed = false; };

// Fillers, then probes, all painted; counts probes too short for their
// request and probes among the fillers' chunks.
Round allocate(const char *name, std::size_t filler, std::size_t probe,
               unsigned n_fill, unsigned n_probe, std::vector<Blk> &all) {
	Round r;
	unsigned char tag = 1;
	std::uintptr_t lo = ~std::uintptr_t(0), hi = 0;
	for(unsigned i = 0; i < n_fill; ++i) {
		auto *p = static_cast<unsigned char *>(kame_pool_malloc(filler));
		if( !p) { std::fprintf(stderr, "%s: malloc(%zu) failed\n", name, filler); r.failed = true; return r; }
		all.push_back({p, filler, tag});
		paint(all.back());
		tag = static_cast<unsigned char>(tag % 250 + 1);
		lo = std::min(lo, reinterpret_cast<std::uintptr_t>(p));
		hi = std::max(hi, reinterpret_cast<std::uintptr_t>(p));
	}
	for(unsigned i = 0; i < n_probe; ++i) {
		auto *p = static_cast<unsigned char *>(kame_pool_malloc(probe));
		if( !p) { std::fprintf(stderr, "%s: malloc(%zu) failed\n", name, probe); r.failed = true; return r; }
		const std::size_t slot_size = kame_pool_malloc_usable_size(p);
		if(slot_size < probe) {
			if(r.short_slots++ < 4)
				std::fprintf(stderr, "%s: malloc(%zu) = %p, its slot holds %zu B\n",
				             name, probe, static_cast<void *>(p), slot_size);
		}
		const auto a = reinterpret_cast<std::uintptr_t>(p);
		if(a > lo && a < hi) ++r.inside;
		all.push_back({p, probe, tag});
		paint(all.back());
		tag = static_cast<unsigned char>(tag % 250 + 1);
	}
	for(auto &b : all)
		if( !intact(b)) ++r.broken;
	return r;
}

unsigned free_shuffled(std::vector<Blk> &all) {
	unsigned broken = 0;
	for(std::size_t i = all.size(); i > 1; --i)
		std::swap(all[i - 1], all[rnd() % i]);
	for(auto &b : all) {
		if( !intact(b)) ++broken;
		kame_pool_free(b.p);
	}
	all.clear();
	return broken;
}

int run_tier(const char *name, std::size_t filler, std::size_t probe, unsigned n_fill, unsigned n_probe) {
	int rc = 0;
	std::vector<Blk> all;
	all.reserve(n_fill + n_probe);
	// Round 1: the fit lists fill as the pins do; the probes take from them.
	Round r1 = allocate(name, filler, probe, n_fill, n_probe, all);
	if(r1.failed) return 1;
	// Another thread frees them: the chunks empty in their bitmaps.
	unsigned broken_x = 0;
	std::thread([&] { broken_x = free_shuffled(all); }).join();
	// Round 2: the owner's misses release empty chunks, some of them filed,
	// and take from what is left on the fit lists.
	Round r2 = allocate(name, filler, probe, n_fill, n_probe, all);
	if(r2.failed) return 1;
	const unsigned broken_own = free_shuffled(all);
	const unsigned broken = r1.broken + broken_x + r2.broken + broken_own;
	std::printf("%s: %u fillers of %zu B, %u probes of %zu B (%u, %u among the fillers): "
	            "%u slots too short, %u blocks overwritten\n",
	            name, n_fill, filler, n_probe, probe, r1.inside, r2.inside,
	            r1.short_slots + r2.short_slots, broken);
	if( !r1.inside || !r2.inside) {
		std::fprintf(stderr, "%s: no probe landed in the fillers' chunks -- no room was reused\n", name);
		rc = 1;
	}
	return (rc || r1.short_slots || r2.short_slots || broken) ? 1 : 0;
}
} // namespace

int main() {
	int rc = 0;
	rc |= run_tier("ALIGN 32", sizeof(std::uintptr_t) == 8 ? 160 : 192, 96, 6000, 3000);
	rc |= run_tier("ALIGN 64", 696, 568, 3000, 3000);
	std::puts(rc ? "FAIL" : "PASS");
	return rc;
}
