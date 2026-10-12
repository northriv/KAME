// A thread that frees one size and goes on to allocate another must not keep
// the first size's memory (allocator.cpp, §tl-cap).  Its own frees go onto
// its list for the bucket, sealed into segments of KAME_TL_SEG_BYTES; past
// KAME_TL_SEGS of them the oldest goes back to its chunks, and a chunk it
// empties is released (parked in the recycle cache, which kame_pool_trim()
// empties).  Before that every freed slot stayed on the list until the
// thread exited: 400 MiB of 64 B freed, then 400 MiB of 1 KiB, held
// 1053 MiB.
//
// Per pair of sizes: PHASE_MB of the first size, painted; all freed;
// kame_pool_trim(); PHASE_MB of the second size, painted and checked.  Live
// units (kame_pool_get_stats) must fall to a small fraction after the trim,
// and the second phase must not stack on the first.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "kame_pool.h"

namespace {
constexpr std::size_t PHASE_MB = 64;

long long units_live() {
	kame_pool_stats_t st;
	std::memset(&st, 0, sizeof st);
	st.version = 1;
	kame_pool_get_stats(&st);
	return static_cast<long long>(st.units_live);
}

std::vector<unsigned char *> fill(std::size_t sz) {
	std::vector<unsigned char *> v(PHASE_MB * 1048576 / sz);
	for(std::size_t i = 0; i < v.size(); ++i) {
		v[i] = static_cast<unsigned char *>(kame_pool_malloc(sz));
		if( !v[i]) { std::fprintf(stderr, "malloc(%zu) failed\n", sz); std::exit(1); }
		std::memset(v[i], static_cast<int>(i & 0xFF), sz);
	}
	return v;
}

unsigned broken_of(const std::vector<unsigned char *> &v, std::size_t sz) {
	unsigned broken = 0;
	for(std::size_t i = 0; i < v.size(); ++i)
		for(std::size_t j = 0; j < sz; j += 61)
			if(v[i][j] != static_cast<unsigned char>(i & 0xFF)) { ++broken; break; }
	return broken;
}

int run(std::size_t a, std::size_t b) {
	// Against the units live before this pair: what the last pair left on
	// the thread's lists (up to the cap per bucket) may go during this one,
	// so a difference can be negative.
	const long long u0 = units_live();
	std::vector<unsigned char *> v = fill(a);
	const long long u1 = units_live() - u0;
	unsigned broken = broken_of(v, a);
	for(auto *p : v) kame_pool_free(p);
	v.clear(); v.shrink_to_fit();
	kame_pool_trim();
	const long long u2 = units_live() - u0;
	v = fill(b);
	const long long u3 = units_live() - u0;
	broken += broken_of(v, b);
	for(auto *p : v) kame_pool_free(p);
	kame_pool_trim();
	std::printf("%zu B -> %zu B, %zu MiB each: live units %lld after the first phase, "
	            "%lld once it is freed and trimmed, %lld after the second; %u blocks overwritten\n",
	            a, b, PHASE_MB, u1, u2, u3, broken);
	int rc = 0;
	if(u2 * 4 > u1) {
		std::fprintf(stderr, "  the freed phase still holds %lld of its %lld units\n", u2, u1);
		rc = 1;
	}
	// The second phase alone takes about what the first did (1 KiB requests
	// round up to 1280 B slots; 64 B and 48 B fill theirs): not their sum.
	if(u3 * 2 > u1 * 3) {
		std::fprintf(stderr, "  the second phase stacked on the first: %lld units against %lld\n", u3, u1);
		rc = 1;
	}
	return (rc || broken) ? 1 : 0;
}
} // namespace

int main() {
	int rc = 0;
	rc |= run(64, 1024);
	rc |= run(1024, 64);
	rc |= run(64, 48);
	rc |= run(4096, 64);
	std::puts(rc ? "FAIL" : "PASS");
	return rc;
}
