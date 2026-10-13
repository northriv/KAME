// Every over-aligned allocation must come back aligned and big enough
// (kame_pool_aligned_alloc / kame_pool_posix_memalign; allocator.cpp,
// new_redirected_aligned).  Up to 4 KiB the pool takes the smallest bucket
// whose kBucketAlign the alignment divides -- a bucket's slots lie ALIGN
// apart from a 256 KiB boundary -- so that table has to say what each
// bucket's template is; past 4 KiB, up to 256 KiB, it takes a dedicated
// chunk.  On ILP32 the table once said 256 for bucket 16, whose template is
// ALIGN 128 there: a 256-aligned request of up to 248 B was 128 B off in
// every other slot.
//
// For each alignment from 32 B to 256 KiB and sizes around it, KEEP blocks
// at a time (so the slots fall at odd and even positions), through both
// calls: each checked for alignment and usable size, painted, checked again
// once its round is allocated, and freed.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>
#include "kame_pool.h"

namespace {
constexpr unsigned KEEP = 64;
// Served by the pool at every alignment here, Windows included.
constexpr std::size_t MAX_SIZE = 64u << 10;

unsigned failures = 0;
void fail(const char *what, std::size_t al, std::size_t sz, const void *p) {
	if(failures++ < 20)
		std::fprintf(stderr, "%s: alignment %zu, size %zu -> %p\n", what, al, sz, p);
}

void round_of(std::size_t al, std::size_t sz, bool posix) {
	std::vector<unsigned char *> v;
	v.reserve(KEEP);
	for(unsigned i = 0; i < KEEP; ++i) {
		void *p = nullptr;
		if(posix) {
			if(kame_pool_posix_memalign(&p, al, sz) != 0) p = nullptr;
		}
		else
			p = kame_pool_aligned_alloc(al, sz);
		if( !p) { fail("no memory", al, sz, p); continue; }
		if(reinterpret_cast<std::uintptr_t>(p) & (al - 1)) fail("misaligned", al, sz, p);
		if(kame_pool_malloc_usable_size(p) < sz) fail("usable size short", al, sz, p);
		std::memset(p, static_cast<int>(v.size() & 0xFF), sz);
		v.push_back(static_cast<unsigned char *>(p));
	}
	for(std::size_t i = 0; i < v.size(); ++i) {
		for(std::size_t j = 0; j < sz; j += 61)
			if(v[i][j] != static_cast<unsigned char>(i & 0xFF)) { fail("overwritten", al, sz, v[i]); break; }
		kame_pool_free(v[i]);
	}
}
} // namespace

int main() {
	const std::size_t common[] = {1, 16, 24, 100, 128, 200, 248, 256, 1000, 1528, 4000, 5000, 17000, 40000};
	unsigned rounds = 0;
	for(std::size_t al = 32; al <= (256u << 10); al <<= 1) {
		std::vector<std::size_t> sizes(std::begin(common), std::end(common));
		for(std::size_t s : {al / 2, al - 8, al, al + 8, 2 * al + 8, 3 * al})
			if(s <= MAX_SIZE) sizes.push_back(s);
		for(std::size_t sz : sizes) {
			round_of(al, sz, false);
			round_of(al, sz, true);
			rounds += 2;
		}
	}
	std::printf("%u rounds of %u blocks, alignments 32 B .. 256 KiB: %u failures\n", rounds, KEEP, failures);
	std::puts(failures ? "FAIL" : "PASS");
	return failures ? 1 : 0;
}
