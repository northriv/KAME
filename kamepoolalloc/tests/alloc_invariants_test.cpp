// §13.170  Regression apparatus distilled from the §13 hunt.
//
// The §13 investigation produced ~30 diagnostic knobs.  Most answered a
// question and were deleted; three survived, by one rule: each tests an
// invariant that is true BY CONSTRUCTION, and each carries a positive control
// so that a zero means "looked and found none" rather than "could not have
// found any".  That distinction is not pedantry — three separate corrections
// in §13 came from probes that reported plausible numbers while being
// structurally incapable of finding anything (§13.155 call sites outside their
// own #ifdef, §13.163(b) a check in a function the workload never calls,
// §13.164 a lookup keyed differently from its insert).
//
// The invariants asserted here:
//
//   1. back_offset derivation.  Every chunk is built as
//        palloc = ALLOC::create(..., chunk_base + ALLOC_CHUNK_HEADER)
//      so `(char *)palloc == chunk_base + ALLOC_CHUNK_HEADER` holds for every
//      chunk in existence.  A wrong base_idx breaks it.  (§13.163b verified
//      1 144 061 498 derivations with zero failures.)
//
//   2. No double free.  A slot parked on the owner freelist keeps its bitmap
//      bit SET, so the list is the only record that it is free; pushing an
//      address already on the list puts it there twice and two pops hand one
//      block to two live objects.  (§13.166: 0 in 40 runs of the shipping
//      configuration.)
//
// Build (the pool must be ACTIVE, i.e. a SHARED library with
// KAMEPOOLALLOC_DYLIB — compiling allocator.cpp into the executable leaves
// new/delete on libc and the checks never run; §13.109):
//
//   g++ -O2 -std=gnu++17 -fPIC -shared -DKAMEPOOLALLOC_DYLIB \
//       -DKAME_POOL_RESOLVE_CHECK -DKAME_POOL_FREE_CENSUS \
//       -o libkamepoolalloc_checked.so allocator.cpp -ldl
//   g++ -O2 -std=gnu++17 alloc_invariants_test.cpp \
//       -o alloc_invariants_test libkamepoolalloc_checked.so -lpthread
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <thread>
#include <random>

extern "C" {
unsigned long long kame_pool_resolve_ok_count() noexcept;
unsigned long long kame_pool_resolve_bad_count() noexcept;
unsigned long long kame_pool_double_free_count() noexcept;
int                kame_pool_double_free_selftest() noexcept;
std::size_t        kame_pool_reserved_bytes() noexcept;
}

namespace {

// Churn shaped like the workload that found these: many threads, mixed sizes,
// and threads that EXIT while still holding blocks — that is what drives the
// orphan-chain adopt path, which is the half §13.150 proved necessary.
void churn(unsigned seed, int iters) {
    std::mt19937 rng(seed);
    std::vector<void *> live;
    live.reserve(512);
    for(int i = 0; i < iters; ++i) {
        unsigned r = rng();
        std::size_t sz = 16u + (r % 2048u);
        if((r & 3u) != 0u || live.empty()) {
            void *p = ::operator new(sz);
            // touch it: an untouched block never reaches the freelist paths
            *static_cast<unsigned char *>(p) = (unsigned char)i;
            live.push_back(p);
        }
        else {
            std::size_t k = rng() % live.size();
            ::operator delete(live[k]);
            live[k] = live.back();
            live.pop_back();
        }
    }
    // Deliberately leave `live` non-empty on some threads: a thread exiting
    // non-empty is what pushes its chunk onto the orphan chain.
    if((seed & 1u) == 0u)
        for(void *p : live) ::operator delete(p);
}

} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int threads = (argc > 1) ? atoi(argv[1]) : 8;
    const int iters   = (argc > 2) ? atoi(argv[2]) : 20000;
    const int rounds  = (argc > 3) ? atoi(argv[3]) : 4;

    if(kame_pool_reserved_bytes() == 0) {
        fprintf(stderr, "FAIL: pool is NOT active (reserved_bytes == 0).\n"
                "      Link against the SHARED libkamepoolalloc built with\n"
                "      -DKAMEPOOLALLOC_DYLIB, else new/delete stay on libc and\n"
                "      every count below is a vacuous zero (§13.109).\n");
        return 2;
    }

    for(int r = 0; r < rounds; ++r) {
        std::vector<std::thread> ts;
        for(int t = 0; t < threads; ++t)
            ts.emplace_back(churn, (unsigned)(r * threads + t + 1), iters);
        for(auto &t : ts) t.join();
    }

    const unsigned long long rok  = kame_pool_resolve_ok_count();
    const unsigned long long rbad = kame_pool_resolve_bad_count();
    const unsigned long long dfree = kame_pool_double_free_count();
    const int dfst = kame_pool_double_free_selftest();

    printf("back_offset derivations verified : %llu (bad %llu)\n", rok, rbad);
    printf("double frees                     : %llu [self-test %s]\n", dfree,
           dfst == 1 ? "PASS" : (dfst == 0 ? "FAIL" : "not run"));

    int rc = 0;
    // A zero is only evidence when the instrument demonstrably runs.
    if(rok == 0) {
        fprintf(stderr, "FAIL: zero derivations checked — the resolve check did\n"
                "      not run, so 'bad = 0' is vacuous.  Was the library built\n"
                "      with -DKAME_POOL_RESOLVE_CHECK?\n");
        rc = 2;
    }
    if(dfst != 1) {
        fprintf(stderr, "FAIL: the double-free detector's positive control did\n"
                "      not pass, so its count cannot be read as a negative.\n");
        rc = 2;
    }
    if(rbad != 0) {
        fprintf(stderr, "FAIL: %llu back_offset derivations were wrong — a free\n"
                "      resolved to a chunk that is not the block's own.\n", rbad);
        rc = 1;
    }
    if(dfree != 0) {
        fprintf(stderr, "FAIL: %llu double frees — a slot was pushed onto a\n"
                "      freelist twice with no pop between.\n", dfree);
        rc = 1;
    }
    if(rc == 0) printf("OK: allocator invariants hold, and both instruments proved they run.\n");
    return rc;
}
