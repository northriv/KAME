// §13.134  Working harness for §13.130's back_offset verifier.
//
// §13.132 reports the verifier's self-test driver segfaulting and the poke
// "never triggering".  Both are harness problems, and both have specific
// causes that cost this side the same time:
//
//  1. **The SIGSEGV is expected, and it comes AFTER the detection.**  Once a
//     back_offset entry is corrupted, every subsequent free of a slot in the
//     affected chunk mis-derives chunk_base -- so the process dies in the free
//     path with rc=139.  That is the whole point (it is the failing runs'
//     signature from a one-byte poke), but it means the interesting output must
//     be flushed before it happens: with block-buffered stdout the buffer is
//     lost and the run looks like it printed nothing.  `setvbuf(_IONBF)` first.
//
//  2. **Poking an UNCLAIMED unit is invisible by design.**  The verifier walks
//     `claim_bitmap` and skips units that are not claimed, because an unclaimed
//     entry legitimately reads 0.  A poke at a fixed index (say unit 7) hits an
//     unclaimed unit on most runs and is correctly ignored -- which looks like
//     "the poke never triggers".  Poke a unit the run has actually claimed: try
//     a range and stop at the first index where the verifier's count rises.
//
// Build (needs the pool ACTIVE, i.e. a SHARED lib with KAMEPOOLALLOC_DYLIB --
// compiling allocator.cpp into the executable leaves new/delete on libc and
// kame_pool_reserved_bytes() reads 0; §13.109):
//
//   c++ -std=c++17 -O2 -pthread -shared -fPIC -DKAMEPOOLALLOC_DYLIB \
//       -DUSE_KAME_ALLOCATOR -DKAME_POOL_VERIFY_BACKOFFSET \
//       -I<kamepoolalloc> <kamepoolalloc>/allocator.cpp -o libkp_vbo.so
//   c++ -std=c++17 -O2 -pthread -I<kamepoolalloc> \
//       backoffset_verify_test.cpp ./libkp_vbo.so -o backoffset_verify_test
//
// Expected output: three rounds with `anomalies=0`, then the positive control
// catching the poked unit, then SIGSEGV.
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" unsigned kame_pool_check_back_offset(void **first_region,
                                               unsigned *first_unit,
                                               unsigned *first_val,
                                               unsigned *first_expect) noexcept;
extern "C" int kame_pool_poke_back_offset(unsigned region_ordinal,
                                          unsigned unit,
                                          unsigned val) noexcept;
extern "C" std::size_t kame_pool_reserved_bytes() noexcept;

static unsigned check(const char *tag) {
    void *r = nullptr; unsigned u = 0, val = 0, exp = 0;
    unsigned bad = kame_pool_check_back_offset(&r, &u, &val, &exp);
    printf("%-22s reserved=%zu  anomalies=%u", tag,
           kame_pool_reserved_bytes(), bad);
    if(bad) printf("  first: region=%p unit=%u val=%u expect=%u", r, u, val, exp);
    printf("\n");
    return bad;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);          // see note 1
    std::vector<void *> v;
    unsigned base_rate = 0;
    for(int round = 0; round < 3; ++round) {
        for(int i = 0; i < 200000; ++i)
            v.push_back(::operator new(32 + (i % 7) * 16));
        for(std::size_t i = 0; i < v.size(); i += 2) {
            ::operator delete(v[i]);
            v[i] = nullptr;
        }
        char tag[32]; snprintf(tag, sizeof tag, "round %d", round);
        base_rate += check(tag);
        std::vector<void *> keep;
        for(void *p : v) if(p) keep.push_back(p);
        v.swap(keep);
    }
    printf("--- base rate over 3 rounds: %u anomalies ---\n", base_rate);

    // Positive control (§13.61): the verifier's zero means nothing until it is
    // shown to fire.  Walk until a CLAIMED unit is hit -- see note 2.
    printf("--- positive control ---\n");
    bool fired = false;
    for(unsigned u = 2; u < 64 && !fired; ++u) {
        if(kame_pool_poke_back_offset(0, u, 0x55) != 0) continue;
        void *r = nullptr; unsigned uu = 0, val = 0, exp = 0;
        unsigned bad = kame_pool_check_back_offset(&r, &uu, &val, &exp);
        printf("poked unit %-3u -> anomalies=%u", u, bad);
        if(bad) { printf("  first(unit=%u val=%u expect=%u)  CAUGHT\n", uu, val, exp);
                  fired = true; }
        else printf("  (unit not claimed; correctly ignored)\n");
    }
    if( !fired) {
        printf("POSITIVE CONTROL FAILED -- verifier did not fire; do not trust "
               "a clean table\n");
        return 2;
    }
    printf("NOTE: freeing from here on will SIGSEGV -- expected, see note 1.\n");
    for(void *p : v) ::operator delete(p);
    return 0;
}
