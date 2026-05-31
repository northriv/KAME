// macOS 16 KiB-page madvise-straddle regression test (fix f0d6a487).
//
// Root cause (debug-build crash, address=0x0, reproduced at process exit):
// §15 places each chunk's K_MAX(4 KiB) header in the 4 KiB *below* its 256 KiB
// unit boundary; the slot region starts AT the boundary.  On a target whose
// page size exceeds K_MAX (Apple arm64 = 16 KiB) the old
//   madvise(chunk_base + ALLOC_PAGE_SIZE, chunk_size - ALLOC_PAGE_SIZE, MADV_FREE)
// has page-UNALIGNED ends; XNU rounds advice ranges OUTWARD, so a released
// chunk's MADV_FREE bleeds into the higher neighbour chunk's header page —
// zeroing a LIVE neighbour's embedded PoolAllocator (vtable + m_flags), which
// then crashes its next virtual dispatch (release_dll's ~PoolAllocator, or
// CrossDeallocBatch::flush's batch_return_to_bitmap — both NULL-vtable calls).
// macOS-only (Linux 4 KiB pages: ends already aligned); became fatal once
// reclaim-on-exit became default (cbd0462c).  NOT a double-free: only a page
// reclaim can zero the +K_MAX-resident header object.
//
// This test makes the (otherwise rare, because MADV_FREE zeroing is lazy)
// failure DETERMINISTIC:
//   1. allocate many size-48 objects spanning many adjacent chunks;
//   2. keep alive whole alternating chunks (even units), free whole chunks
//      (odd units) so the freed ones release → deallocate_chunk → madvise;
//   3. apply memory pressure to force the kernel to actually reclaim (zero)
//      the MADV_FREE'd pages;
//   4. check whether any LIVE keeper chunk's header vtable word got zeroed by
//      an adjacent released chunk's straddling madvise.
//
//   pre-fix : some keeper headers zeroed (or SIGSEGV)  → FAIL (exit 1)
//   post-fix: all keeper headers intact                → PASS (exit 0)
//
// §15 layout (verified against a live crash dump): slot region starts at a
// 256 KiB unit boundary U; chunk_base = U - K_MAX; the embedded PoolAllocator
// (whose first word is its vtable pointer) sits at U - 0xFC0.
//
// Usage: alloc_madvise_straddle_repro [objects] [pressure_MiB]
// (Meaningful only on macOS arm64 / 16 KiB pages and with the pool linked in.)
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
// §15 layout constants (ALLOC_CHUNK_K_MAX / ALLOC_CHUNK_HEADER /
// ALLOC_MIN_CHUNK_SIZE in allocator_prv.h; hardcoded here so the repro stays a
// standalone observer that never includes allocator internals).
constexpr uintptr_t K_MAX = 4096, HEADER = 64, UNIT = 262144 /*256 KiB*/;
constexpr size_t OBJ = 48;   // FS=true size-48 bucket (a crash-report template)
inline uintptr_t unit_of(void *p) { return reinterpret_cast<uintptr_t>(p) & ~(UNIT - 1); }
inline void **hdr_vtable(uintptr_t u) { return reinterpret_cast<void **>(u - K_MAX + HEADER); }
} // namespace

int main(int argc, char **argv) {
    int  N = (argc > 1) ? atoi(argv[1]) : 800000;
    long pressure_mb = (argc > 2) ? atol(argv[2]) : 4096;
    long pg = sysconf(_SC_PAGESIZE);
    printf("[straddle_repro] page=%ld N=%d pressure=%ldMiB obj=%zu\n", pg, N, pressure_mb, OBJ);
    if(pg <= (long)K_MAX)
        printf("[straddle_repro] NOTE: page (%ld) <= K_MAX (%lu) — no straddle possible on this "
               "target; test trivially passes (Linux/4 KiB).\n", pg, (unsigned long)K_MAX);
    fflush(stdout);

    // Owner allocates, then EXITS → all chunks keep BIT_OWNED clear, so a later
    // cross-thread free that empties a chunk triggers its release.
    std::vector<char *> objs(N, nullptr);
    std::thread A([&]() {
        for(int i = 0; i < N; ++i) { objs[i] = new char[OBJ]; std::memset(objs[i], 0xAB, OBJ); }
    });
    A.join();

    // Keep WHOLE alternating chunks live (even sorted units); free the rest.
    std::unordered_map<uintptr_t, int> by_unit;
    for(int i = 0; i < N; ++i) if(objs[i]) by_unit[unit_of(objs[i])]++;
    std::vector<uintptr_t> units;
    for(auto &kv : by_unit) units.push_back(kv.first);
    std::sort(units.begin(), units.end());
    std::unordered_set<uintptr_t> keep;
    for(size_t k = 0; k < units.size(); k += 2) keep.insert(units[k]);

    std::unordered_map<uintptr_t, void *> keep_vt;     // unit -> header vtable BEFORE
    for(uintptr_t u : keep) keep_vt[u] = *hdr_vtable(u);
    printf("[straddle_repro] chunks=%zu live=%zu released=%zu\n",
           units.size(), keep.size(), units.size() - keep.size());
    fflush(stdout);

    for(int i = 0; i < N; ++i)
        if(objs[i] && !keep.count(unit_of(objs[i]))) { delete[] objs[i]; objs[i] = nullptr; }

    // flush the FS=true hold-batch so the freed chunks actually release+madvise.
    { std::vector<char *> f; for(int k = 0; k < 16384; ++k) f.push_back(new char[OBJ]);
      for(char *p : f) delete[] p; }

    // force the kernel to reclaim (zero) the MADV_FREE'd pages.
    {
        const long block = 64L * 1024 * 1024;
        std::vector<char *> big;
        for(long b = 0, nb = (pressure_mb * 1024L * 1024L) / block; b < nb; ++b) {
            char *p = static_cast<char *>(std::malloc(block));
            if(!p) break;
            for(long off = 0; off < block; off += pg) p[off] = (char)b;
            big.push_back(p);
        }
        for(char *p : big) std::free(p);
    }

    int zeroed = 0;
    for(auto &kv : keep_vt) {
        if(kv.second != nullptr && *hdr_vtable(kv.first) == nullptr) {
            if(++zeroed <= 5)
                printf("  STRADDLED: live chunk unit=%p hdr@%p  vtable %p -> 0\n",
                       (void *)kv.first, (void *)hdr_vtable(kv.first), kv.second);
        }
    }
    printf("[straddle_repro] checked=%zu live headers, zeroed=%d\n", keep_vt.size(), zeroed);
    if(zeroed > 0) { printf("[straddle_repro] FAIL: neighbour chunk headers zeroed by straddle\n"); return 1; }
    printf("[straddle_repro] PASS: all live neighbour headers intact\n");
    return 0;
}
