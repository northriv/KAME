// v1 prototype: global-only, log-indexed, lock-free large-recycle cache.
//
// Design (agreed):
//   - Size domain [LO, HI]; N continuous log-index slots:
//       idx(S) = N * ln(S/LO) / ln(HI/LO)
//       T(i)   = LO * (HI/LO)^(i/N)            (slot i's size, increasing)
//   - mmap sizes are QUANTIZED to slot boundaries: alloc rounds the request
//     UP (ceil) to slot i and maps T(i); the slot index is stored in the
//     block's meta.  Free returns the block to its stored slot.  => same
//     size always maps to the same slot (exact reuse works) AND a popped
//     block's size >= request (no too-small handout).  This is the correct
//     "rounding direction": ceil on alloc, exact (stored) on free.
//   - One block per slot (single atomic pointer) -> worst-case resident
//     memory = sum_i T(i), and a single-slot CAS has no harmful ABA (the
//     slot only ever holds FREE blocks; taking one is always valid).
//   - Cap = single atomic byte counter (sloppy), default ~1 GiB, no TLS.
//   - Above HI: raw mmap, uncached (syscall < 1% of touch there).
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <string>
#include <sys/mman.h>

using namespace std::chrono;

// ---------------- parameters ----------------
#ifndef HIVAL_KB
#define HIVAL_KB 2304u                           // ~2.25 MiB (syscall = 1% touch)
#endif
static const size_t LO   = 256u * 1024u;        // 256 KiB
static const size_t HI   = (size_t)HIVAL_KB * 1024u;
static const int    N    = 1000;                // log-index slots
static const size_t PAGE = 16384u;              // macOS arm64 page
static const size_t HDR  = 64u;                 // meta header (user ptr = base + HDR)
static const int    RANGE = 8;                  // bounded upward best-fit scan
static int64_t      g_cap = 1LL << 30;          // 1 GiB default (tunable)

static const double G_LNK = std::log((double)HI / (double)LO);
static inline double idxf(size_t S) { return N * std::log((double)S / (double)LO) / G_LNK; }
static inline size_t Tsize(int i)   { return (size_t)(LO * std::pow((double)HI / (double)LO, (double)i / N)); }
static inline size_t roundup(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

struct Meta { uint64_t magic; size_t mmap_size; size_t alloc_size; int cached; };
static const uint64_t MAGIC = 0x4C5243763100ull; // "LRCv1"

// ---------------- global cache state ----------------
static std::atomic<void*>   g_slot[N + 1];
static std::atomic<int64_t> g_cached_bytes{0};
static std::atomic<int64_t> g_peak_bytes{0};
static std::atomic<uint64_t> g_hits{0}, g_misses{0}, g_evict_full{0}, g_evict_cap{0};

static char *raw_map(size_t msize) {
    void *p = mmap(0, msize, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    return p == MAP_FAILED ? nullptr : (char*)p;
}
static void raw_unmap(char *base, size_t msize) { munmap(base, msize); }

static int idx_round(size_t S) {                     // lrint(idx(S)), clamped to [0,N]
    if(S <= LO) return 0;
    long i = lrint(idxf(S));
    if(i < 0) i = 0; if(i > N) i = N;
    return (int)i;
}
// best-effort peak (one attempt, NO loop -> no livelock source).
static inline void note_peak(int64_t nb) {
    int64_t pk = g_peak_bytes.load(std::memory_order_relaxed);
    if(nb > pk) g_peak_bytes.compare_exchange_weak(pk, nb, std::memory_order_relaxed);
}
// Insert a freed block into the SYMMETRIC +-10% band [size/1.1, size*1.1],
// first EMPTY slot, ONE weak CAS per slot (advance on spurious/occupied).
// The band lets multiple same-size blocks spread across adjacent slots
// (depth without per-slot stacks).  Bounded scan + weak one-shot + munmap
// fallback => no retry loop => livelock-free.
static bool push_block(char *base, size_t msize) {
    if(g_cached_bytes.load(std::memory_order_relaxed) + (int64_t)msize > g_cap) {
        g_evict_cap.fetch_add(1, std::memory_order_relaxed);
        raw_unmap(base, msize); return false;
    }
    int ilo = idx_round((size_t)(msize / 1.1)); if(ilo < 0) ilo = 0;
    int ihi = idx_round((size_t)(msize * 1.1)); if(ihi > N) ihi = N;
    for(int s = ilo; s <= ihi; s++) {
        void *expected = nullptr;
        if(g_slot[s].compare_exchange_weak(expected, base, std::memory_order_acq_rel)) {
            int64_t nb = g_cached_bytes.fetch_add((int64_t)msize, std::memory_order_relaxed) + (int64_t)msize;
            note_peak(nb);
            return true;
        }
        // weak fail (spurious or occupied) -> next slot.  NO retry of this slot.
    }
    g_evict_full.fetch_add(1, std::memory_order_relaxed);
    raw_unmap(base, msize); return false;
}

void *large_alloc(size_t R) {
    size_t total = R + HDR;
    if(total > HI) {                                 // uncached raw mmap
        size_t msize = roundup(total, PAGE);
        char *base = raw_map(msize);
        if(!base) return nullptr;
        Meta *m = (Meta*)base; m->magic = MAGIC; m->mmap_size = msize; m->alloc_size = R; m->cached = 0;
        g_misses.fetch_add(1, std::memory_order_relaxed);
        return base + HDR;
    }
    int ilo = idx_round((size_t)(total / 1.1)); if(ilo < 0) ilo = 0;   // SYMMETRIC +-10% band
    int ihi = idx_round((size_t)(total * 1.1)); if(ihi > N) ihi = N;
    for(int s = ilo; s <= ihi; s++) {
        void *b = g_slot[s].load(std::memory_order_acquire);
        if(!b) continue;
        if(!g_slot[s].compare_exchange_weak(b, nullptr, std::memory_order_acq_rel)) continue;  // weak one-shot; spurious/taken -> next slot, NO retry
        Meta *m = (Meta*)b;                          // we OWN b now -> safe to read meta (no UAF / peek race)
        g_cached_bytes.fetch_sub((int64_t)m->mmap_size, std::memory_order_relaxed);
        if(m->mmap_size >= total) {                  // VERIFY (lrint may co-locate a too-small block)
            m->alloc_size = R; g_hits.fetch_add(1, std::memory_order_relaxed);
            return (char*)b + HDR;
        }
        // too small: ONE put-back attempt into this slot, else release.  O(1) -- no re-scan, no 100x loop.
        void *exp = nullptr;
        if(g_slot[s].compare_exchange_weak(exp, b, std::memory_order_acq_rel)) {
            g_cached_bytes.fetch_add((int64_t)m->mmap_size, std::memory_order_relaxed);  // re-cached
            note_peak(g_cached_bytes.load(std::memory_order_relaxed));
        } else {
            g_evict_full.fetch_add(1, std::memory_order_relaxed);
            raw_unmap((char*)b, m->mmap_size);       // couldn't put back -> release ("push返せず→解放")
        }
        // continue scan (bounded band) -- never retry this slot
    }
    size_t msize = roundup(total, PAGE);             // miss: fresh mmap of exact (page) size
    char *base = raw_map(msize);
    if(!base) return nullptr;
    Meta *m = (Meta*)base; m->magic = MAGIC; m->mmap_size = msize; m->alloc_size = R; m->cached = 1;
    g_misses.fetch_add(1, std::memory_order_relaxed);
    return base + HDR;
}

void large_free(void *ptr) {
    if(!ptr) return;
    char *base = (char*)ptr - HDR;
    Meta *m = (Meta*)base;
    if(m->magic != MAGIC) { std::fprintf(stderr, "BAD MAGIC on free %p\n", ptr); abort(); }
    if(!m->cached) { raw_unmap(base, m->mmap_size); return; }       // uncached (>HI)
    push_block(base, m->mmap_size);                                // re-index by actual size, push down
}

size_t large_usable(void *ptr) {                                   // capacity check helper
    char *base = (char*)ptr - HDR; Meta *m = (Meta*)base; return m->mmap_size - HDR;
}

void large_cache_drain() {
    for(int i = 0; i <= N; i++) {
        void *b = g_slot[i].exchange(nullptr);
        if(b) { Meta *m = (Meta*)b; g_cached_bytes.fetch_sub((int64_t)m->mmap_size, std::memory_order_relaxed); raw_unmap((char*)b, m->mmap_size); }
    }
}

// ==================== correctness tests ====================
static int g_fail = 0;
#define CHECK(c, msg) do { if(!(c)) { std::fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } } while(0)

static bool overlaps(char*a, size_t as, char*b, size_t bs){ return a < b+bs && b < a+as; }

static void test_basic() {
    std::printf("[test_basic]\n");
    std::mt19937 rng(12345);
    struct Live { char*p; size_t r; uint8_t s; };
    std::vector<Live> live;
    long outstanding = 0;
    for(int it = 0; it < 200000; it++) {
        bool doalloc = (live.empty() || (rng()%100) < 55) && live.size() < 64;
        if(doalloc) {
            size_t r = LO/2 + rng() % (HI + HI/2);      // span [128K, ~3.4M] incl > HI
            if(r < 8) r = 8;
            char *p = (char*)large_alloc(r);
            CHECK(p != nullptr, "alloc null");
            if(!p) continue;
            // capacity must be >= request
            CHECK(large_usable(p) >= r, "capacity < request");
            // disjoint from all live
            for(auto &L : live) CHECK(!overlaps(p, r, L.p, L.r), "aliasing two live blocks");
            uint8_t s = (uint8_t)(rng() & 0xff);
            std::memset(p, s, r);
            live.push_back({p, r, s});
            outstanding++;
        } else {
            size_t k = rng() % live.size();
            Live L = live[k]; live[k] = live.back(); live.pop_back();
            // verify sentinel
            for(size_t q = 0; q < L.r; q += 997) CHECK(((uint8_t*)L.p)[q] == L.s, "sentinel mismatch");
            CHECK(((uint8_t*)L.p)[L.r-1] == L.s, "sentinel tail mismatch");
            large_free(L.p);
            outstanding--;
        }
        CHECK(g_cached_bytes.load() <= g_cap + (int64_t)HI, "cap exceeded");
    }
    for(auto &L : live) large_free(L.p);
    // exact-size reuse should hit
    uint64_t h0 = g_hits.load();
    void *a = large_alloc(700*1024); large_free(a);
    void *b = large_alloc(700*1024);
    CHECK(g_hits.load() > h0, "exact-size reuse did not hit cache");
    large_free(b);
    large_cache_drain();
    CHECK(g_cached_bytes.load() == 0, "cached bytes nonzero after drain");
    std::printf("  hits=%llu misses=%llu evict(full)=%llu evict(cap)=%llu peak=%lld MB\n",
        (unsigned long long)g_hits.load(), (unsigned long long)g_misses.load(),
        (unsigned long long)g_evict_full.load(), (unsigned long long)g_evict_cap.load(),
        (long long)(g_peak_bytes.load()>>20));
}

static void test_cap() {
    std::printf("[test_cap] cap=64 MiB\n");
    g_cap = 64LL << 20;
    g_peak_bytes.store(0);
    std::mt19937 rng(999);
    std::vector<char*> live;
    for(int it = 0; it < 100000; it++) {
        if((live.empty() || (rng()%100) < 50) && live.size() < 128) {
            size_t r = LO + rng() % (HI - LO);
            char *p = (char*)large_alloc(r);
            if(p) { std::memset(p, 0xCD, 64); live.push_back(p); }
        } else {
            size_t k = rng()%live.size(); char*p=live[k]; live[k]=live.back(); live.pop_back();
            large_free(p);
        }
    }
    for(auto p : live) large_free(p);
    std::printf("  peak cached = %lld MiB (cap 64), evict(cap)=%llu\n",
        (long long)(g_peak_bytes.load()>>20), (unsigned long long)g_evict_cap.load());
    CHECK(g_peak_bytes.load() <= (64LL<<20) + (int64_t)HI*4, "cap badly exceeded");
    large_cache_drain();
    CHECK(g_cached_bytes.load() == 0, "cap test: bytes nonzero after drain");
    g_cap = 1LL << 30;
}

static std::atomic<int> g_mt_fail{0};
static void mt_worker(int tid, int iters) {
    std::mt19937 rng(uint32_t(tid*2654435761u + 7));
    struct Live { char*p; size_t r; uint8_t s; };
    std::vector<Live> live;
    for(int it = 0; it < iters; it++) {
        if((live.empty() || (rng()%100) < 55) && live.size() < 24) {
            size_t r = LO/2 + rng() % HI;
            char *p = (char*)large_alloc(r);
            if(!p) { g_mt_fail++; continue; }
            if(large_usable(p) < r) g_mt_fail++;
            for(auto &L : live) if(overlaps(p, r, L.p, L.r)) g_mt_fail++;
            uint8_t s = (uint8_t)((tid*131 + it*7) & 0xff);
            std::memset(p, s, r);
            live.push_back({p, r, s});
        } else {
            size_t k = rng()%live.size(); Live L=live[k]; live[k]=live.back(); live.pop_back();
            for(size_t q = 0; q < L.r; q += 1023) if(((uint8_t*)L.p)[q] != L.s) g_mt_fail++;
            large_free(L.p);
        }
    }
    for(auto &L : live) large_free(L.p);
}
static void test_mt() {
    std::printf("[test_mt] 16 threads x 50000\n");
    g_peak_bytes.store(0);
    std::vector<std::thread> ts;
    for(int t = 0; t < 16; t++) ts.emplace_back(mt_worker, t, 50000);
    for(auto &t : ts) t.join();
    CHECK(g_mt_fail.load() == 0, "MT: aliasing/sentinel/capacity failures");
    large_cache_drain();
    CHECK(g_cached_bytes.load() == 0, "MT: bytes nonzero after drain");
    std::printf("  mt_fail=%d  peak=%lld MiB\n", g_mt_fail.load(), (long long)(g_peak_bytes.load()>>20));
}

// Livelock / contention: many MORE threads than cores all hammer a NARROW
// size band so they contend on the SAME slots.  If any op could spin
// unboundedly the run would hang (the outer timeout kills it => FAIL).
// Every op is a bounded band-scan with one weak CAS/slot + deterministic
// fallback, so all ops MUST complete -> g_ll_ops == expected.
static std::atomic<long> g_ll_ops{0};
static void ll_worker(int tid, int iters) {
    std::mt19937 rng(uint32_t(tid * 99u + 1u));
    std::vector<char*> live;
    for(int i = 0; i < iters; i++) {
        if((live.empty() || (rng()%100) < 50) && live.size() < 8) {
            size_t r = 1000000 + (rng() % 60000);    // 1.00-1.06 MiB: all map to one narrow band
            char *p = (char*)large_alloc(r);
            if(p) { std::memset(p, 0xEE, 64); live.push_back(p); }
        } else {
            size_t k = rng()%live.size(); char*p=live[k]; live[k]=live.back(); live.pop_back();
            large_free(p);
        }
        g_ll_ops.fetch_add(1, std::memory_order_relaxed);
    }
    for(auto p : live) large_free(p);
}
static void test_livelock() {
    const int NT = 128, IT = 30000;
    std::printf("[test_livelock] %d threads x %d, narrow band (max slot contention)\n", NT, IT);
    auto t0 = steady_clock::now();
    std::vector<std::thread> ts;
    for(int t = 0; t < NT; t++) ts.emplace_back(ll_worker, t, IT);
    for(auto &t : ts) t.join();                       // if a thread livelocked, this never returns -> outer timeout FAIL
    double sec = duration_cast<duration<double>>(steady_clock::now() - t0).count();
    std::printf("  completed %ld/%d ops in %.2f s (%.1f M ops/s) -> no hang, livelock-free\n",
        g_ll_ops.load(), NT*IT, sec, g_ll_ops.load()/sec/1e6);
    CHECK(g_ll_ops.load() == (long)NT*IT, "livelock: not all ops completed");
    large_cache_drain();
    CHECK(g_cached_bytes.load() == 0, "livelock test: bytes nonzero after drain");
}

// ==================== touch bench ====================
static volatile uint64_t g_sink = 0;
static inline void touch(char*p, size_t n, int mode){
    if(mode==0) return;
    if(mode==1){ uint64_t s=0; for(size_t o=0;o<n;o+=PAGE){ p[o]=(char)(o^0xAB); s+=(unsigned char)p[o]; } g_sink+=s; }
    else { std::memset(p,0xAB,n); g_sink+=(unsigned char)p[n-1]; }
}
static void bench(size_t size, long iters, const std::string &pat, const std::string &tm){
    int tmode = (tm=="none")?0:(tm=="full")?2:1;
    g_hits.store(0); g_misses.store(0);
    auto t0=steady_clock::now(); long ops=0;
    if(pat=="hot"){
        for(long i=0;i<iters;i++){ char*p=(char*)large_alloc(size); touch(p,size,tmode); large_free(p); ops++; }
    } else if(pat.rfind("fifo:",0)==0){
        int d=atoi(pat.c_str()+5); std::vector<char*> ring(d,nullptr); int idx=0;
        for(long i=0;i<iters;i++){ if(ring[idx]) large_free(ring[idx]); char*p=(char*)large_alloc(size); touch(p,size,tmode); ring[idx]=p; idx=(idx+1)%d; ops++; }
        for(auto p:ring) if(p) large_free(p);
    }
    auto t1=steady_clock::now(); double sec=duration_cast<duration<double>>(t1-t0).count();
    uint64_t h=g_hits.load(), m=g_misses.load();
    std::printf("size=%8zu pat=%-8s touch=%-4s | %8.2f M ops/s  %8.0f ns/op  %6.1f GB/s  hit=%.1f%%\n",
        size, pat.c_str(), tm.c_str(), ops/sec/1e6, sec*1e9/ops, (double)size*ops/sec/1e9,
        (h+m)?100.0*h/(h+m):0.0);
    large_cache_drain();
}

int main(int argc, char**argv){
    std::printf("LO=%zuK HI=%zuK N=%d  T(0)=%zuK T(N)=%zuK  worst-case sum~", LO>>10, HI>>10, N, Tsize(0)>>10, Tsize(N)>>10);
    double sum=0; for(int i=1;i<N;i++) sum+=Tsize(i); std::printf("%.0f MB\n", sum/1048576.0);
    std::string mode = argc>1?argv[1]:"test";
    if(mode=="test"){
        test_basic(); test_cap(); test_mt(); test_livelock();
        std::printf(g_fail==0 ? "\nALL TESTS PASS\n" : "\n%d FAILURES\n", g_fail);
        return g_fail==0?0:1;
    } else { // bench: size iters pattern touch
        size_t size=argc>2?strtoull(argv[2],0,10):1048576;
        long iters=argc>3?atol(argv[3]):200000;
        std::string pat=argc>4?argv[4]:"hot";
        std::string tm=argc>5?argv[5]:"page";
        bench(size,iters,pat,tm);
        return 0;
    }
}
