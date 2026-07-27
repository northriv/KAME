/*
 * (§75 / RT_READINESS G7) Per-operation WCET tail harness.
 *
 * Everything else in tests/bench/ measures THROUGHPUT.  A realtime claim
 * lives or dies on the TAIL, so this measures each malloc and each free
 * individually and reports max / p99.9999 — never a mean, which is exactly
 * the statistic that hides a syscall.
 *
 * What it does:
 *   - one measured thread (best-effort elevated priority) + N interferer
 *     threads churning the same size bands, to keep the bitmap CAS, the
 *     recycle cache and the region list contended;
 *   - two arms, INTERLEAVED PER REPETITION (never one arm then the other —
 *     machine state drifts):
 *         RT  = realtime mode + per-thread gating + prewarm
 *         OFF = identical churn with no realtime opt-in
 *   - a log-scale histogram (4 buckets/octave, 1 ns resolution below 64 ns)
 *     so p99.9999 costs O(1) memory and no allocation inside the loop;
 *   - `kame_pool_rt_violations()` asserted to be zero for the RT arm: the
 *     one hard pass/fail here, since absolute latencies are machine-specific
 *     but "a realtime thread entered the kernel" never is.
 *
 * A percentile is printed ONLY when the sample count can support it
 * (>= 10 samples beyond it) — otherwise it would be an artefact of the
 * largest sample, which is what `max` already reports.
 *
 * Usage:
 *   bench_rt_wcet             smoke (CI: seconds, asserts violations == 0)
 *   bench_rt_wcet --full      measurement run (~10^7 samples/band)
 *   bench_rt_wcet --reps N --iters M --threads T
 *
 * Licensed under Apache-2.0 OR GPL-2.0-or-later, as the rest of the tree.
 */
// ------------------------------------------------------- allocator backend
// Default: call kamepoolalloc directly, so the §75 realtime API is available.
// -DWCET_USE_MALLOC: call plain malloc/free instead, so THE SAME harness can
// measure whatever allocator is LD_PRELOADed (glibc, mimalloc, jemalloc, or
// kamepoolalloc's own drop-in).  Comparing worst cases across allocators is
// only meaningful with one harness, one clock and one histogram.
#ifdef WCET_USE_MALLOC
#  include <cstdlib>
#  include <cstring>
#  include <vector>
#  define WCET_MALLOC(sz) std::malloc(sz)
#  define WCET_FREE(p)    std::free(p)
static const char *kWcetBackend = "malloc/free (allocator chosen by LD_PRELOAD)";
// Stand-ins for the realtime API, which is kamepoolalloc's alone.  A stock
// allocator has no equivalent, so the "RT" arm degenerates into a second
// untuned arm — which is the correct control, not a defect.
static inline void kame_pool_set_realtime_thread(int) {}
static inline void kame_pool_set_realtime_mode(int) {}
static inline void kame_pool_rt_drain() {}
static inline void kame_pool_rt_reset_counters() {}
static inline unsigned long long kame_pool_rt_violations() { return 0; }
static inline std::size_t kame_pool_set_thp_policy(int) { return 0; }
static inline unsigned long long kame_pool_rt_deferred_reclaims() { return 0; }
static inline unsigned long long kame_pool_rt_deferred_unmaps() { return 0; }
static inline std::size_t kame_pool_rt_pending_bytes() { return 0; }
enum { KAME_THP_SYSTEM = 0, KAME_THP_ALWAYS = 1, KAME_THP_NEVER = 2 };
//! Portable equivalent of kame_pool_prewarm(): hold `counts[i]` blocks of
//! `sizes[i]` live at once and touch each, so the arena really has to grow,
//! then free them.  Without this the stock arms would be measured cold and
//! the comparison would flatter kamepoolalloc.
static int kame_pool_prewarm(const std::size_t *sizes, const unsigned *counts,
                             unsigned n) {
	for(unsigned i = 0; i < n; i++) {
		std::vector<void *> v;
		v.reserve(counts[i]);
		for(unsigned k = 0; k < counts[i]; k++) {
			void *p = std::malloc(sizes[i]);
			if( !p) break;
			std::memset(p, 0, sizes[i] < 4096 ? sizes[i] : 4096);
			v.push_back(p);
		}
		for(void *p : v) std::free(p);
	}
	return 0;
}
#else
#  include "../../kame_pool.h"
#  define WCET_MALLOC(sz) kame_pool_malloc(sz)
#  define WCET_FREE(p)    kame_pool_free(p)
static const char *kWcetBackend = "kame_pool_malloc/free (direct)";
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#  include <pthread.h>
#  include <sys/qos.h>
#elif defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#endif

// ---------------------------------------------------------------- clock
using clk = std::chrono::steady_clock;
static inline std::uint64_t now_ns() {
	return (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
	           clk::now().time_since_epoch()).count();
}

// ------------------------------------------------------------ histogram
// 1 ns resolution below 64 ns, then 4 buckets per octave.  Fixed size, no
// allocation — the measuring loop must not perturb the allocator under test.
enum { HB = 256 };

struct Hist {
	std::uint64_t bucket[HB];
	std::uint64_t n;
	std::uint64_t max;
	std::uint64_t sum;

	void reset() { std::memset(this, 0, sizeof(*this)); }

	static unsigned idx(std::uint64_t v) {
		if(v < 64) return (unsigned)v;
		unsigned oct = 63u - (unsigned)__builtin_clzll(v);       // >= 6
		unsigned frac = (unsigned)((v >> (oct - 2)) & 3u);
		unsigned i = 64u + (oct - 6u) * 4u + frac;
		return i < (unsigned)HB ? i : (unsigned)HB - 1u;
	}
	//! Upper edge of bucket i — percentiles are reported conservatively.
	static std::uint64_t value(unsigned i) {
		if(i < 64) return i;
		unsigned oct  = 6u + (i - 64u) / 4u;
		unsigned frac = (i - 64u) % 4u;
		return (std::uint64_t)(4u + frac + 1u) << (oct - 2);
	}
	void add(std::uint64_t v) {
		bucket[idx(v)]++;
		n++;
		sum += v;
		if(v > max) max = v;
	}
	//! Smallest value at or below which `p` of the samples fall.
	std::uint64_t pct(double p) const {
		if( !n) return 0;
		std::uint64_t want = (std::uint64_t)(p * (double)n);
		if(want >= n) want = n - 1;
		std::uint64_t acc = 0;
		for(unsigned i = 0; i < HB; i++) {
			acc += bucket[i];
			if(acc > want) return value(i);
		}
		return max;
	}
	//! True iff `n` can support percentile `p` (>= 10 samples beyond it).
	bool supports(double p) const {
		return (double)n * (1.0 - p) >= 10.0;
	}
};

// ------------------------------------------------------------------ bands
struct Band { const char *name; std::size_t size; unsigned live; };
// One per tier: bucketed small, bucketed page-ish, dedicated chunk, large mmap.
static const Band kBands[] = {
	{ "64 B    (bucket)",    64u,                 64u },
	{ "4 KiB   (bucket)",    4096u,               32u },
	{ "256 KiB (dedicated)", 256u * 1024u,         8u },
	{ "8 MiB   (large)",     8u * 1024u * 1024u,   2u },
	// Above LRC_HI (256 MiB) the recycle cache is bypassed BY CONSTRUCTION
	// (`deallocate_large_va`: `mmap_size > LRC_HI || !recycle_push(...)`), so
	// every free here reaches munmap and every alloc reaches mmap.  This is
	// the one band where the RT gate provably acts, which is why it is
	// measured only under `--pressure` (it is also slow and VA-hungry).
	{ "300 MiB (> LRC_HI)",  300u * 1024u * 1024u, 1u },
};
enum { NBANDS = sizeof(kBands) / sizeof(kBands[0]) };
// Bands actually measured: the huge one only under --pressure.
static int g_nbands = NBANDS - 1;

// ------------------------------------------------------- interferer noise
static std::atomic<bool> g_stop{false};

static void interferer() {
	// Deliberately NOT realtime: this is the contention the RT thread must
	// tolerate (bitmap CAS, recycle cache slots, region list walk).
	void *p[16];
	unsigned i = 0;
	while( !g_stop.load(std::memory_order_relaxed)) {
		const Band &b = kBands[i++ & 3u];   // never the huge band
		unsigned n = b.live < 16u ? b.live : 16u;
		unsigned got = 0;
		for(; got < n; got++) {
			p[got] = WCET_MALLOC(b.size);
			if( !p[got]) break;
			*static_cast<char *>(p[got]) = 1;
		}
		for(unsigned j = 0; j < got; j++) WCET_FREE(p[j]);
	}
}

// ------------------------------------------------------------ elevation
static const char *elevate_this_thread() {
#if defined(__APPLE__)
	// QOS_CLASS_USER_INTERACTIVE needs no privilege and is what an audio /
	// control thread would use on macOS.
	if(pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0)
		return "QOS_CLASS_USER_INTERACTIVE";
	return "default (qos elevation failed)";
#elif defined(__linux__)
	sched_param sp;
	std::memset(&sp, 0, sizeof(sp));
	sp.sched_priority = 80;
	if(pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0)
		return "SCHED_FIFO prio 80";
	return "default (SCHED_FIFO needs privilege — tail will include "
	       "scheduler noise)";
#else
	return "default (no elevation on this platform)";
#endif
}


// ==================================================================
// Cross-thread arm: producer allocates, the MEASURED thread frees.
// ==================================================================
// This is the class the same-thread bands above cannot reach, and the one
// that matters most for the STM: a Payload cloned on one thread and released
// on another.  It is also where `CrossDeallocBatch` lives — CAP=1024 entries
// accumulate and then ONE free pays for sorting + merging + CAS-ing the whole
// buffer, so the OFF arm should show a periodic spike that the RT arm (routed
// to the single-slot `push_direct` path) does not.
//
// SIZE MATTERS HERE: only FS=true ALIGN <= 48 uses the batch — sizes 64 B and
// up already take the direct path — so the band must be <= 48 B or the arm
// measures nothing.  32 B it is.
static const std::size_t kXtSize = 32u;

// Fixed-capacity SPSC ring.  Deliberately allocation-free: a queue that
// allocated would pollute the very measurement it feeds.
struct SpscRing {
	enum { N = 8192 };
	void *buf[N];
	std::atomic<unsigned> head{0};      // producer publishes
	std::atomic<unsigned> tail{0};      // consumer consumes
	bool push(void *p) {
		unsigned h = head.load(std::memory_order_relaxed);
		if(h - tail.load(std::memory_order_acquire) >= (unsigned)N)
			return false;               // full
		buf[h % N] = p;
		head.store(h + 1, std::memory_order_release);
		return true;
	}
	void *pop() {
		unsigned t = tail.load(std::memory_order_relaxed);
		if(t == head.load(std::memory_order_acquire)) return nullptr;
		void *p = buf[t % N];
		tail.store(t + 1, std::memory_order_release);
		return p;
	}
};

static SpscRing        g_ring;
static std::atomic<bool> g_xt_stop{false};

static void xt_producer() {
	while( !g_xt_stop.load(std::memory_order_relaxed)) {
		void *p = WCET_MALLOC(kXtSize);
		if( !p) continue;
		*static_cast<char *>(p) = 1;
		while( !g_ring.push(p)) {
			if(g_xt_stop.load(std::memory_order_relaxed)) {
				WCET_FREE(p);      // ring full at shutdown: don't leak it
				return;
			}
		}
	}
}

// Frees `n` blocks from the ring, timing each.  Runs on the measured thread.
static void xt_consume(Hist &h, bool realtime, unsigned n) {
	if(realtime) kame_pool_set_realtime_thread(1);
	unsigned done = 0;
	while(done < n) {
		void *p = g_ring.pop();
		if( !p) continue;               // producer behind; not counted
		std::uint64_t t0 = now_ns();
		WCET_FREE(p);
		std::uint64_t t1 = now_ns();
		h.add(t1 - t0);
		done++;
	}
	if(realtime) kame_pool_set_realtime_thread(0);
}

// ---------------------------------------------------------------- measure
struct Arm {
	Hist alloc[NBANDS];
	Hist free_[NBANDS];
	// (G6a) The USER's first write to the returned block, timed separately.
	// It is not part of malloc's cost, but it is part of the caller's
	// deadline, and it is the only place a transparent-hugepage fault can
	// show up: under THP the kernel may allocate AND ZERO 2 MiB here instead
	// of 4 KiB.  Measured so `--thp never` has something to prove.
	Hist touch[NBANDS];
	unsigned long long violations;
	void reset() {
		for(int i = 0; i < NBANDS; i++) {
			alloc[i].reset(); free_[i].reset(); touch[i].reset();
		}
		violations = 0;
	}
};

// One repetition of one arm.  `realtime` selects the arm.
static void run_rep(Arm &arm, bool realtime, unsigned iters,
                    std::vector<void *> &slots) {
	if(realtime) kame_pool_set_realtime_thread(1);
	for(int bi = 0; bi < g_nbands; bi++) {
		const Band &b = kBands[bi];
		const unsigned live = b.live;
		for(unsigned it = 0; it < iters; it++) {
			for(unsigned k = 0; k < live; k++) {
				std::uint64_t t0 = now_ns();
				void *p = WCET_MALLOC(b.size);
				std::uint64_t t1 = now_ns();
				arm.alloc[bi].add(t1 - t0);
				slots[k] = p;
				if(p) {                                    // first-touch cost
					std::uint64_t t2 = now_ns();
					*static_cast<char *>(p) = (char)k;
					std::uint64_t t3 = now_ns();
					arm.touch[bi].add(t3 - t2);
				}
			}
			for(unsigned k = 0; k < live; k++) {
				std::uint64_t t0 = now_ns();
				WCET_FREE(slots[k]);
				std::uint64_t t1 = now_ns();
				arm.free_[bi].add(t1 - t0);
			}
		}
	}
	if(realtime) kame_pool_set_realtime_thread(0);
}

static void report_hist(const char *label, const Hist &h) {
	static const double kPs[]  = { 0.5, 0.99, 0.999, 0.9999, 0.99999, 0.999999 };
	static const char  *kPn[]  = { "p50", "p99", "p99.9", "p99.99", "p99.999",
	                               "p99.9999" };
	std::printf("    %-22s n=%-10llu mean=%-7llu", label,
	            (unsigned long long)h.n,
	            (unsigned long long)(h.n ? h.sum / h.n : 0));
	for(unsigned i = 0; i < sizeof(kPs) / sizeof(kPs[0]); i++) {
		if(h.supports(kPs[i]))
			std::printf(" %s=%llu", kPn[i],
			            (unsigned long long)h.pct(kPs[i]));
	}
	std::printf("  MAX=%llu ns\n", (unsigned long long)h.max);
}

// ------------------------------------------------- (G6a) cold-fault mode
// The steady-state loop above cannot see a transparent-hugepage fault: it is
// prewarmed, so its pages are already resident, and the large tier hands back
// a pointer whose first page the allocator itself already wrote a header
// into.  The THP hazard lives on COLD memory — the case a realtime program
// hits when its working set outgrows what it prewarmed.
//
// So: take a block above LRC_HI (the recycle cache is bypassed there by
// construction, so every round is a genuinely fresh mmap), and time ONE WRITE
// PER 4 KiB PAGE across it.  Under THP the first touch inside each 2 MiB span
// makes the kernel allocate and zero 2 MiB — a handful of very expensive
// samples and a lot of free ones; under KAME_THP_NEVER every page is its own
// cheap 4 KiB fault.  Mean and tail therefore move in OPPOSITE directions,
// which is precisely the trade the policy exists to let you choose.
static void run_faults(unsigned rounds, std::size_t touch_bytes) {
	const std::size_t BLOCK = 300u * 1024u * 1024u;      // > LRC_HI
	const std::size_t PAGE  = 4096u;
	if(touch_bytes > BLOCK) touch_bytes = BLOCK;
	Hist h; h.reset();
	std::uint64_t worst_span = 0;                        // slowest single page
	for(unsigned r = 0; r < rounds; r++) {
		char *p = static_cast<char *>(WCET_MALLOC(BLOCK));
		if( !p) { std::printf("  (fault mode: allocation failed)\n"); return; }
		for(std::size_t off = 0; off < touch_bytes; off += PAGE) {
			std::uint64_t t0 = now_ns();
			p[off] = (char)r;
			std::uint64_t t1 = now_ns();
			h.add(t1 - t0);
			if(t1 - t0 > worst_span) worst_span = t1 - t0;
		}
		WCET_FREE(p);
		kame_pool_rt_drain();          // give the VA back before the next round
	}
	std::printf("\n  cold first-touch, one write per 4 KiB page on fresh "
	            "(> LRC_HI) memory\n");
	report_hist("cold touch", h);
	// The count of samples far above the 4 KiB-fault cost is the number of
	// hugepage zeroings; printing it makes the mechanism visible rather than
	// inferred from the tail alone.
	std::uint64_t big = 0;
	for(unsigned i = 0; i < (unsigned)HB; i++)
		if(Hist::value(i) > 8000) big += h.bucket[i];
	std::printf("    samples > 8 us: %llu of %llu (%.3f%%)  worst=%llu ns\n",
	            (unsigned long long)big, (unsigned long long)h.n,
	            h.n ? 100.0 * (double)big / (double)h.n : 0.0,
	            (unsigned long long)worst_span);
}

int main(int argc, char **argv) {
	bool full = false, pressure = false;
	unsigned reps = 4, iters = 200, nthreads = 3;
	unsigned xt = 0;   // cross-thread sample count (0 = derive from mode)
	int thp = -1;      // (G6a) -1 = leave the policy alone
	unsigned faults = 0;   // (G6a) cold-fault rounds; 0 = skip that mode
	for(int i = 1; i < argc; i++) {
		if( !std::strcmp(argv[i], "--full")) full = true;
		else if( !std::strcmp(argv[i], "--pressure")) pressure = true;
		else if( !std::strcmp(argv[i], "--xt") && i + 1 < argc)
			xt = (unsigned)std::atoi(argv[++i]);
		else if( !std::strcmp(argv[i], "--reps") && i + 1 < argc)
			reps = (unsigned)std::atoi(argv[++i]);
		else if( !std::strcmp(argv[i], "--iters") && i + 1 < argc)
			iters = (unsigned)std::atoi(argv[++i]);
		else if( !std::strcmp(argv[i], "--threads") && i + 1 < argc)
			nthreads = (unsigned)std::atoi(argv[++i]);
		else if( !std::strcmp(argv[i], "--faults") && i + 1 < argc)
			faults = (unsigned)std::atoi(argv[++i]);
		else if( !std::strcmp(argv[i], "--thp") && i + 1 < argc) {
			const char *a = argv[++i];
			thp = !std::strcmp(a, "always") ? KAME_THP_ALWAYS
			    : !std::strcmp(a, "never")  ? KAME_THP_NEVER
			    : !std::strcmp(a, "system") ? KAME_THP_SYSTEM : -1;
			if(thp < 0) {
				std::fprintf(stderr, "--thp takes system|always|never\n");
				return 2;
			}
		}
		else {
			std::fprintf(stderr,
			    "usage: %s [--full] [--pressure] [--reps N] [--iters M] "
			    "[--threads T] [--thp system|always|never] [--faults R]\n",
			    argv[0]);
			return 2;
		}
	}
	if(full) { reps = 10; iters = 4000; }

	std::printf("== bench_rt_wcet (§75 / G7) ==\n");
	std::printf("backend: %s\n", kWcetBackend);
	std::printf("mode=%s%s reps=%u iters=%u interferers=%u\n",
	            full ? "full" : "smoke", pressure ? " +pressure" : "",
	            reps, iters, nthreads);

	// MEASURED: for every band at or below LRC_HI the recycle cache absorbs
	// the release outright — no madvise, no munmap — even with the cache cap
	// forced to zero (zeroing it is NOT a way to create pressure: a chunk
	// still lands in the per-thread L1, whose byte cut is fixed when the
	// thread ARMS, and the smallest size class fits at idx 0 regardless).
	// So in the default bands the two arms run identical code and the
	// comparison measures scheduler noise only — a real result, but not a
	// test of the gating.
	//
	// `--pressure` therefore adds the > LRC_HI band, where the cache is
	// bypassed by construction so every free reaches munmap.  That is the
	// regime the RT gate exists for, together with cross-thread release and
	// thread exit (both covered by tests/alloc_rt_thread_test.cpp).
	if(pressure) {
		g_nbands = NBANDS;
		std::printf("pressure: added the > LRC_HI band, where the recycle "
		            "cache is bypassed and frees really munmap\n");
	}

	// Clock overhead — the measurement floor.  Every sample above includes
	// one now_ns() pair, so a reported 20 ns op may be mostly this.  Printed
	// rather than subtracted: subtracting would let a negative-looking
	// difference silently clamp.
	{
		Hist ov; ov.reset();
		for(int i = 0; i < 20000; i++) {
			std::uint64_t a = now_ns(), b = now_ns();
			ov.add(b - a);
		}
		std::printf("clock overhead: mean=%llu max=%llu ns\n",
		            (unsigned long long)(ov.sum / (ov.n ? ov.n : 1)),
		            (unsigned long long)ov.max);
	}

	const char *prio = elevate_this_thread();
	std::printf("measured-thread priority: %s\n", prio);

	// Process-wide: silence background maintenance for BOTH arms, so the
	// only difference between them is the per-thread gating under test.
	kame_pool_set_realtime_mode(1);

	// (G6a) THP policy, applied BEFORE prewarm — it has to be, since
	// MADV_NOHUGEPAGE prevents future hugepage faults and khugepaged
	// collapses but does NOT split hugepages that are already established.
	// Both arms share it: it is a process-wide property, not part of the
	// per-thread gating under test, so the A/B for it is across PROCESSES.
	if(thp >= 0) {
		std::size_t re = kame_pool_set_thp_policy(thp);
		std::printf("thp policy=%d (%s), re-advised %zu MiB of existing "
		            "regions\n", thp,
		            thp == KAME_THP_NEVER ? "never"
		            : thp == KAME_THP_ALWAYS ? "always" : "system",
		            re >> 20);
	}

	// Prewarm every band on this thread (the allocator TLS is per-thread).
	{
		// Prewarm the cacheable bands only.  The > LRC_HI band is deliberately
		// NOT prewarmed: it cannot be — the cache is bypassed there, so its
		// mappings are created and destroyed per operation by design.  Its
		// mmap is therefore an expected, counted event, not a violation, which
		// is why `--pressure` relaxes the violation assertion below.
		std::size_t sizes[NBANDS];
		unsigned   counts[NBANDS];
		int nw = NBANDS - 1;
		for(int i = 0; i < nw; i++) {
			sizes[i]  = kBands[i].size;
			counts[i] = kBands[i].live * 2u;      // headroom over the loop
		}
		if(kame_pool_prewarm(sizes, counts, (unsigned)nw) != 0)
			std::printf("WARNING: prewarm did not fit — the RT arm will show "
			            "cold-path outliers\n");
	}

	// (G6a) Cold-fault mode runs BEFORE the interferers start and instead of
	// the steady-state arms: it measures page faults, and the whole point is
	// that nothing else is perturbing the page tables.
	if(faults) {
		run_faults(faults, 32u * 1024u * 1024u);
		return 0;
	}

	std::vector<std::thread> noise;
	for(unsigned i = 0; i < nthreads; i++) noise.emplace_back(interferer);

	Arm rt, off;
	rt.reset();
	off.reset();
	std::vector<void *> slots(64, nullptr);

	kame_pool_rt_reset_counters();
	for(unsigned r = 0; r < reps; r++) {
		// Interleaved, and alternating which arm goes first, so neither arm
		// systematically owns the warmer cache state.
		if(r & 1u) {
			run_rep(off, false, iters, slots);
			run_rep(rt,  true,  iters, slots);
		}
		else {
			run_rep(rt,  true,  iters, slots);
			run_rep(off, false, iters, slots);
		}
		kame_pool_rt_drain();      // settle what the RT arm deferred
	}
	rt.violations = kame_pool_rt_violations();

	g_stop.store(true, std::memory_order_relaxed);
	for(auto &t : noise) t.join();

	for(int bi = 0; bi < g_nbands; bi++) {
		std::printf("\n  %s\n", kBands[bi].name);
		report_hist("RT  malloc",  rt.alloc[bi]);
		report_hist("OFF malloc",  off.alloc[bi]);
		report_hist("RT  1st-touch", rt.touch[bi]);
		report_hist("OFF 1st-touch", off.touch[bi]);
		report_hist("RT  free",    rt.free_[bi]);
		report_hist("OFF free",    off.free_[bi]);
	}

	// ---- cross-thread arm ----
	{
		const unsigned n = xt ? xt : (full ? 2000000u : 120000u);  // >> CAP=1024
		Hist xt_rt, xt_off;
		xt_rt.reset();
		xt_off.reset();
		std::thread prod(xt_producer);
		for(unsigned r = 0; r < reps; r++) {
			if(r & 1u) { xt_consume(xt_off, false, n / reps);
			             xt_consume(xt_rt,  true,  n / reps); }
			else       { xt_consume(xt_rt,  true,  n / reps);
			             xt_consume(xt_off, false, n / reps); }
		}
		g_xt_stop.store(true, std::memory_order_relaxed);
		// Drain so the producer's final push cannot block forever.
		while(void *p = g_ring.pop()) WCET_FREE(p);
		prod.join();
		while(void *p = g_ring.pop()) WCET_FREE(p);

		std::printf("\n  %zu B cross-thread free (producer allocs, we free)\n",
		            kXtSize);
		report_hist("RT  free", xt_rt);
		report_hist("OFF free", xt_off);
		std::printf("    (OFF batches CAP=1024 then one free pays for the "
		            "whole buffer; RT takes push_direct)\n");
	}

	std::printf("\nrt_violations=%llu  deferred_reclaims=%llu  "
	            "deferred_unmaps=%llu  pending_bytes=%zu\n",
	            (unsigned long long)rt.violations,
	            (unsigned long long)kame_pool_rt_deferred_reclaims(),
	            (unsigned long long)kame_pool_rt_deferred_unmaps(),
	            kame_pool_rt_pending_bytes());

	kame_pool_set_realtime_mode(0);

	// The only hard assertion: absolute latencies are machine-specific, but
	// "a realtime thread entered the kernel for a new mapping" is not.  The
	// bands are prewarmed above, so any violation means a path escaped the
	// gating — the regression this harness exists to catch.
	int rc = 0;
	if( !pressure && rt.violations != 0ull) {
		std::printf("FAILED: the realtime arm made %llu new mapping(s) "
		            "despite prewarming\n",
		            (unsigned long long)rt.violations);
		rc = 1;
	}
	else if(pressure) {
		// Expected here, not a failure: the > LRC_HI band bypasses the recycle
		// cache by construction, so each of its allocations maps and each free
		// unmaps.  Counting them is the point — what the RT arm changes is
		// WHEN the unmap happens (rt_drain, outside the critical section),
		// which the band's free-latency tail above shows.
		std::printf("(pressure: %llu mapping(s) expected — the > LRC_HI band "
		            "cannot be cached)\n",
		            (unsigned long long)rt.violations);
	}
	if(kame_pool_rt_pending_bytes() != 0) {
		std::printf("FAILED: %zu bytes still parked after rt_drain\n",
		            kame_pool_rt_pending_bytes());
		rc = 1;
	}
	std::printf("== %s ==\n", rc ? "FAILED" : "PASSED");
	return rc;
}
