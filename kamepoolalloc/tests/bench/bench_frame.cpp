// bench_frame — allocator hitches in a game-style frame loop.
//
// What a game developer asks of a general-purpose allocator is less its
// peak rate than that it never makes a frame late.  This bench runs a
// frame loop shaped like an engine's and reports the frame-time
// distribution and the latency of every allocator call made inside a frame.
//
// Threads: a game (main) thread, a render thread and W job workers meet at
// a barrier at the start and at the end of every frame; a streaming thread
// runs on its own.  In each frame:
//
//   game     replaces entities (a long-lived table of small objects),
//            allocates and frees per-frame temporaries, records render
//            commands (small, some medium), frees the job outputs of the
//            previous frame, and unloads streamed assets once the resident
//            ones exceed a budget;
//   render   frees the previous frame's commands -- every one allocated on
//            the game thread -- and allocates its own temporaries;
//   workers  run jobs: scratch memory allocated, touched and freed (small,
//            a few medium, now and then a 256 KiB - 4 MiB buffer), plus
//            outputs left for the game thread, and their own entity churn.
//   streaming  (asynchronous) loads assets of 1 - 32 MiB at a set rate,
//            writing one byte per 4 KiB as a load would, and hands each to
//            the game thread; the game thread frees the oldest when over
//            budget, so large blocks are freed away from where they were
//            allocated, inside a frame.
//
// Every block is written when allocated (both ends, and every 4 KiB), so a
// page fault on fresh memory lands where it would in a game.  Every malloc
// and free on the frame threads is timed.  The frame time is from the
// release of the start barrier to the release of the end barrier.  Between
// the two the game thread does its idle-time work: with --rt that is
// kame_pool_rt_drain(), timed on its own (a game would run it while
// waiting for vsync), and with --frame-ms the wait for the next frame.
//
// Routed through plain malloc/free and NOT linked against kamepoolalloc:
// the allocator under test is whatever LD_PRELOAD / DYLD_INSERT_LIBRARIES
// supplies, so one binary measures every arm.  --rt configures kamepoolalloc
// as a game would -- kame_pool_set_realtime_mode(1) and
// kame_pool_set_realtime_default(KAME_RT_DEFER) at startup, so no free makes
// a syscall, and kame_pool_rt_drain() once per frame -- when the preloaded
// library exports those calls; with any other allocator it does nothing,
// which makes it the control.  The harness's own arrays come from mmap, so
// the allocator under test sees only the workload.
//
// Usage:
//   bench_frame [--workers N] [--frames N] [--seconds S] [--warmup S]
//               [--scale F] [--work N] [--stream-mibps R]
//               [--stream-budget-mib B] [--frame-ms P] [--rt] [--seed N]
//     workers            job threads                         (default 6)
//     frames             measured frames, at most            (default 10000)
//     seconds            time limit of the measured frames   (default 60)
//     warmup             unmeasured seconds first            (default 3)
//     scale              multiplies the per-frame counts     (default 1)
//     work               compute steps per job               (default 4000)
//     stream-mibps       streaming rate, 0 = off             (default 256)
//     stream-budget-mib  resident streamed assets            (default 256)
//     frame-ms           frame period, 0 = back to back      (default 0)
//
// Output, one line; times in us (frames) and ns (calls).  hitch2x counts the
// frames longer than twice the median; slow_calls_count the calls of 100 us
// or more.  The call percentiles come from a histogram with 8 steps per
// power of two (each value is its step's upper edge); the maxima are exact.
// On Apple silicon the clock steps in 41.7 ns.
//   [bench_frame] threads=8 workers=6 rt=off frames=10000 secs=9.8 fps=1020
//   frame_med_us=950 frame_p99_us=1210 frame_p999_us=1540 frame_max_us=2900
//   hitch2x=3 call_med_ns=42 call_p999_ns=900 call_p9999_ns=4100
//   call_max_ns=310000 slow_calls_count=12 calls_M=405.1 drain_max_us=0
//   stream_loads=290 rss_mib=420 status=ok
// Exit status: 0 ok, 2 usage, 4 malloc failed.
//
// Licensed under Apache-2.0 OR GPL-2.0-or-later, as the rest of the tree.

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#  include <sys/qos.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#  include <x86intrin.h>
#endif

#include "kame_pool.h"   // for kame_pool_stats_t only; nothing is linked

namespace {

// ---------------------------------------------------------------- params
int         g_workers     = 6;
std::size_t g_frames      = 10000;
double      g_seconds     = 60.0;
double      g_warmup      = 3.0;
double      g_scale       = 1.0;
unsigned    g_work        = 4000;
double      g_stream_mibps = 256.0;
double      g_stream_budget_mib = 256.0;
double      g_frame_ms    = 0.0;
bool        g_rt          = false;
bool        g_verbose     = false;
double      g_kame_cache_mib = -1;     // --kame-cache-mib: large-cache cap
std::uint64_t g_seed      = 1;

// Per-frame counts at scale 1.
enum : unsigned {
	G_TABLE = 100000, G_CHURN = 2000, G_TEMP = 2000, G_CMDS = 1500,
	R_TEMP = 1000,
	W_TABLE = 20000, W_CHURN = 500, W_JOBS = 8, J_SMALL = 200, J_MEDIUM = 5,
	W_OUT = 50,
};
constexpr double J_LARGE_P = 0.05;      // a large scratch buffer per job
constexpr double CMD_MEDIUM_P = 0.05;   // a render command that is medium
constexpr int    MAX_THREADS = 66;

// ---------------------------------------------------------------- clock
using clk = std::chrono::steady_clock;
inline std::uint64_t now_ns() {
	return (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
	           clk::now().time_since_epoch()).count();
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
	__asm__ __volatile__("yield");
#endif
}

// Call timing reads the hardware counter directly: steady_clock costs tens
// of ns per read on macOS, which would dilute the differences measured.  No
// isb / lfence: serialising every read slows the surrounding frame more than
// it sharpens a single call's time, and the tail this bench is after is in
// microseconds.
#if defined(__aarch64__)
inline std::uint64_t ticks() {
	std::uint64_t v;
	__asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v) :: "memory");
	return v;
}
#elif defined(__x86_64__) || defined(__i386__)
inline std::uint64_t ticks() { return __rdtsc(); }
#else
inline std::uint64_t ticks() { return now_ns(); }
#endif
double g_ns_per_tick = 1.0;
std::uint64_t g_slow_ticks;      // 100 us in ticks

void calibrate_ticks() {
	const std::uint64_t n0 = now_ns(), t0 = ticks();
	std::uint64_t n1;
	while((n1 = now_ns()) - n0 < 50000000) {}   // 50 ms
	const std::uint64_t t1 = ticks();
	g_ns_per_tick = t1 > t0 ? (double)(n1 - n0) / (double)(t1 - t0) : 1.0;
	g_slow_ticks = (std::uint64_t)(100000.0 / g_ns_per_tick);
}

void *map_zeroed(std::size_t bytes) {
	if( !bytes) bytes = 1;
	void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
	               MAP_PRIVATE | MAP_ANON, -1, 0);
	if(p == MAP_FAILED) {
		std::perror("mmap");
		std::exit(4);
	}
	std::memset(p, 0, bytes);
	return p;
}

// ---------------------------------------------------------------- random
struct Rng {
	std::uint64_t x;
	explicit Rng(std::uint64_t s) : x(s * 0x9e3779b97f4a7c15ull + 1) {}
	std::uint64_t next() {          // xorshift64*
		x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
		return x * 0x2545f4914f6cdd1dull;
	}
	std::size_t below(std::size_t n) { return (std::size_t)((next() >> 11) % n); }
	double unit() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
	std::size_t log_uniform(std::size_t lo, std::size_t hi) {
		return (std::size_t)((double)lo * std::exp(unit() * std::log((double)hi / (double)lo)));
	}
	//! Engine-like small objects: mostly under 256 B, a tail to 4 KiB.
	std::size_t small() {
		const unsigned r = (unsigned)below(100);
		if(r < 70) return 16 + below(240);
		if(r < 90) return 256 + below(768);
		return 1024 + below(3072);
	}
	std::size_t medium() { return log_uniform(4096, 65536); }
	std::size_t large() { return log_uniform(256 * 1024, 4 * 1024 * 1024); }
	std::size_t asset() { return log_uniform(1024 * 1024, 32 * 1024 * 1024); }
};

inline void touch(void *p, std::size_t n, std::uint64_t v) {
	char *c = static_cast<char *>(p);
	c[0] = (char)v;
	c[n - 1] = (char)(v >> 8);
	for(std::size_t off = 4096; off < n; off += 4096) c[off] = (char)v;
}

// ---------------------------------------------------------------- stats
// Latency histogram, 8 steps per power of two.
enum : int { HB = 8 * 41 };
inline int hbucket(std::uint64_t d) {
	if(d < 8) return (int)d;
	const int msb = 63 - __builtin_clzll(d);
	return msb * 8 + (int)((d >> (msb - 3)) & 7);
}
inline std::uint64_t hupper(int b) {        // the step's upper edge
	if(b < 8) return (std::uint64_t)b;
	const int msb = b / 8, sub = b % 8;
	return ((std::uint64_t)(8 + sub + 1) << (msb - 3)) - 1;
}

struct alignas(64) ThreadStats {
	std::uint64_t hist[HB];
	std::uint64_t calls, slow, max_ns, sum_ns, sink;
	bool failed;
	void add(std::uint64_t d) {
		int b = hbucket(d);
		if(b >= HB) b = HB - 1;
		++hist[b];
		++calls;
		sum_ns += d;
		if(d >= g_slow_ticks) ++slow;
		if(d > max_ns) max_ns = d;
	}
};
ThreadStats *g_stats;            // mmap'd, one per frame thread
std::atomic<bool> g_record{false};
std::atomic<bool> g_stop{false};

struct Ctx {
	ThreadStats *st;
	Rng rng;
	bool rec;
};

inline void *tmalloc(Ctx &c, std::size_t n) {
	const std::uint64_t a = ticks();
	void *p = std::malloc(n);
	const std::uint64_t b = ticks();
	if(c.rec) c.st->add(b - a);
	if( !p) c.st->failed = true;
	return p;
}
inline void tfree(Ctx &c, void *p) {
	const std::uint64_t a = ticks();
	std::free(p);
	const std::uint64_t b = ticks();
	if(c.rec) c.st->add(b - a);
}

inline void compute(Ctx &c, unsigned steps) {  // stands in for frame work
	std::uint64_t acc = 0;
	for(unsigned i = 0; i < steps; ++i) acc += c.rng.next() >> 60;
	c.st->sink += acc;
}

// ---------------------------------------------------------------- barrier
struct Barrier {
	alignas(64) std::atomic<unsigned> count{0};
	alignas(64) std::atomic<unsigned> gen{0};
	std::uint64_t release_ns = 0;
	unsigned n = 0;
	void wait() {
		const unsigned g = gen.load(std::memory_order_acquire);
		if(count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
			count.store(0, std::memory_order_relaxed);
			release_ns = now_ns();
			gen.store(g + 1, std::memory_order_release);
			return;
		}
		// Spin, then yield, then nap: a long --frame-ms wait must not burn
		// the cores the next frame needs.
		unsigned spins = 0;
		std::uint64_t t0 = 0;
		while(gen.load(std::memory_order_acquire) == g) {
			if(++spins < 4000) { cpu_relax(); continue; }
			if( !t0) t0 = now_ns();
			if(now_ns() - t0 < 200000) sched_yield();
			else usleep(20);
		}
	}
};
Barrier g_start, g_end;

// ---------------------------------------------------------------- shared
int g_threads;                   // frame threads: game + render + workers
unsigned n_g_churn, n_g_temp, n_g_cmds, n_r_temp, n_w_churn, n_j_small,
         n_j_medium, n_w_out;
void **g_cmds[2];                // render commands, double-buffered
unsigned g_ncmds[2];
void ***g_outs[2];               // [worker] -> job outputs, double-buffered
unsigned *g_nouts[2];
std::uint64_t g_frame;           // written by the game thread between frames

// Streaming: an SPSC ring of loaded assets, streaming thread -> game thread.
enum : std::size_t { SRING = 1024, RESIDENT = 4096 };
struct Asset { void *p; std::size_t n; };
struct StreamRing {
	alignas(64) std::atomic<std::size_t> head{0};
	alignas(64) std::atomic<std::size_t> tail{0};
	alignas(64) Asset slot[SRING];
};
StreamRing g_sring;
// Pointer-width: a 64-bit atomic is a libatomic call on i486 (no CMPXCHG8B),
// which kept the whole test tree from linking there.
std::atomic<std::size_t> g_loads{0};
std::atomic<bool> g_stream_failed{false};

// RT (kamepoolalloc) entry points, if the preloaded allocator has them.
void (*g_rt_drain)() = nullptr;
const char *g_rt_state = "off";

void place_this_thread() {
#if defined(__APPLE__)
	pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

// ---------------------------------------------------------------- roles
void *streaming_main(void *) {
	place_this_thread();
	Rng r(g_seed * 7919 + 17);
	const double bytes_per_ns = g_stream_mibps * 1048576.0 * 1e-9;
	const std::uint64_t t0 = now_ns();
	double loaded = 0;
	while( !g_stop.load(std::memory_order_acquire)) {
		// Hold the average rate: wait until the bytes so far are due.
		const std::uint64_t due = t0 + (std::uint64_t)(loaded / bytes_per_ns);
		const std::uint64_t now = now_ns();
		if(now < due) {
			const std::uint64_t d = due - now;
			usleep((useconds_t)std::min<std::uint64_t>(d / 1000 + 1, 2000));
			continue;
		}
		const std::size_t n = r.asset();
		void *p = std::malloc(n);
		if( !p) { g_stream_failed.store(true); break; }
		touch(p, n, (std::uint64_t)n);
		std::size_t h = g_sring.head.load(std::memory_order_relaxed);
		while(h - g_sring.tail.load(std::memory_order_acquire) >= SRING) {
			if(g_stop.load(std::memory_order_acquire)) { std::free(p); return nullptr; }
			usleep(100);
		}
		g_sring.slot[h & (SRING - 1)] = Asset{p, n};
		g_sring.head.store(h + 1, std::memory_order_release);
		g_loads.fetch_add(1, std::memory_order_relaxed);
		loaded += (double)n;
	}
	return nullptr;
}

// Game thread's resident assets, oldest first.
Asset *g_resident;
std::size_t g_res_head, g_res_tail, g_res_bytes;

void frame_game(Ctx &c, void **table, void **temp) {
	const unsigned cur = (unsigned)(g_frame & 1), prev = cur ^ 1;
	for(unsigned i = 0; i < n_g_churn; ++i) {
		const std::size_t k = c.rng.below(G_TABLE);
		tfree(c, table[k]);
		const std::size_t n = c.rng.small();
		table[k] = tmalloc(c, n);
		if(table[k]) touch(table[k], n, k);
	}
	for(unsigned i = 0; i < n_g_temp; ++i) {
		const std::size_t n = c.rng.small();
		temp[i] = tmalloc(c, n);
		if(temp[i]) touch(temp[i], n, i);
	}
	compute(c, g_work * 4);
	unsigned nc = 0;
	for(unsigned i = 0; i < n_g_cmds; ++i) {
		const std::size_t n = c.rng.unit() < CMD_MEDIUM_P ? c.rng.medium() : c.rng.small();
		void *p = tmalloc(c, n);
		if( !p) continue;
		touch(p, n, i);
		g_cmds[cur][nc++] = p;
	}
	g_ncmds[cur] = nc;
	for(int w = 0; w < g_workers; ++w) {     // last frame's job outputs
		for(unsigned i = 0; i < g_nouts[prev][w]; ++i) tfree(c, g_outs[prev][w][i]);
		g_nouts[prev][w] = 0;
	}
	compute(c, g_work * 4);
	for(unsigned i = n_g_temp; i-- > 0;) if(temp[i]) tfree(c, temp[i]);
	// Streamed assets: take what has loaded, unload the oldest over budget.
	const double budget = g_stream_budget_mib * 1048576.0;
	std::size_t t = g_sring.tail.load(std::memory_order_relaxed);
	const std::size_t h = g_sring.head.load(std::memory_order_acquire);
	for(; t != h; ++t) {
		const Asset a = g_sring.slot[t & (SRING - 1)];
		if(g_res_head - g_res_tail >= RESIDENT) {   // cannot hold it: unload now
			tfree(c, a.p);
			continue;
		}
		g_resident[g_res_head++ % RESIDENT] = a;
		g_res_bytes += a.n;
	}
	g_sring.tail.store(t, std::memory_order_release);
	while((double)g_res_bytes > budget && g_res_tail != g_res_head) {
		const Asset a = g_resident[g_res_tail++ % RESIDENT];
		g_res_bytes -= a.n;
		tfree(c, a.p);
	}
}

void frame_render(Ctx &c, void **temp) {
	const unsigned prev = (unsigned)(g_frame & 1) ^ 1;
	for(unsigned i = 0; i < g_ncmds[prev]; ++i) tfree(c, g_cmds[prev][i]);
	g_ncmds[prev] = 0;
	for(unsigned i = 0; i < n_r_temp; ++i) {
		const std::size_t n = c.rng.small();
		temp[i] = tmalloc(c, n);
		if(temp[i]) touch(temp[i], n, i);
	}
	compute(c, g_work * 2);
	for(unsigned i = n_r_temp; i-- > 0;) if(temp[i]) tfree(c, temp[i]);
}

void frame_worker(int w, Ctx &c, void **table, void **temp) {
	const unsigned cur = (unsigned)(g_frame & 1);
	for(unsigned j = 0; j < W_JOBS; ++j) {
		unsigned nt = 0;
		for(unsigned i = 0; i < n_j_small; ++i) {
			const std::size_t n = c.rng.small();
			void *p = tmalloc(c, n);
			if(p) { touch(p, n, i); temp[nt++] = p; }
		}
		for(unsigned i = 0; i < n_j_medium; ++i) {
			const std::size_t n = c.rng.medium();
			void *p = tmalloc(c, n);
			if(p) { touch(p, n, i); temp[nt++] = p; }
		}
		if(c.rng.unit() < J_LARGE_P) {
			const std::size_t n = c.rng.large();
			void *p = tmalloc(c, n);
			if(p) { touch(p, n, j); temp[nt++] = p; }
		}
		compute(c, g_work);
		for(unsigned i = nt; i-- > 0;) tfree(c, temp[i]);
	}
	unsigned no = 0;
	for(unsigned i = 0; i < n_w_out; ++i) {
		const std::size_t n = c.rng.small();
		void *p = tmalloc(c, n);
		if(p) { touch(p, n, i); g_outs[cur][w][no++] = p; }
	}
	g_nouts[cur][w] = no;
	for(unsigned i = 0; i < n_w_churn; ++i) {
		const std::size_t k = c.rng.below(W_TABLE);
		tfree(c, table[k]);
		const std::size_t n = c.rng.small();
		table[k] = tmalloc(c, n);
		if(table[k]) touch(table[k], n, k);
	}
}

// Frame loop of one thread.  idx 0 is the game thread (the caller of
// main), 1 render, 2.. workers.
std::uint64_t *g_frame_ns;       // mmap'd, measured frames
std::size_t    g_nframes;
std::uint64_t  g_drain_max_ns, g_meas_t0, g_meas_t1;

void thread_loop(int idx) {
	place_this_thread();
	Ctx c{ &g_stats[idx], Rng(g_seed * 1000003 + (std::uint64_t)idx), false};
	std::size_t tsize = idx == 0 ? G_TABLE : idx == 1 ? 0 : W_TABLE;
	const std::size_t tempsize = idx == 0 ? n_g_temp
	                           : idx == 1 ? n_r_temp
	                           : n_j_small + n_j_medium + 1;
	void **table = tsize ? static_cast<void **>(map_zeroed(tsize * sizeof(void *))) : nullptr;
	void **temp = static_cast<void **>(map_zeroed((tempsize + 1) * sizeof(void *)));
	for(std::size_t k = 0; k < tsize; ++k) {      // the long-lived population
		const std::size_t n = c.rng.small();
		table[k] = std::malloc(n);
		if( !table[k]) { c.st->failed = true; tsize = k; break; }
		touch(table[k], n, k);
	}

	const std::uint64_t run_t0 = now_ns();
	bool measuring = false;
	std::size_t measured = 0;
	for(;;) {
		if(idx == 0) {
			// Between frames: decide what the next one is.
			if( !measuring && (double)(now_ns() - run_t0) * 1e-9 >= g_warmup) {
				measuring = true;
				g_record.store(true, std::memory_order_relaxed);
				g_meas_t0 = now_ns();
			}
			if(measuring && (measured >= g_frames
			                 || (double)(now_ns() - g_meas_t0) * 1e-9 >= g_seconds))
				g_stop.store(true, std::memory_order_relaxed);
		}
		g_start.wait();
		if(g_stop.load(std::memory_order_relaxed)) break;
		c.rec = g_record.load(std::memory_order_relaxed);
		if(idx == 0) frame_game(c, table, temp);
		else if(idx == 1) frame_render(c, temp);
		else frame_worker(idx - 2, c, table, temp);
		g_end.wait();
		if(idx == 0) {
			if(c.rec) {
				g_frame_ns[measured++] = g_end.release_ns - g_start.release_ns;
				g_meas_t1 = g_end.release_ns;
			}
			// Idle time: the vsync wait of a game.
			if(g_rt_drain) {
				const std::uint64_t a = now_ns();
				g_rt_drain();
				const std::uint64_t d = now_ns() - a;
				if(c.rec && d > g_drain_max_ns) g_drain_max_ns = d;
			}
			if(g_frame_ms > 0) {
				const std::uint64_t due = g_start.release_ns + (std::uint64_t)(g_frame_ms * 1e6);
				std::uint64_t now;
				while((now = now_ns()) < due) usleep((useconds_t)std::max<std::uint64_t>((due - now) / 1000, 1));
			}
			++g_frame;
		}
	}
	if(idx == 0) g_nframes = measured;
	for(std::size_t k = 0; k < tsize; ++k) std::free(table[k]);
}

void *thread_main(void *arg) {
	thread_loop((int)(std::intptr_t)arg);
	return nullptr;
}

// ---------------------------------------------------------------- report
std::uint64_t pct_frames(std::uint64_t *v, std::size_t n, double p) {
	if( !n) return 0;
	std::size_t k = (std::size_t)((double)n * p + 0.999999);
	if(k == 0) k = 1;
	if(k > n) k = n;
	return v[k - 1];
}

void usage(const char *prog) {
	std::fprintf(stderr,
	    "%s [--workers N] [--frames N] [--seconds S] [--warmup S] [--scale F]\n"
	    "    [--work N] [--stream-mibps R] [--stream-budget-mib B] [--frame-ms P]\n"
	    "    [--rt] [--seed N] [-v]\n", prog);
	std::exit(2);
}

} // namespace

int main(int argc, char **argv) {
	for(int i = 1; i < argc; ++i) {
		const char *a = argv[i];
		const bool more = i + 1 < argc;
		if( !std::strcmp(a, "--workers") && more) g_workers = std::atoi(argv[++i]);
		else if( !std::strcmp(a, "--frames") && more) g_frames = std::strtoull(argv[++i], nullptr, 10);
		else if( !std::strcmp(a, "--seconds") && more) g_seconds = std::atof(argv[++i]);
		else if( !std::strcmp(a, "--warmup") && more) g_warmup = std::atof(argv[++i]);
		else if( !std::strcmp(a, "--scale") && more) g_scale = std::atof(argv[++i]);
		else if( !std::strcmp(a, "--work") && more) g_work = (unsigned)std::strtoul(argv[++i], nullptr, 10);
		else if( !std::strcmp(a, "--stream-mibps") && more) g_stream_mibps = std::atof(argv[++i]);
		else if( !std::strcmp(a, "--stream-budget-mib") && more) g_stream_budget_mib = std::atof(argv[++i]);
		else if( !std::strcmp(a, "--frame-ms") && more) g_frame_ms = std::atof(argv[++i]);
		else if( !std::strcmp(a, "--rt")) g_rt = true;
		else if( !std::strcmp(a, "-v")) g_verbose = true;
		else if( !std::strcmp(a, "--kame-cache-mib") && more) g_kame_cache_mib = std::atof(argv[++i]);
		else if( !std::strcmp(a, "--seed") && more) g_seed = std::strtoull(argv[++i], nullptr, 10);
		else usage(argv[0]);
	}
	if(g_workers < 0 || g_workers + 2 > MAX_THREADS || g_frames == 0
	   || g_scale <= 0 || g_seconds <= 0)
		usage(argv[0]);

	auto scaled = [](unsigned n) { return std::max(1u, (unsigned)std::lround(n * g_scale)); };
	n_g_churn = scaled(G_CHURN); n_g_temp = scaled(G_TEMP); n_g_cmds = scaled(G_CMDS);
	n_r_temp = scaled(R_TEMP); n_w_churn = scaled(W_CHURN);
	n_j_small = scaled(J_SMALL); n_j_medium = scaled(J_MEDIUM); n_w_out = scaled(W_OUT);

	g_threads = 2 + g_workers;
	g_start.n = g_end.n = (unsigned)g_threads;
	g_stats = static_cast<ThreadStats *>(map_zeroed(sizeof(ThreadStats) * (std::size_t)g_threads));
	g_frame_ns = static_cast<std::uint64_t *>(map_zeroed(sizeof(std::uint64_t) * g_frames));
	for(int b = 0; b < 2; ++b) {
		g_cmds[b] = static_cast<void **>(map_zeroed(sizeof(void *) * n_g_cmds));
		g_outs[b] = static_cast<void ***>(map_zeroed(sizeof(void **) * (std::size_t)(g_workers + 1)));
		g_nouts[b] = static_cast<unsigned *>(map_zeroed(sizeof(unsigned) * (std::size_t)(g_workers + 1)));
		for(int w = 0; w < g_workers; ++w)
			g_outs[b][w] = static_cast<void **>(map_zeroed(sizeof(void *) * n_w_out));
	}
	g_resident = static_cast<Asset *>(map_zeroed(sizeof(Asset) * RESIDENT));

	calibrate_ticks();
	if(g_kame_cache_mib >= 0) {
		auto cap = reinterpret_cast<void (*)(std::size_t)>(
		    dlsym(RTLD_DEFAULT, "kame_pool_set_large_cache_cap"));
		if(cap) cap((std::size_t)(g_kame_cache_mib * 1048576.0));
	}
	if(g_rt) {
		auto mode = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "kame_pool_set_realtime_mode"));
		auto dflt = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "kame_pool_set_realtime_default"));
		auto drain = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "kame_pool_rt_drain"));
		if(mode && dflt && drain) {
			mode(1);
			dflt(1);                 // KAME_RT_DEFER for every thread
			g_rt_drain = drain;
			g_rt_state = "defer";
		}
		else g_rt_state = "na";
	}

	pthread_t streamer{};
	const bool streaming = g_stream_mibps > 0;
	if(streaming) pthread_create( &streamer, nullptr, streaming_main, nullptr);
	pthread_t th[MAX_THREADS];
	for(int i = 1; i < g_threads; ++i)
		pthread_create( &th[i], nullptr, thread_main, reinterpret_cast<void *>((std::intptr_t)i));
	thread_loop(0);
	for(int i = 1; i < g_threads; ++i) pthread_join(th[i], nullptr);
	if(streaming) pthread_join(streamer, nullptr);

	// Leftovers (untimed): commands, outputs, assets.
	for(int b = 0; b < 2; ++b) {
		for(unsigned i = 0; i < g_ncmds[b]; ++i) std::free(g_cmds[b][i]);
		for(int w = 0; w < g_workers; ++w)
			for(unsigned i = 0; i < g_nouts[b][w]; ++i) std::free(g_outs[b][w][i]);
	}
	while(g_res_tail != g_res_head) std::free(g_resident[g_res_tail++ % RESIDENT].p);
	for(std::size_t t = g_sring.tail.load(); t != g_sring.head.load(); ++t)
		std::free(g_sring.slot[t & (SRING - 1)].p);

	// Merge the call statistics.
	static std::uint64_t hist[HB];
	std::uint64_t calls = 0, slow = 0, max_ns = 0;
	bool failed = g_stream_failed.load();
	for(int i = 0; i < g_threads; ++i) {
		const ThreadStats &s = g_stats[i];
		for(int b = 0; b < HB; ++b) hist[b] += s.hist[b];
		calls += s.calls; slow += s.slow;
		max_ns = std::max(max_ns, s.max_ns);
		failed = failed || s.failed;
	}
	if(g_verbose) {
		const double span = (double)(g_meas_t1 - g_meas_t0);
		for(int i = 0; i < g_threads; ++i) {
			const ThreadStats &s = g_stats[i];
			std::fprintf(stderr, "  thread %d (%s): calls=%llu in-calls=%.1f%% of the run, "
			             "mean %.0f ns, max %llu ns\n", i,
			             i == 0 ? "game" : i == 1 ? "render" : "worker",
			             (unsigned long long)s.calls,
			             span > 0 ? 100.0 * (double)s.sum_ns * g_ns_per_tick / span : 0.0,
			             s.calls ? (double)s.sum_ns * g_ns_per_tick / (double)s.calls : 0.0,
			             (unsigned long long)((double)s.max_ns * g_ns_per_tick));
		}
	}
	auto hpct = [&](double p) -> std::uint64_t {
		if( !calls) return 0;
		const std::uint64_t want = (std::uint64_t)std::ceil((double)calls * p);
		std::uint64_t seen = 0;
		for(int b = 0; b < HB; ++b) {
			seen += hist[b];
			if(seen >= want)
				return (std::uint64_t)((double)std::min(hupper(b), max_ns) * g_ns_per_tick);
		}
		return (std::uint64_t)((double)max_ns * g_ns_per_tick);
	};

	const std::size_t n = g_nframes;
	std::sort(g_frame_ns, g_frame_ns + n);
	const std::uint64_t med = pct_frames(g_frame_ns, n, 0.5);
	std::size_t hitch = 0;
	for(std::size_t i = 0; i < n; ++i) if(g_frame_ns[i] > 2 * med) ++hitch;
	const double secs = n ? (double)(g_meas_t1 - g_meas_t0) * 1e-9 : 0.0;

	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
	const double rss_mib = (double)ru.ru_maxrss / 1048576.0;      // bytes
#else
	const double rss_mib = (double)ru.ru_maxrss / 1024.0;         // KiB
#endif

	// kamepoolalloc's view, when it is the allocator: what its recycle
	// cache holds (resident, reusable) at the end.
	char kame_info[96] = "";
	if(auto gs = reinterpret_cast<void (*)(kame_pool_stats_t *)>(
	       dlsym(RTLD_DEFAULT, "kame_pool_get_stats"))) {
		kame_pool_stats_t ks{};
		ks.version = KAME_POOL_STATS_VERSION;
		gs( &ks);
		if(ks.version_supported >= 2)
			std::snprintf(kame_info, sizeof(kame_info), " kame_cache_mib=%.0f",
			              (double)ks.cache_bytes / 1048576.0);
	}
	std::printf("[bench_frame] threads=%d workers=%d rt=%s frames=%zu secs=%.2f "
	            "fps=%.0f frame_med_us=%.1f frame_p99_us=%.1f frame_p999_us=%.1f "
	            "frame_max_us=%.1f hitch2x=%zu call_med_ns=%llu call_p999_ns=%llu "
	            "call_p9999_ns=%llu call_max_ns=%llu slow_calls_count=%llu "
	            "calls_M=%.1f drain_max_us=%.1f stream_loads=%llu rss_mib=%.0f%s "
	            "status=%s\n",
	            g_threads, g_workers, g_rt_state, n, secs, secs > 0 ? (double)n / secs : 0.0,
	            med * 1e-3, pct_frames(g_frame_ns, n, 0.99) * 1e-3,
	            pct_frames(g_frame_ns, n, 0.999) * 1e-3, n ? g_frame_ns[n - 1] * 1e-3 : 0.0,
	            hitch, (unsigned long long)hpct(0.5), (unsigned long long)hpct(0.999),
	            (unsigned long long)hpct(0.9999),
	            (unsigned long long)((double)max_ns * g_ns_per_tick),
	            (unsigned long long)slow, (double)calls * 1e-6, g_drain_max_ns * 1e-3,
	            (unsigned long long)g_loads.load(), rss_mib, kame_info,
	            failed ? "malloc_failed" : "ok");
	return failed ? 4 : 0;
}
