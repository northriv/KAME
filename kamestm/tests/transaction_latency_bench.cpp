/*
 * transaction_latency_bench — per-commit LATENCY tail for the STM.
 *
 * Every other STM bench reports throughput (commits/s).  That is the statistic
 * that hides a stall: a run that sleeps 1 ms once per thousand commits looks
 * fine in commits/s and misses a 1 kHz deadline every time.  This one times
 * each `iterate_commit` individually and reports max and percentiles.
 *
 * PURE OBSERVATION — deliberately.  It measures the STM entirely from outside:
 * timestamps around `iterate_commit`, nothing added to kamestm, no realtime
 * mode, no privileged priority, no instrumentation build.  So the numbers are
 * the numbers a normal (Priority::NORMAL) caller sees today, and adding this
 * file cannot regress anything.
 *
 * Workload shapes mirror transaction_payload_integrity_3level_mixed_test, so
 * the latency figures are directly comparable to its throughput figures:
 *
 *   leaf   — each thread commits on its OWN leaf (least contended)
 *   grand  — every thread commits at the root, touching all children:
 *            the 3-level bundle, 1 + 2(N+1) CAS
 *   mixed  — `-x PERCENT` of commits go grand, the rest leaf.  This is the
 *            known-expensive shape: the penalty is bundle CHURN plus
 *            negotiation waiting, not a retry storm.
 *
 * What to look for, and why this runs BEFORE any realtime work:
 *
 *   1. Is the tail quantised to the negotiation sleep chunk?  A contender
 *      that loses the spin band waits on a condition variable for
 *      KAME_NEG_SLEEP_US_PER_MS microseconds (compile-time, default 1000 =
 *      1 ms).  If the tail sits at multiples of that, the sleep dominates
 *      and shrinking or removing it is the whole game.  Rebuild with
 *      -DKAME_NEG_SLEEP_US_PER_MS=250 etc. and compare -- the define is
 *      compile-time, so the sweep costs builds, not runtime branches.
 *   2. How often does a NORMAL commit reach the sleep at all?  Inferred here
 *      without instrumentation, as the fraction of commits at or above one
 *      chunk (`>= sleep-chunk` in the output).  An instrumented build would
 *      answer it exactly but is not throughput-neutral, so it must not be
 *      mixed into the same run.
 *   3. How much of the tail is bundle work rather than waiting?  Compare the
 *      grand arm at 1 thread (no contention: pure bundle cost) against the
 *      contended arms.
 *
 * Usage:
 *   transaction_latency_bench [-t THREADS] [-s SECONDS] [-x GRAND_PERCENT]
 *                             [-w WARMUP_SEC] [-m MODE]
 *     MODE = leaf | grand | mixed | all   (default all)
 *
 * Licensed under GPL-2.0-or-later, as the rest of kamestm/tests.
 */

#include "support_standalone.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "transaction.h"
#include "transaction_impl.h"

//! Same node shape as the 3-level mixed integrity test, so the latency
//! figures here line up with that test's throughput figures.
class MyNode : public Transactional::Node<MyNode> {
public:
    struct Payload : public Transactional::Node<MyNode>::Payload {
        unsigned int m_x = 0;
    };
};
typedef Transactional::Transaction<MyNode> Tr;

// ------------------------------------------------------------------ clock
static inline std::uint64_t now_ns() {
    return (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// -------------------------------------------------------------- histogram
// 1 ns resolution below 64 ns, then 4 buckets per octave: O(1) memory and no
// allocation, so the harness cannot perturb the allocator under the STM.
enum { HB = 256 };
//! Retry accounting for the slow tail.  `iterate_commit` invokes its lambda
//! once per ATTEMPT, so the retry count is observable from outside with no
//! change to kamestm — worth exhausting before instrumenting the library.
#if KAME_STM_NEG_DIAG
//! Negotiation breakdown for SLOW commits only (built with
//! -DKAME_STM_NEG_DIAG=1).  Answers where a slow commit's time went: waiting
//! in cell.wait(), or looping without sleeping — and whether age promotion was
//! ever attempted, let alone granted.  Not throughput-neutral: run separately
//! from the latency tables.
struct NegDiagAcc {
    std::uint64_t n = 0, rounds = 0, sleeps = 0, slept_ns = 0;
    std::uint64_t tries = 0, grants = 0, max_rounds = 0, max_slept = 0;
    std::uint64_t sl_hold = 0, tags_held = 0, sl_priv = 0, tag_list = 0, req = 0;
    void add(const Transactional::detail::NegDiag &d) {
        n++; rounds += d.rounds; sleeps += d.sleeps; slept_ns += d.slept_ns;
        tries += d.priv_tries; grants += d.priv_grants;
        sl_hold += d.sleeps_holding; tags_held += d.tags_held_at_sleep;
        sl_priv += d.sleeps_priv; tag_list += d.tagged_list_at_sleep;
        req += d.req_ns;
        if(d.rounds > max_rounds) max_rounds = d.rounds;
        if(d.slept_ns > max_slept) max_slept = d.slept_ns;
    }
    void merge(const NegDiagAcc &o) {
        n += o.n; rounds += o.rounds; sleeps += o.sleeps; slept_ns += o.slept_ns;
        tries += o.tries; grants += o.grants;
        sl_hold += o.sl_hold; tags_held += o.tags_held; sl_priv += o.sl_priv;
        tag_list += o.tag_list; req += o.req;
        if(o.max_rounds > max_rounds) max_rounds = o.max_rounds;
        if(o.max_slept > max_slept) max_slept = o.max_slept;
    }
};
#endif

struct Retries {
    std::uint64_t slow_n = 0, slow_attempts = 0, slow_max = 0;   // >= threshold
    std::uint64_t all_n = 0, all_attempts = 0;
    void add(std::uint64_t ns, std::uint64_t att, std::uint64_t thresh) {
        all_n++; all_attempts += att;
        if(ns >= thresh) {
            slow_n++; slow_attempts += att;
            if(att > slow_max) slow_max = att;
        }
    }
    void merge(const Retries &o) {
        slow_n += o.slow_n; slow_attempts += o.slow_attempts;
        if(o.slow_max > slow_max) slow_max = o.slow_max;
        all_n += o.all_n; all_attempts += o.all_attempts;
    }
};

struct Hist {
    std::uint64_t bucket[HB];
    std::uint64_t n, max, sum;
    void reset() { std::memset(this, 0, sizeof(*this)); }
    static unsigned idx(std::uint64_t v) {
        if(v < 64) return (unsigned)v;
        unsigned oct = 63u - (unsigned)__builtin_clzll(v);
        unsigned i = 64u + (oct - 6u) * 4u + (unsigned)((v >> (oct - 2)) & 3u);
        return i < (unsigned)HB ? i : (unsigned)HB - 1u;
    }
    static std::uint64_t value(unsigned i) {
        if(i < 64) return i;
        unsigned oct = 6u + (i - 64u) / 4u, frac = (i - 64u) % 4u;
        return (std::uint64_t)(4u + frac + 1u) << (oct - 2);
    }
    void add(std::uint64_t v) {
        bucket[idx(v)]++; n++; sum += v; if(v > max) max = v;
    }
    void merge(const Hist &o) {
        for(unsigned i = 0; i < HB; i++) bucket[i] += o.bucket[i];
        n += o.n; sum += o.sum; if(o.max > max) max = o.max;
    }
    std::uint64_t pct(double p) const {
        if( !n) return 0;
        std::uint64_t want = (std::uint64_t)(p * (double)n), acc = 0;
        if(want >= n) want = n - 1;
        for(unsigned i = 0; i < HB; i++)
            if((acc += bucket[i]) > want) return value(i);
        return max;
    }
    //! A percentile is only meaningful with >= 10 samples beyond it.
    bool supports(double p) const { return (double)n * (1.0 - p) >= 10.0; }
    //! Commits at or above `v` ns — used for the "reached the sleep" estimate.
    std::uint64_t at_or_above(std::uint64_t v) const {
        std::uint64_t acc = 0;
        for(unsigned i = 0; i < HB; i++) if(value(i) >= v) acc += bucket[i];
        return acc;
    }
};

static const std::uint64_t kSleepChunkNs =
    (std::uint64_t)KAME_NEG_SLEEP_US_PER_MS * 1000ull;   // one CV chunk

static void report(const char *label, const Hist &h, double secs) {
    static const double kP[] = { 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999 };
    static const char  *kN[] = { "p50","p90","p99","p99.9","p99.99","p99.999" };
    std::printf("  %-22s n=%-9llu  %6.2f Mcommit/s  mean=%-7llu",
                label, (unsigned long long)h.n,
                secs > 0 ? h.n / secs / 1e6 : 0.0,
                (unsigned long long)(h.n ? h.sum / h.n : 0));
    for(unsigned i = 0; i < sizeof(kP)/sizeof(kP[0]); i++)
        if(h.supports(kP[i]))
            std::printf(" %s=%llu", kN[i], (unsigned long long)h.pct(kP[i]));
    std::printf("  MAX=%llu ns\n", (unsigned long long)h.max);

    // Estimate of how many commits plausibly waited on the negotiation CV.
    // Inferred from the distribution rather than instrumented, because the
    // instrumented build is not throughput-neutral and must not share a run.
    std::uint64_t slow = h.at_or_above(kSleepChunkNs);
    if(h.n)
        std::printf("  %-22s >= 1 sleep chunk (%llu ns): %llu / %llu = %.4f %%\n",
                    "", (unsigned long long)kSleepChunkNs,
                    (unsigned long long)slow, (unsigned long long)h.n,
                    100.0 * (double)slow / (double)h.n);
}

// ------------------------------------------------------------------ run
enum Mode { M_LEAF = 0, M_GRAND = 1, M_MIXED = 2 };

//! A commit at or above this is "slow" for the retry breakdown.  100 us is
//! comfortably past p99.9 in every arm measured so far, and well under the
//! millisecond-scale tail being diagnosed.
static const std::uint64_t kSlowNs = 100000ull;

static Hist run_arm(Mode mode, int threads, double secs, double warmup,
                    int grand_pct, double *out_secs, Retries *out_r
#if KAME_STM_NEG_DIAG
                    , NegDiagAcc *out_d
#endif
                    ) {
    shared_ptr<MyNode> grand(MyNode::create<MyNode>());
    shared_ptr<MyNode> parent(MyNode::create<MyNode>());
    std::vector<shared_ptr<MyNode>> children((size_t)threads);
    for(int i = 0; i < threads; i++)
        children[(size_t)i].reset(MyNode::create<MyNode>());

    grand->iterate_commit([&](Tr &tr) {
        if( !grand->insert(tr, parent, true)) return;
        for(int i = 0; i < threads; i++)
            if( !parent->insert(tr, children[(size_t)i], true)) return;
    });

    std::vector<Hist> per_thread((size_t)threads);
    for(auto &h : per_thread) h.reset();
    std::vector<Retries> per_thread_r((size_t)threads);
#if KAME_STM_NEG_DIAG
    std::vector<NegDiagAcc> per_thread_d((size_t)threads);
#endif
    std::atomic<int> ready{0};
    std::atomic<bool> go{false}, stop{false}, timing{false};

    auto worker = [&](int tid) {
        Hist local; local.reset();
        Retries lr;
#if KAME_STM_NEG_DIAG
        NegDiagAcc ld;
#endif
        shared_ptr<MyNode> leaf = children[(size_t)tid];
        unsigned seq = 0;
        ready.fetch_add(1);
        while( !go.load(std::memory_order_acquire)) { }
        while( !stop.load(std::memory_order_relaxed)) {
            // Decide the scope BEFORE timing so the choice is not measured.
            bool do_grand = (mode == M_GRAND) ||
                (mode == M_MIXED && (int)(seq++ % 100u) < grand_pct);
            const bool measure = timing.load(std::memory_order_relaxed);
            std::uint64_t attempts = 0;
#if KAME_STM_NEG_DIAG
            // Zero the per-thread counters so what we read back belongs to
            // THIS commit alone.
            if(measure) (void)Transactional::neg_diag_snapshot(true);
#endif
            std::uint64_t t0 = measure ? now_ns() : 0;
            if(do_grand)
                grand->iterate_commit([&](Tr &tr) {
                    ++attempts;
                    for(int c = 0; c < threads; c++)
                        tr[*children[(size_t)c]].m_x++;
                });
            else
                leaf->iterate_commit([&](Tr &tr) { ++attempts; tr[*leaf].m_x++; });
            if(measure) {
                std::uint64_t dt = now_ns() - t0;
                local.add(dt);
                lr.add(dt, attempts, kSlowNs);
#if KAME_STM_NEG_DIAG
                auto d = Transactional::neg_diag_snapshot(false);
                if(dt >= kSlowNs) ld.add(d);
#endif
            }
        }
        per_thread[(size_t)tid] = local;
        per_thread_r[(size_t)tid] = lr;
#if KAME_STM_NEG_DIAG
        per_thread_d[(size_t)tid] = ld;
#endif
    };

    std::vector<std::thread> ts;
    ts.reserve((size_t)threads);
    for(int i = 0; i < threads; i++) ts.emplace_back(worker, i);
    while(ready.load() < threads) { }
    go.store(true, std::memory_order_release);

    // Warm up untimed: first-touch faults, chunk claims and the adaptive
    // negotiation state settling are startup effects, not steady-state tail.
    std::this_thread::sleep_for(std::chrono::duration<double>(warmup));
    timing.store(true, std::memory_order_relaxed);
    std::uint64_t t0 = now_ns();
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    timing.store(false, std::memory_order_relaxed);
    double elapsed = (double)(now_ns() - t0) / 1e9;
    stop.store(true, std::memory_order_relaxed);
    for(auto &t : ts) t.join();

    Hist all; all.reset();
    for(auto &h : per_thread) all.merge(h);
    if(out_r) { out_r->merge(Retries()); for(auto &r : per_thread_r) out_r->merge(r); }
#if KAME_STM_NEG_DIAG
    if(out_d) for(auto &d : per_thread_d) out_d->merge(*&d);
#endif
    if(out_secs) *out_secs = elapsed;
    return all;
}

int main(int argc, char **argv) {
    int threads = 4, grand_pct = 10;
    double secs = 2.0, warmup = 0.5;
    const char *mode = "all";
    for(int i = 1; i < argc; i++) {
        if( !std::strcmp(argv[i], "-t") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if( !std::strcmp(argv[i], "-s") && i + 1 < argc) secs = std::atof(argv[++i]);
        else if( !std::strcmp(argv[i], "-w") && i + 1 < argc) warmup = std::atof(argv[++i]);
        else if( !std::strcmp(argv[i], "-x") && i + 1 < argc) grand_pct = std::atoi(argv[++i]);
        else if( !std::strcmp(argv[i], "-m") && i + 1 < argc) mode = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [-t THREADS] [-s SEC] [-w WARMUP] [-x GRAND%%] "
                "[-m leaf|grand|mixed|all]\n", argv[0]);
            return 2;
        }
    }
    if(threads < 1) threads = 1;

    std::printf("== transaction_latency_bench ==\n");
    std::printf("threads=%d  timed=%.1fs (warmup %.1fs)  grand%%=%d  "
                "sleep chunk=%llu ns (KAME_NEG_SLEEP_US_PER_MS=%u)\n",
                threads, secs, warmup, grand_pct,
                (unsigned long long)kSleepChunkNs,
                (unsigned)KAME_NEG_SLEEP_US_PER_MS);

    // Clock floor: every sample includes one now_ns() pair, so figures near
    // this value are quantisation, not measurement.  Printed, never subtracted
    // -- subtracting would let a difference silently clamp at zero.
    {
        Hist ov; ov.reset();
        for(int i = 0; i < 20000; i++) { std::uint64_t a = now_ns(); ov.add(now_ns() - a); }
        std::printf("clock floor: mean=%llu max=%llu ns\n",
                    (unsigned long long)(ov.sum / (ov.n ? ov.n : 1)),
                    (unsigned long long)ov.max);
    }

    const bool all = !std::strcmp(mode, "all");
    struct { const char *name; Mode m; } arms[] = {
        { "leaf  (own node)",  M_LEAF  },
        { "grand (3-lvl bundle)", M_GRAND },
        { "mixed (grand+leaf)", M_MIXED },
    };
    for(auto &a : arms) {
        if( !all && std::strncmp(mode, a.name, std::strlen(mode))) continue;
        double el = 0;
        Retries r;
#if KAME_STM_NEG_DIAG
        NegDiagAcc dg;
        Hist h = run_arm(a.m, threads, secs, warmup, grand_pct, &el, &r, &dg);
#else
        Hist h = run_arm(a.m, threads, secs, warmup, grand_pct, &el, &r);
#endif
        report(a.name, h, el);
        // If slow commits show ~1 attempt, the time is spent INSIDE one
        // attempt (negotiation: spinning or sleeping), not in retrying —
        // which is what tells us where to instrument next.
        std::printf("  %-22s attempts/commit: all=%.3f   slow(>=%llu ns): "
                    "n=%llu mean=%.3f max=%llu\n", "",
                    r.all_n ? (double)r.all_attempts / (double)r.all_n : 0.0,
                    (unsigned long long)kSlowNs, (unsigned long long)r.slow_n,
                    r.slow_n ? (double)r.slow_attempts / (double)r.slow_n : 0.0,
                    (unsigned long long)r.slow_max);
#if KAME_STM_NEG_DIAG
        if(dg.n)
            std::printf("  %-22s SLOW n=%llu | rounds/commit %.2f (max %llu)"
                        " | sleeps/commit %.2f | slept/commit %.0f ns"
                        " (max %llu) | priv tries %.3f grants %.3f\n",
                        "", (unsigned long long)dg.n,
                        (double)dg.rounds  / (double)dg.n,
                        (unsigned long long)dg.max_rounds,
                        (double)dg.sleeps  / (double)dg.n,
                        (double)dg.slept_ns / (double)dg.n,
                        (unsigned long long)dg.max_slept,
                        (double)dg.tries   / (double)dg.n,
                        (double)dg.grants  / (double)dg.n);
        if(dg.n && dg.sleeps)
            std::printf("  %-22s HOLD-AND-WAIT: %.1f%% of sleeps happen while "
                        "still owning >=1 tag (%.2f tags avg); %.1f%% while "
                        "holding privilege\n", "",
                        100.0 * (double)dg.sl_hold / (double)dg.sleeps,
                        dg.sl_hold ? (double)dg.tags_held / (double)dg.sl_hold : 0.0,
                        100.0 * (double)dg.sl_priv / (double)dg.sleeps),
            std::printf("  %-22s   (tagged-list size at sleep: %.2f — 0 means "
                        "nothing was tagged yet, so 'owns none' would be vacuous)\n",
                        "", (double)dg.tag_list / (double)dg.sleeps),
            std::printf("  %-22s   REQUESTED vs ACTUAL sleep: asked %.0f ns/sleep, "
                        "got %.0f ns/sleep  (ratio %.2fx — ~1x means the STM "
                        "chose it, >>1x means the OS did not run us)\n", "",
                        (double)dg.req / (double)dg.sleeps,
                        (double)dg.slept_ns / (double)dg.sleeps,
                        dg.req ? (double)dg.slept_ns / (double)dg.req : 0.0);
#endif
    }
    std::printf("== done ==\n");
    return 0;
}
