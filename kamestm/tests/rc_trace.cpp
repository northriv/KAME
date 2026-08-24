// rc_trace.cpp — runtime for the KAME_RC_TRACE hooks in
// kamepoolalloc/atomic_smart_ptr.h.  Build it into the reproducer:
//
//   c++ ... -DKAME_RC_TRACE kamestm/tests/rc_trace.cpp ...
//
// Design constraints:
//   * called on every local_shared_ptr inc/dec of every atomic_countable
//     object — must not allocate, must add as little contention as possible
//     (per-thread rings; the only shared write is the TSC read on x86-64,
//     which is contention-free).  The bug is timing-sensitive (40-65%
//     per run); if the fire rate collapses under tracing, raise the
//     thread count instead of the ring size.
//   * dump paths (tripwire `die`, gdb-called `kame_rc_dump`) run in a
//     stopped/aborting process — fprintf/dladdr are fine there, but the
//     record path uses nothing beyond TLS and a fixed static array.
//
// Thread-slot recycling: the reproducer creates and joins threads by the
// hundreds, so slots are assigned round-robin modulo MAX_RINGS.  A recycled
// ring overwrites a dead thread's events eventually; the window that
// matters (the fatal inc/dec history of one Packet) is far shorter than a
// ring.  Events carry the real tid, so recycling never misattributes.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
static unsigned rc_trace_tid_() noexcept {
    return (unsigned)syscall(SYS_gettid);
}
#else
static std::atomic<unsigned> s_tid_gen{1};
static unsigned rc_trace_tid_() noexcept {
    static thread_local unsigned tid = s_tid_gen.fetch_add(1, std::memory_order_relaxed);
    return tid;
}
#endif

#if defined(__x86_64__)
#include <x86intrin.h>
static inline unsigned long long rc_trace_seq_() noexcept { return __rdtsc(); }
#else
static std::atomic<unsigned long long> s_seq{1};   // 1-based: dump uses seq==0 as "empty slot"
static inline unsigned long long rc_trace_seq_() noexcept {
    return s_seq.fetch_add(1, std::memory_order_relaxed);
}
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#define RC_TRACE_HAVE_DLADDR 1
#endif

namespace kame_rc_trace {

enum Op : unsigned {  // keep in sync with atomic_smart_ptr.h
    OP_BORN = 1, OP_INC, OP_DEC, OP_DEAD, OP_DEAD_UNIQUE,
    OP_INC_FROM_ZERO, OP_DEC_UNDERFLOW,
};
static const char *op_name_(unsigned op) noexcept {
    switch(op) {
    case OP_BORN: return "BORN";
    case OP_INC: return "INC ";
    case OP_DEC: return "DEC ";
    case OP_DEAD: return "DEAD";
    case OP_DEAD_UNIQUE: return "DEAD(unique)";
    case OP_INC_FROM_ZERO: return "INC-FROM-ZERO **TRIPWIRE**";
    case OP_DEC_UNDERFLOW: return "DEC-UNDERFLOW **TRIPWIRE**";
    default: return "????";
    }
}

struct Ev {
    const void *obj;
    const void *site;
    unsigned long long seq;
    unsigned long long oldc;
    unsigned op;
    unsigned tid;
};

static constexpr unsigned MAX_RINGS = 64;
static constexpr unsigned RING_LOG2 = 14;               // 16384 events/ring
static constexpr unsigned RING = 1u << RING_LOG2;
static constexpr unsigned RING_MASK = RING - 1;
static Ev g_rings[MAX_RINGS][RING];                      // ~50 MB BSS
static std::atomic<unsigned> g_ring_next{0};

struct TL {
    Ev *ring = nullptr;
    unsigned idx = 0;
    unsigned tid = 0;
};
static thread_local TL tl;

void record(const void *obj, unsigned op, unsigned long long oldc,
    const void *site) noexcept {
    TL &t = tl;
    if(!t.ring) {
        t.ring = g_rings[g_ring_next.fetch_add(1, std::memory_order_relaxed)
                         % MAX_RINGS];
        t.tid = rc_trace_tid_();
    }
    Ev &e = t.ring[t.idx++ & RING_MASK];
    e.obj = obj; e.site = site; e.seq = rc_trace_seq_();
    e.oldc = oldc; e.op = op; e.tid = t.tid;
}

static void print_site_(const void *site) {
#ifdef RC_TRACE_HAVE_DLADDR
    Dl_info info;
    if(site && dladdr(const_cast<void *>(site), &info) && info.dli_sname) {
        fprintf(stderr, "%p (%s+0x%lx)", site, info.dli_sname,
            (unsigned long)((const char *)site - (const char *)info.dli_saddr));
        return;
    }
#endif
    fprintf(stderr, "%p", site);
}

} // namespace kame_rc_trace

//! Dump the recorded refcount history of one object, oldest first.
//! From gdb at the poison fault:  call kame_rc_dump((const void*)$rsi)
//! Also called by the tripwires before abort().
extern "C" void kame_rc_dump(const void *obj) {
    using namespace kame_rc_trace;
    static constexpr unsigned MAXOUT = 4096;
    static Ev out[MAXOUT];           // static: dump runs once, no stack blowup
    unsigned n = 0;
    for(unsigned r = 0; r < MAX_RINGS; ++r)
        for(unsigned i = 0; i < RING; ++i) {
            const Ev &e = g_rings[r][i];
            if(e.obj == obj && e.seq && n < MAXOUT)
                out[n++] = e;
        }
    // insertion sort by seq (n is small; dump-time only)
    for(unsigned i = 1; i < n; ++i) {
        Ev key = out[i];
        unsigned j = i;
        for(; j > 0 && out[j-1].seq > key.seq; --j) out[j] = out[j-1];
        out[j] = key;
    }
    fprintf(stderr, "\n==== kame_rc_trace: %u event(s) for obj %p "
        "(oldest first; seq is TSC) ====\n", n, obj);
    for(unsigned i = 0; i < n; ++i) {
        const Ev &e = out[i];
        fprintf(stderr, "  seq=%llu%+lld tid=%u  %-26s rc %llu -> %llu  site=",
            e.seq, i ? (long long)(e.seq - out[0].seq) : 0LL, e.tid,
            op_name_(e.op), e.oldc,
            (e.op == OP_INC || e.op == OP_INC_FROM_ZERO || e.op == OP_BORN)
                ? e.oldc + (e.op == OP_BORN ? 0 : 1)
                : (e.oldc ? e.oldc - 1 : 0));
        print_site_(e.site);
        fprintf(stderr, "\n");
    }
    if(n == MAXOUT)
        fprintf(stderr, "  (output truncated at %u)\n", MAXOUT);
    fprintf(stderr, "==== end of history for %p ====\n\n", obj);
    fflush(stderr);
}

//! Dump the most recent \a nreq events across all threads — context around
//! a fault.  From gdb:  call kame_rc_dump_recent(200)
extern "C" void kame_rc_dump_recent(unsigned nreq) {
    using namespace kame_rc_trace;
    static constexpr unsigned MAXOUT = 4096;
    static Ev out[MAXOUT];
    if(nreq > MAXOUT) nreq = MAXOUT;
    unsigned n = 0;
    for(unsigned r = 0; r < MAX_RINGS; ++r)
        for(unsigned i = 0; i < RING; ++i) {
            const Ev &e = g_rings[r][i];
            if(!e.seq) continue;
            if(n < nreq) {
                out[n++] = e;
            } else {
                unsigned lo = 0;      // replace the oldest kept
                for(unsigned k = 1; k < n; ++k)
                    if(out[k].seq < out[lo].seq) lo = k;
                if(e.seq > out[lo].seq) out[lo] = e;
            }
        }
    for(unsigned i = 1; i < n; ++i) {
        Ev key = out[i];
        unsigned j = i;
        for(; j > 0 && out[j-1].seq > key.seq; --j) out[j] = out[j-1];
        out[j] = key;
    }
    fprintf(stderr, "\n==== kame_rc_trace: most recent %u event(s) ====\n", n);
    for(unsigned i = 0; i < n; ++i) {
        const Ev &e = out[i];
        fprintf(stderr, "  seq=%llu tid=%u obj=%p  %-26s rc=%llu  site=",
            e.seq, e.tid, e.obj, op_name_(e.op), e.oldc);
        print_site_(e.site);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "==== end ====\n\n");
    fflush(stderr);
}

namespace kame_rc_trace {

// ---- v2: destruction stack (per thread) --------------------------------
// reset() brackets `deleter(pref)` with push/pop.  A tripwire that fires
// with a non-empty stack is a REENTRANT release: the anomaly happened
// inside the destruction chain of the listed object(s) -- e.g. a dying
// Packet's list element pointing back at an ancestor under destruction
// (the shape \xc2\xa79.1's RCT3_35 suggests: DEAD(unique) stores 0, the second
// DEC reads 0 before the allocator's poison lands).
struct DtorFrame { const void *obj; const void *site; };
static constexpr unsigned DTOR_MAX = 64;
static thread_local DtorFrame tl_dtor[DTOR_MAX];
static thread_local unsigned tl_dtor_depth = 0;

void push_dtor(const void *obj, const void *site) noexcept {
    if(tl_dtor_depth < DTOR_MAX)
        tl_dtor[tl_dtor_depth] = DtorFrame{obj, site};
    ++tl_dtor_depth;   // depth counts even past capacity (overflow visible)
}
void pop_dtor() noexcept {
    if(tl_dtor_depth) --tl_dtor_depth;
}

static void dump_dtor_stack_() {
    if(!tl_dtor_depth) {
        fprintf(stderr, "  destruction stack: (empty -- not a reentrant release)\n");
        return;
    }
    fprintf(stderr, "  destruction stack (%u deep, innermost last)%s:\n",
        tl_dtor_depth, tl_dtor_depth > DTOR_MAX ? " [TRUNCATED]" : "");
    unsigned n = tl_dtor_depth < DTOR_MAX ? tl_dtor_depth : DTOR_MAX;
    for(unsigned i = 0; i < n; ++i) {
        fprintf(stderr, "    [%u] destroying obj=%p  from ", i, tl_dtor[i].obj);
        print_site_(tl_dtor[i].site);
        fprintf(stderr, "\n");
    }
}

// ---- v2: anomaly registry + modes --------------------------------------
static bool abort_mode_() noexcept {
    static const bool v = [] {
        const char *e = getenv("KAME_RC_TRACE_ABORT");
        return !(e && e[0] == '0');           // default: abort (gdb workflow)
    }();
    return v;
}
struct AnomSlot { std::atomic<const void *> obj{nullptr};
                  std::atomic<unsigned> count{0}; };
static constexpr unsigned ANOM_MAX = 64;
static AnomSlot g_anom[ANOM_MAX];
static std::atomic<unsigned> g_anom_overflow{0};

static void anom_exit_summary_() {
    unsigned total = 0;
    for(unsigned i = 0; i < ANOM_MAX; ++i)
        if(g_anom[i].obj.load(std::memory_order_relaxed)) ++total;
    if(!total && !g_anom_overflow.load(std::memory_order_relaxed)) return;
    fprintf(stderr, "\n==== kame_rc_trace exit summary: %u anomalous object(s)"
        "%s ====\n", total,
        g_anom_overflow.load(std::memory_order_relaxed) ? " (+overflow)" : "");
    for(unsigned i = 0; i < ANOM_MAX; ++i) {
        const void *o = g_anom[i].obj.load(std::memory_order_relaxed);
        if(!o) continue;
        fprintf(stderr, "  obj=%p  anomalies=%u\n", o,
            g_anom[i].count.load(std::memory_order_relaxed));
        kame_rc_dump(o);
    }
    fflush(stderr);
}
static void register_exit_summary_() {
    static std::atomic<bool> once{false};
    bool f = false;
    if(once.compare_exchange_strong(f, true)) atexit(anom_exit_summary_);
}

//! Returns the anomaly ordinal for this object (1 = first).
static unsigned anom_note_(const void *obj) noexcept {
    for(unsigned i = 0; i < ANOM_MAX; ++i) {
        const void *cur = g_anom[i].obj.load(std::memory_order_acquire);
        if(cur == obj)
            return g_anom[i].count.fetch_add(1, std::memory_order_relaxed) + 1;
        if(!cur) {
            const void *expect = nullptr;
            if(g_anom[i].obj.compare_exchange_strong(expect, obj,
                    std::memory_order_acq_rel)) {
                return g_anom[i].count.fetch_add(1, std::memory_order_relaxed) + 1;
            }
            if(expect == obj)
                return g_anom[i].count.fetch_add(1, std::memory_order_relaxed) + 1;
        }
    }
    g_anom_overflow.fetch_add(1, std::memory_order_relaxed);
    return 1;   // treat as first so it is never silently dropped
}

void anomaly(const void *obj, unsigned op, unsigned long long oldc,
    const void *site) noexcept {
    record(obj, op, oldc, site);
    register_exit_summary_();
    unsigned nth = anom_note_(obj);
    if(abort_mode_()) {
        fprintf(stderr, "\nkame_rc_trace: FATAL %s  obj=%p  rc(before)=%llu  at ",
            op_name_(op), obj, oldc);
        print_site_(site);
        fprintf(stderr, "\n");
        dump_dtor_stack_();
        kame_rc_dump(obj);
        abort();
    }
    if(nth == 1) {
        fprintf(stderr, "\nkame_rc_trace: ANOMALY #1 %s  obj=%p  rc(before)=%llu  at ",
            op_name_(op), obj, oldc);
        print_site_(site);
        fprintf(stderr, "\n");
        dump_dtor_stack_();
        kame_rc_dump(obj);
    } else {
        fprintf(stderr, "kame_rc_trace: anomaly #%u %s obj=%p rc=%llu dtor_depth=%u at ",
            nth, op_name_(op), obj, oldc, tl_dtor_depth);
        print_site_(site);
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    // continue: the fetch_add/fetch_sub already happened; later poison
    // dereferences may still crash the run, but everything above is
    // already on stderr.
}

} // namespace kame_rc_trace
