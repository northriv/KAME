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

[[noreturn]] void die(const void *obj, unsigned op, unsigned long long oldc,
    const void *site) noexcept {
    // Record the fatal event itself so it appears in the dump, then dump
    // and abort.  The abort's core/backtrace points at the guilty caller.
    record(obj, op, oldc, site);
    fprintf(stderr, "\nkame_rc_trace: FATAL %s  obj=%p  rc(before)=%llu  at ",
        op_name_(op), obj, oldc);
    print_site_(site);
    fprintf(stderr, "\n");
    kame_rc_dump(obj);
    abort();
}

} // namespace kame_rc_trace
