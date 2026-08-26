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
#include <cstdarg>

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
#include <fcntl.h>
#include <unistd.h>
#define RC_TRACE_HAVE_DLADDR 1
#define RC_TRACE_HAVE_RAW_SINK 1
#endif

// ---- forensic free-poison decode (Â§13.4) -------------------------------
// When the allocator is built with -DKAME_POISON_FORENSIC, a freed pool
// block's first words carry an index into its ring of free records instead
// of a bare magic.  Resolved via dlsym so this TU works with or without
// such an allocator in the process; resolution happens once, on the first
// thread's ring init (NOT in the anomaly path).
#include "kame_poison_forensic.h"
typedef int (*kame_poison_decode_fn)(unsigned long long, kame_freerec *);
typedef unsigned (*kame_poolev_fn)(kame_poolev *, unsigned);
static std::atomic<kame_poison_decode_fn> g_poison_decode{nullptr};
static std::atomic<kame_poolev_fn> g_poolev_fn{nullptr};
static void resolve_poison_decode_() noexcept {
#if defined(__unix__) || defined(__APPLE__)
    static std::atomic<bool> once{false};
    bool f = false;
    if(once.compare_exchange_strong(f, true)) {
        g_poison_decode.store(reinterpret_cast<kame_poison_decode_fn>(
            dlsym(RTLD_DEFAULT, "kame_poison_decode")),
            std::memory_order_release);
        g_poolev_fn.store(reinterpret_cast<kame_poolev_fn>(
            dlsym(RTLD_DEFAULT, "kame_pool_recent_events")),
            std::memory_order_release);
    }
#endif
}
//! Wall clock matching kame_freerec.tsc / kame_poolev.tsc: rdtsc on x86
//! (same as Ev.seq there), cntvct on arm64 (where Ev.seq is only a
//! counter and must NOT be differenced against the allocator's tsc).
static inline unsigned long long rc_wallclock_() noexcept {
#if defined(__x86_64__)
    return __rdtsc();
#elif defined(__aarch64__)
    unsigned long long v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return 0;
#endif
}
static const char *poolev_name_(unsigned k) noexcept {
    switch(k) {
    case KAME_PEV_CHUNK_ALLOC:   return "CHUNK-ALLOC  ";
    case KAME_PEV_CHUNK_RECYCLE: return "CHUNK-RECYCLE";
    case KAME_PEV_CHUNK_RELEASE: return "CHUNK-RELEASE";
    case KAME_PEV_BATCH_RETURN:  return "BATCH-RETURN ";
    case KAME_PEV_DLL_DRAIN:     return "DLL-DRAIN    ";
    case KAME_PEV_CROSS_FLUSH:   return "CROSS-FLUSH  ";
    default: return "?????";
    }
}

namespace kame_rc_trace {

static void install_segv_() noexcept;   // §13.58; defined below the raw helpers

enum Op : unsigned {  // keep in sync with atomic_smart_ptr.h
    OP_BORN = 1, OP_INC, OP_DEC, OP_DEAD, OP_DEAD_UNIQUE,
    OP_INC_FROM_ZERO, OP_DEC_UNDERFLOW,
    OP_WEAK_INC, OP_WEAK_DEC, OP_WEAK_DEAD,
    OP_VADOPT, OP_VMOVE,
    OP_MINE_SHARED, OP_LOOKUP_ESCAPE, OP_DEAD_ELEMENT,
};
static bool is_weak_op_(unsigned op) noexcept {
    return op == OP_WEAK_INC || op == OP_WEAK_DEC || op == OP_WEAK_DEAD;
}
static const char *op_name_(unsigned op) noexcept {
    switch(op) {
    case OP_BORN: return "BORN";
    case OP_INC: return "INC ";
    case OP_DEC: return "DEC ";
    case OP_DEAD: return "DEAD";
    case OP_DEAD_UNIQUE: return "DEAD(unique)";
    case OP_INC_FROM_ZERO: return "INC-FROM-ZERO **TRIPWIRE**";
    case OP_DEC_UNDERFLOW: return "DEC-UNDERFLOW **TRIPWIRE**";
    case OP_WEAK_INC: return "wINC (weak_refcnt)";
    case OP_WEAK_DEC: return "wDEC (weak_refcnt)";
    case OP_WEAK_DEAD: return "wDEAD (weak_refcnt)";
    case OP_VADOPT: return "VADOPT (view association)";
    case OP_DEAD_ELEMENT:
        return "DEAD-ELEMENT **TRIPWIRE** (list element dead at the pre-copy check)";
    case OP_LOOKUP_ESCAPE:
        return "LOOKUP-ESCAPE **TRIPWIRE** (lookup slot outside the pinned tree)";
    case OP_MINE_SHARED:
        return "MINE-SHARED **TRIPWIRE** (copy_branch trusted a mark on a committed-reachable packet)";
    case OP_VMOVE: return "VMOVE (rc_before=src slot)";
    default: return "????";
    }
}

struct Ev {
    const void *obj;
    const void *site;
    const void *slot;       //!< the smart-pointer object performing the op
    const char *tname;      //!< type_name_<T>() literal, or null (BORN)
    unsigned long long seq;
    unsigned long long oldc;
    unsigned op;
    unsigned tid;
};

//! Short type label out of `__PRETTY_FUNCTION__`: everything between
//! "T = " and the closing "]" — enough to tell Packet from PacketList_
//! from PacketWrapper without printing a whole signature.
static void print_type_(const char *tname) {
    if(!tname) { fprintf(stderr, "?"); return; }
    const char *b = strstr(tname, "T = ");
    if(!b) { fprintf(stderr, "%s", tname); return; }
    b += 4;
    const char *e = strchr(b, ']');
    int n = e ? (int)(e - b) : (int)strlen(b);
    if(n > 96) n = 96;
    fprintf(stderr, "%.*s", n, b);
}

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

// ---- raw crash-proof sink -------------------------------------------------
// Why: at 24 threads in abort-mode the other threads are already corrupting
// memory, so a concurrent SIGSEGV can kill the process WHILE the dump is
// being written -- and the pretty dump calls dladdr() per event, which is
// slow enough (symbol-table walk under a loader lock) that the window is
// wide.  One capture was lost exactly that way: header intact, ledger and
// events gone.
//
// So every dump now goes out TWICE: first raw -- straight write(2) of
// unsymbolised lines to a file, NEWEST EVENT FIRST -- then the symbolised
// version to stderr.  Newest-first matters: if the process dies after three
// lines, those three lines are the ones nearest the fault, which is where
// the prior release that caused an underflow lives.
//
// The single most valuable line is emitted before the bulk: PRIOR-RELEASE,
// the most recent DEC / DEAD / DEAD(unique) on this object before the fatal
// op -- i.e. the first releaser in a double release.
static int raw_fd_() noexcept {
#ifdef RC_TRACE_HAVE_RAW_SINK
    static int fd = [] {
        const char *path = getenv("KAME_RC_TRACE_FILE");
        char buf[256];
        if(!path || !*path) {
            snprintf(buf, sizeof buf, "rc_trace.%ld.log", (long)getpid());
            path = buf;
        }
        int f = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
        return f;
    }();
    return fd >= 0 ? fd : 2;
#else
    return 2;
#endif
}
static void raw_write_(const char *b, size_t n) noexcept {
#ifdef RC_TRACE_HAVE_RAW_SINK
    int fd = raw_fd_();
    while(n) {
        ssize_t w = write(fd, b, n);
        if(w <= 0) break;
        b += w; n -= (size_t)w;
    }
#else
    (void)b; (void)n;
#endif
}
//! One formatted line, one write() -- O_APPEND makes short writes to a
//! regular file effectively atomic, so concurrent dumps do not tear lines.
static void raw_line_(const char *fmt, ...) noexcept
    __attribute__((format(printf, 1, 2)));
static void raw_line_(const char *fmt, ...) noexcept {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if(n < 0) return;
    if(n > (int)sizeof buf - 1) n = (int)sizeof buf - 1;
    raw_write_(buf, (size_t)n);
}
//! Short type label, into a caller buffer (no stdio on the raw path).
static const char *type_label_(const char *tname, char *buf, size_t cap) noexcept {
    if(!tname) return "?";
    const char *b = strstr(tname, "T = ");
    if(!b) return tname;
    b += 4;
    const char *e = strchr(b, ']');
    size_t n = e ? (size_t)(e - b) : strlen(b);
    if(n > cap - 1) n = cap - 1;
    memcpy(buf, b, n);
    buf[n] = 0;
    return buf;
}

// ---- destruction-stack storage (hoisted: record() snapshots it) -----------
struct DtorFrame { const void *obj; const void *site; };
static constexpr unsigned DTOR_MAX = 64;
static thread_local DtorFrame tl_dtor[DTOR_MAX];
static thread_local unsigned tl_dtor_depth = 0;

// ---- call-chain capture (v5) ---------------------------------------------
// Â§11.3: nine captures, and the innermost `site` symbol scatters across
// them (clear_fixed x3, bundle x2, local_weak_ptr::reset x2, ...), because at
// -O3 a return address lands wherever the optimiser put the inlined code.  A
// single address cannot name the edge; a few frames can, since the OUTER
// frames are real non-inlined functions (bundle / snapshot / commit / the
// lookup helpers).
//
// Cost: a bounded frame-pointer walk (CHAIN_N loads) on release ops only,
// and one walk at the anomaly.  Opt-in via KAME_RC_TRACE_CHAIN=1 so the
// Â§11.3 fire-rate baseline stays comparable.
//
// REQUIRES the reproducer to be built with -fno-omit-frame-pointer, else the
// walk has nothing to follow.  It is written to fail SHORT rather than wild:
// each candidate frame must be aligned, strictly above the previous one, and
// within 1 MiB of it, so a garbage chain truncates instead of dereferencing
// nonsense.  Re-check the fire rate after enabling it (Â§4's rule).
static constexpr unsigned CHAIN_N = 6;

static bool chain_enabled_() noexcept {
    static const bool v = [] {
        const char *e = getenv("KAME_RC_TRACE_CHAIN");
        return e && e[0] == '1';
    }();
    return v;
}

//! Walk saved frame pointers: fp[0] = caller's fp, fp[1] = return address
//! (true on x86-64 and aarch64 alike).  Returns how many frames were stored.
static unsigned walk_chain_(const void **out, unsigned max) noexcept {
#if defined(__x86_64__) || defined(__aarch64__)
    void **fp = (void **)__builtin_frame_address(0);
    unsigned n = 0;
    uintptr_t prev = 0;
    while(n < max && fp) {
        uintptr_t f = (uintptr_t)fp;
        if(f & (sizeof(void *) - 1)) break;                 // misaligned
        if(prev && (f <= prev || f - prev > (1u << 20))) break;  // implausible
        void *ret = fp[1];
        if(!ret) break;
        out[n++] = ret;
        prev = f;
        fp = (void **)fp[0];
    }
    return n;
#else
    (void)out; (void)max; return 0;
#endif
}

// ---- O(1) last-release cache ---------------------------------------------
// The decisive datum for a double release is "who released this object
// last, before the underflow".  Recovering it from the rings needs a
// 64x16384 scan, and that scan is exactly what a peer thread's SIGSEGV
// interrupts (measured: with the scan first, only the anomaly header
// reached the file).  So maintain it at RECORD time instead: one
// direct-mapped slot per object address, updated on every DEC / DEAD, read
// in O(1) at anomaly time so the line lands immediately after the header.
//
// Racy by construction (a torn slot is possible under concurrent updates to
// the same bucket); the full ring history that follows is the arbiter, this
// is the copy that survives.
struct LastRel {
    std::atomic<const void *> obj{nullptr};
    Ev ev{};
    const void *chain[CHAIN_N]{};
    unsigned chain_n{0};
    //! The containers being destroyed when this release ran (objects on
    //! the dtor stack) -- lets the PRIOR release's containment be compared
    //! against the anomaly's dtor stack: same list appearing in both says
    //! one cascade reached the same holder chain twice.
    const void *dobjs[CHAIN_N]{};
    unsigned dobjs_n{0};
};
static constexpr unsigned LASTREL_SLOTS = 4096;   // power of two
static LastRel g_lastrel[LASTREL_SLOTS];
static inline unsigned lastrel_slot_(const void *obj) noexcept {
    return (unsigned)((((uintptr_t)obj >> 4) * 0x9E3779B97F4A7C15ULL) >> 52)
           & (LASTREL_SLOTS - 1);
}

// ---- O(1) per-object recent-event cache (v7) -------------------------------
// Â§11.5: 4 of 7 captures kept <=1 RC-EV line, because the full history
// needs a 64x16384 ring scan and the peer threads are already corrupting.
// So keep the last RECENT_N events per object in a direct-mapped cache
// maintained at record time, and emit them right after the header --
// before anything that scans.  Racy on bucket collision (two objects
// hashing together thrash the bucket; a torn entry is possible); the
// full RC-EV scan that follows is the arbiter, this is the copy that
// survives.  RECENT_N=16 is enough for Q2 (does the anomalous slot have
// a matching INC nearby?) and for the DEAD->BORN rebirth signature.
struct Recent {
    std::atomic<const void *> obj{nullptr};
    Ev evs[16];
    unsigned idx{0};
};
static constexpr unsigned RECENT_N = 16;
static Recent g_recent[LASTREL_SLOTS];

//! Ring eviction accounting.  Rings are circular AND recycled across
//! threads (slot = round-robin % MAX_RINGS), so old events are dropped
//! silently once a slot has taken RING writes.  That matters for reading
//! a dump: a per-object ledger (BORN/DEAD vs INC/DEC) can only be
//! expected to balance while NOTHING has been evicted.  In abort-mode the
//! process usually dies before any wrap (handoff Â§9's 103/103/206/204 is a
//! genuinely complete history); in continue-mode a long run wraps many
//! times, and an unbalanced ledger then says nothing about coverage.
static std::atomic<unsigned long long> g_writes{0};
static std::atomic<unsigned> g_wrapped_rings{0};

//! Per-op global tallies (relaxed).  Two consumers: the anomaly raw header
//! prints the marker counts so every capture proves the v8 markers were
//! alive in that binary/run (handoff Â§13.2 -- "adopt=0 vmove=0 in the
//! whole file" must be distinguishable from "markers never fired"), and
//! KAME_RC_TRACE_STATS=1 prints the full tally at exit even without an
//! anomaly.
static constexpr unsigned TALLY_N = 32;
static std::atomic<unsigned long long> g_op_tally[TALLY_N];

static void op_tally_exit_() {
    fprintf(stderr, "\n==== kame_rc_trace op tally ====\n");
    for(unsigned i = 1; i < TALLY_N; ++i) {
        unsigned long long v = g_op_tally[i].load(std::memory_order_relaxed);
        if(v) fprintf(stderr, "  %-28s %llu\n", op_name_(i), v);
    }
    fflush(stderr);
}
static void register_stats_() noexcept {
    static std::atomic<bool> once{false};
    bool f = false;
    if(once.compare_exchange_strong(f, true)) {
        const char *e = getenv("KAME_RC_TRACE_STATS");
        if(e && e[0] && e[0] != '0') atexit(op_tally_exit_);
    }
}

// ---- (§13.90) per-object DECREMENT ledger ------------------------------
// §13.89 cornered the defect at "a decrement attributed to nobody": the
// wrapper is the same instance, alive, not double-destructed, and its ctor
// took a count -- yet the count is gone.  The per-object rings answer this
// in principle, but they EVICT (§13.74), which is exactly why the act has
// never been seen.  This is a dedicated, small, non-evicting-by-object
// table: for each object, the last DEC_KEEP decrement sites, direct-mapped
// with the object address stored so a collision is visible rather than
// silently attributed to the wrong object.
namespace {
enum { DEC_SLOTS = 16384, DEC_KEEP = 8 };
struct DecSlot {
    std::atomic<const void *> obj{nullptr};
    std::atomic<unsigned> n{0};
    const void *site[DEC_KEEP]{};
    unsigned long long oldc[DEC_KEEP]{};
    unsigned tid[DEC_KEEP]{};
};
DecSlot g_dec[DEC_SLOTS];
inline unsigned dec_slot_(const void *o) noexcept {
    return (unsigned)((((uintptr_t)o >> 4) * 0x9E3779B97F4A7C15ull) >> 50) % DEC_SLOTS;
}
void dec_note_(const void *obj, const void *site, unsigned long long oldc,
               unsigned tid) noexcept {
    DecSlot &d = g_dec[dec_slot_(obj)];
    if(d.obj.load(std::memory_order_relaxed) != obj) {
        d.obj.store(obj, std::memory_order_relaxed);   // new tenant: restart
        d.n.store(0, std::memory_order_relaxed);
    }
    unsigned i = d.n.fetch_add(1, std::memory_order_relaxed);
    d.site[i % DEC_KEEP] = site;
    d.oldc[i % DEC_KEEP] = oldc;
    d.tid[i % DEC_KEEP]  = tid;
}
}
//! Dump every recorded decrement for \a obj.  Called from the tripwire /
//! probe path, so it runs at the moment the corpse is found.
void dec_dump(const void *obj) noexcept {
    DecSlot &d = g_dec[dec_slot_(obj)];
    if(d.obj.load(std::memory_order_relaxed) != obj) {
        raw_line_("RC-DECLEDGER obj=%p (no entries: slot taken by %p)\n",
            obj, d.obj.load(std::memory_order_relaxed));
        return;
    }
    unsigned n = d.n.load(std::memory_order_relaxed);
    unsigned shown = n < DEC_KEEP ? n : DEC_KEEP;
    raw_line_("RC-DECLEDGER obj=%p total_decs=%u showing=%u (oldest first)\n",
        obj, n, shown);
    unsigned start = n < DEC_KEEP ? 0 : n - DEC_KEEP;
    for(unsigned k = 0; k < shown; ++k) {
        unsigned i = (start + k) % DEC_KEEP;
        raw_line_("  DEC[%u] site=%p rc_before=%llu tid=%u\n",
            start + k, d.site[i], d.oldc[i], d.tid[i]);
    }
}

void record(const void *obj, unsigned op, unsigned long long oldc,
    const void *site, const char *tname, const void *slot) noexcept {
    if(op < TALLY_N) g_op_tally[op].fetch_add(1, std::memory_order_relaxed);
    TL &t = tl;
    if(op == OP_DEC || op == OP_DEAD || op == OP_DEAD_UNIQUE
       || op == OP_DEC_UNDERFLOW)
        dec_note_(obj, site, oldc, rc_trace_tid_());   // §13.90
    if(!t.ring) {
        register_stats_();
        resolve_poison_decode_();
        install_segv_();
        t.ring = g_rings[g_ring_next.fetch_add(1, std::memory_order_relaxed)
                         % MAX_RINGS];
        t.tid = rc_trace_tid_();
    }
    if(t.idx == RING)                  // this thread just filled its ring
        g_wrapped_rings.fetch_add(1, std::memory_order_relaxed);
    g_writes.fetch_add(1, std::memory_order_relaxed);
    Ev &e = t.ring[t.idx++ & RING_MASK];
    e.obj = obj; e.site = site; e.slot = slot; e.tname = tname;
    e.seq = rc_trace_seq_();
    e.oldc = oldc; e.op = op; e.tid = t.tid;
    {
        Recent &rc = g_recent[lastrel_slot_(obj)];
        if(rc.obj.load(std::memory_order_relaxed) != obj) {
            rc.idx = 0;
            rc.obj.store(obj, std::memory_order_relaxed);
        }
        rc.evs[rc.idx++ % RECENT_N] = e;
    }
    if(op == OP_DEC || op == OP_DEAD || op == OP_DEAD_UNIQUE) {
        LastRel &lr = g_lastrel[lastrel_slot_(obj)];
        lr.ev = e;                                   // racy by design
        lr.chain_n = chain_enabled_()
            ? walk_chain_(lr.chain, CHAIN_N) : 0;
        unsigned dn = tl_dtor_depth < CHAIN_N ? tl_dtor_depth : CHAIN_N;
        for(unsigned i = 0; i < dn; ++i) lr.dobjs[i] = tl_dtor[i].obj;
        lr.dobjs_n = dn;
        lr.obj.store(obj, std::memory_order_release);
    }
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

//! Collect this object's events into \a out, oldest first.  Shared static
//! buffer, serialised by \a g_dumping so concurrent dumps do not shred it;
//! best-effort (a spin that gives up rather than deadlocking behind a thread
//! that is about to die).
static constexpr unsigned MAXOUT = 4096;
static Ev g_out[MAXOUT];
static std::atomic<bool> g_dumping{false};

static bool dump_lock_() noexcept {
    for(int i = 0; i < 1000000; ++i) {
        bool f = false;
        if(g_dumping.compare_exchange_weak(f, true, std::memory_order_acq_rel))
            return true;
    }
    return false;   // proceed unserialised rather than hang
}
static void dump_unlock_() noexcept { g_dumping.store(false, std::memory_order_release); }

static unsigned collect_(const void *obj) noexcept {
    unsigned n = 0;
    // Only rings that were ever handed out contain events.
    unsigned nr = g_ring_next.load(std::memory_order_relaxed);
    if(nr > MAX_RINGS) nr = MAX_RINGS;
    for(unsigned r = 0; r < nr; ++r)
        for(unsigned i = 0; i < RING; ++i) {
            const Ev &e = g_rings[r][i];
            if(e.obj == obj && e.seq && n < MAXOUT)
                g_out[n++] = e;
        }
    for(unsigned i = 1; i < n; ++i) {          // insertion sort by seq
        Ev key = g_out[i];
        unsigned j = i;
        for(; j > 0 && g_out[j-1].seq > key.seq; --j) g_out[j] = g_out[j-1];
        g_out[j] = key;
    }
    return n;
}

//! One line per chain, addresses only -- resolve offline with
//!   addr2line -e <binary> -f -C -i <addr>...
//! `-i` is the part that matters: it expands the INLINE stack at each
//! address, which is what a bare symbol lookup could not do (Â§11.3).
static void raw_chain_(const char *kind, const void *obj,
    const void *const *chain, unsigned n) noexcept {
    if(!n) {
        raw_line_("RC-CHAIN-%s obj=%p none (chain capture off or no frames)\n",
            kind, obj);
        return;
    }
    char buf[512];
    int k = snprintf(buf, sizeof buf, "RC-CHAIN-%s obj=%p frames=%u", kind, obj, n);
    for(unsigned i = 0; i < n && k > 0 && k < (int)sizeof buf - 24; ++i)
        k += snprintf(buf + k, sizeof buf - (size_t)k, " %p", chain[i]);
    if(k > 0 && k < (int)sizeof buf - 2) { buf[k++] = '\n'; buf[k] = 0; }
    raw_write_(buf, (size_t)(k > 0 ? k : 0));
}

//! O(1) recent-history emitter: up to RECENT_N events for \a obj from the
//! record-time cache, newest first, no scan.  Emitted before anything slow.
static void raw_recent_(const void *obj) noexcept {
    Recent &rc = g_recent[lastrel_slot_(obj)];
    if(rc.obj.load(std::memory_order_relaxed) != obj) {
        raw_line_("RC-RECENT obj=%p none-cached (bucket taken by another object)\n", obj);
        return;
    }
    unsigned total = rc.idx;
    unsigned n = total < RECENT_N ? total : RECENT_N;
    raw_line_("RC-RECENT obj=%p n=%u of %u (newest first, O(1) cache)\n",
        obj, n, total);
    for(unsigned k = 0; k < n; ++k) {
        const Ev &e = rc.evs[(total - 1 - k) % RECENT_N];
        char tb[128];
        raw_line_("RC-R obj=%p op=%s rc_before=%llu tid=%u seq=%llu slot=%p "
            "site=%p type=%s\n", obj, op_name_(e.op), e.oldc, e.tid, e.seq,
            e.slot, e.site, type_label_(e.tname, tb, sizeof tb));
    }
}

//! O(1) variant, emitted BEFORE any scan so it survives a peer crash.
static void raw_prior_release_fast_(const void *obj) noexcept {
    LastRel &lr = g_lastrel[lastrel_slot_(obj)];
    if(lr.obj.load(std::memory_order_acquire) != obj) {
        raw_line_("RC-PRIOR-RELEASE-FAST obj=%p none-cached\n", obj);
        return;
    }
    Ev e = lr.ev;
    const void *ch[CHAIN_N];
    unsigned cn = lr.chain_n;
    if(cn > CHAIN_N) cn = CHAIN_N;
    for(unsigned i = 0; i < cn; ++i) ch[i] = lr.chain[i];
    char tb[128];
    raw_line_("RC-PRIOR-RELEASE-FAST obj=%p op=%s rc_before=%llu tid=%u "
        "seq=%llu slot=%p site=%p type=%s\n", obj, op_name_(e.op), e.oldc,
        e.tid, e.seq, e.slot, e.site, type_label_(e.tname, tb, sizeof tb));
    raw_chain_("PRIOR", obj, ch, cn);
    if(lr.dobjs_n) {
        char db[384];
        int k = snprintf(db, sizeof db, "RC-PRIOR-DTOR obj=%p depth=%u",
            obj, lr.dobjs_n);
        for(unsigned i = 0; i < lr.dobjs_n && k > 0 && k < (int)sizeof db - 24; ++i)
            k += snprintf(db + k, sizeof db - (size_t)k, " %p", lr.dobjs[i]);
        if(k > 0 && k < (int)sizeof db - 2) { db[k++] = '\n'; db[k] = 0; }
        raw_write_(db, (size_t)(k > 0 ? k : 0));
    }
}

// ---- §13.58 SIGSEGV/SIGBUS forensics ---------------------------------
// §13.57 resolved a crash to "a const member read a value it was never
// constructed with" -- premature slot reuse under a running destructor --
// but the WRITER is blocked behind rr's replay divergence on PREEMPT_RT.
// This handler answers the cheaper half of the question on every NATIVE
// crash: at SIGSEGV, dump (raw sink only, crash-time discipline) the
// fault address, key registers, and for each candidate object base
// (rbp = the destructing object in the §13.57 shape, rdi, the fault
// address) the O(1) caches: prior-release line, recent history, then the
// pool-event tail -- all table lookups, no deref of crash memory.  LAST,
// with SA_RESETHAND already armed so a nested fault just ends the
// process with everything above flushed, it reads candidate word 0 for
// the forensic poison token and prints the RC-FREEREC culprit ("who
// freed this slot, when").  Default ON in tracer builds;
// KAME_RC_TRACE_SEGV=0 disables.
#if (defined(__unix__) || defined(__APPLE__))
#include <signal.h>
#if defined(__linux__)
#include <ucontext.h>
#else
#include <sys/ucontext.h>   // macOS: <ucontext.h> demands _XOPEN_SOURCE
#endif
static void raw_poolev_tail_() noexcept {
    if(kame_poolev_fn pf = g_poolev_fn.load(std::memory_order_acquire)) {
        kame_poolev evs[8];
        unsigned n = pf(evs, 8);
        unsigned long long now = rc_wallclock_();
        for(unsigned i = 0; i < n; ++i)
            raw_line_("RC-POOLEV %s addr=%p aux=%llu tid=%u age_tsc=%llu\n",
                poolev_name_(evs[i].kind), evs[i].addr, evs[i].aux,
                evs[i].tid, now - evs[i].tsc);
    }
}
static void segv_handler_(int sig, siginfo_t *si, void *uc_) noexcept {
    // Candidate object bases.  `this` of the destructing object lives in
    // a callee-saved register or rbp/x29 depending on codegen -- §13.57's
    // frame had it in rbp -- so scan the plausible set rather than betting
    // on one.  All uses below are TABLE lookups keyed by address; nothing
    // dereferences crash memory until the final, deliberately-last W0 read.
    enum { NCAND = 9 };
    const void *cands[NCAND] = {};
    const char *names[NCAND] = {};
    unsigned long long ip = 0;
#if defined(__x86_64__) && defined(__linux__)
    ucontext_t *uc = (ucontext_t *)uc_;
    const greg_t *g = uc->uc_mcontext.gregs;
    ip = (unsigned long long)g[REG_RIP];
    const int regs[] = { REG_RBP, REG_RDI, REG_RBX, REG_R12, REG_R13,
                         REG_R14, REG_R15, REG_RAX };
    const char *rn[] = { "rbp", "rdi", "rbx", "r12", "r13",
                         "r14", "r15", "rax" };
    for(int i = 0; i < 8; ++i) { cands[i] = (const void *)g[regs[i]]; names[i] = rn[i]; }
    raw_line_("\nRC-SEGV sig=%d fault_addr=%p ip=0x%llx rbp=0x%llx rdi=0x%llx "
        "rax=0x%llx rbx=0x%llx r12=0x%llx r13=0x%llx r14=0x%llx r15=0x%llx tid=%u\n",
        sig, si ? si->si_addr : nullptr, ip,
        (unsigned long long)g[REG_RBP], (unsigned long long)g[REG_RDI],
        (unsigned long long)g[REG_RAX], (unsigned long long)g[REG_RBX],
        (unsigned long long)g[REG_R12], (unsigned long long)g[REG_R13],
        (unsigned long long)g[REG_R14], (unsigned long long)g[REG_R15],
        rc_trace_tid_());
#elif defined(__aarch64__) && defined(__APPLE__)
    ucontext_t *uc = (ucontext_t *)uc_;
    ip = uc->uc_mcontext->__ss.__pc;
    cands[0] = (const void *)uc->uc_mcontext->__ss.__fp;   names[0] = "fp";
    cands[1] = (const void *)uc->uc_mcontext->__ss.__x[0]; names[1] = "x0";
    for(int i = 0; i < 6; ++i) {                            // x19..x24 (callee-saved)
        cands[2 + i] = (const void *)uc->uc_mcontext->__ss.__x[19 + i];
        names[2 + i] = "x19+";
    }
    raw_line_("\nRC-SEGV sig=%d fault_addr=%p ip=0x%llx fp=0x%llx x0=0x%llx "
        "x19=0x%llx x20=0x%llx tid=%u\n", sig, si ? si->si_addr : nullptr, ip,
        (unsigned long long)uc->uc_mcontext->__ss.__fp,
        (unsigned long long)uc->uc_mcontext->__ss.__x[0],
        (unsigned long long)uc->uc_mcontext->__ss.__x[19],
        (unsigned long long)uc->uc_mcontext->__ss.__x[20], rc_trace_tid_());
#else
    (void)uc_;
    raw_line_("\nRC-SEGV sig=%d fault_addr=%p (no register decode on this target) tid=%u\n",
        sig, si ? si->si_addr : nullptr, rc_trace_tid_());
#endif
    cands[NCAND - 1] = si ? si->si_addr : nullptr;
    names[NCAND - 1] = "fault_addr";
    for(int i = 0; i < NCAND; ++i) {
        if( !cands[i] || !names[i]) continue;
        if((uintptr_t)cands[i] < 0x1000u) continue;      // not an object base
        bool dup = false;
        for(int j = 0; j < i; ++j) if(cands[j] == cands[i]) dup = true;
        if(dup) continue;
        // Only report candidates the caches actually know -- an
        // unfiltered 9-register dump would bury the signal.
        LastRel &lr = g_lastrel[lastrel_slot_(cands[i])];
        Recent &rc = g_recent[lastrel_slot_(cands[i])];
        bool known = lr.obj.load(std::memory_order_acquire) == cands[i]
                  || rc.obj.load(std::memory_order_relaxed) == cands[i];
        if( !known) continue;
        raw_line_("RC-SEGV-CAND %s=%p (cache HIT)\n", names[i], cands[i]);
        raw_prior_release_fast_(cands[i]);
        raw_recent_(cands[i]);
    }
    raw_poolev_tail_();
    // LAST: deref candidates' word 0 for the forensic token (may nested-
    // fault; SA_RESETHAND means that just ends the process -- everything
    // above is already on disk via the O_APPEND raw sink).
    kame_poison_decode_fn dec = g_poison_decode.load(std::memory_order_acquire);
    for(int i = 0; i < NCAND; ++i) {       // aligned, plausible bases only
        const void *c = cands[i];
        if( !c || ((uintptr_t)c & 7u) || (uintptr_t)c < 0x1000u) continue;
        bool dup = false;
        for(int j = 0; j < i; ++j) if(cands[j] == c) dup = true;
        if(dup) continue;
        unsigned long long w0 = *(volatile const unsigned long long *)c;
        raw_line_("RC-SEGV-W0 %s=%p w0=0x%llx\n", names[i], c, w0);
        if(dec && (w0 >> 48) == KAME_POISON_TAG) {
            kame_freerec fr;
            if(dec(w0, &fr))
                raw_line_("RC-FREEREC freed_ptr=%p size=%llu free_tid=%u "
                    "age_tsc=%llu frames=%u %p %p %p %p\n", fr.ptr,
                    (unsigned long long)fr.size, fr.tid,
                    rc_wallclock_() - fr.tsc, fr.nret,
                    fr.nret > 0 ? fr.ret[0] : nullptr,
                    fr.nret > 1 ? fr.ret[1] : nullptr,
                    fr.nret > 2 ? fr.ret[2] : nullptr,
                    fr.nret > 3 ? fr.ret[3] : nullptr);
        }
    }
    // fall through: SA_RESETHAND restored SIG_DFL; returning re-executes
    // the faulting instruction -> default SIGSEGV -> core/exit code intact.
}
static void install_segv_() noexcept {
    static std::atomic<bool> once{false};
    bool f = false;
    if( !once.compare_exchange_strong(f, true)) return;
    if(const char *e = getenv("KAME_RC_TRACE_SEGV"))
        if(e[0] == '0') return;                      // default ON
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = segv_handler_;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_NODEFER;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
}
#else
static void install_segv_() noexcept {}
#endif

//! The decisive line for a double release: the most recent DEC / DEAD /
//! DEAD(unique) strictly before the newest event (the fatal one), i.e. the
//! FIRST releaser.  Emitted before the bulk dump so it survives a
//! concurrent crash.
static void raw_prior_release_(const void *obj, unsigned n) noexcept {
    if(n < 2) {
        raw_line_("RC-PRIOR-RELEASE obj=%p none (history has %u event(s))\n",
            obj, n);
        return;
    }
    for(unsigned i = n - 1; i-- > 0; ) {
        const Ev &e = g_out[i];
        if(e.op == OP_DEC || e.op == OP_DEAD || e.op == OP_DEAD_UNIQUE) {
            char tb[128];
            raw_line_("RC-PRIOR-RELEASE obj=%p op=%s rc_before=%llu tid=%u "
                "seq=%llu dseq=-%llu site=%p type=%s\n",
                obj, op_name_(e.op), e.oldc, e.tid, e.seq,
                g_out[n-1].seq - e.seq, e.site,
                type_label_(e.tname, tb, sizeof tb));
            return;
        }
    }
    raw_line_("RC-PRIOR-RELEASE obj=%p none in %u recorded event(s)\n", obj, n);
}

//! Raw, unsymbolised, NEWEST FIRST.
static void raw_events_(const void *obj, unsigned n) noexcept {
    unsigned wrapped = g_wrapped_rings.load(std::memory_order_relaxed);
    // Cap the slow (post-scan) dump to the newest events; the RC-RECENT
    // block above already secured the tail, and Â§11.5 showed the long dump
    // rarely survives anyway.  KAME_RC_TRACE_FULL=1 restores everything.
    unsigned cap = n;
    {
        static const char *fe = getenv("KAME_RC_TRACE_FULL");
        if(!(fe && fe[0] == '1') && cap > 40) cap = 40;
    }
    raw_line_("RC-HIST obj=%p events=%u shown=%u evicted_rings=%u total_writes=%llu\n",
        obj, n, cap, wrapped,
        (unsigned long long)g_writes.load(std::memory_order_relaxed));
    unsigned lo = n - cap;
    for(unsigned i = n; i-- > lo; ) {
        const Ev &e = g_out[i];
        char tb[128];
        raw_line_("RC-EV obj=%p i=%u op=%s rc_before=%llu tid=%u seq=%llu "
            "slot=%p site=%p type=%s\n", obj, i, op_name_(e.op), e.oldc,
            e.tid, e.seq, e.slot, e.site, type_label_(e.tname, tb, sizeof tb));
    }
    raw_line_("RC-END obj=%p\n", obj);
}

} // namespace kame_rc_trace

//! Dump the recorded refcount history of one object, oldest first.
//! From gdb at the poison fault:  call kame_rc_dump((const void*)$rsi)
//! Also called by the tripwires before abort().
extern "C" void kame_rc_dump(const void *obj) {
    using namespace kame_rc_trace;
    bool locked = dump_lock_();
    unsigned n = collect_(obj);
    Ev *out = g_out;
    // RAW FIRST: unsymbolised, newest-first, straight write(2) -- survives a
    // concurrent crash during the (dladdr-heavy) pretty phase below.
    raw_prior_release_(obj, n);
    raw_events_(obj, n);
    // Ledger, split strong vs weak (they are different counters).
    unsigned born=0, dead=0, inc=0, dec=0, winc=0, wdec=0, wdead=0, trip=0,
        mark=0;
    const char *ty = nullptr;
    for(unsigned i = 0; i < n; ++i) {
        switch(out[i].op) {
        case OP_BORN: ++born; break;
        case OP_INC: ++inc; break;
        case OP_DEC: ++dec; break;
        case OP_DEAD: case OP_DEAD_UNIQUE: ++dead; break;
        case OP_WEAK_INC: ++winc; break;
        case OP_WEAK_DEC: ++wdec; break;
        case OP_WEAK_DEAD: ++wdead; break;
        case OP_VADOPT: case OP_VMOVE: ++mark; break;   // count-neutral
        default: ++trip; break;
        }
        if(!ty && out[i].tname) ty = out[i].tname;
    }
    unsigned wrapped = g_wrapped_rings.load(std::memory_order_relaxed);
    fprintf(stderr, "\n==== kame_rc_trace: %u event(s) for obj %p "
        "(oldest first; seq is TSC) ====\n", n, obj);
    fprintf(stderr, "  type: ");
    print_type_(ty);
    fprintf(stderr, "\n  ledger: strong BORN %u / DEAD %u / INC %u / DEC %u"
        "   weak wINC %u / wDEC %u / wDEAD %u   markers %u   tripwires %u\n",
        born, dead, inc, dec, winc, wdec, wdead, mark, trip);
    if(wrapped)
        fprintf(stderr, "  ** %u ring(s) have wrapped (%llu events recorded) --"
            " older events were EVICTED, so an unbalanced ledger here says"
            " nothing about tracer coverage **\n",
            wrapped, (unsigned long long)g_writes.load(std::memory_order_relaxed));
    else
        fprintf(stderr, "  no ring has wrapped -- this history is complete\n");
    for(unsigned i = 0; i < n; ++i) {
        const Ev &e = out[i];
        // Count-neutral markers: `oldc` is not a count.  VMOVE carries the
        // SOURCE slot address there; render both with their slot(s) instead
        // of a bogus "rc 0 -> 0" arithmetic line.
        if(e.op == OP_VADOPT || e.op == OP_VMOVE) {
            fprintf(stderr, "  seq=%llu%+lld tid=%u  %-26s slot=%p%s",
                e.seq, i ? (long long)(e.seq - out[0].seq) : 0LL, e.tid,
                op_name_(e.op), e.slot,
                e.op == OP_VMOVE ? "" : "  site=");
            if(e.op == OP_VMOVE)
                fprintf(stderr, " src_slot=%p  site=",
                    (const void *)(uintptr_t)e.oldc);
            print_site_(e.site);
            fprintf(stderr, "\n");
            continue;
        }
        fprintf(stderr, "  seq=%llu%+lld tid=%u  %-26s %s %llu -> %llu  site=",
            e.seq, i ? (long long)(e.seq - out[0].seq) : 0LL, e.tid,
            op_name_(e.op), is_weak_op_(e.op) ? "wrc" : "rc ", e.oldc,
            (e.op == OP_INC || e.op == OP_INC_FROM_ZERO || e.op == OP_BORN
             || e.op == OP_WEAK_INC)
                ? e.oldc + (e.op == OP_BORN ? 0 : 1)
                : (e.oldc ? e.oldc - 1 : 0));
        print_site_(e.site);
        fprintf(stderr, "\n");
    }
    if(n == MAXOUT)
        fprintf(stderr, "  (output truncated at %u)\n", MAXOUT);
    fprintf(stderr, "==== end of history for %p ====\n\n", obj);
    fflush(stderr);
    if(locked) dump_unlock_();
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
        fprintf(stderr, "  seq=%llu tid=%u obj=%p  %-26s rc=%llu  type=",
            e.seq, e.tid, e.obj, op_name_(e.op), e.oldc);
        print_type_(e.tname);
        fprintf(stderr, "  site=");
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
// ---- §13.83 wrapper -> packet-at-construction association ---------------
// (anomaly() is defined below; declare it here so this block can sit next
//  to the other O(1) caches rather than after the reporter.)
// Direct-mapped, same hashing as the other O(1) caches.  Racy by design:
// a lost note only costs a missed check, never a false report -- a report
// requires BOTH a matching wrapper address AND a differing packet.
namespace {
struct WPSlot { std::atomic<const void *> w{nullptr}; const void *p{nullptr};
                long long id{0}; };   //!< §13.89: incarnation id (m_bundle_serial)
constexpr unsigned WP_SLOTS = 8192;
WPSlot g_wp[WP_SLOTS];
inline unsigned wp_slot_(const void *w) noexcept {
    return (unsigned)((((uintptr_t)w >> 4) * 0x9E3779B97F4A7C15ull) >> 51)
           % WP_SLOTS;
}
}
void anomaly(const void *obj, unsigned op, unsigned long long oldc,
    const void *site, const char *tname, const void *slot) noexcept;
static bool wp_report_enabled_() noexcept {
    static const bool v = [] {
        const char *e = getenv("KAME_RC_TRACE_WP_REPORT");
        return e && e[0] && e[0] != '0';
    }();
    return v;
}
void wp_note(const void *wrapper, const void *packet) noexcept {
    if(!wrapper) return;
    WPSlot &s = g_wp[wp_slot_(wrapper)];
    s.p = packet;                                   // publish value first
    s.w.store(wrapper, std::memory_order_release);
}
//! (§13.89) Record the wrapper's INCARNATION (its m_bundle_serial, set at
//! construction and never changed).  Pool storage is reused, so an address
//! alone cannot tell one wrapper from its successor -- and §13.87 leaves
//! "the live wrapper seen at the entry check is a different instance from
//! the one that named the packet at release time" as the untested reading.
void wp_note_id(const void *wrapper, long long id) noexcept {
    if(!wrapper) return;
    WPSlot &s = g_wp[wp_slot_(wrapper)];
    s.id = id;
}
//! Report only when the address still carries our note but the incarnation
//! differs -- i.e. this storage was reused since the note was written.
void wp_check_id(const void *wrapper, long long id_now, const void *site) noexcept {
    if(!wrapper) return;
    WPSlot &s = g_wp[wp_slot_(wrapper)];
    if(s.w.load(std::memory_order_acquire) != wrapper) return;   // no note: miss
    if(s.id == 0 || s.id == id_now) return;                      // same incarnation
    anomaly(wrapper, OP_DEAD_ELEMENT, (unsigned long long)s.id, site,
        "WRAPPER ADDRESS REUSED since construction (rc_before=born serial)",
        (const void *)(uintptr_t)id_now);
}
void wp_check(const void *wrapper, const void *packet,
              const void *site, const char *where) noexcept {
    if(!wrapper) return;
    WPSlot &s = g_wp[wp_slot_(wrapper)];
    if(s.w.load(std::memory_order_acquire) != wrapper) return;  // no note
    const void *born = s.p;
    if(born == packet) return;                       // unchanged: fine
    if( !born) return;
    if(!wp_report_enabled_()) return;
    //!< §13.83 measured: an m_packet change is NORMAL and frequent (195k in
    //!< a 40s run) -- copy-on-write through the non-const packet() is how
    //!< bundle builds its private wrapper.  So "changed" is not a defect
    //!< predicate and this reporter stays OFF unless explicitly asked for
    //!< (KAME_RC_TRACE_WP_REPORT=1), kept only as a frequency probe.
    //!< Born NULL is the bundledBy ctor, whose packet is legitimately
    //!< assigned right after (transaction_impl.h:1444/:1561, both
    //!< pre-publish).  Only the overwrite of an ALREADY-SET counted member
    //!< is interesting -- and skipping this is what flooded the first run.
    (void)where;
    //!< `tname` must have STATIC lifetime: anomaly() stores the pointer in
    //!< the ring and the dump reads it later.  Passing a stack buffer here
    //!< crashed the first attempt (dangling read at dump time), so the two
    //!< pointers travel in the numeric fields instead: rc_before = the
    //!< packet it was BORN with, slot = the packet it names NOW.
    anomaly(wrapper, OP_DEAD_ELEMENT,
        (unsigned long long)(uintptr_t)born, site,
        "wrapper m_packet OVERWRITTEN (rc_before=born packet, slot=now)",
        packet);
}

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

//! KAME_RC_TRACE_TYPES=<substr>: only REPORT anomalies whose type label
//! contains <substr> (recording is unaffected).  Handoff Â§11.1 asked for
//! gating on the ledger-complete classes -- `KAME_RC_TRACE_TYPES=Packet`
//! keeps Packet/PacketList_/PacketWrapper and drops the rest.
static bool type_selected_(const char *tname) noexcept {
    static const char *filt = getenv("KAME_RC_TRACE_TYPES");
    if(!filt || !*filt) return true;
    return tname && strstr(tname, filt) != nullptr;
}

void anomaly(const void *obj, unsigned op, unsigned long long oldc,
    const void *site, const char *tname, const void *slot) noexcept {
    record(obj, op, oldc, site, tname, slot);
    if(!type_selected_(tname)) return;   // recorded, not reported
    register_exit_summary_();
    unsigned nth = anom_note_(obj);
    // ---- RAW PHASE: no dladdr, no stdio buffering.  Everything a later
    // reader needs is on disk before the pretty phase risks dying to a
    // peer thread's SIGSEGV.
    {
        char tb[128];
        raw_line_("\nRC-ANOMALY #%u obj=%p op=%s rc_before=%llu tid=%u "
            "slot=%p site=%p type=%s dtor_depth=%u\n", nth, obj, op_name_(op),
            oldc, rc_trace_tid_(), slot, site,
            type_label_(tname, tb, sizeof tb), tl_dtor_depth);
        // Marker-liveness receipt (Â§13.2): totals across the whole run so
        // far, NOT this object's -- zero here means the markers never
        // fired in this binary, nonzero means absence from a per-object
        // history is a keying fact, not a wiring fault.
        raw_line_("RC-MARKERS-ALIVE adopt=%llu vmove=%llu (run totals)\n",
            g_op_tally[OP_VADOPT].load(std::memory_order_relaxed),
            g_op_tally[OP_VMOVE].load(std::memory_order_relaxed));
        // Forensic poison (Â§13.4): if the stale count carries the 0xBAAD
        // token, name the free that killed the block -- thread, call chain
        // (resolve offline: addr2line -e <alloc.so> -f -C -i), age, and how
        // many stale count ops already hit the word (drift).  This line is
        // immune to tracer-ring eviction: the block itself carried the key.
        if((oldc >> 48) == KAME_POISON_TAG) {
            kame_poison_decode_fn dec =
                g_poison_decode.load(std::memory_order_acquire);
            kame_freerec fr;
            if(dec && dec(oldc, &fr)) {
                // age = stale-op time minus free time.  On x86 both sides are
                // rdtsc (Ev.seq and kame_freerec.tsc share the clock), so this
                // is directly comparable across a capture and across the §13.5
                // batch; on other hosts the two clocks differ -- ignore it.
                raw_line_("RC-FREEREC freed_ptr=%p%s size=%llu free_tid=%u "
                    "tsc=%llu age_tsc=%llu drift=%+d frames=%u %p %p %p %p\n",
                    fr.ptr, fr.ptr == obj ? " (=obj)" : " (DIFFERENT from obj!)",
                    (unsigned long long)fr.size, fr.tid, fr.tsc,
                    rc_wallclock_() - fr.tsc,
                    (int)(oldc & 0xFFFFu) - (int)KAME_POISON_PAD, fr.nret,
                    fr.nret > 0 ? fr.ret[0] : nullptr,
                    fr.nret > 1 ? fr.ret[1] : nullptr,
                    fr.nret > 2 ? fr.ret[2] : nullptr,
                    fr.nret > 3 ? fr.ret[3] : nullptr);
            } else if(dec) {
                raw_line_("RC-FREEREC tag matched, record overwritten "
                    "(free predates the ring window) or plain poison\n");
            } else {
                raw_line_("RC-FREEREC tag matched, no kame_poison_decode in "
                    "process (plain-poison allocator)\n");
            }
        }
        // §13.13: the pool's own timeline around the anomaly.  age_tsc is
        // (now - event); SAME-UNIT marks events in the anomaly object's
        // 256 KiB unit (ALLOC_MIN_CHUNK_SHIFT = 18) -- chunk-level identity
        // without needing the allocator to export chunk_of().
        if(kame_poolev_fn pf = g_poolev_fn.load(std::memory_order_acquire)) {
            kame_poolev evs[12];
            unsigned n = pf(evs, 12);
            unsigned long long now = rc_wallclock_();
            for(unsigned i = 0; i < n; ++i) {
                raw_line_("RC-POOLEV %s addr=%p aux=%llu tid=%u age_tsc=%llu%s\n",
                    poolev_name_(evs[i].kind), evs[i].addr,
                    evs[i].aux, evs[i].tid, now - evs[i].tsc,
                    (((uintptr_t)evs[i].addr >> 18) == ((uintptr_t)obj >> 18))
                        ? "  <== SAME-UNIT as obj" : "");
            }
            if(!n) raw_line_("RC-POOLEV (none recorded)\n");
        }
        if(chain_enabled_()) {
            const void *ch[CHAIN_N];
            unsigned cn = walk_chain_(ch, CHAIN_N);
            raw_chain_("ANOM", obj, ch, cn);
        }
        raw_prior_release_fast_(obj);      // O(1): the decisive line, first
        raw_recent_(obj);                  // O(1): last 16 events, pre-scan
        unsigned dn = tl_dtor_depth < DTOR_MAX ? tl_dtor_depth : DTOR_MAX;
        for(unsigned i = 0; i < dn; ++i)
            raw_line_("RC-DTOR [%u] obj=%p site=%p\n", i,
                tl_dtor[i].obj, tl_dtor[i].site);
    }
    if(abort_mode_()) {
        fprintf(stderr, "\nkame_rc_trace: FATAL %s  obj=%p  rc(before)=%llu  type=",
            op_name_(op), obj, oldc);
        print_type_(tname);
        fprintf(stderr, "  at ");
        print_site_(site);
        fprintf(stderr, "\n");
        dump_dtor_stack_();
        kame_rc_dump(obj);
        abort();
    }
    if(nth == 1) {
        fprintf(stderr, "\nkame_rc_trace: ANOMALY #1 %s  obj=%p  rc(before)=%llu  type=",
            op_name_(op), obj, oldc);
        print_type_(tname);
        fprintf(stderr, "  at ");
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
