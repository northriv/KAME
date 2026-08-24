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

namespace kame_rc_trace {

enum Op : unsigned {  // keep in sync with atomic_smart_ptr.h
    OP_BORN = 1, OP_INC, OP_DEC, OP_DEAD, OP_DEAD_UNIQUE,
    OP_INC_FROM_ZERO, OP_DEC_UNDERFLOW,
    OP_WEAK_INC, OP_WEAK_DEC, OP_WEAK_DEAD,
    OP_VADOPT, OP_VMOVE,
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

void record(const void *obj, unsigned op, unsigned long long oldc,
    const void *site, const char *tname, const void *slot) noexcept {
    TL &t = tl;
    if(!t.ring) {
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
    unsigned born=0, dead=0, inc=0, dec=0, winc=0, wdec=0, wdead=0, trip=0;
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
        "   weak wINC %u / wDEC %u / wDEAD %u   tripwires %u\n",
        born, dead, inc, dec, winc, wdec, wdead, trip);
    if(wrapped)
        fprintf(stderr, "  ** %u ring(s) have wrapped (%llu events recorded) --"
            " older events were EVICTED, so an unbalanced ledger here says"
            " nothing about tracer coverage **\n",
            wrapped, (unsigned long long)g_writes.load(std::memory_order_relaxed));
    else
        fprintf(stderr, "  no ring has wrapped -- this history is complete\n");
    for(unsigned i = 0; i < n; ++i) {
        const Ev &e = out[i];
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
