// §36b dedicated-mode test — INTRUSIVE atomic_shared_ptr with a CUSTOM disposer.
//
// The orphan-chunk reuse (§36b) parks a pool-region chunk in an
// atomic_shared_ptr<chunk> whose refcnt-0 disposal must run the chunk's own
// teardown (deallocate_chunk) — NOT ::operator delete, which the placement-
// new'd, region-resident chunk must never reach.  That is enabled by the
// intrusive control-block mode (`atomic_countable`) plus the opt-in
// `T::atomic_intrusive_dispose(T*)` hook (added to atomic_smart_ptr.h).
//
// This test pins that mode down under heavy concurrency.  A heap-backed stand-
// in `Node` (intrusive + custom disposer) is hammered through atomic_shared_ptr
// store / load / swap / CAS / reset by many threads.  It asserts that the
// tagged-local-ref SMR and the custom disposer compose correctly:
//
//   (1) EXACTLY-ONCE disposal — created == disposed at quiescence (no leak,
//       no double-free).  The disposer paints a magic word and flags any
//       second disposal / corruption (`bad`).
//   (2) The CUSTOM disposer is taken on EVERY release — the class
//       `operator delete` (which the chunk's ::operator delete stands in for)
//       is NEVER called (`class_delete == 0`).
//   (3) No use-after-dispose — every acquired pointer's magic == ALIVE.
//
// Self-contained: header-only atomic_smart_ptr.h + std::thread.  The intrusive
// mode allocates NO control block, so this exercises the §36b path with zero
// pool involvement.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "atomic_smart_ptr.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

static std::atomic<long> g_created{0};
static std::atomic<long> g_disposed{0};
static std::atomic<long> g_class_delete{0};   // must stay 0
static std::atomic<long> g_bad{0};            // double-dispose / corruption / UAF

static const std::uint64_t ALIVE = 0xA11FE10A11FE10ULL;
static const std::uint64_t DEAD  = 0xDEADDEADDEADDEADULL;

//! Intrusive (atomic_countable) stand-in for a pool chunk, with a custom
//! disposer that runs INSTEAD of ::operator delete when the refcnt hits 0.
struct Node : atomic_countable {
    std::uint64_t magic;
    long          x;
    explicit Node(long v) : magic(ALIVE), x(v) {
        g_created.fetch_add(1, std::memory_order_relaxed);
    }
    ~Node() noexcept {}   // magic stamped DEAD by the disposer before this

    //! If the smart-ptr ever routed disposal through `delete p` (the path the
    //! chunk must avoid), this fires — it must stay 0.
    static void operator delete(void *) noexcept {
        g_class_delete.fetch_add(1, std::memory_order_relaxed);
    }

    //! (§36b) Custom disposer — object is STILL LIVE here (so a real chunk
    //! could read its size before tearing down).  Verify single disposal,
    //! then destruct + free via the GLOBAL operator delete (matching the
    //! global `new` below) — never Node::operator delete.
    static void atomic_intrusive_dispose(Node *p) noexcept {
        if(p->magic != ALIVE) {                 // already disposed / corrupted
            g_bad.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        p->magic = DEAD;
        g_disposed.fetch_add(1, std::memory_order_relaxed);
        p->~Node();
        ::operator delete(static_cast<void *>(p));
    }
};

// Compile-time proof that this is the §36b mode (intrusive + custom disposer).
static_assert(ref_traits<Node>::is_intrusive,
              "Node must select the intrusive (atomic_countable) control block");
static_assert(has_intrusive_dispose<Node>::value,
              "Node must expose the custom atomic_intrusive_dispose hook");

static const int SLOTS = 8;
static atomic_shared_ptr<Node> g_slots[SLOTS];

static void check_alive(const local_shared_ptr<Node> &q) {
    if(q && q->magic != ALIVE)
        g_bad.fetch_add(1, std::memory_order_relaxed);
}

static void worker(int tid, int iters) {
    unsigned r = (unsigned)tid * 2654435761u + 1u;
    auto rnd = [&]() { r ^= r << 13; r ^= r >> 17; r ^= r << 5; return r; };
    for(int i = 0; i < iters; i++) {
        local_shared_ptr<Node> p(new Node((long)i));
        int s = (int)(rnd() % (unsigned)SLOTS);
        switch(rnd() % 6u) {
        case 0:                              // store fresh into the atomic slot
            g_slots[s].reset(new Node((long)(i ^ 0x5a)));
            break;
        case 1: {                            // load (acquire) + use
            local_shared_ptr<Node> q(g_slots[s]);
            check_alive(q);
            break;
        }
        case 2:                              // swap local <-> atomic
            p.swap(g_slots[s]);
            break;
        case 3:                              // clear the slot
            g_slots[s].reset();
            break;
        case 4:                              // CAS our p into the slot
            for(local_shared_ptr<Node> q(g_slots[s]);;) {
                check_alive(q);
                if(g_slots[s].compareAndSwap(q, p)) break;
            }
            break;
        case 5:                              // weak-CAS — RE-LOAD the expected
            for(;;) {                        // each retry (weak CAS keeps oldr
                local_shared_ptr<Node> q(g_slots[s]);   // const: no auto-update)
                check_alive(q);
                if(g_slots[s].compareAndSetWeak(q, p)) break;
            }
            break;
        }
    }
}

static int single_thread_sanity() {
    long c0 = g_created, d0 = g_disposed;
    {
        atomic_shared_ptr<Node> a;
        a.reset(new Node(1));                // a holds Node#1
        local_shared_ptr<Node> q(a);         // acquire #1  (a, q)
        check_alive(q);
        local_shared_ptr<Node> r(new Node(2));   // r holds Node#2
        r.swap(a);                           // local.swap(atomic): a=#2, r=#1
        a.reset();                           // drop #2's only ref -> #2 disposed
        // q and r both hold #1; scope end drops both -> #1 disposed
    }                                        // scope end: all refs gone
    long made = g_created - c0, gone = g_disposed - d0;
    bool ok = (made == gone) && made == 2 && g_class_delete == 0 && g_bad == 0;
    std::printf("  [%s] single-thread: created=%ld disposed=%ld class_delete=%ld bad=%ld\n",
                ok ? "ok" : "BAD", made, gone, (long)g_class_delete, (long)g_bad);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    int T = 8, ITERS = 200000;
    if(argc > 1) T = atoi(argv[1]);
    if(argc > 2) ITERS = atoi(argv[2]);

    int fails = single_thread_sanity();

    std::printf("[mt] threads=%d iters=%d slots=%d\n", T, ITERS, SLOTS);
    std::vector<std::thread> ts;
    for(int t = 0; t < T; t++) ts.emplace_back(worker, t, ITERS);
    for(auto &th : ts) th.join();
    for(int s = 0; s < SLOTS; s++) g_slots[s].reset();   // drain all slots

    long created = g_created, disposed = g_disposed;
    long classd = g_class_delete, bad = g_bad;
    std::printf("  created=%ld disposed=%ld class_delete=%ld bad=%ld\n",
                created, disposed, classd, bad);

    bool mt_ok = (created == disposed) && classd == 0 && bad == 0;
    if(!mt_ok) {
        if(created != disposed)
            std::printf("FAIL: leak/double-free (created %ld != disposed %ld)\n",
                        created, disposed);
        if(classd != 0)
            std::printf("FAIL: ::operator delete called %ld times (must be 0)\n", classd);
        if(bad != 0)
            std::printf("FAIL: %ld double-dispose/UAF/corruption events\n", bad);
        fails++;
    }
    std::printf(fails == 0 ? "\nPASS\n" : "\nFAIL\n");
    return fails ? 1 : 0;
}
