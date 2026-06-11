// §36b orphan-chain prototype — a SELF-REFERENTIAL INTRUSIVE node in a
// lock-free SINGLY-LINKED chain, the structure the pool's orphan-chunk
// reuse became (`m_orphan_next` in PoolAllocator).
//
// NAMING NOTE: this file was originally `atomic_intrusive_dll_test.cpp`
// (DLL = doubly-linked) carrying a `m_prev` field as a "raw revalidated
// back-hint".  In practice nothing ever READ that field — it was dead
// store weight — and the production orphan chain has no prev at all:
// Treiber-style push at head, forward-walking scrub, no back-traversal.
// The only real DLL in the allocator is the per-thread owner-mode chunk
// list (`m_dll_prev`/`m_dll_next`), a separate single-thread structure
// unrelated to this PoC.  Renamed + `m_prev` removed so the prototype
// mirrors the production singly-linked chain exactly.
//
// This is the standalone, stress-testable proof-of-concept (mirrors
// atomic_intrusive_dispose_test.cpp's role for the disposer) that pins down
// the two NOVEL mechanisms the allocator integration depends on, in isolation,
// BEFORE they are entangled with allocator.cpp's ~3000 lines:
//
//   (1) SELF-REFERENTIAL INTRUSIVE mode.  An `atomic_shared_ptr<Node>` link
//       lives INSIDE `Node` itself.  Naively, instantiating the member while
//       `Node` is incomplete makes `ref_traits<Node>` take the incomplete-T
//       SFINAE fallback (atomic_smart_ptr.h:292-339) → NON-intrusive (separate
//       control block, `delete`-based disposal), defeating the whole intrusive
//       design AND making `T::atomic_intrusive_dispose` unreachable.  The fix:
//       an EXPLICIT `ref_traits<Node>` specialisation, visible BEFORE the
//       self-referential member, that hard-codes `is_intrusive = true` /
//       `Ref = Node`.  An explicit specialisation outranks both the primary
//       and the `void_t<sizeof(T)>` partial spec, so the member sees intrusive
//       even though `Node` is incomplete (the member only needs the tagged-ptr
//       layout; the refcnt ops are instantiated later, when `Node` is complete).
//       This is exactly what the pool's chunk (`PoolAllocator<...>`) will do via
//       a partial `ref_traits<PoolAllocator<A,F,D>>` specialisation.
//
//   (2) LOCK-FREE SINGLY-LINKED CHAIN under SMR + CUSTOM DISPOSER.  Nodes are
//       pushed at head (Treiber), popped (claimed) at head, and UNLINKED FROM
//       THE MIDDLE when they go "empty" — the orphan structure's three
//       operations.  No back-link: intrusive opts out of weak refs, and the
//       scrub walks forward only (restarting from head on a lost CAS), so a
//       raw prev would be a pure dead store.  The tagged-ref SMR of
//       `atomic_shared_ptr` keeps every node mapped while any thread still
//       references it, so a concurrent unlink can never free a node out from
//       under a traversing/popping thread.
//
// SCOPE OF THIS TEST: MEMORY SAFETY of (1)+(2), not list LINEARIZABILITY.
// It asserts, at quiescence after a final drain:
//   * created == disposed   (no leak, no double-free)
//   * class_delete == 0     (the in-region chunk's `::operator delete` stand-in
//                            is NEVER reached — disposal always took the custom
//                            `atomic_intrusive_dispose` path)
//   * bad == 0              (no use-after-dispose / double-dispose / corruption)
// The LINEARIZABILITY of the concurrent push/pop/mid-unlink protocol is left to
// the TLA+/GenMC verification (see kamestm/tests/tlaplus/, VERIFICATION.md);
// this harness is the realizability + memory-safety companion to that.
//
// Self-contained: header-only atomic_smart_ptr.h + std::thread.
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

struct Node;   // forward declaration so the explicit ref_traits spec below
               // is visible BEFORE the self-referential member instantiates
               // atomic_shared_ptr<Node> (which would otherwise take the
               // incomplete-T → non-intrusive SFINAE fallback).

// (1) force_intrusive_ref opt-in — force INTRUSIVE for the self-referential
// `Node` WITHOUT a `sizeof(Node)` completeness probe (which would be a hard
// error mid-definition).  This is the exact mechanism the allocator uses for
// the chunk via `force_intrusive_ref<PoolAllocator<A,F,D>>`.  (A full
// `ref_traits<Node>` specialisation would also work for a concrete type, but
// the allocator's chunk is a class TEMPLATE — only the opt-in trait composes
// there, so the PoC exercises the opt-in path.)
template <>
struct force_intrusive_ref<Node> : std::true_type {};

//! Self-referential intrusive (atomic_countable) stand-in for a pool chunk.
struct Node : atomic_countable {
    atomic_shared_ptr<Node> m_next;        //!< intrusive self-link (SMR-managed)
    std::atomic<int>        m_marked{0};       //!< logical-delete ("emptied")
    std::uint64_t           magic;
    long                    x;

    explicit Node(long v) : magic(ALIVE), x(v) {
        g_created.fetch_add(1, std::memory_order_relaxed);
    }
    ~Node() noexcept {}   // magic stamped DEAD by the disposer first

    //! The chunk's `::operator delete` stand-in — must NEVER fire.
    static void operator delete(void *) noexcept {
        g_class_delete.fetch_add(1, std::memory_order_relaxed);
    }

    //! (§36b) Custom disposer — object STILL LIVE here (a real chunk reads its
    //! size before tearing down).  Verify single disposal, then destruct + free
    //! via the GLOBAL operator delete (matching the global `new`) — never
    //! Node::operator delete.  `~Node()` drops `m_next`, releasing the
    //! successor's structural ref (cascades, but the test drains head-first so
    //! the chain is short at any drop).
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

// Compile-time proof: the self-referential type still selects intrusive +
// custom disposer (the explicit spec beat the incomplete-T fallback).
static_assert(ref_traits<Node>::is_intrusive,
              "self-referential Node must stay intrusive via the explicit spec");
static_assert(std::is_same<ref_traits<Node>::Ref, Node>::value,
              "intrusive Ref must be Node itself (no separate control block)");
static_assert(has_intrusive_dispose<Node>::value,
              "Node must expose the custom atomic_intrusive_dispose hook");
static_assert(!ref_traits<Node>::has_weak,
              "intrusive opts out of weak refs — chain has no back-link by design");

// ---------------------------------------------------------------------------
// Regression guard — self-referential intrusive CLASS TEMPLATE.
//
// The allocator's chunk (`PoolAllocator<ALIGN,FS,DUMMY>`) is a class TEMPLATE
// whose embedded object holds an `atomic_shared_ptr<self>` orphan-chain link.
// That triggers a circularity the concrete `Node` above does NOT exercise:
// instantiating `atomic_shared_ptr<TNode<...>>` needs the refcount type, and
// taking it from `Ref::Refcnt` would force a qualified-name lookup that
// INSTANTIATES the still-incomplete specialisation → completes it → lays out
// its own `atomic_shared_ptr<self>` member → circular.  GCC rejects this as a
// hard error (clang is lenient); the fix takes `Refcnt` from the (complete)
// trait instead.  This template node keeps that path covered on GCC so the
// chunk integration cannot regress it again.  (Concrete intrusive types escape
// via a complete base's `Refcnt`, so `Node` alone would not have caught it.)
template <int K> struct TNode;
template <int K> struct force_intrusive_ref<TNode<K>> : std::true_type {};
template <int K>
struct TNode {
    typedef uintptr_t Refcnt;
    atomic<Refcnt> refcnt{1};                 // own member (no complete base);
                                              // 1 = adoptable by local_shared_ptr(new)
    atomic_shared_ptr<TNode> m_next;          // injected-class-name self-ref
    long v = 0;
    static void atomic_intrusive_dispose(TNode *p) noexcept {
        p->~TNode();
        ::operator delete(static_cast<void *>(p));
    }
};
static_assert(ref_traits<TNode<1>>::is_intrusive, "template node forced intrusive");
static_assert(std::is_same<ref_traits<TNode<1>>::Ref, TNode<1>>::value,
              "template node Ref is itself");
static_assert(has_intrusive_dispose<TNode<1>>::value, "template node has disposer");

//! Exercise the template path at runtime (so it is not dead code the compiler
//! could skip instantiating): push two onto a chain head, then drain.
static int template_self_ref_smoke() {
    atomic_shared_ptr<TNode<1>> head;
    head.reset(new TNode<1>());
    {
        local_shared_ptr<TNode<1>> a(head);
        local_shared_ptr<TNode<1>> b(new TNode<1>());
        b->m_next = a;
        head.compareAndSwap(a, b);
    }
    head.reset();
    return 0;
}

// ---------------------------------------------------------------------------
// Lock-free orphan chain — head atomic, intrusive forward next, no back-link.
// ---------------------------------------------------------------------------

static atomic_shared_ptr<Node> g_head;

//! Treiber push at head.  `n` is the only ref to a fresh node.
static void push_head(const local_shared_ptr<Node> &n) {
    local_shared_ptr<Node> old(g_head);
    for(;;) {
        n->m_next = old;                       // n.next = current head
        if(g_head.compareAndSwap(old, n)) break;   // old reloaded on failure
    }
}

//! Treiber pop at head (claim).  Returns null when empty.
static local_shared_ptr<Node> pop_head() {
    local_shared_ptr<Node> old(g_head);
    for(;;) {
        if(!old) return local_shared_ptr<Node>();
        local_shared_ptr<Node> nxt(old->m_next);
        if(g_head.compareAndSwap(old, nxt)) return old;   // claimed
        // old reloaded by compareAndSwap on failure; loop.
    }
}

//! Mark a node logically empty (the "m_flags_packed hit 0" trigger).
static void mark_empty(Node *victim) {
    if(victim) victim->m_marked.store(1, std::memory_order_release);
}

//! Physically splice every MARKED node out of the chain (predecessor.next CAS).
//! Memory-safe under SMR regardless of races: a losing CAS just leaves the
//! marked node in place for a later scrub; the node is held alive meanwhile.
//! (Linearizability of this against concurrent push/pop is the TLA+ concern.)
static void scrub() {
    local_shared_ptr<Node> pred;               // null == "predecessor is head"
    local_shared_ptr<Node> cur(g_head);
    while(cur) {
        local_shared_ptr<Node> nxt(cur->m_next);
        if(cur->m_marked.load(std::memory_order_acquire)) {
            bool ok;
            if(!pred)
                ok = g_head.compareAndSet(cur, nxt);     // unlink at head
            else
                ok = pred->m_next.compareAndSet(cur, nxt);
            if(ok) {
                cur = nxt;                     // advance; pred unchanged
                continue;
            }
            // CAS lost (chain changed) — restart from head.
            pred.reset();
            cur = local_shared_ptr<Node>(g_head);
            continue;
        }
        pred = cur;
        cur = nxt;
    }
}

static void check_alive(const local_shared_ptr<Node> &q) {
    if(q && q->magic != ALIVE)
        g_bad.fetch_add(1, std::memory_order_relaxed);
}

static void worker(int tid, int iters) {
    unsigned r = (unsigned)tid * 2654435761u + 1u;
    auto rnd = [&]() { r ^= r << 13; r ^= r >> 17; r ^= r << 5; return r; };
    // Push and pop are BALANCED (2:2) so the list stays bounded — this is a
    // memory-safety test, not a throughput test, and an unbounded list would
    // make the O(N) scrub superlinear.  Mark + scrub (the mid-unlink path) are
    // exercised at lower frequency, and scrub is rate-limited per worker.
    for(int i = 0; i < iters; i++) {
        switch(rnd() % 8u) {
        case 0:                                // push a fresh orphan
        case 1:
            push_head(local_shared_ptr<Node>(new Node((long)i)));
            break;
        case 2:                                // claim (pop) and use, then drop
        case 3: {
            local_shared_ptr<Node> q(pop_head());
            check_alive(q);
            break;
        }
        case 4: {                              // load head + walk a few links
            local_shared_ptr<Node> q(g_head);
            for(int k = 0; q && k < 4; k++) {
                check_alive(q);
                q = local_shared_ptr<Node>(q->m_next);
            }
            break;
        }
        case 5:
        case 6: {                              // mark head's neighbourhood empty
            local_shared_ptr<Node> q(g_head);
            if(q) q = local_shared_ptr<Node>(q->m_next);
            if(q) { check_alive(q); mark_empty(q.get()); }
            break;
        }
        case 7:                                // scrub marked nodes out (rarer)
            scrub();
            break;
        }
    }
}

static int single_thread_sanity() {
    long c0 = g_created, d0 = g_disposed;
    {
        // Build 5, mark the middle one, scrub it out, then drain the rest.
        for(int i = 0; i < 5; i++)
            push_head(local_shared_ptr<Node>(new Node(i)));
        {   // mark the 3rd-from-head
            local_shared_ptr<Node> q(g_head);
            q = local_shared_ptr<Node>(q->m_next);
            q = local_shared_ptr<Node>(q->m_next);
            mark_empty(q.get());
        }
        scrub();                               // physically unlink the marked one
        for(;;) {                              // drain the remaining 4
            local_shared_ptr<Node> q(pop_head());
            if(!q) break;
            check_alive(q);
        }
    }
    long made = g_created - c0, gone = g_disposed - d0;
    bool ok = (made == gone) && made == 5 && g_class_delete == 0 && g_bad == 0;
    std::printf("  [%s] single-thread: created=%ld disposed=%ld class_delete=%ld bad=%ld\n",
                ok ? "ok" : "BAD", made, gone, (long)g_class_delete, (long)g_bad);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    int T = 8, ITERS = 200000;
    if(argc > 1) T = atoi(argv[1]);
    if(argc > 2) ITERS = atoi(argv[2]);

    int fails = single_thread_sanity();
    fails += template_self_ref_smoke();   // self-referential intrusive TEMPLATE

    std::printf("[mt] threads=%d iters=%d\n", T, ITERS);
    std::vector<std::thread> ts;
    for(int t = 0; t < T; t++) ts.emplace_back(worker, t, ITERS);
    for(auto &th : ts) th.join();

    // Final drain: physically remove all marked, then pop everything remaining
    // and clear the head — at quiescence every created node must be disposed.
    scrub();
    for(;;) {
        local_shared_ptr<Node> q(pop_head());
        if(!q) break;
    }
    g_head.reset();

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
