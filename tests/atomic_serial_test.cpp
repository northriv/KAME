// atomic_shared_ptr's opt-in serial (atomic_serial_traits): a CAS whose old
// value was loaded before the word was written -- even back to the same
// pointer -- fails.
//
// Single-threaded and deterministic: the other thread of an ABA is played
// inline between the load and the CAS.  The same script runs on a node type
// without the serial, where the stale CAS succeeds -- the pool allocator's
// orphan-chain bug (orphan_chain_pop: load old, load old->m_orphan_next, CAS;
// another thread pops `old`, adopts it and pushes it back in between).  The
// concurrent protocol itself is model-checked in
// tlaplus/OrphanChain_aba.tla; this checks that atomic_shared_ptr implements
// the serial the model assumes.
//
// Header-only (atomic_smart_ptr.h); no pool, no threads.

#include "atomic_smart_ptr.h"

#include <cstdint>
#include <cstdio>
#include <new>

static int g_failures = 0;
static long g_disposed = 0;

#define CHECK(cond) do { if( !(cond)) { \
    std::fprintf(stderr, "%s:%d: CHECK(%s) failed [%s]\n", __FILE__, __LINE__, #cond, tag); \
    ++g_failures; } } while(0)

template <bool SERIAL> struct SNode;
template <bool SERIAL> struct force_intrusive_ref<SNode<SERIAL> > : std::true_type {};

// Every SNode<true> sits 0x40 into a 4 KiB-aligned block, so its low 12 bits
// are fixed (0x40): room for the 3-bit local refcount and a 9-bit serial.
template <> struct atomic_serial_traits<SNode<true> > {
    static constexpr unsigned LOW_BITS = 12;
    static constexpr uintptr_t LOW_VALUE = 0x40;
};

constexpr int BLOCKS = 8;
alignas(4096) static unsigned char g_blocks[2][BLOCKS][4096];

template <bool SERIAL>
struct SNode : atomic_countable {
    atomic_shared_ptr<SNode> m_next;
    int id;
    explicit SNode(int i) : id(i) {}

    // The bytes below the node read as "refcount 1", so a word dereferenced
    // without decoding (serial bits taken for address bits) looks unique and
    // gets disposed early -- caught as a double disposal.
    static SNode *make(int i) {
        uintptr_t *below = reinterpret_cast<uintptr_t *>(&g_blocks[SERIAL][i][0]);
        for(unsigned k = 0; k < 0x40 / sizeof(uintptr_t); ++k) below[k] = 1;
        return new(&g_blocks[SERIAL][i][0x40]) SNode(i);
    }
    // In-place storage: run the destructor (drops m_next), never free.
    static void atomic_intrusive_dispose(SNode *p) noexcept {
        if(p->id < 0) { ++g_failures; std::fprintf(stderr, "double disposal\n"); return; }
        ++g_disposed;
        p->id = -p->id;
        p->~SNode();
    }
};

static_assert(atomic_serial_on<SNode<true> > && !atomic_serial_on<SNode<false> >, "");

template <bool SERIAL>
static void run(const char *tag) {
    using N = SNode<SERIAL>;
    using L = local_shared_ptr<N>;
    const long disposed0 = g_disposed;
    {
        atomic_shared_ptr<N> head;
        L a(N::make(1)), b(N::make(2)), c(N::make(3));

        // head -> a -> b
        a->m_next = b;
        {
            L h(head);
            CHECK(head.compareAndSet(h, a));
        }

        // --- the orphan chain's ABA, step by step ---------------------------
        L old(head);                           // P: load the head (a)
        L q(head);                             // Q: pop a ...
        L qn(q->m_next);
        CHECK(head.compareAndSet(q, qn));      //    head -> b
        q->m_next = L();                       //    a leaves the chain
        L nxt(old->m_next);                    // P: load a's successor: null now
        {
            L h(head);                         // Q: ... adopt a, exit, push it back
            q->m_next = h;                     //    a -> b
            CHECK(head.compareAndSet(h, q));   //    head -> a -> b
        }
        const bool stale_cas = head.compareAndSet(old, nxt);   // P
        if(SERIAL) {
            CHECK( !stale_cas);                // the word was written since `old`
            L h(head);
            CHECK(h.get() == a.get() && h->m_next == b);
        }
        else {
            CHECK(stale_cas);                  // the bug: b is cut off the chain
            CHECK( !head);
            L h(b);                            // put the chain back for the rest
            L e(head);
            CHECK(head.compareAndSet(e, a));
        }

        // compareAndSwap refreshes the old value and its serial, so a retry
        // from it succeeds.
        {
            L o(head), x(c);
            CHECK(head.compareAndSwap(nxt, x) == false);   // nxt was never the head
            CHECK(nxt.get() == a.get());                     // ... now it holds the head
            CHECK(head.compareAndSwap(nxt, x));              // and is current
            L h(head);
            CHECK(h.get() == c.get());
            CHECK(head.compareAndSet(h, o));                 // back to a
        }

        // Loads alone (tag acquire/release only) leave the serial alone.
        {
            L o(head);
            L r1(head), r2(head), r3(head);
            L copy = o;                        // a copy carries the serial too
            CHECK(head.compareAndSet(copy, o));   // same pointer, unchanged word
        }

        // An empty word keeps its serial: a stale empty load fails.
        {
            L o(head);
            head.reset();                      // store: serial + 1
            CHECK( !head);
            L e1(head);                        // empty, current
            head = a;                          // store
            head.reset();                      // store: empty again
            if(SERIAL)
                CHECK( !head.compareAndSet(e1, b));   // empty then, empty now; written since
            else
                CHECK(head.compareAndSet(e1, b));
            L e2(head);
            if(SERIAL) {
                CHECK( !head);
                CHECK(head.compareAndSet(e2, a));
            }
            L h(head);
            CHECK(h);
        }

        // A word keeps its serial at rest, so destroying a non-null one must
        // decode it: a head that goes out of scope after several stores, and
        // (below) a node disposed while its link still points at the next.
        {
            atomic_shared_ptr<N> h2;
            h2 = a;
            h2 = b;
            h2 = a;
        }
        a->m_next = b;                         // left set: a's disposal drops it
        b->m_next = L();
        c->m_next = L();
        head.reset();
        CHECK(g_disposed == disposed0);        // the locals still hold all three
    }
    CHECK(g_disposed - disposed0 == 3);        // every node disposed exactly once
}

int main() {
    run<false>("no serial");
    run<true>("serial");
    if(g_failures) {
        std::fprintf(stderr, "atomic_serial_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("atomic_serial_test: OK\n");
    return 0;
}
