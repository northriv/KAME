/*
 * xnode_ctorthrow_test.cpp
 *
 * Regression test for the constructor-to-factory hand-off used by
 * XNode::createOrphan (XNode::stl_thisCreating), XQConnectorHolder_
 * (s_conCreating) and XStatusPrinter::create (s_statusPrinterCreating).
 *
 * The pattern: a constructor pushes shared_ptr(this) onto a stack so that
 * constructors can use shared_from_this(); the factory pops the back and
 * returns it as the owner.  The pairing is POSITIONAL, and it silently
 * assumed every constructor completes.
 *
 * What a throwing constructor does to it (2026-07-31; the STM starvation
 * throw made this reachable from node constructors, which run transactions
 * by design, and it was always reachable via XKameError / bad_alloc):
 *
 *   1. `new T` unwinds: the base destructor runs and the MEMORY IS FREED,
 *      while the pushed shared_ptr keeps a refcount on it.  The entry is now
 *      DANGLING AND OWNING.
 *   2. Popping that entry destroys the last owner -> the deleter runs on
 *      freed memory: DOUBLE FREE.
 *   3. Leaving it makes the next factory call adopt freed memory and hand it
 *      to the caller as a live object: USE AFTER FREE.
 *
 * The fix, mirrored here: capture the stack depth before construction; on an
 * exception, neutralise every entry the failed construction left by leaking
 * its control block (the refcount never reaches zero, so the deleter never
 * runs on freed memory — a few dozen bytes on an error path against a double
 * free); and verify the pop by IDENTITY against the pointer the allocation
 * returned rather than trusting the position.
 *
 * Deliberately self-contained (no Qt, no XNode), like xnode_typename_test:
 * the real createOrphan is a template in a Qt-dependent header, so this pins
 * the algorithm the three sites share.  Run with -DXNODE_CTORTHROW_LEGACY to
 * exercise the old logic and watch cases 2 and 3 fail.
 */

#include <cstdio>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>

// ── Instrumentation ──────────────────────────────────────────────────────────

static int g_live = 0;          // constructed - destructed
static int g_destructs = 0;     // total destructor calls
static int g_deleter_calls = 0; // shared_ptr deleters that actually ran

struct Base {
    // The hand-off stack, exactly as in XNode / XQConnector.
    static std::deque<std::shared_ptr<Base>> s_creating;
    std::string tag;
    bool alive = true;
    explicit Base(std::string t) : tag(std::move(t)) {
        ++g_live;
        // shared_ptr(this) with an instrumented deleter so a double free is
        // observable as a second deleter call rather than a crash.
        s_creating.push_back(std::shared_ptr<Base>(this, [](Base *p) {
            ++g_deleter_calls;
            delete p;
        }));
    }
    virtual ~Base() { --g_live; ++g_destructs; alive = false; }
};
std::deque<std::shared_ptr<Base>> Base::s_creating;

//! A node whose own constructor body throws AFTER the base pushed — the
//! shape of a driver constructor whose iterate_commit throws.
struct Throwing : public Base {
    explicit Throwing(std::string t) : Base(std::move(t)) {
        throw std::runtime_error("ctor failed");
    }
};
struct Good : public Base {
    explicit Good(std::string t) : Base(std::move(t)) {}
};

// ── The factory, in both flavours ────────────────────────────────────────────

template <class T>
static std::shared_ptr<T> createOrphan(const char *tag) {
#ifdef XNODE_CTORTHROW_LEGACY
    new T(tag);                             // no guard: the old code
    std::shared_ptr<T> p =
        std::dynamic_pointer_cast<T>(Base::s_creating.back());
    Base::s_creating.pop_back();
    return p;
#else
    const size_t depth = Base::s_creating.size();
    T *raw;
    try {
        raw = new T(tag);
    }
    catch(...) {
        // Dangling AND owning: neutralise by leaking the control block.
        while(Base::s_creating.size() > depth) {
            new std::shared_ptr<Base>(std::move(Base::s_creating.back()));
            Base::s_creating.pop_back();
        }
        throw;
    }
    if(Base::s_creating.empty() ||
        (Base::s_creating.back().get() != static_cast<Base *>(raw)))
        throw std::runtime_error("creation stack desynchronised");
    std::shared_ptr<T> p =
        std::dynamic_pointer_cast<T>(Base::s_creating.back());
    Base::s_creating.pop_back();
    return p;
#endif
}

// ── Cases ────────────────────────────────────────────────────────────────────

int main() {
    int failures = 0;
    auto check = [&](const char *what, bool ok) {
        std::printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
        if( !ok) ++failures;
    };

    // 1. Baseline: a healthy creation still works and the stack stays empty.
    {
        auto p = createOrphan<Good>("first");
        check("healthy creation returns the object", p && (p->tag == "first"));
        check("stack drained after a healthy creation",
              Base::s_creating.empty());
    }

    // 2. A throwing constructor must not leave a poisoned entry — and must
    //    not double-free the object `new` already destroyed.
    const int destructs_before = g_destructs;
    // Deltas, not absolutes: case 1's healthy object was legitimately deleted
    // when its owner went out of scope (the first version of this test
    // compared against zero and failed on its own baseline).
    const int deleters_before_throw = g_deleter_calls;
    bool threw = false;
    try { createOrphan<Throwing>("bad"); }
    catch(std::runtime_error &) { threw = true; }
    check("the constructor's exception propagates", threw);
    check("object destroyed exactly once by the failed new",
          g_destructs == destructs_before + 1);
    check("no deleter ran on the freed memory (no double free)",
          g_deleter_calls == deleters_before_throw);
    check("stack left with no adoptable entry", Base::s_creating.empty());

    // 3. The next creation must get ITS OWN object, not the dead one.
    {
        auto p = createOrphan<Good>("after");
        check("next creation returns a live object", p && p->alive);
        check("...and it is the new one, not the abandoned one",
              p && (p->tag == "after"));
    }

    // 4. Nested: a parent that creates a child and then throws leaves both
    //    entries; depth-based cleanup must remove exactly them.
    struct Parent : public Base {
        explicit Parent(std::string t) : Base(std::move(t)) {
            child = createOrphan<Good>("child");   // completes and pops
            new Good("orphaned-grandchild");       // pushes, never popped
            throw std::runtime_error("parent failed");
        }
        std::shared_ptr<Good> child;
    };
    int deleters_before = g_deleter_calls;
    try { createOrphan<Parent>("parent"); } catch(std::runtime_error &) {}
    // The parent's completed child WAS legitimately owned and released when
    // the parent unwound, so allow that one deleter and no more.
    deleters_before += 1;
    check("nested leftovers cleaned to the entry depth",
          Base::s_creating.empty());
    check("no deleter ran on freed memory during nested cleanup",
          g_deleter_calls == deleters_before);
    {
        auto p = createOrphan<Good>("last");
        check("factory still usable after a nested failure",
              p && p->alive && (p->tag == "last"));
    }

    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
