/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/
#ifndef ATOMIC_SMART_PTR_H_
#define ATOMIC_SMART_PTR_H_

#include "atomic.h"  // integral atomic<T> specialization + atomic_mfence.h
                     // (cyclic include — atomic.h's ATOMIC_H_ guard
                     // is set before it #include's us; the top half
                     // is fully expanded by then.)
#include <functional>
#include <utility>
#include <type_traits>
#include <cstdint>   // uintptr_t — tagged-pointer local refcount (not transitive under libstdc++ 14+)
#include <assert.h>

//! Trait to disable load_shared_() for specific types at compile time.
//! To disable for type T, add `using load_shared_disabled_tag = void;` to T.
//! Detected via SFINAE — no template specialization required.
namespace detail_asp {
    template <typename T, typename = void>
    struct load_shared_enabled_impl : std::true_type {};
    template <typename T>
    struct load_shared_enabled_impl<T, typename std::conditional<true, void,
        typename T::load_shared_disabled_tag>::type> : std::false_type {};
}
template <typename T>
struct load_shared_enabled : detail_asp::load_shared_enabled_impl<T> {};

#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
//if defined as 0, backoff by pause4spin()[=__mm_spin/yield] will be completely killed.
//if > 0, 2^(retry) spin count will by divided by BACKOFF_IN_ATOMIC_SMART_PTR.
    #define BACKOFF_IN_ATOMIC_SMART_PTR 0 //disabled by default, in accord with our tests for ARM64 and high-core-count x86_64.
#endif

//! \brief This is an atomic variant of \a std::unique_ptr.
//! An instance of atomic_unique_ptr can be shared among threads by the use of \a swap(\a _shared_target_).\n
//! Namely, it is destructive reading.
//! Use atomic_shared_ptr when the pointer is required to be shared among scopes and threads.\n
//! This implementation relies on an atomic-swap machine code, e.g. lock xchg, or std::atomic.
//! \sa atomic_shared_ptr, atomic_unique_ptr_test.cpp
template <typename T>
class atomic_unique_ptr {
    typedef T* t_ptr;
public:
    atomic_unique_ptr() noexcept : m_ptr(nullptr) {}

    explicit atomic_unique_ptr(t_ptr t) noexcept : m_ptr(t) {}

    ~atomic_unique_ptr() {delete (t_ptr)m_ptr;}

    void reset(t_ptr t = nullptr) noexcept {
        t_ptr old = m_ptr.exchange(t);
        delete old;
    }
    //! \param[in,out] x \p x is atomically swapped.
    //! Nevertheless, this object is not atomically replaced.
    //! That is, the object pointed by "this" must not be shared among threads.
    void swap(atomic_unique_ptr &x) noexcept {
        m_ptr = x.m_ptr.exchange(m_ptr);
    }

    bool operator!() const noexcept {return !(t_ptr)m_ptr;}
    operator bool() const noexcept {return (t_ptr)m_ptr;}

    //! This function lacks thread-safety.
    T &operator*() const noexcept { assert((t_ptr)m_ptr); return (T &) *(t_ptr)m_ptr;}

    //! This function lacks thread-safety.
    t_ptr operator->() const noexcept { assert((t_ptr)m_ptr); return (t_ptr)m_ptr;}

    //! This function lacks thread-safety.
    t_ptr get() const noexcept { return (t_ptr )m_ptr;}

    atomic_unique_ptr(const atomic_unique_ptr &) = delete;
    atomic_unique_ptr& operator=(const atomic_unique_ptr &) = delete;

private:
    atomic<t_ptr> m_ptr;
};

//! Common non-template base for weak-capable control blocks.  Layout:
//! `[refcnt | weak_refcnt]` at the start of every gref that supports
//! `local_weak_ptr<T>`.  Both `gref_<T>` (separate alloc, default) and
//! `gref_weakable_<T>` (emplaced) inherit from this, enabling
//! `local_weak_ptr<T>` to store a type-erased `gref_weak_base_*`
//! without needing T to be complete at class-definition time.
//=============================================================================
// §biased refcount  —  OPT-IN, PER-TYPE.  ***Auditing the lock-free core?  SKIP
// this whole section and every `if constexpr (is_biased_directpublish<T>::value)`
// one-liner below — they fold to NOTHING for every type that does not inherit the
// marker (i.e. ALL types today; nothing opts in), leaving the plain atomic path.***
//
// A type T that inherits `atomic_biased_directpublish` stores its strong refcount
// NEGATED while owner-private (born -1; copy/reset mutate it non-atomically via
// relaxed load+store), and PUBLISH negates it to +count.  ~3.3× on owner-private
// churn (micro).  CONTRACT — sound ONLY for a DIRECT-PUBLISH control block: one
// that is exclusively the value of an `atomic_shared_ptr<T>` slot and is NEVER
// transitively contained in another control block (a `local_shared_ptr<T>` member
// of some other CB).  Marking a transitively-shared type (e.g. the STM's Packet /
// PacketList / Linkage) is UNSOUND (a reader copying the nested handle races the
// owner's non-atomic store).  Verified by GenMC test 10; see
// [[project_biased_refcount_planA]].  No KAME type opts in (measured: no net win).
struct atomic_biased_directpublish {};
template<class T> struct is_biased_directpublish : std::is_base_of<atomic_biased_directpublish, T> {};

//! All biased refcount logic lives here so the hot paths stay a single skippable
//! `if constexpr` line each.  `rc` is the control block's strong refcnt.
static inline void biased_born_(atomic<uintptr_t> &rc) noexcept {                 //!< fresh CB → private -1
    rc.store((uintptr_t)(-(intptr_t)1), std::memory_order_relaxed);
}
static inline void biased_inc_(atomic<uintptr_t> &rc) noexcept {                  //!< copy: ++count
    uintptr_t v = rc.load(std::memory_order_relaxed);
    if((intptr_t)v < 0) rc.store(v - 1, std::memory_order_relaxed);   //!< private: non-atomic
    else                rc.fetch_add(1, std::memory_order_relaxed);   //!< shared: atomic
}
static inline bool biased_dec_is_dead_(atomic<uintptr_t> &rc) noexcept {          //!< reset: --count → dead?
    uintptr_t v = rc.load(std::memory_order_relaxed);
    if((intptr_t)v < 0) { uintptr_t nv = v + 1;                       //!< private: branchless add+cbz
        rc.store(nv, std::memory_order_relaxed); return nv == 0; }
    return rc.decAndTest();                                           //!< shared: atomic acq_rel
}
static inline void biased_publish_(atomic<uintptr_t> &rc) noexcept {              //!< publish: -count → +count
    uintptr_t v = rc.load(std::memory_order_relaxed);
    if((intptr_t)v < 0) rc.store((uintptr_t)(-(intptr_t)v), std::memory_order_release); //!< idempotent on +
}
static inline uintptr_t biased_count_(uintptr_t v) noexcept {                     //!< magnitude (use_count)
    return (intptr_t)v < 0 ? (uintptr_t)(-(intptr_t)v) : v;
}
//=============================================================================

struct gref_weak_base_ {
    typedef uintptr_t Refcnt;
    atomic<Refcnt> refcnt;
    atomic<Refcnt> weak_refcnt;

    //! Try to promote weak → strong (`local_weak_ptr::lock()`).  CAS
    //! `refcnt` from N to N+1 iff N > 0.  Returns true if the object is
    //! still alive.
    bool try_promote() noexcept {
        Refcnt cur = refcnt.load(std::memory_order_relaxed);
        for(;;) {
            if(cur == 0) return false;
            if(refcnt.compare_exchange_weak(cur, cur + 1,
                    std::memory_order_acq_rel, std::memory_order_relaxed))
                return true;
        }
    }
protected:
    gref_weak_base_() noexcept : refcnt(1), weak_refcnt(1) {}
};

//! Non-intrusive control block with weak support (DEFAULT mode): T
//! allocated separately, gref holds a pointer to it.  Supports
//! `local_weak_ptr<T>`.  Two-counter design: `refcnt` (strong refs) +
//! `weak_refcnt` (weak refs + 1 while strong > 0).  Strong → 0
//! destroys T; control block lives until weak → 0.
//! \sa atomic_shared_ptr, local_weak_ptr
template <typename T>
struct atomic_shared_ptr_gref_ : gref_weak_base_ {
    explicit atomic_shared_ptr_gref_(T *p) noexcept : ptr(p) {}
    ~atomic_shared_ptr_gref_() noexcept {
        assert(refcnt == 0);
        //!< Fast path in `release_strong_zero` deletes without decrementing
        //!< when weak_refcnt == 1 (no live weak_ptr).  Slow path leaves 0.
        //!< Both are valid pre-delete states.
        assert(weak_refcnt == 0 || weak_refcnt == 1);
        //!< T already destroyed in release_strong_zero; ptr is nullptr.
    }
    //! The pointer to the object.
    T *ptr;
    typedef uintptr_t Refcnt;

    static void release_strong_zero(atomic_shared_ptr_gref_ *p) noexcept {
        //!< Destroy T first (separate allocation).
        delete p->ptr;
        p->ptr = nullptr;
        //!< Drop the implicit +1 weak.  Fast path: if weak_refcnt == 1
        //!< no live `local_weak_ptr<T>` exists (the "1" IS the implicit),
        //!< so skip the atomic fetch_sub.
        if(p->weak_refcnt.load(std::memory_order_acquire) == 1) {
            delete p;
        }
        else if(p->weak_refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete p;
        }
    }
    static void release_weak_zero(atomic_shared_ptr_gref_ *p) noexcept {
        delete p;
    }

    atomic_shared_ptr_gref_(const atomic_shared_ptr_gref_ &) = delete;
};

//! ===========================================================================
//!  USAGE — picking a control-block mode for atomic_shared_ptr<T> /
//!  local_shared_ptr<T>.  You inherit an (optional) MARKER on T; ref_traits<T>
//!  (further down) reads it and selects the layout.  Nothing else to wire up.
//!
//!    default (inherit nothing) : 2 allocations (T + control block), weak OK
//!        struct T { ... };
//!        local_shared_ptr<T> p(new T(...));
//!
//!    : atomic_emplaced   (== atomic_weakable) : 1 allocation (T embedded), weak OK
//!        struct T : atomic_emplaced { ... };
//!        auto p = make_local_shared<T>(args...);
//!        // emplaced T MUST use make_local_shared(), NOT local_shared_ptr<T>(new T)
//!
//!    : atomic_strictrefonly : 2 allocations, NO weak (saves 8 B + 1 atomic op)
//!        struct T : atomic_strictrefonly { ... };
//!        local_shared_ptr<T> p(new T(...));
//!
//!    : atomic_countable : intrusive — refcnt lives INSIDE T, no separate
//!        control block, fastest hot path, NO weak.  Auto-detected when T is
//!        complete at first use.
//!        struct T : atomic_countable { ... };
//!
//!    self-referential intrusive node (T embeds an atomic_shared_ptr<T> link, so
//!    T is INCOMPLETE at first use -> marker auto-detection cannot see it):
//!    opt in explicitly with `force_intrusive_ref<T>` (below), and give T the
//!    intrusive contract -- `typedef ... Refcnt;` + `atomic<Refcnt> refcnt;`
//!    (and, optionally, a `void atomic_intrusive_dispose() noexcept` method).
//!
//!  The per-mode cost/feature MATRIX is the table just below (next to
//!  `atomic_emplaced`).  Worked examples: tests/atomic_intrusive_dispose_test.cpp
//!  and tests/atomic_intrusive_chain_test.cpp (self-referential).
//! ===========================================================================

//! Opt-out marker: T uses strict reference counting only (no
//! `weak_refcnt`).  Inherit from this to suppress `local_weak_ptr<T>`
//! support — saves 8 bytes per control block and one atomic op at
//! construction.  Cannot be combined with `atomic_emplaced` or
//! `atomic_weakable` (the SFINAE branches are mutually exclusive).
struct atomic_strictrefonly {};

//! Non-intrusive control block WITHOUT weak support: T allocated
//! separately, gref holds a pointer.  Used when `T : atomic_strictrefonly`.
//! \sa atomic_shared_ptr
template <typename T>
struct atomic_shared_ptr_gref_strictrefonly_ {
    explicit atomic_shared_ptr_gref_strictrefonly_(T *p) noexcept : ptr(p), refcnt(1) {}
    //!< Polymorphic T's dispatch via virtual destructor on T* (e.g.
    //!< Payload : virtual ~Payload).  Non-polymorphic T's are
    //!< constructed only with Y == T (static_assert in
    //!< local_shared_ptr).
    ~atomic_shared_ptr_gref_strictrefonly_() noexcept { assert(refcnt == 0); delete ptr; }
    //! The pointer to the object.
    T *ptr;
    typedef uintptr_t Refcnt;
    //! Reference counter.
    atomic<Refcnt> refcnt;

    atomic_shared_ptr_gref_strictrefonly_(const atomic_shared_ptr_gref_strictrefonly_ &) = delete;
};

//! Marker base classes (opt-in).  Empty structs (or trivial — see
//! `atomic_countable` below) — empty base optimisation makes them
//! zero-cost in `sizeof(T)` when applicable.
//!
//! Choosing a mode:
//!
//! | Marker                | Alloc | weak_ptr | get() cost | Use case        |
//! |-----------------------|-------|----------|------------|-----------------|
//! | `atomic_countable`    | 1×    | no       | branchless | hottest types   |
//! | `atomic_emplaced`     | 1×    | yes      | offset+null| weakable hot    |
//! | `atomic_strictrefonly`| 2×    | no       | offset+null| small / cold    |
//! | (none — default)      | 2×    | yes      | offset+null| anything else   |
//!
//! `atomic_weakable` is a back-compat alias for `atomic_emplaced`.
//!
//! All bookkeeping (`Ref` type, deleter, layout) is then driven from
//! `ref_traits<T>` — see below.
struct atomic_emplaced {};
struct atomic_weakable : atomic_emplaced {};

//! Intrusive refcnt base — T inherits and gets a built-in refcnt field.
//! sizeof(T) includes the refcnt; `local_shared_ptr<T>` stores T*
//! directly (no separate control block).  Fastest hot path.
struct atomic_countable {
    atomic_countable() noexcept : refcnt(1) {}
    atomic_countable(const atomic_countable &) noexcept : refcnt(1) {}
    ~atomic_countable() { assert(refcnt == 0); }

    atomic_countable &operator=(const atomic_countable &) = delete;

    typedef uintptr_t Refcnt;
    atomic<Refcnt> refcnt;
};

//! Non-intrusive control block, T embedded + weak refcount — single
//! allocation, supports `local_weak_ptr<T>`.  Used when
//! `T : atomic_emplaced` (including `T : atomic_weakable`).
//! Two-counter design inherited from `gref_weak_base_`:
//!   * `refcnt`      — strong refs (local_shared_ptr / atomic_shared_ptr)
//!   * `weak_refcnt` — weak refs + 1 while strong > 0 ("alive" sentinel)
//! Strong → 0 destroys T; control block lives until weak → 0.
template <typename T>
struct atomic_shared_ptr_gref_weakable_ : gref_weak_base_ {
    typedef uintptr_t Refcnt;
    alignas(T) unsigned char data_storage[sizeof(T)];

    template <typename ...Args>
    explicit atomic_shared_ptr_gref_weakable_(Args&&... args) {
        new ( &data_storage) T(std::forward<Args>(args)...);
    }
    ~atomic_shared_ptr_gref_weakable_() noexcept {
        assert(refcnt == 0);
        //!< Fast path in `release_strong_zero` deletes without decrementing
        //!< when weak_refcnt == 1 (no live weak_ptr).  Slow path leaves 0.
        //!< Both are valid pre-delete states.
        assert(weak_refcnt == 0 || weak_refcnt == 1);
    }

    T *ptr_() noexcept { return reinterpret_cast<T *>( &data_storage); }
    const T *ptr_() const noexcept {
        return reinterpret_cast<const T *>( &data_storage);
    }

    static void release_strong_zero(atomic_shared_ptr_gref_weakable_ *p) noexcept {
        p->ptr_()->~T();
        //!< Drop the implicit +1 weak.  Fast path: if weak_refcnt == 1
        //!< no live `local_weak_ptr<T>` exists (the "1" IS the implicit),
        //!< so skip the atomic fetch_sub.  Safe — no new weak_ptr can
        //!< come into existence (no strong → no copy from shared; no
        //!< existing weak → no copy from weak).
        if(p->weak_refcnt.load(std::memory_order_acquire) == 1) {
            delete p;
        }
        else if(p->weak_refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete p;
        }
    }
    static void release_weak_zero(atomic_shared_ptr_gref_weakable_ *p) noexcept {
        delete p;
    }
    //! try_promote() inherited from gref_weak_base_.

    atomic_shared_ptr_gref_weakable_(const atomic_shared_ptr_gref_weakable_ &) = delete;
};

template <typename X, typename Y, typename Z, typename E> struct atomic_shared_ptr_base;
template <typename X> class atomic_shared_ptr;
template <typename X, typename Y> class local_shared_ptr;
template <typename X> class scoped_atomic_view;
template <typename X> class local_weak_ptr;

//! Compile-time mode selection for atomic_shared_ptr_base.  Encodes
//! "which `Ref` does T use" and the boolean flags that govern the
//! deleter / get() / reset_unsafe paths.  All other logic lives in
//! the single `atomic_shared_ptr_base` template below.
//!
//! Incomplete-T support: a self-referencing struct
//! (`struct N { local_shared_ptr<N> next; };`) instantiates
//! `local_shared_ptr<N>` BEFORE N is complete — at that point
//! `std::is_base_of<atomic_countable, N>` is a hard error rather
//! than `false`.  So gate the marker-detection on `sizeof(T)` via
//! SFINAE: when T is incomplete, all three category flags default
//! to `false` and `Ref = atomic_shared_ptr_gref_<T>` (the default
//! two-alloc gref — its only T usage is `T *ptr;` which works on
//! incomplete T).
//!
//! Trade-off: self-referential types lose the intrusive/emplaced/
//! strict optimisations.  Intrusive (`atomic_countable`) is naturally
//! incompatible via AUTO-detection — the `is_base_of` marker probe needs T
//! complete — but a SELF-REFERENTIAL intrusive type (one that embeds an
//! `atomic_shared_ptr<T>` link inside itself, e.g. a lock-free list/DLL node)
//! can OPT IN explicitly via `force_intrusive_ref<T>` (below), which is
//! consulted WITHOUT requiring T complete.  `atomic_emplaced` just saves the
//! second allocation, so its pay-back is bounded.  The first instantiation of
//! `ref_traits<T>` wins (template-instantiation caching is per-args), so any
//! other user of `local_shared_ptr<T>` in the same TU sees the same decision.

//! (§36b) Opt-in to force the INTRUSIVE control block for a type that the
//! auto-detection cannot see as intrusive because it is INCOMPLETE at the
//! point of first use — specifically a SELF-REFERENTIAL intrusive type holding
//! an `atomic_shared_ptr<T>` link inside itself.  Specialise to `std::true_type`
//! (before the first use of `atomic_shared_ptr<T>` / `local_shared_ptr<T>`):
//!
//!     template <...> struct force_intrusive_ref<MyNode<...>> : std::true_type {};
//!
//! The forced type then takes the intrusive `Ref = T` path (no separate control
//! block; disposal via `T::atomic_intrusive_dispose` if present, else `delete`),
//! and `local_weak_ptr<T>` is disabled (`has_weak == false`).  T must supply the
//! intrusive contract — a `typedef ... Refcnt;` and an `atomic<Refcnt> refcnt;`
//! member — exactly as `atomic_countable` would; it need NOT inherit
//! `atomic_countable` (so it can avoid that base's `~assert(refcnt==0)` when the
//! type has non-refcount disposal paths of its own).
//!
//! This is a COMPILE-TIME trait-dispatch hook only; it does NOT touch the
//! lock-free refcount / CAS / tagged-local-ref SMR core (its GenMC/TLA
//! verification is unaffected).
template <typename T> struct force_intrusive_ref : std::false_type {};

//! (§36c) Opt-out of marker AUTO-detection for a NON-intrusive type that is
//! INCOMPLETE — or whose instantiation would be CIRCULAR — at the first use of
//! `local_shared_ptr<T>` / `atomic_shared_ptr<T>`.  The `sizeof(T)` completeness
//! probe in `ref_traits_auto` instantiates `T`; for a plain incomplete class
//! (`struct N;`) that soft-fails and the incomplete fallback is chosen, but for
//! a not-yet-instantiated class TEMPLATE id (e.g. `Holder<A>` used while `A` is
//! still being defined, where `Holder<A>` transitively needs `A` complete) the
//! probe forces `Holder<A>`'s instantiation and the resulting error is a HARD
//! error, NOT soft SFINAE — so `local_shared_ptr<Holder<A>>` cannot be a member
//! of `A` the way `std::shared_ptr<Holder<A>>` can.  Specialise this to
//! `std::true_type` (before that first use) to skip the probe and take the
//! default non-intrusive two-alloc `gref_<T>` control block — identical to the
//! auto-detected incomplete fallback, and the only `T` usage is `T *ptr;` which
//! is valid on an incomplete T.  `local_weak_ptr<T>` stays available.
//!
//!     template <class X> struct force_incomplete_ref<Holder<X>> : std::true_type {};
//!
//! Mutually exclusive with `force_intrusive_ref<T>` (intrusive wins).  Pure
//! compile-time trait dispatch — does NOT touch the lock-free SMR core.
template <typename T> struct force_incomplete_ref : std::false_type {};

//! AUTO-detection traits (used only for NON-forced types).  A `sizeof(T)`
//! completeness gate distinguishes the two: incomplete T → non-intrusive
//! gref_<T> fallback; complete T → marker-base (`is_base_of`) detection.
//! Kept SEPARATE from the dispatcher `ref_traits` below so that the
//! `sizeof(T)` probe is NEVER substituted for a force-intrusive type (which is
//! typically self-referential and INCOMPLETE at first use — a `sizeof` there is
//! a hard error, not soft SFINAE, because it happens mid-definition).
template <typename T, typename = void>
struct ref_traits_auto {
    //! Incomplete T: default to plain non-intrusive gref_<T>.
    static constexpr bool is_intrusive = false;
    static constexpr bool is_emplaced  = false;
    static constexpr bool is_strict    = false;
    using Ref = atomic_shared_ptr_gref_<T>;
    static constexpr bool has_weak = true;
};

template <typename T>
struct ref_traits_auto<T, std::void_t<decltype(sizeof(T))>> {
    //! Complete T: full marker-base detection.
    static constexpr bool is_intrusive
        = std::is_base_of<atomic_countable, T>::value;
    static constexpr bool is_emplaced
        = std::is_base_of<atomic_emplaced, T>::value && !is_intrusive;
    static constexpr bool is_strict
        = std::is_base_of<atomic_strictrefonly, T>::value && !is_intrusive;

    //!< intrusive → T itself ; emplaced → gref_weakable_<T> ;
    //!< strict   → gref_strictrefonly_<T> ; otherwise → gref_<T>.
    using Ref = typename std::conditional<is_intrusive, T,
                typename std::conditional<is_emplaced, atomic_shared_ptr_gref_weakable_<T>,
                typename std::conditional<is_strict,   atomic_shared_ptr_gref_strictrefonly_<T>,
                                                       atomic_shared_ptr_gref_<T>>::type>::type>::type;

    //!< Whether `local_weak_ptr<T>` is allowed (gref_weak_base_ is in
    //!< the Ref chain).  Intrusive and strict opt out.
    static constexpr bool has_weak = !is_intrusive && !is_strict;
};

//! Dispatcher.  The `int Mode` second parameter — defaulted from the two
//! force traits (each needs only T's template-id, NOT a complete T) — picks:
//!   0 = AUTO-detection (`sizeof(T)` marker probe; requires T completable),
//!   1 = forced INTRUSIVE   (`force_intrusive_ref<T>`; NO `sizeof` anywhere),
//!   2 = forced INCOMPLETE  (`force_incomplete_ref<T>`; non-intrusive gref_<T>
//!       fallback, NO `sizeof` — for circular/incomplete template-id members).
//! Intrusive wins if both are (mis)specialised.
template <typename T>
constexpr int ref_force_mode() noexcept {
    return force_intrusive_ref<T>::value ? 1
         : (force_incomplete_ref<T>::value ? 2 : 0);
}
template <typename T, int Mode = ref_force_mode<T>()>
struct ref_traits : ref_traits_auto<T> {};

//! (§36b) Forced-intrusive — `Ref = T` (no separate control block; disposal via
//! `T::atomic_intrusive_dispose` if present, else `delete`), `local_weak_ptr<T>`
//! disabled.  Reached WITHOUT a `sizeof(T)` probe, so it is valid even when T is
//! incomplete (a self-referential intrusive list/DLL node).
//!
//! Provides `Refcnt` HERE (in the trait, which is complete) so that
//! `atomic_shared_ptr_base` can take the refcount type from the trait rather
//! than from `Ref::Refcnt`.  For a self-referential intrusive class TEMPLATE,
//! `Ref::Refcnt` would force a qualified-name lookup that INSTANTIATES (and so
//! completes) the still-incomplete chunk specialisation → circular with its
//! own `atomic_shared_ptr<chunk>` member → a hard error on GCC (clang is
//! lenient).  A concrete intrusive type escapes this because its `Refcnt` is
//! reachable via a complete base (e.g. `atomic_countable`) without instantiating
//! the type; a template specialisation has no such escape.  All control blocks
//! in this header use `uintptr_t` for the count, so the trait fixes it here.
template <typename T>
struct ref_traits<T, 1> {
    static constexpr bool is_intrusive = true;
    static constexpr bool is_emplaced  = false;
    static constexpr bool is_strict    = false;
    using Ref = T;
    using Refcnt = uintptr_t;
    static constexpr bool has_weak = false;
};

//! (§36c) Forced-incomplete — default non-intrusive two-alloc `gref_<T>`
//! control block (its only T usage is `T *ptr;`, valid on incomplete T), NO
//! `sizeof(T)` probe.  Mirrors the auto-detected incomplete fallback so a
//! circular/incomplete template-id can be a `local_shared_ptr<T>` member.
template <typename T>
struct ref_traits<T, 2> {
    static constexpr bool is_intrusive = false;
    static constexpr bool is_emplaced  = false;
    static constexpr bool is_strict    = false;
    using Ref = atomic_shared_ptr_gref_<T>;
    static constexpr bool has_weak = true;
};

//! (§36b) Opt-in custom disposer for the INTRUSIVE mode (`atomic_countable`).
//! If `T` provides a static `T::atomic_intrusive_dispose(T*)`, the `deleter`
//! calls it — with the object still LIVE (so it can read members, e.g. a
//! region/chunk size) — INSTEAD of the heap `delete p` when the intrusive
//! refcnt reaches 0.  This lets a placement-new'd pool-region object dispose
//! via its own teardown (run `~T()` + `deallocate_chunk`) rather than
//! `::operator delete`.  It affects ONLY the terminal, single-threaded release
//! leaf (the unique last releaser) — it does NOT touch the lock-free
//! refcount / CAS / tagged-local-ref protocol, so the GenMC/TLA verification
//! of the concurrency core is unchanged.
template <typename T, typename = void>
struct has_intrusive_dispose : std::false_type {};
template <typename T>
struct has_intrusive_dispose<
    T, std::void_t<decltype(T::atomic_intrusive_dispose(std::declval<T*>()))>>
    : std::true_type {};

//! \brief Single base class for atomic_shared_ptr / local_shared_ptr.
//! Mode is driven by `ref_traits<T>`; all four paths
//! (default / strict / emplaced / intrusive) share this template.
//!
//! `LOCAL_REF_CAPACITY` (lower bits of the tagged pointer available for
//! the local refcount): equals the minimum alignment guaranteed for
//! `Ref` storage.  `Ref` heap-allocated via `new` is `sizeof(intptr_t)`-
//! aligned; for intrusive, `Ref = T` and T has `atomic<uintptr_t> refcnt`
//! so alignment is at least `sizeof(double)`.  Both give 3 usable bits
//! (max local refcount = 7) on 64-bit.
//!
//! Do NOT use `alignas` / `alignof` on Ref — see CLAUDE.md.
//!
//! `KAME_LOCAL_REF_CAPACITY_OVERRIDE`: stress-testing knob; force a
//! smaller capacity to simulate high-CPU contention without changing
//! actual allocator alignment.
template <typename T, typename reflocal_t, typename reflocal_var_t,
          typename Enable = void>
struct atomic_shared_ptr_base {
protected:
    using Traits = ref_traits<T>;
    using Ref = typename Traits::Ref;
    // Take `Refcnt` from the TRAIT when it provides one, else from `Ref`.  The
    // trait is always complete, whereas `Ref::Refcnt` on a self-referential
    // intrusive class TEMPLATE forces the chunk's instantiation → circular with
    // its own `atomic_shared_ptr<chunk>` member (hard error on GCC).  See the
    // forced-intrusive `ref_traits<T,true>` spec (which sets `Refcnt`).
    template <typename TR, typename = void>
    struct refcnt_of_ { using type = typename Ref::Refcnt; };
    template <typename TR>
    struct refcnt_of_<TR, std::void_t<typename TR::Refcnt>> {
        using type = typename TR::Refcnt;
    };
    using Refcnt = typename refcnt_of_<Traits>::type;

    static int deleter(Ref *p) noexcept {
        if constexpr (Traits::is_intrusive) {
            if constexpr (has_intrusive_dispose<T>::value) {
                //!< (§36b) custom region disposer — object still LIVE so it
                //!< can read its size etc., then run ~T() + deallocate_chunk.
                T::atomic_intrusive_dispose(p);
            } else {
                //!< T's dtor runs (incl. ~atomic_countable's `assert(refcnt == 0)`).
                delete p;
            }
        } else if constexpr (Traits::has_weak) {
            //!< Two-counter release: destroy T, drop implicit weak.
            Ref::release_strong_zero(p);
        } else {
            //!< strict mode: `~Ref` does `delete ptr`.
            delete p;
        }
        return 1;
    }

    //! Adopt a freshly-allocated object.  Emplaced types disallow this
    //! path (use `make_local_shared<T>(args)`) — the static_assert fires
    //! only if reset_unsafe is actually instantiated.
    template<typename Y> void reset_unsafe(Y *y) noexcept(Traits::is_intrusive) {
        static_assert( !Traits::is_emplaced,
            "Emplaced T: use make_local_shared<T>(args), not local_shared_ptr<T>(T*)");
        if constexpr (Traits::is_intrusive) {
            m_ref = (reflocal_t)static_cast<T*>(y);   //!< T's ctor set refcnt=1.
        } else {
            m_ref = (reflocal_t)new Ref(y);           //!< Wrap y in a fresh CB.
        }
        if constexpr (is_biased_directpublish<T>::value)   //!< §biased — skippable: born private (-1)
            biased_born_(((Ref*)(reflocal_t)m_ref)->refcnt);
    }

    T *get() noexcept {
        if constexpr (Traits::is_intrusive) {
            //!< Branchless: `(T*)0` is `nullptr`, no offset (Ref IS T).
            return (T*)(reflocal_t)this->m_ref;
        } else if(this->m_ref) {
            Ref *p = (Ref*)(reflocal_t)this->m_ref;
            if constexpr (Traits::is_emplaced) return p->ptr_();
            else                               return p->ptr;
        }
        return nullptr;
    }
    const T *get() const noexcept {
        return const_cast<atomic_shared_ptr_base *>(this)->get();
    }

    int _use_count_() const noexcept {
        uintptr_t v = ((const Ref*)(reflocal_t)this->m_ref)->refcnt;
        if constexpr (is_biased_directpublish<T>::value) return (int)biased_count_(v); //!< §biased — skippable
        return (int)v;
    }

    reflocal_var_t m_ref;

#ifdef KAME_LOCAL_REF_CAPACITY_OVERRIDE
    enum {LOCAL_REF_CAPACITY = KAME_LOCAL_REF_CAPACITY_OVERRIDE};
#else
    enum {LOCAL_REF_CAPACITY =
        (Traits::is_intrusive ? sizeof(double) : sizeof(intptr_t))};
#endif
};
//! \brief This class provides non-reentrant interfaces for atomic_shared_ptr: operator->(), operator*() and so on.\n
//! Use this class in non-reentrant scopes instead of costly atomic_shared_ptr.
//! \sa atomic_shared_ptr, atomic_unique_ptr, atomic_shared_ptr_test.cpp.
template <typename T, typename reflocal_var_t = uintptr_t>
class local_shared_ptr : protected atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t> {
public:
    local_shared_ptr() noexcept { this->m_ref = (TaggedPtr)nullptr; }

    template<typename Y> explicit local_shared_ptr(Y *y) {
        // For non-polymorphic T, Y must equal T (otherwise the
        // virtual-less destructor in the deleter path would slice).
        // Polymorphic T's accept derived Y via virtual ~T.
        static_assert(std::is_same<T, Y>::value || std::has_virtual_destructor<T>::value,
            "local_shared_ptr<T>(Y*): T must be polymorphic when Y != T");
        this->reset_unsafe(y);
    }

    explicit local_shared_ptr(const atomic_shared_ptr<T> &t) noexcept { this->m_ref = reinterpret_cast<TaggedPtr>(t.load_shared_()); }
    template<typename Y> local_shared_ptr(const atomic_shared_ptr<Y> &y) {
        static_assert(sizeof(static_cast<const T*>(y.get())), "");
        this->m_ref = reinterpret_cast<TaggedPtr>(y.load_shared_());
    }
    inline local_shared_ptr(const local_shared_ptr<T, reflocal_var_t> &t) noexcept;
    template<typename Y, typename Z> inline local_shared_ptr(const local_shared_ptr<Y, Z> &y) noexcept;
    local_shared_ptr(local_shared_ptr<T, reflocal_var_t> &&t) noexcept {
        this->m_ref = t.m_ref;
        t.m_ref = (TaggedPtr)nullptr;
    }
    template<typename Y, typename Z> local_shared_ptr(local_shared_ptr<Y, Z> &&y) noexcept {
        this->m_ref = y.m_ref;
        y.m_ref = (TaggedPtr)nullptr;
    }
    inline ~local_shared_ptr();

    local_shared_ptr &operator=(const local_shared_ptr &t) noexcept {
        local_shared_ptr(t).swap( *this);
        return *this;
    }
    template<typename Y, typename Z> local_shared_ptr &operator=(const local_shared_ptr<Y, Z> &y) noexcept {
        local_shared_ptr(y).swap( *this);
        return *this;
    }
    local_shared_ptr &operator=(local_shared_ptr &&t) noexcept {
        t.swap( *this);
        t.reset();
        return *this;
    }
    template<typename Y, typename Z> local_shared_ptr &operator=(local_shared_ptr<Y, Z> &&y) noexcept {
        y.swap( *this);
        y.reset();
        return *this;
    }
    //! \param[in] t The pointer held by this instance is replaced with that of \a t.
    local_shared_ptr &operator=(const atomic_shared_ptr<T> &t) noexcept {
        this->reset();
        this->m_ref = reinterpret_cast<TaggedPtr>(t.load_shared_());
        return *this;
    }
    //! \param[in] y The pointer held by this instance is replaced with that of \a y.
    template<typename Y> local_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) noexcept {
        static_assert(sizeof(static_cast<const T*>(y.get())), "");
        this->reset();
        this->m_ref = reinterpret_cast<TaggedPtr>(y.load_shared_());
        return *this;
    }

    //! \param[in,out] x \p The pointer held by \a x is swapped with that of this instance.
    inline void swap(local_shared_ptr &x) noexcept;
    //! \param[in,out] x \p The pointer held by \a x is atomically swapped with that of this instance.
    void swap(atomic_shared_ptr<T> &x) noexcept;

    //! The pointer held by this instance is reset to null pointer.
    inline void reset() noexcept;
    //! The pointer held by this instance is reset with a pointer \a y.
    template<typename Y> void reset(Y *y) {
        static_assert(std::is_same<T, Y>::value || std::has_virtual_destructor<T>::value,
            "local_shared_ptr<T>::reset(Y*): T must be polymorphic when Y != T");
        reset();
        this->reset_unsafe(y);
    }

    //! Const-transparent (std::shared_ptr semantics): `get()`,
    //! `operator*`, `operator->` always return non-const `T*` / `T&`
    //! regardless of the constness of `*this`.  The smart pointer's
    //! constness applies to the pointer itself (cannot reset / swap),
    //! not to the pointee.  This matches what migrating code from
    //! `std::shared_ptr<T>` expects.
    T *get() const noexcept {
        return const_cast<local_shared_ptr *>(this)->atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::get();
    }

    T &operator*() const noexcept { assert( *this); return *get();}
    T *operator->() const noexcept { assert( *this); return get();}

    bool operator!() const noexcept {return !this->m_ref;}
    operator bool() const noexcept {return this->m_ref;}

    template<typename Y, typename Z> bool operator==(const local_shared_ptr<Y, Z> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->ref_ptr_() == (const Ref *)x.ref_ptr_());}
    template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->ref_ptr_() == (const Ref *)x.ref_ptr_());}
    template<typename Y, typename Z> bool operator!=(const local_shared_ptr<Y, Z> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->ref_ptr_() != (const Ref *)x.ref_ptr_());}
    template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->ref_ptr_() != (const Ref *)x.ref_ptr_());}

    int use_count() const noexcept { return this->_use_count_();}
    bool unique() const noexcept {return use_count() == 1;}

    //!< Tag for the `local_weak_ptr::lock()`-internal ctor below.
    struct adopt_promoted_t {};
    //!< Adopt a Ref* with the strong refcnt already bumped (typically
    //!< by a successful `try_promote()` from `local_weak_ptr::lock()`).
    //!< Local count starts at 0; the global +1 already exists.
    inline local_shared_ptr(adopt_promoted_t, typename atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::Ref *r) noexcept {
        this->m_ref = reinterpret_cast<TaggedPtr>(r);
    }

protected:
    template <typename Y, typename Z> friend class local_shared_ptr;
    template <typename Y> friend class atomic_shared_ptr;
    template <typename Y> friend class scoped_atomic_view;  // operator local_shared_ptr<T>() needs m_ref
    template <typename Y> friend class local_weak_ptr;  // lock() / promote path
    typedef typename atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::Ref Ref;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::Refcnt Refcnt;
    typedef uintptr_t TaggedPtr;

    //! A pointer to global reference struct.
    Ref* ref_ptr_() const noexcept {return (Ref *)(TaggedPtr)(this->m_ref);}
};

//! \brief Weak counterpart of `local_shared_ptr<T>`.  Works with any
//! `T` whose control block carries a `weak_refcnt` — i.e., the default
//! mode (no marker) and `atomic_emplaced` / `atomic_weakable`.
//! NOT supported when `T : atomic_strictrefonly` (compile error at
//! point of use).
//!
//! Stores a type-erased `gref_weak_base_*` so the class can be
//! instantiated when `T` is only forward-declared (e.g.,
//! `PacketWrapper` holds `local_weak_ptr<Linkage>` before Linkage's
//! full definition).  Method bodies that need T complete (`reset`,
//! `lock`) use `if constexpr` / SFINAE at instantiation time.
//!
//! `lock()` atomically promotes to `local_shared_ptr<T>` if `T` is
//! still alive.
template <typename T>
class local_weak_ptr {
public:
    local_weak_ptr() noexcept : m_ref(nullptr) {}

    //! Construct from a `local_shared_ptr<T>` — bumps weak_refcnt by 1.
    //! T must be complete at this point (method body instantiation).
    template <typename Z>
    explicit local_weak_ptr(const local_shared_ptr<T, Z> &sp) noexcept
        : m_ref(nullptr) {
        static_assert(!std::is_base_of<atomic_strictrefonly, T>::value,
            "local_weak_ptr<T>: T must not inherit from atomic_strictrefonly");
        if(sp.m_ref) {
            using Ref = typename local_shared_ptr<T, Z>::Ref;
            m_ref = static_cast<gref_weak_base_ *>(
                reinterpret_cast<Ref *>(static_cast<uintptr_t>(sp.m_ref)));
            m_ref->weak_refcnt.fetch_add(1, std::memory_order_acq_rel);
            //!< §biased — skippable: taking a weak handle (promotable cross-thread)
            //!< is a publish point, so a biased CB is negated -count→+count here.
            if constexpr (is_biased_directpublish<T>::value) biased_publish_(m_ref->refcnt);
        }
    }

    //! Copy: straightforward global fetch_add (mirrors local_shared_ptr).
    local_weak_ptr(const local_weak_ptr &o) noexcept : m_ref(o.m_ref) {
        if(m_ref)
            m_ref->weak_refcnt.fetch_add(1, std::memory_order_acq_rel);
    }

    local_weak_ptr(local_weak_ptr &&o) noexcept : m_ref(o.m_ref) { o.m_ref = nullptr; }

    ~local_weak_ptr() noexcept { reset(); }

    local_weak_ptr &operator=(const local_weak_ptr &o) noexcept {
        local_weak_ptr(o).swap( *this);
        return *this;
    }
    template <typename Z>
    local_weak_ptr &operator=(const local_shared_ptr<T, Z> &sp) noexcept {
        local_weak_ptr(sp).swap( *this);
        return *this;
    }
    local_weak_ptr &operator=(local_weak_ptr &&o) noexcept {
        o.swap( *this);
        o.reset();
        return *this;
    }

    void swap(local_weak_ptr &o) noexcept {
        gref_weak_base_ *t = m_ref; m_ref = o.m_ref; o.m_ref = t;
    }

    //! Drop weak ref; if last, free the CB.
    //! T must be complete when instantiated (to select the right
    //! derived Ref type for deletion).
    void reset() noexcept {
        if(m_ref) {
            if(m_ref->weak_refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                //!< Select the correct derived gref type for delete.
                //!< T is complete at this point (deferred template instantiation).
                if constexpr (ref_traits<T>::is_emplaced) {
                    using Ref = atomic_shared_ptr_gref_weakable_<T>;
                    Ref::release_weak_zero(static_cast<Ref *>(m_ref));
                }
                else {
                    using Ref = atomic_shared_ptr_gref_<T>;
                    Ref::release_weak_zero(static_cast<Ref *>(m_ref));
                }
            }
            m_ref = nullptr;
        }
    }

    //! Control-block identity test against a LIVE \a sp WITHOUT a
    //! weak->strong promotion — zero refcount traffic (lock() costs a
    //! try_promote + a release RMW).  This never dereferences the weak
    //! side; it only compares control-block addresses.  m_ref is a bare
    //! CB pointer (the weak handle carries no local-count tag) and
    //! sp.ref_ptr_() returns sp's masked CB pointer, so equality is
    //! exact for any tag layout of \a sp.
    //!
    //! Correctness contract: the caller must pass a sp that is KNOWN
    //! ALIVE (refcnt >= 1).  A live CB's address is occupied, so a true
    //! result means genuine identity — no freed-and-reused-CB ABA.  A
    //! dead-but-weakly-pinned CB on THIS side simply won't match a
    //! distinct live CB, degrading to a normal miss.  Returns false when
    //! either side is empty (null != a live CB; matches only null==null).
    template <typename Z>
    bool same_control_block(const local_shared_ptr<T, Z> &sp) const noexcept {
        return m_ref == static_cast<const gref_weak_base_ *>(sp.ref_ptr_());
    }

    //! Promote to `local_shared_ptr<T>`; returns empty when expired.
    //! T must be complete when instantiated.
    local_shared_ptr<T> lock() const noexcept {
        if( !m_ref || !m_ref->try_promote()) return local_shared_ptr<T>();
        //!< Downcast to the actual Ref type for local_shared_ptr's adopt ctor.
        using Ref = typename local_shared_ptr<T>::Ref;
        return local_shared_ptr<T>(
            typename local_shared_ptr<T>::adopt_promoted_t{},
            static_cast<Ref *>(m_ref));
    }

    bool expired() const noexcept {
        return !m_ref || m_ref->refcnt.load(std::memory_order_acquire) == 0;
    }

    bool operator!() const noexcept {return !m_ref;}
    explicit operator bool() const noexcept {return m_ref;}

private:
    gref_weak_base_ *m_ref;
};

/*! \brief This is an atomic variant of \a std::shared_ptr, and can be shared by atomic and lock-free means.\n
 *
* \a atomic_shared_ptr can be shared among threads by the use of \a operator=(_target_), \a swap(_target_).
* An instance of \a atomic_shared_ptr<T> holds:\n
* 	a) a pointer to \a atomic_shared_ptr_gref_<T>, which is a struct. consisting of a pointer to a T-type object and a global reference counter.\n
* 	b) a local (temporary) reference counter, which is embedded in the above pointer by using several LSBs that should be usually zero.\n
* The values of a) and b), \a m_ref, are atomically handled with CAS machine codes.
* The purpose of b) the local reference counter is to tell the "observation" to the shared target before increasing the global reference counter.
* This process is implemented in \a acquire_tag_ref_().\n
* A function \a release_tag_ref_() tries to decrease the local counter first. When it fails, the global counter is decreased.\n
* To swap the pointer and local reference counter (which will be reset to zero), the setter must adds the local counting to the global counter before swapping.
* \sa atomic_unique_ptr, local_shared_ptr, atomic_shared_ptr_test.cpp.
 */
template <typename T>
class atomic_shared_ptr : protected local_shared_ptr<T, atomic<uintptr_t>> {
public:
    atomic_shared_ptr() noexcept : local_shared_ptr<T, atomic<uintptr_t>>() {}

    template<typename Y> explicit atomic_shared_ptr(Y *y) : local_shared_ptr<T, atomic<uintptr_t>>(y) { publish_clear_priv_(); }
    atomic_shared_ptr(const atomic_shared_ptr<T> &t) noexcept : local_shared_ptr<T, atomic<uintptr_t>>(t) {} //!< source already shared (PRIV clear)
    template<typename Y> atomic_shared_ptr(const atomic_shared_ptr<Y> &y) noexcept : local_shared_ptr<T, atomic<uintptr_t>>(y) {} //!< source already shared
    atomic_shared_ptr(const local_shared_ptr<T> &t) noexcept : local_shared_ptr<T, atomic<uintptr_t>>(t) { publish_clear_priv_(); }
    template<typename Y> atomic_shared_ptr(const local_shared_ptr<Y> &y) noexcept : local_shared_ptr<T, atomic<uintptr_t>>(y) { publish_clear_priv_(); }
    atomic_shared_ptr(atomic_shared_ptr<T> &&t) noexcept {
        operator=(std::move(t));
    }
    template<typename Y> atomic_shared_ptr(atomic_shared_ptr<Y> &&y) noexcept {
        operator=(std::move(y));
    }

    ~atomic_shared_ptr() {}

    //! \param[in] t The pointer held by this instance is atomically replaced with that of \a t.
    atomic_shared_ptr &operator=(const atomic_shared_ptr &t) noexcept {
        local_shared_ptr<T>(t).swap( *this);
        return *this;
    }
    //! \param[in] y The pointer held by this instance is atomically replaced with that of \a y.
    template<typename Y> atomic_shared_ptr &operator=(const local_shared_ptr<Y> &y) noexcept {
        local_shared_ptr<T>(y).swap( *this);
        return *this;
    }
    atomic_shared_ptr &operator=(local_shared_ptr<T> &&t) noexcept {
        t.swap( *this);
        t.reset();
        return *this;
    }
    template<typename Y> atomic_shared_ptr &operator=(local_shared_ptr<Y> &&y) noexcept {
        y.swap( *this);
        y.reset();
        return *this;
    }
    //! The pointer held by this instance is atomically reset to null pointer.
    void reset() noexcept {
        local_shared_ptr<T>().swap( *this);
    }
    //! The pointer held by this instance is atomically reset with a pointer \a y.
    template<typename Y> void reset(Y *y) {
        local_shared_ptr<T>(y).swap( *this);
    }

    //! \return true if succeeded.
    //! \sa compareAndSwap()
    bool compareAndSet(const local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue) noexcept;
    //! \return true if succeeded.
    //! \sa compareAndSet()
    bool compareAndSwap(local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue) noexcept;
    //! \return true if succeeded.
    //! \sa compareAndSet()
    bool compareAndSetWeak(const local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue) noexcept;

    //! \return true if succeeded.
    //! \brief Weakly version using a pre-acquired \a scoped_atomic_view.
    //!   On success, \a scoped is reset to Empty (tag consumed by CAS). On
    //!   weak failure (CAS contention), \a scoped remains TagHeld for retry.
    //!   On pointer change since acquire, \a scoped is eagerly cleaned up
    //!   to Empty so the caller can detect it via \a scoped.operator bool().
    inline bool compareAndSetWeak(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newvalue) noexcept;

    //! \brief Like compareAndSetWeak(scoped, newr) but on success \a scoped
    //!   transitions to Owned(newr) instead of Empty.  Saves a reload when
    //!   the caller needs to keep tracking the new value after CAS success.
    //!   Entry does fetch_add(2) instead of (1); failure undo is fetch_sub(2).
    inline bool compareAndSetWeakRetain(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newvalue) noexcept;

    //! \brief STRONG (spinning) version of compareAndSetWeak(scoped, newr).
    //!   Internal CAS retry loop on spurious weak failure; returns false
    //!   only on pointer mismatch (real contention). Intended for use by
    //!   the privileged thread (s_privileged_tidstamp holder), where
    //!   fair_mode blocks all other CAS — no peer to contend with.
    inline bool compareAndSetStrong(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newvalue) noexcept;
    //! \brief STRONG + RETAIN_NEWR variant — see compareAndSetStrong and
    //!   compareAndSetWeakRetain.
    inline bool compareAndSetStrongRetain(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newvalue) noexcept;

    bool operator!() const noexcept {return !this->m_ref;}
    operator bool() const noexcept {return this->m_ref;}

    template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (ref_ptr_() == (const Ref*)x.ref_ptr_());}
    template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (ref_ptr_() == (const Ref*)x.ref_ptr_());}
    template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (ref_ptr_() != (const Ref*)x.ref_ptr_());}
    template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (ref_ptr_() != (const Ref*)x.ref_ptr_());}
    //! Comparison with scoped_atomic_view (staleness check etc.).
    bool operator==(const scoped_atomic_view<T> &x) const noexcept {
        return (ref_ptr_() == x.ref_ptr_());}
    bool operator!=(const scoped_atomic_view<T> &x) const noexcept {
        return (ref_ptr_() != x.ref_ptr_());}
protected:
    template <typename Y, typename Z> friend class local_shared_ptr;
    template <typename Y> friend class atomic_shared_ptr;
    template <typename Y> friend class scoped_atomic_view;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, atomic<uintptr_t>>::Ref Ref;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, atomic<uintptr_t>>::Refcnt Refcnt;
    typedef atomic<uintptr_t> TaggedPtr;
    //! A pointer to global reference struct.
    Ref* ref_ptr_() const noexcept {
        auto ref = this->m_ref.load(std::memory_order_relaxed);
        return (Ref*)(ref & (~(uintptr_t)(this->LOCAL_REF_CAPACITY - 1)));
    }
    //! (§biased) PUBLISH from a value/copy constructor that installs a CB
    //! straight into this atomic slot (so it is NOT routed through
    //! swap()/compareAndSet_impl_, the two RMW publish points).  Negates a still
    //! private (-count) CB to +count (release) so the now-shared CB uses the
    //! atomic refcnt path henceforth; idempotent on an already-shared (+) CB.
    //! At construction *this is not yet reachable by other threads, so the
    //! base copy-ctor's owner-side bump preceding this is sound.  Folds away
    //! (empty) when the gate is OFF — off-path codegen stays byte-identical.
    void publish_clear_priv_() noexcept {
        if constexpr (is_biased_directpublish<T>::value)         //!< §biased — skippable
            if(Ref *p = ref_ptr_()) biased_publish_(p->refcnt);
    }
    //! Single atomic load returning both the pointer and the local refcount.
    std::pair<Ref*, Refcnt> load_tagged_() const noexcept {
        auto ref = this->m_ref.load(std::memory_order_relaxed);
        return {(Ref*)(ref & (~(uintptr_t)(this->LOCAL_REF_CAPACITY - 1))),
                (Refcnt)(ref & (uintptr_t)(this->LOCAL_REF_CAPACITY - 1))};
    }

    //internal functions below.
    //! Atomically scans \a m_ref and increases the global reference counter.
    //! \a load_shared_() is used for atomically coping the pointer.
    inline Ref *load_shared_() const noexcept;
    //! Atomically scans \a m_ref and increases the  local (temporary) reference counter.
    //! use \a release_tag_ref_() to release the temporary reference.
    inline std::pair<Ref*, bool> acquire_tag_ref_(Refcnt *, bool weakly = false) const noexcept;
    //! Tries to decrease local (temporary) reference counter.
    //! In case the reference is lost, \a release_tag_ref_() releases the global reference counter instead.
    //! When \a left_global_rcnt > 0, undoes step 4's
    //! excess (left_global_rcnt - 1) on tag-success, or combines undo+release on pointer-changed.
    //! \param[in] single_attempt  If true, drain CAS is single-shot;
    //!   on CAS-loss, returns false WITHOUT global fetch_sub.
    inline bool release_tag_ref_(Ref *, Refcnt added_global_rcnt,
                                  bool single_attempt = false) const noexcept;

    //! Unified CAS template covering all flavours of compareAndSet/Swap.
    //!
    //! The OldrT parameter is deduced from the call site and selects the
    //! variant via constexpr if branches:
    //!   - OldrT = const local_shared_ptr<T> (Set):
    //!       no acquire_tag_ref_ (oldr keeps pref alive); step4 = +T;
    //!       failure undo via plain fetch_sub(T, relaxed).
    //!   - OldrT = local_shared_ptr<T> (Swap):
    //!       acquire_tag_ref_ required (will update oldr on mismatch);
    //!       step4 = +(T-1); failure undo via release_tag_ref_(pref, T).
    //!   - OldrT = scoped_atomic_view<T> (SetScoped, weak only):
    //!       scoped already holds tag (no acquire); step4 = +(T-1);
    //!       failure undo via plain fetch_sub(T-1, relaxed); on success,
    //!       scoped's tag is consumed by CAS (m_pref reset to nullptr).
    //! NewrT = const local_shared_ptr<T> (caller retains ownership):
    //!   fetch_add(1) at start, fetch_sub(1) at WEAK-failure undo.
    //!   m_ref takes its own +1 via the fetch_add; caller's local
    //!   var keeps its +1 separately.
    template<typename OldrT, typename NewrT, bool WEAK = false, bool RETAIN_NEWR = false>
    inline bool compareAndSet_impl_(OldrT &oldr,
        NewrT &newr) noexcept;
private:
};

//! \brief RAII scoped tag holder on \a atomic_shared_ptr<T>.
//!
//! Acquires a tag ref on the supplied atomic_shared_ptr's m_ref in the
//! constructor (1 CAS). Releases on destruction. Move-only.
//!
//! Three logical states:
//!   - Empty (m_pref == nullptr):
//!       - asp was nullptr, or
//!       - weakly acquire failed (\a acquire_succeeded() == false), or
//!       - tag was consumed by a successful compareAndSetWeak.
//!   - TagHeld (m_pref != nullptr, m_tag_held == true):
//!       Tag still held in m_ref's tag count. Destructor calls
//!       release_tag_ref_(pref, 1u). compareAndSetWeak uses scoped path.
//!   - Owned (m_pref != nullptr, m_tag_held == false):
//!       Tag was promoted to refcnt at construction time
//!       (fetch_add(rcnt) + release_tag_ref_). Destructor does plain
//!       fetch_sub(1, acq_rel) + delete check (like local_shared_ptr).
//!       compareAndSetWeak uses Set path (regular const local_shared_ptr
//!       semantics).
//!
//! Adaptive promotion:
//!   The optional \a promote_threshold parameter selects between TagHeld and
//!   Owned at construction. If the tag count after acquire (rcnt_new) is >=
//!   threshold, the scoped is promoted to Owned (frees a tag slot). Useful
//!   when many concurrent readers risk filling LOCAL_REF_CAPACITY.
//!     - threshold = 1: always promote → equivalent to load_shared_().
//!     - threshold = LOCAL_REF_CAPACITY-1 (DEFER, default): promote only at
//!       the LAST slot. Reserved for the privileged thread (single contender
//!       guarantee, see ScopedNegotiateLinkage); keeps TagHeld cheap.
//!     - threshold = LOCAL_REF_CAPACITY-2 (ADAPTIVE): promote one slot earlier
//!       — the thread that lands at the second-to-last slot pre-emptively
//!       drains tag bits, leaving room (the LAST slot) for the privileged
//!       thread.  Used by all non-privileged acquires.
//!
//!   A "never-promote" mode (formerly threshold = LOCAL_REF_CAPACITY) was
//!   removed — without promote, peer-thread TagHeld views accumulate and
//!   block zero-reset CAS indefinitely (see livelock fixed in c363629a).
template <typename T>
class scoped_atomic_view {
public:
    typedef typename atomic_shared_ptr<T>::Ref Ref;
    typedef typename atomic_shared_ptr<T>::Refcnt Refcnt;

    enum {
        LOCAL_REF_CAPACITY = atomic_shared_ptr<T>::LOCAL_REF_CAPACITY,
        DEFER_THRESHOLD = LOCAL_REF_CAPACITY - 1,        //!< promote at last slot (privileged-only)
        ADAPTIVE_THRESHOLD = LOCAL_REF_CAPACITY - 2,     //!< promote one slot early (non-privileged)
    };

    scoped_atomic_view() noexcept
        : m_asp(nullptr), m_pref(nullptr), m_tag_held(false),
          m_acquire_succeeded(true) {}

    //! Acquires a tag ref on \a asp.m_ref.
    //! \param[in] promote_threshold Tag-count threshold for adaptive
    //!   promotion. After acquire bumps tag to rcnt_new, if rcnt_new >=
    //!   promote_threshold the tag is promoted to refcnt (Owned mode).
    //!   Default \a ADAPTIVE_THRESHOLD promotes one slot before the cap
    //!   (= reserve last slot for the privileged thread).  Use
    //!   \a DEFER_THRESHOLD only on the privileged path.
    //! \param[in] weakly If true, the acquire CAS is single-shot — on
    //!   contention, this constructs an Empty instance with
    //!   \a acquire_succeeded() == false. Strong (default) loops until success.
    //! \note On Empty after construction, distinguish:
    //!   - \a acquire_succeeded() == true  AND \a m_pref == nullptr →
    //!       asp held nullptr (genuinely empty atomic_shared_ptr).
    //!   - \a acquire_succeeded() == false AND \a m_pref == nullptr →
    //!       weakly = true and the acquire CAS lost (caller should retry).
    explicit scoped_atomic_view(atomic_shared_ptr<T> &asp,
                                     Refcnt promote_threshold = ADAPTIVE_THRESHOLD,
                                     bool weakly = false) noexcept
        : m_asp(&asp), m_pref(nullptr), m_tag_held(false),
          m_acquire_succeeded(true) {
        Refcnt rcnt;
        auto [p, ok] = asp.acquire_tag_ref_( &rcnt, weakly);
        if(p && ok) {
            if(rcnt >= promote_threshold) {
                // Promote: tag → refcnt. Bit-identical to load_shared_.
                p->refcnt.fetch_add(rcnt, std::memory_order_relaxed);
                asp.release_tag_ref_(p, rcnt);
                m_pref = p;
                m_tag_held = false;  // sentinel for Owned
            } else {
                m_pref = p;
                m_tag_held = true;  // TagHeld
            }
        } else {
            m_acquire_succeeded = ok;  // false on weak fail; true on null asp
        }
    }

    //! Move-construct from a `local_shared_ptr<T>&&`.  Takes ownership of
    //! `from`'s +1 refcount with ZERO atomic ops — the new view starts in
    //! Owned mode (`m_tag_held == 0`) reusing `from`'s refcount.
    //! `asp` is the atomic_shared_ptr the view is bound to (used for the
    //! weak-CAS scoped path and for release on dtor).  Caller is
    //! responsible that the moved-from `local_shared_ptr` was a valid
    //! reference to `asp`'s current value at construction time (we do
    //! NOT verify; standard move-semantics caveat).
    scoped_atomic_view(atomic_shared_ptr<T> &asp, local_shared_ptr<T> &&from) noexcept
        : m_asp(&asp), m_pref(nullptr), m_tag_held(false),
          m_acquire_succeeded(true) {
        if(from.m_ref) {
            m_pref = (Ref *)from.m_ref;
            // m_tag_held stays 0 → Owned mode.
            from.m_ref = 0;  // empty out the source
        }
    }

    //! Replace the current view with a value from `from`, taking
    //! ownership of `from`'s +1 refcount (zero atomic ops on the
    //! transfer).  The previous view is released first.
    //!
    //! Use this after a successful CAS to update the view to the new
    //! linkage value without paying a load_shared_ — e.g. a multi-phase
    //! CAS protocol where each phase advances the linkage and we want
    //! the view to track without re-reading the atomic_shared_ptr.
    void assign_from_local(local_shared_ptr<T> &&from) noexcept {
        release_();
        if(from.m_ref) {
            m_pref = (Ref *)from.m_ref;
            m_tag_held = false;  // Owned mode
            from.m_ref = 0;  // empty out the source
        } else {
            m_pref = nullptr;
            m_tag_held = false;
        }
        m_acquire_succeeded = true;
    }

    scoped_atomic_view(scoped_atomic_view &&other) noexcept
        : m_asp(other.m_asp), m_pref(other.m_pref),
          m_tag_held(other.m_tag_held),
          m_acquire_succeeded(other.m_acquire_succeeded) {
        other.m_pref = nullptr;
        other.m_tag_held = false;
        other.m_acquire_succeeded = true;
    }
    scoped_atomic_view &operator=(scoped_atomic_view &&other) noexcept {
        if(this != &other) {
            release_();
            m_asp = other.m_asp;
            m_pref = other.m_pref;
            m_tag_held = other.m_tag_held;
            m_acquire_succeeded = other.m_acquire_succeeded;
            other.m_pref = nullptr;
            other.m_tag_held = false;
            other.m_acquire_succeeded = true;
        }
        return *this;
    }
    scoped_atomic_view(const scoped_atomic_view &) = delete;
    scoped_atomic_view &operator=(const scoped_atomic_view &) = delete;

    //! Exchange internal state with another view.  Useful for
    //! "transferring ownership through a sub-routine":
    //!   void f(scoped_atomic_view<T> &out) {
    //!       scoped_atomic_view<T> local(*some_asp, ADAPTIVE_THRESHOLD);
    //!       ... use local for CAS / read ...
    //!       out.swap(local);  // hand it back to caller; local goes
    //!                         // out of scope releasing the *old* out.
    //!   }
    //! No tag refcount op happens — just a stateless rearrangement.
    void swap(scoped_atomic_view &other) noexcept {
        std::swap(m_asp, other.m_asp);
        std::swap(m_pref, other.m_pref);
        std::swap(m_tag_held, other.m_tag_held);
        std::swap(m_acquire_succeeded, other.m_acquire_succeeded);
    }

    ~scoped_atomic_view() noexcept { release_(); }

    //! \brief Convert to local_shared_ptr<T>. Internally promotes if needed.
    //!   - From TagHeld: promote (tag → refcnt; bit-identical to load_shared_),
    //!     then fetch_add(1) for the new ref. Scoped transitions to Owned.
    //!   - From Owned: just fetch_add(1) for the new ref. Scoped retains.
    //!   - From Empty: returns empty.
    //! \note lvalue conversion — scoped remains usable after the call.
    //!   For scoped that holds a long-lived ref, consider whether
    //!   the extra fetch_add(1) is worth it vs. holding the scoped directly.
    operator local_shared_ptr<T>() & noexcept {
        local_shared_ptr<T> ret;
        if(m_pref) {
            if(m_tag_held) {
                // TagHeld → Promote (zero-reset): load CURRENT tag count
                // and drain it all in one shot, transferring rcnt_now
                // refs to global.  This absorbs all current tag holders'
                // IOUs (not just our acquire-time snapshot), helping
                // keep tag bits low for other threads' acquires.
                promote_tagheld_();
                m_tag_held = false;  // mode flips to Owned
            }
            // Owned: scoped already has +1 in refcnt. Add another +1 for the
            //   new local_shared_ptr's own ownership.
            m_pref->refcnt.fetch_add(1, std::memory_order_relaxed);
            ret.m_ref = (uintptr_t)m_pref;
        }
        return ret;
    }

    //! \brief rvalue (move) conversion — transfer ownership to a
    //! `local_shared_ptr<T>`, leaving this view empty.
    //!   - From Owned: ZERO atomic ops — the +1 refcnt is just transferred.
    //!   - From TagHeld: promote (zero-reset, current rcnt) but skip the
    //!     fetch_add(1) for the new local_shared_ptr's ownership (the
    //!     promoted ref IS the new ownership).
    //!   - From Empty: returns empty.
    //! Use `std::move(scoped)` to explicitly opt in.  Saves 1 atomic op
    //! vs the lvalue conversion when the view will not be used again.
    operator local_shared_ptr<T>() && noexcept {
        local_shared_ptr<T> ret;
        if(m_pref) {
            if(m_tag_held) {
                promote_tagheld_();
            }
            // Transfer m_pref to ret.  Empty out self so dtor is a no-op.
            ret.m_ref = (uintptr_t)m_pref;
            m_pref = nullptr;
            m_tag_held = false;
        }
        return ret;
    }

    bool operator!() const noexcept { return m_pref == nullptr; }
    explicit operator bool() const noexcept { return m_pref != nullptr; }

    //! \return false only when weakly acquire CAS lost; true otherwise (incl. null asp).
    bool acquire_succeeded() const noexcept { return m_acquire_succeeded; }

    //! \return true if currently in TagHeld mode (vs Owned or Empty).
    bool is_tag_held() const noexcept { return m_pref && m_tag_held; }
    //! \return true if currently in Owned mode (promoted at construction).
    bool is_owned() const noexcept { return m_pref && !m_tag_held; }

    //! Smart-pointer accessors (return T*).
    T *get() const noexcept {
        if( !m_pref) return nullptr;
        if constexpr (ref_traits<T>::is_intrusive) {
            return m_pref;                  //!< Intrusive: Ref IS T.
        } else if constexpr (ref_traits<T>::is_emplaced) {
            return m_pref->ptr_();
        } else {
            return m_pref->ptr;
        }
    }
    T *operator->() const noexcept { assert(m_pref); return get(); }
    T &operator*() const noexcept { assert(m_pref); return *get(); }

    Ref *ref_ptr_() const noexcept { return m_pref; }

    //! Identity comparison against atomic_shared_ptr (e.g. Linkage).
    //! Pure relaxed load + pointer comparison — no load_shared_,
    //! no refcount manipulation.  scoped_atomic_view is a friend of
    //! atomic_shared_ptr, so ref_ptr_() access is valid.
    //! Reverse direction (asp != view) uses inherited
    //! local_shared_ptr::operator!=(scoped_atomic_view) — no friend
    //! needed (adding one would create ambiguity with the inherited
    //! member via Linkage's inheritance chain).
    bool operator==(const atomic_shared_ptr<T> &rhs) const noexcept {
        return m_pref == rhs.ref_ptr_();
    }
    bool operator!=(const atomic_shared_ptr<T> &rhs) const noexcept {
        return m_pref != rhs.ref_ptr_();
    }

private:
    //! Promote TagHeld → Owned via "zero-reset": load current tag
    //! count and drain ALL of them, transferring rcnt_now refs to
    //! global in a single fetch_add + drain CAS.
    //!
    //! Compared to the previous "use rcnt_at_acquire" pattern:
    //!   - Same atomic op count on the success path
    //!     (1 fetch_add + 1 CAS via release_tag_ref_, drains all,
    //!     sub_amount = 0, no extra fetch_sub)
    //!   - Captures CURRENT state (others' tags acquired AFTER our
    //!     acquire are also drained), helping keep tag bits at 0 for
    //!     subsequent acquires
    //!   - On ptr-change (swapper absorbed our tag) or rcnt_now == 0
    //!     (some drainer already absorbed us), fall back to plain
    //!     fetch_add(1) — our tag is already accounted in global.
    //!
    //! Caller is responsible for setting m_tag_held = false after,
    //! since this function only handles the atomic-state transition.
    void promote_tagheld_() noexcept {
        auto [cur_ptr, rcnt_now] = m_asp->load_tagged_();
        if(cur_ptr == m_pref && rcnt_now > 0) {
            // Pre-pay rcnt_now to global: covers all rcnt_now tag
            // holders (us + others present at this moment).  Drain
            // CAS in release_tag_ref_ tries to remove rcnt_now tags;
            // sub_amount = rcnt_now - drained, so net global change
            // = drained.  Our own +1 is part of those drained refs.
            m_pref->refcnt.fetch_add(rcnt_now, std::memory_order_relaxed);
            m_asp->release_tag_ref_(m_pref, rcnt_now);
        } else {
            // ptr changed (swapper absorbed) or tag already drained
            // (another load_shared_ / promote took our tag and
            // converted it to global).  Either way our +1 is in
            // global; just add 1 more for the Owned ref we want.
            // Wait — we already had +1 absorbed; we need to gain
            // a +1 for "Owned" mode.  Since absorption transferred
            // our tag's implicit ref to global, we already have it.
            // No fetch_add needed.
        }
    }

    //! Release TagHeld via "zero-reset": load current tag count and
    //! drain ALL tags, paying others' IOUs to global.  After this
    //! call tag bits are 0 (assuming no race), letting other threads'
    //! acquires succeed without weakly-failing on capacity.
    //!
    //! Atomic op count: 1 fetch_add + 1 CAS (via release_tag_ref_).
    //! Compared to the simple release_tag_ref_(pref, 1) (1 CAS),
    //! this costs +1 op per release but bounds tag accumulation —
    //! crucial under high-contender:capacity ratios.
    //!
    //! Math (state at call: tag = T including our +1, global = G):
    //!   - Pre-pay (T-1) to global: G' = G + T - 1
    //!   - release_tag_ref_(pref, T) drains drained tags;
    //!     sub_amount = T - drained, fetch_sub.
    //!   - Net global: G + T - 1 - (T - drained) = G + drained - 1
    //!     - Full drain (drained = T): G + T - 1, tag = 0.  ✓
    //!     - Partial drain: G + drained - 1, tag = T - drained.  ✓
    //!     - Fall-through (ptr changed): drained = 0, sub_amount = T,
    //!       net global = G - 1, tag wherever.  ✓ (our +1 was absorbed
    //!       by swapper into G already; -1 releases our share)
    //! All cases: net true ref change = -1.  Verified by induction
    //! on the standard refcnt invariant (true_refs = global + tag
    //! when m_ref still points to pref).
    bool release_tagheld_zeroreset_(bool single_attempt) noexcept {
        auto [cur_ptr, rcnt_now] = m_asp->load_tagged_();
        if(cur_ptr == m_pref && rcnt_now > 0) {
            if(rcnt_now > 1) {
                m_pref->refcnt.fetch_add(rcnt_now - 1,
                    std::memory_order_relaxed);
            }
            return m_asp->release_tag_ref_(m_pref, rcnt_now, single_attempt);
        }
        // ptr changed or tag already drained — our +1 is in global.
        if(m_pref->refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            m_asp->deleter(m_pref);
        }
        return true;
    }

public:
    //! Single-attempt release.  Returns true on success (view becomes
    //! empty); false if drain CAS lost (view stays valid; caller must
    //! retry).  Tracks rcnt_added across iterations so we only
    //! fetch_add / fetch_sub the DIFF when observed tag count changes.
    //! All fetch_sub use acq_rel + delete check (memory ordering
    //! correctness even though the typical case won't drop refcnt to 0).
    //! Caller initialises rcnt_added=0 before first call.
    bool try_release_single_attempt(uintptr_t &rcnt_added) noexcept {
        if( !m_pref) return true;
        // Helper: skip fetch_sub when sub == 0 (a fetch_sub(0, acq_rel)
        // is not a no-op — its delete check fires if OLD refcnt == 0,
        // a race that can occur if m_ref was reset between our CAS and
        // this fetch_sub).  Captures m_pref and m_asp for deleter.
        auto sub_with_delete_check = [this](uintptr_t sub) {
            if(sub) {
                if(m_pref->refcnt.fetch_sub(sub,
                        std::memory_order_acq_rel) == sub) {
                    m_asp->deleter(m_pref);
                }
            }
        };
        if( !m_tag_held) {
            // Owned: plain fetch_sub(1).  Plus undo any pre-pay (Owned
            // mode shouldn't have pre-pay normally; safety).
            sub_with_delete_check(rcnt_added + 1);
            m_pref = nullptr;
            rcnt_added = 0;
            return true;
        }
        auto [cur_ptr, rcnt_now] = m_asp->load_tagged_();
        if(cur_ptr != m_pref || rcnt_now == 0) {
            // ptr changed (swapper absorbed our +1) or tag drained.
            // Release our +1 (now in refcnt) + undo our pre-pay.
            sub_with_delete_check(rcnt_added + 1);
            m_pref = nullptr;
            m_tag_held = false;
            rcnt_added = 0;
            return true;
        }
        // TagHeld + ptr unchanged.  One-shot pre-pay LOCAL_REF_CAPACITY
        // (an upper bound — rcnt_now never exceeds CAPACITY-1, so
        // CAP-1 + slack always covers needed).  After the first entry,
        // rcnt_added stays at CAP and the branch is never re-entered.
        uintptr_t needed = rcnt_now - 1;
        if(needed > rcnt_added) {
            // This route fires at most once per loop.
            m_pref->refcnt.fetch_add(LOCAL_REF_CAPACITY,
                std::memory_order_relaxed);
            rcnt_added = LOCAL_REF_CAPACITY;
        }
        // Single-shot drain CAS: tag rcnt_now → 0.
        if(const_cast<atomic_shared_ptr<T> *>(m_asp)->m_ref.compare_set_weak(
            (uintptr_t)m_pref + rcnt_now,
            (uintptr_t)m_pref + 0)) {
            // Adjust for excess pre-pay.
            sub_with_delete_check(rcnt_added - needed);
            rcnt_added = 0;
            m_pref = nullptr;
            m_tag_held = false;
            return true;
        }
        return false;
    }

private:
    void release_() noexcept {
        if(m_pref) {
            if(m_tag_held) {
                (void)release_tagheld_zeroreset_(/*single_attempt=*/false);
            } else {
                // Owned → plain fetch_sub(1) + delete check.
                if(m_pref->refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    m_asp->deleter(m_pref);
                }
            }
            m_pref = nullptr;
            m_tag_held = false;
        }
    }

    atomic_shared_ptr<T> *m_asp;
    Ref *m_pref;
    bool m_tag_held;
    bool m_acquire_succeeded;

    template <typename Y> friend class atomic_shared_ptr;
};

//! \brief Construct a `local_shared_ptr<T>` with `T(args...)`.
//!
//! Compile-time dispatch via `ref_traits<T>`:
//!   * `atomic_emplaced`   → single alloc, T placed in `gref_weakable_`.
//!   * `atomic_countable`  → single alloc, T itself is the control block
//!                           (refcnt set by `atomic_countable`'s ctor).
//!   * `atomic_strictrefonly` or default → 2 allocs (T + Ref).
template <typename T, class... Args>
local_shared_ptr<T> make_local_shared(Args&&... args) {
    if constexpr (ref_traits<T>::is_emplaced) {
        using Ref = atomic_shared_ptr_gref_weakable_<T>;
        Ref *r = new Ref(std::forward<Args>(args)...);
        return local_shared_ptr<T>(
            typename local_shared_ptr<T>::adopt_promoted_t{}, r);
    }
    else {
        //!< Non-emplaced — `local_shared_ptr<T>(T*)` adopts the raw
        //!< pointer.  Intrusive: 1 alloc; default/strict: 2 allocs.
        //!< strictrefonly T: gref_strictrefonly_ without weak.
        return local_shared_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

template <typename T, class Alloc, class... Args>
local_shared_ptr<T> allocate_local_shared(Alloc &base_alloc, Args&&... args) {
    typename Alloc::template rebind<T>::other alloc(base_alloc);
    auto p = alloc.allocate(1);
    alloc.construct(p, std::forward<Args>(args)...);
    auto deleter = [alloc, p]() mutable {
        alloc.destroy(p);
        alloc.deallocate(p, 1);
    };
    return local_shared_ptr<T>(p, deleter);
}

template <typename T, typename reflocal_var_t>
inline local_shared_ptr<T, reflocal_var_t>::local_shared_ptr(const local_shared_ptr &y) noexcept {
    static_assert(sizeof(static_cast<const T*>(y.get())), "");
    this->m_ref = (TaggedPtr)y.m_ref;
    if(Ref *p = ref_ptr_()) {
        if constexpr (is_biased_directpublish<T>::value) biased_inc_(p->refcnt); //!< §biased — skippable
        else p->refcnt.fetch_add(1, std::memory_order_relaxed);
    }
}

template <typename T, typename reflocal_var_t>
template<typename Y, typename Z>
inline local_shared_ptr<T, reflocal_var_t>::local_shared_ptr(const local_shared_ptr<Y, Z> &y) noexcept {
    static_assert(sizeof(static_cast<const T*>(y.get())), "");
    this->m_ref = (TaggedPtr)y.m_ref;
    if(Ref *p = ref_ptr_()) {
        if constexpr (is_biased_directpublish<T>::value) biased_inc_(p->refcnt); //!< §biased — skippable
        else p->refcnt.fetch_add(1, std::memory_order_relaxed);
    }
}

template <typename T, typename reflocal_var_t>
inline local_shared_ptr<T, reflocal_var_t>::~local_shared_ptr() {
    reset();
}

template <typename T, typename reflocal_var_t>
inline void
local_shared_ptr<T, reflocal_var_t>::reset() noexcept {
    Ref *pref = ref_ptr_();
    if( !pref) return;
    if constexpr (is_biased_directpublish<T>::value) {        //!< §biased — skippable block
        if(biased_dec_is_dead_(pref->refcnt)) this->deleter(pref);
        this->m_ref = (TaggedPtr)nullptr;
        return;
    }
    // decreases global reference counter.
    if(unique()) {
        pref->refcnt.store(0, std::memory_order_relaxed);
        this->deleter(pref);
    }
    else if(pref->refcnt.decAndTest()) {
        this->deleter(pref);
    }
    this->m_ref = (TaggedPtr)nullptr;
}
//=============================================================================
// acquire_tag_ref_() — atomically read m_ref and increment the local refcount
//   (Comments by Claude Opus — based on source code analysis)
//
// The local refcount is embedded in the lower bits of the pointer (the bits
// guaranteed zero by allocator alignment). Incrementing it via CAS tells
// other threads "someone is observing this pointer — don't free the Ref yet"
// without touching the global (heap-allocated) reference counter.
//
// Invariant: local refcount < LOCAL_REF_CAPACITY. If the counter
// would overflow (extremely unlikely — requires that many concurrent readers),
// spin-wait until a slot opens.
//
// The caller MUST call release_tag_ref_() after it is done with the pointer.
//=============================================================================
template <typename T>
inline std::pair<typename atomic_shared_ptr<T>::Ref *, bool>
atomic_shared_ptr<T>::acquire_tag_ref_(Refcnt *rcnt, bool weakly) const noexcept {
    Ref *pref;
    Refcnt rcnt_new;
    for(int spins = 1;; spins *= 2) {
        auto [p, rcnt_old] = load_tagged_();
        pref = p;
        if( !pref) {
            // target is null.
            *rcnt = rcnt_old;
            return {(Ref*)nullptr, true};
        }
        rcnt_new = rcnt_old + 1u;
        /*
        static int rcnt_max = 0;
        if(rcnt_new > (int)rcnt_max) {
            rcnt_max = rcnt_new;
            fprintf(stderr, "max_rcnt=%d\n", rcnt_max);
        }
        */
        // Weak callers fail-fast on either overflow OR CAS-loss with
        // no pause (caller retries at a higher level).
        if(weakly) {
            if(rcnt_new < this->LOCAL_REF_CAPACITY
               && const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_weak(
                   TaggedPtr((uintptr_t)pref + rcnt_old),
                   TaggedPtr((uintptr_t)pref + rcnt_new)))
                break;
            return {(Ref*)nullptr, false};
        }
        // Strong path: pause on overflow, exponential backoff on CAS loss.
        if(rcnt_new < this->LOCAL_REF_CAPACITY) {
            if(const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_weak(
                TaggedPtr((uintptr_t)pref + rcnt_old),
                TaggedPtr((uintptr_t)pref + rcnt_new)))
                break;
        }
        else {
            pause4spin();
        }
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
        for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
            pause4spin(); //exponential backoff.
#else
        (void)spins;
#endif
    }
    assert(rcnt_new);
    *rcnt = rcnt_new;
    return {pref, true};
}
template <typename T>
inline typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::load_shared_() const noexcept {
    static_assert(load_shared_enabled<T>::value,
        "load_shared_ is disabled for this type; use scoped_atomic_view instead");
    Refcnt rcnt;
    auto [pref, success] = acquire_tag_ref_( &rcnt);
    if( !pref) return (Ref*)nullptr;
    // Transfer all rcnt tag refs to global at once (instead of just +1). The
    // matching release_tag_ref_(pref, rcnt) attempts to drain rcnt tag refs in
    // a single CAS, stealing from other threads' tag refs. Those threads then
    // fall back to fetch_sub(1) on refcnt on their own release_tag_ref_, which
    // shifts contention from m_ref (the hot tagged pointer) to refcnt (per
    // object). When rcnt=1 this is identical to the previous +1 / K=1 pattern.
    pref->refcnt.fetch_add(rcnt, std::memory_order_relaxed);
    release_tag_ref_(pref, rcnt);
    return pref;
}

// release_tag_ref_() — release the local reference acquired by acquire_tag_ref_().
// Tries to decrement the local refcount via CAS. If the pointer has been
// swapped out by another thread since acquire_tag_ref_(), the local counter
// is gone — fall back to decrementing the global refcount instead (the
// swapper transferred local counts to global before swapping).
//
// added_global_rcnt (default 1): total number of global refcount units that
//   the caller has pre-added to pref->refcnt on top of the 1 local ref being
//   released. Callers use this to batch-release excess refs in one shot:
//
//   compareAndSwap_ / compareAndSwapWeak_ / swap (CAS failure path):
//     Step 4 pre-added (rcnt_old - 1) to global. Pass added_global_rcnt=rcnt_old
//     to release that excess and our own local ref in a single operation.
//
//   load_shared_ (rcnt-bulk transfer):
//     fetch_add(rcnt) pre-adds rcnt to global (instead of the usual +1). Pass
//     added_global_rcnt=rcnt so the tag drain and global undo are consistent.
//
// Same-pointer CAS success path:
//   Drains local_release = min(rcnt_old, added_global_rcnt) from the tag.
//   Remaining excess = (added_global_rcnt - local_release) is undone from global
//   via fetch_sub(excess, acq_rel) with a delete check. MUST be acq_rel (not
//   relaxed): a concurrent local_reset can drop refcnt to exactly the excess
//   amount; our fetch_sub would then reach 0 and the deleter must fire.
//
// Pointer-changed path:
//   Combines excess undo + our own 1 ref into a single fetch_sub(added_global_rcnt,
//   acq_rel) with delete check — one fewer atomic op vs. the two-step old code.
template <typename T>
inline bool atomic_shared_ptr<T>::release_tag_ref_(Ref *pref, Refcnt added_global_rcnt,
                                                    bool single_attempt) const noexcept {
    Refcnt sub_amount = added_global_rcnt;
    for(int spins = 1;; spins *= 2) {
        auto [cur_ptr, rcnt_old] = load_tagged_();
        if(rcnt_old && (cur_ptr == pref)) {
            Refcnt local_release = std::min(rcnt_old, added_global_rcnt); //1 by default.
            Refcnt rcnt_new = rcnt_old - local_release;
            // trying to dec. reference counter if stored pointer is unchanged.
            if(const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_weak(
                TaggedPtr((uintptr_t)pref + rcnt_old),
                TaggedPtr((uintptr_t)pref + rcnt_new))) {
                //decreases the rest of global counting.
                // CRITICAL: must be acq_rel + delete check, NOT relaxed.
                // Concurrent local_reset() can drop refcnt to (added_global_rcnt -
                // local_release), and our fetch_sub then takes it to 0. Without the
                // delete check the object leaks. Discovered via GenMC test 7
                sub_amount = added_global_rcnt - local_release;
                break;
            }
            // CAS lost.  single_attempt callers return false WITHOUT
            // doing the global fetch_sub — caller's pre-pay IOU stays
            // in pref->refcnt and is balanced by a later call.
            if(single_attempt)
                return false;
            auto [cur_ptr, rcnt_old] = load_tagged_();
            if((cur_ptr == pref) && rcnt_old) {
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
                for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
                    pause4spin(); //exponential backoff.
#else
                (void)spins;
#endif
                continue; // pointer unchanged, retry.
            }
        }
        // local reference has released by other processes.
        break;
    }
    if(sub_amount) {
        if(pref->refcnt.fetch_sub(sub_amount, std::memory_order_acq_rel) == sub_amount) {
            const_cast<atomic_shared_ptr*>(this)->deleter(pref);
        }
    }
    return true;
}

//=============================================================================
// compareAndSet_impl_<OldrT, WEAK, RETAIN_NEWR>() — unified atomic CAS on the shared pointer
//
// OldrT-driven dispatch via constexpr if:
//   - OldrT = const local_shared_ptr<T> (Set):
//       no acquire (oldr keeps pref alive); step4 = +T;
//       failure undo via plain fetch_sub(T, relaxed).
//   - OldrT = local_shared_ptr<T> (Swap, ACQUIRE):
//       acquire_tag_ref_() to pin pref while updating oldr on mismatch;
//       step4 = +(T-1); failure undo via release_tag_ref_(pref, T).
//   - OldrT = scoped_atomic_view<T> (SetScoped, WEAK only):
//       scoped already holds tag; step4 = +(T-1);
//       failure undo via plain fetch_sub(T-1, relaxed) (eager); on success,
//       scoped's tag is consumed by CAS (m_pref reset to nullptr).
//
// RETAIN_NEWR (SCOPED+WEAK only): on CAS success, scoped transitions to
//   Owned mode on newr instead of going Empty.  Entry does fetch_add(2)
//   instead of (1) — one for m_ref, one for scoped's Owned ref.  Failure
//   undo is fetch_sub(2) (same op count, different amount).  Useful when
//   the caller needs to keep tracking the new value after CAS (e.g.
//   bundle Phase 2 → Phase 4: scoped retains newr, eliminating reload).
//
// Common steps:
//   1. Pre-increment newr's global refcount (optimistic).
//   2. Read the current pointer (acquire_tag_ref_ or load_tagged_).
//   3. Pointer mismatch → undo + return false.
//   4. Pre-pay other tag holders via fetch_add(step4_amount).
//   5. CAS m_ref: (pref + rcnt_old) → (newr + 0).
//      On failure, undo step 4; retry (or return false if WEAK).
//   6. On success, decrement pref's global refcount (m_ref no longer owns it).
//=============================================================================
template <typename T>
template<typename OldrT, typename NewrT, bool WEAK, bool RETAIN_NEWR>
inline bool
atomic_shared_ptr<T>::compareAndSet_impl_(
    OldrT &oldr,
    NewrT &newr) noexcept {

    using OldrPlain = typename std::remove_cv<OldrT>::type;
    using NewrPlain = typename std::remove_cv<NewrT>::type;
    // `static constexpr` (not plain constexpr): these are used inside the
    // lambdas below (`if constexpr (SCOPED)`, NEWR_ADD in new_refcnt_undo).
    // A function-local constexpr is a constant on GCC/Clang and needs no
    // capture, but MSVC rejects that (C2131 in `if constexpr`, C3493 "cannot
    // be implicitly captured").  Static storage makes them true constants
    // usable in nested lambdas on every compiler; the value is identical.
    static constexpr bool SCOPED = std::is_same<OldrPlain, scoped_atomic_view<T>>::value;
    static constexpr bool ACQUIRE = !std::is_const<OldrT>::value && !SCOPED;
    // SCOPED + STRONG: enabled for the privileged-thread fast path.
    // Privilege is exclusive (s_privileged_tidstamp slot) and fair_mode
    // blocks all other threads' CAS on this linkage, so the strong-spin
    // has no peer to contend with — guaranteed forward progress, no
    // livelock. Caller is responsible for invoking strong-mode only
    // when privileged.
    static_assert( !(RETAIN_NEWR && !SCOPED),
        "RETAIN_NEWR requires SCOPED (scoped_atomic_view oldr)");

    auto oldr_pref = [&]() -> Ref* {
        if constexpr (SCOPED) return oldr.m_pref;
        else return oldr.ref_ptr_();
    };
    auto newr_pref = [&]() -> Ref* {
        return newr.ref_ptr_();
    };
    // RETAIN_NEWR adds +1 for scoped's Owned ref on newr after CAS success.
    static constexpr Refcnt NEWR_ADD = RETAIN_NEWR ? 2u : 1u;  // static: see SCOPED note
    auto new_refcnt_undo = [&newr]() {
        if(newr.ref_ptr_()) {
            if constexpr ( !RETAIN_NEWR) {
                if(newr.use_count() == 2) //unique at start pt., and was +1.
                    { newr.ref_ptr_()->refcnt--; return; }
            }
            newr.ref_ptr_()->refcnt.fetch_sub(NEWR_ADD, std::memory_order_relaxed);
        }
    };

    //!< §biased — skippable: newr is about to be installed into this atomic slot,
    //!< so a biased CB is negated -count→+count BEFORE the optimistic count bump.
    if constexpr (is_biased_directpublish<T>::value)
        if(Ref *np = newr_pref()) biased_publish_(np->refcnt);
    // Optimistic +NEWR_ADD for m_ref's implicit ref (+ scoped's Owned
    // ref when RETAIN_NEWR) on success; will undo on WEAK-failure or
    // pointer-mismatch.
    if(newr_pref()) {
        if constexpr ( !RETAIN_NEWR) {
            if(newr.unique())
                { newr_pref()->refcnt++; }
            else
                newr_pref()->refcnt.fetch_add(1, std::memory_order_relaxed);
        }
        else {
            newr_pref()->refcnt.fetch_add(NEWR_ADD, std::memory_order_relaxed);
        }
    }
    for(int spins = 1;; spins *= 2) {
        Ref *pref;
        Refcnt rcnt_old;

        if constexpr (ACQUIRE) {
            auto [p, success] = acquire_tag_ref_( &rcnt_old, WEAK);
            if constexpr (WEAK) {
                if( !success) {
                    new_refcnt_undo();
                    return false;
                }
            }
            pref = p;
        } else {
            std::tie(pref, rcnt_old) = load_tagged_();
        }

        if(pref != oldr_pref()) {
            // pointer mismatch
            if constexpr (ACQUIRE) {
                if(pref) {
                    pref->refcnt.fetch_add(1, std::memory_order_relaxed);
                    release_tag_ref_(pref, 1u);
                }
            }
            new_refcnt_undo();
            if constexpr (ACQUIRE) {
                if(oldr.ref_ptr_()) {
                    // decreasing global reference counter.
                    if(oldr.ref_ptr_()->refcnt.decAndTest()) {
                        this->deleter(oldr.ref_ptr_());
                    }
                }
                oldr.m_ref = (uintptr_t)pref;
            } else if constexpr (SCOPED) {
                // For TagHeld: pointer changed since acquire; our tag was
                //   absorbed by the swapper (their step 4 pre-paid us +1).
                //   Eagerly clean up so scoped becomes Empty and the caller
                //   can detect "tag gone" via scoped.operator bool() == false.
                // For Owned: scoped still holds its +1 in OLD pref's refcnt.
                //   No special cleanup; caller sees scoped as still valid
                //   but the CAS returned false (caller may retry or destruct).
                if(oldr.m_tag_held) {
                    // TagHeld
                    release_tag_ref_(oldr.m_pref, 1u);
                    oldr.m_pref = nullptr;
                    oldr.m_tag_held = false;
                }
            }
            return false;
        }

        // step 4: pre-pay other tag holders.
        //
        //   - Swap (ACQUIRE): step4 = +(T-1) — own acquired tag is consumed
        //     by the CAS (no pre-pay needed for self).
        //   - Set: step4 = +T — caller's oldr keeps pref alive separately
        //     via its +1 in refcnt, so all T tag holders are external.
        //   - SCOPED: step4 = +T — treat scoped's tag as if it were already
        //     +1 in refcnt (because: in the ABSORBED case our CAS will
        //     consume scoped's tag along with others, and we owe -1 in the
        //     success path; in the DRAINED case some drainer already
        //     pre-paid scoped +1, and we likewise owe -1). Either way,
        //     a fetch_sub(2) at success consumes both m_ref's release and
        //     scoped's tag-share uniformly. This avoids needing to detect
        //     ABSORBED vs DRAINED at runtime.
        Refcnt step4_amount;
        if constexpr (ACQUIRE) {
            step4_amount = (rcnt_old > 1u) ? rcnt_old - 1u : 0u;
        } else {
            step4_amount = rcnt_old;
        }
        if(pref && step4_amount) {
            pref->refcnt.fetch_add(step4_amount, std::memory_order_relaxed);
        }

        // CAS m_ref: pref + rcnt_old → newr + 0
        Refcnt rcnt_new = 0;
        if(this->m_ref.compare_set_weak(
                TaggedPtr((uintptr_t)pref + rcnt_old),
                TaggedPtr((uintptr_t)newr_pref() + rcnt_new))) {
            if(pref) {
                // Release m_ref's implicit ownership.
                // For SCOPED in TagHeld mode, additionally consume scoped's
                // tag-share in the same fetch_sub (sub = 2). For Owned mode
                // and non-SCOPED, scoped's/oldr's +1 stays — sub = 1.
                //   TagHeld (refcnt >= 2 always at this point):
                //     ABSORBED: step4=+T pre-paid T including scoped →
                //       refcnt >= R_init + T - (T-1) = R_init + 1 >= 2.
                //     DRAINED:  drainer pre-paid +1 → R_init >= 2;
                //       step4=+T (T>=0) → refcnt >= 2.
                // RETAIN_NEWR + Owned: scoped is reassigned to newr below,
                //   so its Owned +1 on OLD pref must also be released here
                //   (sub = 2).  Without this, OLD pref's refcnt leaks +1
                //   per Owned-RETAIN call — the bug appears at low
                //   LOCAL_REF_CAPACITY where Owned mode is hit frequently
                //   (e.g., CAP=4 ADAPTIVE=2 → any rcnt>=2 acquire promotes).
                Refcnt sub = 1u;
                if constexpr (SCOPED) {
                    if(oldr.m_tag_held) sub = 2u;  // TagHeld
                    else if constexpr (RETAIN_NEWR) sub = 2u;  // Owned + RETAIN
                    // else Owned non-RETAIN: sub = 1 (scoped keeps OLD)
                }
                if(pref->refcnt.fetch_sub(sub, std::memory_order_acq_rel) == sub) {
                    const_cast<atomic_shared_ptr*>(this)->deleter(pref);
                }
            }
            if constexpr (SCOPED) {
                if constexpr (RETAIN_NEWR) {
                    // Transition scoped to Owned(newr).  The extra +1
                    // from NEWR_ADD at entry provides the Owned ref.
                    if(oldr.m_tag_held) oldr.m_tag_held = false;
                    oldr.m_pref = newr_pref();
                    // oldr is now Owned on newr.  Destructor will
                    // release_tag_ref_(newr, 1u) → fetch_sub(1).
                } else {
                    if(oldr.m_tag_held) {
                        // TagHeld: tag-share consumed via fetch_sub(2). Mark Empty.
                        oldr.m_pref = nullptr;
                        oldr.m_tag_held = false;
                    }
                    // Owned: scoped retains its +1 in pref's refcnt — but pref is
                    //   no longer in m_ref. Caller may still hold it as if from
                    //   compareAndSet on a const local_shared_ptr; destructor
                    //   eventually releases via fetch_sub(1).
                }
            }
            return true;
        }

        // CAS failure — undo step 4.
        if constexpr (ACQUIRE) {
            // Swap: batch undo via release_tag_ref_(pref, rcnt_old)
            //   = drain CAS for tag (rcnt_old refs) + global undo combined.
            if(pref) {
                assert(rcnt_old);
                release_tag_ref_(pref, rcnt_old);
            }
        } else {
            // Set / SCOPED: a held ref keeps pref alive (refcnt >= 2),
            //   so plain relaxed fetch_sub is safe.
            //   - Set:    caller's oldr provides the +1.
            //   - SCOPED: scoped's tag is still held (in m_ref's tag count
            //             OR pre-paid by an external drainer); destructor's
            //             release_tag_ref_(pref, 1u) handles cleanup.
            if(pref && step4_amount) {
                pref->refcnt.fetch_sub(step4_amount, std::memory_order_relaxed);
            }
        }
        if constexpr (WEAK) {
            // Roll back the optimistic newr fetch_add(1) from the entry of
            // this function — STRONG mode keeps it across retries, but WEAK
            // returns false without retry, so the +1 must be undone.
            new_refcnt_undo();
            return false;
        }
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
        for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
            pause4spin(); //exponential backoff.
#else
        (void)spins;
#endif
    }
}

template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSwap(local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<local_shared_ptr<T>, const local_shared_ptr<T>, false>(oldr, newr);
}
template <typename T>
bool
atomic_shared_ptr<T>::compareAndSet(const local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<const local_shared_ptr<T>, const local_shared_ptr<T>, false>(oldr, newr);
}
template <typename T>
bool
atomic_shared_ptr<T>::compareAndSetWeak(const local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<const local_shared_ptr<T>, const local_shared_ptr<T>, true>(oldr, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetWeak(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<scoped_atomic_view<T>, const local_shared_ptr<T>, true>(scoped, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetWeakRetain(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<scoped_atomic_view<T>, const local_shared_ptr<T>, true, true>(scoped, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetStrong(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newr) noexcept {
    // STRONG (WEAK=false): the impl_'s outer for-loop spins on weak CAS
    // failure.  Pointer mismatch (oldr.m_pref != m_ref's load) returns
    // false unconditionally — the only "real contention" exit.
    return compareAndSet_impl_<scoped_atomic_view<T>, const local_shared_ptr<T>, false, false>(scoped, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetStrongRetain(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<scoped_atomic_view<T>, const local_shared_ptr<T>, false, true>(scoped, newr);
}

template <typename T, typename reflocal_var_t>
inline void
local_shared_ptr<T, reflocal_var_t>::swap(local_shared_ptr &r) noexcept {
    TaggedPtr x = this->m_ref;
    this->m_ref = (TaggedPtr)r.m_ref;
    r.m_ref = x;
}

template <typename T, typename reflocal_var_t>
void
local_shared_ptr<T, reflocal_var_t>::swap(atomic_shared_ptr<T> &r) noexcept {
    //!< §biased — skippable: this's CB is about to enter the atomic slot r, so a
    //!< biased CB is negated -count→+count (release) before the install CAS.
    if constexpr (is_biased_directpublish<T>::value)
        if(Ref *sp = ref_ptr_()) biased_publish_(sp->refcnt);
    for(int spins = 1;; spins *= 2) {
        Refcnt rcnt_old, rcnt_new;
        auto [pref, success] = r.acquire_tag_ref_( &rcnt_old);
        if(pref && (rcnt_old != 1u)) {
            pref->refcnt.fetch_add(rcnt_old - 1u, std::memory_order_relaxed);
        }
        rcnt_new = 0;
        if(r.m_ref.compare_set_weak(
            TaggedPtr((uintptr_t)pref + rcnt_old),
            TaggedPtr((uintptr_t)this->m_ref + rcnt_new))) {
            this->m_ref = (TaggedPtr)pref;
            return;
        }
        if(pref) {
            assert(rcnt_old);
            r.release_tag_ref_(pref, rcnt_old);
        }
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
        for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
            pause4spin(); //exponential backoff.
#else
        (void)spins;
#endif
    }
}

//! Atomic access for a copy-able class which does not require
//! transactional writing.  General-case specialization of the
//! `atomic<T>` template forward-declared in `atomic.h`.  Lives here
//! (rather than in atomic.h) because it depends on
//! `atomic_shared_ptr` / `local_shared_ptr` which are defined above
//! in this file; placing the definition in atomic.h would create a
//! cyclic include (`atomic_smart_ptr.h` already #includes `atomic.h`
//! for the integral specialization).
template <typename T, class Enable>
class atomic {
public:
    atomic() : m_var(new T) {}
    atomic(T t) : m_var(new T(t)) {}
    atomic(const atomic &t) noexcept : m_var(t.m_var) {}
    operator T() const noexcept {
        local_shared_ptr<T> x = m_var;
        return *x;
    }
    atomic &operator=(T t) {
        m_var.reset(new T(t));
        return *this;
    }
    atomic &operator=(const atomic &x) noexcept {
        m_var = x.m_var;
        return *this;
    }
    bool compare_set_strong(const T &oldv, const T &newv) {
        local_shared_ptr<T> oldx(m_var);
        if( *oldx != oldv)
            return false;
        local_shared_ptr<T> newx(new T(newv));
        bool ret = m_var.compareAndSet(oldx, newx);
        return ret;
    }
protected:
    atomic_shared_ptr<T> m_var;
};

#endif /*ATOMIC_SMART_PTR_H_*/
