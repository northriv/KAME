/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            ../LICENSE-APACHE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see ../LICENSE-GPL-2.0).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/

/*
 * fast_vector_test — union-discipline regression test for
 * Transactional::fast_vector<T, N>.
 *
 * fast_vector stores up to max_fixed_size elements inline in `T m_array[]`,
 * union'd with a std::vector<T> used only past that point.  `is_fixed()`
 * true means the INLINE array is the active union member, and m_vector must
 * not be touched at all -- not even a const capacity()/size() read.
 *
 * The bug this pins (2026-08-08): shrink_to_fit() had the test inverted
 *
 *     if( !is_fixed()) return;        // returns when a real vector EXISTS
 *     if(m_vector.capacity() - m_vector.size() > max_fixed_size)
 *         m_vector.shrink_to_fit();   // runs when m_vector is INACTIVE
 *
 * so it read m_array's bytes reinterpreted as a std::vector's three
 * pointers and then reallocated through them.  Talker::connect and
 * Talker::disconnect (transaction_signal.h) call shrink_to_fit() on every
 * connect/disconnect, and a Talker with <= max_fixed_size listeners -- the
 * normal case -- is always in inline mode.  Whether it actually corrupted
 * memory depended on the stale bytes past the live elements, so it
 * presented as a sporadic SIGSEGV inside Talker::disconnect at shutdown
 * rather than a deterministic failure.
 *
 * Because the failure is byte-pattern dependent, this test deliberately
 * poisons the storage before placement-new so the inactive-member read has
 * garbage to find, and sweeps element counts across the inline/heap
 * boundary.  Run it under -fsanitize=address,undefined for full strength:
 * the pre-fix header trips UBSan inside std::vector::capacity() even on
 * the runs where it does not crash outright.
 */

#include "support_standalone.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "fast_vector.h"

using Transactional::fast_vector;

static int g_failures = 0;

#define CHECK(cond) do { \
    if( !(cond)) { \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while(0)

// Mirrors the instantiation that crashed: Talker's listener list is a
// fast_vector<std::weak_ptr<ListenerBase>, 1>.  With sizeof(weak_ptr)==16
// on LP64, max_fixed_size works out to 4, so counts 0..4 exercise inline
// mode and 5+ exercise the std::vector fallback.
using W = std::weak_ptr<int>;
using FV = fast_vector<W, 1>;

//! Build a fast_vector on deliberately poisoned storage, so that any read
//! of the inactive union member sees plausible-looking garbage rather than
//! zeroes.  Returns via placement new into \a buf.
static FV *make_poisoned(void *buf, unsigned char pattern) {
    std::memset(buf, pattern, sizeof(FV));
    return new (buf) FV();
}

//! size(), element identity and iteration must agree in both storage modes.
static void check_contents(FV &v, const std::vector<std::shared_ptr<int>> &keep,
                           int n, const char *where) {
    CHECK(v.size() == (size_t)n);
    CHECK(v.empty() == (n == 0));
    for(int i = 0; i < n; ++i) {
        auto sp = v[i].lock();
        if( !sp || *sp != i) {
            std::printf("FAIL %s: element %d wrong\n", where, i);
            ++g_failures;
        }
    }
    // begin()/end() must span exactly size() elements -- and must not form
    // an out-of-range subscript when empty (the &m_vector[0] /
    // &m_vector[size()] spellings did).
    CHECK((size_t)(v.end() - v.begin()) == (size_t)n);
    int seen = 0;
    for(auto it = v.begin(); it != v.end(); ++it) ++seen;
    CHECK(seen == n);
    (void)keep;
}

int main() {
    std::vector<std::shared_ptr<int>> keep;
    for(int i = 0; i < 64; ++i) keep.push_back(std::make_shared<int>(i));

    std::printf("sizeof(weak_ptr)=%zu sizeof(std::vector<W>)=%zu sizeof(FV)=%zu\n",
                sizeof(W), sizeof(std::vector<W>), sizeof(FV));

    // (1) shrink_to_fit() must be safe at every size, in both modes, on
    //     poisoned storage.  This is the direct regression for the
    //     inverted is_fixed() test.
    for(int n = 0; n <= 40; ++n) {
        alignas(FV) unsigned char buf[sizeof(FV)];
        FV *v = make_poisoned(buf, 0xAB);
        for(int i = 0; i < n; ++i) v->push_back(W(keep[i]));
        check_contents( *v, keep, n, "after fill");

        v->shrink_to_fit();                 // <-- crashed pre-fix
        check_contents( *v, keep, n, "after shrink_to_fit");

        v->shrink_to_fit();                 // idempotent
        check_contents( *v, keep, n, "after 2nd shrink_to_fit");
        v->~FV();
    }

    // (2) Talker::disconnect's exact shape: erase elements one at a time
    //     from the front, then shrink.  Also covers erase() returning a
    //     one-past-the-end iterator without dereferencing it.
    for(int n = 0; n <= 12; ++n) {
        alignas(FV) unsigned char buf[sizeof(FV)];
        FV *v = make_poisoned(buf, 0x5A);
        for(int i = 0; i < n; ++i) v->push_back(W(keep[i]));
        int expected = n;
        while( !v->empty()) {
            auto it = v->erase(v->begin());
            --expected;
            CHECK(v->size() == (size_t)expected);
            CHECK(it == v->begin());        // erased at front
            v->shrink_to_fit();
        }
        CHECK(v->size() == 0);
        v->~FV();
    }

    // (3) erase() of the LAST element specifically -- the case that made
    //     the heap branch return &*m_vector.end().
    for(int n = 5; n <= 10; ++n) {          // n > max_fixed_size => heap mode
        FV v;
        for(int i = 0; i < n; ++i) v.push_back(W(keep[i]));
        auto it = v.erase(v.end() - 1);
        CHECK(it == v.end());
        CHECK(v.size() == (size_t)(n - 1));
        v.shrink_to_fit();
        CHECK(v.size() == (size_t)(n - 1));
    }

    // (4) resize() -- now calls shrink_to_fit() internally in heap mode.
    for(int n : {6, 12, 40}) {
        FV v;
        for(int i = 0; i < n; ++i) v.push_back(W(keep[i]));
        v.resize(n / 2);
        CHECK(v.size() == (size_t)(n / 2));
        for(int i = 0; i < n / 2; ++i) { auto sp = v[i].lock(); CHECK(sp && *sp == i); }
        v.resize(n);                        // grow back
        CHECK(v.size() == (size_t)n);
    }

    // (5) copy / move construction and assignment across the boundary.
    for(int n : {0, 1, 3, 4, 5, 8, 40}) {
        FV a;
        for(int i = 0; i < n; ++i) a.push_back(W(keep[i]));

        FV b(a);
        a.shrink_to_fit(); b.shrink_to_fit();
        check_contents(b, keep, n, "copy ctor");

        FV c(std::move(b));
        c.shrink_to_fit();
        check_contents(c, keep, n, "move ctor");

        FV d; d.push_back(W(keep[63]));     // non-empty target
        d = c;
        check_contents(d, keep, n, "copy assign");

        FV e; e.push_back(W(keep[62]));
        e = std::move(d);
        check_contents(e, keep, n, "move assign");
    }

    // (6) Growth across the inline->heap transition keeps every element.
    {
        FV v;
        for(int i = 0; i < 40; ++i) {
            v.push_back(W(keep[i]));
            CHECK(v.size() == (size_t)(i + 1));
            v.shrink_to_fit();              // shrink at every step
            auto sp = v[i].lock();
            CHECK(sp && *sp == i);
            CHECK(v.front().lock() && *v.front().lock() == 0);
            CHECK(v.back().lock() && *v.back().lock() == i);
        }
        check_contents(v, keep, 40, "grow-with-shrink");
    }

    if(g_failures) {
        std::printf("fast_vector_test: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("fast_vector_test: all checks passed\n");
    return 0;
}
