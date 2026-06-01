# `contrib/` — experimental adaptors

Header-only adaptor shims that bridge `kamepoolalloc` to other ecosystems.
**Status: experimental** unless otherwise noted on the per-file header — the
core pool API is the supported surface, these shims are convenience layers
that ride on it and may evolve without notice.

## What's here

| File | Status | What it is |
|---|---|---|
| [`pmr_resource.hpp`](pmr_resource.hpp) | experimental | `std::pmr::memory_resource` — the C++17 standard "swap out the allocator behind any STL container at runtime" interface.  **One adaptor covers every `std::pmr::vector / unordered_map / string / list / deque / map`** plus any third-party library that consumes `std::pmr::memory_resource *` (Boost.Container, Folly, abseil).  Stateless singleton via `kame::pmr::pool_resource()`. |
| [`ros2_allocator.hpp`](ros2_allocator.hpp) | experimental | C++17 Allocator (`kame::pool_allocator<T>`) for `rclcpp::Publisher<Msg, Alloc>` / `Subscription<Msg, Alloc>` / `Executor` and for STL containers in ROS 2 real-time callbacks.  Stateless (`is_always_equal = true_type`); routes via `kame_pool_malloc` / `kame_pool_free`. |
| [`aligned_allocator.hpp`](aligned_allocator.hpp) | experimental | Over-aligned C++17 Allocator (`kame::pool_aligned_allocator<T, Align>`) for Eigen / SIMD / cacheline-aligned buffers.  Drop-in for `Eigen::aligned_allocator<T>` on POSIX (Windows over-aligned support pending — see header). |

## How they fit together

`pmr_resource.hpp` is the **broadest** entry point — if your code is C++17
and uses `std::pmr`-aware containers, just point them at
`kame::pmr::pool_resource()` and you're done.  The other adaptors are for:

- **`ros2_allocator.hpp`** — when you need a *concrete*
  `std::allocator<T>`-shaped allocator (not a `polymorphic_allocator`)
  because rclcpp's templates require it.
- **`aligned_allocator.hpp`** — when you need stronger alignment than the
  pool's default 16 B (Eigen's 32 B AVX2 alignment, AVX-512's 64 B, page
  alignment 4096 B).  PMR can also serve over-aligned via `do_allocate`'s
  second argument, but a concrete allocator gives you `static constexpr
  alignment` for templates that probe it.

## Concept-conformance tests (no third-party deps)

Each adaptor ships with a smoke test that exercises the same surface its
target ecosystem consumes — but **without** linking that ecosystem, so
kamepoolalloc has no `find_package(rclcpp)` / `find_package(Eigen3)` / etc.
build dependency:

| Test | Covers |
|---|---|
| `test_pmr_resource` | `pmr::vector` / `pmr::string` / `pmr::unordered_map` / `pmr::map` / `pmr::list` / `pmr::deque` / `polymorphic_allocator<T>` / `set_default_resource` / `is_equal` |
| `test_ros2_allocator` | `std::vector` / `std::list` / `std::map` / `std::basic_string` / `std::allocate_shared`, rebind / propagate_on_* / equality |
| `test_aligned_allocator` | Align ∈ {16, 32, 64}, runtime alignment check, `std::vector::data()` alignment, rebind carries `Align` |

All three end with a `kame_pool_get_stats()` sanity check (regions > 0) so
"silently routing to libc malloc" cannot regress.

## When NOT to use these on a hard-real-time path

The pool's cold-claim path can mmap a fresh 32 MiB region (one-shot per
working-set growth) which is NOT bounded for hard-RT.  Same idiom as
TLSF: **pre-warm before entering the time-critical loop**:

```cpp
kame_pool_set_realtime_mode(1);   // silence background maintenance
for (std::size_t sz : your_RT_size_classes)
    if (void *p = kame_pool_malloc(sz)) kame_pool_free(p);
// ... now enter your 1 kHz control loop ...
```

For soft-RT and AD perception (33 ms / frame) the lock-free TLS freelist
pop is fast enough without pre-warm.

## Recipes that don't ship as files

Some ecosystems need their headers to compile any adaptor; rather than
gate the contrib build on those, here are paste-ready recipes the user
copies into their downstream package.

### OpenCV — `cv::MatAllocator`

OpenCV's `cv::Mat` allocator interface lives in `<opencv2/core.hpp>` and
differs slightly between 3.x and 4.x.  In a downstream package that
already pulls OpenCV in, drop this next to your translation unit and call
`cv::Mat::setDefaultAllocator(kame::cv_pool_allocator())` at startup:

```cpp
#include <opencv2/core.hpp>
#include <kamepoolalloc/contrib/../kame_pool.h>

namespace kame {
class CvPoolAllocator : public cv::MatAllocator {
public:
    cv::UMatData *allocate(int dims, const int *sizes, int type,
                           void *data0, std::size_t *step,
                           cv::AccessFlag /*flags*/,
                           cv::UMatUsageFlags usageFlags) const override {
        std::size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; --i) {
            if (step) {
                if (data0 && step[i] != CV_AUTOSTEP) {
                    total = step[i];
                    continue;
                }
                step[i] = total;
            }
            total *= sizes[i];
        }
        auto *u = new cv::UMatData(this);
        u->size = total;
        if (data0) {
            u->data = u->origdata = (uchar *)data0;
            u->flags |= cv::UMatData::USER_ALLOCATED;
        } else {
            u->data = u->origdata = (uchar *)kame_pool_malloc(total);
        }
        u->urefcount = 0;
        u->refcount = 0;
        return u;
    }
    bool allocate(cv::UMatData *u, cv::AccessFlag /*flags*/,
                  cv::UMatUsageFlags /*usageFlags*/) const override {
        return u != nullptr;
    }
    void deallocate(cv::UMatData *u) const override {
        if (!u) return;
        if (!(u->flags & cv::UMatData::USER_ALLOCATED))
            kame_pool_free(u->origdata);
        delete u;
    }
};
inline cv::MatAllocator *cv_pool_allocator() {
    static CvPoolAllocator inst;
    return &inst;
}
} // namespace kame

// At process startup:
//     cv::Mat::setDefaultAllocator(kame::cv_pool_allocator());
```

Not shipped as a header here because the `cv::UMatData` / `cv::AccessFlag`
ABI varies across OpenCV minor versions; downstream maintainers can pin
the snippet to their `cv::` version and contribute back a proper header
if a common subset emerges.

## Contributing more adaptors

Same dual-license header, `#include "../kame_pool.h"` only (no internal
`allocator_prv.h` surface — that's not ABI-stable), one concept-conformance
test file registered in `kamepoolalloc/tests/CMakeLists.txt`.  Auto-skip
on platforms / toolchains where the ecosystem isn't available (via
`__has_include` or a runtime probe), as `test_pmr_resource.cpp` and
`test_aligned_allocator.cpp` already do.
