# `contrib/` — experimental adaptors

Header-only adaptor shims that bridge `kamepoolalloc` to other ecosystems.
**Status: experimental** unless otherwise noted on the per-file header — the
core pool API is the supported surface, these shims are convenience layers
that ride on it and may evolve without notice.

| File | Status | What it is |
|---|---|---|
| [`ros2_allocator.hpp`](ros2_allocator.hpp) | experimental | C++17 Allocator adaptor (`kame::pool_allocator<T>`) suitable for `rclcpp::Publisher<Msg, Alloc>` / `Subscription<Msg, Alloc>` / `Executor` and for STL containers in ROS 2 real-time callbacks.  Stateless (`is_always_equal = true_type`); routes via `kame_pool_malloc` / `kame_pool_free`. |

## Concept conformance check

[`test_ros2_allocator.cpp`](test_ros2_allocator.cpp) is registered as the
`test_ros2_allocator` ctest target — it does NOT link rclcpp (would force a
ROS 2 build dependency on kamepoolalloc).  Instead it exercises the SAME
C++17 Allocator surface rclcpp internally consumes:

  * `std::vector<T, Alloc>` push/copy/move
  * `std::list<T, Alloc>` (node-based; one-at-a-time allocate)
  * `std::map<K,V, …, Alloc>` (forces `rebind_alloc<__node>`)
  * `std::basic_string<char, …, Alloc<char>>`
  * `std::allocate_shared<T>(alloc, ...)` (matches publisher message ctor)
  * Stateless-allocator equality + propagate_on_* traits

If this test passes, the rclcpp integration is structurally sound.  The
remaining unknown is the ROS-side wiring (executor / memory strategy /
QoS) — which is downstream-package territory, not allocator territory.

## When NOT to use this on a hard-real-time path

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

## Contributing more adaptors

Same dual-license header, `#include "../kame_pool.h"` only (no internal
allocator_prv.h surface — that's not ABI-stable), one concept-conformance
test file registered in `kamepoolalloc/tests/CMakeLists.txt`.  Land
inside `#if`-gates if the host ecosystem isn't ubiquitous.
