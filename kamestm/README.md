# kamestm

[![License: Apache-2.0 OR GPL-2.0+](https://img.shields.io/badge/License-Apache--2.0_OR_GPL--2.0%2B-blue.svg)](#license)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)]()

Lock-free software transactional memory (STM) primitives — the
snapshot/transaction core from the [KAME](https://github.com/northriv/KAME)
measurement framework, extracted as a stand-alone, **header-only**
library plus one small `.cpp`.

Dual-licensed under your choice of **Apache 2.0 OR GPL-2.0-or-later**
so it embeds cleanly into permissive / proprietary projects (Apache
path) or links into GPLv2-only projects such as KAME itself (GPL path).

## What's in here

The library covers Layer 0 (atomic primitives) and Layer 1 (snapshot
+ transaction commit) of the KAME STM design:

| Header | Role |
|---|---|
| `atomic.h`, `atomic_prv_basic.h`, `atomic_prv_std.h` | Portable barriers + CAS over `std::atomic_thread_fence` |
| `atomic_smart_ptr.h` | Tagged-pointer lock-free `local_shared_ptr<T>` (the engine under every Snapshot) |
| `atomic_queue.h` | Lock-free MPMC queue |
| `mutex.h` | Lightweight wrappers around `std::mutex` / `std::shared_mutex` |
| `threadlocal.h` + `threadlocal.cpp` | `XThreadLocal<T, Tag>` with deterministic per-thread teardown |
| `xtime.h` + `xtime.cpp` | Monotonic time helpers used by Lamport-clock serial numbers |
| `transaction.h`, `transaction_definitions.h`, `transaction_impl.h`, `transaction_signal.h` | The STM core: `Snapshot<XN>`, `Transaction<XN>`, `Node<XN>`, `Talker<...>` |
| `transaction_negotiation.h`, `transaction_neg_impl.h` | Negotiated retries (`iterate_commit_negotiated`) |

Out of scope (lives in `kame/` proper): `XNode`, the higher-level
node hierarchy on top of `Transactional::Node`.  Pull-out of that
layer is tracked separately.

## Dependencies

- C++17 toolchain (gcc 9+, clang 10+, MSVC stays on `std::allocator` for now).
- [`kamepoolalloc`](../kamepoolalloc) — sibling library providing
  `Transactional::allocator<T>` and the lock-free pool used by every
  Snapshot allocation.  The STM core includes
  `kamepoolalloc/allocator.h` via the consumer's INCLUDEPATH; falling
  back to `std::allocator` requires defining `USE_STD_ALLOCATOR`
  before including `transaction.h`.

## Build / Use

This is intended to be consumed from a parent build (KAME itself, or
a downstream user's CMake/qmake project).  Add to your INCLUDEPATH:

```cmake
target_include_directories(your_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/path/to/kamestm
    ${CMAKE_CURRENT_SOURCE_DIR}/path/to/kamepoolalloc)
```

Compile `kamestm/threadlocal.cpp` + `kamestm/xtime.cpp` into your
target.

A stand-alone `kamestm.pro` / `CMakeLists.txt` producing a
`libkamestm.dylib` is on the roadmap.

## License

**Dual-licensed under your choice of EITHER:**

- **Apache License, Version 2.0** — see [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
  Best for embedding into permissive / proprietary projects.
- **GNU GPL, version 2 of the License, or (at your option) any later version**
  — see [LICENSE-GPL-2.0](LICENSE-GPL-2.0).
  Best for linking into GPLv2-only projects such as KAME itself.

Pick whichever license suits your downstream project; see [LICENSE](LICENSE)
for the full dual-grant statement.

Copyright (C) 2002-2026 Kentaro Kitagawa &lt;kitag@issp.u-tokyo.ac.jp&gt;,
The University of Tokyo, ISSP.

Both license grants explicitly require preservation of the copyright
notice and the choice-of-license clause when redistributing this
software, in source or binary form.
