# kamepoolalloc/tests/CMakeLists.txt
#
# Self-contained build for kamepoolalloc's tests.  When this directory
# is the top-level of a CMake build, the file builds the kamepoolalloc
# SHARED library and every test in this folder.  When included via
# `add_subdirectory(...)` from a parent project (KAME's
# tests/CMakeLists.txt), the parent's USE_KAME_ALLOCATOR /
# kamepoolalloc / Threads decisions propagate and we only build the
# tests themselves.

cmake_minimum_required(VERSION 3.14)
if(NOT PROJECT_NAME)
    project(kamepoolalloc_tests LANGUAGES CXX C)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Reuse parent USE_KAME_ALLOCATOR if defined; otherwise compute the
# default (same logic as the historic top-level tests/CMakeLists.txt).
if(NOT DEFINED USE_KAME_ALLOCATOR)
    if(WIN32)
        set(USE_KAME_ALLOCATOR OFF)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
        set(USE_KAME_ALLOCATOR OFF)
    else()
        set(USE_KAME_ALLOCATOR ON)
    endif()
    message(STATUS "kamepoolalloc/tests: USE_KAME_ALLOCATOR = ${USE_KAME_ALLOCATOR}")
endif()

# Build the `kamepoolalloc` SHARED library target if it doesn't already
# exist (e.g. when this directory is built standalone).  The library
# lives one level up (`../allocator.cpp`).
if(USE_KAME_ALLOCATOR AND NOT TARGET kamepoolalloc)
    add_library(kamepoolalloc SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/../allocator.cpp)
    target_include_directories(kamepoolalloc PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_compile_definitions(kamepoolalloc PUBLIC KAMEPOOLALLOC_DYLIB)
    # Hot-path codegen: hide the (inline, COMDAT/weak) internal functions so
    # intra-DSO self-calls bind DIRECTLY instead of through the PLT.  Without
    # `-fvisibility-inlines-hidden`, `kame_free`'s call to the inline
    # `PoolAllocatorBase::deallocate` goes `call …@plt` on every free; with it
    # the symbol drops to local and the call is direct.  `-fno-semantic-
    # interposition` does the same for the non-inline self-calls
    # (`new_redirected_large`, `radix_lookup_slow`).  Neither changes which
    # symbols are exported — `malloc`/`free`/… stay global and interposable,
    # so the LD_PRELOAD drop-in semantics are untouched.
    target_compile_options(kamepoolalloc PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fvisibility-inlines-hidden;-fno-semantic-interposition>
        # MSVC on Japanese Windows (CP932 system code page) misreads UTF-8
        # box-drawing / Japanese characters in comments, potentially swallowing
        # newlines and hiding #define directives (symptom: C2065 on macros
        # defined just before a box-drawing art block).  /utf-8 fixes this.
        $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
    set_target_properties(kamepoolalloc PROPERTIES
        INSTALL_NAME_DIR "@rpath"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()
if(NOT USE_KAME_ALLOCATOR)
    add_compile_definitions(DISABLE_POOL_ALLOCATOR)
endif()

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Headers: only the library itself (`../`).  kamepoolalloc's tests are
# self-contained — they no longer pull kamestm's Qt-free test harness
# (`support_standalone.*` / `threadlocal.cpp` / `atomic_smart_ptr.h`); the
# includes were vestigial (no symbol from them was actually used), so the
# standalone repo builds with nothing but this directory.
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(test_link_libs Threads::Threads)
if(USE_KAME_ALLOCATOR)
    list(APPEND test_link_libs kamepoolalloc)
endif()

set(kamepoolalloc_test_names
    alloc_stress_test
    alloc_minimal_bench
    alloc_bucket34_repro)

foreach(t IN LISTS kamepoolalloc_test_names)
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} ${test_link_libs})
endforeach()

# Pure-C smoke test of the kame_pool.h public ABI.  No C++ support
# files — exercises only the C-linkage public surface.
if(USE_KAME_ALLOCATOR)
    add_executable(c_api_test c_api_test.c)
    target_link_libraries(c_api_test kamepoolalloc Threads::Threads)
    set_property(TARGET c_api_test PROPERTY C_STANDARD 99)

    # Pure-C intercept smoke test: validate that stdlib malloc/calloc/realloc/free
    # are routed through the pool (§31 IAT redirect on Windows; __interpose on
    # macOS; strong-symbol override on Linux).  Probe: kame_pool_malloc_usable_size
    # returns 0 for foreign pointers and >= requested_size for pool pointers.
    add_executable(malloc_intercept_test malloc_intercept_test.c)
    target_link_libraries(malloc_intercept_test kamepoolalloc Threads::Threads)
    set_property(TARGET malloc_intercept_test PROPERTY C_STANDARD 99)

    # (§27) Huge-alloc (> 32 MiB) round-trip / no-over-satisfy / MT test.
    # Uses the kame_pool_* C API directly, so it is meaningful only with the
    # pool linked in (DISABLE_POOL_ALLOCATOR has no kame_pool_malloc symbol).
    add_executable(alloc_huge_test alloc_huge_test.cpp)
    target_link_libraries(alloc_huge_test kamepoolalloc Threads::Threads)

    # (§28) Cap-eviction (kame_pool_set_large_cache_cap synchronous shrink) test.
    # Verifies the two-phase K-then-N reduction, MT race-safety, and round-trip
    # behaviour.  Pool-only (uses the kame_pool_* C API directly).
    add_executable(alloc_evict_test alloc_evict_test.cpp)
    target_link_libraries(alloc_evict_test kamepoolalloc Threads::Threads)

    # macOS 16 KiB-page madvise-straddle regression (fix f0d6a487).  Forces the
    # otherwise-lazy MADV_FREE reclaim with memory pressure, then checks that a
    # released chunk's reclaim never zeroes a live neighbour chunk's header.
    # Meaningful only where page size > ALLOC_CHUNK_K_MAX (Apple arm64 = 16 KiB);
    # on Linux (4 KiB == K_MAX) it self-skips and trivially passes.
    add_executable(alloc_madvise_straddle_repro alloc_madvise_straddle_repro.cpp)
    target_link_libraries(alloc_madvise_straddle_repro kamepoolalloc Threads::Threads)

    # (§28.2) kame_pool_get_stats v2 tier-attribution fields (cache_bytes,
    # dedicated_chunk_bytes, large_alloc_count/bytes) instrumentation test.
    add_executable(alloc_stats_test alloc_stats_test.cpp)
    target_link_libraries(alloc_stats_test kamepoolalloc Threads::Threads)

    # (§30) kame_pool_set_realtime_mode() preset toggle — verifies the
    # bundled lazy-drain / auto-tune / thread-exit-reclaim knobs flip
    # together and restore on enable=0.
    add_executable(alloc_realtime_mode_test alloc_realtime_mode_test.cpp)
    target_link_libraries(alloc_realtime_mode_test kamepoolalloc Threads::Threads)

    # (contrib) C++17 Allocator concept conformance for
    # `kame::pool_allocator<T>` — exercises rebind, propagate_on_*,
    # equality, allocate_shared, and STL containers (vector / list /
    # map / string).  Same Allocator surface rclcpp uses; passing here
    # is a structural pre-flight before any ROS 2 integration.
    add_executable(test_ros2_allocator ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/test_ros2_allocator.cpp)
    target_link_libraries(test_ros2_allocator kamepoolalloc Threads::Threads)

    # (contrib) `std::pmr::memory_resource` adaptor — the C++17 standard
    # interface; one adaptor covers every PMR-aware STL container plus
    # any third-party library (Boost.Container, Folly, abseil, …) that
    # consumes `std::pmr::memory_resource *`.  Auto-skips on toolchains
    # without `<memory_resource>` via `__has_include`.
    add_executable(test_pmr_resource ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/test_pmr_resource.cpp)
    target_link_libraries(test_pmr_resource kamepoolalloc Threads::Threads)

    # (contrib) Over-aligned C++17 Allocator (`kame::pool_aligned_allocator
    # <T, Align>`) for Eigen / SIMD / cacheline-aligned buffer use.
    # Drop-in for `Eigen::aligned_allocator<T>` on POSIX; auto-skips the
    # over-aligned section on Windows (current `kame_pool_aligned_alloc`
    # limitation) and still exercises the Align=16 default path.
    add_executable(test_aligned_allocator ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/test_aligned_allocator.cpp)
    target_link_libraries(test_aligned_allocator kamepoolalloc Threads::Threads)

    # Tuning-diagnostic — run on the target machine to characterise VMM
    # cost / TLB shootdown / kamepoolalloc throughput, and print a
    # recommendation block.  NOT registered as a ctest (long-running,
    # machine-specific output meant for human reading) — invoke directly:
    #     ./alloc_tune_report [seconds-per-bench, default 2]
    add_executable(alloc_tune_report alloc_tune_report.cpp)
    target_link_libraries(alloc_tune_report kamepoolalloc Threads::Threads)

    # Third-party-style microbenches (mimalloc-bench / glibc-bench shapes).
    # NOT registered as ctests — perf-only.  See bench/README.md for usage.
    #
    # `bench_loop`         — single-thread hot malloc/free.
    # `bench_xthread`      — cross-thread producer/consumer (xmalloc-test
    #                        shape) routed via `malloc`/`free`.  Picks up
    #                        whatever allocator the linker resolves first
    #                        (LD_PRELOAD overrides apply).
    # `bench_xthread_pool` — same workload but calls `kame_pool_malloc`
    #                        directly.  Use the two side-by-side to
    #                        attribute deltas to the strong-symbol override
    #                        layer vs the pool core.
    add_executable(bench_loop bench/bench_loop.c)
    set_property(TARGET bench_loop PROPERTY C_STANDARD 99)
    target_link_libraries(bench_loop Threads::Threads)

    add_executable(bench_xthread bench/bench_xthread.cpp)
    target_link_libraries(bench_xthread Threads::Threads)

    add_executable(bench_xthread_pool bench/bench_xthread.cpp)
    target_compile_definitions(bench_xthread_pool PRIVATE XTHREAD_USE_KAME_POOL)
    target_link_libraries(bench_xthread_pool kamepoolalloc Threads::Threads)
endif()

# --- CTest registration ---
enable_testing()
if(USE_KAME_ALLOCATOR)
    add_test(NAME c_api_test COMMAND c_api_test)
    add_test(NAME malloc_intercept_test COMMAND malloc_intercept_test)
    add_test(NAME alloc_huge_test COMMAND alloc_huge_test)
    add_test(NAME alloc_evict_test COMMAND alloc_evict_test)
    add_test(NAME alloc_stats_test COMMAND alloc_stats_test)
    add_test(NAME alloc_realtime_mode_test COMMAND alloc_realtime_mode_test)
    add_test(NAME test_ros2_allocator COMMAND test_ros2_allocator)
    add_test(NAME test_pmr_resource COMMAND test_pmr_resource)
    add_test(NAME test_aligned_allocator COMMAND test_aligned_allocator)
    if(APPLE)
        # macOS-only (16 KiB pages): forces MADV_FREE reclaim via memory
        # pressure to make the straddle deterministic.  Args: objects, pressure_MiB.
        add_test(NAME alloc_madvise_straddle_repro
                 COMMAND alloc_madvise_straddle_repro 800000 4096)
    endif()
endif()
