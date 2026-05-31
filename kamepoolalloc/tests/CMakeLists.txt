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
endif()

# --- CTest registration ---
enable_testing()
if(USE_KAME_ALLOCATOR)
    add_test(NAME c_api_test COMMAND c_api_test)
    add_test(NAME alloc_huge_test COMMAND alloc_huge_test)
    add_test(NAME alloc_evict_test COMMAND alloc_evict_test)
    if(APPLE)
        # macOS-only (16 KiB pages): forces MADV_FREE reclaim via memory
        # pressure to make the straddle deterministic.  Args: objects, pressure_MiB.
        add_test(NAME alloc_madvise_straddle_repro
                 COMMAND alloc_madvise_straddle_repro 800000 4096)
    endif()
endif()
