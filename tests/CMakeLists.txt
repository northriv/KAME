# tests/CMakeLists.txt
#
# Thin coordinator: each sibling library (kamepoolalloc, kamestm) owns
# its own tests/ subdirectory with a self-contained CMakeLists.txt.
# This file just delegates via add_subdirectory() and chooses a single
# shared USE_KAME_ALLOCATOR up-front so the kamepoolalloc SHARED
# library target is defined once and reused by both sets of tests.
#
# (Kame-specific tests — xnode_typename_test, nidaqmx_*_test,
# bench_kame_full — live in this directory and use the Qt-aware kame/
# tree; they're not registered in CMake yet, only via per-test .pro
# files.  Add inline targets here as they get a CMake build path.)

cmake_minimum_required(VERSION 3.14)
project(kame_tests LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Shared option resolved once here so kamepoolalloc/tests/ and
# kamestm/tests/ see the same value.
if(NOT DEFINED USE_KAME_ALLOCATOR)
    if(WIN32)
        set(USE_KAME_ALLOCATOR OFF)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
        set(USE_KAME_ALLOCATOR OFF)
    else()
        set(USE_KAME_ALLOCATOR ON)
    endif()
endif()
message(STATUS "USE_KAME_ALLOCATOR = ${USE_KAME_ALLOCATOR}")

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Subdir order matters: kamepoolalloc's tests/ creates the
# `kamepoolalloc` SHARED-lib target; kamestm's tests/ reuses it via
# the `if(NOT TARGET kamepoolalloc)` guard.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kamepoolalloc/tests
                 ${CMAKE_CURRENT_BINARY_DIR}/kamepoolalloc-tests)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kamestm/tests
                 ${CMAKE_CURRENT_BINARY_DIR}/kamestm-tests)

enable_testing()
