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

# Headers:
#   - kamepoolalloc/ (the library itself, ../)
#   - kamestm/ (atomic.h, transaction.h transitively pulled by some tests)
#   - kamestm/tests/ (support_standalone.h — sibling library's tests
#                     own the Qt-free test harness used here too)
#   - kame/ (real support.h fallback when the standalone shim is not
#            preincluded — only used on AOCC paths that lack -include)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kamestm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kamestm/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kame)

# Pre-include `support_standalone.h` for C++ TUs; pure-C TUs (c_api_test)
# skip it (no C++ headers in support_standalone.h's chain).  AOCC 4.0
# crashes on -include, fall back to per-file #include in that case.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-include ${CMAKE_CURRENT_SOURCE_DIR}/../../kamestm/tests/support_standalone.h")
check_cxx_source_compiles("int main(){return 0;}" KAMEPOOLALLOC_TESTS_HAVE_FORCE_INCLUDE)
unset(CMAKE_REQUIRED_FLAGS)
if(KAMEPOOLALLOC_TESTS_HAVE_FORCE_INCLUDE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-include${CMAKE_CURRENT_SOURCE_DIR}/../../kamestm/tests/support_standalone.h>)
endif()

# support_SRCS: shared test harness sources.  Pulled from kamestm/tests/
# (the canonical home of `support_standalone.*`) and kamestm/
# (threadlocal.cpp — the type-erased TLS dispatcher every `XThreadLocal`
# call lands on; in production it lives in kame.dll, the standalone
# test binary needs its own copy).
set(support_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kamestm/tests/support_standalone.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kamestm/threadlocal.cpp)

set(test_link_libs Threads::Threads)
if(USE_KAME_ALLOCATOR)
    list(APPEND test_link_libs kamepoolalloc)
endif()

set(kamepoolalloc_test_names
    alloc_stress_test
    alloc_minimal_bench
    alloc_bucket34_repro)

foreach(t IN LISTS kamepoolalloc_test_names)
    add_executable(${t} ${t}.cpp ${support_SRCS})
    target_link_libraries(${t} ${test_link_libs})
endforeach()

# Pure-C smoke test of the kame_pool.h public ABI.  No C++ support
# files — exercises only the C-linkage public surface.
if(USE_KAME_ALLOCATOR)
    add_executable(c_api_test c_api_test.c)
    target_link_libraries(c_api_test kamepoolalloc Threads::Threads)
    set_property(TARGET c_api_test PROPERTY C_STANDARD 99)
endif()

# --- CTest registration ---
enable_testing()
if(USE_KAME_ALLOCATOR)
    add_test(NAME c_api_test COMMAND c_api_test)
endif()
