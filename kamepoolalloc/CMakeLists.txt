# kamepoolalloc/CMakeLists.txt — canonical standalone build of the
# kamepoolalloc lock-free pool allocator (dual-licensed Apache-2.0 OR
# GPL-2.0-or-later).  Lets an external project consume the library via
#
#   add_subdirectory(kamepoolalloc)         # in-tree
#   find_package(kamepoolalloc CONFIG)       # after `cmake --install`
#   target_link_libraries(app PRIVATE kamepoolalloc::kamepoolalloc)
#
# Production KAME builds inline-compile allocator.cpp (qmake kame.pro /
# kame/CMakeLists.txt SOURCES) and do NOT link this shared library; this
# file is purely for the standalone library + its optional test scaffold.
#
# NOTE: the `add_library(kamepoolalloc ...)` definition below is intentionally
# mirrored by tests/CMakeLists.txt's `if(NOT TARGET kamepoolalloc)` fallback,
# which builds the same target when tests/ is configured as its OWN top level
# (`cd tests && cmake ..`, and the KAME repo-root tests/ coordinator).  Keep
# the two compile-definition / option / property sets in sync.

cmake_minimum_required(VERSION 3.16)
project(kamepoolalloc VERSION 8.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Must precede the target's $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/...>
# use below — otherwise CMAKE_INSTALL_INCLUDEDIR is empty and the installed
# INTERFACE_INCLUDE_DIRECTORIES degrades to an absolute "/kamepoolalloc".
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(kamepoolalloc SHARED ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp)
add_library(kamepoolalloc::kamepoolalloc ALIAS kamepoolalloc)

# Public C API header is kame_pool.h.  In-tree consumers (incl. tests/) include
# it as "kame_pool.h"; the install layout (include/kamepoolalloc on the path)
# keeps that spelling working for installed consumers too.
target_include_directories(kamepoolalloc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kamepoolalloc>)
target_compile_definitions(kamepoolalloc PUBLIC KAMEPOOLALLOC_DYLIB)

# Hot-path codegen (see tests/CMakeLists.txt for the rationale): bind intra-DSO
# self-calls directly instead of via the PLT; /utf-8 fixes MSVC CP932 comment
# mis-parsing.  PRIVATE — does not change which symbols are exported, so the
# malloc/free drop-in interposition semantics are preserved.
target_compile_options(kamepoolalloc PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fvisibility-inlines-hidden;-fno-semantic-interposition>
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_link_libraries(kamepoolalloc PUBLIC Threads::Threads)
set_target_properties(kamepoolalloc PROPERTIES
    INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH TRUE
    SOVERSION ${PROJECT_VERSION_MAJOR})

# ---- install + export: enables find_package(kamepoolalloc CONFIG) ----
install(TARGETS kamepoolalloc EXPORT kamepoolallocTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# Public headers: the C API (kame_pool.h) PLUS the lock-free atomic primitives
# (atomic*.h) — these are the single shared home of atomic_shared_ptr et al.,
# included by the STM (kamestm) and by any external lock-free consumer, so they
# ship as part of the public surface.
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/kame_pool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/atomic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/atomic_mfence.h
    ${CMAKE_CURRENT_SOURCE_DIR}/atomic_smart_ptr.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kamepoolalloc)
install(EXPORT kamepoolallocTargets
    NAMESPACE kamepoolalloc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kamepoolalloc)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kamepoolalloc-config-version.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kamepoolalloc-config.cmake"
"include(CMakeFindDependencyMacro)
find_dependency(Threads)
include(\"\${CMAKE_CURRENT_LIST_DIR}/kamepoolallocTargets.cmake\")
")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kamepoolalloc-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kamepoolalloc-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kamepoolalloc)

# ---- optional test scaffold ----
# Build the tests only when this is the top-level project (portable check —
# avoids requiring cmake 3.21 PROJECT_IS_TOP_LEVEL).  A consumer that
# add_subdirectory()'s us gets the library alone.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_kpa_default_tests ON)
else()
    set(_kpa_default_tests OFF)
endif()
option(KAMEPOOLALLOC_BUILD_TESTS "Build the kamepoolalloc test scaffold" ${_kpa_default_tests})
if(KAMEPOOLALLOC_BUILD_TESTS)
    enable_testing()
    set(USE_KAME_ALLOCATOR ON)          # tests/ reuses the target above via its NOT-TARGET guard
    add_subdirectory(tests)
endif()
