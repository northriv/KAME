# kamestm/CMakeLists.txt — canonical standalone build of the kamestm
# software-transactional-memory library (dual-licensed Apache-2.0 OR
# GPL-2.0-or-later).  Lets an external project consume it via
#
#   add_subdirectory(kamestm)              # in-tree (kamepoolalloc sibling must exist)
#   find_package(kamestm CONFIG)            # after `cmake --install`
#   target_link_libraries(app PRIVATE kamestm::kamestm)
#
# DEPENDENCY on kamepoolalloc is HEADER-ONLY: the lock-free atomic primitives
# (atomic.h / atomic_smart_ptr.h / atomic_mfence.h) were relocated to
# kamepoolalloc/ as their single home, and the STM #includes them.  kamestm
# calls ZERO libkamepoolalloc *runtime* symbols (see kamestm.pro), so we do NOT
# link libkamepoolalloc — we only need its include directory.  The package
# config find_dependency()'s kamepoolalloc so those headers are present; the
# install layout assumes both are installed under the same prefix.
#
# Production KAME builds inline/SUBDIRS-compile these sources (qmake kamestm.pro
# / kame/CMakeLists.txt); this file is for the standalone library + its optional
# test scaffold.  The test scaffold's own lib target (kamestm/tests/CMakeLists.txt)
# is unchanged and keeps working independently.

cmake_minimum_required(VERSION 3.16)
project(kamestm VERSION 8.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

include(GNUInstallDirs)   # must precede $<INSTALL_INTERFACE:${CMAKE_INSTALL_*}>
include(CMakePackageConfigHelpers)

add_library(kamestm SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/threadlocal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/xthread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/xtime.cpp)
add_library(kamestm::kamestm ALIAS kamestm)

target_compile_features(kamestm PUBLIC cxx_std_17)
# PUBLIC includes: kamestm's own headers (transaction.h …) AND kamepoolalloc's
# atomic*.h (header-only dep).  No libkamepoolalloc link — see header note.
target_include_directories(kamestm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../kamepoolalloc>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kamestm>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kamepoolalloc>)
target_link_libraries(kamestm PUBLIC Threads::Threads)
set_target_properties(kamestm PROPERTIES
    INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH TRUE
    SOVERSION ${PROJECT_VERSION_MAJOR})

# ---- install + export: enables find_package(kamestm CONFIG) ----
install(TARGETS kamestm EXPORT kamestmTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/atomic_queue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fast_vector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/support.h
    ${CMAKE_CURRENT_SOURCE_DIR}/threadlocal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transaction.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transaction_detail.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transaction_definitions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transaction_impl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transaction_negotiation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transaction_neg_impl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transaction_signal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/xthread.h
    ${CMAKE_CURRENT_SOURCE_DIR}/xtime.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kamestm)
install(EXPORT kamestmTargets
    NAMESPACE kamestm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kamestm)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kamestm-config-version.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
# Header-only dependency on kamepoolalloc (for atomic*.h) — pull its package in
# so its headers are on the consumer's path; no runtime link is added.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kamestm-config.cmake"
"include(CMakeFindDependencyMacro)
find_dependency(Threads)
find_dependency(kamepoolalloc)
include(\"\${CMAKE_CURRENT_LIST_DIR}/kamestmTargets.cmake\")
")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kamestm-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kamestm-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kamestm)

# ---- optional test scaffold (top-level only; portable check) ----
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_kstm_default_tests ON)
else()
    set(_kstm_default_tests OFF)
endif()
option(KAMESTM_BUILD_TESTS "Build the kamestm test scaffold" ${_kstm_default_tests})
if(KAMESTM_BUILD_TESTS)
    enable_testing()
    set(USE_KAME_ALLOCATOR ON)
    add_subdirectory(tests)
endif()
