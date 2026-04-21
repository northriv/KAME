/*
 * tests/allocator.cpp — brings the KAME pool allocator into the standalone
 * test binaries (operator new/new[]/delete/delete[] overrides) as a real
 * translation unit, instead of being pulled in via an #include inside
 * support_standalone.cpp.
 *
 * Compiled whenever DISABLE_POOL_ALLOCATOR is not defined. The CMake build
 * gates this via -DUSE_KAME_ALLOCATOR=ON/OFF (tests/CMakeLists.txt); the
 * qmake build (tests.pri) compiles it unconditionally and relies on
 * allocator.h setting USE_STD_ALLOCATOR on non-x86 so this TU collapses
 * to nothing there.
 */
#ifndef DISABLE_POOL_ALLOCATOR
    #include "../kame/allocator.cpp"
#endif
