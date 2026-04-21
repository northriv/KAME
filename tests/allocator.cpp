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
 *
 * activateAllocator() must be called for the pool to actually be used —
 * without it, new_redirected() in allocator_prv.h falls through to
 * malloc(). In kame/main.cpp that's done explicitly from main(); the
 * standalone test binaries each have their own main() and don't know
 * about the pool, so we flip the switch here from a static-init object.
 * It runs before main() — during dyld image load — but that's fine:
 * allocations before this constructor runs simply take the malloc branch,
 * which is the exact behaviour main.cpp relies on pre-activateAllocator.
 */
#ifndef DISABLE_POOL_ALLOCATOR
    #include "../kame/allocator.cpp"

    namespace {
        struct KamePoolActivator {
            KamePoolActivator() noexcept { activateAllocator(); }
        };
        const KamePoolActivator g_kame_pool_activator;
    }
#endif
