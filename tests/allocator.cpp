/*
 * tests/allocator.cpp — activator shim for the standalone test binaries.
 *
 * Previously this file `#include`d the entire allocator translation unit
 * inline.  Now the pool allocator is factored out into its own shared
 * library (`kamepoolalloc`, built from `../kamepoolalloc/allocator.cpp`),
 * so this TU only needs to flip the activation switch.
 *
 * The shared-library split lets dyld process the `__DATA,__interpose`
 * section that captures `free()` calls from libc++ / libsystem_pthread
 * during thread teardown — dyld only honours interposing from MH_DYLIB
 * images, never from MH_EXECUTE.
 *
 * `activateAllocator()` runs before `main()` via a static-init object
 * (just like the previous inline-include design did). Allocations before
 * the activator constructor runs simply take the malloc branch — the
 * same pre-activation behaviour kame/main.cpp relies on.
 */
#ifndef DISABLE_POOL_ALLOCATOR
    // Exported from libkamepoolalloc.dylib (allocator.cpp).  Declared
    // here without including the heavy allocator.h header, which would
    // pull in <array>, <vector>, ALLOC_TLS, PoolAllocator templates,
    // etc. just to compile a one-line static-init shim.
    extern void activateAllocator();

    namespace {
        struct KamePoolActivator {
            KamePoolActivator() noexcept { activateAllocator(); }
        };
        const KamePoolActivator g_kame_pool_activator;
    }
#endif
