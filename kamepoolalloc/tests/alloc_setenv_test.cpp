/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            kamepoolalloc/LICENSE-APACHE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (see kamepoolalloc/LICENSE-GPL-2.0).

        Pick whichever license suits your project.
***************************************************************************/

// alloc_setenv_test — setenv() whose own malloc claims a fresh chunk.
//
// setenv holds libc's environment lock while it mallocs the new "NAME=value"
// string.  With the pool interposing malloc, that malloc comes to the pool,
// and on a thread with no chunk of that size class yet it claims one.  The
// chunk claim used to read its knobs (KAME_POOL_DISABLE_PREFILL,
// KAME_ALLOC_PREWARM, KAME_POOL_VERBOSE, ...) lazily by getenv -- which on
// macOS (Darwin 27) takes the same environment lock, an os_unfair_lock: the
// process aborted with "BUG IN CLIENT OF LIBPLATFORM: Trying to recursively
// lock an os_unfair_lock" (SIGKILL).  CPython's start-up setenv hit it on every
// run under DYLD_INSERT_LIBRARIES (2026-11-17).  The knobs are now
// read once at load (KamePoolEnv in allocator.cpp).
//
// Here: fresh threads, each setenv'ing values of many lengths (so that every
// size class claims its first chunk on that thread inside setenv), reading
// them back by getenv.  Before the fix this aborts on macOS; elsewhere getenv
// takes no lock and the test passes either way.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "kame_pool.h"

int main() {
	const int THREADS = 8;
	int failures = 0;
	std::vector<std::thread> threads;
	std::vector<int> bad(THREADS, 0);
	for(int t = 0; t < THREADS; ++t) {
		threads.emplace_back([t, &bad] {
			char name[64];
			std::snprintf(name, sizeof(name), "KAME_SETENV_TEST_%d", t);
			for(std::size_t len = 1; len < 40000; len = len * 5 / 4 + 1) {
				std::string value(len, static_cast<char>('a' + t));
				if(setenv(name, value.c_str(), 1) != 0) { ++bad[t]; continue; }
				const char *got = std::getenv(name);
				if( !got || std::strlen(got) != len) ++bad[t];
			}
			unsetenv(name);
		});
	}
	for(auto &th : threads) th.join();
	for(int t = 0; t < THREADS; ++t) failures += bad[t];
	// The pool took setenv's malloc (else the test proves nothing): a malloc
	// of the same kind on this thread is the pool's.
	void *p = std::malloc(100);
	bool pooled = kame_pool_malloc_usable_size(p) >= 100;
	std::free(p);
	std::printf("alloc_setenv_test: %d failures, malloc %s the pool's\n", failures, pooled ? "is" : "is NOT");
	return failures == 0 ? 0 : 1;
}
