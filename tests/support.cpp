/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "support.h"
#include "atomic.h"
#include "allocator.cpp"

int my_assert(char const*s, int d) {
        fprintf(stderr, "Err:%s:%d\n", s, d);
        abort();
        return -1;
}

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
X86CPUSpec::X86CPUSpec() {
	uint32_t stepinfo, features_ext, features;
#if defined __LP64__ || defined __LLP64__
	asm volatile("push %%rbx; cpuid; pop %%rbx"
#else
	asm volatile("push %%ebx; cpuid; pop %%ebx"
#endif
	: "=a" (stepinfo), "=c" (features_ext), "=d" (features) : "a" (0x1));
	verSSE = (features & (1uL << 25)) ? 1 : 0;
	if(verSSE && (features & (1uL << 26)))
		verSSE = 2;
	if((verSSE == 2) && (features_ext & (1uL << 0)))
		verSSE = 3;
#ifdef __APPLE__
	hasMonitor = false;
#else
	hasMonitor = (verSSE == 3) && (features_ext & (1uL << 3));
#endif
	monitorSizeSmallest = 0L;
	monitorSizeLargest = 0L;
	if(hasMonitor) {
		uint32_t monsize_s, monsize_l;
#if defined __LP64__ || defined __LLP64__
		asm volatile("push %%rbx; cpuid; mov %%ebx, %%ecx; pop %%rbx"
#else
		asm volatile("push %%ebx; cpuid; mov %%ebx, %%ecx; pop %%ebx"
#endif
		: "=a" (monsize_s), "=c" (monsize_l) : "a" (0x5) : "%edx");
		monitorSizeSmallest = monsize_s;
		monitorSizeLargest = monsize_l;
	}
	fprintf(stderr,
#if defined __LP64__
		"x86-64, LP64 + "
#else
	#if defined __LLP64__
			"x86-64, LLP64 + "
	#else
			"x86-32 + "
	#endif
#endif
		"SSE%u, monitor=%u, mon_smallest=%u, mon_larget=%u\n"
		, verSSE, (unsigned int)hasMonitor, monitorSizeSmallest, monitorSizeLargest);
}
const X86CPUSpec cg_cpuSpec;
#endif




