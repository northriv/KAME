/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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

void my_assert(char const*s, int d) {
        fprintf(stderr, "Err:%s:%d\n", s, d);
        abort();
}

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__
X86CPUSpec::X86CPUSpec() {
	uint32_t stepinfo, features_ext, features;
	asm volatile("push %%ebx; cpuid; pop %%ebx"
	: "=a" (stepinfo), "=c" (features_ext), "=d" (features) : "a" (0x1));
	verSSE = (features & (1uL << 25)) ? 1 : 0;
	if(verSSE && (features & (1uL << 26)))
		verSSE = 2;
	if((verSSE == 2) && (features_ext & (1uL << 0)))
		verSSE = 3;
#ifdef MACOSX
	hasMonitor = false;
#else 
	hasMonitor = (verSSE == 3) && (features_ext & (1uL << 3));
#endif
	monitorSizeSmallest = 0L;
	monitorSizeLargest = 0L;
	if(hasMonitor) {
		uint32_t monsize_s, monsize_l;
		asm volatile("push %%ebx; cpuid; mov %%ebx, %%ecx; pop %%ebx"
		: "=a" (monsize_s), "=c" (monsize_l) : "a" (0x5) : "%edx");
		monitorSizeSmallest = monsize_s;
		monitorSizeLargest = monsize_l;
	}
	fprintf(stderr, "SSE%u, monitor=%u, mon_smallest=%u, mon_larget=%u\n"
		, verSSE, (unsigned int)hasMonitor, monitorSizeSmallest, monitorSizeLargest);
}
const X86CPUSpec cg_cpuSpec;
#endif

	
	
	
