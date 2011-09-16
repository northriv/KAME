/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xtime.h"
#include <sys/time.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>

void msecsleep(unsigned int ms) {
	XTime t0(XTime::now());
	XTime t1 =  t0;
	t0 += ms * 1e-3;
	while(t1 < t0) {
		struct timespec req;
		req.tv_sec = (int)(t0 - t1);
		req.tv_nsec = lrint((t0 - t1 - req.tv_sec) * 1e9);
		if( !nanosleep(&req, NULL))
			break;
		t1 = XTime::now();
		switch(errno) {
		case EINTR:
			continue;
		default:
			abort();
		}
	}
}

unsigned int timeStamp() {
#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
    uint64_t r;
    asm volatile("rdtsc" : "=A" (r));
    return (unsigned long)(r / (uint64_t)256);
#elif defined __powerpc__ || defined __POWERPC__ || defined __ppc__
    uint32_t rx, ry, rz;
    asm volatile("1: \n"
				 "mftbu %[rx]\n"
				 "mftb %[ry]\n"
				 "mftbu %[rz]\n"
				 "cmpw %[rz], %[rx]\n"
				 "bne- 1b"
				 : [rx] "=&r" (rx), [ry] "=&r" (ry), [rz] "=&r" (rz)
				 :: "cc");
    uint64_t r = rx;
    r = (r << 32u) + ry;
    return (unsigned int)(r);
#else
    XTime time(XTime::now());
    return (unsigned long)(time.usec() + time.sec() * 1000000uL);
#endif
}


XTime
XTime::now() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return XTime(tv.tv_sec, tv.tv_usec);
};

XString
XTime::getTimeStr(bool subsecond) const {
    if( *this) {
		char str[100];
		ctime_r( &tv_sec, str);
		str[strlen(str) - 1] = '\0';
		if(subsecond)
			sprintf(str + strlen(str), " +%.3dms", (int)tv_usec/1000);
		return str;
    }
    else {
        return XString();
    }
}
XString
XTime::getTimeFmtStr(const char *fmt, bool subsecond) const {
    if( *this) {
		char str[100];
		struct tm time;
		localtime_r( &tv_sec, &time);
		strftime(str, 100, fmt, &time);
		if(subsecond)
			sprintf(str + strlen(str), " +%.3f", 1e-6 * tv_usec);
		return str;
    }
    else {
        return XString();
    }
}

