/***************************************************************************
        Copyright (C) 2002-2014 Kentaro Kitagawa
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
#include <string.h>
#include <stdint.h>
#include "atomic.h"

#ifdef USE_QTHREAD
    #include <QThread>
    #include <QDateTime>
#endif
#ifdef _MSC_VER
    #include <windows.h>
    #include <time.h>
#else
    #include <sys/time.h>
    #include <errno.h>
#endif

void msecsleep(unsigned int ms) noexcept {
#ifdef USE_QTHREAD
    XTime t0(XTime::now());
    for(;;) {
        unsigned int elapsed_ms = XTime::now().diff_msec(t0);
        if(elapsed_ms >= ms)
            break;
        if(ms - elapsed_ms >= 30) {
            QThread::msleep(ms - elapsed_ms);
        }
        else {
            XTime t1(XTime::now());
            QThread::yieldCurrentThread();
            if(XTime::now().diff_msec(t1) > 1)
                QThread::msleep(std::min(10u, ms - elapsed_ms)); //needs time slicing.
            else {
                pause4spin();
                pause4spin();
                pause4spin();
                pause4spin();
            }
        }
    }
#else //USE_QTHREAD
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
#endif //USE_QTHREAD
}

timestamp_t timeStamp() noexcept {
#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
    memoryBarrier();
    uint64_t r;
    #ifdef _MSC_VER
        r =  __rdtsc();
    #else
        asm volatile("rdtsc" : "=A" (r));
    #endif
    memoryBarrier();
    return r;
#else
    XTime time(XTime::now());
    return (time.usec() + time.sec() * 1000000uL);
#endif
}

static atomic<timestamp_t> s_time_stamp_cnt_per_ms = 0;
static atomic<timestamp_t> s_time_stamp_calc;

DECLSPEC_KAME timestamp_t timeStampCountsPerMilliSec() noexcept {
    if( !s_time_stamp_cnt_per_ms) {
        for(;;) {
            timestamp_t time_stamp_start = timeStamp();
            s_time_stamp_calc = time_stamp_start;
            XTime time_start(XTime::now());
            timestamp_t time_stamp_start2 = timeStamp();
            msecsleep(20);
            timestamp_t dt = timeStamp() - time_stamp_start2;
            unsigned int msec = XTime::now().diff_msec(time_start);
            timestamp_t dt2 = timeStamp() - time_stamp_start;
            s_time_stamp_cnt_per_ms = (dt+dt2)/2 / msec;
            fprintf(stderr, "Clocks per ms = %d\n", (int)s_time_stamp_cnt_per_ms);
            if((double)dt2 / dt < 1.2)
                break;
        }
    }
    return s_time_stamp_cnt_per_ms;
}

XTime
XTime::now() noexcept {
#ifdef USE_QTHREAD
    qint64 x = QDateTime::currentMSecsSinceEpoch();
    return XTime(x / 1000LL, (x % 1000LL) * 1000l);
#else //USE_QTHREAD
    timeval tv;
    gettimeofday(&tv, NULL);
    return XTime(tv.tv_sec, tv.tv_usec);
#endif //USE_QTHREAD
};

XString
XTime::getTimeStr(bool subsecond) const {
    if(isSet()) {
        char str[100];
#if defined _MSC_VER || defined(__MINGW64__)
        __time32_t t32 = tv_sec;
        __time64_t t64 = tv_sec;
        if(sizeof(tv_sec) == 4)
            _ctime32_s(str, sizeof(str - 1), &t32);
        else
            _ctime64_s(str, sizeof(str - 1), &t64);
#else
        ctime_r( &tv_sec, str);
#endif
        str[strlen(str) - 1] = '\0';
        if(subsecond)
            sprintf(str + strlen(str), " +%.3dms", (int)tv_usec/1000);
        return {str};
    }
    else {
        return {};
    }
}
XString
XTime::getTimeFmtStr(const char *fmt, bool subsecond) const {
    if(isSet()) {
        struct tm time;
#if defined _MSC_VER || defined(__MINGW64__)
        __time32_t t32 = tv_sec;
        __time64_t t64 = tv_sec;
        if(sizeof(tv_sec) == 4)
            _localtime32_s( &time, &t32);
        else
            _localtime64_s( &time, &t64);
#else
        localtime_r( &tv_sec, &time);
#endif
        char str[100];
        strftime(str, 100, fmt, &time);
        if(subsecond)
            sprintf(str + strlen(str), " +%.3f", 1e-6 * tv_usec);
        return {str};
    }
    else {
        return {};
    }
}

