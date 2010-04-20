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
#ifndef XTIME_H_
#define XTIME_H_

#include <support.h>
#include <math.h>

//! sleep in ms
void msecsleep(unsigned int ms);

//! fetch CPU counter.
unsigned int timeStamp();

class XTime
{
public:
    XTime() : tv_sec(0), tv_usec(0) {}
    XTime(long sec, long usec) : tv_sec(sec), tv_usec(usec) {}
    XTime(const XTime &x) :
        tv_sec(x.tv_sec), tv_usec(x.tv_usec) {}
    XTime &operator=(const XTime &x) {
        tv_sec = x.tv_sec; tv_usec = x.tv_usec;
        return *this;
    }
    double operator-(const XTime &x) const {
        return (tv_sec - x.tv_sec) + (tv_usec - x.tv_usec) * 1e-6;
    }
    long diff_usec(const XTime &x) const {
        return (tv_sec - x.tv_sec) * 1000000L + ((tv_usec - x.tv_usec));
    }
    long diff_msec(const XTime &x) const {
        return (tv_sec - x.tv_sec) * 1000L + ((tv_usec - x.tv_usec) / 1000L);
    }
    long diff_sec(const XTime &x) const {
        return tv_sec - x.tv_sec;
    }
    XTime &operator+=(double sec_d) {
        long sec = floor(sec_d + tv_sec + 1e-6 * tv_usec);
        long usec = (lrint(1e6 * (tv_sec - sec + sec_d) + tv_usec));
        tv_sec = sec;
        tv_usec = usec;
        ASSERT((tv_usec >= 0) && (tv_usec < 1000000));
        return *this;
    }
    XTime &operator-=(double sec) {
        *this += -sec;
        return *this;
    }
    bool operator==(const XTime &x) const {
        return (tv_sec == x.tv_sec) && (tv_usec == x.tv_usec);
    }
    bool operator!=(const XTime &x) const {
        return (tv_sec != x.tv_sec) || (tv_usec != x.tv_usec);
    }
    bool operator<(const XTime &x) const  {
        return (tv_sec < x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec < x.tv_usec));
    }
    bool operator<=(const XTime &x) const  {
        return (tv_sec <= x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec <= x.tv_usec));
    }
    bool operator>(const XTime &x) const  {
        return (tv_sec > x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec > x.tv_usec));
    }
    bool operator>=(const XTime &x) const  {
        return (tv_sec >= x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec >= x.tv_usec));
    }
    bool operator!() const {
        return (tv_sec == 0) && (tv_usec == 0);
    }
    operator bool() const {
        return (tv_sec != 0) || (tv_usec != 0);
    }
    long sec() const {return tv_sec;}
    long usec() const {return tv_usec;}
    static XTime now();
    XString getTimeStr(bool subsecond = true) const;
    XString getTimeFmtStr(const char *fmt, bool subsecond = true) const
		__attribute__ ((format(strftime,2, 0)));
private:
    long tv_sec;
    long tv_usec;
};

#endif /*XTIME_H_*/
