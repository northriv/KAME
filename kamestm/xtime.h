/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/
#ifndef XTIME_H_
#define XTIME_H_

#include <support.h>  // angle: INCLUDEPATH-priority (kame/ full vs kamestm/ standalone) — see transaction_detail.h
#include <math.h>

#include <chrono>
//#include <thread>
using namespace std::chrono;
//using namespace std::this_thread;

//! Sleeps in ms
DECLSPEC_KAME void msecsleep(unsigned int ms) noexcept; //<!\todo {std::this_thread::sleep_for(std::chrono::milliseconds(ms));}

//! Fetches CPU counter.
using timestamp_t = uint64_t;
DECLSPEC_KAME timestamp_t timeStamp() noexcept;
DECLSPEC_KAME timestamp_t timeStampCountsPerMilliSec() noexcept;

class DECLSPEC_KAME XTime {
public:
    XTime() noexcept : tv_sec(0), tv_usec(0) {}
    XTime(long sec, long usec) noexcept : tv_sec(sec), tv_usec(usec) {}
    XTime(const system_clock::time_point &);
    operator system_clock::time_point() const;

    double operator-(const XTime &x) const noexcept {
        return (tv_sec - x.tv_sec) + (tv_usec - x.tv_usec) * 1e-6;
    }
    long diff_usec(const XTime &x) const noexcept {
        return (tv_sec - x.tv_sec) * 1000000L + ((tv_usec - x.tv_usec));
    }
    long diff_msec(const XTime &x) const noexcept {
        return (tv_sec - x.tv_sec) * 1000L + ((tv_usec - x.tv_usec) / 1000L);
    }
    long diff_sec(const XTime &x) const noexcept {
        return tv_sec - x.tv_sec;
    }
    XTime &operator+=(double sec_d) noexcept {
        long sec = floor(sec_d + tv_sec + 1e-6 * tv_usec);
        long usec = (lrint(1e6 * (tv_sec - sec + sec_d) + tv_usec));
        tv_sec = sec;
        tv_usec = usec;
        assert((tv_usec >= 0) && (tv_usec < 1000000));
        return *this;
    }
    XTime &operator-=(double sec) noexcept {
        *this += -sec;
        return *this;
    }
    bool operator==(const XTime &x) const noexcept {
        return (tv_sec == x.tv_sec) && (tv_usec == x.tv_usec);
    }
    bool operator!=(const XTime &x) const noexcept {
        return (tv_sec != x.tv_sec) || (tv_usec != x.tv_usec);
    }
    bool operator<(const XTime &x) const noexcept  {
        return (tv_sec < x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec < x.tv_usec));
    }
    bool operator<=(const XTime &x) const noexcept  {
        return (tv_sec <= x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec <= x.tv_usec));
    }
    bool operator>(const XTime &x) const noexcept  {
        return (tv_sec > x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec > x.tv_usec));
    }
    bool operator>=(const XTime &x) const noexcept  {
        return (tv_sec >= x.tv_sec) || ((tv_sec == x.tv_sec) && (tv_usec >= x.tv_usec));
    }
    bool operator!() const noexcept {
        return (tv_sec == 0) && (tv_usec == 0);
    }
    bool isSet() const noexcept {
        return (tv_sec != 0) || (tv_usec != 0);
    }
    long sec() const noexcept {return tv_sec;}
    long usec() const noexcept {return tv_usec;}
    static XTime now() noexcept;
    XString getTimeStr(bool subsecond = true) const;
    XString getTimeFmtStr(const char *fmt, bool subsecond = true) const
#if defined __GNUC__ || defined __clang__
        __attribute__ ((format(strftime,2, 0)))
#endif
    ;

private:
    long tv_sec;
    long tv_usec;
};

#endif /*XTIME_H_*/
