#include "xtime.h"
#include <sys/time.h>

void msecsleep(unsigned int ms)
{
  if(!ms) usleep(0);
  struct timespec rem, req;
  req.tv_sec = ms / 1000;
  req.tv_nsec = ((ms % 1000) * 1000000);
  while(nanosleep(&req, &rem)) req = rem;
  return;
}

unsigned long timeStamp() {
#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__
    uint64_t r;
    asm volatile("rdtsc" : "=A" (r));
    return (unsigned int)(r / 256u);
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

std::string
XTime::getTimeStr(bool subsecond) const
{
    if(*this) {
      char str[100];
      ctime_r(&tv_sec, str);
      str[strlen(str) - 1] = '\0';
      if(subsecond)
          sprintf(str + strlen(str), " +%.3dms", (int)tv_usec/1000);
      return std::string(str);
    }
    else {
        return std::string();
    }
}
std::string
XTime::getTimeFmtStr(const char *fmt, bool subsecond) const
{
    if(*this) {
      char str[100];
      struct tm time;
      localtime_r(&tv_sec, &time);
      strftime(str, 100, fmt, &time);
      if(subsecond)
          sprintf(str + strlen(str), " +%.3f", 1e-6 * tv_usec);
      return std::string(str);
    }
    else {
        return std::string();
    }
}

