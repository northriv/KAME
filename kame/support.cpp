#include <errno.h>
#include <string.h>
#include <klocale.h>
#include "support.h"
#include "xtime.h"
#include "measure.h"
#include "threadlocal.h"

#if defined __linux__ || defined MACOSX
#undef TRAP_FPE
 #if defined TRAP_FPE && defined __linux__
  #include <fpu_control.h>
  static void __attribute__ ((constructor)) trapfpe (void)
  {
   fpu_control_t cw =
     _FPU_DEFAULT & ~(_FPU_MASK_IM | _FPU_MASK_ZM | _FPU_MASK_OM);
  _FPU_SETCW(cw);
  }
 #endif
#endif // __linux__

#ifdef HAVE_LIBGCCPP
    static void kame_cleanup(void *obj, void *data) {
        char *msg = (char*)data;
        gErrPrint(QString("Memory Leak! addr=%1, %2")
            .arg((unsigned int)obj, 0, 16)
            .arg(QString::fromUtf8(msg)));
        free(msg);
    }
    void *kame_gc::operator new(size_t size) {
        char *buf = (char*)malloc(256);
        snprintf(buf, 256, "size=%u, time=%s", 
            (unsigned int)size, (const char*)XTime::now().getTimeStr().utf8());        
        return ::operator new(size, UseGC, kame_cleanup, buf);
    }
    void kame_gc::operator delete(void *obj) {
        GC_register_finalizer_ignore_self( GC_base(obj), 0, 0, 0, 0 );
        gc::operator delete(obj);
    }
#endif

#ifndef NDEBUG
  #ifdef __linux__
    #include <execinfo.h>
  #endif
    void my_assert(const char *file, int line)
    {
        fprintf(stderr, "assertion failed %s:%d\n",file,line);
     #ifdef __linux__
        void *trace[128];
        int n = backtrace(trace, sizeof(trace) / sizeof(trace[0]));
        backtrace_symbols_fd(trace, n, 1);
     #endif
        abort();
    }
#endif // NDEBUG

XKameError::XKameError(const QString &s, const char *file, int line)
     : m_msg(s), m_file(file), m_line(line), m_errno(errno) {
    errno = 0;
}
void
XKameError::print(const QString &header) {
    print(header + m_msg, m_file, m_line, m_errno);
}
void
XKameError::print() {
    print("");
}
void
XKameError::print(const QString &msg, const char *file, int line, int _errno) {
    char buf[256];
    if(_errno) {
        errno = 0;
        strerror_r(_errno, buf, sizeof(buf));
        if(!errno)
                 _gErrPrint(msg + " " + QString::fromUtf8(buf), file, line);
        else
                 _gErrPrint(msg + " (strerror failed)", file, line);
        errno = 0;
    }
    else {
         _gErrPrint(msg, file, line);
    }    
}

const QString &
XKameError::msg() const
{
    return m_msg;
}

bool g_bLogDbgPrint;

#include <iostream>
#include <fstream>

#include <thread.h>
static std::ofstream g_debugofs("/tmp/kame.log", std::ios::out);
static XMutex g_debug_mutex;

double roundlog10(double val)
{
int i = lrint(log10(val));
  return pow(10.0, (double)i);
}
double setprec(double val, double prec)
{
  double x;

  if(prec <= 1e-100) return val;
  x = roundlog10(prec/2);
  double f = rint(val / x);
  double z = (fabs(f) < (double)0x8fffffff) ? ((int)f) * x : f * x;
  return  z;
}


//---------------------------------------------------------------------------
#include "xtime.h"


void
_dbgPrint(const QString &str, const char *file, int line)
{
  if(!g_bLogDbgPrint) return;
  XScopedLock<XMutex> lock(g_debug_mutex);
  g_debugofs 
  	<< (const char*)(QString("0x%1:%2:%3:%4 %5")
        .arg((unsigned int)threadID(), 0, 16)
        .arg(XTime::now().getTimeStr())
        .arg(file)
        .arg(line)
        .arg(str))
        .local8Bit()
  	<< std::endl;
}
void
_gErrPrint(const QString &str, const char *file, int line)
{
   {
      XScopedLock<XMutex> lock(g_debug_mutex);
      g_debugofs 
        << (const char*)(QString("Err:0x%1:%2:%3:%4 %5")
            .arg((unsigned int)threadID(), 0, 16)
            .arg(XTime::now().getTimeStr())
            .arg(file)
            .arg(line)
            .arg(str))
            .local8Bit()
        << std::endl;
      fprintf(stderr, "err:%s:%d %s\n", file, line, (const char*)str.local8Bit());
   }
  shared_ptr<XStatusPrinter> statusprinter = g_statusPrinter;
  if(statusprinter) statusprinter->printError(str);
}

QString formatDouble(const char *fmt, double var)
{
    char cbuf[128];
      if(strlen(fmt) == 0) {
          snprintf(cbuf, sizeof(cbuf), "%g", var);
          return QString(cbuf);
      }
      
      if(!strncmp(fmt, "TIME:", 5)) {
            XTime time;
            time += var;
            if(fmt[5]) 
                return time.getTimeFmtStr(fmt + 5, false);
            else
                return time.getTimeStr(false);
      }
      snprintf(cbuf, sizeof(cbuf), fmt, var);
      return QString(cbuf);
}
void formatDoubleValidator(QString &fmt) {
    if(fmt.isEmpty()) return;

    std::string buf(fmt.local8Bit());

    if(!strncmp(buf.c_str(), "TIME:", 5)) return;
    
    int arg_cnt = 0;
    for(unsigned int pos = 0;;) {
        pos = buf.find('%', pos);
        if(pos == std::string::npos) break;
        pos++;
        if(buf[pos] == '%') {
            continue;
        }
        arg_cnt++;
        if(arg_cnt > 1) {
            throw XKameError(i18n("Illegal Format, too many %s."), __FILE__, __LINE__);
        }
        char conv;
        if((sscanf(buf.c_str() + pos, "%*[+-'0# ]%*f%c", &conv) != 1) &&
            (sscanf(buf.c_str() + pos, "%*[+-'0# ]%c", &conv) != 1) &&
            (sscanf(buf.c_str() + pos, "%*f%c", &conv) != 1) &&
            (sscanf(buf.c_str() + pos, "%c", &conv) != 1)) {
            throw XKameError(i18n("Illegal Format."), __FILE__, __LINE__);                
        }
        if(std::string("eEgGf").find(conv) == std::string::npos)
            throw XKameError(i18n("Illegal Format, no float conversion."), __FILE__, __LINE__);  
    }
    if(arg_cnt == 0)
        throw XKameError(i18n("Illegal Format, no %."), __FILE__, __LINE__);
}

static XThreadLocal<unsigned int> stl_random_seed;
namespace KAME {
    unsigned int rand() {
        return rand_r(&(*stl_random_seed));
    }
}
