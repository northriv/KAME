//Every KAME source must include this header
//---------------------------------------------------------------------------

#ifndef supportH
#define supportH

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef WORDS_BIGENDIAN
    #ifndef __BIG_ENDIAN__
        #define __BIG_ENDIAN__
    #endif
#endif

#if SIZEOF_SHORT != 2
 #error sizeof short is not 2.
#endif
#if SIZEOF_FLOAT != 4
 #error sizeof float is not 4.
#endif
#if SIZEOF_DOUBLE != 8
 #error sizeof double is not 8.
#endif
/*
    #if SIZEOF_LONG == 4
     typedef long int int32_t;
     typedef unsigned long int uint32_t;
    #else
        #if SIZEOF_INT == 4
          typedef int int32_t;
          typedef unsigned int uint32_t;
        #else
          #error Could not define 32bit integer.
        #endif
    #endif
*/

#ifdef HAVE_LIBGCCPP
    //Boehm GC stuff
    #define GC_OPERATOR_NEW_ARRAY
    #define GC_NAME_CONFLICT
    //default size results in falure; "too many root sets", see private/gc_priv.h
    #define LARGE_CONFIG
    #ifdef __linux__
        #define GC_LINUX_THREADS
        #define _REENTRANT
    #endif
    #define GC_DEBUG
    #include <gc_cpp.h>
    #include <gc_allocator.h>
    #if defined MACOSX
    // for buggy pthread library of GC
        #define BUGGY_PTHRAD_COND_WAIT_USEC 10000
        #define BUGGY_PTHRAD_COND
    #endif
    class kame_gc : public gc {
    public:
        void *operator new(size_t size);
        void operator delete(void *obj);
    };
#else
 #if defined __WIN32__ || defined WINDOWS
 #else
    #include <pthread.h>
 #endif
#endif

#define DEBUG_XTHREAD 1

#ifdef DEBUG_XTHREAD
#else
  #define DEBUG_XTHREAD 0
#endif
#undef ASSERT
#ifdef NDEBUG
 #define ASSERT(expr)
 #define C_ASSERT(expr)
#else
 #define ASSERT(expr) _my_assert((expr), __FILE__, __LINE__)
 #define C_ASSERT(expr) _my_cassert(sizeof(char [ ( expr ) ? 1 : -1 ]))
 void my_assert(const char *file, int line);
 inline void _my_assert(bool var, const char *file, int line) {
    if (!var) my_assert(file, line);
 }
 inline void _my_cassert(size_t ) {}
#endif

//boost
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
using boost::scoped_ptr;
using boost::shared_ptr;
using boost::weak_ptr;
using boost::enable_shared_from_this;
using boost::dynamic_pointer_cast;

#define PI 3.14159265358979

#include <math.h>

#include <qstring.h>

//! Debug printing.
#define dbgPrint(msg) _dbgPrint(msg, __FILE__, __LINE__)
void
_dbgPrint(const QString &str, const char *file, int line);
//! Global Error Message/Printing.
#define gErrPrint(msg) _gErrPrint(msg, __FILE__, __LINE__)
#define gWarnPrint(msg) _gWarningPrint(msg, __FILE__, __LINE__)
void
_gErrPrint(const QString &str, const char *file, int line);
void
_gWarnPrint(const QString &str, const char *file, int line);

//! Base of exception
struct XKameError {
    //! errno is read and cleared after a construction
    XKameError(const QString &s, const char *file, int line);
    void print();
    void print(const QString &header);
    static void print(const QString &msg, const char *file, int line, int _errno);
    const QString &msg() const;
private:
    QString m_msg;
    const char *m_file;
    int m_line;
    int m_errno;
};

//! If true, Log all dbgPrint().
extern bool g_bLogDbgPrint;

//! round value to the nearest 10s. ex. 42.3 to 10, 120 to 100
double roundlog10(double val);
//! round value within demanded precision.
//! ex. 38.32, 40 to 30, 0.4234, 0.01 to 0.42
double setprec(double val, double prec);

//! convert control characters to visible (ex. \xx).
std::string dumpCString(const char *cstr);

//! \sa printf()
std::string formatString(const char *format, ...)
     __attribute__ ((format(printf,1,2)));

std::string formatDouble(const char *fmt, double val);
//! validator
//! throw XKameError
//! \sa XValueNode
void formatDoubleValidator(std::string &fmt);

namespace KAME {
    unsigned int rand();
    //! thread-safe version of i18n().
    //! this is not needed in QT4 or later.
    QString i18n(const char* eng);    
}

//---------------------------------------------------------------------------
#endif
