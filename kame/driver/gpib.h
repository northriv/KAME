#ifndef GPIB_H_
#define GPIB_H_

#include "interface.h"

#ifdef HAVE_CONFIG_H
 #include <config.h>
 #ifdef HAVE_LINUX_GPIB
  #define GPIB_LINUX_NI
 #endif
#endif

#if defined WINDOWS || defined __WIN32__
 #define GPIB_WIN32_NI
 #define GPIB_WIN32_IF_4304
#endif // WINDOWS || __WIN32__

#if defined GPIB_LINUX_NI || defined GPIB_WIN32_NI
#define GPIB_NI
#endif

#if defined GPIB_WIN32_IF_4304 || defined GPIB_NI
#define USE_GPIB
#endif

#ifdef GPIB_NI

class XNIGPIBPort : public XPort
{
public:
 XNIGPIBPort(XInterface *interface);
 virtual ~XNIGPIBPort();
 
  virtual void open() throw (XInterface::XCommError &);
  virtual void send(const char *str) throw (XInterface::XCommError &);
  virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &);
  virtual void receive() throw (XInterface::XCommError &);
  virtual void receive(unsigned int length) throw (XInterface::XCommError &);

private:
  int m_ud;
  //! send IFC
  void gpib_reset() throw (XInterface::XCommError &);
  void gpib_spoll_before_write() throw (XInterface::XCommError &);
  void gpib_spoll_before_read() throw (XInterface::XCommError &);
  QString gpibStatus(const QString &msg);
  static int s_cntOpened;
  static XMutex s_lock;
};

typedef XNIGPIBPort XGPIBPort;

#endif /*GPIB_NI*/

#endif /*GPIB_H_*/
