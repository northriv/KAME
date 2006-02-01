#ifndef SERIAL_H_
#define SERIAL_H_

#include "interface.h"

#if  defined __linux__ || defined MACOSX
#define SERIAL_POSIX
#endif //__linux__ || LINUX

#if defined WINDOWS || defined __WIN32__
 #define SERIAL_WIN32
#endif // WINDOWS || __WIN32__

#if defined SERIAL_WIN32 || defined SERIAL_POSIX
#define USE_SERIAL
#endif

#ifdef SERIAL_POSIX

class XPosixSerialPort : public XPort
{
public:
 XPosixSerialPort(XInterface *interface);
 virtual ~XPosixSerialPort();
 
  virtual void open() throw (XInterface::XCommError &);
  virtual void send(const char *str) throw (XInterface::XCommError &);
  virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &);
  virtual void receive() throw (XInterface::XCommError &);
  virtual void receive(unsigned int length) throw (XInterface::XCommError &);  
private:
  int m_scifd;
};

typedef XPosixSerialPort XSerialPort;

#endif /*SERIAL_POSIX*/

#endif /*SERIAL_H_*/
