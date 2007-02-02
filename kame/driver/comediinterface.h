#ifndef COMEDI_INTERFACE_H_
#define COMEDI_INTERFACE_H_

#include "interface.h"

#ifdef HAVE_CONFIG_H
 #include <config.h>
 #ifdef HAVE_COMEDI

#include <comedilib.h>

class XComediPort : public XPort
{
public:
 XComediPort(XInterface *interface);
 virtual ~XComediPort();
 
  virtual void open() throw (XInterface::XCommError &);
  virtual void send(const char *str) throw (XInterface::XCommError &);
  virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &);
  virtual void receive() throw (XInterface::XCommError &);
  virtual void receive(unsigned int length) throw (XInterface::XCommError &);

private:
  void comedi_close() throw (XInterface::XCommError &);
  comedi_t *pDev;
  QString errmsg(const QString &str);
};
 	
 #endif //HAVE_COMEDI
#endif
 
#endif // COMEDI_INTERFACE_H_
 