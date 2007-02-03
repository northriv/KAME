#ifndef COMEDI_INTERFACE_H_
#define COMEDI_INTERFACE_H_

#include "interface.h"

#ifdef HAVE_CONFIG_H
 #include <config.h>
 #ifdef HAVE_COMEDI

#include <comedilib.h>

class XComediInterface : public XInterface
{
 XNODE_OBJECT
protected:
 XComediInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
public:
 virtual ~XComediInterface() {}
   
  virtual void open() throw (XInterfaceError &);
  //! This can be called even if has already closed.
  virtual void close();
  
  virtual bool isOpened() const {return pDev;}
private:
  comedi_t *pDev;
};
 	
 #endif //HAVE_COMEDI
#endif
 
#endif // COMEDI_INTERFACE_H_
 