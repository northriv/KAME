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
 XComediInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver, int subdevice_type);
public:
 virtual ~XComediInterface() {}

  virtual bool isOpened() const {return m_pDev;}
  
  int numChannels();
  comedi_t *comedi_dev() const {return m_pDev;}
  int comedi_subdev() const {return m_subdevice;}
  int comedi_fd() {return comedi_fileno(m_pDev);}
  void comedi_command_test(comedi_cmd *cmd);

  //! for asynchronous IO.
  inline void write(const char *sendbuf, int size);
  inline int read(char *rdbuf, unsigned int length);
protected:
  virtual void open() throw (XInterfaceError &);
  //! This can be called even if has already closed.
  virtual void close() throw (XInterfaceError &);

private:
  comedi_t *m_pDev;
  int m_subdevice;
  int m_subdevice_type;
};
 	
void
XComediInterface::write(const char *sendbuf, int size)
{
	ASSERT(isOpened());
	::write(comedi_fd(), sendbuf, size);
}
int
XComediInterface::read(char *rdbuf, unsigned int length)
{
	ASSERT(isOpened());
	return ::read(comedi_fd(), sendbuf, size);
}
 	
 #endif //HAVE_COMEDI
#endif
 
#endif // COMEDI_INTERFACE_H_
 