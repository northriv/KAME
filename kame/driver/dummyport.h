#ifndef DUMMYPORT_H_
#define DUMMYPORT_H_
#include "interface.h"

#include <fstream>

class XDummyPort : public XPort {
public:
      XDummyPort(XInterface *interface);
      virtual ~XDummyPort();
      virtual void open() throw (XInterface::XCommError &);
      virtual void send(const char *str) throw (XInterface::XCommError &);
      virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &);
      virtual void receive() throw (XInterface::XCommError &);
      virtual void receive(unsigned int length) throw (XInterface::XCommError &);
private:
    std::ofstream m_stream;
};

#endif /*DUMMYPORT_H_*/
