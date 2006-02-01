#include "dummyport.h"

XDummyPort::XDummyPort(XInterface *interface) :
    XPort(interface),
    m_stream()
{
}
XDummyPort::~XDummyPort()
{
    m_stream.close();
}
void
XDummyPort::open() throw (XInterface::XCommError &)
{
    m_stream.open("/tmp/kamedummyport.log", std::ios::out);
}
void
XDummyPort::send(const char *str) throw (XInterface::XCommError &)
{
    m_stream << "send:"
        << str << std::endl;
}
void
XDummyPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &)
{
    m_stream << "write:";
    m_stream.write(sendbuf, size);
    m_stream << std::endl;
}
void
XDummyPort::receive() throw (XInterface::XCommError &)
{
    m_stream << "receive:"
         << std::endl;
    buffer().resize(1);
    buffer()[0] = '\0';
}
void
XDummyPort::receive(unsigned int length) throw (XInterface::XCommError &)
{
    m_stream << "receive length = :"
        << length << std::endl;
    buffer().resize(length);
    buffer()[0] = '\0';
}
