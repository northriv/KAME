//---------------------------------------------------------------------------
#include "measure.h"
#include "interface.h"
#include "xnodeconnector.h"
#include <string>
#include <stdarg.h>
#include "driver.h"
#include "gpib.h"
#include "serial.h"
#include "dummyport.h"
#include "klocale.h"

//---------------------------------------------------------------------------
#define SNPRINT_BUF_SIZE 128

XThreadLocal<std::vector<char> > XPort::s_tlBuffer;

XInterface::XInterfaceError::XInterfaceError(const QString &msg, const char *file, int line)
 : XKameError(msg, file, line) {}
XInterface::XConvError::XConvError(const char *file, int line)
 : XInterfaceError(i18n("Conversion Error"), file, line) {}
XInterface::XCommError::XCommError(const QString &msg, const char *file, int line)
     :  XInterfaceError(i18n("Communication Error") + QString(", ") + msg, file, line) {}
XInterface::XOpenInterfaceError::XOpenInterfaceError(const char *file, int line)
     :  XInterfaceError(i18n("Open Interface Error"), file, line) {}


XInterface::XInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) : 
    XNode(name, runtime), 
    m_bGPIBUseSerialPollOnWrite(true),
    m_bGPIBUseSerialPollOnRead(true),
    m_gpibWaitBeforeWrite(1),
    m_gpibWaitBeforeRead(2),
    m_gpibWaitBeforeSPoll(1),
    m_gpibMAVbit(0x10),
    m_driver(driver),
    m_device(create<XComboNode>("Device", false)),
    m_port(create<XStringNode>("Port", false)),
    m_address(create<XUIntNode>("Address", false)),
    m_baudrate(create<XUIntNode>("BaudRate", false)),
    m_opened(create<XBoolNode>("Opened", true)),
    m_script_send(create<XStringNode>("Send", true)),
    m_script_query(create<XStringNode>("Query", true))
{
#ifdef USE_GPIB
  device()->add("GPIB");
#endif
  device()->add("SERIAL");
  device()->add("DUMMY");
  baudrate()->value(9600);
  
  m_lsnOnSendRequested = m_script_send->onValueChanged().connectWeak(
            false, shared_from_this(), &XInterface::onSendRequested);
  m_lsnOnQueryRequested = m_script_query->onValueChanged().connectWeak(
            false, shared_from_this(), &XInterface::onQueryRequested);
}
void
XInterface::setEOS(const char *str) {
    m_eos = str;
}
         
void
XInterface::open() throw (XInterfaceError &)
{
  lock();
  try {
      
      if(isOpened()) {
          throw XInterfaceError(i18n("Port has already opened"), __FILE__, __LINE__);
      }
        
      g_statusPrinter->printMessage(QString(driver()->getName()) + i18n(": Starting..."));
    
      {
      shared_ptr<XPort> port;
        #ifdef USE_GPIB
          if(device()->to_str() == "GPIB") {
            port.reset(new XGPIBPort(this));
          }
        #endif
          if(device()->to_str() == "SERIAL") {
            port.reset(new XSerialPort(this));
          }
          if(device()->to_str() == "DUMMY") {
            port.reset(new XDummyPort(this));
          }
          
          if(!port) throw XOpenInterfaceError(__FILE__, __LINE__);
            
          port->open();
          m_xport.swap(port);
      }
  }
  catch (XInterfaceError &e) {
          gErrPrint(QString(driver()->getName()) + i18n(": Opening port failed, because"));
          m_xport.reset();
          
          unlock();
          throw e;
  }
  //g_statusPrinter->clear();
        
  unlock();

}
void
XInterface::close()
{
  lock();
//  if(isOpened()) 
//    g_statusPrinter->printMessage(QString(driver()->getName()) + i18n(": Stopping..."));
  m_xport.reset();
  //g_statusPrinter->clear();

  unlock();
}
int
XInterface::scanf(const char *fmt, ...) const {
  int ret;
  va_list ap;

  va_start(ap, fmt);

  ret = vsscanf(&buffer()[0], fmt, ap);

  va_end(ap);
  return ret;    
}
double
XInterface::toDouble() const throw (XInterface::XConvError &) {
    double x;
    int ret = sscanf(&buffer()[0], "%lf", &x);
    if(ret != 1) throw XConvError(__FILE__, __LINE__);
    return x;
}
int
XInterface::toInt() const throw (XInterface::XConvError &) {
    int x;
    int ret = sscanf(&buffer()[0], "%d", &x);
    if(ret != 1) throw XConvError(__FILE__, __LINE__);
    return x;
}
unsigned int
XInterface::toUInt() const throw (XInterface::XConvError &) {
    unsigned int x;
    int ret = sscanf(&buffer()[0], "%u", &x);
    if(ret != 1) throw XConvError(__FILE__, __LINE__);
    return x;
}

const std::vector<char> &
XInterface::buffer() const {return m_xport->buffer();}

void
XInterface::send(const char *str) throw (XCommError &)
{
  lock();
  try {
      dbgPrint(driver()->getName() + " Sending:\"" + str + "\"");
      m_xport->send(str);
  }
  catch (XCommError &e) {
      e.print(driver()->getName() + i18n(" SendError, because "));
      unlock();
      throw e;
  }
  unlock();
}
void
XInterface::sendf(const char *fmt, ...) throw (XInterfaceError &)
{
  va_list ap;
  int buf_size = SNPRINT_BUF_SIZE;
  std::vector<char> buf;
  for(;;) {
      buf.resize(buf_size);
   int ret;
    
      va_start(ap, fmt);
    
      ret = vsnprintf(&buf[0], buf_size, fmt, ap);
    
      va_end(ap);
      
      if(ret < 0) throw XConvError(__FILE__, __LINE__);
      if(ret < buf_size) break;
      
      buf_size *= 2;
  }
  
  this->send(&buf[0]);
}
void
XInterface::write(const char *sendbuf, int size) throw (XCommError &)
{
  lock();
  try {
      dbgPrint(driver()->getName() + QString().sprintf(" Sending %d bytes", size));
      m_xport->write(sendbuf, size);
  }
  catch (XCommError &e) {
      e.print(driver()->getName() + i18n(" SendError, because "));
      unlock();
      throw e;
  }
  unlock();
}
void
XInterface::receive() throw (XCommError &)
{
  lock();
  try {
      dbgPrint(driver()->getName() + " Receiving...");
      m_xport->receive();
      ASSERT(buffer().size());
      dbgPrint(driver()->getName() + " Received;\"" + (const char*)&buffer()[0] + "\"");
  }
  catch (XCommError &e) {
        e.print(driver()->getName() + i18n(" ReceiveError, because "));
        unlock();
        throw e;
  }
  unlock();
}
void
XInterface::receive(unsigned int length) throw (XCommError &)
{
  lock();
  try {
      dbgPrint(driver()->getName() + QString(" Receiving %1 bytes...").arg(length));
      m_xport->receive(length);
      dbgPrint(driver()->getName() + QString("%1 bytes Received.").arg(buffer().size())); 
  }
  catch (XCommError &e) {
      e.print(driver()->getName() + i18n(" ReceiveError, because "));
      unlock();
      throw e;
  }
  unlock();
}
void
XInterface::query(const char *str) throw (XCommError &)
{
  lock();
  try {
      send(str);
      receive();
  }
  catch (XCommError &e) {
      unlock();
      throw e;
  }
  unlock();
}
void
XInterface::queryf(const char *fmt, ...) throw (XInterfaceError &)
{
  va_list ap;
  int buf_size = SNPRINT_BUF_SIZE;
  std::vector<char> buf;
  for(;;) {
      buf.resize(buf_size);
   int ret;
    
      va_start(ap, fmt);
    
      ret = vsnprintf(&buf[0], buf_size, fmt, ap);
    
      va_end(ap);
      
      if(ret < 0) throw XConvError(__FILE__, __LINE__);
      if(ret < buf_size) break;
      
      buf_size *= 2;
  }

  this->query(&buf[0]);
}
void
XInterface::onSendRequested(const shared_ptr<XValueNodeBase> &)
{
shared_ptr<XPort> port = m_xport;
    if(!port)
       throw XInterfaceError(i18n("Port is not opened."), __FILE__, __LINE__);
    port->send(m_script_send->to_str());
}
void
XInterface::onQueryRequested(const shared_ptr<XValueNodeBase> &)
{
shared_ptr<XPort> port = m_xport;    
    if(!port)
       throw XInterfaceError(i18n("Port is not opened."), __FILE__, __LINE__);
    lock();
    try {
        port->send(m_script_query->to_str());
        port->receive();
    }
    catch (XKameError &e) {
        unlock();
        throw e;
    }
    m_lsnOnQueryRequested->mask();
    m_script_query->value(&port->buffer()[0]);
    m_lsnOnQueryRequested->unmask();
    unlock();
}

XPort::XPort(XInterface *interface)
 : m_pInterface(interface)
{
  m_pInterface->opened()->value(true);
  m_pInterface->device()->setUIEnabled(false);
  m_pInterface->port()->setUIEnabled(false);
  m_pInterface->address()->setUIEnabled(false);
  m_pInterface->baudrate()->setUIEnabled(false);   
}
XPort::~XPort()
{
  m_pInterface->device()->setUIEnabled(true);
  m_pInterface->port()->setUIEnabled(true);
  m_pInterface->address()->setUIEnabled(true);
  m_pInterface->baudrate()->setUIEnabled(true);
  m_pInterface->opened()->value(false);
}
