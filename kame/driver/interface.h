#ifndef INTERFACE_H_
#define INTERFACE_H_

#include "xnode.h"
#include "xlistnode.h"
#include "xitemnode.h"
#include <vector>

class XDriver;
class XPort;
//#include <stdarg.h>

class XInterface : public XNode
{
 XNODE_OBJECT
protected:
 XInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
public:
 virtual ~XInterface() {}
 
 struct XInterfaceError : public XKameError {
    XInterfaceError(const QString &msg, const char *file, int line);
 };
 struct XConvError : public XInterfaceError {
    XConvError(const char *file, int line);
 };
 struct XCommError : public XInterfaceError {
    XCommError(const QString &, const char *file, int line);
 };
 struct XOpenInterfaceError : public XInterfaceError {
    XOpenInterfaceError(const char *file, int line);
 };
 
 shared_ptr<XDriver> driver() const {return m_driver.lock();}
 
 const shared_ptr<XComboNode> &device() const {return m_device;}
  //! GPIB port or Serial port device
 const shared_ptr<XStringNode> &port() const {return m_port;}
  //! GPIB address
 const shared_ptr<XUIntNode> &address() const {return m_address;}
  //! Serial port baud rate
 const shared_ptr<XUIntNode> &baudrate() const {return m_baudrate;}
 //! True if interface is opened
 const shared_ptr<XBoolNode> &opened() const {return m_opened;}

  void lock() {m_mutex.lock();}
  void unlock() {m_mutex.unlock();}
  bool isLocked() const {return m_mutex.isLockedByCurrentThread();}

  XRecursiveMutex &mutex() {return m_mutex;}
  
  //! Buffer is Thread-Local-Strage.
  //! Therefore, be careful when you access multi-interfaces in one thread.
  //! \sa XThreadLocal
  const std::vector<char> &buffer() const;
  //! error-check is user's responsibility.
  int scanf(const char *format, ...) const
   __attribute__ ((format(scanf,2,3)));
  double toDouble() const throw (XConvError &);
  int toInt() const throw (XConvError &);
  unsigned int toUInt() const throw (XConvError &);
  
  virtual void open() throw (XInterfaceError &);
  //! This can be called even if has already closed.
  virtual void close();
  
  void send(const std::string &str) throw (XCommError &);
  virtual void send(const char *str) throw (XCommError &);
  //! format version of send()
  //! \sa printf()
  void sendf(const char *format, ...) throw (XInterfaceError &)
     __attribute__ ((format(printf,2,3)));
  virtual void write(const char *sendbuf, int size) throw (XCommError &);
  virtual void receive() throw (XCommError &);
  virtual void receive(unsigned int length) throw (XCommError &);
  void query(const std::string &str) throw (XCommError &);
  virtual void query(const char *str) throw (XCommError &);
  //! format version of query()
  //! \sa printf()
  void queryf(const char *format, ...) throw (XInterfaceError &)
     __attribute__ ((format(printf,2,3)));
  
  void setEOS(const char *str);
  void setGPIBUseSerialPollOnWrite(bool x) {m_bGPIBUseSerialPollOnWrite = x;}
  void setGPIBUseSerialPollOnRead(bool x) {m_bGPIBUseSerialPollOnRead = x;}
  void setGPIBWaitBeforeWrite(int msec) {m_gpibWaitBeforeWrite = msec;}
  void setGPIBWaitBeforeRead(int msec) {m_gpibWaitBeforeRead = msec;}
  void setGPIBWaitBeforeSPoll(int msec) {m_gpibWaitBeforeSPoll = msec;}
  void setGPIBMAVbit(unsigned char x) {m_gpibMAVbit = x;}
  
  const std::string &eos() const {return m_eos;}
  bool gpibUseSerialPollOnWrite() const {return m_bGPIBUseSerialPollOnWrite;}
  bool gpibUseSerialPollOnRead() const {return m_bGPIBUseSerialPollOnRead;}
  int gpibWaitBeforeWrite() const {return m_gpibWaitBeforeWrite;}
  int gpibWaitBeforeRead() const {return m_gpibWaitBeforeRead;}
  int gpibWaitBeforeSPoll() const {return m_gpibWaitBeforeSPoll;}
  unsigned char gpibMAVbit() const {return m_gpibMAVbit;}
  
  bool isOpened() const {return m_xport;}
private:
  std::string m_eos;
  bool m_bGPIBUseSerialPollOnWrite;
  bool m_bGPIBUseSerialPollOnRead;
  int m_gpibWaitBeforeWrite;
  int m_gpibWaitBeforeRead;
  int m_gpibWaitBeforeSPoll;
  unsigned char m_gpibMAVbit; //! don't check if zero
  weak_ptr<XDriver> m_driver;
  bool m_bOpened;
  shared_ptr<XComboNode> m_device;
  shared_ptr<XStringNode> m_port;
  shared_ptr<XUIntNode> m_address;
  shared_ptr<XUIntNode> m_baudrate;
  shared_ptr<XBoolNode> m_opened;
  
  shared_ptr<XPort> m_xport;
  
  //! for scripting
  shared_ptr<XStringNode> m_script_send;
  shared_ptr<XStringNode> m_script_query;
  shared_ptr<XListener> m_lsnOnSendRequested;
  shared_ptr<XListener> m_lsnOnQueryRequested;
  void onSendRequested(const shared_ptr<XValueNodeBase> &);
  void onQueryRequested(const shared_ptr<XValueNodeBase> &);
    
  XRecursiveMutex m_mutex;
};

class XPort {
public:
      XPort(XInterface *interface);
      virtual ~XPort();
      virtual void open() throw (XInterface::XCommError &) = 0;
      virtual void send(const char *str) throw (XInterface::XCommError &) = 0;
      virtual void write(const char *sendbuf, int size) throw (XInterface::XCommError &) = 0;
      virtual void receive() throw (XInterface::XCommError &) = 0;
      virtual void receive(unsigned int length) throw (XInterface::XCommError &) = 0;
      //! Buffer is Thread-Local-Strage.
      //! Therefore, be careful when you access multi-interfaces in one thread.
      //! \sa XThreadLocal
      std::vector<char>& buffer() {return *s_tlBuffer;}
protected:
      static XThreadLocal<std::vector<char> > s_tlBuffer;
      XInterface *m_pInterface;
};

class XInterfaceList : public XAliasListNode<XInterface>
{
 XNODE_OBJECT
protected:
    XInterfaceList(const char *name, bool runtime) : XAliasListNode<XInterface>(name, runtime) {}
};

#endif /*INTERFACE_H_*/
