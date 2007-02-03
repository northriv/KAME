#ifndef PRIMARYDRIVER_H_
#define PRIMARYDRIVER_H_

#include "driver.h"
#include "interface.h"

class XPrimaryDriver : public XDriver
{
 XNODE_OBJECT
 protected:
  XPrimaryDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XPrimaryDriver() {}
  
  //! show all forms belonging to driver
  virtual void showForms() = 0;
  
  template <typename tVar>
  static void push(tVar, std::vector<char> &buf);

  template <typename tVar>
  static tVar pop(std::vector<char>::iterator &it);
  
  virtual const shared_ptr<XRecordDependency> dependency() const 
        {return shared_ptr<XRecordDependency>();}


  //! Shut down your threads, unconnect GUI, and deactivate signals
  //! this may be called even if driver has already stopped.
  virtual void stop() = 0;  
 protected:
  //! Start up your threads, connect GUI, and activate signals
  virtual void start() = 0;
  //! Be called for closing interfaces.
  virtual void afterStop() = 0;  
    
  //! this is called when raw is written 
  //! unless dependency is broken
  //! convert raw to record
  //! \sa analyze()
  virtual void analyzeRaw() throw (XRecordError&) = 0;
  
  //! clear thread-local raw buffer.
  void clearRaw() {rawData().clear();}
  //! will call analyzeRaw() if dependency.
  //! unless dependency is broken.
  //! \arg time_awared time when a visible phenomenon started
  //! \arg time_recorded usually pass \p XTime::now()
  //! \sa timeAwared()
  //! \sa time()
  void finishWritingRaw(const XTime &time_awared, const XTime &time_recorded);
  //! raw data. Thread-Local storaged.
  std::vector<char> &rawData() {return *s_tlRawData;}

  //! These are FIFO (fast in fast out)
  //! push raw data to raw record
  template <typename tVar>
  void push(tVar);
  //! read raw record
  template <typename tVar>
  tVar pop() throw (XBufferUnderflowRecordError&);

 private:
  friend class XRawStreamRecordReader;
  friend class XRawStreamRecorder;

  //! raw data
  static XThreadLocal<std::vector<char> > s_tlRawData;
  typedef std::vector<char>::iterator RawData_it;
  static XThreadLocal<RawData_it> s_tl_pop_it;
  
  static void _push_char(char, std::vector<char> &buf);
  static void _push_short(short, std::vector<char> &buf);
  static void _push_int32(int32_t, std::vector<char> &buf);
  static void _push_double(double, std::vector<char> &buf);
  static char _pop_char(std::vector<char>::iterator &it);
  static short _pop_short(std::vector<char>::iterator &it);
  static int32_t _pop_int32(std::vector<char>::iterator &it);
  static double _pop_double(std::vector<char>::iterator &it);
};

#endif /*PRIMARYDRIVER_H_*/
