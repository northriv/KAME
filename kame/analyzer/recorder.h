//---------------------------------------------------------------------------

#ifndef recorderH
#define recorderH
//---------------------------------------------------------------------------
#include "xnode.h"
#include "xnodeconnector.h"
#include "driver.h"

#include <fstream>

#define MAX_RAW_RECORD_SIZE 100000

class XRawStream : public XNode
{
 XNODE_OBJECT
 protected:
  XRawStream(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist);
 public:
  virtual ~XRawStream();
  const shared_ptr<XStringNode> &filename() const {return m_filename;}  
 protected:
  shared_ptr<XDriverList> m_drivers;
  //! file descriptor of GZip
  void *m_pGFD;
  XMutex m_filemutex;
 private:
  shared_ptr<XStringNode> m_filename;

};
class XRawStreamRecorder : public XRawStream
{
 XNODE_OBJECT
 protected:
  XRawStreamRecorder(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist);
 public:
  virtual ~XRawStreamRecorder() {}
  const shared_ptr<XBoolNode> &recording() const {return m_recording;}
 protected:
  virtual void onCatch(const shared_ptr<XNode> &node);
  virtual void onRelease(const shared_ptr<XNode> &node);  
 private:
  void onOpen(const shared_ptr<XValueNodeBase> &);
  
  shared_ptr<XListener> m_lsnOnRecord;
  shared_ptr<XListener> m_lsnOnCatch;
  shared_ptr<XListener> m_lsnOnRelease;
  shared_ptr<XListener> m_lsnOnFlush;
  shared_ptr<XListener> m_lsnOnOpen;
  
  void onRecord(const shared_ptr<XDriver> &driver);
  void onFlush(const shared_ptr<XValueNodeBase> &);
  shared_ptr<XBoolNode> m_recording;
};


class XScalarEntryList;

class XTextWriter : public XNode
{
 XNODE_OBJECT
 protected:
  XTextWriter(const char *name, bool runtime,
     const shared_ptr<XDriverList> &driverlist, const shared_ptr<XScalarEntryList> &entrylist);
 public:
  virtual ~XTextWriter() {}

  const shared_ptr<XStringNode> &filename() const {return m_filename;}
  const shared_ptr<XBoolNode> &recording() const {return m_recording;}
  const shared_ptr<XStringNode> &lastLine() const {return m_lastLine;}

 protected:
  virtual void onCatch(const shared_ptr<XNode> &node);
  virtual void onRelease(const shared_ptr<XNode> &node);  
 private:
  shared_ptr<XDriverList> m_drivers;
  shared_ptr<XScalarEntryList> m_entries;
  shared_ptr<XStringNode> m_filename;
  shared_ptr<XStringNode> m_lastLine;
  shared_ptr<XBoolNode> m_recording;
  shared_ptr<XListener> m_lsnOnRecord;
  shared_ptr<XListener> m_lsnOnFlush;
  shared_ptr<XListener> m_lsnOnCatch;
  shared_ptr<XListener> m_lsnOnRelease; 
  shared_ptr<XListener> m_lsnOnLastLineChanged; 
  shared_ptr<XListener> m_lsnOnFilenameChanged;
  void onRecord(const shared_ptr<XDriver> &driver);
  void onFlush(const shared_ptr<XValueNodeBase> &);
  void onLastLineChanged(const shared_ptr<XValueNodeBase> &);
  void onFilenameChanged(const shared_ptr<XValueNodeBase> &);
    
  std::fstream m_stream;
  XRecursiveMutex m_filemutex;
};


#endif
