//---------------------------------------------------------------------------

#ifndef recordreaderformH
#define recordreaderformH
//---------------------------------------------------------------------------
#include "xnodeconnector.h"

class FrmRecordReader;
class XRawStreamRecordReader;
class XRawStreamRecordReaderConnector : public XQConnector
{
 Q_OBJECT
 XQCON_OBJECT
 protected:
  XRawStreamRecordReaderConnector(
    const shared_ptr<XRawStreamRecordReader> &reader, FrmRecordReader *form);
 public:
  virtual ~XRawStreamRecordReaderConnector() {}

 private:
  shared_ptr<XRawStreamRecordReader> m_reader;
  FrmRecordReader *m_pForm;
  
  xqcon_ptr m_conRecordFile, m_conFF, m_conRW, m_conStop,
     m_conFirst, m_conNext, m_conBack, m_conPosString, m_conSpeed;    
};
  
#endif
