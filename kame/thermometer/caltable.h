//----------------------------------------------------------------------------
#ifndef caltableH
#define caltableH
//----------------------------------------------------------------------------

#include "caltableform.h"
#include "thermometer.h"
#include "xnodeconnector.h"
//----------------------------------------------------------------------------

class FrmCalTable;

class XConCalTable : public XQConnector
{
 Q_OBJECT
 XQCON_OBJECT
 protected:
  XConCalTable(const shared_ptr<XThermometerList> &list, FrmCalTable *form);
 public:
  virtual ~XConCalTable() {}

  const shared_ptr<XNode> &dump() const {return m_dump;}
  const shared_ptr<XDoubleNode> &temp() const {return m_temp;}
  const shared_ptr<XDoubleNode> &value() const {return m_value;}  
  shared_ptr<XItemNode<XThermometerList, XThermometer> >
     &thermometer() {return m_thermometer;}  
  
 private:
  shared_ptr<XThermometerList> m_list; 
 
  shared_ptr<XNode> m_dump;
  shared_ptr<XDoubleNode> m_temp, m_value;
  shared_ptr<XItemNode<XThermometerList, XThermometer> > m_thermometer;
  xqcon_ptr m_conThermo, m_conTemp, m_conValue, m_conDump;
  
  shared_ptr<XListener> m_lsnTemp, m_lsnValue;
  shared_ptr<XListener> m_lsnDump;
  
  void onTempChanged(const shared_ptr<XValueNodeBase> &);
  void onValueChanged(const shared_ptr<XValueNodeBase> &);  
  void onDumpTouched(const shared_ptr<XNode> &);
  FrmCalTable *m_pForm;
};

//----------------------------------------------------------------------------
#endif
