//---------------------------------------------------------------------
#include "caltable.h"
#include "measure.h"
#include <qpushbutton.h>
#include <fstream>
//---------------------------------------------------------------------

XConCalTable::XConCalTable
(const shared_ptr<XThermometerList> &list, FrmCalTable *form)
  :  XQConnector(list, form), 
    m_list(list),
    m_dump(createOrphan<XNode>("dump") ),
    m_temp(createOrphan<XDoubleNode>("temp") ),
    m_value(createOrphan<XDoubleNode>("value") ),
    m_thermometer(createOrphan<XItemNode<XThermometerList, XThermometer> >(
        "thermometer", false, list) ),
    m_conThermo(xqcon_create<XQComboBoxConnector>(m_thermometer, (QComboBox *)form->cmbThermometer) ),
    m_conTemp(xqcon_create<XQLineEditConnector>(m_temp, form->edTemp, false)),
    m_conValue(xqcon_create<XQLineEditConnector>(m_value, form->edValue, false)),
    m_conDump(xqcon_create<XQButtonConnector>(m_dump, form->btnDump)),
    m_pForm(form)
{
  m_lsnTemp = temp()->onValueChanged().connectWeak(false, 
    shared_from_this(),
    &XConCalTable::onTempChanged);
  m_lsnValue = value()->onValueChanged().connectWeak(false,
    shared_from_this(),
    &XConCalTable::onValueChanged);
  m_lsnDump = dump()->onTouch().connectWeak(false,
    shared_from_this(),
    &XConCalTable::onDumpTouched);
}

void
XConCalTable::onTempChanged(const shared_ptr<XValueNodeBase> &)
{
  shared_ptr<XThermometer> thermo = *thermometer();
  if(!thermo) return;
  double ret = thermo->getRawValue(*temp());
  m_lsnValue->mask();
  value()->value(ret);    
  m_lsnValue->unmask();
}
void
XConCalTable::onValueChanged(const shared_ptr<XValueNodeBase> &)
{
  shared_ptr<XThermometer> thermo = *thermometer();
  if(!thermo) return;
  double ret = thermo->getTemp(*value());
  m_lsnTemp->mask();
  temp()->value(ret);
  m_lsnTemp->unmask();
}
void
XConCalTable::onDumpTouched(const shared_ptr<XNode> &)
{
  QString buf;
  shared_ptr<XThermometer> thermo = *thermometer();
  if(!thermo) return;
  std::ofstream stream("calibration.dat", std::ios::out);
  stream << "#temp res err" << std::endl;
  for(double t = *thermo->tempMin(); t < *thermo->tempMax(); t+=0.01)
    {
      double r = thermo->getRawValue(t);
      stream << t << " " << r
	     <<" " << (thermo->getTemp(r) - t) << std::endl;
    }
}

