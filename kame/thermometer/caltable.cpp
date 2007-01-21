//---------------------------------------------------------------------
#include "caltable.h"
#include "measure.h"
#include <qpushbutton.h>
#include <fstream>
#include <klocale.h>
#include "graph.h"
#include "graphwidget.h"
#include "graphnurlform.h"
#include "xwavengraph.h"
//---------------------------------------------------------------------

XConCalTable::XConCalTable
(const shared_ptr<XThermometerList> &list, FrmCalTable *form)
  :  XQConnector(list, form), 
    m_list(list),
    m_display(createOrphan<XNode>("display") ),
    m_temp(createOrphan<XDoubleNode>("temp") ),
    m_value(createOrphan<XDoubleNode>("value") ),
    m_thermometer(createOrphan<XItemNode<XThermometerList, XThermometer> >(
        "thermometer", false, list) ),
    m_conThermo(xqcon_create<XQComboBoxConnector>(m_thermometer, (QComboBox *)form->cmbThermometer) ),
    m_conTemp(xqcon_create<XQLineEditConnector>(m_temp, form->edTemp, false)),
    m_conValue(xqcon_create<XQLineEditConnector>(m_value, form->edValue, false)),
    m_conDisplay(xqcon_create<XQButtonConnector>(m_display, form->btnDisplay)),
    m_pForm(form),
    m_waveform(new FrmGraphNURL(g_pFrmMain)),
    m_wave(createOrphan<XWaveNGraph>("Waveform", true, m_waveform.get()))
{
  m_lsnTemp = temp()->onValueChanged().connectWeak(false, 
    shared_from_this(),
    &XConCalTable::onTempChanged);
  m_lsnValue = value()->onValueChanged().connectWeak(false,
    shared_from_this(),
    &XConCalTable::onValueChanged);
  m_lsnDisplay = display()->onTouch().connectWeak(false,
    shared_from_this(),
    &XConCalTable::onDisplayTouched);

  m_waveform->setCaption(KAME::i18n("Thermometer Calibration"));
  {
      const char *labels[] = {"Temp. [K]", "Value", "T(v(T))-T [K]"};
      m_wave->setColCount(3, labels);
      m_wave->selectAxes(0, 1, 2);
      m_wave->plot1()->label()->value(KAME::i18n("Curve"));
      m_wave->plot1()->drawPoints()->value(false);
      m_wave->plot2()->label()->value(KAME::i18n("Error"));
      m_wave->plot2()->drawPoints()->value(false);
      shared_ptr<XAxis> axisx = *m_wave->plot1()->axisX();
      axisx->logScale()->value(true);
      shared_ptr<XAxis> axisy = *m_wave->plot1()->axisY();
      axisy->logScale()->value(true);
      m_wave->clear();
  }
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
XConCalTable::onDisplayTouched(const shared_ptr<XNode> &)
{
  shared_ptr<XThermometer> thermo = *thermometer();
  if(!thermo) {
      m_wave->clear();
      return;
  }
  const int length = 1000;
  double step = (log(*thermo->tempMax()) - log(*thermo->tempMin())) / length;
  {   XScopedWriteLock<XWaveNGraph> lock(*m_wave);
      m_wave->setRowCount(length);
      double lt = log(*thermo->tempMin());
      for(int i = 0; i < length; i++)
        {
          double t = exp(lt);
          double r = thermo->getRawValue(t);
          m_wave->cols(0)[i] = t;
          m_wave->cols(1)[i] = r;
          m_wave->cols(2)[i] = thermo->getTemp(r) - t;
          lt += step;
        }
  }
  m_waveform->show();
  m_waveform->raise();  
}
