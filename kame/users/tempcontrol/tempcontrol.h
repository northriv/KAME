#ifndef tempcontrolH
#define tempcontrolH
//---------------------------------------------------------------------------
#include "thermometer/thermometer.h"
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class FrmTempControl;

class XTempControl : public XPrimaryDriver
{
 XNODE_OBJECT
 protected:
  XTempControl(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  //! usually nothing to do
  virtual ~XTempControl() {}
  //! show all forms belonging to driver
  virtual void showForms();
  
  class XChannel : public XNode
  {
  XNODE_OBJECT
  protected:
    XChannel(const char *name, bool runtime, const shared_ptr<XThermometerList> &list);
  public:
    shared_ptr<XItemNode<XThermometerList, XThermometer> > &thermometer() 
            {return m_thermometer;}
  const   shared_ptr<XComboNode> &excitation() const {return m_excitation;}
  private:
    shared_ptr<XItemNode<XThermometerList, XThermometer> > m_thermometer;
    shared_ptr<XComboNode> m_excitation;
  };
  
  typedef  XAliasListNode<XChannel> XChannelList;
  
  const shared_ptr<XChannelList> &channels() const {return m_channels;}
  //! heater-control channel
  const shared_ptr<XItemNode<XChannelList, XChannel> > &currentChannel() const {return m_currentChannel;}
  const shared_ptr<XDoubleNode> &targetTemp() const {return m_targetTemp;}
  const shared_ptr<XDoubleNode> &manualPower() const {return m_manualPower;}
  const shared_ptr<XDoubleNode> &prop() const {return m_prop;}
  const shared_ptr<XDoubleNode> &interval() const {return m_int;}
  const shared_ptr<XDoubleNode> &deriv() const {return m_deriv;}
  const shared_ptr<XComboNode> &heaterMode() const {return m_heaterMode;}
  const shared_ptr<XComboNode> &powerRange() const {return m_powerRange;}
  const shared_ptr<XDoubleNode> &heaterPower() const {return m_heaterPower;}
  const shared_ptr<XDoubleNode> &sourceTemp() const {return m_sourceTemp;}
  //! holds an averaged error between target temp and actual one
  const shared_ptr<XDoubleNode> &stabilized() const {return m_stabilized;}
 protected:
  //! Start up your threads, connect GUI, and activate signals
  virtual void start();
  //! Shut down your threads, unconnect GUI, and deactivate signals
  //! this may be called even if driver has already stopped.
  virtual void stop();
  
  //! this is called when raw is written 
  //! unless dependency is broken
  //! convert raw to record
  virtual void analyzeRaw() throw (XRecordError&);
  //! this is called after analyze() or analyzeRaw()
  //! record is readLocked
  virtual void visualize();
  
  //! register channel names in your constructor
  //! \param multiread if true, simultaneous reading of multi channels.
  //! \param channel_names array of pointers to channel name. ends with null pointer.
  void createChannels(const shared_ptr<XScalarEntryList> &scalarentries,
    const shared_ptr<XThermometerList> &thermometers, 
    bool multiread, const char **channel_names, const char **excitations);
  
  //! read raw value from the instrument
  virtual double getRaw(shared_ptr<XChannel> &channel) = 0;
  //! obtain current heater power
  //! \sa m_heaterPowerUnit()
  virtual double getHeater() = 0;
  //! ex. "W", "dB", or so
  virtual const char *m_heaterPowerUnit() = 0;
  
  virtual void onPChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onIChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onDChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onTargetTempChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onManualPowerChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onHeaterModeChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onPowerRangeChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onExcitationChanged(const shared_ptr<XValueNodeBase> &) = 0;

 private:
  shared_ptr<XChannelList> m_channels;
  shared_ptr<XItemNode<XChannelList, XChannel> > m_currentChannel;
  shared_ptr<XItemNode<XChannelList, XChannel> > m_setupChannel;
  shared_ptr<XDoubleNode> m_targetTemp;
  shared_ptr<XDoubleNode> m_manualPower;
  shared_ptr<XDoubleNode> m_prop, m_int, m_deriv;
  shared_ptr<XComboNode> m_heaterMode;
  shared_ptr<XComboNode> m_powerRange;
  shared_ptr<XDoubleNode> m_heaterPower, m_sourceTemp;
  //! holds an averaged error between target temp and actual one
  shared_ptr<XDoubleNode> m_stabilized;
  
  shared_ptr<XListener> m_lsnOnPChanged, m_lsnOnIChanged, m_lsnOnDChanged,
    m_lsnOnTargetTempChanged, m_lsnOnManualPowerChanged, m_lsnOnHeaterModeChanged,
    m_lsnOnPowerRangeChanged, m_lsnOnCurrentChannelChanged,
    m_lsnOnSetupChannelChanged, m_lsnOnExcitationChanged;

  void onSetupChannelChanged(const shared_ptr<XValueNodeBase> &);

  xqcon_ptr m_conCurrentChannel, m_conSetupChannel,
    m_conHeaterMode, m_conPowerRange,
    m_conExcitation, m_conThermometer;
  xqcon_ptr m_conTargetTemp, m_conManualPower, m_conP, m_conI, m_conD;
  xqcon_ptr m_conHeater;
  xqcon_ptr m_conTemp;
  
  std::deque<shared_ptr<XScalarEntry> > m_entry_temps;
  std::deque<shared_ptr<XScalarEntry> > m_entry_raws;
 
  shared_ptr<XThread<XTempControl> > m_thread;
  qshared_ptr<FrmTempControl> m_form;
  bool m_multiread;
   
  void *execute(const atomic<bool> &);
  
};

#endif
