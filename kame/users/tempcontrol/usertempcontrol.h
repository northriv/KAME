//---------------------------------------------------------------------------

#ifndef usertempcontrolH
#define usertempcontrolH

#include "tempcontrol.h"
#include "oxforddriver.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------
//ITC503 Oxford
class XITC503 : public XOxfordDriver<XTempControl>
{
 XNODE_OBJECT
 protected:
  XITC503(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  ~XITC503() {}

 protected:
  virtual double getRaw(shared_ptr<XChannel> &channel);
  //! obtain current heater power
  //! \sa m_heaterPowerUnit()
  virtual double getHeater();
  //! ex. "W", "dB", or so
  virtual const char *m_heaterPowerUnit() {return "%";}
  
  virtual void afterStart();
  
  virtual void onPChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onIChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onDChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTargetTempChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onManualPowerChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onHeaterModeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onPowerRangeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onExcitationChanged(const shared_ptr<XValueNodeBase> &);
 private:
};

//AVS47-IB
//AVS47 and TS530A
class XAVS47IB:public XCharDeviceDriver<XTempControl>
{
 XNODE_OBJECT
 protected:
  XAVS47IB(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  ~XAVS47IB() {}

 protected:
  virtual double getRaw(shared_ptr<XChannel> &channel);
  //! obtain current heater power
  //! \sa m_heaterPowerUnit()
  virtual double getHeater();
  //! ex. "W", "dB", or so
  virtual const char *m_heaterPowerUnit() {return "W";}
  
  virtual void afterStart();
  virtual void beforeStop();
  
  virtual void onPChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onIChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onDChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTargetTempChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onManualPowerChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onHeaterModeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onPowerRangeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onExcitationChanged(const shared_ptr<XValueNodeBase> &);
 private:
  double read(const char *str);

  void setTemp(double temp);
  void setHeaterMode(int mode) {}
  int setPoint();
  //AVS-47 COMMANDS
  int setRange(unsigned int range);
  double getRes();
  int getRange();
  //TS-530 COMMANDS
  int setBias(unsigned int bias);
  void setPowerRange(int range);

  int m_autorange_wait;
};

//Cryo-con base class
class XCryocon : public XCharDeviceDriver<XTempControl>
{
 XNODE_OBJECT
 protected:
  XCryocon(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XCryocon() {}

 protected:
  virtual double getRaw(shared_ptr<XChannel> &channel);
  //! obtain current heater power
  //! \sa m_heaterPowerUnit()
  virtual double getHeater();
  //! ex. "W", "dB", or so
  virtual const char *m_heaterPowerUnit() {return "%";}
  
  virtual void afterStart();
  
  virtual void onPChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onIChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onDChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTargetTempChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onManualPowerChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onHeaterModeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onPowerRangeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onExcitationChanged(const shared_ptr<XValueNodeBase> &);
 private:
  void setTemp(double temp);
  //        void SetChannel(XChannel *channel);
  void setHeaterMode();
  void getChannel();
  int control();
  int stopControl();
  double getInput(shared_ptr<XChannel> &channel);
  int setHeaterSetPoint(double value);
};

//Cryo-con Model 32 Cryogenic Inst.
class XCryoconM32:public XCryocon
{
 XNODE_OBJECT
 protected:
  XCryoconM32(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  ~XCryoconM32() {}

 protected:
};

//Cryo-con Model 62 Cryogenic Inst.
class XCryoconM62:public XCryocon
{
 XNODE_OBJECT
 protected:
  XCryoconM62(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  ~XCryoconM62() {}

 protected:
  virtual void afterStart();
};

//LakeShore 340
class XLakeShore340:public XCharDeviceDriver<XTempControl>
{
 XNODE_OBJECT
 protected:
  XLakeShore340(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  ~XLakeShore340() {}

 protected:
  virtual double getRaw(shared_ptr<XChannel> &channel);
  //! obtain current heater power
  //! \sa m_heaterPowerUnit()
  virtual double getHeater();
  //! ex. "W", "dB", or so
  virtual const char *m_heaterPowerUnit() {return "%";}
  
  virtual void afterStart();
  
  virtual void onPChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onIChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onDChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTargetTempChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onManualPowerChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onHeaterModeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onPowerRangeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onExcitationChanged(const shared_ptr<XValueNodeBase> &);
 private:
};
#endif
