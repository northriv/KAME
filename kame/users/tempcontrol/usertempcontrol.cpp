//---------------------------------------------------------------------------
#include "usertempcontrol.h"
#include "charinterface.h"

XITC503::XITC503(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
 : XOxfordDriver<XTempControl>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  const char *channels_create[] = {"1", "2", "3", 0L};
  const char *excitations_create[] = {0L};
  createChannels(scalarentries, thermometers, true, channels_create, excitations_create);
  heaterMode()->add("PID");
  heaterMode()->add("Man");
}
void
XITC503::open() throw (XInterface::XInterfaceError &)
{
  start();
  powerRange()->setUIEnabled(false);
}
double
XITC503::getRaw(shared_ptr<XChannel> &channel)
{
  ASSERT(interface()->isLocked());
  interface()->send("X");
  return read(QString(channel->getName()).toInt());
}
double
XITC503::getHeater()
{
  return read(5);
}
void
XITC503::onPChanged(const shared_ptr<XValueNodeBase> &)
{
  interface()->send("P" + prop()->to_str());
}
void
XITC503::onIChanged(const shared_ptr<XValueNodeBase> &){
  interface()->send("I" + interval()->to_str());
}
void
XITC503::onDChanged(const shared_ptr<XValueNodeBase> &){
  interface()->send("D" + deriv()->to_str());
}
void
XITC503::onTargetTempChanged(const shared_ptr<XValueNodeBase> &){
  if(heaterMode()->to_str() == "PID")
    interface()->send("T" + targetTemp()->to_str());
}
void
XITC503::onManualPowerChanged(const shared_ptr<XValueNodeBase> &){
  if(heaterMode()->to_str() == "Man")
    interface()->send("O" + manualPower()->to_str());
}
void
XITC503::onHeaterModeChanged(const shared_ptr<XValueNodeBase> &){
}
void
XITC503::onPowerRangeChanged(const shared_ptr<XValueNodeBase> &){
}
void
XITC503::onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &){
  interface()->send("H" + currentChannel()->to_str());
}
void
XITC503::onExcitationChanged(const shared_ptr<XValueNodeBase> &)
{
}

XAVS47IB::XAVS47IB(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
  XCharDeviceDriver<XTempControl>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  const char *channels_create[] = {"0", "1", "2", "3", "4", "5", "6", "7", 0L};
  const char *excitations_create[] = {"0", "3uV", "10uV", "30uV", "100uV", "300uV", "1mV", "3mV", 0L};
  createChannels(scalarentries, thermometers, false, channels_create, excitations_create);
  
  heaterMode()->add("PID");
  powerRange()->add("0");
  powerRange()->add("1uW");
  powerRange()->add("10uW");
  powerRange()->add("100uW");
  powerRange()->add("1mW");
  powerRange()->add("10mW");
  powerRange()->add("100mW");
  powerRange()->add("1W");

  //    UseSerialPollOnWrite = false;
  //    UseSerialPollOnRead = false;
  interface()->setGPIBWaitBeforeWrite(10); //10msec
  interface()->setGPIBWaitBeforeRead(10); //10msec
}
double
XAVS47IB::read(const char *str)
{
  double x = 0;
  interface()->queryf("%s?", str);
  char buf[4];
  if(interface()->scanf("%3s %lf", buf, &x) != 2)
    throw XInterface::XConvError(__FILE__, __LINE__);
  if(strncmp(buf, str, 3))
    throw XInterface::XConvError(__FILE__, __LINE__);
  return x;
}
void
XAVS47IB::onPChanged(const shared_ptr<XValueNodeBase> &)
{
  int ip = lrint(*prop());
  if(ip > 60) ip = 60;
  if(ip < 5) ip = 5;
  ip = lrint(ip / 5.0 - 1.0);
  interface()->sendf("PRO %u", ip);
}
void
XAVS47IB::onIChanged(const shared_ptr<XValueNodeBase> &)
{
  int ii = lrint(*interval());
  if(ii > 4000) ii = 4000;
  ii = (ii < 2) ? 0 : lrint(log10((double)ii) * 3.0);
  interface()->sendf("ITC %u", ii);
}
void
XAVS47IB::onDChanged(const shared_ptr<XValueNodeBase> &)
{
  int id = lrint(*deriv());
  id = (id < 1) ? 0 : lrint(log10((double)id) * 3.0) + 1;
  interface()->sendf("DTC %u", id);
}
void
XAVS47IB::onTargetTempChanged(const shared_ptr<XValueNodeBase> &) {
  setPoint();
}
void
XAVS47IB::onManualPowerChanged(const shared_ptr<XValueNodeBase> &) {}
void
XAVS47IB::onHeaterModeChanged(const shared_ptr<XValueNodeBase> &) {}
void
XAVS47IB::onPowerRangeChanged(const shared_ptr<XValueNodeBase> &) {
  setPowerRange(*powerRange());
}
void
XAVS47IB::onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &) {
  shared_ptr<XChannel> ch = *currentChannel();
  if(!ch) return;
  { XScopedLock<XInterface> lock(*interface());
      interface()->send("ARN 0;INP 0;ARN 0;RAN 7");
      interface()->sendf("DIS 0;MUX %u;ARN 0", QString(currentChannel()->to_str()).toInt());
      if(*ch->excitation() >= 1)
        interface()->sendf("EXC %u", (unsigned int)(*ch->excitation()));
      msecsleep(1500);
      interface()->send("ARN 0;INP 1;ARN 0;RAN 6");
      m_autorange_wait = 0;
  }
}
void
XAVS47IB::onExcitationChanged(const shared_ptr<XValueNodeBase> &node) {
	if(!interface()->isOpened()) return;
      shared_ptr<XComboNode> excitation = dynamic_pointer_cast<XComboNode>(node);
      interface()->sendf("EXC %u", (unsigned int)(*excitation));
      m_autorange_wait = 0;
}

double
XAVS47IB::getRaw(shared_ptr<XChannel> &)
{
  return getRes();
}
void
XAVS47IB::open() throw (XInterface::XInterfaceError &)
{
  msecsleep(200);
  interface()->send("REM 1;ARN 0;DIS 0");
  currentChannel()->str(formatString("%d",(int)lrint(read("MUX"))));
  
  start();

  manualPower()->setUIEnabled(false);
}
void
XAVS47IB::afterStop()
{
	try {
	  interface()->send("REM 0"); //LOCAL
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
	}
  
  close();
}

int
XAVS47IB::setRange(unsigned int range)
{
  int rangebuf = (int)*powerRange();
  interface()->send("POW 0");
  if(range > 7) range = 7;
  interface()->queryf("ARN 0;RAN %u;*OPC?", range);
  setPoint();
  interface()->sendf("POW %u", rangebuf);
  m_autorange_wait = 0;
  return 0;
}

double
XAVS47IB::getRes()
{
  double x;
  { XScopedLock<XInterface> lock(*interface());
      int wait = interface()->gpibWaitBeforeRead();
      interface()->setGPIBWaitBeforeRead(300);
      interface()->query("AVE 1;*OPC?");
      interface()->setGPIBWaitBeforeRead(wait);
      x = read("AVE");
  }      
  if(m_autorange_wait++ > 10)
  {
     int range = getRange();
     if(lrint(read("OVL")) == 0)
    	{
    	  if(fabs(x) < 0.1 * pow(10.0, range - 1))
    	    setRange(std::max(range - 1, 1));
    	  if(fabs(x) > 1.6 * pow(10.0, range - 1))
    	    setRange(std::min(range + 1, 7));
    	}
    else
    	{
    	   setRange(std::min(range + 1, 7));
    	}  
  }
  return x;
}
int
XAVS47IB::getRange()
{
  return lrint(read("RAN"));
}
int
XAVS47IB::setPoint()
{
  shared_ptr<XChannel> ch = *currentChannel();
  if(!ch) return -1;
  shared_ptr<XThermometer> thermo = *ch->thermometer();
  if(!thermo) return -1;
  double res = thermo->getRawValue(*targetTemp());
  //the unit is 100uV
  int val = lrint(10000.0 * res / pow(10.0, getRange() - 1));
  val = std::min(val, 20000);
  interface()->sendf("SPT %d", val);
  return 0;
}

int
XAVS47IB::setBias(unsigned int bias)
{
  interface()->sendf("BIA %u", bias);
  return 0;
}
void
XAVS47IB::setPowerRange(int range)
{
  interface()->sendf("POW %u", range);
}
double
XAVS47IB::getHeater()
{
  return read("HTP");
}

XCryocon::XCryocon(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
 XCharDeviceDriver<XTempControl>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  heaterMode()->add("OFF");
  heaterMode()->add("PID");
  heaterMode()->add("MAN");
  interface()->setEOS("");
  interface()->setGPIBUseSerialPollOnWrite(false);
  interface()->setGPIBUseSerialPollOnRead (false);
  interface()->setGPIBWaitBeforeWrite(20);
  //    ExclusiveWaitAfterWrite = 10;
  interface()->setGPIBWaitBeforeRead(20);
}
XCryoconM62::XCryoconM62(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
 XCryocon(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  const char *channels_create[] = {"A", "B", 0L};
  const char *excitations_create[] = {"10UV", "30UV", "100UV", "333UV", "1.0MV", "3.3MV", 0L};
  createChannels(scalarentries, thermometers, true, channels_create, excitations_create);
}
XCryoconM32::XCryoconM32(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
 XCryocon(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  const char *channels_create[] = {"A", "B", 0L};
  const char *excitations_create[] = {"CI", "10MV", "3MV", "1MV", 0L};
  createChannels(scalarentries, thermometers, true, channels_create, excitations_create);
  powerRange()->add("HI");
  powerRange()->add("MID");
  powerRange()->add("LOW");
}
void
XCryocon::open() throw (XInterface::XInterfaceError &)
{
  getChannel();
  interface()->query("HEATER:RANGE?");
  powerRange()->str(QString(&interface()->buffer()[0]).stripWhiteSpace());
  interface()->query("HEATER:PMAN?");
  manualPower()->str(std::string(&interface()->buffer()[0]));
  interface()->query("HEATER:TYPE?");
  QString s(&interface()->buffer()[0]);
  heaterMode()->str(s.stripWhiteSpace());
  interface()->query("INPUT A:VBIAS?");

  atomic_shared_ptr<const XNode::NodeList> list(channels()->children());
  shared_ptr<XChannel> ch0 = dynamic_pointer_cast<XChannel>(list->at(0));
  shared_ptr<XChannel> ch1 = dynamic_pointer_cast<XChannel>(list->at(1));

  ch0->excitation()->str(QString(&interface()->buffer()[0]).stripWhiteSpace());
  interface()->query("INPUT B:VBIAS?");
  ch1->excitation()->str(QString(&interface()->buffer()[0]).stripWhiteSpace());
  interface()->query("HEATER:PGAIN?");
  prop()->str(std::string(&interface()->buffer()[0]));
  interface()->query("HEATER:IGAIN?");
  interval()->str(std::string(&interface()->buffer()[0]));
  interface()->query("HEATER:DGAIN?");
  deriv()->str(std::string(&interface()->buffer()[0]));
  
  start();
}
void
XCryoconM62::open() throw (XInterface::XInterfaceError &)
{
  powerRange()->clear();
  interface()->query("HEATER:LOAD?");
  if(interface()->toInt() == 50)
    {
      powerRange()->add("0.05W");
      powerRange()->add("0.5W");
      powerRange()->add("5.0W");
      powerRange()->add("50W");
    }
  else
    {
      powerRange()->add("0.03W");
      powerRange()->add("0.3W");
      powerRange()->add("2.5W");
      powerRange()->add("25W");
    }
  XCryocon::open();
}
void
XCryocon::onPChanged(const shared_ptr<XValueNodeBase> &) {
  interface()->send("HEATER:PGAIN " + prop()->to_str());
}
void
XCryocon::onIChanged(const shared_ptr<XValueNodeBase> &) {
  interface()->send("HEATER:IGAIN " + interval()->to_str());
}
void
XCryocon::onDChanged(const shared_ptr<XValueNodeBase> &) {
  interface()->send("HEATER:DGAIN " + deriv()->to_str());
}
void
XCryocon::onTargetTempChanged(const shared_ptr<XValueNodeBase> &) {
    setTemp(*targetTemp());
}
void
XCryocon::onManualPowerChanged(const shared_ptr<XValueNodeBase> &) {
  interface()->sendf("HEATER:PMAN %f", (double)*manualPower());
}
void
XCryocon::onHeaterModeChanged(const shared_ptr<XValueNodeBase> &) {
  setHeaterMode();
}
void
XCryocon::onPowerRangeChanged(const shared_ptr<XValueNodeBase> &) {
  interface()->send("HEATER:RANGE " + powerRange()->to_str());
}
void
XCryocon::onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &) {
  interface()->send("HEATER:SOURCE " + currentChannel()->to_str());
}
void
XCryocon::onExcitationChanged(const shared_ptr<XValueNodeBase> &node) {
	if(!interface()->isOpened()) return;
      shared_ptr<XChannel> ch;
      atomic_shared_ptr<const XNode::NodeList> list(channels()->children());
      if(list) { 
          for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
              shared_ptr<XChannel> _ch = dynamic_pointer_cast<XChannel>(*it);
             if(_ch->excitation() == node)
                ch = ch;
          }
      }
      interface()->send("INPUT " + ch->getName() +
       ":VBIAS " + node->to_str());
}
void
XCryocon::setTemp(double temp)
{
  if(temp > 0)
    control();
  else
    stopControl();

  shared_ptr<XThermometer> thermo = *(shared_ptr<XChannel>(*currentChannel()))->thermometer();
  if(thermo)
    setHeaterSetPoint(thermo->getRawValue(temp));
  else
    setHeaterSetPoint(temp);
}
double
XCryocon::getRaw(shared_ptr<XChannel> &channel)
{
  double x;
  x = getInput(channel);
  return x;
}
void
XCryocon::getChannel()
{
  interface()->query("HEATER:SOURCE?");
  char s[3];
  if(interface()->scanf("CH%s", s) != 1) return;
  currentChannel()->str(std::string(s));
}
void
XCryocon::setHeaterMode(void)
{
  if(heaterMode()->to_str() == "Off")
    stopControl();
  else
    control();

  interface()->send("HEATER:TYPE " + heaterMode()->to_str());
}
double
XCryocon::getHeater(void)
{
  interface()->query("HEATER:OUTP?");
  return interface()->toDouble();
}

int
XCryocon::control()
{
  interface()->send("CONTROL");
  return 0;
}
int
XCryocon::stopControl()
{
  interface()->send("STOP");
  return 0;
}
double
XCryocon::getInput(shared_ptr<XChannel> &channel)
{
  interface()->query("INPUT? " + channel->getName());
  double x;
  if(interface()->scanf("%lf", &x) != 1) x = 0.0;
  return x;
}

int
XCryocon::setHeaterSetPoint(double value)
{
  interface()->sendf("HEATER:SETPT %f", value);
  return 0;
}

XLakeShore340::XLakeShore340(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
 : XCharDeviceDriver<XTempControl>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  const char *channels_create[] = {"A", "B", 0L};
  const char *excitations_create[] = {0L};
  createChannels(scalarentries, thermometers, true, channels_create, excitations_create);
  heaterMode()->add("Off");
  heaterMode()->add("PID");
  heaterMode()->add("Man");
  interface()->setEOS("");
}

double
XLakeShore340::getRaw(shared_ptr<XChannel> &channel)
{
  shared_ptr<XThermometer> thermo = *(shared_ptr<XChannel>(*currentChannel()))->thermometer();
  if(thermo)
     interface()->query("SRDG? " + channel->getName());
  else
     interface()->query("KRDG? " + channel->getName());
  return interface()->toDouble();
}
double
XLakeShore340::getHeater()
{
  interface()->query("HTR?");
  return interface()->toDouble();
}
void
XLakeShore340::onPChanged(const shared_ptr<XValueNodeBase> &)
{
  interface()->sendf("PID 1,%f", (double)*prop());
}
void
XLakeShore340::onIChanged(const shared_ptr<XValueNodeBase> &)
{
  interface()->sendf("PID 1,,%f", (double)*interval());
}
void
XLakeShore340::onDChanged(const shared_ptr<XValueNodeBase> &)
{
  interface()->sendf("PID 1,,,%f", (double)*deriv());
}
void
XLakeShore340::onTargetTempChanged(const shared_ptr<XValueNodeBase> &)
{
  double temp = *targetTemp();
  shared_ptr<XThermometer> thermo = *(shared_ptr<XChannel>(*currentChannel()))->thermometer();
  if(thermo)
    {
      interface()->sendf("CSET 1,%s,3,1", (const char*)currentChannel()->to_str().c_str());
      temp = thermo->getRawValue(temp);
    }
  else
    {
      interface()->sendf("CSET 1,%s,1,1", (const char*)currentChannel()->to_str().c_str());
    }
  interface()->sendf("SETP 1,%f", temp);
}
void
XLakeShore340::onManualPowerChanged(const shared_ptr<XValueNodeBase> &)
{
  interface()->sendf("MOUT 1,%f", (double)*manualPower());
}
void
XLakeShore340::onHeaterModeChanged(const shared_ptr<XValueNodeBase> &)
{
  if(heaterMode()->to_str() == "Off")
    {
      interface()->send("RANGE 0");
    }
  if(heaterMode()->to_str() == "PID")
    {
      interface()->send("CMODE 1");
    }
  if(heaterMode()->to_str() == "Man")
    {
      interface()->send("CMODE 3");
    }
}
void
XLakeShore340::onPowerRangeChanged(const shared_ptr<XValueNodeBase> &)
{
  interface()->sendf("RANGE %d", *powerRange() + 1);
}
void
XLakeShore340::onCurrentChannelChanged(const shared_ptr<XValueNodeBase> &)
{
  interface()->sendf("CSET 1,%s", (const char *)currentChannel()->to_str().c_str());
}
void
XLakeShore340::onExcitationChanged(const shared_ptr<XValueNodeBase> &)
{
}
void
XLakeShore340::open() throw (XInterface::XInterfaceError &)
{
  interface()->query("CSET?");
  currentChannel()->str(std::string(&interface()->buffer()[0]));
  interface()->query("CDISP? 1");
  int res, maxcurr;
  if(interface()->scanf("%*d,%d", &res) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
  interface()->query("CLIMIT? 1");
  if(interface()->scanf("%*f,%*f,%*f,%d", &maxcurr) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);

  for(int i = 1; i < 6; i++)
    {
      powerRange()->add(QString().sprintf("%.1f W", 
        (double)pow(10.0, i - 5.0)  * pow(maxcurr, 2.0) * res));
    }
  interface()->query("CMODE? 1");
  switch(interface()->toInt()) {
  case 1:
    heaterMode()->str(std::string("PID"));
    break;
  case 3:
    heaterMode()->str(std::string("Man"));
    break;
  default:
    break;
  }
  interface()->query("RANGE?");
  int range = interface()->toInt();
  if(range == 0)
    heaterMode()->str(std::string("Off"));
  else
    powerRange()->value(range - 1);

  interface()->query("MOUT?");
  manualPower()->value(interface()->toDouble());
  interface()->query("PID? 1");
  double p, i, d;
  if(interface()->scanf("%lf,%lf,%lf", &p, &i, &d) != 3)
        throw XInterface::XConvError(__FILE__, __LINE__);
  prop()->value(p);
  interval()->value(i);
  deriv()->value(d);
  
  start();
}

