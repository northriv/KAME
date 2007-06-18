/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "usertempcontrol.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, CryoconM32, "Cryocon M32 temp. controller");
REGISTER_TYPE(XDriverList, CryoconM62, "Cryocon M62 temp. controller");
REGISTER_TYPE(XDriverList, LakeShore340, "LakeShore 340 temp. controller");
REGISTER_TYPE(XDriverList, AVS47IB, "Picowatt AVS-47 bridge");
REGISTER_TYPE(XDriverList, ITC503, "Oxford ITC-503 temp. controller");
REGISTER_TYPE(XDriverList, NeoceraLTC21, "Neocera LT-21 temp. controller");

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
}
void
XITC503::open() throw (XInterface::XInterfaceError &)
{
	start();

	if(!shared_ptr<XDCSource>(*extDCSource())) {
	  	heaterMode()->clear();
		heaterMode()->add("PID");
		heaterMode()->add("Man");
	}
	powerRange()->setUIEnabled(false);
}
double
XITC503::getRaw(shared_ptr<XChannel> &channel)
{
	interface()->send("X");
	return read(QString(channel->getName()).toInt());
}
double
XITC503::getHeater()
{
	return read(5);
}
void
XITC503::onPChanged(double p)
{
	interface()->sendf("P%f", p);
}
void
XITC503::onIChanged(double i){
	interface()->sendf("I%f", i);
}
void
XITC503::onDChanged(double d){
	interface()->sendf("D%f", d);
}
void
XITC503::onTargetTempChanged(double temp){
	if(heaterMode()->to_str() == "PID")
		interface()->sendf("T%f", temp);
}
void
XITC503::onManualPowerChanged(double pow){
	if(heaterMode()->to_str() == "Man")
		interface()->sendf("O%f", pow);
}
void
XITC503::onHeaterModeChanged(int){
}
void
XITC503::onPowerRangeChanged(int){
}
void
XITC503::onCurrentChannelChanged(const shared_ptr<XChannel> &ch){
	interface()->send("H" + ch->getName());
}
void
XITC503::onExcitationChanged(const shared_ptr<XChannel> &, int)
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
  
	//    UseSerialPollOnWrite = false;
	//    UseSerialPollOnRead = false;
	interface()->setGPIBWaitBeforeWrite(10); //10msec
	interface()->setGPIBWaitBeforeRead(10); //10msec

//	manualPower()->disable();
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
XAVS47IB::onPChanged(double p)
{
	int ip = lrint(p);
	if(ip > 60) ip = 60;
	if(ip < 5) ip = 5;
	ip = lrint(ip / 5.0 - 1.0);
	interface()->sendf("PRO %u", ip);
}
void
XAVS47IB::onIChanged(double i)
{
	int ii = lrint(i);
	if(ii > 4000) ii = 4000;
	ii = (ii < 2) ? 0 : lrint(log10((double)ii) * 3.0);
	interface()->sendf("ITC %u", ii);
}
void
XAVS47IB::onDChanged(double d)
{
	int id = lrint(d);
	id = (id < 1) ? 0 : lrint(log10((double)id) * 3.0) + 1;
	interface()->sendf("DTC %u", id);
}
void
XAVS47IB::onTargetTempChanged(double) {
	setPoint();
}
void
XAVS47IB::onManualPowerChanged(double) {}
void
XAVS47IB::onHeaterModeChanged(int) {}
void
XAVS47IB::onPowerRangeChanged(int ran) {
	setPowerRange(ran);
}
void
XAVS47IB::onCurrentChannelChanged(const shared_ptr<XChannel> &ch) {
	interface()->send("ARN 0;INP 0;ARN 0;RAN 7");
	interface()->sendf("DIS 0;MUX %u;ARN 0", QString(currentChannel()->to_str()).toInt());
	if(*ch->excitation() >= 1)
		interface()->sendf("EXC %u", (unsigned int)(*ch->excitation()));
	msecsleep(1500);
	interface()->send("ARN 0;INP 1;ARN 0;RAN 6");
	m_autorange_wait = 0;
}
void
XAVS47IB::onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) {
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
	if(ch != shared_ptr<XChannel>(*currentChannel())) return;
	interface()->sendf("EXC %u", (unsigned int)exc);
	m_autorange_wait = 0;

	powerRange()->add("0");
	powerRange()->add("1uW");
	powerRange()->add("10uW");
	powerRange()->add("100uW");
	powerRange()->add("1mW");
	powerRange()->add("10mW");
	powerRange()->add("100mW");
	powerRange()->add("1W");
}

double
XAVS47IB::getRaw(shared_ptr<XChannel> &)
{
	return getRes();
}
void
XAVS47IB::open() throw (XInterface::XInterfaceError &)
{
	msecsleep(50);
	interface()->send("REM 1;ARN 0;DIS 0");
	currentChannel()->str(formatString("%d", (int)lrint(read("MUX"))));
	onCurrentChannelChanged(*currentChannel());

	start();

	if(!shared_ptr<XDCSource>(*extDCSource())) {
	  	heaterMode()->clear();
		heaterMode()->add("PID");
	}
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
}
void
XCryocon::open() throw (XInterface::XInterfaceError &)
{
	atomic_shared_ptr<const XNode::NodeList> list(channels()->children());
	shared_ptr<XChannel> ch0 = dynamic_pointer_cast<XChannel>(list->at(0));
	shared_ptr<XChannel> ch1 = dynamic_pointer_cast<XChannel>(list->at(1));
	interface()->query("INPUT A:VBIAS?");
	ch0->excitation()->str(QString(&interface()->buffer()[0]).stripWhiteSpace());
	interface()->query("INPUT B:VBIAS?");
	ch1->excitation()->str(QString(&interface()->buffer()[0]).stripWhiteSpace());

	powerRange()->clear();
	interface()->query("HEATER:RANGE?");
	powerRange()->str(QString(&interface()->buffer()[0]).stripWhiteSpace());

	if(!shared_ptr<XDCSource>(*extDCSource())) {
		getChannel();
		interface()->query("HEATER:PMAN?");
		manualPower()->str(std::string(&interface()->buffer()[0]));
		interface()->query("HEATER:PGAIN?");
		prop()->str(std::string(&interface()->buffer()[0]));
		interface()->query("HEATER:IGAIN?");
		interval()->str(std::string(&interface()->buffer()[0]));
		interface()->query("HEATER:DGAIN?");
		deriv()->str(std::string(&interface()->buffer()[0]));

		if(!shared_ptr<XDCSource>(*extDCSource())) {
		  	heaterMode()->clear();
			heaterMode()->add("OFF");
			heaterMode()->add("PID");
			heaterMode()->add("MAN");
		}
		interface()->query("HEATER:TYPE?");
		QString s(&interface()->buffer()[0]);
		heaterMode()->str(s.stripWhiteSpace());
	}

	start();
}
void
XCryoconM32::open() throw (XInterface::XInterfaceError &)
{
	XCryocon::open();

	powerRange()->add("HI");
	powerRange()->add("MID");
	powerRange()->add("LOW");
}
void
XCryoconM62::open() throw (XInterface::XInterfaceError &)
{
	XCryocon::open();

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
}
void
XCryocon::onPChanged(double p) {
	interface()->sendf("HEATER:PGAIN %f", p);
}
void
XCryocon::onIChanged(double i) {
	interface()->sendf("HEATER:IGAIN %f", i);
}
void
XCryocon::onDChanged(double d) {
	interface()->sendf("HEATER:DGAIN %f", d);
}
void
XCryocon::onTargetTempChanged(double temp) {
    setTemp(temp);
}
void
XCryocon::onManualPowerChanged(double pow) {
	interface()->sendf("HEATER:PMAN %f", pow);
}
void
XCryocon::onHeaterModeChanged(int) {
	setHeaterMode();
}
void
XCryocon::onPowerRangeChanged(int) {
	interface()->send("HEATER:RANGE " + powerRange()->to_str());
}
void
XCryocon::onCurrentChannelChanged(const shared_ptr<XChannel> &ch) {
	interface()->send("HEATER:SOURCE " + ch->getName());
}
void
XCryocon::onExcitationChanged(const shared_ptr<XChannel> &ch, int) {
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
	interface()->send("INPUT " + ch->getName() +
					  ":VBIAS " + ch->excitation()->to_str());
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

XNeoceraLTC21::XNeoceraLTC21(const char *name, bool runtime,
							 const shared_ptr<XScalarEntryList> &scalarentries,
							 const shared_ptr<XInterfaceList> &interfaces,
							 const shared_ptr<XThermometerList> &thermometers,
							 const shared_ptr<XDriverList> &drivers)
	: XCharDeviceDriver<XTempControl>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
	const char *channels_create[] = {"1", "2", 0L};
	const char *excitations_create[] = {0L};
//	const char *excitations_create[] = {"1mV", "320uV", "100uV", "32uV", "10uV", 0L};
	createChannels(scalarentries, thermometers, true, channels_create, excitations_create);
	interface()->setEOS("");
	powerRange()->add("0");
	powerRange()->add("0.05W");
	powerRange()->add("0.5W");
	powerRange()->add("5W");
	powerRange()->add("50W");
}
void
XNeoceraLTC21::control()
{
	interface()->send("SCONT;");
}
void
XNeoceraLTC21::monitor()
{
	interface()->send("SMON;");
}

double
XNeoceraLTC21::getRaw(shared_ptr<XChannel> &channel)
{
	interface()->query("QSAMP?" + channel->getName() + ";");
	double x;
	if(interface()->scanf("%7lf", &x) != 1)
		return 0.0;
	return x;
}
double
XNeoceraLTC21::getHeater()
{
	interface()->query("QHEAT?;");
	double x;
	if(interface()->scanf("%5lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
void
XNeoceraLTC21::setHeater()
{
	interface()->sendf("SPID1,%f,%f,%f,%f,100.0;",
		 (double)*prop(), (double)*interval(), (double)*deriv(), (double)*manualPower());
}
void
XNeoceraLTC21::onPChanged(double /*p*/)
{
	setHeater();
}
void
XNeoceraLTC21::onIChanged(double /*i*/)
{
	setHeater();
}
void
XNeoceraLTC21::onDChanged(double /*d*/)
{
	setHeater();
}
void
XNeoceraLTC21::onTargetTempChanged(double temp)
{
	interface()->sendf("SETP1,%.5f;", temp);
}
void
XNeoceraLTC21::onManualPowerChanged(double /*pow*/)
{
	setHeater();
}
void
XNeoceraLTC21::onHeaterModeChanged(int x)
{
	if(x < 6) {
		interface()->sendf("SHCONT%d;", x);
		control();
	}
	else
		monitor();
}
void
XNeoceraLTC21::onPowerRangeChanged(int ran)
{
	interface()->sendf("SHMXPWR%d;", ran);
}
void
XNeoceraLTC21::onCurrentChannelChanged(const shared_ptr<XChannel> &cch)
{
	int ch = atoi(cch->getName().c_str());
	if(ch < 1) ch = 3;
	interface()->sendf("SOSEN1,%d;", ch);
}
void
XNeoceraLTC21::onExcitationChanged(const shared_ptr<XChannel> &, int)
{
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
}
void
XNeoceraLTC21::open() throw (XInterface::XInterfaceError &)
{
	if(!shared_ptr<XDCSource>(*extDCSource())) {
		interface()->query("QOUT?1;");
		int sens, cmode, range;
		if(interface()->scanf("%1d;%1d;%1d;", &sens, &cmode, &range) != 3)
			throw XInterface::XConvError(__FILE__, __LINE__);
		currentChannel()->str(formatString("%d", sens));
		
		heaterMode()->clear();
		heaterMode()->add("AUTO P");
		heaterMode()->add("AUTO PI");
		heaterMode()->add("AUTO PID");
		heaterMode()->add("PID");
		heaterMode()->add("TABLE");
		heaterMode()->add("DEFAULT");
		heaterMode()->add("MONITOR");

		heaterMode()->value(cmode);
		powerRange()->value(range);
	
		interface()->query("QPID?1;");
		double p, i, d, power, limit;
		if(interface()->scanf("%lf;%lf;%lf;%lf;%lf;", &p, &i, &d, &power, &limit) != 5)
	        throw XInterface::XConvError(__FILE__, __LINE__);
		prop()->value(p);
		interval()->value(i);
		deriv()->value(d);
		manualPower()->value(power);
	}
	start();
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
	interface()->setEOS("");
	interface()->setGPIBUseSerialPollOnWrite(false);
	interface()->setGPIBUseSerialPollOnRead (false);
	interface()->setGPIBWaitBeforeWrite(20);
	//    ExclusiveWaitAfterWrite = 10;
	interface()->setGPIBWaitBeforeRead(20);	
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
XLakeShore340::onPChanged(double p)
{
	interface()->sendf("PID 1,%f", p);
}
void
XLakeShore340::onIChanged(double i)
{
	interface()->sendf("PID 1,,%f", i);
}
void
XLakeShore340::onDChanged(double d)
{
	interface()->sendf("PID 1,,,%f", d);
}
void
XLakeShore340::onTargetTempChanged(double temp)
{
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
XLakeShore340::onManualPowerChanged(double pow)
{
	interface()->sendf("MOUT 1,%f", pow);
}
void
XLakeShore340::onHeaterModeChanged(int)
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
XLakeShore340::onPowerRangeChanged(int ran)
{
	interface()->sendf("RANGE %d", ran + 1);
}
void
XLakeShore340::onCurrentChannelChanged(const shared_ptr<XChannel> &ch)
{
	interface()->sendf("CSET 1,%s", (const char *)ch->getName().c_str());
}
void
XLakeShore340::onExcitationChanged(const shared_ptr<XChannel> &, int)
{
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
}
void
XLakeShore340::open() throw (XInterface::XInterfaceError &)
{
	interface()->query("CDISP? 1");
	int res, maxcurr;
	if(interface()->scanf("%*d,%d", &res) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
	interface()->query("CLIMIT? 1");
	if(interface()->scanf("%*f,%*f,%*f,%d", &maxcurr) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);

	powerRange()->clear();
	for(int i = 1; i < 6; i++) {
		powerRange()->add(QString().sprintf("%.1f W", 
											(double)pow(10.0, i - 5.0)  * pow(maxcurr, 2.0) * res));
	}
	if(!shared_ptr<XDCSource>(*extDCSource())) {
		interface()->query("CSET?");
		int ch;
		if(interface()->scanf("%d", &ch) != 1)
			currentChannel()->str(formatString("%d", ch));
		
		heaterMode()->clear();
		heaterMode()->add("Off");
		heaterMode()->add("PID");
		heaterMode()->add("Man");

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
	}
	start();
}

