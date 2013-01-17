/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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
REGISTER_TYPE(XDriverList, LakeShore370, "LakeShore 370 temp. controller");
REGISTER_TYPE(XDriverList, AVS47IB, "Picowatt AVS-47 bridge");
REGISTER_TYPE(XDriverList, ITC503, "Oxford ITC-503 temp. controller");
REGISTER_TYPE(XDriverList, NeoceraLTC21, "Neocera LTC-21 temp. controller");
REGISTER_TYPE(XDriverList, KE2700w7700, "Keithley 2700&7700 as temp. controller");

XITC503::XITC503(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XOxfordDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "1", "2", "3", 0L };
	const char *excitations_create[] = { 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create);
}
void XITC503::open() throw (XKameError &) {
	start();

	for(Transaction tr( *this);; ++tr) {
		const Snapshot &shot(tr);
		if( !shared_ptr<XDCSource>(shot[ *extDCSource()])) {
			tr[ *heaterMode()].clear();
			tr[ *heaterMode()].add("PID");
			tr[ *heaterMode()].add("Man");
		}
		tr[ *powerRange()].setUIEnabled(false);
		if(tr.commit())
			break;
	}
}
double XITC503::getRaw(shared_ptr<XChannel> &channel) {
	interface()->send("X");
	return read(QString(channel->getName()).toInt());
}
double XITC503::getTemp(shared_ptr<XChannel> &channel) {
	interface()->send("X");
	return read(QString(channel->getName()).toInt());
}
double XITC503::getHeater() {
	return read(5);
}
void XITC503::onPChanged(double p) {
	interface()->sendf("P%f", p);
}
void XITC503::onIChanged(double i) {
	interface()->sendf("I%f", i);
}
void XITC503::onDChanged(double d) {
	interface()->sendf("D%f", d);
}
void XITC503::onTargetTempChanged(double temp) {
	if(( **heaterMode())->to_str() == "PID")
		interface()->sendf("T%f", temp);
}
void XITC503::onManualPowerChanged(double pow) {
	if(( **heaterMode())->to_str() == "Man")
		interface()->sendf("O%f", pow);
}
void XITC503::onHeaterModeChanged(int) {
}
void XITC503::onPowerRangeChanged(int) {
}
void XITC503::onCurrentChannelChanged(const shared_ptr<XChannel> &ch) {
	interface()->send("H" + ch->getName());
}
void XITC503::onExcitationChanged(const shared_ptr<XChannel> &, int) {
}

XAVS47IB::XAVS47IB(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "0", "1", "2", "3", "4", "5", "6", "7",
		0L };
	const char *excitations_create[] = { "0", "3uV", "10uV", "30uV", "100uV",
		"300uV", "1mV", "3mV", 0L };
	createChannels(ref(tr_meas), meas, false, channels_create,
		excitations_create);

	//    UseSerialPollOnWrite = false;
	//    UseSerialPollOnRead = false;
	interface()->setGPIBWaitBeforeWrite(10); //10msec
	interface()->setGPIBWaitBeforeRead(10); //10msec

	//	manualPower()->disable();
}
double XAVS47IB::read(const char *str) {
	double x = 0;
	interface()->queryf("%s?", str);
	char buf[4];
	if(interface()->scanf("%3s %lf", buf, &x) != 2)
		throw XInterface::XConvError(__FILE__, __LINE__);
	if(strncmp(buf, str, 3))
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
void XAVS47IB::onPChanged(double p) {
	int ip = lrint(p);
	if(ip > 60)
		ip = 60;
	if(ip < 5)
		ip = 5;
	ip = lrint(ip / 5.0 - 1.0);
	interface()->sendf("PRO %u", ip);
}
void XAVS47IB::onIChanged(double i) {
	int ii = lrint(i);
	if(ii > 4000)
		ii = 4000;
	ii = (ii < 2) ? 0 : lrint(log10((double) ii) * 3.0);
	interface()->sendf("ITC %u", ii);
}
void XAVS47IB::onDChanged(double d) {
	int id = lrint(d);
	id = (id < 1) ? 0 : lrint(log10((double) id) * 3.0) + 1;
	interface()->sendf("DTC %u", id);
}
void XAVS47IB::onTargetTempChanged(double) {
	setPoint();
}
void XAVS47IB::onManualPowerChanged(double) {
}
void XAVS47IB::onHeaterModeChanged(int) {
}
void XAVS47IB::onPowerRangeChanged(int ran) {
	setPowerRange(ran);
}
void XAVS47IB::onCurrentChannelChanged(const shared_ptr<XChannel> &ch) {
	Snapshot shot( *this);
	interface()->send("ARN 0;INP 0;ARN 0;RAN 7");
	interface()->sendf("DIS 0;MUX %u;ARN 0",
		QString(shot[ *currentChannel()].to_str()).toInt());
	if(shot[ *ch->excitation()] >= 1)
		interface()->sendf("EXC %u", (unsigned int) (shot[ *ch->excitation()]));
	msecsleep(1500);
	interface()->send("ARN 0;INP 1;ARN 0;RAN 6");
	m_autorange_wait = 0;
}
void XAVS47IB::onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
	Snapshot shot( *this);
	if(ch != shared_ptr<XChannel>(shot[ *currentChannel()]))
		return;
	interface()->sendf("EXC %u", (unsigned int) exc);
	m_autorange_wait = 0;

	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange()].add("0");
		tr[ *powerRange()].add("1uW");
		tr[ *powerRange()].add("10uW");
		tr[ *powerRange()].add("100uW");
		tr[ *powerRange()].add("1mW");
		tr[ *powerRange()].add("10mW");
		tr[ *powerRange()].add("100mW");
		tr[ *powerRange()].add("1W");
		if(tr.commit())
			break;
	}
}
double XAVS47IB::getRaw(shared_ptr<XChannel> &) {
	return getRes();
}
double XAVS47IB::getTemp(shared_ptr<XChannel> &) {
	return getRes();
}
void XAVS47IB::open() throw (XKameError &) {
	msecsleep(50);
	interface()->send("REM 1;ARN 0;DIS 0");
	trans( *currentChannel()).str(formatString("%d", (int) lrint(read("MUX"))));
	onCurrentChannelChanged( ***currentChannel());

	start();

	for(Transaction tr( *this);; ++tr) {
		const Snapshot &shot(tr);
		if( !shared_ptr<XDCSource>(shot[ *extDCSource()])) {
			tr[ *heaterMode()].clear();
			tr[ *heaterMode()].add("PID");
		}
		if(tr.commit())
			break;
	}
}
void XAVS47IB::closeInterface() {
	try {
		interface()->send("REM 0"); //LOCAL
	}
	catch(XInterface::XInterfaceError &e) {
		e.print(getLabel());
	}

	close();
}

int XAVS47IB::setRange(unsigned int range) {
	int rangebuf = ***powerRange();
	interface()->send("POW 0");
	if(range > 7)
		range = 7;
	interface()->queryf("ARN 0;RAN %u;*OPC?", range);
	setPoint();
	interface()->sendf("POW %u", rangebuf);

	m_autorange_wait = 0;
	return 0;
}

double XAVS47IB::getRes() {
	double x;
	{
		XScopedLock<XInterface> lock( *interface());
		int wait = interface()->gpibWaitBeforeRead();
		interface()->setGPIBWaitBeforeRead(300);
		interface()->query("AVE 1;*OPC?");
		interface()->setGPIBWaitBeforeRead(wait);
		x = read("AVE");
	}
	if(m_autorange_wait++ > 10) {
		int range = getRange();
		if(lrint(read("OVL")) == 0) {
			if(fabs(x) < 0.1 * pow(10.0, range - 1))
				setRange(std::max(range - 1, 1));
			if(fabs(x) > 1.6 * pow(10.0, range - 1))
				setRange(std::min(range + 1, 7));
		}
		else {
			setRange(std::min(range + 1, 7));
		}
	}
	return x;
}
int XAVS47IB::getRange() {
	return lrint(read("RAN"));
}
int XAVS47IB::setPoint() {
	Snapshot shot( *this);
	shared_ptr<XChannel> ch = shot[ *currentChannel()];
	if( !ch)
		return -1;
	shared_ptr<XThermometer> thermo = shot[ *ch->thermometer()];
	if( !thermo)
		return -1;
	double res = thermo->getRawValue(shot[ *targetTemp()]);
	//the unit is 100uV
	int val = lrint(10000.0 * res / pow(10.0, getRange() - 1));
	val = std::min(val, 20000);
	interface()->sendf("SPT %d", val);
	return 0;
}

int XAVS47IB::setBias(unsigned int bias) {
	interface()->sendf("BIA %u", bias);
	return 0;
}
void XAVS47IB::setPowerRange(int range) {
	interface()->sendf("POW %u", range);
}
double XAVS47IB::getHeater() {
	return read("HTP");
}

XCryocon::XCryocon(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
	interface()->setEOS("");
	interface()->setGPIBUseSerialPollOnWrite(false);
	interface()->setGPIBUseSerialPollOnRead(false);
	interface()->setGPIBWaitBeforeWrite(20);
	//    ExclusiveWaitAfterWrite = 10;
	interface()->setGPIBWaitBeforeRead(20);
}
XCryoconM62::XCryoconM62(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCryocon(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "A", "B", 0L };
	const char *excitations_create[] = { "10UV", "30UV", "100UV", "333UV",
		"1.0MV", "3.3MV", 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create);
}
XCryoconM32::XCryoconM32(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCryocon(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "A", "B", 0L };
	const char *excitations_create[] = { "CI", "10MV", "3MV", "1MV", 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create);
}
void XCryocon::open() throw (XKameError &) {
	Snapshot shot( *channels());
	const XNode::NodeList &list( *shot.list());
	shared_ptr<XChannel> ch0 = static_pointer_cast<XChannel>(list.at(0));
	shared_ptr<XChannel> ch1 = static_pointer_cast<XChannel>(list.at(1));
	interface()->query("INPUT A:VBIAS?");
	trans( *ch0->excitation()).str(QString( &interface()->buffer()[0]).simplified());
	interface()->query("INPUT B:VBIAS?");
	trans( *ch1->excitation()).str(QString( &interface()->buffer()[0]).simplified());

	trans( *powerRange()).clear();
	interface()->query("HEATER:RANGE?");
	trans( *powerRange()).str(QString( &interface()->buffer()[0]).simplified());

	if( !shared_ptr<XDCSource>( ***extDCSource())) {
		getChannel();
		interface()->query("HEATER:PMAN?");
		trans( *manualPower()).str(XString( &interface()->buffer()[0]));
		interface()->query("HEATER:PGAIN?");
		trans( *prop()).str(XString( &interface()->buffer()[0]));
		interface()->query("HEATER:IGAIN?");
		trans( *interval()).str(XString( &interface()->buffer()[0]));
		interface()->query("HEATER:DGAIN?");
		trans( *deriv()).str(XString( &interface()->buffer()[0]));

		for(Transaction tr( *this);; ++tr) {
			tr[ *heaterMode()].clear();
			tr[ *heaterMode()].add("OFF");
			tr[ *heaterMode()].add("PID");
			tr[ *heaterMode()].add("MAN");
			if(tr.commit())
				break;
		}
		interface()->query("HEATER:TYPE?");
		QString s( &interface()->buffer()[0]);
		trans( *heaterMode()).str(s.simplified());
	}

	start();
}
void XCryoconM32::open() throw (XKameError &) {
	XCryocon::open();

	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange()].add("HI");
		tr[ *powerRange()].add("MID");
		tr[ *powerRange()].add("LOW");
		if(tr.commit())
			break;
	}
}
void XCryoconM62::open() throw (XKameError &) {
	XCryocon::open();

	interface()->query("HEATER:LOAD?");
	for(Transaction tr( *this);; ++tr) {
		if(interface()->toInt() == 50) {
			tr[ *powerRange()].add("0.05W");
			tr[ *powerRange()].add("0.5W");
			tr[ *powerRange()].add("5.0W");
			tr[ *powerRange()].add("50W");
		}
		else {
			tr[ *powerRange()].add("0.03W");
			tr[ *powerRange()].add("0.3W");
			tr[ *powerRange()].add("2.5W");
			tr[ *powerRange()].add("25W");
		}
		if(tr.commit())
			break;
	}
}
void XCryocon::onPChanged(double p) {
	interface()->sendf("HEATER:PGAIN %f", p);
}
void XCryocon::onIChanged(double i) {
	interface()->sendf("HEATER:IGAIN %f", i);
}
void XCryocon::onDChanged(double d) {
	interface()->sendf("HEATER:DGAIN %f", d);
}
void XCryocon::onTargetTempChanged(double temp) {
	setTemp(temp);
}
void XCryocon::onManualPowerChanged(double pow) {
	interface()->sendf("HEATER:PMAN %f", pow);
}
void XCryocon::onHeaterModeChanged(int) {
	setHeaterMode();
}
void XCryocon::onPowerRangeChanged(int) {
	interface()->send("HEATER:RANGE " + ( **powerRange())->to_str());
}
void XCryocon::onCurrentChannelChanged(const shared_ptr<XChannel> &ch) {
	interface()->send("HEATER:SOURCE " + ch->getName());
}
void XCryocon::onExcitationChanged(const shared_ptr<XChannel> &ch, int) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
	interface()->send("INPUT " + ch->getName() + ":VBIAS "
		+ ( **ch->excitation())->to_str());
}
void XCryocon::setTemp(double temp) {
	if(temp > 0)
		control();
	else
		stopControl();

	Snapshot shot( *this);
	shared_ptr<XThermometer> thermo = shot[ *(shared_ptr<XChannel>(shot[ *currentChannel()])->thermometer())];
	if(thermo)
		setHeaterSetPoint(thermo->getRawValue(temp));
	else
		setHeaterSetPoint(temp);
}
double XCryocon::getRaw(shared_ptr<XChannel> &channel) {
	double x;
	x = getInput(channel);
	return x;
}
double XCryocon::getTemp(shared_ptr<XChannel> &channel) {
	double x;
	x = getInput(channel);
	return x;
}
void XCryocon::getChannel() {
	interface()->query("HEATER:SOURCE?");
	char s[3];
	if(interface()->scanf("CH%s", s) != 1)
		return;
	trans( *currentChannel()).str(XString(s));
}
void XCryocon::setHeaterMode(void) {
	Snapshot shot( *this);
	if(shot[ *heaterMode()].to_str() == "Off")
		stopControl();
	else
		control();

	interface()->send("HEATER:TYPE " + shot[ *heaterMode()].to_str());
}
double XCryocon::getHeater(void) {
	interface()->query("HEATER:OUTP?");
	return interface()->toDouble();
}

int XCryocon::control() {
	interface()->send("CONTROL");
	return 0;
}
int XCryocon::stopControl() {
	interface()->send("STOP");
	return 0;
}
double XCryocon::getInput(shared_ptr<XChannel> &channel) {
	interface()->query("INPUT? " + channel->getName());
	double x;
	if(interface()->scanf("%lf", &x) != 1)
		x = 0.0;
	return x;
}

int XCryocon::setHeaterSetPoint(double value) {
	interface()->sendf("HEATER:SETPT %f", value);
	return 0;
}

XNeoceraLTC21::XNeoceraLTC21(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "1", "2", 0L };
	const char *excitations_create[] = { 0L };
	//	const char *excitations_create[] = {"1mV", "320uV", "100uV", "32uV", "10uV", 0L};
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create);
	interface()->setEOS("");
	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange()].add("0");
		tr[ *powerRange()].add("0.05W");
		tr[ *powerRange()].add("0.5W");
		tr[ *powerRange()].add("5W");
		tr[ *powerRange()].add("50W");
		if(tr.commit())
			break;
	}
}
void XNeoceraLTC21::control() {
	interface()->send("SCONT;");
}
void XNeoceraLTC21::monitor() {
	interface()->send("SMON;");
}

double XNeoceraLTC21::getRaw(shared_ptr<XChannel> &channel) {
	interface()->query("QSAMP?" + channel->getName() + ";");
	double x;
	if(interface()->scanf("%7lf", &x) != 1)
		return 0.0;
	return x;
}
double XNeoceraLTC21::getTemp(shared_ptr<XChannel> &channel) {
	interface()->query("QSAMP?" + channel->getName() + ";");
	double x;
	if(interface()->scanf("%7lf", &x) != 1)
		return 0.0;
	return x;
}
double XNeoceraLTC21::getHeater() {
	interface()->query("QHEAT?;");
	double x;
	if(interface()->scanf("%5lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
void XNeoceraLTC21::setHeater() {
	Snapshot shot( *this);
	interface()->sendf("SPID1,%f,%f,%f,%f,100.0;", (double)shot[ *prop()],
		(double)shot[ *interval()], (double)shot[ *deriv()], (double)shot[ *manualPower()]);
}
void XNeoceraLTC21::onPChanged(double /*p*/) {
	setHeater();
}
void XNeoceraLTC21::onIChanged(double /*i*/) {
	setHeater();
}
void XNeoceraLTC21::onDChanged(double /*d*/) {
	setHeater();
}
void XNeoceraLTC21::onTargetTempChanged(double temp) {
	interface()->sendf("SETP1,%.5f;", temp);
}
void XNeoceraLTC21::onManualPowerChanged(double /*pow*/) {
	setHeater();
}
void XNeoceraLTC21::onHeaterModeChanged(int x) {
	if(x < 6) {
		interface()->sendf("SHCONT%d;", x);
		control();
	}
	else
		monitor();
}
void XNeoceraLTC21::onPowerRangeChanged(int ran) {
	interface()->sendf("SHMXPWR%d;", ran);
}
void XNeoceraLTC21::onCurrentChannelChanged(const shared_ptr<XChannel> &cch) {
	int ch = atoi(cch->getName().c_str());
	if(ch < 1)
		ch = 3;
	interface()->sendf("SOSEN1,%d;", ch);
}
void XNeoceraLTC21::onExcitationChanged(const shared_ptr<XChannel> &, int) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
}
void XNeoceraLTC21::open() throw (XKameError &) {
	if( !shared_ptr<XDCSource>( ***extDCSource())) {
		interface()->query("QOUT?1;");
		int sens, cmode, range;
		if(interface()->scanf("%1d;%1d;%1d;", &sens, &cmode, &range) != 3)
			throw XInterface::XConvError(__FILE__, __LINE__);
		for(Transaction tr( *this);; ++tr) {
			tr[ *currentChannel()].str(formatString("%d", sens));

			tr[ *heaterMode()].clear();
			tr[ *heaterMode()].add("AUTO P");
			tr[ *heaterMode()].add("AUTO PI");
			tr[ *heaterMode()].add("AUTO PID");
			tr[ *heaterMode()].add("PID");
			tr[ *heaterMode()].add("TABLE");
			tr[ *heaterMode()].add("DEFAULT");
			tr[ *heaterMode()].add("MONITOR");

			tr[ *heaterMode()] = cmode;
			tr[ *powerRange()] = range;
			if(tr.commit())
				break;
		}

		interface()->query("QPID?1;");
		double p, i, d, power, limit;
		if(interface()->scanf("%lf;%lf;%lf;%lf;%lf;", &p, &i, &d, &power,
			&limit) != 5)
			throw XInterface::XConvError(__FILE__, __LINE__);
		for(Transaction tr( *this);; ++tr) {
			tr[ *prop()] = p;
			tr[ *interval()] = i;
			tr[ *deriv()] = d;
			tr[ *manualPower()] = power;
			if(tr.commit())
				break;
		}
	}
	start();
}


XLakeShore::XLakeShore(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
	interface()->setEOS("\r\n");
	interface()->setGPIBUseSerialPollOnWrite(false);
	interface()->setGPIBUseSerialPollOnRead(false);
	interface()->setGPIBWaitBeforeWrite(40);
	//    ExclusiveWaitAfterWrite = 10;
	interface()->setGPIBWaitBeforeRead(40);
}

XLakeShore340::XLakeShore340(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XLakeShore (name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "A", "B", 0L };
	const char *excitations_create[] = { 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create);
}

double XLakeShore340::getRaw(shared_ptr<XChannel> &channel) {
	interface()->query("SRDG? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShore340::getTemp(shared_ptr<XChannel> &channel) {
	interface()->query("KRDG? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShore340::getHeater() {
	interface()->query("HTR?");
	return interface()->toDouble();
}
void XLakeShore340::onPChanged(double p) {
	interface()->sendf("PID 1,%f", p);
}
void XLakeShore340::onIChanged(double i) {
	interface()->sendf("PID 1,,%f", i);
}
void XLakeShore340::onDChanged(double d) {
	interface()->sendf("PID 1,,,%f", d);
}
void XLakeShore340::onTargetTempChanged(double temp) {
	Snapshot shot( *this);
	shared_ptr<XThermometer> thermo = shot[ *shared_ptr<XChannel> ( shot[ *currentChannel()])->thermometer()];
	if(thermo) {
		interface()->sendf("CSET 1,%s,3,1",
			(const char*)shot[ *currentChannel()].to_str().c_str());
		temp = thermo->getRawValue(temp);
	}
	else {
		interface()->sendf("CSET 1,%s,1,1",
			(const char*)shot[ *currentChannel()].to_str().c_str());
	}
	interface()->sendf("SETP 1,%f", temp);
}
void XLakeShore340::onManualPowerChanged(double pow) {
	interface()->sendf("MOUT 1,%f", pow);
}
void XLakeShore340::onHeaterModeChanged(int) {
	Snapshot shot( *this);
	if(shot[ *heaterMode()].to_str() == "Off") {
		interface()->send("RANGE 0");
	}
	if(shot[ *heaterMode()].to_str() == "PID") {
		interface()->send("CMODE 1");
	}
	if(shot[ *heaterMode()].to_str() == "Man") {
		interface()->send("CMODE 3");
	}
}
void XLakeShore340::onPowerRangeChanged(int ran) {
	interface()->sendf("RANGE %d", ran + 1);
}
void XLakeShore340::onCurrentChannelChanged(const shared_ptr<XChannel> &ch) {
	interface()->sendf("CSET 1,%s", (const char *) ch->getName().c_str());
}
void XLakeShore340::onExcitationChanged(const shared_ptr<XChannel> &, int) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
}
void XLakeShore340::open() throw (XKameError &) {
	interface()->query("CDISP? 1");
	int res, maxcurr_idx;
	if(interface()->scanf("%*d,%d", &res) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	interface()->query("CLIMIT? 1");
	if(interface()->scanf("%*f,%*f,%*f,%d", &maxcurr_idx) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	double maxcurr = pow(2.0, maxcurr_idx) * 0.125;
	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange()].clear();
		for(int i = 1; i < 6; i++) {
			tr[ *powerRange()].add(formatString("%.2g W", (double) pow(10.0, i - 5.0)
				* pow(maxcurr, 2.0) * res));
		}
		if(tr.commit())
			break;
	}
	if( !shared_ptr<XDCSource>( ***extDCSource())) {
		interface()->query("CSET? 1");
		for(Transaction tr( *this);; ++tr) {
			char ch[2];
			if(interface()->scanf("%1s", ch) == 1)
				tr[ *currentChannel()].str(XString(ch));

			tr[ *heaterMode()].clear();
			tr[ *heaterMode()].add("Off");
			tr[ *heaterMode()].add("PID");
			tr[ *heaterMode()].add("Man");
			if(tr.commit())
				break;
		}

		interface()->query("CMODE? 1");
		switch(interface()->toInt()) {
		case 1:
			trans( *heaterMode()).str(XString("PID"));
			break;
		case 3:
			trans( *heaterMode()).str(XString("Man"));
			break;
		default:
			break;
		}
		interface()->query("RANGE?");
		int range = interface()->toInt();
		if(range == 0)
			trans( *heaterMode()).str(XString("Off"));
		else
			trans( *powerRange()) = range - 1;

		interface()->query("MOUT? 1");
		trans( *manualPower()) = interface()->toDouble();
		interface()->query("PID? 1");
		double p, i, d;
		if(interface()->scanf("%lf,%lf,%lf", &p, &i, &d) != 3)
			throw XInterface::XConvError(__FILE__, __LINE__);
		for(Transaction tr( *this);; ++tr) {
			tr[ *prop()] = p;
			tr[ *interval()] = i;
			tr[ *deriv()] = d;
			if(tr.commit())
				break;
		}
	}
	start();
}

XLakeShore370::XLakeShore370(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XLakeShore(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "1", "2", "3", "4", "5", "6", "7", "8", 0L };
	const char *excitations_create[] = { 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create);
}

double XLakeShore370::getRaw(shared_ptr<XChannel> &channel) {
	interface()->query("RDGR? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShore370::getTemp(shared_ptr<XChannel> &channel) {
	interface()->query("RDGK? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShore370::getHeater() {
	interface()->query("HTR?");
	return interface()->toDouble();
}
void XLakeShore370::onPChanged(double p) {
	interface()->sendf("PID %f", p);
}
void XLakeShore370::onIChanged(double i) {
	interface()->sendf("PID ,,%f", i);
}
void XLakeShore370::onDChanged(double d) {
	interface()->sendf("PID ,,,%f", d);
}
void XLakeShore370::onTargetTempChanged(double temp) {
	Snapshot shot( *this);
	shared_ptr<XThermometer> thermo = shot[ *shared_ptr<XChannel> ( shot[ *currentChannel()])->thermometer()];
	if(thermo) {
		interface()->sendf("CSET %s,,2",
			(const char*)shot[ *currentChannel()].to_str().c_str());
		temp = thermo->getRawValue(temp);
	}
	else {
		interface()->sendf("CSET %s,,1",
			(const char*)shot[ *currentChannel()].to_str().c_str());
	}
	interface()->sendf("SETP %f", temp);
}
void XLakeShore370::onManualPowerChanged(double pow) {
	interface()->sendf("MOUT 1,%f", pow);
}
void XLakeShore370::onHeaterModeChanged(int) {
	Snapshot shot( *this);
	if(shot[ *heaterMode()].to_str() == "Off") {
		interface()->send("CMODE 4");
	}
	if(shot[ *heaterMode()].to_str() == "PID") {
		interface()->send("CMODE 1");
	}
	if(shot[ *heaterMode()].to_str() == "Man") {
		interface()->send("CMODE 3");
	}
}
void XLakeShore370::onPowerRangeChanged(int ran) {
	interface()->sendf("HTRRNG %d", ran + 1);
}
void XLakeShore370::onCurrentChannelChanged(const shared_ptr<XChannel> &ch) {
	interface()->sendf("CSET %s", (const char *) ch->getName().c_str());
}
void XLakeShore370::onExcitationChanged(const shared_ptr<XChannel> &, int) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
}
void XLakeShore370::open() throw (XKameError &) {
	interface()->query("CSET?");
	int ctrl_ch, units, htr_limit;
	double htr_res;
	if(interface()->scanf("%d,%*d,%d,%*d,%*d,%d,%lf", &ctrl_ch, &units, &htr_limit, &htr_res) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange()].clear();
		for(int i = 1; i < htr_limit; i++) {
			double pwr = htr_res * (pow(10.0, i) * 1e-7);
			if(pwr < 0.1)
				tr[ *powerRange()].add(formatString("%.2g uW", pwr * 1e3));
			else
				tr[ *powerRange()].add(formatString("%.2g mW", pwr));
		}
		if(tr.commit())
			break;
	}
	if( !shared_ptr<XDCSource>( ***extDCSource())) {
		for(Transaction tr( *this);; ++tr) {
			tr[ *currentChannel()] = ctrl_ch - 1;
			tr[ *heaterMode()].clear();
			tr[ *heaterMode()].add("Off");
			tr[ *heaterMode()].add("PID");
			tr[ *heaterMode()].add("Man");
			if(tr.commit())
				break;
		}

		interface()->query("CMODE?");
		switch(interface()->toInt()) {
		case 1:
			trans( *heaterMode()).str(XString("PID"));
			break;
		case 3:
			trans( *heaterMode()).str(XString("Man"));
			break;
		default:
			break;
		}
		interface()->query("HTRRNG?");
		int range = interface()->toInt();
		if(range == 0)
			trans( *heaterMode()).str(XString("Off"));
		else
			trans( *powerRange()) = range - 1;

		interface()->query("MOUT? 1");
		trans( *manualPower()) = interface()->toDouble();
		interface()->query("PID?");
		double p, i, d;
		if(interface()->scanf("%lf,%lf,%lf", &p, &i, &d) != 3)
			throw XInterface::XConvError(__FILE__, __LINE__);
		for(Transaction tr( *this);; ++tr) {
			tr[ *prop()] = p;
			tr[ *interval()] = i;
			tr[ *deriv()] = d;
			if(tr.commit())
				break;
		}
	}
	start();
}


XKE2700w7700::XKE2700w7700(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl>(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", 0L };
	const char *excitations_create[] = { 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create);
}
void XKE2700w7700::open() throw (XKameError &) {
	start();
	interface()->send("TRAC:CLE"); //Clears buffer.
	interface()->send("INIT:CONT OFF");
	interface()->send("TRIG:SOUR IMM"); //Immediate trigger.
	interface()->send("TRIG:COUN 1"); //1 scan.
}
double XKE2700w7700::getRaw(shared_ptr<XChannel> &channel) {
	int ch = atoi(channel->getName().c_str());
	interface()->sendf("ROUT:CLOS (@1%1d%1d)", ch / 10, ch % 10);
	interface()->query("READ?");
	double x;
	if(interface()->scanf("%lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
double XKE2700w7700::getTemp(shared_ptr<XChannel> &channel) {
	return getRaw(channel);
}
double XKE2700w7700::getHeater() {
	return 0.0;
}
void XKE2700w7700::onPChanged(double p) {
}
void XKE2700w7700::onIChanged(double i) {
}
void XKE2700w7700::onDChanged(double d) {
}
void XKE2700w7700::onTargetTempChanged(double temp) {
}
void XKE2700w7700::onManualPowerChanged(double pow) {
}
void XKE2700w7700::onHeaterModeChanged(int) {
}
void XKE2700w7700::onPowerRangeChanged(int) {
}
void XKE2700w7700::onCurrentChannelChanged(const shared_ptr<XChannel> &ch) {
}
void XKE2700w7700::onExcitationChanged(const shared_ptr<XChannel> &, int) {
}
