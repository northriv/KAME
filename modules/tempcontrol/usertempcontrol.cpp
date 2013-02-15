/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
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
		excitations_create, 1);
}
void XITC503::open() throw (XKameError &) {
	start();

	for(Transaction tr( *this);; ++tr) {
		const Snapshot &shot(tr);
		if( !hasExtDevice(shot, 0)) {
			tr[ *heaterMode(0)].clear();
			tr[ *heaterMode(0)].add("PID");
			tr[ *heaterMode(0)].add("Man");
			tr[ *powerMax(0)].setUIEnabled(false);
			tr[ *powerMin(0)].setUIEnabled(false);
		}
		tr[ *powerRange(0)].setUIEnabled(false);
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
double XITC503::getHeater(unsigned int /*loop*/) {
	return read(5);
}
void XITC503::onPChanged(unsigned int /*loop*/, double p) {
	interface()->sendf("P%f", p);
}
void XITC503::onIChanged(unsigned int /*loop*/, double i) {
	interface()->sendf("I%f", i);
}
void XITC503::onDChanged(unsigned int /*loop*/, double d) {
	interface()->sendf("D%f", d);
}
void XITC503::onTargetTempChanged(unsigned int /*loop*/, double temp) {
	if(( **heaterMode(0))->to_str() == "PID")
		interface()->sendf("T%f", temp);
}
void XITC503::onManualPowerChanged(unsigned int /*loop*/, double pow) {
	if(( **heaterMode(0))->to_str() == "Man")
		interface()->sendf("O%f", pow);
}
void XITC503::onHeaterModeChanged(unsigned int /*loop*/, int) {
}
void XITC503::onPowerRangeChanged(unsigned int /*loop*/, int) {
}
void XITC503::onCurrentChannelChanged(unsigned int /*loop*/, const shared_ptr<XChannel> &ch) {
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
		excitations_create, 1);

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
void XAVS47IB::onPChanged(unsigned int /*loop*/, double p) {
	int ip = lrint(p);
	if(ip > 60)
		ip = 60;
	if(ip < 5)
		ip = 5;
	ip = lrint(ip / 5.0 - 1.0);
	interface()->sendf("PRO %u", ip);
}
void XAVS47IB::onIChanged(unsigned int /*loop*/, double i) {
	int ii = lrint(i);
	if(ii > 4000)
		ii = 4000;
	ii = (ii < 2) ? 0 : lrint(log10((double) ii) * 3.0);
	interface()->sendf("ITC %u", ii);
}
void XAVS47IB::onDChanged(unsigned int /*loop*/, double d) {
	int id = lrint(d);
	id = (id < 1) ? 0 : lrint(log10((double) id) * 3.0) + 1;
	interface()->sendf("DTC %u", id);
}
void XAVS47IB::onTargetTempChanged(unsigned int /*loop*/, double) {
	setPoint();
}
void XAVS47IB::onManualPowerChanged(unsigned int /*loop*/, double) {
}
void XAVS47IB::onHeaterModeChanged(unsigned int /*loop*/, int) {
}
void XAVS47IB::onPowerRangeChanged(unsigned int /*loop*/, int ran) {
	setPowerRange(ran);
}
void XAVS47IB::onCurrentChannelChanged(unsigned int /*loop*/, const shared_ptr<XChannel> &ch) {
	Snapshot shot( *this);
	interface()->send("ARN 0;INP 0;ARN 0;RAN 7");
	interface()->sendf("DIS 0;MUX %u;ARN 0",
		QString(shot[ *currentChannel(0)].to_str()).toInt());
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
	if(ch != shared_ptr<XChannel>(shot[ *currentChannel(0)]))
		return;
	interface()->sendf("EXC %u", (unsigned int) exc);
	m_autorange_wait = 0;

	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange(0)].add("0");
		tr[ *powerRange(0)].add("1uW");
		tr[ *powerRange(0)].add("10uW");
		tr[ *powerRange(0)].add("100uW");
		tr[ *powerRange(0)].add("1mW");
		tr[ *powerRange(0)].add("10mW");
		tr[ *powerRange(0)].add("100mW");
		tr[ *powerRange(0)].add("1W");
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
	trans( *currentChannel(0)).str(formatString("%d", (int) lrint(read("MUX"))));
	onCurrentChannelChanged(0, ***currentChannel(0));

	start();

	for(Transaction tr( *this);; ++tr) {
		const Snapshot &shot(tr);
		if( !hasExtDevice(shot, 0)) {
			tr[ *heaterMode(0)].clear();
			tr[ *heaterMode(0)].add("PID");
			tr[ *powerMax(0)].setUIEnabled(false);
			tr[ *powerMin(0)].setUIEnabled(false);
		}
		if(tr.commit())
			break;
	}
}
void XAVS47IB::closeInterface() {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
	try {
		interface()->send("REM 0"); //LOCAL
	}
	catch(XInterface::XInterfaceError &e) {
		e.print(getLabel());
	}

	close();
}

int XAVS47IB::setRange(unsigned int range) {
	int rangebuf = ***powerRange(0);
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
	shared_ptr<XChannel> ch = shot[ *currentChannel(0)];
	if( !ch)
		return -1;
	shared_ptr<XThermometer> thermo = shot[ *ch->thermometer()];
	if( !thermo)
		return -1;
	double res = thermo->getRawValue(shot[ *targetTemp(0)]);
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
double XAVS47IB::getHeater(unsigned int /*loop*/) {
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
		excitations_create, 2);
}
XCryoconM32::XCryoconM32(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCryocon(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "A", "B", 0L };
	const char *excitations_create[] = { "CI", "10MV", "3MV", "1MV", 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create, 2);
}
void XCryocon::open() throw (XKameError &) {
	Snapshot shot( *this);
	const XNode::NodeList &list( *channels()->list());
	assert(list.size() == 2);
	shared_ptr<XChannel> ch0 = static_pointer_cast<XChannel>(list.at(0));
	shared_ptr<XChannel> ch1 = static_pointer_cast<XChannel>(list.at(1));
	interface()->query("INPUT A:VBIAS?");
	trans( *ch0->excitation()).str(interface()->toStrSimplified());
	interface()->query("INPUT B:VBIAS?");
	trans( *ch1->excitation()).str(interface()->toStrSimplified());

	for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
		trans( *powerRange(idx)).clear();
		if( !hasExtDevice(shot, idx)) {
			getChannel(idx);
			interface()->queryf("LOOP %u:PMAN?", idx + 1);
			trans( *manualPower(idx)).str(XString( &interface()->buffer()[0]));
			interface()->queryf("LOOP %u:PGAIN?", idx + 1);
			trans( *prop(idx)).str(XString( &interface()->buffer()[0]));
			interface()->queryf("LOOP %u:IGAIN?", idx + 1);
			trans( *interval(idx)).str(XString( &interface()->buffer()[0]));
			interface()->queryf("LOOP %u:DGAIN?", idx + 1);
			trans( *deriv(idx)).str(XString( &interface()->buffer()[0]));

			for(Transaction tr( *this);; ++tr) {
				tr[ *heaterMode(idx)].clear();
				tr[ *heaterMode(idx)].add("OFF");
				tr[ *heaterMode(idx)].add("PID");
				tr[ *heaterMode(idx)].add("MAN");
				tr[ *powerMin(idx)].setUIEnabled(false);
				if(tr.commit())
					break;
			}
			interface()->queryf("LOOP %u:TYPE?", idx + 1);
			QString s( &interface()->buffer()[0]);
			trans( *heaterMode(idx)).str(s.simplified());
		}
	}

	interface()->queryf("LOOP %u:RANGE?", 1);
	trans( *powerRange(0)).str(QString( &interface()->buffer()[0]).simplified());

	start();
}
void XCryoconM32::open() throw (XKameError &) {
	XCryocon::open();

	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange(0)].add("HI");
		tr[ *powerRange(0)].add("MID");
		tr[ *powerRange(0)].add("LOW");
		if(tr.commit())
			break;
	}
	Snapshot shot( *this);
	for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
		if( !hasExtDevice(shot, idx)) {
			interface()->queryf("LOOP %u:MAXPWR?", idx + 1);
			trans( *powerMax(idx)).str(XString( &interface()->buffer()[0]));
		}
	}
}
void XCryoconM32::onPowerMaxChanged(unsigned int loop, double x) {
	interface()->sendf("LOOP %u:MAXPWR %f ", loop, x);
}
void XCryoconM62::open() throw (XKameError &) {
	XCryocon::open();

	for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
		powerMax(idx)->setUIEnabled(false);
	}
	//LOOP 1
	interface()->query("HEATER:LOAD?");
	for(Transaction tr( *this);; ++tr) {
		if(interface()->toInt() == 50) {
			tr[ *powerRange(0)].add("0.05W");
			tr[ *powerRange(0)].add("0.5W");
			tr[ *powerRange(0)].add("5.0W");
			tr[ *powerRange(0)].add("50W");
		}
		else {
			tr[ *powerRange(0)].add("0.03W");
			tr[ *powerRange(0)].add("0.3W");
			tr[ *powerRange(0)].add("2.5W");
			tr[ *powerRange(0)].add("25W");
		}
		if(tr.commit())
			break;
	}
}
void XCryocon::onPChanged(unsigned int loop, double p) {
	interface()->sendf("LOOP %u:PGAIN %f", loop + 1, p);
}
void XCryocon::onIChanged(unsigned int loop, double i) {
	interface()->sendf("LOOP %u:IGAIN %f", loop + 1, i);
}
void XCryocon::onDChanged(unsigned int loop, double d) {
	interface()->sendf("LOOP %u:DGAIN %f", loop + 1, d);
}
void XCryocon::onTargetTempChanged(unsigned int loop, double temp) {
	setTemp(loop, temp);
}
void XCryocon::onManualPowerChanged(unsigned int loop, double pow) {
	interface()->sendf("LOOP %u:PMAN %f", loop + 1, pow);
}
void XCryocon::onHeaterModeChanged(unsigned int loop, int) {
	setHeaterMode(loop);
}
void XCryocon::onPowerRangeChanged(unsigned int loop, int) {
	if(loop != 0)
		return;
	interface()->sendf("LOOP %u:RANGE %s", loop + 1, ( **powerRange(loop))->to_str().c_str());
}
void XCryocon::onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) {
	interface()->sendf("LOOP %u:SOURCE %s", loop + 1, ch->getName().c_str());
}
void XCryocon::onExcitationChanged(const shared_ptr<XChannel> &ch, int) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
	interface()->send("INPUT " + ch->getName() + ":VBIAS "
		+ ( **ch->excitation())->to_str());
}
void XCryocon::setTemp(unsigned int loop, double temp) {
	if(temp > 0)
		control();
	else
		stopControl();

	Snapshot shot( *this);
	shared_ptr<XThermometer> thermo = shot[ *(shared_ptr<XChannel>(shot[ *currentChannel(loop)])->thermometer())];
	if(thermo)
		setHeaterSetPoint(loop, thermo->getRawValue(temp));
	else
		setHeaterSetPoint(loop, temp);
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
void XCryocon::getChannel(unsigned int loop) {
	interface()->queryf("LOOP %u:SOURCE?", loop + 1);
	char s[3];
	if(interface()->scanf("CH%s", s) != 1)
		return;
	trans( *currentChannel(loop)).str(XString(s));
}
void XCryocon::setHeaterMode(unsigned int loop) {
	Snapshot shot( *this);
	if(shot[ *heaterMode(loop)].to_str() == "Off")
		stopControl();
	else
		control();

	interface()->sendf("LOOP %u:TYPE %s", loop + 1, shot[ *heaterMode(loop)].to_str().c_str());
}
double XCryocon::getHeater(unsigned int loop) {
	interface()->queryf("LOOP %u:OUTP?", loop + 1);
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

int XCryocon::setHeaterSetPoint(unsigned int loop, double value) {
	interface()->sendf("LOOP %u:SETPT %f", loop + 1, value);
	return 0;
}

XNeoceraLTC21::XNeoceraLTC21(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "1", "2", 0L };
	const char *excitations_create[] = { 0L };
	//	const char *excitations_create[] = {"1mV", "320uV", "100uV", "32uV", "10uV", 0L};
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create, 2);
	interface()->setEOS("");
	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange(0)].add("0");
		tr[ *powerRange(0)].add("0.05W");
		tr[ *powerRange(0)].add("0.5W");
		tr[ *powerRange(0)].add("5W");
		tr[ *powerRange(0)].add("50W");
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
double XNeoceraLTC21::getHeater(unsigned int loop) {
	if(loop != 0)
		return 0.0;
	interface()->query("QHEAT?;");
	double x;
	if(interface()->scanf("%5lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
void XNeoceraLTC21::setHeater(unsigned int loop) {
	Snapshot shot( *this);
	interface()->sendf("SPID%u,%f,%f,%f,%f,100.0,%f;", loop + 1, (double)shot[ *prop(loop)],
		(double)shot[ *interval(loop)], (double)shot[ *deriv(loop)], (double)shot[ *manualPower(loop)],
		(double)shot[ *powerMax(loop)]);
}
void XNeoceraLTC21::onPChanged(unsigned int loop, double /*p*/) {
	setHeater(loop);
}
void XNeoceraLTC21::onIChanged(unsigned int loop, double /*i*/) {
	setHeater(loop);
}
void XNeoceraLTC21::onDChanged(unsigned int loop, double /*d*/) {
	setHeater(loop);
}
void XNeoceraLTC21::onTargetTempChanged(unsigned int loop, double temp) {
	interface()->sendf("SETP%u,%.5f;", loop + 1, temp);
}
void XNeoceraLTC21::onManualPowerChanged(unsigned int loop, double /*pow*/) {
	setHeater(loop);
}
void XNeoceraLTC21::onPowerMaxChanged(unsigned int loop, double /*pow*/) {
	setHeater(loop);
}
void XNeoceraLTC21::onHeaterModeChanged(unsigned int loop, int x) {
	if(loop != 0)
		return;
	if(x < 6) {
		interface()->sendf("SHCONT%d;", x);
		control();
	}
	else
		monitor();
}
void XNeoceraLTC21::onPowerRangeChanged(unsigned int loop, int ran) {
	if(loop != 0)
		return;
	interface()->sendf("SHMXPWR%d;", ran);
}
void XNeoceraLTC21::onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &cch) {
	int ch = atoi(cch->getName().c_str());
	if(ch < 1)
		ch = 3;
	interface()->sendf("SOSEN%u,%d;", loop + 1, ch);
}
void XNeoceraLTC21::onExcitationChanged(const shared_ptr<XChannel> &, int) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
}
void XNeoceraLTC21::open() throw (XKameError &) {
	Snapshot shot( *this);
	for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
		if( !hasExtDevice(shot, idx)) {
			interface()->queryf("QOUT?%u;", idx + 1);
			int sens, cmode, range;
			if(idx == 0) {
				if(interface()->scanf("%1d;%1d;%1d;", &sens, &cmode, &range) != 3)
					throw XInterface::XConvError(__FILE__, __LINE__);
			}
			else {
				if(interface()->scanf("%1d;%1d;", &sens, &cmode) != 2)
					throw XInterface::XConvError(__FILE__, __LINE__);
			}
			for(Transaction tr( *this);; ++tr) {
				tr[ *currentChannel(idx)].str(formatString("%d", sens));

				tr[ *heaterMode(idx)].clear();
				tr[ *heaterMode(idx)].add("AUTO P");
				tr[ *heaterMode(idx)].add("AUTO PI");
				tr[ *heaterMode(idx)].add("AUTO PID");
				tr[ *heaterMode(idx)].add("PID");
				tr[ *heaterMode(idx)].add("TABLE");
				tr[ *heaterMode(idx)].add("DEFAULT");
				tr[ *heaterMode(idx)].add("MONITOR");

				tr[ *heaterMode(idx)] = cmode;
				if(idx == 0)
					tr[ *powerRange(idx)] = range;
				if(tr.commit())
					break;
			}

			interface()->queryf("QPID?%u;", idx + 1);
			double p, i, d, power, limit;
			if(interface()->scanf("%lf;%lf;%lf;%lf;%lf;", &p, &i, &d, &power,
				&limit) != 5)
				throw XInterface::XConvError(__FILE__, __LINE__);
			for(Transaction tr( *this);; ++tr) {
				tr[ *prop(idx)] = p;
				tr[ *interval(idx)] = i;
				tr[ *deriv(idx)] = d;
				tr[ *manualPower(idx)] = power;
				tr[ *powerMax(idx)] = limit;
				tr[ *powerMin(0)].setUIEnabled(false);
				if(tr.commit())
					break;
			}
		}
	}
	start();
}


XLakeShoreBridge::XLakeShoreBridge(const char *name, bool runtime,
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
	XLakeShoreBridge (name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "A", "B", 0L };
	const char *excitations_create[] = { 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create, 2);
}

double XLakeShore340::getRaw(shared_ptr<XChannel> &channel) {
	interface()->query("SRDG? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShore340::getTemp(shared_ptr<XChannel> &channel) {
	interface()->query("KRDG? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShore340::getHeater(unsigned int loop) {
	if(loop == 0) {
		interface()->query("HTR?");
	}
	else {
		interface()->query("ANALOG?2");
		int mode;
		if(interface()->scanf("%*d,%d", &mode) != 1)
			throw XInterface::XConvError(__FILE__, __LINE__);
		if(mode != 3)
			return 0.0; //AOUT2 is not in loop mode.
		interface()->query("AOUT?2");
	}
	return interface()->toDouble();
}
void XLakeShore340::onPChanged(unsigned int loop, double p) {
	interface()->sendf("PID %u,%f", loop + 1, p);
}
void XLakeShore340::onIChanged(unsigned int loop, double i) {
	interface()->sendf("PID %u,,%f", loop + 1, i);
}
void XLakeShore340::onDChanged(unsigned int loop, double d) {
	interface()->sendf("PID %u,,,%f", loop + 1, d);
}
void XLakeShore340::onTargetTempChanged(unsigned int loop, double temp) {
	Snapshot shot( *this);
	shared_ptr<XThermometer> thermo = shot[ *shared_ptr<XChannel> ( shot[ *currentChannel(loop)])->thermometer()];
	if(thermo) {
		interface()->sendf("CSET %u,%s,3,1", loop + 1,
			(const char*)shot[ *currentChannel(loop)].to_str().c_str());
		temp = thermo->getRawValue(temp);
	}
	else {
		interface()->sendf("CSET %u,%s,1,1", loop + 1,
			(const char*)shot[ *currentChannel(loop)].to_str().c_str());
	}
	interface()->sendf("SETP %u,%f", loop + 1, temp);
}
void XLakeShore340::onManualPowerChanged(unsigned int loop, double pow) {
	interface()->sendf("MOUT %u,%f", loop + 1, pow);
}
void XLakeShore340::onPowerMaxChanged(unsigned int loop, double pow) {
	if(loop == 0)
		interface()->sendf("CLIMI %f", pow);
}
void XLakeShore340::onHeaterModeChanged(unsigned int loop, int) {
	Snapshot shot( *this);
	if(shot[ *heaterMode(loop)].to_str() == "PID") {
		interface()->sendf("CMODE %u,1", loop + 1);
	}
	if(shot[ *heaterMode(loop)].to_str() == "Man") {
		interface()->sendf("CMODE %u,3", loop + 1);
	}
}
void XLakeShore340::onPowerRangeChanged(unsigned int loop, int ran) {
	if(loop != 0)
		return;
	interface()->sendf("RANGE %d", ran);
}
void XLakeShore340::onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) {
	interface()->sendf("CSET %u,%s", loop + 1, (const char *) ch->getName().c_str());
}
void XLakeShore340::onExcitationChanged(const shared_ptr<XChannel> &, int) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
}
void XLakeShore340::open() throw (XKameError &) {
	Snapshot shot( *this);
	for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
		interface()->queryf("CDISP? %u", idx + 1);
		int res, maxcurr_idx;
		if(interface()->scanf("%*d,%d", &res) != 1)
			throw XInterface::XConvError(__FILE__, __LINE__);
		interface()->queryf("CLIMIT? %u", idx + 1);
		if(interface()->scanf("%*f,%*f,%*f,%d", &maxcurr_idx) != 1)
			throw XInterface::XConvError(__FILE__, __LINE__);

		interface()->query("CLIMI?");
		double max_curr_loop1 = interface()->toDouble();

		double maxcurr = pow(2.0, maxcurr_idx) * 0.125;
		for(Transaction tr( *this);; ++tr) {
			tr[ *powerRange(idx)].clear();
			if(idx == 0) {
				tr[ *powerRange(idx)].add("0");
				for(int i = 1; i < 6; i++) {
					tr[ *powerRange(idx)].add(formatString("%.2g W", (double) pow(10.0, i - 5.0)
						* pow(maxcurr, 2.0) * res));
				}
			}
			if(tr.commit())
				break;
		}
		if( !hasExtDevice(shot, idx)) {
			interface()->queryf("CSET? %u", idx + 1);
			for(Transaction tr( *this);; ++tr) {
				char ch[2];
				if(interface()->scanf("%1s", ch) == 1)
					tr[ *currentChannel(idx)].str(XString(ch));

				tr[ *heaterMode(idx)].clear();
				tr[ *heaterMode(idx)].add("PID");
				tr[ *heaterMode(idx)].add("Man");

				if(idx == 0)
					tr[ *powerMax(idx)] = max_curr_loop1;
				else
					tr[ *powerMax(idx)].setUIEnabled(false);
				tr[ *powerMin(idx)].setUIEnabled(false);
				if(tr.commit())
					break;
			}

			interface()->queryf("CMODE? %u", idx + 1);
			switch(interface()->toInt()) {
			case 1:
				trans( *heaterMode(idx)).str(XString("PID"));
				break;
			case 3:
				trans( *heaterMode(idx)).str(XString("Man"));
				break;
			default:
				break;
			}
			if(idx == 0) {
				interface()->query("RANGE?");
				int range = interface()->toInt();
				trans( *powerRange(0)) = range;
			}

			interface()->queryf("MOUT? %u", idx + 1);
			trans( *manualPower(idx)) = interface()->toDouble();
			interface()->queryf("PID? %u", idx + 1);
			double p, i, d;
			if(interface()->scanf("%lf,%lf,%lf", &p, &i, &d) != 3)
				throw XInterface::XConvError(__FILE__, __LINE__);
			for(Transaction tr( *this);; ++tr) {
				tr[ *prop(idx)] = p;
				tr[ *interval(idx)] = i;
				tr[ *deriv(idx)] = d;
				if(tr.commit())
					break;
			}
		}
	}
	start();
}

XLakeShore370::XLakeShore370(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XLakeShoreBridge(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = { "1", "2", "3", "4", "5", "6", "7", "8", 0L };
	const char *excitations_create[] = { 0L };
	createChannels(ref(tr_meas), meas, true, channels_create,
		excitations_create, 1);
}

double XLakeShore370::getRaw(shared_ptr<XChannel> &channel) {
	interface()->query("RDGR? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShore370::getTemp(shared_ptr<XChannel> &channel) {
	interface()->query("RDGK? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShore370::getHeater(unsigned int /*loop*/) {
	interface()->query("HTR?");
	return interface()->toDouble();
}
void XLakeShore370::onPChanged(unsigned int /*loop*/, double p) {
	interface()->sendf("PID %f", p);
}
void XLakeShore370::onIChanged(unsigned int /*loop*/, double i) {
	interface()->sendf("PID ,,%f", i);
}
void XLakeShore370::onDChanged(unsigned int /*loop*/, double d) {
	interface()->sendf("PID ,,,%f", d);
}
void XLakeShore370::onTargetTempChanged(unsigned int /*loop*/, double temp) {
	Snapshot shot( *this);
	shared_ptr<XThermometer> thermo = shot[ *shared_ptr<XChannel> ( shot[ *currentChannel(0)])->thermometer()];
	if(thermo) {
		interface()->sendf("CSET %s,,2",
			(const char*)shot[ *currentChannel(0)].to_str().c_str());
		temp = thermo->getRawValue(temp);
	}
	else {
		interface()->sendf("CSET %s,,1",
			(const char*)shot[ *currentChannel(0)].to_str().c_str());
	}
	interface()->sendf("SETP %f", temp);
}
void XLakeShore370::onManualPowerChanged(unsigned int /*loop*/, double pow) {
	interface()->sendf("MOUT %f", pow);
}
void XLakeShore370::onHeaterModeChanged(unsigned int /*loop*/, int) {
	Snapshot shot( *this);
	if(shot[ *heaterMode(0)].to_str() == "Off") {
		interface()->send("CMODE 4");
	}
	if(shot[ *heaterMode(0)].to_str() == "PID") {
		interface()->send("CMODE 1");
	}
	if(shot[ *heaterMode(0)].to_str() == "Man") {
		interface()->send("CMODE 3");
	}
}
void XLakeShore370::onPowerRangeChanged(unsigned int /*loop*/, int ran) {
	interface()->sendf("HTRRNG %d", ran);
}
void XLakeShore370::onCurrentChannelChanged(unsigned int /*loop*/, const shared_ptr<XChannel> &ch) {
	interface()->sendf("CSET %s", (const char *) ch->getName().c_str());
}
void XLakeShore370::onExcitationChanged(const shared_ptr<XChannel> &, int) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
}
void XLakeShore370::open() throw (XKameError &) {
	Snapshot shot( *this);;
	interface()->query("CSET?");
	int ctrl_ch, units, htr_limit;
	double htr_res;
	if(interface()->scanf("%d,%*d,%d,%*d,%*d,%d,%lf", &ctrl_ch, &units, &htr_limit, &htr_res) != 4)
		throw XInterface::XConvError(__FILE__, __LINE__);

	for(Transaction tr( *this);; ++tr) {
		tr[ *powerRange(0)].clear();
		tr[ *powerRange(0)].add("0");
		for(int i = 1; i < htr_limit; i++) {
			double pwr = htr_res * (pow(10.0, i) * 1e-7);
			if(pwr < 0.1)
				tr[ *powerRange(0)].add(formatString("%.2g uW", pwr * 1e3));
			else
				tr[ *powerRange(0)].add(formatString("%.2g mW", pwr));
		}
		if(tr.commit())
			break;
	}
	if( !hasExtDevice(shot, 0)) {
		for(Transaction tr( *this);; ++tr) {
			tr[ *currentChannel(0)].str(formatString("%d", ctrl_ch ));
			tr[ *heaterMode(0)].clear();
			tr[ *heaterMode(0)].add("Off");
			tr[ *heaterMode(0)].add("PID");
			tr[ *heaterMode(0)].add("Man");
			if(tr.commit())
				break;
		}

		interface()->query("CMODE?");
		switch(interface()->toInt()) {
		case 1:
			trans( *heaterMode(0)).str(XString("PID"));
			break;
		case 3:
			trans( *heaterMode(0)).str(XString("Man"));
			break;
		default:
			break;
		}
		interface()->query("HTRRNG?");
		int range = interface()->toInt();
		trans( *powerRange(0)) = range;

		interface()->query("MOUT?");
		trans( *manualPower(0)) = interface()->toDouble();
		interface()->query("PID?");
		double p, i, d;
		if(interface()->scanf("%lf,%lf,%lf", &p, &i, &d) != 3)
			throw XInterface::XConvError(__FILE__, __LINE__);
		for(Transaction tr( *this);; ++tr) {
			tr[ *prop(0)] = p;
			tr[ *interval(0)] = i;
			tr[ *deriv(0)] = d;
			tr[ *powerMax(0)].setUIEnabled(false);
			tr[ *powerMin(0)].setUIEnabled(false);
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
		excitations_create, 0);
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
double XKE2700w7700::getHeater(unsigned int) {
	return 0.0;
}

