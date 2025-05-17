/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
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
REGISTER_TYPE(XDriverList, LakeShore350, "LakeShore 350 temp. controller");
REGISTER_TYPE(XDriverList, LakeShore370, "LakeShore 370 AC res. bridge");
REGISTER_TYPE(XDriverList, AVS47IB, "Picowatt AVS-47 AC res. bridge");
REGISTER_TYPE(XDriverList, ITC503, "Oxford ITC-503 temp. controller");
REGISTER_TYPE(XDriverList, NeoceraLTC21, "Neocera LTC-21 temp. controller");
REGISTER_TYPE(XDriverList, LinearResearch700, "LinearResearch LR-700  AC res. bridge");

XITC503::XITC503(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XOxfordDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
    createChannels(ref(tr_meas), meas, true,
        {"1", "2", "3"},
        {"HEATER", "GASFLOW"});
}
void XITC503::open() {
	start();

    interface()->query("X");
    int stat, automan, locrem, sweep, ctrlsens, autopid;
    if(interface()->scanf("X%1dA%dC%1dS%2dH%1dL%1d", &stat, &automan, &locrem, &sweep, &ctrlsens, &autopid) != 6)
        throw XInterface::XConvError(__FILE__, __LINE__);

	iterate_commit([=](Transaction &tr){
		const Snapshot &shot(tr);
        for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
            if( !hasExtDevice(shot, idx)) {
                tr[ *heaterMode(idx)].clear();
                tr[ *heaterMode(idx)].add({"AUTO", "MAN"});
                tr[ *powerMax(idx)].setUIEnabled(false);
                tr[ *powerMin(idx)].setUIEnabled(false);
                tr[ *currentChannel(idx)].str(formatString("%d", ctrlsens));
                tr[ *heaterMode(idx)] = (automan & ((idx == 0) ? 1 : 2)) ? 0 : 1;
            }
            tr[ *powerRange(idx)].setUIEnabled(false);
        }
    });

    double t = read(0);
    trans( *targetTemp(0)) = t;
    trans( *targetTemp(1)).setUIEnabled(false);
    double p = read(8);
    double i = read(9);
    double d = read(10);
    trans( *prop(0)) = p;
    trans( *interval(0)) = i;
    trans( *deriv(0)) = d;
    trans( *prop(1)).setUIEnabled(false);
    trans( *interval(1)).setUIEnabled(false);
    trans( *deriv(1)).setUIEnabled(false);
}
double XITC503::getRaw(shared_ptr<XChannel> &channel) {
	interface()->send("X");
	return read(QString(channel->getName()).toInt());
}
double XITC503::getTemp(shared_ptr<XChannel> &channel) {
	interface()->send("X");
	return read(QString(channel->getName()).toInt());
}
double XITC503::getHeater(unsigned int loop) {
    return read(loop ? 7: 5); //loop1 is for gas flow.
}
void XITC503::onPChanged(unsigned int loop, double p) {
    if(loop) return;
	interface()->sendf("P%f", p);
}
void XITC503::onIChanged(unsigned int loop, double i) {
    if(loop) return;
    interface()->sendf("I%f", i);
}
void XITC503::onDChanged(unsigned int loop, double d) {
    if(loop) return;
    interface()->sendf("D%f", d);
}
void XITC503::onTargetTempChanged(unsigned int loop, double temp) {
    if(loop) return;
    if(( **heaterMode(0))->to_str() == "AUTO")
		interface()->sendf("T%f", temp);
}
void XITC503::onManualPowerChanged(unsigned int loop, double pow) {
    if(loop == 0) {
        if(( **heaterMode(0))->to_str() == "MAN")
            interface()->sendf("O%f", pow);
    }
    else {
        if(( **heaterMode(1))->to_str() == "MAN")
            interface()->sendf("G%f", pow);
    }
}
void XITC503::onHeaterModeChanged(unsigned int loop, int) {
    int mode = ((( **heaterMode(0))->to_str() == "MAN") ? 0 : 1) + ((( **heaterMode(1))->to_str() == "MAN") ? 0 : 2);
    interface()->sendf("A%d", mode);
}
void XITC503::onPowerRangeChanged(unsigned int /*loop*/, int) {
}
void XITC503::onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) {
    if(loop) return;
	interface()->send("H" + ch->getName());
}
void XITC503::onExcitationChanged(const shared_ptr<XChannel> &, int) {
}

XAVS47IB::XAVS47IB(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
    createChannels(ref(tr_meas), meas, false,
        {"0", "1", "2", "3", "4", "5", "6", "7"},
        {"Loop"});

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

	iterate_commit([=](Transaction &tr){
        tr[ *powerRange(0)].add({"0", "1uW", "10uW", "100uW", "1mW", "10mW", "100mW", "1W"});
    });
}
double XAVS47IB::getRaw(shared_ptr<XChannel> &) {
	return getRes();
}
double XAVS47IB::getTemp(shared_ptr<XChannel> &) {
	return getRes();
}
void XAVS47IB::open() {
	msecsleep(50);
	interface()->send("REM 1;ARN 0;DIS 0");
	trans( *currentChannel(0)).str(formatString("%d", (int) lrint(read("MUX"))));
	onCurrentChannelChanged(0, ***currentChannel(0));

	start();

	iterate_commit([=](Transaction &tr){
		const Snapshot &shot(tr);
		if( !hasExtDevice(shot, 0)) {
			tr[ *heaterMode(0)].clear();
			tr[ *heaterMode(0)].add("PID");
			tr[ *powerMax(0)].setUIEnabled(false);
			tr[ *powerMin(0)].setUIEnabled(false);
		}
    });
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
    XCharDeviceDriver<XTempControl, XCryoconCharInterface> (name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("");
    interface()->setGPIBUseSerialPollOnWrite(false);
    interface()->setGPIBUseSerialPollOnRead(false);
    interface()->setGPIBWaitBeforeWrite(20);
    //    ExclusiveWaitAfterWrite = 10;
    interface()->setGPIBWaitBeforeRead(20);
    interface()->setSerialEOS("\n");
    interface()->setSerialBaudRate(9600);
    interface()->setSerialStopBits(1);
    interface()->setSerialFlushBeforeWrite(true);
}
XCryoconM62::XCryoconM62(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCryocon(name, runtime, ref(tr_meas), meas) {
    createChannels(ref(tr_meas), meas, true,
        {"A", "B"},
        {"HEATER", "AOUT"});
}
XCryoconM32::XCryoconM32(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCryocon(name, runtime, ref(tr_meas), meas) {
    createChannels(ref(tr_meas), meas, true,
        {"A", "B"},
        {"Loop#1", "Loop#2"});
}
void XCryocon::open() {
    Snapshot shot_ch( *channels());
    const XNode::NodeList &list( *shot_ch.list());
    assert(list.size() == 2);
    shared_ptr<XChannel> ch0 = static_pointer_cast<XChannel>(list.at(0));
    shared_ptr<XChannel> ch1 = static_pointer_cast<XChannel>(list.at(1));
    interface()->query("INPUT A:VBIAS?");
    trans( *ch0->excitation()).str(interface()->toStrSimplified());
    interface()->query("INPUT B:VBIAS?");
    trans( *ch1->excitation()).str(interface()->toStrSimplified());

    Snapshot shot( *this);
    for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
        trans( *powerRange(idx)).clear();
        if( !hasExtDevice(shot, idx)) {
            getChannel(idx);
            interface()->queryf("%s:PMAN?", loopString(idx));
            trans( *manualPower(idx)).str(XString( &interface()->buffer()[0]));
            interface()->queryf("%s:PGAIN?", loopString(idx));
            trans( *prop(idx)).str(XString( &interface()->buffer()[0]));
            interface()->queryf("%s:IGAIN?", loopString(idx));
            trans( *interval(idx)).str(XString( &interface()->buffer()[0]));
            interface()->queryf("%s:DGAIN?", loopString(idx));
            trans( *deriv(idx)).str(XString( &interface()->buffer()[0]));

            iterate_commit([=](Transaction &tr){
                tr[ *heaterMode(idx)].clear();
                tr[ *heaterMode(idx)].add({"OFF", "PID", "MAN"});
                tr[ *powerMin(idx)].setUIEnabled(false);
            });
            interface()->queryf("%s:TYPE?", loopString(idx));
            trans( *heaterMode(idx)).str(interface()->toStrSimplified());
        }
    }

    interface()->queryf("%s:RANGE?", loopString(0));
    trans( *powerRange(0)).str(interface()->toStrSimplified());

    start();
}
void XCryoconM32::open() {
    XCryocon::open();

    iterate_commit([=](Transaction &tr){
        tr[ *powerRange(0)].add({"HI", "MID", "LOW"});
    });
    Snapshot shot( *this);
    for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
        if( !hasExtDevice(shot, idx)) {
            interface()->queryf("%s:MAXPWR?", loopString(idx));
            trans( *powerMax(idx)).str(XString( &interface()->buffer()[0]));
        }
    }
}
void XCryoconM32::onPowerMaxChanged(unsigned int loop, double x) {
    interface()->sendf("%s:MAXPWR %f ", loopString(loop), x);
}
void XCryoconM62::open() {
    XCryocon::open();

    for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
        powerMax(idx)->setUIEnabled(false);
    }
    //LOOP 1
    interface()->query("HEATER:LOAD?");
    iterate_commit([=](Transaction &tr){
        if(interface()->toInt() == 50) {
            tr[ *powerRange(0)].add({"0.05W", "0.5W", "5.0W", "50W"});
        }
        else {
            tr[ *powerRange(0)].add({"0.03W", "0.3W", "2.5W", "25W"});
        }
    });
}
void XCryocon::onPChanged(unsigned int loop, double p) {
    interface()->sendf("%s:PGAIN %f", loopString(loop), p);
}
void XCryocon::onIChanged(unsigned int loop, double i) {
    interface()->sendf("%s:IGAIN %f", loopString(loop), i);
}
void XCryocon::onDChanged(unsigned int loop, double d) {
    interface()->sendf("%s:DGAIN %f", loopString(loop), d);
}
void XCryocon::onTargetTempChanged(unsigned int loop, double temp) {
    setTemp(loop, temp);
}
void XCryocon::onManualPowerChanged(unsigned int loop, double pow) {
    interface()->sendf("%s:PMAN %f", loopString(loop), pow);
}
void XCryocon::onHeaterModeChanged(unsigned int loop, int) {
    setHeaterMode(loop);
}
void XCryocon::onPowerRangeChanged(unsigned int loop, int) {
    if(loop != 0)
        return;
    interface()->sendf("%s:RANGE %s", loopString(loop), ( **powerRange(loop))->to_str().c_str());
}
void XCryocon::onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) {
    interface()->sendf("%s:SOURCE %s", loopString(loop), ch->getName().c_str());
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
    if(shared_ptr<XChannel> ch = shot[ *currentChannel(loop)]) {
        shared_ptr<XThermometer> thermo = shot[ *ch->thermometer()];
        if(thermo)
            setHeaterSetPoint(loop, thermo->getRawValue(temp));
        else
            setHeaterSetPoint(loop, temp);
    }
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
    interface()->queryf("%s:SOURCE?", loopString(loop));
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

    interface()->sendf("%s:TYPE %s", loopString(loop), shot[ *heaterMode(loop)].to_str().c_str());
}
double XCryocon::getHeater(unsigned int loop) {
    interface()->queryf("%s:OUTP?", loopString(loop));
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
    interface()->sendf("%s:SETPT %f", loopString(loop), value);
    return 0;
}

XNeoceraLTC21::XNeoceraLTC21(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
    createChannels(ref(tr_meas), meas, true,
        {"1", "2"},
        {"Loop1", "Loop2"});
    interface()->setEOS("");
	interface()->setSerialEOS("\n");
	iterate_commit([=](Transaction &tr){
    tr[ *powerRange(0)].add({"0", "0.05W", "0.5W", "5W", "50W"});
    });
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
void XNeoceraLTC21::open() {
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
			iterate_commit([=](Transaction &tr){
				tr[ *currentChannel(idx)].str(formatString("%d", sens));

				tr[ *heaterMode(idx)].clear();
                tr[ *heaterMode(idx)].add({"AUTO P", "AUTO PI", "AUTO PID", "PID", "TABLE", "DEFAULT", "MONITOR"});

				tr[ *heaterMode(idx)] = cmode;
				if(idx == 0)
					tr[ *powerRange(idx)] = range;
            });

			interface()->queryf("QPID?%u;", idx + 1);
			double p, i, d, power, limit;
			if(interface()->scanf("%lf;%lf;%lf;%lf;%lf;", &p, &i, &d, &power,
				&limit) != 5)
				throw XInterface::XConvError(__FILE__, __LINE__);
			iterate_commit([=](Transaction &tr){
				tr[ *prop(idx)] = p;
				tr[ *interval(idx)] = i;
				tr[ *deriv(idx)] = d;
				tr[ *manualPower(idx)] = power;
				tr[ *powerMax(idx)] = limit;
				tr[ *powerMin(0)].setUIEnabled(false);
            });
		}
	}
	start();
}


XLakeShoreBridge::XLakeShoreBridge(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
	interface()->setEOS("\r\n");
    interface()->setGPIBMAVbit(4); //valid read??? but serial poll does not respond.
    interface()->setGPIBUseSerialPollOnWrite(false);
    interface()->setGPIBUseSerialPollOnRead(false);
    interface()->setGPIBWaitBeforeWrite(40);
    //    ExclusiveWaitAfterWrite = 10;
    interface()->setGPIBWaitBeforeRead(40);
    interface()->setSerialStopBits(1);
    interface()->setSerialBaudRate(9600);
    interface()->setSerialParity(XCharInterface::PARITY_ODD);
    interface()->setSerial7Bits(true);
    interface()->setSerialEOS("\r\n");
}

XLakeShore340::XLakeShore340(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XLakeShoreBridge (name, runtime, ref(tr_meas), meas) {

    createChannels(ref(tr_meas), meas, true,
        {"A", "B"},
        {"Loop1", "Loop2"},
        false, false); //Assuming scanner is not installed.
}

double XLakeShoreBridge::getRaw(shared_ptr<XChannel> &channel) {
	interface()->query("SRDG? " + channel->getName());
	return interface()->toDouble();
}
double XLakeShoreBridge::getTemp(shared_ptr<XChannel> &channel) {
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
void XLakeShoreBridge::onPChanged(unsigned int loop, double p) {
    interface()->queryf("PID? %u", loop + 1);
    double i, d;
    if(interface()->scanf("%*lf,%lf,%lf", &i, &d) != 2)
        throw XInterface::XConvError(__FILE__, __LINE__);
    interface()->sendf("PID %u,%f,%f,%f", loop + 1, p, i, d);
}
void XLakeShoreBridge::onIChanged(unsigned int loop, double i) {
    interface()->queryf("PID? %u", loop + 1);
    double p, d;
    if(interface()->scanf("%lf,%*lf,%lf", &p, &d) != 2)
        throw XInterface::XConvError(__FILE__, __LINE__);
    interface()->sendf("PID %u,%f,%f,%f", loop + 1, p, i, d);
}
void XLakeShoreBridge::onDChanged(unsigned int loop, double d) {
    interface()->queryf("PID? %u", loop + 1);
    double p, i;
    if(interface()->scanf("%lf,%lf", &p, &i) != 2)
        throw XInterface::XConvError(__FILE__, __LINE__);
    interface()->sendf("PID %u,%f,%f,%f", loop + 1, p, i, d);
}
void XLakeShore340::onTargetTempChanged(unsigned int loop, double temp) {
	Snapshot shot( *this);
    if(shared_ptr<XChannel> ch = shot[ *currentChannel(loop)]) {
        shared_ptr<XThermometer> thermo = shot[ *ch->thermometer()];
        if(thermo) {
            interface()->sendf("CSET %u,%s,3,1", loop + 1,
                (const char*)shot[ *ch->thermometer()].to_str().c_str());
            temp = thermo->getRawValue(temp);
        }
        else {
            interface()->sendf("CSET %u,%s,1,1", loop + 1,
                (const char*)shot[ *ch->thermometer()].to_str().c_str());
        }
        interface()->sendf("SETP %u,%f", loop + 1, temp);
    }
}
void XLakeShoreBridge::onManualPowerChanged(unsigned int loop, double pow) {
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
    if( !ch) return;
	interface()->sendf("CSET %u,%s", loop + 1, (const char *) ch->getName().c_str());
}
void XLakeShore340::open() {
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
        double max_curr_loop1 = 0.0;
        try {
            max_curr_loop1 = interface()->toDouble();
        }
        catch (XInterface::XConvError &) {
        //older firmware
            max_curr_loop1 = 2.0;
            powerMax(0)->setUIEnabled(false);
        }

		double maxcurr = pow(2.0, maxcurr_idx) * 0.125;
		iterate_commit([=](Transaction &tr){
			tr[ *powerRange(idx)].clear();
			if(idx == 0) {
				tr[ *powerRange(idx)].add("0");
				for(int i = 1; i < 6; i++) {
					tr[ *powerRange(idx)].add(formatString("%.2g W", (double) pow(10.0, i - 5.0)
						* pow(maxcurr, 2.0) * res));
				}
			}
        });
		if( !hasExtDevice(shot, idx)) {
			interface()->queryf("CSET? %u", idx + 1);
			iterate_commit([=](Transaction &tr){
				char ch[2];
				if(interface()->scanf("%1s", ch) == 1)
					tr[ *currentChannel(idx)].str(XString(ch));

				tr[ *heaterMode(idx)].clear();
                tr[ *heaterMode(idx)].add({"PID", "Man"});

				if(idx == 0)
					tr[ *powerMax(idx)] = max_curr_loop1;
				else
					tr[ *powerMax(idx)].setUIEnabled(false);
				tr[ *powerMin(idx)].setUIEnabled(false);
            });

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
			iterate_commit([=](Transaction &tr){
				tr[ *prop(idx)] = p;
				tr[ *interval(idx)] = i;
				tr[ *deriv(idx)] = d;
            });
		}
	}
	start();
}

XLakeShore350::XLakeShore350(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XLakeShoreBridge (name, runtime, ref(tr_meas), meas) {

    createChannels(ref(tr_meas), meas, true,
        {"A", "B", "C", "D"},
        {"OUTPUT1", "OUTPUT2", "AOUT1", "AOUT2"},
        false, false); //Assuming scanner is not installed.
}

double XLakeShore350::getHeater(unsigned int loop) {
    if(loop <= 1)
        interface()->queryf("HTR? %u", loop + 1);
    else
        interface()->queryf("AOUT? %u", loop + 1);
    return interface()->toDouble();
}
void XLakeShore350::onTargetTempChanged(unsigned int loop, double temp) {
    Snapshot shot( *this);
    if(shared_ptr<XChannel> ch = shot[ *currentChannel(loop)]) {
        shared_ptr<XThermometer> thermo = shot[ *ch->thermometer()];
        if(thermo) {
            temp = thermo->getRawValue(temp);
        }
        interface()->sendf("SETP %u,%f", loop + 1, temp);
    }
}
void XLakeShore350::onPowerMaxChanged(unsigned int loop, double pow) {
    interface()->queryf("HTRSET %d", loop + 1);
    int res,maxc,maxuc,currorpw;
    interface()->scanf("%d,%d,%d,%d", &res, &maxc, &maxuc, &currorpw);
    if(loop == 0)
        interface()->sendf("HTRSET %d,%d,0,%.1f,%d", loop + 1, res, pow, currorpw);
}
void XLakeShore350::onHeaterModeChanged(unsigned int loop, int) {
    Snapshot shot( *this);
    onCurrentChannelChanged(loop, shot[ *currentChannel(loop)]);
}
void XLakeShore350::onPowerRangeChanged(unsigned int loop, int ran) {
    interface()->sendf("RANGE %d,%d", loop + 1, ran);
}
void XLakeShore350::onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) {
    Snapshot shot( *this);
    if( !ch) return;
    int chno = ch->getName().c_str()[0] - 'A';
    interface()->sendf("OUTMODE %u,%d,%d", loop + 1, (unsigned int)shot[ *heaterMode(loop)], chno + 1);
}
void XLakeShore350::open() {
    Snapshot shot( *this);
    int res, maxcurr_idx;
    double max_curr_loop1 = 0.0;
    interface()->query("HTRSET? 1");
    if(interface()->scanf("%d,%d,%lf", &res, &maxcurr_idx, &max_curr_loop1) != 3)
        throw XInterface::XConvError(__FILE__, __LINE__);
    res *= 25;
    for(unsigned int idx = 0; idx < numOfLoops(); ++idx) {
        if(idx != 0)
            powerMax(idx)->setUIEnabled(false);

        double maxcurr = 0.1; //100mA for OUTPUT2
        if(idx == 0)
            maxcurr = pow(2.0, maxcurr_idx / 2.0) / 2.0;
        else
            res = 100.0;
        iterate_commit([=](Transaction &tr){
            tr[ *powerRange(idx)].clear();
            tr[ *powerRange(idx)].add("0");
            if(idx <= 1) {
                for(int i = 1; i < 6; i++) {
                    double power = pow(10.0, i - 5.0) * pow(maxcurr, 2.0) * res;
                    if(power < 0.9e-3)
                        tr[ *powerRange(idx)].add(formatString("%.3g uW", power * 1e6));
                    else if (power < 0.9)
                        tr[ *powerRange(idx)].add(formatString("%.3g mW", power * 1e3));
                    else
                        tr[ *powerRange(idx)].add(formatString("%.3g W", power));
                }
            }
            else
                tr[ *powerRange(idx)].add("On");
        });
        if( !hasExtDevice(shot, idx)) {
            interface()->queryf("OUTMODE? %u", idx + 1);
            int mode, srcch;
            if(interface()->scanf("%d,%d", &mode, &srcch) != 2)
                throw XInterface::XConvError(__FILE__, __LINE__);
            iterate_commit([=](Transaction &tr){
                if((srcch > 0) && (srcch <= shot.size( channels())))
                    tr[ *currentChannel(idx)] = shot.list( channels())->at(srcch - 1);
                tr[ *heaterMode(idx)].clear();
                if(idx <= 1)
                    tr[ *heaterMode(idx)].add({"Off", "PID", "ZONE", "OpenLoop"});
                else
                    tr[ *heaterMode(idx)].add({"Off", "PID", "ZONE", "OpenLoop", "MonOut", "WarmUp"});

                tr[ *heaterMode(idx)] = mode;

                if(idx == 0)
                    tr[ *powerMax(idx)] = max_curr_loop1;
                else
                    tr[ *powerMax(idx)].setUIEnabled(false);
                tr[ *powerMin(idx)].setUIEnabled(false);
            });

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
            iterate_commit([=](Transaction &tr){
                tr[ *prop(idx)] = p;
                tr[ *interval(idx)] = i;
                tr[ *deriv(idx)] = d;
            });
        }
    }
    start();
}
void
XLakeShore350::onSetupChannelChanged(const shared_ptr<XChannel> &channel) {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened() || !channel)
        return;
    interface()->query("INNAME? " + channel->getName());
    auto inname = interface()->toStrSimplified();
    interface()->query("INCRV? " + channel->getName());
    int incrv = interface()->toUInt();
    XString crvinfo;
    if(incrv) {
        interface()->queryf("CRVHDR? %d",incrv);
        crvinfo = interface()->toStrSimplified();
    }
    interface()->query("INTYPE? " + channel->getName());
    unsigned int sen_type = 0, autorange = 0, range = 0,
        compensation = 0, units = 0, sen_exc = 0;
    interface()->scanf("%u,%u,%u,%u,%u,%u", &sen_type, &autorange, &range, &compensation, &units, &sen_exc);
    try {
        channel->iterate_commit([=](Transaction &tr){
            tr[ *channel->excitation()].clear();
            XString curr_exc;
            if(sen_type == 3) {
                tr[ *channel->excitation()].add({"1 mV", "10 mV"});
                curr_exc = tr[ *channel->excitation()].itemStrings().at(sen_exc).label;
            }
            tr[ *channel->info()] =
                inname + ":" +
                std::map<int,std::string>{{0, "Disabled"},{1, "Diode"},{2, "Platinum RTD"},{3, "NTC RTD"},
                                           {4, "Thermocouple"},{5, "Capacitance"}}.at(sen_type) +
                ", \n" + "Autorange: " + std::string(autorange ? "On" : "Off") + ", \n" +
                curr_exc + ", \n" +
                "Units:" +
                std::map<int,std::string>{{1, "Kelvin"},{2, "Celsius"},{3, "Sensor"}}.at(units) +
                ", \n" + crvinfo;
        });
    }
    catch(std::out_of_range &e) {
    }
}
void XLakeShore350::onExcitationChanged(const shared_ptr<XChannel> &channel, int exc) {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened())
        return;
    interface()->query("INTYPE? " + channel->getName());
    unsigned int sen_type = 0, autorange = 0, range = 0,
        compensation = 0, units = 0, sen_exc = 0;
    interface()->scanf("%u,%u,%u,%u,%u,%u", &sen_type, &autorange, &range, &compensation, &units, &sen_exc);
    interface()->sendf("INTYPE %s,%d,%d,%d,%d,%d,%d", channel->getName().c_str(),
        sen_type, autorange, range, compensation, units, exc);
}

XLakeShore370::XLakeShore370(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XLakeShoreBridge(name, runtime, ref(tr_meas), meas) {

    createChannels(ref(tr_meas), meas, true,
        {"1", "2", "3", "4", "5", "6", "7", "8"},
        {"Loop"},
        true, true); //Assuming scanner is used.
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
    double i, d;
    if(interface()->scanf("%*lf,%lf,%lf", &i, &d) != 2)
        throw XInterface::XConvError(__FILE__, __LINE__);

    interface()->sendf("PID %6f,%6f,%6f", p, i, d);
}
void XLakeShore370::onIChanged(unsigned int /*loop*/, double i) {
    double p, d;
    if(interface()->scanf("%lf,%*lf,%lf", &p, &d) != 2)
        throw XInterface::XConvError(__FILE__, __LINE__);

    interface()->sendf("PID %6f,%6f,%6f", p, i, d);
}
void XLakeShore370::onDChanged(unsigned int /*loop*/, double d) {
    double p, i;
    if(interface()->scanf("%lf,%lf,%*lf", &p, &i) != 2)
        throw XInterface::XConvError(__FILE__, __LINE__);

    interface()->sendf("PID %6f,%6f,%6f", p, i, d);
}
void XLakeShore370::onTargetTempChanged(unsigned int /*loop*/, double temp) {
	Snapshot shot( *this);
    if(shared_ptr<XChannel> ch = shot[ *currentChannel(0)]) {
        shared_ptr<XThermometer> thermo = shot[ *ch->thermometer()];
        if(thermo) {
        //     interface()->sendf("CSET %s,,2",
        //         (const char*)shot[ *currentChannel(0)].to_str().c_str());
            temp = thermo->getRawValue(temp);
        }
        else {
            // interface()->sendf("CSET %s,,1",
            //     (const char*)shot[ *currentChannel(0)].to_str().c_str());
        }
        interface()->sendf("SETP %g", temp);
    }
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
void XLakeShore370::open() {
    Snapshot shot( *this);
    interface()->send("*CLS");
    interface()->query("*IDN?");
    m_is372 = (interface()->toStrSimplified().find("MODEL372") != std::string::npos);
    interface()->query("CSET?");
	int ctrl_ch, units, htr_limit;
	double htr_res;
	if(interface()->scanf("%d,%*d,%d,%*d,%*d,%d,%lf", &ctrl_ch, &units, &htr_limit, &htr_res) != 4)
		throw XInterface::XConvError(__FILE__, __LINE__);

	iterate_commit([=](Transaction &tr){
		tr[ *powerRange(0)].clear();
		tr[ *powerRange(0)].add("0");
		for(int i = 1; i < htr_limit; i++) {
			double pwr = htr_res * (pow(10.0, i) * 1e-7);
			if(pwr < 0.1)
				tr[ *powerRange(0)].add(formatString("%.2g uW", pwr * 1e3));
			else
				tr[ *powerRange(0)].add(formatString("%.2g mW", pwr));
		}
    });
	if( !hasExtDevice(shot, 0)) {
		iterate_commit([=](Transaction &tr){
			tr[ *currentChannel(0)].str(formatString("%d", ctrl_ch ));
			tr[ *heaterMode(0)].clear();
            tr[ *heaterMode(0)].add({"Off", "PID", "Man"});
        });

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
		iterate_commit([=](Transaction &tr){
			tr[ *prop(0)] = p;
			tr[ *interval(0)] = i;
			tr[ *deriv(0)] = d;
			tr[ *powerMax(0)].setUIEnabled(false);
			tr[ *powerMin(0)].setUIEnabled(false);
        });
	}
	start();
}
void
XLakeShore370::onSetupChannelChanged(const shared_ptr<XChannel> &channel) {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened() || !channel)
        return;
    interface()->query("INSET? " + channel->getName());
    unsigned int offon = 0, dwell = 0, pause = 0,
        crvno = 0, tempco = 0;
    interface()->scanf("%u,%u,%u,%u,%u", &offon, &dwell, &pause, &crvno, &tempco);
    XString crvinfo;
    if(crvno) {
        interface()->queryf("CRVHDR? %d",crvno);
        crvinfo = interface()->toStrSimplified();
    }
    unsigned int curr_mode = 0, excitation = 0, range = 0, autorange = 0, cs_off = 0;
    if(is372())
        interface()->query("INTYPE? " + channel->getName());
    else
        interface()->query("RDGRNG? " + channel->getName());
    interface()->scanf("%u,%u,%u,%u,%u", &curr_mode, &excitation, &range, &autorange, &cs_off);
    XString exc_unit = "V";
    double exc_begin = 2.0e-6;
    unsigned int exc_end = 12;
    if(curr_mode) {
        exc_unit = "A";
        exc_begin = 1.0e-12;
        exc_end = 22;
    }
    const auto si = std::map<int,std::string>{{-4,"p"},{-3,"n"},{-2,"u"},{-1,"m"},{0,""},
                                               {1,"k"},{2,"M"}};

    double r = 2.0e-3 * pow(10.0, 0.5 * (range - 1));
    int k = (int)floor(log10(r)/3.0);
    r /= pow(10.0, 3 * k);
    XString rangestr = formatString("%.3g ", r) + si.at(k) + "Ohm";

    interface()->query("SCAN?");
    unsigned int autoscan;
    interface()->scanf("%*u,%u", &autoscan);
    try {
        channel->iterate_commit([=](Transaction &tr){
            tr[ *channel->excitation()].clear();
            for(unsigned int i = 0; i < exc_end; ++i) {
                double e = exc_begin * pow(10.0, 0.5 * i);
                int k = (int)floor(log10(e)/3.0);
                e /= pow(10.0, 3 * k);
                tr[ *channel->excitation()].add(formatString("%.3g ", e) + si.at(k) + exc_unit);
            }
            XString curr_exc = tr[ *channel->excitation()].itemStrings().at(excitation - 1).label;
            tr[ *channel->excitation()] = excitation - 1;
            tr[ *channel->scanDwellSeconds()] = (double)dwell;

            tr[ *channel->info()] =
                std::string(offon ? "On" : "Off") +
                ", \n" +
                "Autorange: " + std::string(autorange ? "On" : "Off") +
                ", " + rangestr + ", \n" +
                "Excitation: " + curr_exc + ", " +
                std::string(cs_off ? "Off" : "On") + ", \n" +
                formatString("Dwell: %u sec., Pause: %u sec.,\n", dwell, pause) +
               "Autoscan: " + std::string(autoscan ? "On" : "Off") +
               ", \n" +
                crvinfo;
        });
    }
    catch(std::out_of_range &e) {
    }
}

void XLakeShore370::onExcitationChanged(const shared_ptr<XChannel> &channel, int exc) {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened())
        return;
    unsigned int curr_mode = 0, range = 0, autorange = 0, cs_off = 0,units=0;
    if(is372()) {
        interface()->query("INTYPE? " + channel->getName());
        interface()->scanf("%u,%*u,%u,%u,%u,%u", &curr_mode, &range, &autorange, &cs_off,&units);
        interface()->sendf("INTYPE %s,%u,%d,%u,%u,%u,%u", channel->getName().c_str(), curr_mode,
                           exc + 1, range, autorange, cs_off,units);
    }
    else {
        interface()->query("RDGRNG? " + channel->getName());
        interface()->scanf("%u,%*u,%u,%u,%u", &curr_mode, &range, &autorange, &cs_off);
        interface()->sendf("RDGRNG %s,%u,%d,%u,%u,%u", channel->getName().c_str(), curr_mode,
                           exc + 1, range, autorange, cs_off);
    }
}
void XLakeShore370::onChannelEnableChanged(const shared_ptr<XChannel> &channel, bool enable) {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened())
        return;
    interface()->query("INSET? " + channel->getName());
    unsigned int offon = 0, dwell = 0, pause = 0,
        crvno = 0, tempco = 0;
    interface()->scanf("%u,%u,%u,%u,%u", &offon, &dwell, &pause, &crvno, &tempco);
    interface()->sendf("INSET %s,%d,%u,%u,%u,%u", channel->getName().c_str(), enable ? 1 : 0,
        dwell, pause, crvno, tempco);
    if(enable)
        interface()->sendf("SCAN %s,%d", channel->getName().c_str(), 1);
}
void XLakeShore370::onScanDwellSecChanged(const shared_ptr<XChannel> &channel, double sec) {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened())
        return;
    interface()->query("INSET? " + channel->getName());
    unsigned int offon = 0, dwell = 0, pause = 0,
        crvno = 0, tempco = 0;
    interface()->scanf("%u,%u,%u,%u,%u", &offon, &dwell, &pause, &crvno, &tempco);
    dwell = std::max(1L, lrint(sec));
    pause = std::max(3L, lrint(sec));
    interface()->sendf("INSET %s,%u,%u,%u,%u,%u", channel->getName().c_str(), offon,
        dwell, pause, crvno, tempco);
}

XLinearResearch700::XLinearResearch700(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XTempControl> (name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\n");
    interface()->setGPIBUseSerialPollOnWrite(false);
    interface()->setGPIBUseSerialPollOnRead(false);
    interface()->setGPIBWaitBeforeWrite(40);
    interface()->setGPIBWaitBeforeRead(40);

    createChannels(ref(tr_meas), meas, true,
        {"0"},
        {"Loop"});
    iterate_commit([=](Transaction &tr){
        tr[ *powerRange(0)].add({"30uA", "100uA", "300uA", "1mA", "3mA", "10mA", "30mA", "100mA", "300mA", "1A", "3A"});
    });
}

double
XLinearResearch700::parseResponseMessage() {
    double v; char unit;
    int ret = interface()->scanf("%lf%c", &v, &unit);
    if((ret != 1) && (ret != 2))
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(ret == 2) {
        if(unit == 'K')
            v *= 1e3;
        if(unit == 'M')
            v *= 1e6;
        if(unit == 'U')
            v *= 1e-6;
    }
    return v;
}

double XLinearResearch700::getRaw(shared_ptr<XChannel> &channel) {
    interface()->query("GET 0");
    double res =  parseResponseMessage();
    return res;
}
double XLinearResearch700::getTemp(shared_ptr<XChannel> &channel) {
    return getRaw(channel);
}
double XLinearResearch700::getHeater(unsigned int loop) {
    interface()->query("GET 8");
    double v;
    if(interface()->scanf("%lf V", &v) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    return v;
}
void XLinearResearch700::onPChanged(unsigned int loop, double p) {
    int x = lrint(log10(p / 0.1) * 3);
    interface()->sendf("HEATER G=%02d", x);
}
void XLinearResearch700::onIChanged(unsigned int loop, double i) {
    int x = lrint(log10(i / 0.2) * 3);
    interface()->sendf("HEATER T=%02d", x);
}
void XLinearResearch700::onDChanged(unsigned int loop, double d) {
}
void XLinearResearch700::onTargetTempChanged(unsigned int loop, double temp) {
    Snapshot shot( *this);
    if(shared_ptr<XChannel> ch = shot[ *currentChannel(loop)]) {
        shared_ptr<XThermometer> thermo = shot[ *ch->thermometer()];
        if(thermo) {
            temp = thermo->getRawValue(temp);
        }
        interface()->sendf("OFFSET R=%f", temp);
    }
}
void XLinearResearch700::onManualPowerChanged(unsigned int loop, double pow) {
    interface()->sendf("HEATER Q=%+03d", (int)lrint(pow * 10.0));
}
void XLinearResearch700::onHeaterModeChanged(unsigned int loop, int) {
    Snapshot shot( *this);
    if(shot[ *heaterMode(loop)].to_str() == "Normal") {
        interface()->sendf("HEATER L=0");
        interface()->send("HEATER 1"); //ON
        return;
    }
    if(shot[ *heaterMode(loop)].to_str() == "Inverted") {
        interface()->sendf("HEATER L=1");
        interface()->send("HEATER 1"); //ON
        return;
    }
    if(shot[ *heaterMode(loop)].to_str() == "Open") {
        interface()->sendf("HEATER L=2");
        interface()->send("HEATER 1"); //ON
        return;
    }
    interface()->send("HEATER 0"); //OFF
}
void XLinearResearch700::onPowerRangeChanged(unsigned int loop, int ran) {
    interface()->sendf("HEATER R=%02d", ran);
}
void XLinearResearch700::onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) {
}
void XLinearResearch700::onExcitationChanged(const shared_ptr<XChannel> &, int exc) {
    interface()->sendf("EXCITATION %d", exc);
}
void XLinearResearch700::open() {
    Snapshot shot_ch( *channels());
    const XNode::NodeList &list( *shot_ch.list());
    assert(list.size() == 1);
    shared_ptr<XChannel> ch0 = static_pointer_cast<XChannel>(list.at(0));

    interface()->query("GET 6");
    int range, exc, vexc, fil, mode, ll, snum;
    if(interface()->scanf("%1dR,%1dE,%3d%%,%1dF,%1dM,%1dL,%2dS", &range, &exc, &vexc, &fil, &mode, &ll, &snum) != 7)
        throw XInterface::XConvError(__FILE__, __LINE__);

    iterate_commit([=](Transaction &tr){
        tr[ *powerRange(0)] = range;
        tr[ *ch0->excitation()] = exc;
    });
    start();
}


