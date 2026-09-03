/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "charinterface.h"
#include "userdcsource.h"

REGISTER_TYPE(XDriverList, YK7651, "YOKOGAWA 7651 dc source");
REGISTER_TYPE(XDriverList, ADVR6142, "ADVANTEST TR6142/R6142/R6144 DC V/DC A source");
REGISTER_TYPE(XDriverList, MicroTaskTCS, "MICROTASK/Leiden Triple Current Source");
REGISTER_TYPE(XDriverList, OptotuneICC4C2000, "Optotune ICC4C-2000 current controller");

XYK7651::XYK7651(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
   : XCharDeviceDriver<XDCSource>(name, runtime, ref(tr_meas), meas) {
	//Displayed as V / A rather than the 7651's raw "F" codes (F1 = DC voltage source,
	//F5 = DC current source), which is what this combo used to show. XComboNode has no
	//separate display label -- itemStrings() returns {s, s} and XQComboBoxConnector writes the
	//selection back through .label -- so the shown text IS the stored text, and .kam files hold
	//it verbatim. changeFunction() therefore carries a shim that maps a legacy "F1"/"F5"
	//loaded from an older .kam onto these entries.
	iterate_commit([=](Transaction &tr){
		tr[ *function()].add("V");
		tr[ *function()].add("A");
    });
	channel()->disable();
	interface()->setGPIBUseSerialPollOnRead(false);
	interface()->setGPIBUseSerialPollOnWrite(false);
}
void
XYK7651::open() {
	this->start();
	msecsleep(3000); // wait for instrumental reset.
}
void
XYK7651::changeFunction(int /*ch*/, int func) {
	//"Function" IS the V/A switch: it sends the 7651's "F" command.
	//  combo index 0 ("V") -> F1 = DC VOLTAGE source,  index 1 ("A") -> F5 = DC CURRENT source.
	//It also decides which set of ranges is meaningful, so the Range list is rebuilt to match.
	if(func < 0) {
		//Nothing selected. A .kam saved while this combo still showed the raw "F" codes holds
		//Function.load("F1"/"F5"); such a string matches no item, so XComboNode keeps it with
		//index -1. Remap it here and return: the write re-enters with a valid index. Done
		//before any interface lock, since it touches STM.
		XString stale = Snapshot( *this)[ *function()].to_str();
		if(stale == "F1") {
			trans( *function()) = 0;
		}
		else if(stale == "F5") {
			trans( *function()) = 1;
		}
		return;
	}
	const bool is_volt = (func == 0);
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened()) return;
	iterate_commit([=](Transaction &tr){
		tr[ *range()].clear();
		if(is_volt) {
			//DC V, sent as R2..R6 by changeRange(). The top range is 30V (max output
			//+-32.000V per the 7651 spec) -- the instrument has NO 100V range.
			tr[ *range()].add("10mV");
			tr[ *range()].add("100mV");
			tr[ *range()].add("1V");
			tr[ *range()].add("10V");
			tr[ *range()].add("30V");
		}
		else {
			//DC A, sent as R4..R6 by changeRange(). Max output +-120.000mA on the 100mA range.
			tr[ *range()].add("1mA");
			tr[ *range()].add("10mA");
			tr[ *range()].add("100mA");
		}
    });
	//Derive the F code from the index, NOT from the combo label. The label used to be sent
	//verbatim (to_str() + "E"), which is why the raw device codes were what the UI displayed.
	interface()->send(is_volt ? "F1E" : "F5E");
}
void
XYK7651::changeOutput(int /*ch*/, bool x) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened()) return;
	interface()->sendf("O%uE", x ? 1 : 0);
}
void
XYK7651::changeValue(int /*ch*/, double x, bool autorange) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened()) return;
	if(autorange)
		interface()->sendf("SA%.10fE", x);
	else
		interface()->sendf("S%.10fE", x);
}
double
XYK7651::max(int /*ch*/, bool autorange) const {
	Snapshot shot( *this);
	int ran = shot[ *range()];
	//Nominal full scale per range. The 7651 can actually source 1.2x these, but the nominal
	//value is the right thing to advertise: tempcontrol clamps its output against max().
	if(shot[ *function()] == 0) {
		//DC V: 10mV, 100mV, 1V, 10V, 30V. NOT a power of ten at the top -- the 7651's highest
		//voltage range is 30V, so the old 10e-3*10^ran formula wrongly reported 100V.
		static const double fs_v[] = {10e-3, 100e-3, 1.0, 10.0, 30.0};
		if(autorange || (ran < 0) || (ran >= (int)(sizeof(fs_v) / sizeof(fs_v[0]))))
			ran = (int)(sizeof(fs_v) / sizeof(fs_v[0])) - 1;
		return fs_v[ran];
	}
	else {
		//DC A: 1mA, 10mA, 100mA.
		static const double fs_a[] = {1e-3, 10e-3, 100e-3};
		if(autorange || (ran < 0) || (ran >= (int)(sizeof(fs_a) / sizeof(fs_a[0]))))
			ran = (int)(sizeof(fs_a) / sizeof(fs_a[0])) - 1;
		return fs_a[ran];
	}
}
void
XYK7651::changeRange(int /*ch*/, int ran) {
	Snapshot shot( *this);
	{
		XScopedLock<XInterface> lock( *interface());
		if( !interface()->isOpened()) return;
		//The 7651 numbers all ranges in one "R" sequence, so the combo index needs a per-mode
		//offset: DC V index 0..4 -> R2..R6 (10mV,100mV,1V,10V,30V);
		//        DC A index 0..2 -> R4..R6 (1mA,10mA,100mA).
		if(shot[ *function()] == 0) {
			if(ran == -1)
				ran = 4;
			ran += 2;
		}
		else {
			if(ran == -1)
				ran = 2;
			ran += 4;
		}
		interface()->sendf("R%dE", ran);
	}
}

XADVR6142::XADVR6142(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
   : XCharDeviceDriver<XDCSource>(name, runtime, ref(tr_meas), meas) {
	iterate_commit([=](Transaction &tr){
		tr[ *function()].add("V [V]");
		tr[ *function()].add("I [A]");
    });
	channel()->disable();
	interface()->setEOS("\r\n");
}
void
XADVR6142::open() {
	this->start();
}
void
XADVR6142::changeFunction(int /*ch*/, int ) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened()) return;
	iterate_commit([=](Transaction &tr){
		const Snapshot &shot(tr);
		if(shot[ *function()] == 0) {
			tr[ *range()].clear();
			tr[ *range()].add("10mV");
			tr[ *range()].add("100mV");
			tr[ *range()].add("1V");
			tr[ *range()].add("10V");
			tr[ *range()].add("30V");
		}
		else {
			tr[ *range()].clear();
			tr[ *range()].add("1mA");
			tr[ *range()].add("10mA");
			tr[ *range()].add("100mA");
		}
    });
}
void
XADVR6142::changeOutput(int /*ch*/, bool x) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened()) return;
	if(x)
		interface()->send("E");
	else
		interface()->send("H");
}
void
XADVR6142::changeValue(int /*ch*/, double x, bool autorange) {
	XScopedLock<XInterface> lock( *interface());
	Snapshot shot( *this);
	if( !interface()->isOpened()) return;
	if(autorange) {
		if(shot[ *function()] == 0) {
			interface()->sendf("D%.8fV", x);
		}
		else {
			x *= 1e3;
			interface()->sendf("D%.8fMA", x);
		}
	}
	else {
		if(shot[ *function()] == 0) {
			if(shot[ *range()] <= 1)
				x *= 1e3;
		}
		else {
			x *= 1e3;
		}
		interface()->sendf("D%.8f", x);
	}
}
double
XADVR6142::max(int /*ch*/, bool autorange) const {
	Snapshot shot( *this);
	int ran = shot[ *range()];
	if(shot[ *function()] == 0) {
		if(autorange || (ran == -1))
			ran = 4;
		if(ran == 4)
			return 30;
		return 10e-3 * pow(10.0, (double)ran);
	}
	else {
		if(autorange || (ran == -1))
			ran = 2;
		return 1e-3 * pow(10.0, (double)ran);
	}
}
void
XADVR6142::changeRange(int /*ch*/, int ran) {
	Snapshot shot( *this);
	{
		XScopedLock<XInterface> lock( *interface());
		if( !interface()->isOpened()) return;
		if(shot[ *function()] == 0) {
			if(ran == -1)
				ran = 2;
			ran += 2;
			interface()->sendf("V%d", ran);
		}
		else {
			if(ran == -1)
				ran = 2;
			ran += 1;
			interface()->sendf("I%d", ran);
		}
	}
}


XOptotuneICC4C2000::XOptotuneICC4C2000(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
   : XCharDeviceDriver<XDCSource>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\r\n");
    interface()->setSerialBaudRate(256000);
    interface()->setSerialStopBits(1);
	iterate_commit([=](Transaction &tr){
		tr[ *channel()].add("1");
		tr[ *channel()].add("2");
		tr[ *channel()].add("3");
        tr[ *channel()].add("4");
        tr[ *function()].add({"mA"});
        tr[ *function()].disable();
        tr[ *range()].add({"2000mA"});
        tr[ *range()].disable();
        tr[ *output()].disable();
        tr[ *interface()->device()] = "SERIAL";
    });
}
XDCSource::Status
XOptotuneICC4C2000::queryStatus(int ch) {
    Status st;
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened()) return st;
    interface()->queryf("SETCHANNEL=%i", ch);
    interface()->query("GETCURRENT");
    st.value = interface()->toDouble();
    st.valid = true;
    return st;
}
void
XOptotuneICC4C2000::changeValue(int ch, double x, bool autorange) {
	{
		XScopedLock<XInterface> lock( *interface());
		if(!interface()->isOpened()) return;
        interface()->queryf("SETCHANNEL=%i", ch);
        interface()->queryf("SETCURRENT=%g", x);
	}
}
void
XOptotuneICC4C2000::open() {
	this->start();
    interface()->query("GETID");
	fprintf(stderr, "%s\n", (const char*)&interface()->buffer()[0]);
    interface()->query("GETDEVICESN");
    fprintf(stderr, "%s\n", (const char*)&interface()->buffer()[0]);
}
double
XOptotuneICC4C2000::max(int ch, bool autorange) const {
    return 2000.0;
}

XMicroTaskTCS::XMicroTaskTCS(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
   : XCharDeviceDriver<XDCSource>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\n");
    interface()->setSerialBaudRate(9600);
    interface()->setSerialStopBits(2);
    iterate_commit([=](Transaction &tr){
        tr[ *channel()].add("1");
        tr[ *channel()].add("2");
        tr[ *channel()].add("3");
        tr[ *function()].disable();
        tr[ *range()].add("99uA");
        tr[ *range()].add("0.99uA");
        tr[ *range()].add("9.9mA");
        tr[ *range()].add("99mA");
    });
}
XDCSource::Status
XMicroTaskTCS::queryStatus(int ch) {
    Status st;
    unsigned int ran[3];
    unsigned int v[3];
    unsigned int o[3];
    {
        XScopedLock<XInterface> lock( *interface());
        if( !interface()->isOpened()) return st;
        interface()->query("STATUS?");
        if(interface()->scanf("%*u%*u,%u,%u,%u,%*u,%u,%u,%u,%*u,%u,%u,%u,%*u",
            &ran[0], &v[0], &o[0],
            &ran[1], &v[1], &o[1],
            &ran[2], &v[2], &o[2]) != 9)
            throw XInterface::XConvError(__FILE__, __LINE__);
    }
    st.value  = pow(10.0, (double)ran[ch] - 1) * 1e-6 * v[ch];
    st.output = (bool)o[ch];
    st.range  = (int)ran[ch] - 1;
    st.valid  = true;
    return st;
}
void
XMicroTaskTCS::changeOutput(int ch, bool x) {
    {
        XScopedLock<XInterface> lock( *interface());
        if(!interface()->isOpened()) return;
        unsigned int v[3];
        interface()->query("STATUS?");
        if(interface()->scanf("%*u%*u,%*u,%*u,%u,%*u,%*u,%*u,%u,%*u,%*u,%*u,%u,%*u", &v[0], &v[1], &v[2])
            != 3)
            throw XInterface::XConvError(__FILE__, __LINE__);
        for(int i = 0; i < 3; i++) {
            if(ch != i)
                v[i] = 0;
            else
                v[i] ^= x ? 1 : 0;
        }
        interface()->sendf("SETUP 0,0,%u,0,0,0,%u,0,0,0,%u,0", v[0], v[1], v[2]);
        interface()->receive(2);
    }
    updateStatus();
}
void
XMicroTaskTCS::changeValue(int ch, double x, bool autorange) {
    {
        XScopedLock<XInterface> lock( *interface());
        if(!interface()->isOpened()) return;
        if((x >= 0.099) || (x < 0))
            throw XInterface::XInterfaceError(i18n("Value is out of range."), __FILE__, __LINE__);
        if(autorange) {
            interface()->sendf("SETDAC %u 0 %u", (unsigned int)(ch + 1), (unsigned int)lrint(x * 1e6));
            interface()->receive(1);
        }
        else {
            unsigned int ran[3];
            interface()->query("STATUS?");
            if(interface()->scanf("%*u%*u,%u,%*u,%*u,%*u,%u,%*u,%*u,%*u,%u,%*u,%*u,%*u",
                &ran[0], &ran[1], &ran[2]) != 3)
                throw XInterface::XConvError(__FILE__, __LINE__);
            int v = lrint(x / (pow(10.0, (double)ran[ch] - 1) * 1e-6));
            v = std::max(std::min(v, 99), 0);
            interface()->sendf("DAC %u %u", (unsigned int)(ch + 1), (unsigned int)v);
            interface()->receive(2);
        }
    }
    updateStatus();
}
void
XMicroTaskTCS::changeRange(int ch, int newran) {
    {
        XScopedLock<XInterface> lock( *interface());
        if(!interface()->isOpened()) return;
        unsigned int ran[3], v[3];
        interface()->query("STATUS?");
        if(interface()->scanf("%*u%*u,%u,%u,%*u,%*u,%u,%u,%*u,%*u,%u,%u,%*u,%*u",
            &ran[0], &v[0],
            &ran[1], &v[1],
            &ran[2], &v[2]) != 6)
            throw XInterface::XConvError(__FILE__, __LINE__);
        double x = pow(10.0, (double)ran[ch] - 1) * 1e-6 * v[ch];
        int newv = lrint(x / (pow(10.0, (double)newran) * 1e-6));
        newv = std::max(std::min(newv, 99), 0);
        interface()->sendf("SETDAC %u %u %u",
            (unsigned int)(ch + 1), (unsigned int)(newran + 1), (unsigned int)newv);
        interface()->receive(1);
    }
    updateStatus();
}
double
XMicroTaskTCS::max(int ch, bool autorange) const {
    if(autorange) return 0.099;
    {
        XScopedLock<XInterface> lock( *interface());
        if(!interface()->isOpened()) return 0.099;
        unsigned int ran[3];
        interface()->query("STATUS?");
        if(interface()->scanf("%*u%*u,%u,%*u,%*u,%*u,%u,%*u,%*u,%*u,%u,%*u,%*u,%*u",
            &ran[0], &ran[1], &ran[2]) != 3)
            throw XInterface::XConvError(__FILE__, __LINE__);
        return pow(10.0, (double)(ran[ch] - 1)) * 99e-6;
    }
;}
void
XMicroTaskTCS::open() {
    this->start();
    interface()->query("ID?");
    fprintf(stderr, "%s\n", (const char*)&interface()->buffer()[0]);
}
