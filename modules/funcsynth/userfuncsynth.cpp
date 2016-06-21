/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "userfuncsynth.h"
#include "charinterface.h"
#include "analyzer.h"

REGISTER_TYPE(XDriverList, WAVEFACTORY, "NF WAVE-FACTORY pulse generator");

XWAVEFACTORY::XWAVEFACTORY(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XFuncSynth>(name, runtime, ref(tr_meas), meas) {

	iterate_commit([=](Transaction &tr){
		tr[ *function()].add("SINUSOID");
		tr[ *function()].add("TRIANGLE");
		tr[ *function()].add("SQUARE");
		tr[ *function()].add("PRAMP");
		tr[ *function()].add("NRAMP");
		tr[ *function()].add("USER");
		tr[ *function()].add("VSQUARE");
		tr[ *mode()].add("NORMAL");
		tr[ *mode()].add("BURST");
		tr[ *mode()].add("SWEEP");
		tr[ *mode()].add("MODULATION");
		tr[ *mode()].add("NOISE");
		tr[ *mode()].add("DC");
    });
}
/*
  double
  XWAVEFACTORY::Read(void)
  {
  string buf;
  Query("?PHS", &buf);
  double x = 0;
  sscanf(buf.c_str(), "%*s %lf", &x);
  return x;
  }
*/
void
XWAVEFACTORY::onOutputChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("SIG %d", shot[ *output()] ? 1 : 0);
}

void
XWAVEFACTORY::onTrigTouched(const Snapshot &shot, XTouchableNode *) {
	interface()->send("TRG 1");
}

void
XWAVEFACTORY::onModeChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("OMO %d", (int)shot[ *mode()]);
}

void
XWAVEFACTORY::onFunctionChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("FNC %d", (int)shot[ *function()] + 1);
}

void
XWAVEFACTORY::onFreqChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("FRQ %e" , (double)shot[ *freq()]);
}

void
XWAVEFACTORY::onAmpChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("AMV %e" , (double)shot[ *amp()]);
}

void
XWAVEFACTORY::onPhaseChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("PHS %e" , (double)shot[ *phase()]);
}

void
XWAVEFACTORY::onOffsetChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("OFS %e" , (double)shot[ *offset()]);
}
