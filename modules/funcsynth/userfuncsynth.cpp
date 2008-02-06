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
#include "userfuncsynth.h"
#include "charinterface.h"
#include "analyzer.h"

REGISTER_TYPE(XDriverList, WAVEFACTORY, "NF WAVE-FACTORY pulse generator");

XWAVEFACTORY::XWAVEFACTORY(const char *name, bool runtime, 
						   const shared_ptr<XScalarEntryList> &scalarentries,
						   const shared_ptr<XInterfaceList> &interfaces,
						   const shared_ptr<XThermometerList> &thermometers,
						   const shared_ptr<XDriverList> &drivers) : 
    XCharDeviceDriver<XFuncSynth>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
	function()->add("SINUSOID");
	function()->add("TRIANGLE");
	function()->add("SQUARE");
	function()->add("PRAMP");
	function()->add("NRAMP");
	function()->add("USER");
	function()->add("VSQUARE");
	mode()->add("NORMAL");
	mode()->add("BURST");
	mode()->add("SWEEP");
	mode()->add("MODULATION");
	mode()->add("NOISE");
	mode()->add("DC");
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
XWAVEFACTORY::onOutputChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("SIG %d", *output() ? 1 : 0);
}

void
XWAVEFACTORY::onTrigTouched(const shared_ptr<XNode> &)
{
	interface()->send("TRG 1");
}

void
XWAVEFACTORY::onModeChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("OMO %d", (int)*mode());
}

void
XWAVEFACTORY::onFunctionChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("FNC %d", (int)*function() + 1);
}

void
XWAVEFACTORY::onFreqChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("FRQ %e" , (double)*freq());
}

void
XWAVEFACTORY::onAmpChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("AMV %e" , (double)*amp());
}

void
XWAVEFACTORY::onPhaseChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("PHS %e" , (double)*phase());
}


void
XWAVEFACTORY::onOffsetChanged(const shared_ptr<XValueNodeBase> &)
{
    interface()->sendf("OFS %e" , (double)*offset());
}
