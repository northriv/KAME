/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "lecroy.h"
#include "charinterface.h"
#include "xwavengraph.h"

REGISTER_TYPE(XDriverList, LecroyDSO, "Lecroy/Iwatsu X-Stream DSO");

//---------------------------------------------------------------------------
XLecroyDSO::XLecroyDSO(const char *name, bool runtime,
		   const shared_ptr<XScalarEntryList> &scalarentries,
		   const shared_ptr<XInterfaceList> &interfaces,
		   const shared_ptr<XThermometerList> &thermometers,
		   const shared_ptr<XDriverList> &drivers) :
	XCharDeviceDriver<XDSO>(name, runtime, scalarentries, interfaces, thermometers, drivers) {
	const char* ch[] = {"C1", "C2", "C3", "C4", "M1", "M2", "M3", "M4", 0L};
	for(int i = 0; ch[i]; i++) {
		trace1()->add(ch[i]);
		trace2()->add(ch[i]);
	}
	const char* sc[] = {"0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10",
						"20", "50", "100", 0L};
	for(int i = 0; sc[i]; i++)
	{
		vFullScale1()->add(sc[i]);
		vFullScale2()->add(sc[i]);
	}
	const char* tr[] = {"EXT", "EXT10", "CH1", "CH2", "CH3", "CH4", "LINE", 0L};
	for(int i = 0; tr[i]; i++)
	{
		trigSource()->add(tr[i]);
	}

	interface()->setGPIBWaitBeforeWrite(20); //20msec
	interface()->setGPIBWaitBeforeSPoll(10); //10msec
	interface()->setEOS("");

	recordLength()->value(10000);
}

void
XLecroyDSO::open() throw (XInterface::XInterfaceError &)
{
	interface()->send("COMM_HEADER OFF");
	interface()->send("COMM_FORMAT DEF9,WORD,BIN");
	interface()->send("COMM_ORDER HI");
	onAverageChanged(average());

	start();
}
void 
XLecroyDSO::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
	int avg = *average();
	avg = std::max(1, avg);
	const char *atype = *singleSequence() ? "SUMMED" : "CONTINUOUS";
    std::string ch = trace1()->to_str();
    if(!ch.empty())
		interface()->sendf("F1:DEFINE EQN,'AVG(%s)',AVGTYPE,%s,SWEEPS,%d",
			ch.c_str(), atype, avg);
    ch = trace2()->to_str();
    if(!ch.empty())
		interface()->sendf("F2:DEFINE EQN,'AVG(%s)',AVGTYPE,%s,SWEEPS,%d",
			ch.c_str(), atype, avg);
}

void
XLecroyDSO::onSingleChanged(const shared_ptr<XValueNodeBase> &node) {
	onAverageChanged(node);
}
void
XLecroyDSO::onTrigSourceChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("TRIG_SELECT ,,%s", trigSource()->to_str().c_str());
}
void
XLecroyDSO::onTrigPosChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("%s:TRIG_DELAY %f PCT", trigSource()->to_str().c_str(), (double)*trigPos());
}
void
XLecroyDSO::onTrigLevelChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("%s:TRIG_LEVEL %g V", trigSource()->to_str().c_str(), (double)*trigLevel());
}
void
XLecroyDSO::onTrigFallingChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("%s:TRIG_SLOPE %s", trigSource()->to_str().c_str(), (*trigFalling() ? "NEG" : "POS"));
}
void
XLecroyDSO::onTimeWidthChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("TIME_DIV %.1g", (double)*timeWidth()/10.0);
}
void
XLecroyDSO::onVFullScale1Changed(const shared_ptr<XValueNodeBase> &) {
    std::string ch = trace1()->to_str();
	if(ch.empty()) return;
	interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(vFullScale1()->to_str().c_str())/10.0);
}
void
XLecroyDSO::onVFullScale2Changed(const shared_ptr<XValueNodeBase> &) {
    std::string ch = trace2()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(vFullScale2()->to_str().c_str())/10.0);
}
void
XLecroyDSO::onVOffset1Changed(const shared_ptr<XValueNodeBase> &) {
    std::string ch = trace1()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)*vOffset1());
}
void
XLecroyDSO::onVOffset2Changed(const shared_ptr<XValueNodeBase> &) {
    std::string ch = trace2()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)*vOffset2());
}
void
XLecroyDSO::onRecordLengthChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("MEMORY_SIZE %s",
					  recordLength()->to_str().c_str());
}
void
XLecroyDSO::onForceTriggerTouched(const shared_ptr<XNode> &) {
	interface()->send("ARM:FORCE_TRIGER");
}

void
XLecroyDSO::startSequence() {
	interface()->send("ARM:CLEAR_SWEEPS");
}

int
XLecroyDSO::acqCount(bool *seq_busy) {
	unsigned int n = lrint(inspectDouble("SWEEPS_PER_ACQ", "F1"));
	*seq_busy = (n < *average());
	return n;
}

double
XLecroyDSO::inspectDouble(const char *req, const std::string &trace) {
	interface()->queryf("%s:INSPECT? '%s'", trace.c_str(), req);
	double x;
	interface()->scanf("\"%*s : %lf", &x);
	return x;
}

double
XLecroyDSO::getTimeInterval() {
	return inspectDouble("HORIZ_INTERVAL", trace1()->to_str());
}

void
XLecroyDSO::getWave(std::deque<std::string> &channels)
{
	XScopedLock<XInterface> lock(*interface());
	for(unsigned int i = 0; i < channels.size(); i++) {
		interface()->sendf("F%u:WAVEFORM? ALL", i);
		interface()->receive(10);
		rawData().insert(rawData().end(), 
						 interface()->buffer().begin(), interface()->buffer().end());
		int blks;
		interface()->scanf("#9%8u", &blks);
		interface()->receive(blks + 1);
		rawData().insert(rawData().end(), 
						 interface()->buffer().begin(), interface()->buffer().end());
	}
}
void
XLecroyDSO::convertRaw() throw (XRecordError&) {
	
    //MSB first

    /*
    unsigned char *buf = new unsigned char[256];
    unsigned char *s;

    for(;;)
    {
        n = (m + 40 < width) ? 40 : (width - m);
        Send("CFMT IND0,WORD,BIN");
        Send((AnsiString)"WFSU SP,0,NP," +
                n + ",FP," + (pos + m) + ",SN,0");
        Lock();
        Send(Channel + ":WF? DAT1");
        Receive(buf, 255);
        Unlock();
        s = buf;
        for(int i = 0; i < 100; i++)
        {
            if(buf[i] == '#')
            {
                s = buf + i + 2;
                break;
            }
        }

        for(int i = 0; i < n; i++)
        {
             val = (int)*s * 0x100 +
                ((*s >= 0x80) ? -0x10000 : 0);
             val += *(s + 1);
             if(VGain * val > 0.3)
                val = val + 1;
             (*wbuf)[m + i] = VGain * val - VOffset;
             s += 2;
        }

        m += n;
        if(m >= width) break;
    }
    delete buf;
   }
    return 0;
    */	
}
