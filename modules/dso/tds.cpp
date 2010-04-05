/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "tds.h"
#include "charinterface.h"
#include "xwavengraph.h"

REGISTER_TYPE(XDriverList, TDS, "Tektronix DSO");

//---------------------------------------------------------------------------
XTDS::XTDS(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XDSO>(name, runtime, ref(tr_meas), meas) {
	const char* ch[] = {"CH1", "CH2", "CH3", "CH4", "MATH1", "MATH2", 0L};
	for(int i = 0; ch[i]; i++) {
		trace1()->add(ch[i]);
		trace2()->add(ch[i]);
		trace3()->add(ch[i]);
		trace4()->add(ch[i]);
	}
	const char* sc[] = {"0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10",
						"20", "50", "100", 0L};
	for(int i = 0; sc[i]; i++) {
		vFullScale1()->add(sc[i]);
		vFullScale2()->add(sc[i]);
		vFullScale3()->add(sc[i]);
		vFullScale4()->add(sc[i]);
	}
	const char* tr[] = {"EXT", "EXT10", "CH1", "CH2", "CH3", "CH4", "LINE", 0L};
	for(int i = 0; tr[i]; i++) {
		trigSource()->add(tr[i]);
	}

	interface()->setGPIBWaitBeforeWrite(20); //20msec
	interface()->setGPIBWaitBeforeSPoll(10); //10msec

	recordLength()->value(10000);
}

void
XTDS::open() throw (XInterface::XInterfaceError &) {
	interface()->send("HEADER ON");
	interface()->query("ACQ:STOPAFTER?");
	char buf[10];
	if(interface()->scanf(":ACQ%*s %9s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	singleSequence()->value(!strncmp(buf, "SEQ", 3));
  
	interface()->query("ACQ:MODE?");
	if(interface()->scanf(":ACQ%*s %9s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	if( !strncmp(buf, "AVE", 3)) {
		interface()->query("ACQ:NUMAVG?");
		int x;
		if(interface()->scanf(":ACQ%*s %d", &x) != 1)
			throw XInterface::XConvError(__FILE__, __LINE__);
		average()->value(x);
	}
	if( !strncmp(buf, "SAM", 3))
		average()->value(1);
	interface()->send("DATA:ENC RPB;WIDTH 2"); //MSB first RIB
  
	start();
}
void 
XTDS::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
	if( *average() == 1) {
		interface()->send("ACQ:MODE SAMPLE");
	}
	else {
		interface()->send("ACQ:MODE AVE;NUMAVG " + average()->to_str());
	}
}

void
XTDS::onSingleChanged(const shared_ptr<XValueNodeBase> &) {
	if( *singleSequence()) {
		interface()->send("ACQ:STOPAFTER SEQUENCE;STATE ON");
	}
	else {
		interface()->send("ACQ:STOPAFTER RUNSTOP;STATE ON");
	}
}
void
XTDS::onTrigSourceChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->send("TRIG:A:EDG:SOU " + trigSource()->to_str());
}
void
XTDS::onTrigPosChanged(const shared_ptr<XValueNodeBase> &) {
    if( *trigPos() >= 0)
		interface()->sendf("HOR:DELAY:STATE OFF;TIME %.2g", (double)*trigPos());
    else
		interface()->sendf("HOR:DELAY:STATE ON;TIME %.2g", -( *trigPos() - 50.0)/100.0* *timeWidth());
}
void
XTDS::onTrigLevelChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("TRIG:A:EDG:LEV %g", (double)*trigLevel());
}
void
XTDS::onTrigFallingChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("TRIG:A:EDG:SLOP %s", (*trigFalling() ? "FALL" : "RISE"));
}
void
XTDS::onTimeWidthChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("HOR:MAIN:SCALE %.1g", (double)*timeWidth()/10.0);
}
void
XTDS::onVFullScale1Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace1()->to_str();
	if(ch.empty()) return;
	interface()->sendf("%s:SCALE %.1g", ch.c_str(), atof(vFullScale1()->to_str().c_str())/10.0);
}
void
XTDS::onVFullScale2Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace2()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:SCALE %.1g", ch.c_str(), atof(vFullScale2()->to_str().c_str())/10.0);
}
void
XTDS::onVFullScale3Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace3()->to_str();
	if(ch.empty()) return;
	interface()->sendf("%s:SCALE %.1g", ch.c_str(), atof(vFullScale3()->to_str().c_str())/10.0);
}
void
XTDS::onVFullScale4Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace4()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:SCALE %.1g", ch.c_str(), atof(vFullScale4()->to_str().c_str())/10.0);
}
void
XTDS::onVOffset1Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace1()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g", ch.c_str(), (double)*vOffset1());
}
void
XTDS::onVOffset2Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace2()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g", ch.c_str(), (double)*vOffset2());
}
void
XTDS::onVOffset3Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace3()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g", ch.c_str(), (double)*vOffset3());
}
void
XTDS::onVOffset4Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace4()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g", ch.c_str(), (double)*vOffset4());
}
void
XTDS::onRecordLengthChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->send("HOR:RECORD " + 
					  recordLength()->to_str());
}
void
XTDS::onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) {
	interface()->send("TRIG FORC");
}

void
XTDS::startSequence() {
	interface()->send("ACQ:STATE ON");
}

int
XTDS::acqCount(bool *seq_busy) {
	interface()->query("ACQ:NUMACQ?;:BUSY?");
	int n;
	int busy;
	if(interface()->scanf(":ACQ%*s %d;:BUSY %d", &n, &busy) != 2)
		throw XInterface::XConvError(__FILE__, __LINE__);
	*seq_busy = busy;
	return n;
}

double
XTDS::getTimeInterval() {
	interface()->query("WFMP?");
	char *cp = strstr(&interface()->buffer()[0], "XIN");
	if( !cp) throw XInterface::XConvError(__FILE__, __LINE__);
	double x;
	int ret = sscanf(cp, "%*s %lf", &x);
	if(ret != 1) throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}

void
XTDS::getWave(shared_ptr<RawData> &writer, std::deque<XString> &channels) {
	XScopedLock<XInterface> lock( *interface());
	int pos = 1;
	int width = 20000;
	for(std::deque<XString>::iterator it = channels.begin(); it != channels.end(); it++) {
		int rsize = (2 * width + 1024);
		interface()->sendf("DATA:SOURCE %s;START %u;STOP %u;:WAVF?",
						   (const char *)it->c_str(),
						   pos, pos + width);
		interface()->receive(rsize);
		writer->insert(writer->end(),
						 interface()->buffer().begin(), interface()->buffer().end());
	}
}
void
XTDS::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	Snapshot &shot(tr);

	double xin = 0;
	double yin[256], yoff[256];
	int width = 0;
	double xoff = 0;
	int triggerpos;

	int size = reader.size();
	std::vector<char> bufcpy(reader.data());
	bufcpy.push_back('\0');
	char *buf = &bufcpy[0];
  
	int ch_cnt = 0;
	//scan # of channels etc.
	char *cp = buf;
	for(;;) {
		if(cp >= &buf[size]) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
		if( *cp == ':') cp++;
		if( !strncasecmp(cp, "XIN", 3))
			sscanf(cp, "%*s %lf", &xin);
		if( !strncasecmp(cp, "PT_O", 4))
			sscanf(cp, "%*s %d", &triggerpos);
		if( !strncasecmp(cp, "XZE", 3))
			sscanf(cp, "%*s %lf", &xoff);
		if( !strncasecmp(cp, "YMU", 3))
			sscanf(cp, "%*s %lf", &yin[ch_cnt - 1]);
		if( !strncasecmp(cp, "YOF", 3))
			sscanf(cp, "%*s %lf", &yoff[ch_cnt - 1]);
		if( !strncasecmp(cp, "NR_P", 4)) {
			ch_cnt++;
			sscanf(cp, "%*s %d", &width);
		}
		if( !strncasecmp(cp, "CURV", 4)) {
			for(;;) {
				cp = index(cp, '#');
				if( !cp) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
				int x;
				if(sscanf(cp, "#%1d", &x) != 1) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
				char fmt[9];
				if(snprintf(fmt, sizeof(fmt), "#%%*1d%%%ud", x) < 0)
					throw XBufferUnderflowRecordError(__FILE__, __LINE__);
				int yyy;
				if(sscanf(cp, fmt, &yyy) != 1) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
				if(yyy == 0) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
				cp += 2 + x;
           
				cp += yyy;
				if( *cp != ',') break;
			}
		}
		char *ncp = index(cp, ';');
		if( !ncp)
			cp = index(cp, ':');
		else
			cp = ncp;
		if( !cp) break;
		cp++;
	}
	if((width <= 0) || (width > size / 2)) throw XBufferUnderflowRecordError(__FILE__, __LINE__);

	if(triggerpos != 0)
		xoff = -triggerpos * xin;

	tr[ *this].setParameters(ch_cnt, xoff, xin, width);
  
	cp = buf;
	for(int j = 0; j < ch_cnt; j++) {
		double *wave = tr[ *this].waveDisp(j);
		cp = index(cp, '#');
		if( !cp) break;
		int x;
		if(sscanf(cp, "#%1d", &x) != 1) break;
		char fmt[9];
		if(snprintf(fmt, sizeof(fmt), "#%%*1d%%%ud", x) < 0)
			throw XBufferUnderflowRecordError(__FILE__, __LINE__);
		int yyy;
		if(sscanf(cp, fmt, &yyy) != 1) break;
		if(yyy == 0) break;
		cp += 2 + x;

		int i = 0;
		for(; i < std::min(width, yyy/2); i++) {
			double val = *((unsigned char *)cp) * 0x100;
			val += *((unsigned char *)cp + 1);
			*(wave++) = yin[j] * (val - yoff[j] - 0.5);
			cp += 2;
		}
		for(; i < width; i++) {
			*(wave++) = 0.0;
		}
	}  
}
