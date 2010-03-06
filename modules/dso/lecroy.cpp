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
#include "lecroy.h"
#include "charinterface.h"
#include "xwavengraph.h"

REGISTER_TYPE(XDriverList, LecroyDSO, "Lecroy/Iwatsu DSO");

//---------------------------------------------------------------------------
XLecroyDSO::XLecroyDSO(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XDSO>(name, runtime, ref(tr_meas), meas) {
	const char* ch[] = {"C1", "C2", "C3", "C4", "M1", "M2", "M3", "M4", 0L};
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
	const char* tr[] = {"C1", "C2", "C3", "C4", "LINE", "EX", "EX10", "PA", "ETM10", 0L};
	for(int i = 0; tr[i]; i++) {
		trigSource()->add(tr[i]);
	}

	interface()->setGPIBWaitBeforeWrite(20); //20msec
	interface()->setGPIBWaitBeforeSPoll(10); //10msec
	interface()->setEOS("\n");

	recordLength()->value(10000);
}

void
XLecroyDSO::open() throw (XInterface::XInterfaceError &) {
	interface()->send("COMM_HEADER OFF");
	interface()->send("COMM_FORMAT DEF9,WORD,BIN");
    //LSB first for litte endian.
	interface()->send("COMM_ORDER LO");
	onAverageChanged(average());

	start();
}
void 
XLecroyDSO::onTrace1Changed(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XInterface> lock(*interface());
    XString ch = trace1()->to_str();
    if(!ch.empty()) {
		interface()->sendf("%s:TRACE ON", ch.c_str());
    }
    onAverageChanged(average());
}
void 
XLecroyDSO::onTrace2Changed(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XInterface> lock(*interface());
    XString ch = trace2()->to_str();
    if(!ch.empty()) {
		interface()->sendf("%s:TRACE ON", ch.c_str());
    }
    onAverageChanged(average());
}
void
XLecroyDSO::onTrace3Changed(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XInterface> lock(*interface());
    XString ch = trace3()->to_str();
    if(!ch.empty()) {
		interface()->sendf("%s:TRACE ON", ch.c_str());
    }
    onAverageChanged(average());
}
void
XLecroyDSO::onTrace4Changed(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XInterface> lock(*interface());
    XString ch = trace4()->to_str();
    if(!ch.empty()) {
		interface()->sendf("%s:TRACE ON", ch.c_str());
    }
    onAverageChanged(average());
}
void
XLecroyDSO::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XInterface> lock(*interface());
	interface()->send("TRIG_MODE STOP");
	int avg = *average();
	avg = std::max(1, avg);
	bool sseq = *singleSequence();
	if(avg == 1) {
//		interface()->send("F1:TRACE OFF");
//		interface()->send("F2:TRACE OFF");
		if(sseq) {
			interface()->send("ARM");
		}
		else {
			interface()->send("TRIG_MODE NORM");			
		}
	}
	else {
//		const char *atype = sseq ? "SUMMED" : "CONTINUOUS";
		const char *atype = sseq ? "AVGS" : "AVGC";
	    XString ch = trace1()->to_str();
	    if(!ch.empty()) {
//			interface()->sendf("TA:DEFINE EQN,'AVG(%s)',AVGTYPE,%s,SWEEPS,%d",
//				ch.c_str(), atype, avg);
			interface()->sendf("TA:DEFINE EQN,'%s(%s)',SWEEPS,%d",
				atype, ch.c_str(), avg);
			interface()->send("TA:TRACE ON");
	    }
	    ch = trace2()->to_str();
	    if(!ch.empty()) {
			interface()->sendf("TB:DEFINE EQN,'%s(%s)',SWEEPS,%d",
				atype, ch.c_str(), avg);
			interface()->send("TB:TRACE ON");
	    }
		interface()->send("TRIG_MODE NORM");
	    ch = trace3()->to_str();
	    if(!ch.empty()) {
			interface()->sendf("TB:DEFINE EQN,'%s(%s)',SWEEPS,%d",
				atype, ch.c_str(), avg);
			interface()->send("TB:TRACE ON");
	    }
		interface()->send("TRIG_MODE NORM");
	    ch = trace4()->to_str();
	    if(!ch.empty()) {
			interface()->sendf("TB:DEFINE EQN,'%s(%s)',SWEEPS,%d",
				atype, ch.c_str(), avg);
			interface()->send("TB:TRACE ON");
	    }
		interface()->send("TRIG_MODE NORM");
	}
}

void
XLecroyDSO::onSingleChanged(const shared_ptr<XValueNodeBase> &node) {
	onAverageChanged(node);
}
void
XLecroyDSO::onTrigSourceChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("TRIG_SELECT EDGE,SR,%s", trigSource()->to_str().c_str());
}
void
XLecroyDSO::onTrigPosChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("TRIG_DELAY %fPCT", (double)*trigPos());
}
void
XLecroyDSO::onTrigLevelChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("%s:TRIG_LEVEL %gV", trigSource()->to_str().c_str(), (double)*trigLevel());
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
    XString ch = trace1()->to_str();
	if(ch.empty()) return;
	interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(vFullScale1()->to_str().c_str())/10.0);
}
void
XLecroyDSO::onVFullScale2Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace2()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(vFullScale2()->to_str().c_str())/10.0);
}
void
XLecroyDSO::onVFullScale3Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace3()->to_str();
	if(ch.empty()) return;
	interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(vFullScale3()->to_str().c_str())/10.0);
}
void
XLecroyDSO::onVFullScale4Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace4()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(vFullScale4()->to_str().c_str())/10.0);
}
void
XLecroyDSO::onVOffset1Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace1()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)*vOffset1());
}
void
XLecroyDSO::onVOffset2Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace2()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)*vOffset2());
}
void
XLecroyDSO::onVOffset3Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace3()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)*vOffset3());
}
void
XLecroyDSO::onVOffset4Changed(const shared_ptr<XValueNodeBase> &) {
    XString ch = trace4()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)*vOffset4());
}
void
XLecroyDSO::onRecordLengthChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("MEMORY_SIZE %s",
					  recordLength()->to_str().c_str());
}
void
XLecroyDSO::onForceTriggerTouched(const shared_ptr<XNode> &) {
	XScopedLock<XInterface> lock(*interface());
	//	interface()->send("FORCE_TRIGER");
	if((*average() <= 1) && *singleSequence()) {
		interface()->send("ARM");
	}
	else {
		interface()->send("TRIG_MODE NORM");
	}
}

void
XLecroyDSO::startSequence() {
	XScopedLock<XInterface> lock(*interface());
	if((*average() <= 1) && *singleSequence())
		interface()->send("ARM");
	else
		interface()->send("TRIG_MODE NORM");
	interface()->send("CLEAR_SWEEPS");
}

int
XLecroyDSO::acqCount(bool *seq_busy) {
	bool sseq = *singleSequence();
	unsigned int n = 0;
	int avg = *average();
	avg = std::max(1, avg);
	if(!trace1()->to_str().empty()) {
		interface()->queryf("%s:TRACE?", trace1()->to_str().c_str());
		if(!strncmp(&interface()->buffer()[0], "ON", 2)) {
			//trace1 is displayed.
			XString ch = (avg > 1) ? XString("TA") : trace1()->to_str();
			n = lrint(inspectDouble("SWEEPS_PER_ACQ", ch));
		}
	}
	if(!sseq || (avg < 2)) {
		interface()->query("INR?");
		if(interface()->toInt() & 1) {
			if(!sseq)
				m_totalCount++;
			else
				n = 1;
		}
		if(!sseq)
			n = m_totalCount;
	}
	*seq_busy = (n < avg);
	return n;
}

double
XLecroyDSO::inspectDouble(const char *req, const XString &trace) {
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
XLecroyDSO::getWave(std::deque<XString> &channels)
{
	XScopedLock<XInterface> lock(*interface());
//	interface()->send("TRIG_MODE STOP");
	try {
		push<unsigned int32_t>(channels.size());
		for(unsigned int i = 0; i < std::min((unsigned int)channels.size(), 4u); i++) {
			XString ch = channels[i];
			if(*average() > 1) {
				const char *fch[] = {"TA", "TB", "TC", "TD"};
				ch = fch[i];
			}
			interface()->sendf("%s:WAVEFORM? ALL", ch.c_str());
			msecsleep(50);
			interface()->receive(4); //For "ALL,"
			if(interface()->buffer().size() != 4)
				throw XInterface::XCommError(i18n("Invalid waveform"), __FILE__, __LINE__);
			interface()->setGPIBUseSerialPollOnRead(false);
			interface()->receive(2); //For "#9"
			if(interface()->buffer().size() != 2)
				throw XInterface::XCommError(i18n("Invalid waveform"), __FILE__, __LINE__);
			rawData().insert(rawData().end(), 
							 interface()->buffer().begin(), interface()->buffer().end());
			unsigned int n;
			interface()->scanf("#%1u", &n);
			interface()->receive(n);
			if(interface()->buffer().size() != n)
				throw XInterface::XCommError(i18n("Invalid waveform"), __FILE__, __LINE__);
			rawData().insert(rawData().end(), 
							 interface()->buffer().begin(), interface()->buffer().end());
			int blks = interface()->toUInt();
			XTime tstart = XTime::now();
			for(int retry = 0;; retry++) {
				interface()->receive(blks);
				blks -= interface()->buffer().size();
				rawData().insert(rawData().end(), 
								 interface()->buffer().begin(), interface()->buffer().end());
				if(blks <= 0)
					break;
				if(XTime::now() - tstart > 3.0)
					break; //timeout.
				msecsleep(20);
			}
			if(blks != 0)
				throw XInterface::XCommError(i18n("Invalid waveform"), __FILE__, __LINE__);
			interface()->receive(); //For LF.
			interface()->setGPIBUseSerialPollOnRead(true);
		}
	}
	catch (XInterface::XInterfaceError &e) {
		interface()->setGPIBUseSerialPollOnRead(true);
		throw e;
	}
//	if(!*singleSequence())
//		interface()->send("TRIG_MODE NORM");
}
void
XLecroyDSO::convertRaw() throw (XRecordError&) {
#define WAVEDESC_WAVE_ARRAY_COUNT 116
#define DATA_BLOCK 346
	
	unsigned int ch_cnt = pop<unsigned int32_t>();
	for(unsigned int ch = 0; ch < ch_cnt; ch++) {
		std::vector<char>::iterator dit = rawDataPopIterator();
		unsigned int n;
		sscanf(&*dit, "#%1u", &n);
		dit += n + 2;
		if(strncmp(&*dit, "WAVEDESC", 8)) {
			throw XRecordError(i18n("Invalid waveform"), __FILE__, __LINE__);
		}
		dit += DATA_BLOCK;
		rawDataPopIterator() += WAVEDESC_WAVE_ARRAY_COUNT + n + 2;
		int32_t count = pop<int32_t>();
		pop<int32_t>();
		int32_t first_valid = pop<int32_t>();
		int32_t last_valid = pop<int32_t>();
		int32_t first = pop<int32_t>();
		pop<int32_t>();
		pop<int32_t>();
		pop<int32_t>();
		int32_t acqcount = pop<int32_t>();
		pop<int16_t>();
		pop<int16_t>();
		float vgain = pop<float>();
		float voffset = pop<float>();
		pop<float>();
		pop<float>();
		pop<int16_t>();
		pop<int16_t>();
		float interval = pop<float>();
		double hoffset = pop<double>();
		
		fprintf(stderr, "first_valid=%d,last_valid=%d,first=%d,acqcount=%d,count=%d\n",
			(int)first_valid, (int)last_valid, (int)first, (int)acqcount, (int)count);
		if(ch == 0) {
			if((count < 0) || 
				(rawData().size() < (count * 2 + DATA_BLOCK + n + 2) * ch_cnt))
				throw XBufferUnderflowRecordError(__FILE__, __LINE__);
			setParameters(ch_cnt, hoffset, interval, count);
		}
		
		double *wave = waveDisp(ch);
		rawDataPopIterator() = dit;
		for(int i = 0; i < std::min(count, (int32_t)lengthDisp()); i++) {
			int16_t x = pop<int16_t>();
			float v = voffset + vgain * x;
			*wave++ = v;
		}
	}
}
