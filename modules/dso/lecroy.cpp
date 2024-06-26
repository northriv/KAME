/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
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

REGISTER_TYPE(XDriverList, LecroyDSO, "Lecroy/Teledyne/Iwatsu DSO");

//---------------------------------------------------------------------------
XLecroyDSO::XLecroyDSO(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XDSO>(name, runtime, ref(tr_meas), meas) {
	iterate_commit([=](Transaction &tr){
        for(auto &&x: {trace1(), trace2(), trace3(), trace4()})
            tr[ *x].add({"C1", "C2", "C3", "C4", "M1", "M2", "M3", "M4"});
        for(auto &&x: {vFullScale1(), vFullScale2(), vFullScale3(), vFullScale4()})
            tr[ *x].add({"0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"});
        tr[ *trigSource()].add({"C1", "C2", "C3", "C4", "LINE", "EX", "EX10", "PA", "ETM10"});
    });
//	interface()->setGPIBWaitBeforeWrite(20); //20msec
//    interface()->setGPIBWaitBeforeSPoll(10); //10msec
    interface()->setGPIBUseSerialPollOnRead(false);
    interface()->setGPIBUseSerialPollOnWrite(false);
    interface()->setEOS("\n");

	trans( *recordLength()) = 10000;
}

void
XLecroyDSO::open() {
	interface()->send("COMM_HEADER OFF");
	interface()->send("COMM_FORMAT DEF9,WORD,BIN");
    //LSB first for litte endian.
	interface()->send("COMM_ORDER LO");

    interface()->query("TIME_DIV?");
    trans( *timeWidth()) = interface()->toDouble() * 10.0;

    interface()->query("MEMORY_SIZE?");
    XString str = interface()->toStrSimplified();
    double x = interface()->toDouble();
    if(str.find("MA") != std::string::npos)
        x *= 1e6;
    if(str.find("K") != std::string::npos)
        x *= 1e3;
    trans( *recordLength()) = lrint(x);

    Snapshot shot_this( *this);
    onAverageChanged(shot_this, average().get());

	start();
}
bool
XLecroyDSO::isWaveMaster() {
    interface()->query("*IDN?");
    if(interface()->toStr().find("WAVEMASTER") != std::string::npos) return true;
    char buf[256];
    if(interface()->scanf("LECROY,LT%s", buf) == 1) return false;
    if(interface()->scanf("LECROY,LC%s", buf) == 1) return false;
    int num;
    if(interface()->scanf("LECROY,9%d", &num) == 1) return false;
//    if(interface()->toStr().find("MXI") != std::string::npos) return true;
    return true;
}

void
XLecroyDSO::activateTrace(const char *name) {
    interface()->queryf("%s:TRACE?", name);
    if( !strncmp( &interface()->buffer()[0], "OFF", 2)) {
        interface()->queryf("%s:TRACE ON;*OPC?", name);
        msecsleep(500);
        Snapshot shot_this( *this);
        onAverageChanged(shot_this, average().get());
    }
}

void 
XLecroyDSO::onTrace1Changed(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XInterface> lock( *interface());
    XString ch = ( **trace1())->to_str();
    if( !ch.empty()) {
        activateTrace(ch.c_str());
    }
}
void 
XLecroyDSO::onTrace2Changed(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XInterface> lock( *interface());
    XString ch = ( **trace2())->to_str();
    if( !ch.empty()) {
        activateTrace(ch.c_str());
    }
}
void
XLecroyDSO::onTrace3Changed(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XInterface> lock( *interface());
    XString ch = ( **trace3())->to_str();
    if( !ch.empty()) {
        activateTrace(ch.c_str());
    }
}
void
XLecroyDSO::onTrace4Changed(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XInterface> lock( *interface());
    XString ch = ( **trace4())->to_str();
    if( !ch.empty()) {
        activateTrace(ch.c_str());
    }
}
void
XLecroyDSO::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XInterface> lock( *interface());
	Snapshot shot_this( *this);
	interface()->send("TRIG_MODE STOP");
	int avg = shot_this[ *average()];
	avg = std::max(1, avg);
	bool sseq = shot_this[ *singleSequence()];
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
        bool wavemaster = isWaveMaster();
        const char *atype = sseq ? "SUMMED" : "CONTINUOUS";
        if( !wavemaster)
            atype = sseq ? "AVGS" : "AVGC";
        XString chs[] = {shot_this[ *trace1()].to_str(), shot_this[ *trace2()].to_str(),
                         shot_this[ *trace3()].to_str(), shot_this[ *trace4()].to_str()};
        const char *tchs[] = {"TA", "TB", "TC", "TD"};
//        if(wavemaster)
//            tcshs = {"F1", "F2", "F3", "F4"}; //not nescessary at the moment
        auto tch = tchs;
        for(auto it = chs; it != chs + 4; ++it) {
            if( !it->empty()) {
                if(wavemaster) {
//                    interface()->sendf("%s:DEFINE EQN,'AVG(%s)',AVETYPE,%s,SWEEPS,%d",
                    interface()->sendf("%s:DEFINE EQN,'AVG(%s)',AVERAGETYPE,%s,SWEEPS,%d",
                        *tch, it->c_str(), atype, avg);
                }
                else {
                    interface()->sendf("%s:DEFINE EQN,'%s(%s)',SWEEPS,%d",
                        *tch, atype, it->c_str(), avg);
                }
                interface()->sendf("%s:TRACE ON", *tch);
                tch++;
            }
        }
		interface()->send("TRIG_MODE NORM");
	}
    startSequence();
}

void
XLecroyDSO::onSingleChanged(const Snapshot &shot, XValueNodeBase *node) {
    Snapshot shot_this( *this);
    onAverageChanged(shot_this, average().get());
}
void
XLecroyDSO::onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("TRIG_SELECT EDGE,SR,%s", shot[ *trigSource()].to_str().c_str());
}
void
XLecroyDSO::onTrigPosChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("TRIG_DELAY %fPCT", (double)shot[ *trigPos()]);
}
void
XLecroyDSO::onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
	interface()->sendf("%s:TRIG_LEVEL %gV", shot_this[ *trigSource()].to_str().c_str(), (double)shot_this[ *trigLevel()]);
}
void
XLecroyDSO::onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
	interface()->sendf("%s:TRIG_SLOPE %s", shot_this[ *trigSource()].to_str().c_str(),
		(shot_this[ *trigFalling()] ? "NEG" : "POS"));
}
void
XLecroyDSO::onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("TIME_DIV %.1g", (double)shot[ *timeWidth()] / 10.0);
}
void
XLecroyDSO::onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    XString ch = shot_this[ *trace1()].to_str();
	if(ch.empty()) return;
	interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(shot_this[ *vFullScale1()].to_str().c_str()) / 10.0);
}
void
XLecroyDSO::onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    XString ch = shot_this[ *trace2()].to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(shot_this[ *vFullScale2()].to_str().c_str()) / 10.0);
}
void
XLecroyDSO::onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    XString ch = shot_this[ *trace3()].to_str();
	if(ch.empty()) return;
	interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(shot_this[ *vFullScale3()].to_str().c_str()) / 10.0);
}
void
XLecroyDSO::onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    XString ch = shot_this[ *trace4()].to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:VOLT_DIV %.1g", ch.c_str(), atof(shot_this[ *vFullScale4()].to_str().c_str()) / 10.0);
}
void
XLecroyDSO::onVOffset1Changed(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    XString ch = shot_this[ *trace1()].to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)shot_this[ *vOffset1()]);
}
void
XLecroyDSO::onVOffset2Changed(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    XString ch = shot_this[ *trace2()].to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)shot_this[ *vOffset2()]);
}
void
XLecroyDSO::onVOffset3Changed(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    XString ch = shot_this[ *trace3()].to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)shot_this[ *vOffset3()]);
}
void
XLecroyDSO::onVOffset4Changed(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    XString ch = shot_this[ *trace4()].to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g V", ch.c_str(), (double)shot_this[ *vOffset4()]);
}
void
XLecroyDSO::onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("MEMORY_SIZE %.2g", (double)(unsigned int)shot[ *recordLength()]);
}
void
XLecroyDSO::onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) {
	XScopedLock<XInterface> lock( *interface());
    Snapshot shot_this( *this);
	//	interface()->send("FORCE_TRIGER");
	if((shot_this[ *average()] <= 1) && shot_this[ *singleSequence()]) {
		interface()->send("ARM");
	}
	else {
		interface()->send("TRIG_MODE NORM");
	}
}

void
XLecroyDSO::startSequence() {
	XScopedLock<XInterface> lock( *interface());
    Snapshot shot_this( *this);
    interface()->send("STOP;CLEAR_SWEEPS");
    if((shot_this[ *average()] <= 1) && shot_this[ *singleSequence()])
		interface()->send("ARM");
	else
		interface()->send("TRIG_MODE NORM");
    m_totalCount = 0;
}

int
XLecroyDSO::acqCount(bool *seq_busy) {
    Snapshot shot_this( *this);
    bool sseq = shot_this[ *singleSequence()];
    unsigned int n = 0;
    int avg = shot_this[ *average()];
    avg = std::max(1, avg);
    if( !shot_this[ *trace1()].to_str().empty()) {
        interface()->queryf("%s:TRACE?", shot_this[ *trace1()].to_str().c_str());
        if( !strncmp( &interface()->buffer()[0], "ON", 2)) {
            //trace1 is displayed.
            XString ch = (avg > 1) ? XString("TA") : shot_this[ *trace1()].to_str();
            n = lrint(inspectDouble("SWEEPS_PER_ACQ", ch));
        }
    }
    if( !sseq || (avg < 2)) {
        interface()->query("INR?");
        if(interface()->toInt() & 1) {
            if( !sseq)
                m_totalCount++;
            else
                n = 1;
        }
        if( !sseq)
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
	return inspectDouble("HORIZ_INTERVAL", ( **trace1())->to_str());
}

void
XLecroyDSO::getWave(shared_ptr<RawData> &writer, std::deque<XString> &channels) {
	Snapshot shot( *this);
	XScopedLock<XInterface> lock( *interface());
//	interface()->send("TRIG_MODE STOP");
	try {
		writer->push<uint32_t>(channels.size());
		for(unsigned int i = 0; i < std::min((unsigned int)channels.size(), 4u); i++) {
			XString ch = channels[i];
			if(shot[ *average()] > 1) {
				const char *fch[] = {"TA", "TB", "TC", "TD"};
				ch = fch[i];
			}
			interface()->sendf("%s:WAVEFORM? ALL", ch.c_str());
			msecsleep(50);
			interface()->receive(4); //For "ALL,"
			if(interface()->buffer().size() != 4)
				throw XInterface::XCommError(i18n("Invalid waveform"), __FILE__, __LINE__);
			interface()->setGPIBUseSerialPollOnRead(false);
            interface()->receive(2); //For "#9" or "#A"
			if(interface()->buffer().size() != 2)
				throw XInterface::XCommError(i18n("Invalid waveform"), __FILE__, __LINE__);
			writer->insert(writer->end(),
							 interface()->buffer().begin(), interface()->buffer().end());
			unsigned int n;
            interface()->scanf("#%1x", &n);
			interface()->receive(n);
			if(interface()->buffer().size() != n)
				throw XInterface::XCommError(i18n("Invalid waveform"), __FILE__, __LINE__);
			writer->insert(writer->end(),
							 interface()->buffer().begin(), interface()->buffer().end());
			int blks = interface()->toUInt();
			XTime tstart = XTime::now();
			for(int retry = 0;; retry++) {
				interface()->receive(blks);
				blks -= interface()->buffer().size();
				writer->insert(writer->end(),
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
//			interface()->setGPIBUseSerialPollOnRead(true);
		}
	}
	catch (XInterface::XInterfaceError &e) {
//		interface()->setGPIBUseSerialPollOnRead(true);
		throw e;
	}
//	if(!*singleSequence())
//		interface()->send("TRIG_MODE NORM");
}
void
XLecroyDSO::convertRaw(RawDataReader &reader, Transaction &tr) {
		Snapshot &shot(tr);

#define WAVEDESC_WAVE_ARRAY_COUNT 116
#define DATA_BLOCK 346
	
	unsigned int ch_cnt = reader.pop<uint32_t>();
	for(unsigned int ch = 0; ch < ch_cnt; ch++) {
		std::vector<char>::const_iterator dit = reader.popIterator();
		unsigned int n;
		sscanf( &*dit, "#%1u", &n);
		dit += n + 2;
		if(strncmp( &*dit, "WAVEDESC", 8)) {
			throw XRecordError(i18n("Invalid waveform"), __FILE__, __LINE__);
		}
		dit += DATA_BLOCK;
		reader.popIterator() += WAVEDESC_WAVE_ARRAY_COUNT + n + 2;
		int32_t count = reader.pop<int32_t>();
		reader.pop<int32_t>();
		int32_t first_valid = reader.pop<int32_t>();
		int32_t last_valid = reader.pop<int32_t>();
		int32_t first = reader.pop<int32_t>();
		reader.pop<int32_t>();
		reader.pop<int32_t>();
		reader.pop<int32_t>();
		int32_t acqcount = reader.pop<int32_t>();
		reader.pop<int16_t>();
		reader.pop<int16_t>();
		float vgain = reader.pop<float>();
		float voffset = reader.pop<float>();
		reader.pop<float>();
		reader.pop<float>();
		reader.pop<int16_t>();
		reader.pop<int16_t>();
		float interval = reader.pop<float>();
		double hoffset = reader.pop<double>();
		
		fprintf(stderr, "first_valid=%d,last_valid=%d,first=%d,acqcount=%d,count=%d\n",
			(int)first_valid, (int)last_valid, (int)first, (int)acqcount, (int)count);
		if(ch == 0) {
			if((count < 0) || 
				(reader.size() < (count * 2 + DATA_BLOCK + n + 2) * ch_cnt))
				throw XBufferUnderflowRecordError(__FILE__, __LINE__);
			tr[ *this].setParameters(ch_cnt, hoffset, interval, count);
		}
		
		double *wave = tr[ *this].waveDisp(ch);
		reader.popIterator() = dit;
		int length = shot[ *this].lengthDisp();
		for(int i = 0; i < std::min(count, (int32_t)length); i++) {
			int16_t x = reader.pop<int16_t>();
			float v = voffset + vgain * x;
			*wave++ = v;
		}
	}
}
