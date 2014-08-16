/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "thamwaydso.h"

#include "xwavengraph.h"

REGISTER_TYPE(XDriverList, ThamwayDVUSBDSO, "Thamway DV14U25 A/D conversion board");

#define NUM_MAX_CH 2
#define INTERNAL_CLOCK 25e6 //Hz

#define ADDR_STS 0
#define ADDR_CTRL 1
#define ADDR_CFG 2
#define ADDR_DIVISOR 3
#define ADDR_SAMPLES_LSW 4
#define ADDR_SAMPLES_MSW 6
#define ADDR_AVGM1_LSW 0x8 //average count minus 1
#define ADDR_AVGM1_MSW 0xa
#define ADDR_ACQCNTM1_LSW 0x8 //acquision count minus 1
#define ADDR_ACQCNTM1_MSW 0xa
#define ADDR_FRAMESM1 0xc //# of frames minus 1

#define ADDR_BURST_CH1 0x18
#define ADDR_BURST_CH2 0x19
#define ADDR_CH1_SET_MEM_ADDR_LSW 0x10
#define ADDR_CH1_SET_MEM_ADDR_MSW 0x12
#define ADDR_CH2_SET_MEM_ADDR_LSW 0x14
#define ADDR_CH2_SET_MEM_ADDR_MSW 0x16
#define ADDR_CH1_GET_MEM_ADDR_LSW 0x4
#define ADDR_CH1_GET_MEM_ADDR_MSW 0x6
#define ADDR_CH2_GET_MEM_ADDR_LSW 0x10
#define ADDR_CH2_GET_MEM_ADDR_MSW 0x12

#define MAX_SMPL 0x80000u //512kwords

XThamwayDVUSBDSO::XThamwayDVUSBDSO(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XDSO, XWinCUSBInterface>(name, runtime, ref(tr_meas), meas) {

    const char* sc[] = {"5", 0L};
    for(Transaction tr( *this);; ++tr) {
        tr[ *recordLength()] = 2000;
        tr[ *timeWidth()] = 1e-2;
        tr[ *average()] = 1;
        for(int i = 0; sc[i]; i++) {
            tr[ *vFullScale1()].add(sc[i]);
            tr[ *vFullScale2()].add(sc[i]);
        }
        tr[ *vFullScale1()] = "5";
        tr[ *vFullScale2()] = "5";
        if(tr.commit())
            break;
    }

    trace1()->disable();
    trace2()->disable();
    vFullScale3()->disable();
    vFullScale4()->disable();
    vOffset1()->disable();
    vOffset2()->disable();
    vOffset3()->disable();
    vOffset4()->disable();
    trigPos()->disable();
    trigSource()->disable();
    trigLevel()->disable();
    trigFalling()->disable();
}
XThamwayDVUSBDSO::~XThamwayDVUSBDSO() {
}
void
XThamwayDVUSBDSO::open() throw (XKameError &) {
    XScopedLock<XInterface> lock( *interface());
    XString idn = interface()->getIDN();
    gWarnPrint("Pulser IDN=" + idn);

    int smps = interface()->readRegister16(ADDR_SAMPLES_MSW);
    smps = smps * 0x10000L * smps + interface()->readRegister16(ADDR_SAMPLES_LSW);
    int avg = acqCount(0);
    double intv = getTimeInterval();
    for(Transaction tr( *this);; ++tr) {
        tr[ *recordLength()] = smps;
        tr[ *timeWidth()] = smps * intv;
        tr[ *average()] = avg;
        if(tr.commit())
            break;
    }
    interface()->writeToRegister8(ADDR_FRAMESM1, 0); //1 frame.

    this->start();
}
void
XThamwayDVUSBDSO::close() throw (XKameError &) {
    interface()->stop();
}
void
XThamwayDVUSBDSO::onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) {
//    XScopedLock<XInterface> lock( *interface());
//    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.

//    interface()->writeToRegister8(ADDR_CTRL, 1); //starts.
}

void
XThamwayDVUSBDSO::startSequence() {
    XScopedLock<XInterface> lock( *interface());
    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.
    interface()->writeToRegister8(ADDR_CTRL, 1); //starts.
}

int
XThamwayDVUSBDSO::acqCount(bool *seq_busy) {
    XScopedLock<XInterface> lock( *interface());
    if(seq_busy)
        *seq_busy = interface()->singleRead(ADDR_STS) & 4;
    int acq = interface()->readRegister16(ADDR_ACQCNTM1_MSW);
    acq = acq * 0x10000L + interface()->readRegister16(ADDR_ACQCNTM1_LSW);
    acq++;
    return acq;
}

double
XThamwayDVUSBDSO::getTimeInterval() {
    XScopedLock<XInterface> lock( *interface());
    int div = interface()->singleRead(ADDR_DIVISOR);
    int pres = interface()->singleRead(ADDR_CFG) % 0x8;
    double clk = INTERNAL_CLOCK / pow(2.0, pres) / div;
    return 1.0/clk;
}

void
XThamwayDVUSBDSO::getWave(shared_ptr<RawData> &writer, std::deque<XString> &) {
    XScopedLock<XInterface> lock( *interface());

    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.
    interface()->writeToRegister16(ADDR_CH1_GET_MEM_ADDR_LSW, 0);
    interface()->writeToRegister16(ADDR_CH1_GET_MEM_ADDR_MSW, 0);
    interface()->writeToRegister16(ADDR_CH2_GET_MEM_ADDR_LSW, 0);
    interface()->writeToRegister16(ADDR_CH2_GET_MEM_ADDR_MSW, 0);

    int smps = interface()->readRegister16(ADDR_SAMPLES_MSW);
    smps = smps * 0x10000L * smps + interface()->readRegister16(ADDR_SAMPLES_LSW);

    writer->push((uint16_t)2); //channels
    writer->push((uint32_t)0); //reserve
    writer->push((uint32_t)0); //reserve
    writer->push((uint32_t)acqCount(0));
    writer->push((uint32_t)smps);
    writer->push((double)getTimeInterval());
    std::vector<uint8_t> buf(smps);
    //Ch1
    interface()->burstRead(ADDR_BURST_CH1, &buf[0], smps);
    writer->push((double)1.0/3276.8); //[V/bit]
    writer->push((double)-2.5); //offset[V]
    writer->insert(writer->end(), buf.begin(), buf.end());
    //Ch2
    buf.clear();
    interface()->burstRead(ADDR_BURST_CH2, &buf[0], smps);
    writer->push((double)1.0/3276.8); //[V/bit]
    writer->push((double)-2.5); //offset[V]
    writer->insert(writer->end(), buf.begin(), buf.end());
}
void
XThamwayDVUSBDSO::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    const unsigned int num_ch = reader.pop<uint16_t>();
    reader.pop<uint32_t>(); //reserve
    reader.pop<uint32_t>(); //reserve
    unsigned int accumCount = reader.pop<uint32_t>();
    unsigned int len = reader.pop<uint32_t>();
    double interval = reader.pop<double>();

    tr[ *this].setParameters(num_ch, 0.0, interval, len);

    for(unsigned int j = 0; j < num_ch; j++) {
        double prop = reader.pop<double>();
        double offset = reader.pop<double>();
        double *wave = tr[ *this].waveDisp(j);
        for(unsigned int i = 0; i < len; i++) {
            *wave++ = prop * reader.pop<uint32_t>() + offset;
        }
    }
}

void
XThamwayDVUSBDSO::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *interface());
    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.

    unsigned int avg = shot[ *average()];
    avg--;
    interface()->writeToRegister16(ADDR_AVGM1_LSW, avg % 0x10000uL);
    interface()->writeToRegister16(ADDR_AVGM1_MSW, avg / 0x10000uL);

    interface()->writeToRegister8(ADDR_CTRL, 1); //starts.
}

void
XThamwayDVUSBDSO::onSingleChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onTrigPosChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *interface());
    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.

    int smps = interface()->readRegister16(ADDR_SAMPLES_MSW);
    smps = smps * 0x10000L * smps + interface()->readRegister16(ADDR_SAMPLES_LSW);

    double interval = shot[ *timeWidth()] / smps;
    int div = std::max(1L, lrint(INTERNAL_CLOCK * interval));
    int pres = std::min(7, std::max(0, (int)floor(log(div / 256.0) / log(2.0)) + 1));

    uint8_t cfg = interface()->readRegister8(ADDR_CFG);
    interface()->writeToRegister8(ADDR_CFG, cfg | pres);

    div = std::min(255L, lrint(div / pow(2.0, pres)));
    interface()->writeToRegister8(ADDR_DIVISOR, div);

    interface()->writeToRegister8(ADDR_CTRL, 1); //starts.
}
void
XThamwayDVUSBDSO::onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *interface());
    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.

    unsigned int smps = shot[ *recordLength()];
    interface()->writeToRegister16(ADDR_SAMPLES_LSW, smps % 0x10000uL);
    interface()->writeToRegister16(ADDR_SAMPLES_MSW, smps / 0x10000uL);

    interface()->writeToRegister8(ADDR_CTRL, 1); //starts.
}
void
XThamwayDVUSBDSO::onTrace1Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onTrace2Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onTrace3Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onTrace4Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onVOffset1Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onVOffset2Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onVOffset3Changed(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayDVUSBDSO::onVOffset4Changed(const Snapshot &shot, XValueNodeBase *) {
}
