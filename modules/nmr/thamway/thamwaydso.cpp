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
//#define ADDR_CH1_GET_MEM_ADDR_LSW 0x4
//#define ADDR_CH1_GET_MEM_ADDR_MSW 0x6
//#define ADDR_CH2_GET_MEM_ADDR_LSW 0x10
//#define ADDR_CH2_GET_MEM_ADDR_MSW 0x12

#define MAX_SMPL 0x80000u //512kwords
#define EXTRA_SMPL 1000u

XThamwayDVUSBDSO::XThamwayDVUSBDSO(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XDSO, XThamwayDVCUSBInterface>(name, runtime, ref(tr_meas), meas) {

    std::vector<shared_ptr<XNode>> unnecessary_ui{
        vFullScale3(), vFullScale4(), vOffset1(), vOffset2(), vOffset3(), vOffset4(),
        trigPos(), trigSource(), trigLevel(), trigFalling(), forceTrigger(),
        fetchMode()
    };

    iterate_commit([=](Transaction &tr){
        tr[ *recordLength()] = 2000;
        tr[ *timeWidth()] = 1e-2;
        tr[ *fetchMode()] = "Sequence";
        for(auto &&x: {vFullScale1(), vFullScale2()}) {
            tr[ *x].add({"5"});
            tr[ *x] = "5";
        }
        for(auto &&x: {trace1(), trace2()}) {
            tr[ *x].add({"CH1", "CH2"});
        }
        tr[ *trace1()] = "CH2";
        tr[ *trace2()] = "CH1";
        tr[ *average()] = 1;
        for(auto &&x: unnecessary_ui)
            tr[ *x].disable();
    });
}
XThamwayDVUSBDSO::~XThamwayDVUSBDSO() {
}
void
XThamwayDVUSBDSO::open() {
    XScopedLock<XInterface> lock( *interface());
    XString idn = interface()->getIDN();
    gMessagePrint(formatString_tr(I18N_NOOP("%s successfully opened\n"), idn.c_str()));
    auto pos = idn.find("BIT=");
    if(pos == std::string::npos)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(sscanf(idn.c_str() + pos, "BIT=%u", &m_adConvBits) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);

// Initial states after powerup are undefined.
    int smps = interface()->readRegister16(ADDR_SAMPLES_MSW);
    smps = smps * 0x10000L + interface()->readRegister16(ADDR_SAMPLES_LSW);
    smps = std::min((int)MAX_SMPL, std::max(10000 + (int)EXTRA_SMPL, smps));
    smps -= EXTRA_SMPL;
//    int smps = 25000;
    double intv = std::max(2.0 / INTERNAL_CLOCK, getTimeInterval());
//    fprintf(stderr, "smps%u,avg%u,intv%g\n",smps,avg,intv);
    iterate_commit([=](Transaction &tr){
        tr[ *recordLength()] = smps;
        tr[ *timeWidth()] = smps * intv;
    });

    m_pending = true;
    Snapshot shot( *this);
//    onTimeWidthChanged(shot, 0);
    onRecordLengthChanged(shot, 0);
    onAverageChanged(shot, 0);
    interface()->writeToRegister8(ADDR_FRAMESM1, 0); //1 frame.
    m_pending = false;

    this->start();

    startSequence();
}
void
XThamwayDVUSBDSO::close() {
    interface()->stop();
}
void
XThamwayDVUSBDSO::onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) {
//    XScopedLock<XInterface> lock( *interface());
//    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.

//    interface()->writeToRegister8(ADDR_STS, 0x80); //soft trigger? undocumented.

//    startSequence();
}

void
XThamwayDVUSBDSO::startSequence() {
    XScopedLock<XInterface> lock( *interface());
    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.
    if( !m_pending) {
        interface()->writeToRegister16(ADDR_CH1_SET_MEM_ADDR_LSW, 0);
        interface()->writeToRegister16(ADDR_CH1_SET_MEM_ADDR_MSW, 0);
        interface()->writeToRegister16(ADDR_CH2_SET_MEM_ADDR_LSW, 0);
        interface()->writeToRegister16(ADDR_CH2_SET_MEM_ADDR_MSW, 0);

        interface()->writeToRegister8(ADDR_CTRL, 1); //starts.
    }
}

int
XThamwayDVUSBDSO::acqCount(bool *seq_busy) {
    XScopedLock<XInterface> lock( *interface());
    if(m_pending) {
        if(seq_busy) *seq_busy = true;
        return 0;
    }
    uint8_t sts = interface()->singleRead(ADDR_STS);
    bool is_started = sts & 2;
    bool is_ad_finished = sts & 4;
    int acq = interface()->readRegister16(ADDR_ACQCNTM1_MSW);
    acq = acq * 0x10000L + interface()->readRegister16(ADDR_ACQCNTM1_LSW);
    if(is_started && is_ad_finished) acq++;
    if(seq_busy) {
        *seq_busy = !is_started || !is_ad_finished;
    }
    return acq;
}

double
XThamwayDVUSBDSO::getTimeInterval() {
    XScopedLock<XInterface> lock( *interface());
    int div = interface()->singleRead(ADDR_DIVISOR);
    int pres = interface()->singleRead(ADDR_CFG) % 0x8u;
    double clk = INTERNAL_CLOCK / pow(2.0, pres) / std::max(div, 1);
    return 1.0/clk;
}

void
XThamwayDVUSBDSO::getWave(shared_ptr<RawData> &writer, std::deque<XString> &) {
    XScopedLock<XInterface> lock( *interface());

    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.
    interface()->writeToRegister16(ADDR_CH1_SET_MEM_ADDR_LSW, 0);
    interface()->writeToRegister16(ADDR_CH1_SET_MEM_ADDR_MSW, 0);
    interface()->writeToRegister16(ADDR_CH2_SET_MEM_ADDR_LSW, 0);
    interface()->writeToRegister16(ADDR_CH2_SET_MEM_ADDR_MSW, 0);

    int smps = interface()->readRegister16(ADDR_SAMPLES_MSW);
    smps = smps * 0x10000L + interface()->readRegister16(ADDR_SAMPLES_LSW);
    smps -= EXTRA_SMPL;
//    fprintf(stderr, "samps%d\n", smps);
    Snapshot shot( *this);
//    smps = shot[ *recordLength()];
    if(smps > MAX_SMPL)
        throw XInterface::XInterfaceError(i18n("# of samples exceeded the limit."), __FILE__, __LINE__);

    std::deque<uint8_t> adds;
    if(shot[ *trace1()] >= 0) adds.push_back( (shot[ *trace1()] == 0) ? ADDR_BURST_CH1 : ADDR_BURST_CH2 );
    if(shot[ *trace2()] >= 0) adds.push_back( (shot[ *trace2()] == 0) ? ADDR_BURST_CH1 : ADDR_BURST_CH2 );

    writer->push((uint16_t)adds.size()); //channels
    writer->push((uint32_t)0); //reserve
    writer->push((uint32_t)0); //reserve
    int acq = interface()->readRegister16(ADDR_ACQCNTM1_MSW);
    acq = acq * 0x10000L + interface()->readRegister16(ADDR_ACQCNTM1_LSW);
    acq++;
    writer->push((uint32_t)acq);
    writer->push((uint32_t)smps);
    writer->push((double)getTimeInterval());
    std::vector<uint8_t> buf(smps * sizeof(uint32_t));
    for(auto it = adds.begin(); it != adds.end(); ++it) {
        interface()->burstRead( *it, &buf[0], buf.size());
        double res = 2.5 / pow(2, m_adConvBits - 1);
        writer->push(res); //[V/bit]
        writer->push((double)-2.5); //offset[V]
        writer->insert(writer->end(), buf.begin(), buf.end());
    }
}
void
XThamwayDVUSBDSO::convertRaw(RawDataReader &reader, Transaction &tr) {
    const unsigned int num_ch = reader.pop<uint16_t>();
    reader.pop<uint32_t>(); //reserve
    reader.pop<uint32_t>(); //reserve
    unsigned int accum_count = reader.pop<uint32_t>();
    unsigned int len = reader.pop<uint32_t>();
    double interval = reader.pop<double>();

    tr[ *this].setParameters(num_ch, 0.0, interval, len);

    for(unsigned int j = 0; j < num_ch; j++) {
        double prop = reader.pop<double>() / accum_count;
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

    startSequence();
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
//    int smps = interface()->readRegister16(ADDR_SAMPLES_MSW);
//    smps = smps * 0x10000L + interface()->readRegister16(ADDR_SAMPLES_LSW);
//    smps--;
    int smps = Snapshot( *this)[ *recordLength()];

    double interval = shot[ *timeWidth()] / smps;
    int div = std::max(2L, lrint(INTERNAL_CLOCK * interval));
    int pres = std::min(7, std::max(0, (int)floor(log(div / 256.0) / log(2.0)) + 1));

    div = std::max(1L, lrint(div / pow(2.0, pres)));
    if(div > 255)
        throw XInterface::XInterfaceError(i18n("Too long time intervals."), __FILE__, __LINE__);

    XScopedLock<XInterface> lock( *interface());
    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.

    uint8_t cfg = 0x20; //8:ext_clock 0x40:flip
    interface()->writeToRegister8(ADDR_CFG, cfg | pres);
    interface()->writeToRegister8(ADDR_DIVISOR, div);

    startSequence();
}
void
XThamwayDVUSBDSO::onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *interface());
    interface()->writeToRegister8(ADDR_CTRL, 0); //stops.

    unsigned int smps = shot[ *recordLength()];
    smps += EXTRA_SMPL;
    interface()->writeToRegister16(ADDR_SAMPLES_LSW, smps % 0x10000uL);
    interface()->writeToRegister16(ADDR_SAMPLES_MSW, smps / 0x10000uL);

    onTimeWidthChanged(Snapshot( *this), 0);
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

