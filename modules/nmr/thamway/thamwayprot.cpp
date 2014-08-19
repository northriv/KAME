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
#include "analyzer.h"
#include "charinterface.h"
#include "thamwayprot.h"
#include "ui_thamwayprotform.h"
#include <QStatusBar>

REGISTER_TYPE(XDriverList, ThamwayT300ImpedanceAnalyzer, "Thamway T300-1049A Impedance Analyzer");
REGISTER_TYPE(XDriverList, ThamwayCharPROT, "Thamway PROT NMR.EXE TCP/IP Control");
#ifdef USE_EZUSB
    REGISTER_TYPE(XDriverList, ThamwayUSBPROT, "Thamway PROT NMR USB Control");
    XThamwayUSBPROT::XThamwayUSBPROT(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XThamwayPROT<XThamwayMODCUSBInterface>(name, runtime, ref(tr_meas), meas) {
        interface()->setEOS("\r\n");
    }
    void XThamwayUSBPROT::start() {
        interface()->query("*IDN?");
        fprintf(stderr, "PROT:%s\n", interface()->toStr().c_str());
        XThamwayPROT::start();
    }
#endif

XThamwayCharPROT::XThamwayCharPROT(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XThamwayPROT<XCharInterface>(name, runtime, ref(tr_meas), meas) {
    trans( *this->interface()->port()) = "127.0.0.1:5025";
    trans( *this->interface()->device()) = "TCP/IP";
    this->interface()->setEOS("\n");
}

template <class tInterface>
XThamwayPROT<tInterface>::XThamwayPROT(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XSG, tInterface>(name, runtime, ref(tr_meas), meas),
    m_rxGain(XSG::create<XDoubleNode>("RXGain", true, "%.0f")),
    m_rxPhase(XSG::create<XDoubleNode>("RXPhase", true, "%.1f")),
    m_rxLPFBW(XSG::create<XDoubleNode>("RXLPFBW", true, "%.4g")),
    m_form(new FrmThamwayPROT(g_pFrmMain)) {

    m_form->statusBar()->hide();
    m_form->setWindowTitle(i18n("Thamway PROT Control - ") + this->getLabel() );

    m_conRFON = xqcon_create<XQToggleButtonConnector>(this->rfON(), m_form->m_ckbRFON);
    m_form->m_dblOutput->setRange(-40, 0);
    m_form->m_dblOutput->setSingleStep(1.0);
    m_conOLevel = xqcon_create<XQDoubleSpinBoxConnector>(this->oLevel(), m_form->m_dblOutput, m_form->m_slOutput);
    m_conFreq = xqcon_create<XQLineEditConnector>(this->freq(), m_form->m_edFreq);
    m_form->m_dblRXGain->setRange(0, 95);
    m_form->m_dblRXGain->setSingleStep(2.0);
    m_conRXGain = xqcon_create<XQDoubleSpinBoxConnector>(m_rxGain, m_form->m_dblRXGain, m_form->m_slRXGain);
    m_form->m_dblRXPhase->setRange(0, 360);
    m_form->m_dblRXPhase->setSingleStep(1.0);
    m_conRXPhase = xqcon_create<XQDoubleSpinBoxConnector>(m_rxPhase, m_form->m_dblRXPhase, m_form->m_slRXPhase);
    m_conRXLPFBW = xqcon_create<XQLineEditConnector>(m_rxLPFBW, m_form->m_edRXLPFBW);

    this->rfON()->setUIEnabled(false);
    this->oLevel()->setUIEnabled(false);
    this->freq()->setUIEnabled(false);
    this->amON()->disable();
    this->fmON()->disable();
    rxGain()->setUIEnabled(false);
    rxPhase()->setUIEnabled(false);
    rxLPFBW()->setUIEnabled(false);
}
template <class tInterface>
void
XThamwayPROT<tInterface>::showForms() {
    m_form->showNormal();
    m_form->raise();
}
template <class tInterface>
void
XThamwayPROT<tInterface>::start() {
    XScopedLock<XInterface> lock( *this->interface());
    this->interface()->query("FREQR");
    double f;
    for(int i = 0; ; ++i) {
        if(this->interface()->scanf("FREQR%lf", &f) == 1)
            break;
        if(i > 2)
            throw XInterface::XConvError(__FILE__, __LINE__);
        this->interface()->receive(); //flushing not-welcome message if any, although this is TCP connection.
    }
    this->interface()->query("ATT1R");
    double olevel;
    if(this->interface()->scanf("ATT1R%lf", &olevel) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    olevel = log10(olevel / 1023.0);
    this->interface()->query("GAINR");
    double gain;
    if(this->interface()->scanf("GAINR%lf", &gain) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    this->interface()->query("PHASR");
    double phase;
    if(this->interface()->scanf("PHASR%lf", &phase) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    this->interface()->query("LPF1R");
    double bw;
    if(this->interface()->scanf("LPF1R%lf", &bw) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    for(Transaction tr( *this);; ++tr) {
        tr[ *this->freq()] = f;
        tr[ *this->oLevel()] = olevel;
        tr[ *this->rxGain()] = gain;
        tr[ *this->rxPhase()] = phase;
        tr[ *this->rxLPFBW()] = bw;
        if(tr.commit())
            break;
    }

    XSG::start();

    rxGain()->setUIEnabled(true);
    rxPhase()->setUIEnabled(true);
    rxLPFBW()->setUIEnabled(true);

    for(Transaction tr( *this);; ++tr) {
        m_lsnRFON = tr[ *this->rfON()].onValueChanged().connectWeakly(
            this->shared_from_this(), &XThamwayPROT::onRFONChanged);
        m_lsnOLevel = tr[ *this->oLevel()].onValueChanged().connectWeakly(
            this->shared_from_this(), &XThamwayPROT::onOLevelChanged);
        m_lsnFreq = tr[ *this->freq()].onValueChanged().connectWeakly(
            this->shared_from_this(), &XThamwayPROT::onFreqChanged);
        m_lsnRXGain = tr[ *rxGain()].onValueChanged().connectWeakly(
            this->shared_from_this(), &XThamwayPROT::onRXGainChanged);
        m_lsnRXPhase = tr[ *rxPhase()].onValueChanged().connectWeakly(
            this->shared_from_this(), &XThamwayPROT::onRXPhaseChanged);
        m_lsnRXLPFBW = tr[ *rxLPFBW()].onValueChanged().connectWeakly(
            this->shared_from_this(), &XThamwayPROT::onRXLPFBWChanged);
        if(tr.commit())
            break;
    }
}
template <class tInterface>
void
XThamwayPROT<tInterface>::stop() {
    rxGain()->setUIEnabled(false);
    rxPhase()->setUIEnabled(false);
    rxLPFBW()->setUIEnabled(false);

    m_lsnRFON.reset();
    m_lsnOLevel.reset();
    m_lsnFreq.reset();
    m_lsnRXGain.reset();
    m_lsnRXPhase.reset();
    m_lsnRXLPFBW.reset();

    XSG::stop();
}

template <class tInterface>
void
XThamwayPROT<tInterface>::changeFreq(double mhz) {
    XScopedLock<XInterface> lock( *this->interface());
    this->interface()->sendf("FREQW%010.6f", mhz);
    msecsleep(50); //wait stabilization of PLL
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
    this->interface()->sendf("RFSWW%s", shot[ *this->rfON()] ? "1" : "0");
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    this->interface()->sendf("ATT1W%04.0f", (double)pow(10, shot[ *this->oLevel()] / 10.0) * 1023.0);
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onRXGainChanged(const Snapshot &shot, XValueNodeBase *) {
    double gain = shot[ *rxGain()];
    gain = std::min(95.0, std::max(0.0, gain));
    this->interface()->sendf("GAINW%02.0f", gain);
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onRXPhaseChanged(const Snapshot &shot, XValueNodeBase *) {
    double phase = shot[ *rxPhase()];
    phase -= floor(phase / 360.0) * 360.0;
    this->interface()->sendf("PHASW%05.1f", phase);
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onRXLPFBWChanged(const Snapshot &shot, XValueNodeBase *) {
    double bw = shot[ *rxLPFBW()];
    bw = std::min(200.0, std::max(0.0, bw));
    this->interface()->sendf("LPF1W%07.0f", bw);
}

template class XThamwayPROT<class XCharInterface>;
#ifdef USE_EZUSB
    template class XThamwayPROT<class XThamwayMODCUSBInterface>;
#endif

XThamwayT300ImpedanceAnalyzer::XThamwayT300ImpedanceAnalyzer(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\r\n");
    interface()->setSerialBaudRate(115200);
    interface()->setSerialStopBits(1);
    interface()->setSerialFlushBeforeWrite(true);
    trans( *interface()->device()) = "SERIAL";

    average()->disable();

    calThru()->disable();
}

void
XThamwayT300ImpedanceAnalyzer::open() throw (XKameError &) {
    interface()->query("GET START?");
    double freq;
    if(interface()->scanf("START %lf", &freq) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    trans( *startFreq()) = freq;
    interface()->query("GET STOP?");
    if(interface()->scanf("STOP %lf", &freq) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    trans( *stopFreq()) = freq;
    unsigned int samples;
    interface()->query("GET SAMPLE?");
    if(interface()->scanf("SAMPLE %u", &samples) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    trans( *points()) = samples;

//    interface()->send("RJX FORMAT"); //R+jX format
    interface()->send("LOG FORMAT"); //log format

    start();
}
void
XThamwayT300ImpedanceAnalyzer::onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("SET START %f", (double)shot[ *startFreq()]);
}
void
XThamwayT300ImpedanceAnalyzer::onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("SET STOP %f", (double)shot[ *stopFreq()]);
}
void
XThamwayT300ImpedanceAnalyzer::onPointsChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("SET SAMPLE %u", (unsigned int)shot[ *points()]);
}
void
XThamwayT300ImpedanceAnalyzer::onCalOpenTouched(const Snapshot &shot, XTouchableNode *) {
    interface()->query("CAL OPEN");
    if(interface()->toStrSimplified() != "CAL OPEN DONE")
        throw XInterface::XInterfaceError("Calibration has failed.", __FILE__, __LINE__);
}
void
XThamwayT300ImpedanceAnalyzer::onCalShortTouched(const Snapshot &shot, XTouchableNode *) {
    interface()->query("CAL SHORT");
    if(interface()->toStrSimplified() != "CAL SHORT DONE")
        throw XInterface::XInterfaceError("Calibration has failed.", __FILE__, __LINE__);
}
void
XThamwayT300ImpedanceAnalyzer::onCalTermTouched(const Snapshot &shot, XTouchableNode *) {
    interface()->query("CAL LOAD");
    if(interface()->toStrSimplified() != "CAL LOAD DONE")
        throw XInterface::XInterfaceError("Calibration has failed.", __FILE__, __LINE__);
}

void
XThamwayT300ImpedanceAnalyzer::getMarkerPos(unsigned int num, double &x, double &y) {
    if(num > 2)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    Snapshot shot( *this);
    const auto trace = shot[ *this].trace();
    double v = (num == 0) ? 1e10 : -1.0;
    unsigned int idx;
    for(unsigned int i = 0; i < shot[ *this].length(); i++) {
        double z = std::norm( trace[i]);
        if(num == 0) {
            if(v > z) {
                v = z;
                idx = i;
            }
        }
        else {
            if(v < z) {
                v = z;
                idx = i;
            }
        }
    }
    x *= idx * shot[ *this].freqInterval() + shot[ *this].startFreq();
    y = log10(v) * 20.0;
}
void
XThamwayT300ImpedanceAnalyzer::oneSweep() {
}
void
XThamwayT300ImpedanceAnalyzer::startContSweep() {
}
void
XThamwayT300ImpedanceAnalyzer::acquireTrace(shared_ptr<RawData> &writer, unsigned int ch) {
    XScopedLock<XInterface> lock( *interface());
    uint32_t len;
    interface()->query("MEAS ON");
    if(interface()->scanf("LOG MAG,F=,M=,%u", &len) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    writer->push((uint32_t)0); //LOG MAG
    writer->push(len);
    for(unsigned int i = 0; i < len; ++i) {
        interface()->receive();
        float freq, amp;
        if(interface()->scanf("%f,%f", &freq, &amp) != 2)
            throw XInterface::XConvError(__FILE__, __LINE__);
        writer->push(freq);
        writer->push(amp);
    }
}
void
XThamwayT300ImpedanceAnalyzer::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    const Snapshot &shot(tr);
    uint32_t stype = reader.pop<uint32_t>();
    uint32_t samples = reader.pop<uint32_t>();
    switch(stype) {
    case 0: //Log format.
        break;
    default:
        throw XRecordError(i18n("Given format is not supported."), __FILE__, __LINE__);
    }

    tr[ *this].trace_().resize(samples);
    float startfreq = 0.0, stopfreq = 0.0;
    for(unsigned int i = 0; i < samples; i++) {
        float freq = reader.pop<float>();
        if(i == 0)
            startfreq = freq;
        if(i == samples - 1)
            stopfreq = freq;
        tr[ *this].trace_()[i] = pow(10.0, reader.pop<float>() / 20.0);
    }
    tr[ *this].m_startFreq = startfreq;
    tr[ *this].m_freqInterval = (stopfreq - startfreq) / (samples - 1);
}
