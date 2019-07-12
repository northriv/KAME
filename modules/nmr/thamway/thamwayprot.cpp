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
#include "analyzer.h"
#include "charinterface.h"
#include "thamwayprot.h"
#include "ui_thamwayprotform.h"
#include <QStatusBar>

REGISTER_TYPE(XDriverList, ThamwayT300ImpedanceAnalyzer, "Thamway T300-1049A Impedance Analyzer");
REGISTER_TYPE(XDriverList, ThamwayCharPROT, "Thamway PROT NMR.EXE TCP/IP Control");
#ifdef USE_THAMWAY_USB
    REGISTER_TYPE(XDriverList, ThamwayUSBPROT, "Thamway PROT NMR USB Control");
    XThamwayUSBPROT::XThamwayUSBPROT(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XThamwayPROT<XThamwayMODCUSBInterface>(name, runtime, ref(tr_meas), meas) {
        interface()->setEOS("\r\n");
    }
    void XThamwayUSBPROT::start() {
        interface()->query("*IDN?");
        fprintf(stderr, "PROT:%s\n", interface()->toStr().c_str());
        XThamwayPROT<XThamwayMODCUSBInterface>::start();
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
    m_fwdPWR(XSG::create<XDoubleNode>("FWDPWR", true, "%.1f")),
    m_bwdPWR(XSG::create<XDoubleNode>("BWDPWR", true, "%.1f")),
    m_ampWarn(XSG::create<XBoolNode>("AmpWarn", true)),
    m_form(new FrmThamwayPROT) {

    m_form->statusBar()->hide();
    m_form->setWindowTitle(i18n("Thamway PROT Control - ") + this->getLabel() );

    //Ranges should be preset in prior to connectors.
    m_form->m_dblOutput->setRange(0, 1023);
    m_form->m_dblOutput->setSingleStep(1);
    m_form->m_dblRXGain->setRange(0, 95);
    m_form->m_dblRXGain->setSingleStep(2.0);
    m_form->m_dblRXPhase->setRange(0, 360);
    m_form->m_dblRXPhase->setSingleStep(1.0);

//    XSG::m_conUIs.clear();
//    XSG::m_form.reset();
    m_conUIs = {
        xqcon_create<XQToggleButtonConnector>(this->rfON(), m_form->m_ckbRFON),
        xqcon_create<XQDoubleSpinBoxConnector>(this->oLevel(), m_form->m_dblOutput, m_form->m_slOutput),
        xqcon_create<XQLineEditConnector>(this->freq(), m_form->m_edFreq),
        xqcon_create<XQDoubleSpinBoxConnector>(m_rxGain, m_form->m_dblRXGain, m_form->m_slRXGain),
        xqcon_create<XQDoubleSpinBoxConnector>(m_rxPhase, m_form->m_dblRXPhase, m_form->m_slRXPhase),
        xqcon_create<XQLineEditConnector>(m_rxLPFBW, m_form->m_edRXLPFBW),
        xqcon_create<XQLCDNumberConnector>(m_fwdPWR, m_form->m_lcdFWD),
        xqcon_create<XQLCDNumberConnector>(m_bwdPWR, m_form->m_lcdBWD),
        xqcon_create<XQLedConnector>(m_ampWarn, m_form->m_ledWarn),
    };
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
    fetchStatus({}, true);

    XSG::start();

    rxGain()->setUIEnabled(true);
    rxPhase()->setUIEnabled(true);
    rxLPFBW()->setUIEnabled(true);

    this->iterate_commit([=](Transaction &tr){
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
    });

    m_thread.reset(new XThread{XThamwayPROT<tInterface>::shared_from_this(), &XThamwayPROT<tInterface>::fetchStatus, false});
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

    if(m_thread) {
        m_thread->terminate();
        m_thread->join();
        m_thread.reset();
    }

    XSG::stop();
}

template <class tInterface>
double
XThamwayPROT<tInterface>::query(const char *cmd) {
    msecsleep(10);
    this->interface()->query(cmd);
    for(int i = 0; ; ++i) {
        double v;
        if(this->interface()->scanf((XString(cmd) + "%lf").c_str(), &v) == 1)
            return v;
        if(i > 4)
            throw XInterface::XConvError(__FILE__, __LINE__);
        this->interface()->receive(); //flushing not-welcome message if any.
        msecsleep(i * 10);
    }
}

template <class tInterface>
void
XThamwayPROT<tInterface>::fetchStatus(const atomic<bool>& terminated, bool single) {
    for(;;) {
        try {
            XScopedLock<XInterface> lock( *this->interface());
            msecsleep(100);
            Transaction tr( *this);

            double f = query("FREQR");
            double olevel = query("ATT1R");
        //    olevel = log10(olevel / 1023.0) * 20.0;
            double gain = query("GAINR");
            double phase = query("PHASR");
            double bw = query("LPF1R");
            bw *= 1e-3;
            int sw = (int)lrint(query("RFSWR"));
            double fwd = query("FWDPR");
            double bwd = query("BWDPR");
            int warn = (int)lrint(query("STTSR"));

            for(;;){
                if(fabs(tr[ *this->freq()] - f) > 1e-6) {
                    tr[ *this->freq()] = f;
                    tr.unmark(m_lsnFreq);
                }
                if(fabs(tr[ *this->oLevel()] - olevel) > 1e-3) {
                    tr[ *this->oLevel()] = olevel;
                    tr.unmark(m_lsnOLevel);
                }
                if(fabs(tr[ *this->rxGain()] - gain) > 1e-3) {
                    tr[ *this->rxGain()] = gain;
                    tr.unmark(m_lsnRXGain);
                }
                if(fabs(tr[ *this->rxPhase()] - phase) > 1e-3) {
                    tr[ *this->rxPhase()] = phase;
                    tr.unmark(m_lsnRXPhase);
                }
                if(fabs(tr[ *this->rxLPFBW()] - bw) > 1e-3) {
                    tr[ *this->rxLPFBW()] = bw;
                    tr.unmark(m_lsnRXLPFBW);
                }
                if(tr[ *this->rfON()] != sw) {
                    tr[ *this->rfON()] = sw;
                    tr.unmark(m_lsnRFON);
                }
                tr[ *this->fwdPWR()] = fwd;
                tr[ *this->bwdPWR()] = bwd;
                tr[ *this->ampWarn()] = warn;
//                if( !single && (XTime::now() - m_timeUIinteracted < 0.2))
//                    break;
                if(tr.commitOrNext())
                    break;
                if( !single)
                    break; //fetched status are discarded because of possible user interaction.
            }
        }
        catch (XInterface::XInterfaceError &e) {
            e.print();
        }
        if(single || terminated)
            break;
        msecsleep(100);
    }
}

template <class tInterface>
void
XThamwayPROT<tInterface>::changeFreq(double mhz) {
    XScopedLock<XInterface> lock( *this->interface());
    msecsleep(20);
    this->interface()->sendf("FREQW%010.6f", mhz);
    msecsleep(50); //wait stabilization of PLL
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *this->interface());
    msecsleep(20);
    this->interface()->sendf("RFSWW%s", shot[ *this->rfON()] ? "1" : "0");
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *this->interface());
    msecsleep(20);
    int olevel = (int)shot[ *this->oLevel()]; //pow(10, shot[ *this->oLevel()] / 20.0) * 1023.0);
    olevel = std::min(1023, std::max(0, olevel));
    this->interface()->sendf("ATT1W%04.0f", (double)olevel);
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onRXGainChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *this->interface());
    msecsleep(20);
    double gain = shot[ *rxGain()];
    gain = std::min(95.0, std::max(0.0, gain));
    this->interface()->sendf("GAINW%02.0f", gain);
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onRXPhaseChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *this->interface());
    msecsleep(20);
    double phase = shot[ *rxPhase()];
    phase -= floor(phase / 360.0) * 360.0;
    this->interface()->sendf("PHASW%05.1f", phase);
}
template <class tInterface>
void
XThamwayPROT<tInterface>::onRXLPFBWChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *this->interface());
    msecsleep(20);
    double bw = shot[ *rxLPFBW()];
//    bw = std::min(200.0, std::max(0.0, bw));
    this->interface()->sendf("LPF1W%07.0f", bw * 1e3);
}

template class XThamwayPROT<class XCharInterface>;
#ifdef USE_THAMWAY_USB
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
    power()->disable();

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
