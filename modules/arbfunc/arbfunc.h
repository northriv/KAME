/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef ARBFUNC_H
#define ARBFUNC_H

#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class Ui_FrmArbFuncGen;
typedef QForm<QMainWindow, Ui_FrmArbFuncGen> FrmArbFuncGen;

class XArbFuncGen : public XPrimaryDriver {
public:
    XArbFuncGen(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! usually nothing to do
    virtual ~XArbFuncGen() = default;
    //! show all forms belonging to driver
    virtual void showForms() override;

    const shared_ptr<XBoolNode> output() const {return m_output;}
    const shared_ptr<XBoolNode> burst() const {return m_burst;}
    const shared_ptr<XDoubleNode> burstPhase() const {return m_burstPhase;} //!< Burst Start Phase[deg.]
    //! Phase of the CONTINUOUS waveform [deg.], sent as SCPI "PHAS". Distinct from
    //! burstPhase(), which is "BURS:PHAS" and only applies while burst() is set. Meaningful
    //! only when the generators share a timebase (10 MHz reference in/out chained), which is
    //! what fixes the relative phase between two units running continuously.
    const shared_ptr<XDoubleNode> phase() const {return m_phase;} //!< [deg.]
    const shared_ptr<XUIntNode> burstCycles() const {return m_burstCycles;} //!< Burst cycle count, 0 = INFinity
    const shared_ptr<XComboNode> trigSrc() const {return m_trigSrc;}
    const shared_ptr<XComboNode> waveform() const {return m_waveform;}
    const shared_ptr<XDoubleNode> freq() const {return m_freq;} //!< [Hz]
    const shared_ptr<XDoubleNode> ampl() const {return m_ampl;} //!< [V]
    const shared_ptr<XDoubleNode> offset() const {return m_offset;} //!< [V]
    const shared_ptr<XDoubleNode> duty() const {return m_duty;} //!< [%]
    const shared_ptr<XDoubleNode> pulseWidth() const {return m_pulseWidth;} //!< [s], 0 = specify by Duty instead
    const shared_ptr<XDoubleNode> pulsePeriod() const {return m_pulsePeriod;} //!< [s], 0 = follow Freq (period = 1/Freq)
    const shared_ptr<XTouchableNode> softwareTrig() const {return m_softwareTrig;} //!< sends a bus/software trigger (*TRG)

protected:
    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;

    //! Starts up your threads, connects GUI, and activates signals.
    virtual void start() override;
    //! Shuts down your threads, unconnects GUI, and deactivates signals
    //! This function may be called even if driver has already stopped.
    virtual void stop() override;

    virtual void changeOutput(bool active) = 0;
    virtual void changePulseCond() = 0;
    //! Issues a single software/bus trigger (e.g. SCPI "*TRG"). Effective when the trigger
    //! source is BUS; arms a burst on demand from the UI button or from a script.
    virtual void sendSoftwareTrigger() = 0;
private:
    const shared_ptr<XBoolNode> m_output;
    const shared_ptr<XBoolNode> m_burst;
    const shared_ptr<XDoubleNode> m_burstPhase;
    const shared_ptr<XDoubleNode> m_phase;
    const shared_ptr<XUIntNode> m_burstCycles;
    const shared_ptr<XComboNode> m_trigSrc, m_waveform;
    const shared_ptr<XDoubleNode> m_freq, m_ampl, m_offset, m_duty;
    const shared_ptr<XDoubleNode> m_pulseWidth, m_pulsePeriod;
    const shared_ptr<XTouchableNode> m_softwareTrig;

    shared_ptr<Listener> m_lsnOnCondChanged, m_lsnOnOutputChanged, m_lsnOnSoftTrigTouched;

    void onOutputChanged(const Snapshot &shot, XValueNodeBase *);
    void onCondChanged(const Snapshot &shot, XValueNodeBase *);
    void onSoftTrigTouched(const Snapshot &shot, XTouchableNode *);

    const qshared_ptr<FrmArbFuncGen> m_form;

    std::deque<xqcon_ptr> m_conUIs;
};
#endif // ARBFUNC_H
