/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef relaydriverH
#define relaydriverH

#include "primarydriver.h"
#include "xnodeconnector.h"

class QMainWindow;
class Ui_FrmRelay;
typedef QForm<QMainWindow, Ui_FrmRelay> FrmRelay;

//! Base class for relay/digital-output controllers.
class DECLSPEC_SHARED XRelayDriver : public XPrimaryDriver {
public:
    static constexpr unsigned int maxNumChannels = 8;

    XRelayDriver(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
        unsigned int num_channels = maxNumChannels);
    //! usually nothing to do
    virtual ~XRelayDriver() {}
    //! show all forms belonging to driver
    virtual void showForms() override;

    unsigned int numChannels() const {return (unsigned int)m_channelOutputs.size();}
    //! \arg ch [0, numChannels() - 1]
    const shared_ptr<XBoolNode> &channelOutput(unsigned int ch) const {return m_channelOutputs.at(ch);}

    //! driver specific part below
    //! \arg ch [0, numChannels() - 1]
    virtual void changeOutput(unsigned int ch, bool on) = 0;
    //! Reads the current states back from the device if supported, during start().
    virtual void queryStatus(Transaction &tr) {}

    struct Payload : public XPrimaryDriver::Payload {
        //! Recorded states; bit0 corresponds to the first channel.
        unsigned int bits() const {return m_bits;}
    private:
        friend class XRelayDriver;
        unsigned int m_bits = 0;
    };
protected:
    //! Starts up your threads, connects GUI, and activates signals.
    virtual void start() override;
    //! Shuts down your threads, unconnects GUI, and deactivates signals
    //! This function may be called even if driver has already stopped.
    virtual void stop() override;

    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;

    const qshared_ptr<FrmRelay> &form() const {return m_form;}
private:
    std::deque<shared_ptr<XBoolNode>> m_channelOutputs;
    std::deque<xqcon_ptr> m_conChannels;
    std::deque<shared_ptr<Listener>> m_lsnOutputs;

    void onOutputChanged(const Snapshot &shot, XValueNodeBase *node);

    const qshared_ptr<FrmRelay> m_form;

    void finish(const XTime &time_awared);
};

#endif
