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
//---------------------------------------------------------------------------

#ifndef lasermoduleH
#define lasermoduleH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"
#include <deque>
#include <vector>

class XScalarEntry;
class QMainWindow;
class Ui_FrmLaserModule;
typedef QForm<QMainWindow, Ui_FrmLaserModule> FrmLaserModule;

//! Abstract base for laser-diode controllers, single- OR multi-channel.
//!
//! A controller is modelled as two independent lists of channels: LaserChannel (current source)
//! and TecChannel (thermo-electric temperature control). This split -- rather than one combined
//! "module" per slot -- matches modular mainframes (e.g. Newport/ILX LDC-3900) whose slots may
//! hold a laser-only, TEC-only, or combination module, and it lets the UI show laser and TEC on
//! separate panels. A conventional single-channel instrument is simply the N=1 special case:
//! e.g. a bare current source is createLaserChannels(1)+createTecChannels(0); a current source
//! with a TEC is (1)+(1); the LDC-3900 mainframe is (4)+(4).
//!
//! Concrete drivers derive XCharDeviceDriver<XLaserModule> (one shared interface for the whole
//! instrument), call createLaserChannels()/createTecChannels() from their constructor to declare
//! how many channels exist (NOT hard-coded here), and implement the per-channel instrument hooks
//! (readLaser/readTec/setLaser*/setTec*/readErrors). All the generic machinery -- the polling
//! thread, the raw-record pipeline, the QToolBox form with unused pages hidden, and the
//! non-finite ("-INF"/absent-module) guard -- lives here and is shared.
class DECLSPEC_SHARED XLaserModule : public XPrimaryDriverWithThread {
public:
    XLaserModule(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLaserModule() {}
    //! Shows all forms belonging to driver.
    virtual void showForms() override;

    //! One laser current-source channel, bound to a fixed slot (1-based). Its readbacks are
    //! populated only while readLaser() reports the channel present; its setpoints/enable are
    //! one-directional command nodes (set by user/script -> driver hook; never written back from
    //! the readback, which would fight user input and re-fire the listener every poll).
    class DECLSPEC_SHARED LaserChannel : public XNode {
    public:
        LaserChannel(const char *name, bool runtime, unsigned int slot,
            const shared_ptr<XLaserModule> &driver);
        unsigned int slot() const {return m_slot;}
        const shared_ptr<XScalarEntry> &current() const {return m_current;} //![mA]
        const shared_ptr<XScalarEntry> &power() const {return m_power;} //![mW]
        const shared_ptr<XScalarEntry> &voltage() const {return m_voltage;} //![V]
        const shared_ptr<XDoubleNode> &setCurrent() const {return m_setCurrent;} //![mA]
        const shared_ptr<XDoubleNode> &setPower() const {return m_setPower;} //![mW]
        const shared_ptr<XBoolNode> &enabled() const {return m_enabled;}
        void start();
        void stop();
    private:
        friend class XLaserModule;
        const unsigned int m_slot;
        weak_ptr<XLaserModule> m_driver;
        const shared_ptr<XScalarEntry> m_current, m_power, m_voltage;
        const shared_ptr<XDoubleNode> m_setCurrent, m_setPower;
        const shared_ptr<XBoolNode> m_enabled;
        shared_ptr<Listener> m_lsnSetCurrent, m_lsnSetPower, m_lsnEnabled;
        void onSetCurrentChanged(const Snapshot &shot, XValueNodeBase *);
        void onSetPowerChanged(const Snapshot &shot, XValueNodeBase *);
        void onEnabledChanged(const Snapshot &shot, XValueNodeBase *);
    };

    //! One TEC channel, bound to a fixed slot (1-based). Independent of the same-numbered
    //! LaserChannel (a modular mainframe treats laser and TEC channel selection independently).
    class DECLSPEC_SHARED TecChannel : public XNode {
    public:
        TecChannel(const char *name, bool runtime, unsigned int slot,
            const shared_ptr<XLaserModule> &driver);
        unsigned int slot() const {return m_slot;}
        const shared_ptr<XScalarEntry> &temp() const {return m_temp;} //![degC]
        const shared_ptr<XDoubleNode> &setTemp() const {return m_setTemp;} //![degC]
        const shared_ptr<XBoolNode> &enabled() const {return m_enabled;}
        void start();
        void stop();
    private:
        friend class XLaserModule;
        const unsigned int m_slot;
        weak_ptr<XLaserModule> m_driver;
        const shared_ptr<XScalarEntry> m_temp;
        const shared_ptr<XDoubleNode> m_setTemp;
        const shared_ptr<XBoolNode> m_enabled;
        shared_ptr<Listener> m_lsnSetTemp, m_lsnEnabled;
        void onSetTempChanged(const Snapshot &shot, XValueNodeBase *);
        void onEnabledChanged(const Snapshot &shot, XValueNodeBase *);
    };

    unsigned int numLaserChannels() const {return (unsigned int)m_laserChannels.size();}
    unsigned int numTecChannels() const {return (unsigned int)m_tecChannels.size();}
    //! \param slot 1-based.
    const shared_ptr<LaserChannel> &laserChannel(unsigned int slot) const {return m_laserChannels.at(slot - 1);}
    const shared_ptr<TecChannel> &tecChannel(unsigned int slot) const {return m_tecChannels.at(slot - 1);}
    const shared_ptr<XStringNode> &status() const {return m_status;}

    struct Payload : public XPrimaryDriver::Payload {};

protected:
    //! Declare the instrument's channels; call from the concrete driver's constructor (additive:
    //! a subclass may add TEC channels on top of a base that already created laser channels).
    //! Creates the channel objects + scalar entries, wires them to the QToolBox pages, and
    //! reveals the corresponding panel (unused pages stay hidden; an empty list keeps its whole
    //! panel hidden). The form starts fully hidden in the XLaserModule constructor.
    void createLaserChannels(Transaction &tr_meas, const shared_ptr<XMeasure> &meas, unsigned int n);
    void createTecChannels(Transaction &tr_meas, const shared_ptr<XMeasure> &meas, unsigned int n);

    //! Per-channel instrument hooks. \a slot is 1-based (ignored by single-channel instruments).
    //! read*: fill the out-params and return true if the channel is present; return false for an
    //! absent/not-applicable slot (e.g. "-INF"), which the base then skips this poll. Individual
    //! readback fields may still be non-finite even when present (e.g. optical power on a source
    //! with no monitor photodiode) -- the base records each field only if std::isfinite().
    virtual bool readLaser(unsigned int slot, double &current_mA, double &power_mW,
        double &voltage_V, bool &output_on) {return false;}
    virtual bool readTec(unsigned int slot, double &temp_C, bool &output_on) {return false;}
    virtual void setLaserCurrent(unsigned int slot, double mA) {}
    virtual void setLaserPower(unsigned int slot, double mW) {}
    virtual void setLaserOutput(unsigned int slot, bool on) {}
    virtual void setTecTemp(unsigned int slot, double degC) {}
    virtual void setTecOutput(unsigned int slot, bool on) {}
    //! Instrument-wide error query (not per channel); folded into the status string. Optional.
    virtual XString readErrors() {return XString();}

    //! Pops the readings pushed by execute() into the scalar entries and the status node.
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    virtual void visualize(const Snapshot &shot) override;
    void *execute(const atomic<bool> &) override;

private:
    std::deque<shared_ptr<LaserChannel> > m_laserChannels;
    std::deque<shared_ptr<TecChannel> > m_tecChannels;

    const shared_ptr<XStringNode> m_status;
    const qshared_ptr<FrmLaserModule> m_form;
    std::deque<xqcon_ptr> m_conUIs;
};

//---------------------------------------------------------------------------

#endif
