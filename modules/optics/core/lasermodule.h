/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef lasermoduleH
#define lasermoduleH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmLaserModule;
typedef QForm<QMainWindow, Ui_FrmLaserModule> FrmLaserModule;

//! Base class for digital storage oscilloscope.
class DECLSPEC_SHARED XLaserModule : public XPrimaryDriverWithThread {
public:
    XLaserModule(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! usually nothing to do.
    virtual ~XLaserModule() {}
    //! Shows all forms belonging to driver.
    virtual void showForms() override;

    struct Payload : public XPrimaryDriver::Payload {
        double temperature() const {return m_temperature;} //! [degC]
        double current() const {return m_current;} //! [mA]
        double power() const {return m_power;} //! [mW]
        double voltage() const {return m_voltage;} //! [V]
    private:
        friend class XLaserModule;
        double m_temperature, m_current, m_power, m_voltage;
    };

    const shared_ptr<XScalarEntry> &temperature() const {return m_temperature;}
    const shared_ptr<XScalarEntry> &current() const {return m_current;}
    const shared_ptr<XScalarEntry> &power() const {return m_power;}
    const shared_ptr<XScalarEntry> &voltage() const {return m_voltage;}
    const shared_ptr<XStringNode> &status() const {return m_status;}
    const shared_ptr<XBoolNode> &enabled() const {return m_enabled;}
    const shared_ptr<XDoubleNode> &setCurrent() const {return m_setCurrent;}
    const shared_ptr<XDoubleNode> &setPower() const {return m_setPower;}
    const shared_ptr<XDoubleNode> &setTemp() const {return m_setTemp;}
protected:

    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;

protected:
    struct ModuleStatus {XString status; double temperature = 0, current = 0, power = 0, voltage = 0;};
    virtual ModuleStatus readStatus() = 0;

    virtual void onEnabledChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onCurrentChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onPowerChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onTempChanged(const Snapshot &shot, XValueNodeBase *) = 0;

    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;

    const shared_ptr<XScalarEntry> m_temperature;
    const shared_ptr<XScalarEntry> m_current;
    const shared_ptr<XScalarEntry> m_power;
    const shared_ptr<XScalarEntry> m_voltage;
private:
    const shared_ptr<XStringNode> m_status;
    const shared_ptr<XBoolNode> m_enabled;
    const shared_ptr<XDoubleNode> m_setCurrent;
    const shared_ptr<XDoubleNode> m_setPower;
    const shared_ptr<XDoubleNode> m_setTemp;

    const qshared_ptr<FrmLaserModule> m_form;

    shared_ptr<Listener> m_lsnOnEnabledChanged,
        m_lsnOnCurrentChanged, m_lsnOnPowerChanged, m_lsnOnTempChanged;

    std::deque<xqcon_ptr> m_conUIs;

    void *execute(const atomic<bool> &) override;
};

//---------------------------------------------------------------------------

#endif
