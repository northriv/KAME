/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef PUMPCONTROLLER_H
#define PUMPCONTROLLER_H

#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class Ui_FrmPumpController;
typedef QForm<QMainWindow, Ui_FrmPumpController> FrmPumpControl;

class XPumpControl : public XPrimaryDriverWithThread {
public:
    XPumpControl(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! usually nothing to do
    virtual ~XPumpControl() = default;
    //! show all forms belonging to driver
    virtual void showForms() override;

    const shared_ptr<XScalarEntry> &entryPressure() const {return m_entryPressure;}

    const shared_ptr<XBoolNode> activate() const {return m_activate;}
    const shared_ptr<XBoolNode> heating() const {return m_heating;}
    const shared_ptr<XBoolNode> standby() const {return m_standby;}
    const shared_ptr<XDoubleNode> standbyRotationSpeed() const {return m_standbyRotationSpeed;}
    const shared_ptr<XDoubleNode> maxDrivePower() const {return m_maxDrivePower;}
protected:
    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;

    //! reads a rotation speed from the instrument
    virtual double getRotationSpeed() = 0; //[Hz]
    //! reads runtime value from the instrument
    virtual double getRuntime() = 0; //[hrs]
    //! reads pressure sensor value from the instrument
    virtual double getPressure() = 0; //[Pa]
    //! reads temperatures from the instrument
    virtual std::deque<XString> getTempLabels() = 0;
    virtual std::deque<double> getTemps() = 0; //[degC]
    //! reads warning status from the instrument
    virtual std::pair<unsigned int, XString> getWarning() = 0;
    //! reads error status from the instrument
    virtual std::pair<unsigned int, XString> getError() = 0;

    virtual void changeMode(bool active, bool stby, bool heating) = 0;
    virtual void changeMaxDrivePower(double p) = 0; //[%]
    virtual void changeStandbyRotationSpeed(double p) = 0; //[%]
private:
    const shared_ptr<XBoolNode> m_activate, m_heating, m_standby;
    const shared_ptr<XBoolNode> m_warning, m_error;
    const shared_ptr<XDoubleNode> m_rotationSpeed, m_runtime;
    std::deque<shared_ptr<XDoubleNode>> m_temps;
    const shared_ptr<XDoubleNode> m_standbyRotationSpeed, m_maxDrivePower;

    shared_ptr<Listener> m_lsnOnModeChanged, m_lsnOnStanbyRotationSpeedChanged, m_lsnMaxDriverPowerChanged;

    void onModeChanged(const Snapshot &shot, XValueNodeBase *);
    void onStandbyRotationSpeedChanged(const Snapshot &shot, XValueNodeBase *);
    void onMaxDriverPowerChanged(const Snapshot &shot, XValueNodeBase *);

    shared_ptr<XScalarEntry> m_entryPressure;

    const qshared_ptr<FrmPumpControl> m_form;

    std::deque<xqcon_ptr> m_conUIs, m_conTempUIs;

    virtual void *execute(const atomic<bool> &) override;
};
#endif // PUMPCONTROLLER_H
