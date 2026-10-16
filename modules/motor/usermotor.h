/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/

#ifndef USERMOTOR_H_
#define USERMOTOR_H_

#include "motor.h"
#include <cstdint>   //!< Kept locally: this header's include chain does
                     //!< not reach support.h.
#include "modbusrtuinterface.h"
#include "chardevicedriver.h"

//ORIENTAL MOTOR FLEX CRK series.
class XFlexCRK : public XModbusRTUDriver<XMotorDriver>  {
public:
	XFlexCRK(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XFlexCRK() {}

protected:
protected:
    virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) override;
    virtual void changeConditions(const Snapshot &shot) override;
    virtual void getConditions() override;
    virtual void setTarget(const Snapshot &shot, double target) override;
    virtual void setActive(bool active) override;
    virtual void setAUXBits(unsigned int bits) override;
    virtual void setForward() override; //!< continuous rotation.
    virtual void setReverse() override;//!< continuous rotation.
    virtual void stopRotation() override; //!< stops motor and waits for deceleration.
    virtual void toHomePosition() override; //!< goes to the origin by sensor(s).
    //! stores current settings to the NV memory of the instrumeMotornt.
    virtual void storeToROM() override;
    virtual void clearPosition() override;
protected:
    //! Alarm/warning codes already reported, to keep the 100 ms poll from
    //! reprinting a condition that persists. Touched only by getStatus(),
    //! i.e. the poll thread.
    uint32_t m_alarmReported = 0, m_warnReported = 0;
private:
	void sendStopSignal(bool wait);
};

//ORIENTAL MOTOR CVD series with RS-485.
class XOrientalMotorCVD2B : public XModbusRTUDriver<XMotorDriver>  {
public:
    XOrientalMotorCVD2B(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XOrientalMotorCVD2B() {}

protected:
    virtual bool isPresetTo2Phase() const {return true;}
protected:
    virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) override;
    virtual void changeConditions(const Snapshot &shot) override;
    virtual void getConditions() override;
    virtual void setTarget(const Snapshot &shot, double target) override;
    virtual void setActive(bool active) override;
    virtual void setAUXBits(unsigned int bits) override;
    virtual void setForward() override; //!< continuous rotation.
    virtual void setReverse() override;//!< continuous rotation.
    virtual void stopRotation() override; //!< stops motor and waits for deceleration.
    virtual void toHomePosition() override; //!< goes to the origin by sensor(s).
    //! stores current settings to the NV memory of the instrumeMotornt.
    virtual void storeToROM() override;
    virtual void clearPosition() override;
protected:
    //! \sa XFlexCRK::m_alarmReported
    uint32_t m_alarmReported = 0, m_infoReported = 0;
private:
    void sendStopSignal(bool wait);

    static const std::vector<uint32_t> s_resolutions_2B, s_resolutions_5B;
};
class XOrientalMotorCVD5B : public XOrientalMotorCVD2B  {
public:
    XOrientalMotorCVD5B(const char *name, bool runtime,
                        Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XOrientalMotorCVD2B(name, runtime, ref(tr_meas), meas) {}
    virtual ~XOrientalMotorCVD5B() {}

protected:
    virtual bool isPresetTo2Phase() const override {return false;}
};

//ORIENTAL MOTOR FLEX AR/DG2 series.
class XFlexAR : public XFlexCRK {
public:
	XFlexAR(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XFlexCRK(name, runtime, ref(tr_meas), meas) {}
	virtual ~XFlexAR() {}

    //! \arg points, speeds: [# of devices][# of points].
    //! \arg slaves: if any, devices to be started simultatneously.
    virtual void runSequentially(const std::vector<std::vector<double>> &points,
        const std::vector<std::vector<double>> &speeds, const std::vector<shared_ptr<XMotorDriver>> &slaves) override;
protected:
protected:
    virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) override;
    virtual void changeConditions(const Snapshot &shot) override;
    virtual void getConditions() override;
    virtual void setTarget(const Snapshot &shot, double target) override;
    virtual void setActive(bool active) override;
    virtual void setAUXBits(unsigned int bits) override;
    virtual void setForward() override; //!< continuous rotation.
    virtual void setReverse() override;//!< continuous rotation.
    virtual void stopRotation() override;//!< stops motor and waits for deceleration.
    virtual void toHomePosition() override; //!< goes to the origin by sensor(s).
    //! stores current settings to the NV memory of the instrument.
    virtual void storeToROM() override;
    virtual void clearPosition() override;
private:
	void sendStopSignal(bool wait);
    void prepairSequence(const std::vector<double> &points, const std::vector<double> &speeds);
};

//ORIENTAL MOTOR EMP401.
class XEMP401 : public XCharDeviceDriver<XMotorDriver>  {
public:
	XEMP401(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XEMP401() {}
protected:
protected:
    virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) override;
    virtual void changeConditions(const Snapshot &shot) override;
    virtual void getConditions() override;
    virtual void setTarget(const Snapshot &shot, double target) override;
    virtual void setActive(bool active) override;
    virtual void setAUXBits(unsigned int bits) override;
    virtual void setForward() override; //!< continuous rotation.
    virtual void setReverse() override;//!< continuous rotation.
    virtual void stopRotation() override; //!< stops motor and waits for deceleration.
    virtual void toHomePosition() override {}
    //! stores current settings to the NV memory of the instrumeMotornt.
    virtual void storeToROM() override;
    virtual void clearPosition() override;
private:
	void waitForCursor();
};

template <class T>
using XSharedSerialPortDriver = XModbusRTUDriver<T>; //uses only subset of ModbusDriver.
//Sigma optics piezo-assited motor controller PAMC-104.
class XSigmaPAMC104 : public XSharedSerialPortDriver<XMotorDriver>  {
public:
    XSigmaPAMC104(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XSigmaPAMC104() {}
protected:
protected:
    virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) override;
    virtual void changeConditions(const Snapshot &shot) override {}
    virtual void getConditions() override {}
    virtual void setTarget(const Snapshot &shot, double target) override;
    virtual void setActive(bool active) override {}
    virtual void setAUXBits(unsigned int bits) override {}
    virtual void setForward() override; //!< continuous rotation.
    virtual void setReverse() override;//!< continuous rotation.
    virtual void stopRotation() override; //!< stops motor and waits for deceleration.
    virtual void toHomePosition() override {}
    //! stores current settings to the NV memory of the instrumeMotornt.
    virtual void storeToROM() override {}
    virtual void clearPosition() override;
private:
    char channelChar(const Snapshot &shot);
    double m_pulsesTotal;
};

//! Kamoer M1-STP smart peristaltic pump, via RS485 (Modbus RTU).
//! One revolution of the pump head is 360 deg.; volume = Position / 360 * (ml per rev.),
//! e.g. by a calibrated entry on Position. Speed is in pulses/s of StepMotor pulses per rev.,
//! 60 by default, so that Speed reads in rpm.
//! Open loop like XSigmaPAMC104: the pump cannot report how far it has turned,
//! so Position is what was commanded, interpolated by time while running.
class XKamoerM1STP : public XModbusRTUDriver<XMotorDriver>  {
public:
    XKamoerM1STP(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XKamoerM1STP() {}
protected:
    virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) override;
    virtual void changeConditions(const Snapshot &shot) override;
    virtual void getConditions() override;
    virtual void setTarget(const Snapshot &shot, double target) override;
    virtual void setActive(bool active) override {}
    virtual void setAUXBits(unsigned int bits) override {}
    virtual void setForward() override; //!< continuous rotation.
    virtual void setReverse() override;//!< continuous rotation.
    virtual void stopRotation() override;
    virtual void toHomePosition() override; //!< throws: the pump has no home sensor.
    virtual void storeToROM() override {}
    virtual void clearPosition() override;
private:
    double rpm(const Snapshot &shot) const;
    //! [deg.] turned since m_moveStarted, as estimated from the speed.
    double estimatedMove(const XTime &now) const;
    //! Folds the move so far into m_posBase and restarts the clock.
    void rebase(const XTime &now);
    void stopRun();
    //! \arg r: [rpm]. \arg degrees: +inf for continuous rotation.
    void startRun(double r, int dir, double degrees);

    //! The open-loop state, guarded by the interface lock: getStatus() runs on
    //! the poll thread, the others on whichever thread touched the node.
    double m_posBase = 0.0; //!< [deg.] at m_moveStarted.
    double m_moveRemain = 0.0; //!< [deg.] left in this run, +inf when continuous.
    double m_degPerSec = 0.0;
    int m_dir = 0; //!< +1 forward, -1 reverse, 0 stopped.
    XTime m_moveStarted;
    //! \sa XFlexCRK::m_alarmReported
    uint16_t m_abnormalReported = 0;
};

#endif /* USERMOTOR_H_ */
