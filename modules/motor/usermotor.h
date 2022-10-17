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

#ifndef USERMOTOR_H_
#define USERMOTOR_H_

#include "motor.h"
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
	//! stores current settings to the NV memory of the instrumeMotornt.
    virtual void storeToROM() override;
    virtual void clearPosition() override;
private:
	void sendStopSignal(bool wait);
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
        const std::vector<std::vector<double>> &speeds, const std::vector<const shared_ptr<XMotorDriver>> &slaves) override;
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
	//! stores current settings to the NV memory of the instrumeMotornt.
    virtual void storeToROM() override;
    virtual void clearPosition() override;
private:
	void waitForCursor();
};

#endif /* USERMOTOR_H_ */
