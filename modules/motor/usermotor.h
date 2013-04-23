/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

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
	virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready);
	virtual void changeConditions(const Snapshot &shot);
	virtual void getConditions(Transaction &tr);
	virtual void setTarget(const Snapshot &shot, double target);
	virtual void setActive(bool active);
	virtual void setAUXBits(unsigned int bits);
	virtual void setForward(); //!< continuous rotation.
	virtual void setReverse();//!< continuous rotation.
	virtual void stopRotation(); //!< stops motor and waits for deceleration.
	//! stores current settings to the NV memory of the instrumeMotornt.
	virtual void storeToROM();
	virtual void clearPosition();
private:
	void sendStopSignal(bool wait);
};

//ORIENTAL MOTOR FLEX AR/DG2 series.
class XFlexAR : public XFlexCRK {
public:
	XFlexAR(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XFlexCRK(name, runtime, ref(tr_meas), meas) {}
	virtual ~XFlexAR() {}
protected:
protected:
	virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready);
	virtual void changeConditions(const Snapshot &shot);
	virtual void getConditions(Transaction &tr);
	virtual void setTarget(const Snapshot &shot, double target);
	virtual void setActive(bool active);
	virtual void setAUXBits(unsigned int bits);
	virtual void setForward(); //!< continuous rotation.
	virtual void setReverse();//!< continuous rotation.
	virtual void stopRotation();//!< stops motor and waits for deceleration.
	//! stores current settings to the NV memory of the instrument.
	virtual void storeToROM();
	virtual void clearPosition();
private:
	void sendStopSignal(bool wait);
};

//ORIENTAL MOTOR EMP401.
class XEMP401 : public XCharDeviceDriver<XMotorDriver>  {
public:
	XEMP401(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XEMP401() {}
protected:
protected:
	virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready);
	virtual void changeConditions(const Snapshot &shot);
	virtual void getConditions(Transaction &tr);
	virtual void setTarget(const Snapshot &shot, double target);
	virtual void setActive(bool active);
	virtual void setAUXBits(unsigned int bits);
	virtual void setForward(); //!< continuous rotation.
	virtual void setReverse();//!< continuous rotation.
	virtual void stopRotation(); //!< stops motor and waits for deceleration.
	//! stores current settings to the NV memory of the instrumeMotornt.
	virtual void storeToROM();
	virtual void clearPosition();
private:
	void waitForCursor();
};

#endif /* USERMOTOR_H_ */
