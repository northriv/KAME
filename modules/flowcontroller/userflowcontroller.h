/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/

#ifndef USERFLOWCONTROLLER_H_
#define USERFLOWCONTROLLER_H_

#include "flowcontroller.h"
#include "fujikininterface.h"

//Fujikin FCST1000 Series Mass Flow Controllers.
class XFCST1000 : public XFujikinProtocolDriver<XFlowControllerDriver>  {
public:
	XFCST1000(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XFCST1000() {}
protected:
	virtual bool isController(); //! distinguishes monitors and controllers.
	virtual bool isUnitInSLM(); //! false for SCCM.
	virtual double getFullScale();

	virtual void getStatus(double &flow, double &valve_v, bool &alarm, bool &warning);
	virtual void setValveState(bool open);
	virtual void changeControl(bool ctrl);
	virtual void changeSetPoint(double target);
	virtual void setRampTime(double time);

	enum ClassID {NetworkClass = 0x03, DeviceManagerClass = 0x64, ExceptionClass = 0x65,
		GasCalibrationClass = 0x66, FlowMeterClass = 0x68, FlowControllerClass = 0x69};
private:
};

#endif /* USERFLOWCONTROLLER_H_ */
