/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "pulserdriver.h"
#include "chardevicedriver.h"
#include <vector>

//! My pulser driver
class XH8Pulser : public XCharDeviceDriver<XPulser> {
public:
	XH8Pulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XH8Pulser() {}

	struct Payload : public XPulser::Payload {
	private:
		friend class XH8Pulser;
		struct h8ushort {unsigned char msb; unsigned char lsb;};
		std::vector<h8ushort> m_zippedPatterns;
	};

	//! time resolution [ms]
    virtual double resolution() const;
protected:
	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open() throw (XInterface::XInterfaceError &);
    //! Sends patterns to pulser or turn-off
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern);
    //! Converts RelPatList to native patterns
    virtual void createNativePatterns(Transaction &tr);
    virtual double resolutionQAM() const {return 0.0;}
    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const;
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return false;}
private:
	//! Add 1 pulse pattern
	//! \param term a period to next pattern
	//! \param pattern a pattern for digital, to appear
	int pulseAdd(Transaction &tr, uint64_t term, uint16_t pattern);
};
