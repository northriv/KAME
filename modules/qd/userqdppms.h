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
#ifndef userqdppmsH
#define userqdppmsH

#include "chardevicedriver.h"
#include "qdppms.h"
//---------------------------------------------------------------------------

//! GPIB/serial interface for Quantum Design PPMS Model6000 or later
class DECLSPEC_SHARED XQDPPMS6000 : public XCharDeviceDriver<XQDPPMS> {
public:
    XQDPPMS6000(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! usually nothing to do
    virtual ~XQDPPMS6000() = default;

protected:
protected:
    virtual double getField();
    virtual double getPosition();
    virtual double getTemp();
    virtual double getTempRotator();
    virtual double getHeliumLevel();
private:
};

#endif
