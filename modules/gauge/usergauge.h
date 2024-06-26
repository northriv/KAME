/***************************************************************************
        Copyright (C) 2002-2020 Kentaro Kitagawa
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
#ifndef usergaugeH
#define usergaugeH

#include "gauge.h"
#include "pfeifferprotocol.h"
//---------------------------------------------------------------------------
//Pfeiffer TPG361/362 Gauge Measurement Unit
class XTPG362 : public XPfeifferProtocolDriver<XGauge> {
public:
    XTPG362(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XTPG362() {}
protected:
    virtual double getPressure(unsigned int ch);
};

#endif

