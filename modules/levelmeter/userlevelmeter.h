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
//---------------------------------------------------------------------------
#ifndef userlevelmeterH
#define userlevelmeterH

#include "levelmeter.h"
#include "oxforddriver.h"
//---------------------------------------------------------------------------
//OXFORD ILM helim level meter
class XILM : public XOxfordDriver<XLevelMeter> {
public:
	XILM(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XILM() {}
protected:
	virtual double getLevel(unsigned int ch);
};

//LakeShore LM-500 level meter
class XLM500 : public XCharDeviceDriver<XLevelMeter> {
public:
	XLM500(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XLM500() {}
protected:
	virtual double getLevel(unsigned int ch);
};
#endif

