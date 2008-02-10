/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
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
#ifndef userlevelmeterH
#define userlevelmeterH

#include "levelmeter.h"
#include "oxforddriver.h"
//---------------------------------------------------------------------------
//OXFORD ILM helim level meter
class XILM : public XOxfordDriver<XLevelMeter>
{
	XNODE_OBJECT
protected:
	XILM(const char *name, bool runtime,
		   const shared_ptr<XScalarEntryList> &scalarentries,
		   const shared_ptr<XInterfaceList> &interfaces,
		   const shared_ptr<XThermometerList> &thermometers,
		   const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XILM() {}
protected:
	virtual double getLevel(unsigned int ch);
};

//LakeShore LM-500 level meter
class XLM500 : public XCharDeviceDriver<XLevelMeter>
{
	XNODE_OBJECT
protected:
	XLM500(const char *name, bool runtime,
		   const shared_ptr<XScalarEntryList> &scalarentries,
		   const shared_ptr<XInterfaceList> &interfaces,
		   const shared_ptr<XThermometerList> &thermometers,
		   const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XLM500() {}
protected:
	virtual double getLevel(unsigned int ch);
};
#endif

