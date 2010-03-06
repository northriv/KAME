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
//---------------------------------------------------------------------------

#ifndef thermometerH
#define thermometerH

#include "xnode.h"
#include "xlistnode.h"
#include "cspline.h"

class XThermometer : public XNode {
public:
	XThermometer(const char *name, bool runtime);
	virtual ~XThermometer() {}

	virtual double getTemp(double res) const = 0;
	virtual double getRawValue(double temp) const = 0;

	const shared_ptr<XDoubleNode> &tempMin() const {return m_tempMin;} 
	const shared_ptr<XDoubleNode> &tempMax() const {return m_tempMax;} 
private:
	const shared_ptr<XDoubleNode> m_tempMin, m_tempMax;
};

class XThermometerList : public XCustomTypeListNode<XThermometer> {
public:
	XThermometerList(const char *name, bool runtime);
	virtual ~XThermometerList() {}

	DEFINE_TYPE_HOLDER
protected:
	virtual shared_ptr<XNode> createByTypename(
        const XString &type, const XString &name) {
		shared_ptr<XNode> ptr = (creator(type))(name.c_str(), false);
		if(ptr) insert(ptr);
		return ptr;
	}
};

//chebichev polynominal
class XLakeShore : public XThermometer {
public:
	XLakeShore(const char *name, bool runtime);
	virtual ~XLakeShore() {}
  
	double getTemp(double res) const;
	double getRawValue(double temp) const;
    
	const shared_ptr<XDoubleNode> &resMin() const {return m_resMin;}
	const shared_ptr<XDoubleNode> &resMax() const {return m_resMax;}
	typedef XListNode<XDoubleNode> XDoubleListNode;
	const shared_ptr<XDoubleListNode> &zu() const {return m_zu;}
	const shared_ptr<XDoubleListNode> &zl() const {return m_zl;}
	typedef XListNode<XDoubleListNode> XDouble2DNode;
	const shared_ptr<XDouble2DNode> &ai() const {return m_ai;}
private:
	const shared_ptr<XDoubleNode> m_resMin, m_resMax;
	const shared_ptr<XDoubleListNode> m_zu, m_zl;
	const shared_ptr<XDouble2DNode> m_ai;

};

class XScientificInstruments : public XThermometer {
public:
	XScientificInstruments(const char *name, bool runtime);
	virtual ~XScientificInstruments() {}

	double getTemp(double res) const;
	double getRawValue(double temp) const;

	const shared_ptr<XDoubleNode> &resMin() const {return m_resMin;}
	const shared_ptr<XDoubleNode> &resMax() const {return m_resMax;}
	typedef XListNode<XDoubleNode> XDoubleListNode;
	const shared_ptr<XDoubleListNode> &abcde() const {return m_abcde;}
	const shared_ptr<XDoubleListNode> &abc() const {return m_abc;}
	const shared_ptr<XDoubleNode> &rCrossover() const {return m_rCrossover;}
private:
	const shared_ptr<XDoubleNode> m_resMin, m_resMax;
	const shared_ptr<XDoubleListNode> m_abcde, m_abc;
	const shared_ptr<XDoubleNode> m_rCrossover;    
};

//! Cubic (natural) spline approximation.
class XApproxThermometer : public XThermometer {
public:
	XApproxThermometer(const char *name, bool runtime);

	double getTemp(double res) const;
	double getRawValue(double temp) const;

	typedef XListNode<XDoubleNode> XDoubleListNode;
	const shared_ptr<XDoubleListNode> &resList() const {return m_resList;}
	const shared_ptr<XDoubleListNode> &tempList() const {return m_tempList;}  
private:
	const shared_ptr<XDoubleListNode> m_resList, m_tempList;
	mutable atomic_shared_ptr<CSplineApprox> m_approx, m_approx_inv;
};

//---------------------------------------------------------------------------
#endif
