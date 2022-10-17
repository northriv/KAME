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
#ifndef fourresH
#define fourresH
//---------------------------------------------------------------------------
#include "dmm.h"
#include "dcsource.h"
#include "secondarydriver.h"

class XScalarEntry;
class Ui_FrmFourRes;
typedef QForm<QMainWindow, Ui_FrmFourRes> FrmFourRes;

//! Measure Resistance By Switching Polarity of DC Source
class XFourRes : public XSecondaryDriver {
public:
	XFourRes(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	~XFourRes ();

	//! Shows all forms belonging to driver
	virtual void showForms();
protected:
	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
	//! Checks if the connected drivers have valid time stamps.
	//! \return true if dependency is resolved.
	//! This function must be reentrant unlike analyze().
	virtual bool checkDependency(const Snapshot &shot_this,
		const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) const;
public:
	struct Payload : public XSecondaryDriver::Payload {
	private:
			friend class XFourRes;
			double value_inverted;
	};

	const shared_ptr<XScalarEntry> &resistance() const {return m_resistance;}

	const shared_ptr<XItemNode < XDriverList, XDMM > > &dmm() const {return m_dmm;}
	const shared_ptr<XItemNode < XDriverList, XDCSource > > &dcsource() const {return m_dcsource;}

    const shared_ptr<XUIntNode> &dmmChannel() const {return m_dmmChannel;}
    const shared_ptr<XBoolNode> &control() const {return m_control;}

private:
	const shared_ptr<XScalarEntry> m_resistance;

	const shared_ptr<XItemNode < XDriverList, XDMM> > m_dmm;
	const shared_ptr<XItemNode < XDriverList, XDCSource > > m_dcsource;
    const shared_ptr<XUIntNode> m_dmmChannel;
    const shared_ptr<XBoolNode> m_control;

    std::deque<xqcon_ptr> m_conUIs;
	const qshared_ptr<FrmFourRes> m_form;
};
  
#endif
