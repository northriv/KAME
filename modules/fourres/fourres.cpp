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
#include "fourres.h"
#include "ui_fourresform.h"
#include "interface.h"
#include "analyzer.h"

XFourRes::XFourRes(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
	: XSecondaryDriver(name, runtime, ref(tr_meas), meas),
	  m_resistance(create<XScalarEntry>("Resistance", false,
								   dynamic_pointer_cast<XDriver>(shared_from_this()))),
	  m_dmm(create<XItemNode < XDriverList, XDMM> >(
		  "DMM", false, ref(tr_meas), meas->drivers(), true)),
	  m_dcsource(create<XItemNode < XDriverList, XDCSource> >(
		  "DCSource", false, ref(tr_meas), meas->drivers(), true)),
	  m_control(create<XBoolNode>("Control", false)),
	  m_form(new FrmFourRes(g_pFrmMain)) {

    m_form->setWindowTitle(i18n("Resistance Measurement with Switching Polarity - ") + getLabel() );

    meas->scalarEntries()->insert(tr_meas, resistance());

    connect(dmm());
    connect(dcsource());
	for(Transaction tr( *this);; ++tr) {
		tr[ *control()] = false;
		tr[ *this].value_inverted = 0.0;

		m_lsnOnControlChanged = tr[ *control()].onValueChanged().connectWeakly(
			shared_from_this(), &XFourRes::onControlChanged);

		if(tr.commit())
			break;
	}
	m_conControl = xqcon_create<XQToggleButtonConnector>(m_control, m_form->m_ckbControl);
	m_conDMM = xqcon_create<XQComboBoxConnector>(m_dmm, m_form->m_cmbDMM, ref(tr_meas));
	m_conDCSource = xqcon_create<XQComboBoxConnector>(m_dcsource, m_form->m_cmbDCSource, ref(tr_meas));
	m_conRes = xqcon_create<XQLCDNumberConnector> (m_resistance->value(), m_form->m_lcdRes);
}
XFourRes::~XFourRes () {
}
void
XFourRes::showForms() {
	m_form->show();
	m_form->raise();
}
void
XFourRes::onControlChanged(const Snapshot &shot, XValueNodeBase *node) {
}
bool
XFourRes::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
    shared_ptr<XDMM> dmm__ = shot_this[ *dmm()];
    shared_ptr<XDCSource> dcsource__ = shot_this[ *dcsource()];
    if( !dmm__ || !dcsource__) return false;
    if(emitter != dmm__.get()) return false;
    if( !shot_this[ *control()]) return false;
	return true;
}

void
XFourRes::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) throw (XRecordError&) {
	Snapshot &shot_this(tr);
    shared_ptr<XDMM> dmm__ = shot_this[ *dmm()];
    shared_ptr<XDCSource> dcsource__ = shot_this[ *dcsource()];

	double curr = shot_others[ *dcsource__->value()];
	double var = shot_emitter[ *dmm__].value();

	if(curr < 0) {
		tr[ *this].value_inverted = var;
	}
	else {
		if(shot_this[ *this].value_inverted != 0.0)
			resistance()->value(tr, (shot_this[ *this].value_inverted + var) / 2);
	}

	trans[ *dcsource__->output()] = -curr; //Invert polarity.
}

void
XFourRes::visualize(const Snapshot &shot) {
}
