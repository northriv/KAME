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
#include "fourres.h"
#include "ui_fourresform.h"
#include "interface.h"
#include "analyzer.h"

REGISTER_TYPE(XDriverList, FourRes, "Resistance Measurement with Polarity Switching");

XFourRes::XFourRes(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
	: XSecondaryDriver(name, runtime, ref(tr_meas), meas),
	  m_resistance(create<XScalarEntry>("Resistance", false,
								   dynamic_pointer_cast<XDriver>(shared_from_this()))),
	  m_dmm(create<XItemNode < XDriverList, XDMM> >(
		  "DMM", false, ref(tr_meas), meas->drivers(), true)),
	  m_dcsource(create<XItemNode < XDriverList, XDCSource> >(
		  "DCSource", false, ref(tr_meas), meas->drivers(), true)),
      m_dmmChannel(create<XUIntNode>("DMMChannel", false)),
      m_control(create<XBoolNode>("Control", true)),
	  m_form(new FrmFourRes) {

    m_form->setWindowTitle(i18n("Resistance Measurement with Switching Polarity - ") + getLabel() );

    meas->scalarEntries()->insert(tr_meas, resistance());

    connect(dmm());
    connect(dcsource());
	iterate_commit([=](Transaction &tr){
        tr[ *dmmChannel()] = 1;
        tr[ *control()] = false;
        tr[ *this].value_inverted = 0.0;
    });
	m_conControl = xqcon_create<XQToggleButtonConnector>(m_control, m_form->m_ckbControl);
	m_conDMM = xqcon_create<XQComboBoxConnector>(m_dmm, m_form->m_cmbDMM, ref(tr_meas));
	m_conDCSource = xqcon_create<XQComboBoxConnector>(m_dcsource, m_form->m_cmbDCSource, ref(tr_meas));
	m_conRes = xqcon_create<XQLCDNumberConnector> (m_resistance->value(), m_form->m_lcdRes);
}
XFourRes::~XFourRes () {
}
void
XFourRes::showForms() {
    m_form->showNormal();
	m_form->raise();
}
bool
XFourRes::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
    shared_ptr<XDMM> dmm__ = shot_this[ *dmm()];
    shared_ptr<XDCSource> dcsource__ = shot_this[ *dcsource()];
    if( !dmm__ || !dcsource__) return false;
    if(emitter != dmm__.get()) return false;
	return true;
}

void
XFourRes::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) {
	Snapshot &shot_this(tr);
    shared_ptr<XDMM> dmm__ = shot_this[ *dmm()];
    shared_ptr<XDCSource> dcsource__ = shot_this[ *dcsource()];

    if(shot_emitter[ *dmm__].timeAwared() < shot_others[ *dcsource__].time() + 0.05)
		throw XSkippedRecordError(__FILE__, __LINE__);

    double curr = shot_others[ *dcsource__->value()];
    double var = shot_emitter[ *dmm__].value(shot_this[ *dmmChannel()] - 1);

	if(curr < 0) {
		tr[ *this].value_inverted = var;
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	else {
		if(shot_this[ *this].value_inverted == 0.0)
			throw XSkippedRecordError(__FILE__, __LINE__);
		resistance()->value(tr, (var - shot_this[ *this].value_inverted) / 2 / curr);
		tr[ *this].value_inverted = 0.0;
	}
}

void
XFourRes::visualize(const Snapshot &shot) {
	if(shot[ *control()]) {
		shared_ptr<XDCSource> dcsource__ = shot[ *dcsource()];
        dcsource__->iterate_commit([=](Transaction &tr){
			double curr = tr[ *dcsource__->value()];
			tr[ *dcsource__->value()] = -curr; //Invert polarity.
            msecsleep(100);
        });
	}
}
