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
//---------------------------------------------------------------------------
#include "ui_dmmform.h"
#include <QStatusBar>
#include "dmm.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XDMM::XDMM(const char *name, bool runtime, 
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas, unsigned int max_num_channels) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_function(create<XComboNode>("Function", false)),
    m_waitInms(create<XUIntNode>("WaitInms", false)),
    m_form(new FrmDMM),
    m_maxNumOfChannels(max_num_channels) {

    for(unsigned int i = 0; i < maxNumOfChannels(); ++i) {
        m_entries.push_back(create<XScalarEntry>(
            (maxNumOfChannels() > 1) ? formatString("Channel%u", i + 1).c_str() : "Value",
            false, dynamic_pointer_cast<XDriver>(shared_from_this())));
        meas->scalarEntries()->insert(tr_meas, m_entries.back());
    }
	iterate_commit([=](Transaction &tr){
		tr[ *m_waitInms] = 100;
    });
	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("DMM - ") + getLabel() );
	m_function->setUIEnabled(false);
	m_waitInms->setUIEnabled(false);
	m_conFunction = xqcon_create<XQComboBoxConnector>(m_function, m_form->m_cmbFunction, Snapshot( *m_function));
	m_conWaitInms = xqcon_create<XQSpinBoxUnsignedConnector>(m_waitInms, m_form->m_numWait);
}

void
XDMM::showForms() {
//! impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XDMM::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    if(maxNumOfChannels() > 1) {
        unsigned int num_ch = reader.pop<uint32_t>();
        for(unsigned int i = 0; i < num_ch; ++i) {
            double x = reader.pop<double>();
            m_entries[i]->value(tr, x);
            tr[ *this].write_(x, i);
        }
    }
    else {
        double x = reader.pop<double>();
        tr[ *this].write_(x);
        m_entries[0]->value(tr, x);
    }
}
void
XDMM::visualize(const Snapshot &shot) {
}

void
XDMM::onFunctionChanged(const Snapshot &shot, XValueNodeBase *node) {
    try {
        changeFunction();
    }
    catch (XKameError &e) {
		e.print(getLabel() + " " + i18n("DMM Error"));
    }
}

void *
XDMM::execute(const atomic<bool> &terminated) {
	changeFunction();

    m_function->setUIEnabled(true);
    m_waitInms->setUIEnabled(true);
        
	iterate_commit([=](Transaction &tr){
	    m_lsnOnFunctionChanged =
	        tr[ *function()].onValueChanged().connectWeakly(
				shared_from_this(), &XDMM::onFunctionChanged);
    });
	while( !terminated) {
		msecsleep( ***waitInms());
		if(( **function())->to_str().empty()) continue;
      
		auto writer = std::make_shared<RawData>();
		XTime time_awared = XTime::now();
		// try/catch exception of communication errors
		try {
            if(maxNumOfChannels() > 1) {
                auto var = oneShotMultiRead();
                writer->push(static_cast<uint32_t>(var.size()));
                for(double x: var)
                    writer->push(x);
            }
            else {
                double x = oneShotRead();
                writer->push(x);
            }
		}
		catch (XKameError &e) {
			e.print(getLabel() + " " + i18n("DMM Read Error"));
			continue;
		}
		finishWritingRaw(writer, time_awared, XTime::now());
	}
    
    m_function->setUIEnabled(false);
    m_waitInms->setUIEnabled(false);

    m_lsnOnFunctionChanged.reset();

    return NULL;
}

