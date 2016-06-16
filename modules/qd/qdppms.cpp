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
#include "qdppms.h"
#include "analyzer.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, QDPPMS6000, "Quantum Design PPMS low-level interface");

XQDPPMS6000::XQDPPMS6000(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XPrimaryDriverWithThread>(name, runtime, ref(tr_meas), meas),
    m_temp(create<XScalarEntry>("Temp", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_temp_rotator(create<XScalarEntry>("TempRotator", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_field(create<XScalarEntry>("Field", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_position(create<XScalarEntry>("Position", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_heliumLevel(create<XDoubleNode>("HeliumLevel", true)),
    m_form(new FrmQDPPMS(g_pFrmMain)) {
    meas->scalarEntries()->insert(tr_meas, m_temp);
    meas->scalarEntries()->insert(tr_meas, m_temp_rotator);
    meas->scalarEntries()->insert(tr_meas, m_field);
    meas->scalarEntries()->insert(tr_meas, m_position);

    m_form->setWindowTitle(XString("QDPPMS - " + getLabel() ));

    m_conTemp = xqcon_create<XQLCDNumberConnector>(temp()->value(), m_form->m_lcdTemp);
    m_conTempRotator = xqcon_create<XQLCDNumberConnector>(temp_rotator()->value(), m_form->m_lcdTempRotator);
    m_conField = xqcon_create<XQLCDNumberConnector>(field()->value(), m_form->m_lcdField);
    m_conPosition = xqcon_create<XQLCDNumberConnector>(position()->value(), m_form->m_lcdPosition);
    m_conHeliumLevel = xqcon_create<XQLCDNumberConnector>(heliumLevel(), m_form->m_lcdHeliumLevel);

    interface()->setEOS("");
    interface()->setSerialEOS("\r\n");
}
void
XQDPPMS6000::showForms() {
    m_form->showNormal();
    m_form->raise();
}
void
XQDPPMS6000::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    tr[ *this].m_sampleTemp = reader.pop<float>();
    tr[ *this].m_sampleTempRotator = reader.pop<float>();
    tr[ *this].m_magnetField = reader.pop<float>();
    tr[ *this].m_samplePosition = reader.pop<float>();
    m_temp->value(tr, tr[ *this].m_sampleTemp);
    m_temp_rotator->value(tr, tr[ *this].m_sampleTempRotator);
    m_field->value(tr, tr[ *this].m_magnetField);
    m_position->value(tr, tr[*this].m_samplePosition);
}
void
XQDPPMS6000::visualize(const Snapshot &shot) {
}



void *
XQDPPMS6000::execute(const atomic<bool> &terminated) {

    while( !terminated) {
        msecsleep(100);
        double magnet_field;
        double sample_temp;
        double sample_temp_rotator;
        double sample_position;
        double helium_level;
        int is_user_temp;

        try {
            // Reading....
            interface()->query("GetDat? 2");
            if( interface()->scanf("2,%*lf,%lf", &sample_temp) != 1)
                throw XInterface::XConvError(__FILE__, __LINE__);
            interface()->query("USERTEMP?");
            if( interface()->scanf("%d,%*lf,%*lf,%*lf,%*lf",&is_user_temp) != 1)
                throw XInterface::XConvError(__FILE__, __LINE__);
            else if(is_user_temp){
                int user_temp_channel = lrint(pow(2,is_user_temp));
                interface()->queryf("GetDat? %d", user_temp_channel);
                if( interface()->scanf("%*d,%*lf,%lf", &sample_temp_rotator) != 1)
                    throw XInterface::XConvError(__FILE__, __LINE__);
            }
            interface()->query("GetDat? 4");
            if( interface()->scanf("4,%*lf,%lf", &magnet_field) != 1)
                throw XInterface::XConvError(__FILE__, __LINE__);
            interface()->query("GetDat? 8");
            if( interface()->scanf("8,%*lf,%lf", &sample_position) != 1)
                throw XInterface::XConvError(__FILE__, __LINE__);
            interface()->query("LEVEL?");
            if( interface()->scanf("%lf,%*lf", &helium_level) != 1)
                throw XInterface::XConvError(__FILE__, __LINE__);
        }
        catch (XKameError &e) {
            e.print(getLabel() + "; ");
            continue;
        }
        shared_ptr<RawData> writer(new RawData);
        writer->push((float)sample_temp);
        if(is_user_temp){
            writer->push((float)sample_temp_rotator);
        }
        else{
            writer->push(0.0f);
        }
        writer->push((float)magnet_field);
        writer->push((float)sample_position);

        finishWritingRaw(writer, XTime::now(), XTime::now());

        for(Transaction tr( *this);; ++tr) {
            Snapshot &shot(tr);
            tr[ *heliumLevel()] = helium_level;
            if(tr.commit())
                break;
        }
    }

    return NULL;
}
