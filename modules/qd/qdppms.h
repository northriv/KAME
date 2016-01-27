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
#ifndef qdppmsH
#define qdppmsH

#include "chardevicedriver.h"
//---------------------------------------------------------------------------

#include "primarydriverwiththread.h"
#include "xnodeconnector.h"
#include "ui_qdppmsform.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmQDPPMS;
typedef QForm<QMainWindow, Ui_FrmQDPPMS> FrmQDPPMS;

//! GPIB/serial interface for Quantum Design PPMS Model6000 or later
class DECLSPEC_SHARED XQDPPMS6000 : public XCharDeviceDriver<XPrimaryDriverWithThread> {
public:
    XQDPPMS6000(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! usually nothing to do
    virtual ~XQDPPMS6000() {}
    //! Shows all forms belonging to driver
    virtual void showForms();

    struct Payload : public XPrimaryDriver::Payload {
        double temp() const {return m_sampleTemp;}
        double magnetField() const {return m_magnetField;}
        double position() const {return m_samplePosition;}
    private:
        friend class XQDPPMS6000;
        double m_magnetField;
        double m_samplePosition;
        double m_sampleTemp;
    };
protected:
    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot);

    //! driver specific part below
    const shared_ptr<XScalarEntry> &field() const {return m_field;}
    const shared_ptr<XScalarEntry> &position() const {return m_position;}
    const shared_ptr<XScalarEntry> &temp() const {return m_temp;}
    const shared_ptr<XDoubleNode> &heliumLevel() const {return m_heliumLevel;}

protected:
private:
    const shared_ptr<XScalarEntry> m_field, m_position, m_temp;

    const shared_ptr<XDoubleNode> m_heliumLevel;

    xqcon_ptr m_conField, m_conTemp, m_conPosition, m_conHeliumLevel;

    const qshared_ptr<FrmQDPPMS> m_form;

    void *execute(const atomic<bool> &);
};

#endif
