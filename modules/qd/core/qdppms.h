/***************************************************************************
        Copyright (C) 2002-2016 Shota Suetsugu and Kentaro Kitagawa
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

//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmQDPPMS;
typedef QForm<QMainWindow, Ui_FrmQDPPMS> FrmQDPPMS;

//! GPIB/serial interface for Quantum Design PPMS Model6000 or later
class DECLSPEC_SHARED XQDPPMS : public XPrimaryDriverWithThread {
public:
    XQDPPMS(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! usually nothing to do
    virtual ~XQDPPMS() = default;
    //! Shows all forms belonging to driver
    virtual void showForms() override;

    struct Payload : public XPrimaryDriver::Payload {
        double temp() const {return m_sampleTemp;}
        double user_temp() const {return m_sampleUserTemp;}
        double magnetField() const {return m_magnetField;}
        double position() const {return m_samplePosition;}
    private:
        friend class XQDPPMS;
        double m_magnetField; //Oe
        double m_samplePosition;
        double m_sampleTemp;
        double m_sampleUserTemp;
    };
protected:
    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;

    //! driver specific part below
    const shared_ptr<XScalarEntry> &field() const {return m_field;} //Oe
    const shared_ptr<XScalarEntry> &position() const {return m_position;}
    const shared_ptr<XScalarEntry> &temp() const {return m_temp;}
    const shared_ptr<XScalarEntry> &user_temp() const {return m_user_temp;}
    const shared_ptr<XDoubleNode> &heliumLevel() const {return m_heliumLevel;}

    const shared_ptr<XDoubleNode> &targetField() const {return m_targetField;}
    const shared_ptr<XDoubleNode> &fieldSweepRate() const {return m_fieldSweepRate;}
    const shared_ptr<XComboNode> &fieldApproachMode() const {return m_fieldApproachMode;}
    const shared_ptr<XComboNode> &fieldMagnetMode() const {return m_fieldMagnetMode;}
    const shared_ptr<XStringNode> &fieldStatus() const {return m_fieldStatus;}

    const shared_ptr<XDoubleNode> &targetPosition() const {return m_targetPosition;}
    const shared_ptr<XComboNode> &positionApproachMode() const {return m_positionApproachMode;}
    const shared_ptr<XIntNode> &positionSlowDownCode() const {return m_positionSlowDownCode;}
    const shared_ptr<XStringNode> &positionStatus() const {return m_positionStatus;}

    const shared_ptr<XDoubleNode> &targetTemp() const {return m_targetTemp;}
    const shared_ptr<XDoubleNode> &tempSweepRate() const {return m_tempSweepRate;}
    const shared_ptr<XComboNode> &tempApproachMode() const {return m_tempApproachMode;}
    const shared_ptr<XStringNode> &tempStatus() const {return m_tempStatus;}
    const shared_ptr<XIntNode> &userTempChannel() const {return m_userTempChannel;}
protected:
    virtual void setField(double field, double rate, int approach_mode, int magnet_mode) = 0; //T
    virtual void setPosition(double position, int mode, int slow_down_code) = 0;
    virtual void setTemp(double temp, double rate, int approach_mode) = 0;
    virtual double getField() = 0; //Oe
    virtual double getPosition() = 0;
    virtual double getTemp() = 0;
    virtual double getUserTemp(int channel) = 0;
    virtual double getHeliumLevel() = 0;
    virtual int getStatus() = 0;
private:
    virtual void onFieldChanged(const Snapshot &shot,  XValueNodeBase *);
    virtual void onPositionChanged(const Snapshot &shot,  XValueNodeBase *);
    virtual void onTempChanged(const Snapshot &shot,  XValueNodeBase *);

    const shared_ptr<XScalarEntry> m_field, m_position, m_temp, m_user_temp;

    const shared_ptr<XDoubleNode> m_heliumLevel;

    const shared_ptr<XDoubleNode> m_targetField, m_fieldSweepRate;
    const shared_ptr<XComboNode> m_fieldApproachMode, m_fieldMagnetMode;
    const shared_ptr<XStringNode> m_fieldStatus;

    const shared_ptr<XDoubleNode> m_targetPosition;
    const shared_ptr<XComboNode> m_positionApproachMode;
    const shared_ptr<XIntNode> m_positionSlowDownCode;
    const shared_ptr<XStringNode> m_positionStatus;

    const shared_ptr<XDoubleNode> m_targetTemp, m_tempSweepRate;
    const shared_ptr<XComboNode> m_tempApproachMode;
    const shared_ptr<XStringNode> m_tempStatus;
    const shared_ptr<XIntNode> m_userTempChannel;

    shared_ptr<Listener> m_lsnFieldSet, m_lsnTempSet, m_lsnPositionSet;

    std::deque<xqcon_ptr> m_conUIs;

    const qshared_ptr<FrmQDPPMS> m_form;

    virtual void *execute(const atomic<bool> &) override;
};

#endif
