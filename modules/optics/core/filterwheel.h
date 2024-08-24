/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
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

#ifndef FilterWheelH
#define FilterWheelH
//---------------------------------------------------------------------------
#include "secondarydriver.h"
#include "xnodeconnector.h"

class XDigitalCamera;
class QMainWindow;
class Ui_FrmFilterWheel;
typedef QForm<QMainWindow, Ui_FrmFilterWheel> FrmFilterWheel;

class XScalarEntry;

//! Base class for filter wheel rotator,
class DECLSPEC_SHARED XFilterWheel : public XSecondaryDriver {
public:
    XFilterWheel(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
    virtual ~XFilterWheel();
	//! Shows all forms belonging to driver.
    virtual void showForms() override;

    //! driver specific part below
    static constexpr unsigned int MaxFilterCount = 6;
    virtual unsigned int filterCount() const {return 6;}
    const shared_ptr<XUIntNode> &target() const {return m_target;}
    const shared_ptr<XStringNode> &filterLabel(unsigned int index) const {return m_filterLabels.at(index);}
    const shared_ptr<XUIntNode> &dwellCount(unsigned int index) const {return m_dwellCounts.at(index);}
    const shared_ptr<XDoubleNode> &stmAngle(unsigned int index) const {return m_stmAngles.at(index);}
    const shared_ptr<XDoubleNode> &angleErrorWithin() const {return m_angleErrorWithin;} //!< [deg.]
    const shared_ptr<XDoubleNode> &waitAfterMove() const { return m_waitAfterMove;} //!< [s]
    const shared_ptr<XBoolNode> &goAroundAfterShot() const {return m_goAroundAfterShot;}

    const shared_ptr<XScalarEntry> &currentWheelIndex() const {return m_currentWheelIndex;}

    struct Payload : public XSecondaryDriver::Payload {
        unsigned int dwellIndex() const {return m_dwellIndex;}
        int wheelIndexOfFrame(const XTime &time) const {
            return (time == m_timeLastFrame) ? m_wheelIndexOfLastFrame : -1;} //!< -1: unknown
    protected:
        friend class XFilterWheel;
        unsigned int m_dwellIndex = 0;
        unsigned int m_nextWheelIndex = 0;
        int wheelIndex() const {return m_wheelIndex;} //!< -1: not ready
        int m_wheelIndex = 0;
        int m_wheelIndexOfLastFrame = 0;
        XTime m_timeLastFrame = {};
        XTime m_timeFilterMoved;
    };

    const shared_ptr<XItemNode<XDriverList, XDigitalCamera> > &camera() const {return m_camera;}

protected:
    //! This function is called when a connected driver emit a signal
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
         XDriver *emitter) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;

    //! \return -1 if unstable yet.
    virtual int currentWheelPosition(const Snapshot &shot_this, const Snapshot &shot_stm) = 0;

    const shared_ptr<XItemNode<XDriverList, XDigitalCamera> > m_camera;
    const shared_ptr<XUIntNode> m_target;
    std::deque<shared_ptr<XStringNode>> m_filterLabels;
    std::deque<shared_ptr<XUIntNode>> m_dwellCounts;
    std::deque<shared_ptr<XDoubleNode>> m_stmAngles;
    const shared_ptr<XDoubleNode> m_angleErrorWithin; //!< [deg.]
    const shared_ptr<XDoubleNode> m_waitAfterMove; //!< [s]
    const shared_ptr<XBoolNode> m_goAroundAfterShot;
    const shared_ptr<XScalarEntry> m_currentWheelIndex;

    const qshared_ptr<FrmFilterWheel> m_form;

    virtual void onTargetChanged(const Snapshot &shot, XValueNodeBase *) = 0;
private:
    shared_ptr<Listener> m_lsnOnTargetChanged;

    std::deque<xqcon_ptr> m_conUIs;
};

//---------------------------------------------------------------------------

#endif
