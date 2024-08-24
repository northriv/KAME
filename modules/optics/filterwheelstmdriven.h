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

#ifndef FilterWheelSTMDrivenH
#define FilterWheelSTMDrivenH
//---------------------------------------------------------------------------
#include "filterwheel.h"

class XMotorDriver;
class QMainWindow;

class XScalarEntry;

//! Base class for filter wheel rotator,
class DECLSPEC_SHARED XFilterWheelSTMDriven : public XFilterWheel {
public:
    XFilterWheelSTMDriven(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
    virtual ~XFilterWheelSTMDriven() {}
protected:
    //! Checks if the connected drivers have valid time stamps.
    //! \return true if dependency is resolved.
    //! This function must be reentrant unlike analyze().
    virtual bool checkDependency(const Snapshot &shot_this,
        const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const override;

    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &stm() const {return m_stm;}

    //! \return -1 if unstable yet.
    virtual int currentWheelPosition(const Snapshot &shot_this, const Snapshot &shot_stm) override;

    virtual void onTargetChanged(const Snapshot &shot, XValueNodeBase *) override;
private:
    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > m_stm;
    std::deque<xqcon_ptr> m_conUIs;
};

//---------------------------------------------------------------------------

#endif
