/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef USERLASERMODULE_H
#define USERLASERMODULE_H

#include "lasermodule.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------

//! COHERENT Stringray laser module
class XCoherentStingray : public XCharDeviceDriver<XLaserModule> {
public:
    XCoherentStingray(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XCoherentStingray() {}
protected:
    virtual ModuleStatus readStatus() override;
    virtual void onEnabledChanged(const Snapshot &shot, XValueNodeBase *) override;
private:
};

#endif // USERLASERMODULE_H
