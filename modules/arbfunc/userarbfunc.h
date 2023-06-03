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
//---------------------------------------------------------------------------

#ifndef USERARBFUNCH
#define USERARBFUNCH

#include "arbfunc.h"
#include "chardevicedriver.h"
#include "charinterface.h"


class XArbFuncGenSCPI : public XCharDeviceDriver<XArbFuncGen> {
public:
    XArbFuncGenSCPI(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XArbFuncGenSCPI() {}

protected:
    virtual void changeOutput(bool active) override;
    virtual void changePulseCond() override;

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;
private:

};

#endif
