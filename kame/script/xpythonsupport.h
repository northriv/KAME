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

#ifndef xpythonsupportH
#define xpythonsupportH

#ifdef USE_PYBIND11

#include "xscriptingthread.h"

class XMeasure;

//! Python scripting support, containing a thread running python monitor program.
//! The monitor program synchronize Ruby threads and XScriptingThread objects.
//! \sa XScriptingThread
class XPython : public XScriptingThreadList {
public:
    XPython(const char *name, bool runtime, const shared_ptr<XMeasure> &measure);
    virtual ~XPython();
protected:
    virtual void *execute(const atomic<bool> &) override;
    void my_defout(shared_ptr<XNode> node, const std::string &msg);
    std::string my_defin(shared_ptr<XNode> node);
private:
};

#endif //USE_PYBIND11
//---------------------------------------------------------------------------
#endif //
