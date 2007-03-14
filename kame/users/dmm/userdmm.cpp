/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#include "userdmm.h"
#include "charinterface.h"
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, KE2000, "Keithley 2000/2001 DMM");
REGISTER_TYPE(XDriverList, KE2182, "Keithley 2182 nanovolt meter");
REGISTER_TYPE(XDriverList, HP34420A, "Agilent 34420A nanovolt meter");

void
XDMMSCPI::changeFunction()
{
    std::string func = function()->to_str();
    if(!func.empty())
        interface()->sendf(":CONF:%s", func.c_str());
}
double
XDMMSCPI::fetch()
{
    interface()->query(":FETC?");
    return interface()->toDouble();
}
double
XDMMSCPI::oneShotRead()
{
    interface()->query(":READ?");
    return interface()->toDouble();
}
double
XDMMSCPI::measure(const std::string &func)
{
    interface()->queryf(":MEAS:%s?", func.c_str());
    return interface()->toDouble();
}
