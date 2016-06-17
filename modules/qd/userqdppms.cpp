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
#include "userqdppms.h"
#include "charinterface.h"


REGISTER_TYPE(XDriverList, QDPPMS6000, "Quantum Design PPMS low-level interface");

XQDPPMS6000::XQDPPMS6000(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XQDPPMS>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("");
    interface()->setSerialEOS("\r\n");
}

double
XQDPPMS6000::getField(){
    double magnet_field;
    interface()->query("GetDat? 4");
    if( interface()->scanf("4,%*f,%lf", &magnet_field) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    return magnet_field;
}

double
XQDPPMS6000::getPosition(){
    double sample_position;
    interface()->query("GetDat? 8");
    if( interface()->scanf("8,%*f,%lf", &sample_position) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    return sample_position;
}

double
XQDPPMS6000::getTemp(){
    double sample_temp;
    interface()->query("GetDat? 2");
    if( interface()->scanf("2,%*f,%lf", &sample_temp) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    return sample_temp;
}

double
XQDPPMS6000::getTempRotator(){
    double sample_temp_rotator;
    int is_user_temp;
    interface()->query("USERTEMP?");
    if( interface()->scanf("%d,%*f,%*f,%*f,%*f",&is_user_temp) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(is_user_temp){
        int user_temp_channel = lrint(pow(2,is_user_temp));
        interface()->queryf("GetDat? %d", user_temp_channel);
        if( interface()->scanf("%*d,%*f,%lf", &sample_temp_rotator) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
    }
    else{
        sample_temp_rotator = 0.0;
    }
    return sample_temp_rotator;
}

double
XQDPPMS6000::getHeliumLevel(){
    double helium_level;
    interface()->query("LEVEL?");
    if( interface()->scanf("%lf,%*f", &helium_level) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    return helium_level;
}
