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
#include "userqdppms.h"
#include "charinterface.h"


REGISTER_TYPE(XDriverList, QDPPMS6000, "Quantum Design PPMS low-level interface");

XQDPPMS6000::XQDPPMS6000(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XQDPPMS>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("");
    interface()->setSerialEOS("\r\n");
}

void
XQDPPMS6000::open() throw (XKameError &) {
    interface()->query("FIELD?");
    double field, field_rate;
    int field_approach_mode, magnet_mode;
    if(interface()->scanf("%lf,%lf,%d,%d;", &field, &field_rate, &field_approach_mode, &magnet_mode) != 4)
        throw XInterface::XConvError(__FILE__, __LINE__);
    double position;
    int mode;
    int slow_down_code;
    interface()->query("MOVE?");
    if(interface()->scanf("%lf,%d,%d;", &position, &mode, &slow_down_code) != 3)
        throw XInterface::XConvError(__FILE__, __LINE__);
    double temp;
    double temp_rate;
    int temp_approach_mode;
    interface()->query("TEMP?");
    if(interface()->scanf("%lf,%lf,%d;", &temp, &temp_rate, &temp_approach_mode) != 3)
        throw XInterface::XConvError(__FILE__, __LINE__);
    iterate_commit([=](Transaction &tr){
       tr[ *fieldSweepRate()] = field_rate * 1e-4;
       tr[ *targetField()] = field * 1e-4;
       tr[ *fieldApproachMode()] = field_approach_mode;
       tr[ *tempSweepRate()] = temp_rate;
       tr[ *targetTemp()] = temp;
       tr[ *tempApproachMode()] = temp_approach_mode;
       tr[ *targetPosition()] = position;
       tr[ *positionApproachMode()] = mode;
       tr[ *positionSlowDownCode()] = slow_down_code;
    });

    start();
}

void
XQDPPMS6000::setField(double field, double rate, int approach_mode, int magnet_mode){
    interface()->sendf("FIELD %f %f %d %d", field * 1e4, rate * 1e4, approach_mode, magnet_mode);
}

void
XQDPPMS6000::setPosition(double position, int mode, int slow_down_code){
    interface()->sendf("MOVE %f %d %d", position, mode, slow_down_code);
}

void
XQDPPMS6000::setTemp(double temp, double rate, int approach_mode){
    if(temp > 0.0){
        interface()->sendf("TEMP %f %f %d", temp, rate, approach_mode);
    }
    else if(1 > getField() && ((getStatus() >> 4) & 0xf) == 0x1){
        interface()->send("SHUTDOWN");
    }
    else{
        throw XInterface::XInterfaceError(i18n("Erase magnetic field before shutdown."), __FILE__, __LINE__);
    }
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
XQDPPMS6000::getUserTemp(){
    double sample_user_temp;
    int is_user_temp;
    interface()->query("USERTEMP?");
    if( interface()->scanf("%d,%*f,%*f,%*f,%*f",&is_user_temp) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(is_user_temp){
        int user_temp_channel = lrint(pow(2,is_user_temp));
        interface()->queryf("GetDat? %d", user_temp_channel);
        if( interface()->scanf("%*d,%*f,%lf", &sample_user_temp) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
    }
    else{
        sample_user_temp = 0.0;
    }
    return sample_user_temp;
}

double
XQDPPMS6000::getHeliumLevel(){
    double helium_level;
    interface()->query("LEVEL?");
    if( interface()->scanf("%lf,%*f", &helium_level) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    return helium_level;
}

int
XQDPPMS6000::getStatus(){
    int status;
    interface()->query("GetDat? 1");
    if( interface()->scanf("1,%*f,%d", &status) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    return status;
}
