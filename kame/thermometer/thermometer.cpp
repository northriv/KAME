/***************************************************************************
		Copyright (C) 2002-2025 Kentaro Kitagawa
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
#include <math.h>

#include "thermometer.h"
#include "rand.h"

//---------------------------------------------------------------------------
XCalibrationCurveList::XCalibrationCurveList(const char *name, bool runtime) :
    XCustomTypeListNode<XCalibrationCurve>(name, runtime) {
}
DECLARE_TYPE_HOLDER(XCalibrationCurveList)
REGISTER_TYPE(XCalibrationCurveList, LakeShore, "LakeShore")
;
REGISTER_TYPE(XCalibrationCurveList, ScientificInstruments, "Scientific Instruments")
;
REGISTER_TYPE(XCalibrationCurveList, ApproxThermometer, "Cubic-spline (Thermometer)")
;
REGISTER_TYPE(XCalibrationCurveList, ApproxCalibration, "Cubic-spline (Generic)")
;

XCalibrationCurve::XCalibrationCurve(const char *name, bool runtime) :
    XNode(name, runtime),
    m_outMin(create<XDoubleNode>("TMin", false)),
    m_outMax(create<XDoubleNode>("TMax", false)) {
    trans( *outMin()) = 1e-3;
    trans( *outMax()) = 1e3;
}

XLakeShore::XLakeShore(const char *name, bool runtime) :
    XResistanceThermometer(name, runtime),
    m_resMin(create<XDoubleNode>("RMin", false)),
    m_resMax(create<XDoubleNode>("RMax", false)),
    m_zu(create<XDoubleListNode>("Zu", false)),
    m_zl(create<XDoubleListNode>("Zl", false)),
    m_ai(create<XDouble2DNode>("Ai", false)) {
}

double XLakeShore::getRaw(double temp) const {
    Snapshot shot( *this);
    //using Newton's method
    double x, y, dypdx, val;
    if(temp < shot[ *outMin()])
        return shot[ *resMax()];
    if(temp > shot[ *outMax()])
        return shot[ *resMin()];
    val = shot[ *resMin()];
    for(double dy = 0.0001;; dy *= 2) {
        if(dy > 1.0)
            return shot[ *resMin()];
        double t = randMT19937();
        x = (log10(shot[ *resMax()]) * t + log10(shot[ *resMin()]) * (1 - t));
        for(int i = 0; i < 100; i++) {
            y = getOutput(pow(10, x)) - temp;
            if(fabs(y) < dy) {
                val = pow(10, x);
                return val;
            }
            dypdx = (y - (getOutput(pow(10, x - 0.00001)) - temp)) / 0.00001;
            if(dypdx != 0)
                x -= y / dypdx;
            if((x > log10(shot[ *resMax()])) || (x < log10(shot[ *resMin()]))
                || (dypdx == 0)) {
                double t = randMT19937();
                x = (log10(shot[ *resMax()]) * t + log10(shot[ *resMin()]) * (1 - t));
            }
        }
    }
    return val;
}

double XLakeShore::getOutput(double res) const {
    Snapshot shot( *this);
    double temp = 0, z, u = 0;
    if(res > shot[ *resMax()])
        return shot[ *outMin()];
    if(res < shot[ *resMin()])
        return shot[ *outMax()];
    z = log10(res);
    unsigned int n;
    if( !shot.size(zu()))
        return 0;
    const auto &zu_list( *shot.list(zu()));
    if( !shot.size(zl()))
        return 0;
    const auto &zl_list( *shot.list(zl()));
    for(n = 0; n < zu_list.size(); n++) {
        double zu = shot[ *static_pointer_cast<XDoubleNode>(zu_list.at(n))];
        double zl = shot[ *static_pointer_cast<XDoubleNode>(zl_list.at(n))];
        u = (z - zu + z - zl) / (zu - zl);
        if((u >= -1) && (u <= 1))
            break;
    }
    if(n >= zu_list.size())
        return 0;
    if( !shot.size(ai()))
        return 0;
    const auto &ai_list( *shot.list(ai()));
    if( !shot.size(ai_list[n]))
        return 0;
    const auto &ai_n_list( *shot.list(ai_list[n]));
    for(unsigned int i = 0; i < ai_n_list.size(); i++) {
        double ai_n_i = shot[ *static_pointer_cast<XDoubleNode>(ai_n_list.at(i))];
        temp += ai_n_i * cos(i * acos(u));
    }
    return temp;
}

XScientificInstruments::XScientificInstruments(const char *name, bool runtime) :
    XResistanceThermometer(name, runtime),
    m_resMin(create<XDoubleNode>("RMin", false)),
    m_resMax(create<XDoubleNode>("RMax", false)),
    m_abcde(create<XDoubleListNode>("ABCDE", false)),
    m_abc(create<XDoubleListNode>("ABC", false)),
    m_rCrossover(create<XDoubleNode>("RCrossover", false)) {
}

double XScientificInstruments::getRaw(double temp) const {
    Snapshot shot( *this);
    //using Newton's method
    double x, y, dypdx, val;
    if(temp < shot[ *outMin()])
        return shot[ *resMax()];
    if(temp > shot[ *outMax()])
        return shot[ *resMin()];
    val = shot[ *resMin()];
    for(double dy = 0.0001;; dy *= 2) {
        if(dy > 1.0)
            return shot[ *resMin()];
        double t = randMT19937();
        x = (log10(shot[ *resMax()]) * t + log10(shot[ *resMin()]) * (1 - t));
        for(int i = 0; i < 100; i++) {
            y = getOutput(pow(10, x)) - temp;
            if(fabs(y) < dy) {
                val = pow(10, x);
                return val;
            }
            dypdx = (y - (getOutput(pow(10, x - 0.00001)) - temp)) / 0.00001;
            if(dypdx != 0)
                x -= y / dypdx;
            if((x > log10(shot[ *resMax()])) || (x < log10(shot[ *resMin()]))
                || (dypdx == 0)) {
                double t = randMT19937();
                x = (log10(shot[ *resMax()]) * t + log10(shot[ *resMin()]) * (1 - t));
            }
        }
    }
    return val;
}

double XScientificInstruments::getOutput(double res) const {
    Snapshot shot( *this);
    if(res > shot[ *resMax()])
        return shot[ *outMin()];
    if(res < shot[ *resMin()])
        return shot[ *outMax()];
    double y = 0.0;
    double lx = log(res);
    if(res > shot[ *rCrossover()]) {
        if( !shot.size(abcde())) return 0;
        const auto &abcde_list( *shot.list(abcde()));
        if(abcde_list.size() >= 5) {
            double a = shot[ *static_pointer_cast<XDoubleNode>(abcde_list.at(0))];
            double b = shot[ *static_pointer_cast<XDoubleNode>(abcde_list.at(1))];
            double c = shot[ *static_pointer_cast<XDoubleNode>(abcde_list.at(2))];
            double d = shot[ *static_pointer_cast<XDoubleNode>(abcde_list.at(3))];
            double e = shot[ *static_pointer_cast<XDoubleNode>(abcde_list.at(4))];
            y = (a + c * lx + e * lx * lx) / (1.0 + b * lx + d * lx * lx);
        }
        return y;
    } else {
        if( !shot.size(abc())) return 0;
        const auto &abc_list( *shot.list(abc()));
        if(abc_list.size() >= 3) {
            double a = shot[ *static_pointer_cast<XDoubleNode>(abc_list.at(0))];
            double b = shot[ *static_pointer_cast<XDoubleNode>(abc_list.at(1))];
            double c = shot[ *static_pointer_cast<XDoubleNode>(abc_list.at(2))];
            y = 1.0 / (a + b * res * lx + c * res * res);
        }
        return y;
    }
}
