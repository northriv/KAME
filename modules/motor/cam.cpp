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

#include "cam.h"
#include "ui_camform.h"
#include "interface.h"
#include "analyzer.h"
#include <iostream>
#include <sstream>
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, MicroCAM, "Micro CAM z,x,phi");

XMicroCAM::XMicroCAM(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XSecondaryDriver(name, runtime, ref(tr_meas), meas),
      m_stms{
        create<XItemNode < XDriverList, XMotorDriver> >(
            "STMZ", false, ref(tr_meas), meas->drivers(), true),
        create<XItemNode < XDriverList, XMotorDriver> >(
            "STMX", false, ref(tr_meas), meas->drivers(), true),
        create<XItemNode < XDriverList, XMotorDriver> >(
            "STMA", false, ref(tr_meas), meas->drivers(), true),
          },
      m_currValues{
          create<XScalarEntry>("CurrZ", false, dynamic_pointer_cast<XDriver>(shared_from_this())),
          create<XScalarEntry>("CurrX", false, dynamic_pointer_cast<XDriver>(shared_from_this())),
          create<XScalarEntry>("CurrA", false, dynamic_pointer_cast<XDriver>(shared_from_this())),
          },
      m_gearRatios{
          create<XDoubleNode>("GearRatioZ", false),
          create<XDoubleNode>("GearRatioX", false),
          create<XDoubleNode>("GearRatioA", false),
          },
      m_maxSpeeds{
          create<XDoubleNode>("MaxSpeedZ", false),
          create<XDoubleNode>("MaxSpeedX", false),
          create<XDoubleNode>("MaxSpeedA", false),
          },
      m_endmillRadius(create<XDoubleNode>("EndmillRadius", false)),
      m_offsetX(create<XDoubleNode>("OffsetX", false)),
      m_feedXY(create<XDoubleNode>("FeedXY", false)),
      m_feedZ(create<XDoubleNode>("FeedZ", false)),
      m_cutDepthXY(create<XDoubleNode>("CutDepthXY", false)),
      m_cutDepthZ(create<XDoubleNode>("CutDepthZ", false)),
      m_escapeToHome(create<XTouchableNode>("EscapeToHome", true)),
      m_setZeroPositions(create<XTouchableNode>("SetZeroPositions", true)),

      m_pos1{
          create<XDoubleNode>("Position1Z", false),
          create<XDoubleNode>("Position1X", false),
          create<XDoubleNode>("Position1A", false),
          },
      m_pos2{
          create<XDoubleNode>("Position2Z", false),
          create<XDoubleNode>("Position2X", false),
          create<XDoubleNode>("Position2A", false),
          },
      m_startZ(create<XDoubleNode>("StartPositionZ", false)),
      m_roughness(create<XDoubleNode>("Roughness", false)),
      m_execute(create<XTouchableNode>("Execute", true)),
      m_cutNow(create<XTouchableNode>("CutNow", true)),
      m_appendToList(create<XTouchableNode>("AppendToList", true)),
      m_slipping(create<XBoolNode>("Slipping", true)),
      m_running(create<XBoolNode>("Running", true)),
      m_runningStatus(create<XStringNode>("RunningStatus", true)),
      m_form(new FrmCAM) {

    m_form->setWindowTitle(i18n("Controller of Micro CAM") + getLabel() );
    for(auto &e: m_currValues)
        meas->scalarEntries()->insert(tr_meas, e);

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(stm(Axis::X), m_form->m_cmbSTMX, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(stm(Axis::Z), m_form->m_cmbSTMZ, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(stm(Axis::A), m_form->m_cmbSTMA, ref(tr_meas)),
        xqcon_create<XQLCDNumberConnector>(currValue(Axis::Z)->value(), m_form->m_lcdZ),
        xqcon_create<XQLCDNumberConnector>(currValue(Axis::X)->value(), m_form->m_lcdX),
        xqcon_create<XQLCDNumberConnector>(currValue(Axis::A)->value(), m_form->m_lcdA),
        xqcon_create<XQLineEditConnector>(maxSpeed(Axis::Z), m_form->m_edMaxSpeedZ),
        xqcon_create<XQLineEditConnector>(maxSpeed(Axis::X), m_form->m_edMaxSpeedX),
        xqcon_create<XQLineEditConnector>(maxSpeed(Axis::A), m_form->m_edMaxSpeedA),
        xqcon_create<XQLineEditConnector>(gearRatio(Axis::Z), m_form->m_edRatioZ),
        xqcon_create<XQLineEditConnector>(gearRatio(Axis::X), m_form->m_edRatioX),
        xqcon_create<XQLineEditConnector>(gearRatio(Axis::A), m_form->m_edRatioA),
        xqcon_create<XQLineEditConnector>(endmillRadius(), m_form->m_edEndmillRadius),
        xqcon_create<XQLineEditConnector>(offsetX(), m_form->m_edOffsetX),
        xqcon_create<XQLineEditConnector>(feedZ(), m_form->m_edFeedZ),
        xqcon_create<XQLineEditConnector>(feedXY(), m_form->m_edFeedXY),
        xqcon_create<XQLineEditConnector>(cutDepthZ(), m_form->m_edCutDepthZ),
        xqcon_create<XQLineEditConnector>(cutDepthXY(), m_form->m_edCutDepthXY),
        xqcon_create<XQLineEditConnector>(pos1(Axis::Z), m_form->m_edZ1),
        xqcon_create<XQLineEditConnector>(pos1(Axis::X), m_form->m_edX1),
        xqcon_create<XQLineEditConnector>(pos1(Axis::A), m_form->m_edA1),
        xqcon_create<XQLineEditConnector>(pos2(Axis::Z), m_form->m_edZ2),
        xqcon_create<XQLineEditConnector>(pos2(Axis::X), m_form->m_edX2),
        xqcon_create<XQLineEditConnector>(pos2(Axis::A), m_form->m_edA2),
        xqcon_create<XQLineEditConnector>(startZ(), m_form->m_edZ0),
        xqcon_create<XQLineEditConnector>(roughness(), m_form->m_edRoughness),
        xqcon_create<XQLedConnector>(m_slipping, m_form->m_ledSlipped),
        xqcon_create<XQLedConnector>(m_running, m_form->m_ledRunning),
        xqcon_create<XQButtonConnector>(m_setZeroPositions, m_form->m_btnSetZeroPos),
        xqcon_create<XQButtonConnector>(m_escapeToHome, m_form->m_btnEscape),
        xqcon_create<XQButtonConnector>(m_cutNow, m_form->m_btnCutNow),
        xqcon_create<XQButtonConnector>(m_appendToList, m_form->m_btnAppend),
        xqcon_create<XQButtonConnector>(m_execute, m_form->m_btnExec),
        xqcon_create<XQLabelConnector>(runningStatus(), m_form->m_lblStatus),
    };

    for(auto &c: m_stms)
        connect(c);

    iterate_commit([=](Transaction &tr){
        tr[ *m_slipping] = false;
        tr[ *m_running] = false;
        tr[ *gearRatio(Axis::Z)] = 360.0 / 1.0; // Linear Actuator, thread pitch
        tr[ *gearRatio(Axis::X)] = 48.0/16.0 * 360.0 / 0.1; //reduction ratio of timing belt * micrometer
        tr[ *gearRatio(Axis::A)] = 72.0/18.0 * 360.0 / 10.0; //reduction ratio of timing belt * rotary table
        tr[ *maxSpeed(Axis::Z)] = 10.0;
        tr[ *maxSpeed(Axis::X)] = 0.1;
        tr[ *maxSpeed(Axis::A)] = 10.0;
        tr[ *cutDepthZ()] = 0.2;
        tr[ *cutDepthXY()] = 0.2;
        tr[ *feedZ()] = 0.1;
        tr[ *feedXY()] = 0.1;
        tr[ *roughness()] = 10;
        m_lsnOnEscapeTouched = tr[ *escapeToHome()].onTouch().connectWeakly(
            shared_from_this(), &XMicroCAM::onEscapeTouched);
        m_lsnOnSetZeroTouched = tr[ *setZeroPositions()].onTouch().connectWeakly(
            shared_from_this(), &XMicroCAM::onSetZeroTouched);
        m_lsnOnCutNowTouched = tr[ *cutNow()].onTouch().connectWeakly(
            shared_from_this(), &XMicroCAM::onCutNowTouched);
        m_lsnOnAppendToListTouched = tr[ *appendToList()].onTouch().connectWeakly(
            shared_from_this(), &XMicroCAM::onAppendToListTouched);
        m_lsnOnExecuteTouched = tr[ *execute()].onTouch().connectWeakly(
            shared_from_this(), &XMicroCAM::onExecuteTouched);
    });
}

void XMicroCAM::showForms() {
    m_form->showNormal();
    m_form->raise();
}

double
XMicroCAM::posToSTM(const Snapshot &shot, Axis axis, double pos) {
    if(axis == Axis::X)
        pos -= shot[ *offsetX()];
    pos *= shot[ *gearRatio(axis)];
    return pos;
}
double
XMicroCAM::stmToPos(const Snapshot &shot, Axis axis, double stmpos) {
    stmpos /= shot[ *gearRatio(axis)];
    if(axis == Axis::X)
        stmpos += shot[ *offsetX()];
    return stmpos;
}

double
XMicroCAM::feedToSTMHz(const Snapshot &shot, Axis axis, double feed, const shared_ptr<XMotorDriver> &stm) {
    double deg_per_sec = (posToSTM(shot, axis, feed) - posToSTM(shot, axis, 0.0)) * 60.0;
    return deg_per_sec / 360.0 * Snapshot( *stm)[ *stm->stepMotor()];
}

void
XMicroCAM::setSpeed(const Snapshot &shot, Axis axis, double rate) {
    if(rate < 0)
        rate = shot[ *maxSpeed(axis)] / 60.0; //[mm/min]
    const shared_ptr<XMotorDriver> stm__ = shot[ *stm(axis)];
    if(!stm__)
        return;
    trans( *stm__->speed()) = feedToSTMHz(shot, axis, rate, stm__);
}

void
XMicroCAM::setTarget(const Snapshot &shot, Axis axis, double target) {
    const shared_ptr<XMotorDriver> stm__ = shot[ *stm(axis)];
    if(!stm__)
        return;
    trans( *stm__->target()) = posToSTM(shot, axis, target);
}

void XMicroCAM::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
                        XDriver *emitter) {
    Snapshot &shot_this(tr);
    std::deque<shared_ptr<XMotorDriver>> stms;
    for(auto &stm: m_stms) {
        const shared_ptr<XMotorDriver> stm__ = shot_this[ *stm];
        stms.push_back(stm__);
    }
    tr[ *this].isAllReady = true;
    tr[ *this].isSlipping = false;
    for(unsigned int i = 0; i < NUM_AXES; ++i) {
        auto axis = static_cast<Axis>(i);
        double var;
        Snapshot shot_stm = shot_others;
        if(stms[i].get() == emitter)
            shot_stm = shot_emitter;
        var = shot_stm[ *stms[i]->position()->value()];
        var = stmToPos(shot_this, axis, var);
        tr[ *this].values[i] = var;
        currValue(axis)->value(tr, var);
        if( !shot_stm[ *stms[i]->ready()])
            if(i != static_cast<unsigned int>(Axis::A))
                tr[ *this].isAllReady = false;
        if(shot_stm[ *stms[i]->slipping()])
            tr[ *this].isSlipping = true;
    }

}

void XMicroCAM::visualize(const Snapshot &shot) {
    std::string line_to_do;
    iterate_commit([=, &line_to_do](Transaction &tr){
        tr[ *running()] = shot[ *this].isRunning;
        tr[ *slipping()] = shot[ *this].isSlipping;

        if(shot[ *this].isRunning) {
            if(shot[ *this].isAllReady && (XTime::now() - shot[ *this].lineStartedTime > 1.0)) {
                //runs every 1sec.
                tr[ *this].lineStartedTime = XTime::now();
                std::stringstream ss;
                ss << shot[ *this].codeLines;
                int lineno = 0;
                while(std::getline(ss, line_to_do)) {
                    lineno++;
                    if(lineno == tr[ *this].codeLinePos + 1) {
                        tr[ *this].codeLinePos++;
                        break;
                    }
                }
                if(line_to_do.empty())
                    tr[ *runningStatus()] = "Code underflows. Confusing...";
            }
            double rest = tr[ *this].estFinishTime - XTime::now();
            tr[ *runningStatus()] = formatString("Line %d: Est. Time Remain %d min..", tr[ *this].codeLinePos, (int)lrint(rest/60.0));
        }
        else {
            tr[ *runningStatus()] = "Idle";
        }

    });

    if( !shot[ *this].isRunning || line_to_do.empty())
        return;

    //execute one line.
    std::cerr << line_to_do << std::endl;

    return;

    double x, z, f;
    int word;
    if(sscanf(line_to_do.c_str(), "G01X%lfF%lf", &x, &f) == 2) {
        setSpeed(shot, Axis::X, f);
        setTarget(shot, Axis::X, x);
    }
    else if(sscanf(line_to_do.c_str(), "G01Z%lfF%lf", &z, &f) == 2) {
        setSpeed(shot, Axis::Z, f);
        setTarget(shot, Axis::Z, z);
    }
    else if(sscanf(line_to_do.c_str(), "G01X%lfZ%lfF%lf", &x, &z, &f) == 3) {
        double dx = fabs(x - shot[ *this].values[static_cast<int>(Axis::X)]);
        double dz = fabs(z - shot[ *this].values[static_cast<int>(Axis::Z)]);
        double fx = f * dx / sqrt(dx*dx + dz*dz);
        double fz = f * dz / sqrt(dx*dx + dz*dz);
        setSpeed(shot, Axis::X, fx);
        setSpeed(shot, Axis::Z, fz);
        setTarget(shot, Axis::X, x);
        setTarget(shot, Axis::Z, z);
    }
    else if(sscanf(line_to_do.c_str(), "G00X%lf", &x) == 1) {
        setMaxSpeed(shot, Axis::X);
        setTarget(shot, Axis::X, x);
    }
    else if(sscanf(line_to_do.c_str(), "G01Z%lf", &z) == 1) {
        setMaxSpeed(shot, Axis::Z);
        setTarget(shot, Axis::Z, z);
    }
    else if(sscanf(line_to_do.c_str(), "G01X%lfZ%lf", &x, &z) == 2) {
        setMaxSpeed(shot, Axis::X);
        setMaxSpeed(shot, Axis::Z);
        setTarget(shot, Axis::X, x);
        setTarget(shot, Axis::Z, z);
    }
    else if(sscanf(line_to_do.c_str(), "S%lfM03", &f) == 1) {
        f *= 360.0; //deg/min
        const shared_ptr<XMotorDriver> stm__ = shot[ *stm(Axis::A)];
        if(!stm__)
            return;
        trans( *stm__->forwardMotor()).touch();
    }
    else if(sscanf(line_to_do.c_str(), "M%2d", &word) == 1) {
        switch(word) {
        case 2:
            trans( *this).isRunning = false;
        case 5:
            {
                const shared_ptr<XMotorDriver> stm__ = shot[ *stm(Axis::A)];
                if(!stm__)
                    return;
                trans( *stm__->stopMotor()).touch();
            }
        default:
            break;
        }
    }

}
bool XMicroCAM::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
                                XDriver *emitter) const {
    for(auto &stm: m_stms) {
        const shared_ptr<XMotorDriver> stm__ = shot_this[ *stm];
        if(!stm__)
            return false;
        for(auto &stm2: m_stms) {
            if(stm == stm2) continue;
            const shared_ptr<XMotorDriver> stm2__ = shot_this[ *stm2];
            if(stm2__ == stm__)
                return false;
        }
    }
    const shared_ptr<XMotorDriver> stm__ = shot_this[ *stm(Axis::Z)];
    if(emitter != stm__.get())
        return false;
    return true;
}

void XMicroCAM::execCut() {
    iterate_commit([=](Transaction &tr){
        Snapshot shot(tr);
        if( !shot[ *this].isAllReady)
            throw XInterface::XInterfaceError(getLabel() +
                i18n("Motor is not ready to go."), __FILE__, __LINE__);
        if(shot[ *this].isRunning)
            throw XInterface::XInterfaceError(getLabel() +
                i18n("Already running."), __FILE__, __LINE__);
        tr[ *this].isRunning = true;
        tr[ *this].codeLines = std::make_shared<XString>(m_form->m_txtCode->toPlainText());
        tr[ *this].codeLinePos = 0;
        tr[ *this].lineStartedTime = XTime::now();
        tr[ *this].estFinishTime = XTime::now();
        std::stringstream ss;
        ss << shot[ *this].codeLines;
        std::string line;
        double currX = 0.0, currZ = HOME_Z;
        while(std::getline(ss, line)) {
            double x, z, f;
            double dist = -1;
            if(sscanf(line.c_str(), "G01X%lfF%lf", &x, &f) == 2) {
                dist = fabs(x - currX);
                currX = x;
            }
            else if(sscanf(line.c_str(), "G01Z%lfF%lf", &z, &f) == 2) {
                dist = fabs(z - currZ);
                currZ = z;
            }
            else if(sscanf(line.c_str(), "G01X%lfZ%lfF%lf", &x, &z, &f) == 3) {
                dist = sqrt(pow(x - currX, 2) + pow(z - currZ, 2));
                currX = x; currZ = z;
            }
            if(dist > 0)
                tr[ *this].estFinishTime += dist / f * 60.0;
        }
    });

}
void XMicroCAM::onExecuteTouched(const Snapshot &shot, XTouchableNode *) {
    Snapshot shot_this( *this);
    try {
        execCut();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while starting, "));
        return;
    }
}

void XMicroCAM::onCutNowTouched(const Snapshot &shot, XTouchableNode *) {
    Snapshot shot_this( *this);
    try {
        m_form->m_txtCode->setText(genCode(shot_this));
        execCut();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while starting, "));
        return;
    }
}
XString XMicroCAM::genCode(const Snapshot &shot_this) {
    if(shot_this[ *pos1(Axis::Z)] >= shot_this[ *startZ()])
        throw XInterface::XInterfaceError(getLabel() +
            i18n("Z0 must be > Z1."), __FILE__, __LINE__);
    if(shot_this[ *pos2(Axis::Z)] >= shot_this[ *startZ()])
        throw XInterface::XInterfaceError(getLabel() +
            i18n("Z0 must be > Z2."), __FILE__, __LINE__);

    double z0 = shot_this[ *startZ()];
    double z1 = shot_this[ *pos1(Axis::Z)];
    double z2 = shot_this[ *pos2(Axis::Z)];
    double x1 = shot_this[ *pos1(Axis::X)];
    double x2 = shot_this[ *pos2(Axis::X)];
    double cut_z =  shot_this[ *cutDepthZ()];
    double cut_xy =  shot_this[ *cutDepthXY()];

    XString newcodes;
    newcodes += formatString("(Cyl.Cut %.4f, %.4f -> %.4f, %.4f from %.4f)\n", z1, x1, z2, x2, z0);
    double currZ = HOME_Z, currX = 1e10;
    auto fn_code_ln = [&](double z, double x, double rate = -1.0){
        if((currX == x) && (currZ == z)) {
            return; //zero cut, ignoring.
        }
        newcodes += (rate >= 0.0) ? "G01" : "G00";
        if(currX != x) {
            newcodes += formatString("X%.4f", x);
            currX = x;
        }
        if(currZ != z) {
            newcodes += formatString("Z%.4f", z);
            currZ = z;
        }
        if(rate > 0) {
            newcodes += formatString("F%.4f", rate);
        }
        newcodes += "\n";
    };
    //phi -> inf. rot. w/ max speed
    double rpm_a = shot_this[ *maxSpeed(Axis::A)] * 60.0 / 360.0;
    newcodes += formatString("S%.4fM03\n", rpm_a);
    //x -> x1 w/ max speed
    fn_code_ln(currZ, x1);
    //z -> z0 w/ max speed
    fn_code_ln(z0, currX);
    double feed_z = shot_this[ *feedZ()];
    double feed_xy = cut_xy * fabs(rpm_a); //[mm/min.]
    feed_xy = std::min((double)shot_this[ *feedXY()], feed_xy);
    if(x1 == x2) {
        if(z1 != z2)
            throw XInterface::XInterfaceError(getLabel() +
                i18n("Z1 must be = Z2."), __FILE__, __LINE__);
        fn_code_ln(z2, currX, feed_z);
    }
    else {
        //z1' = 0, z2' = 0
        double currZ1 = z0, currZ2 = z0;
        //loop until z1' = z1 and z2' = z2
        for(bool finish_cut = false; !finish_cut;) {
            //x -> x1
            fn_code_ln(currZ, x1, feed_xy);
            //z1' -= z_cut, z2' -= dz_cut, if z1' = z1 and z2' = z2, (r feed speed) = roughness
            currZ1 -= cut_z;
            if(currZ1 < z1)
                currZ1 = z1;
            currZ2 -= cut_z;
            if(currZ2 < z2)
                currZ2 = z2;
            if((currZ1 == z1) && (currZ2 == z2)) {
                finish_cut = true;
                cut_xy = shot_this[ *roughness()] * 1e3; //[mm]
            }
            //z -> z1' w/ given z feed speed
            fn_code_ln(currZ1, currX, feed_z);
            //x -> x2, z -> z2', |2pi / dA/dt| dXdt = (X cut).
            feed_xy = std::min(feed_xy, cut_xy * fabs(rpm_a));
            if(z1 != z2)
                feed_xy = std::min(feed_xy, feed_z);
            fn_code_ln(currZ2, x2, feed_xy);
            if(newcodes.length() > 2000) {
                throw XInterface::XInterfaceError(getLabel() +
                    i18n("Too small cut depth or incorrect direction."), __FILE__, __LINE__);
            }
        }
    }
    //z -> z0
    fn_code_ln(z0, currX, feed_z);
//    //escape
//    fn_code_ln(HOME_Z, currX);
    newcodes += "M05\n";
    newcodes += "M02\n";

    return newcodes;
}
void XMicroCAM::onAppendToListTouched(const Snapshot &shot, XTouchableNode *) {
    Snapshot shot_this( *this);
    try {
        m_form->m_txtCode->append(genCode(shot_this));
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while preparing list, "));
        return;
    }
}
void XMicroCAM::onEscapeTouched(const Snapshot &shot, XTouchableNode *) {
    Snapshot shot_this( *this);
    try {
        trans( *this).isRunning = false;
        //Stops all STMs
        for(auto &stm: m_stms) {
            const shared_ptr<XMotorDriver> stm__ = shot_this[ *stm];
            if(!stm__)
                continue;
            trans( *stm__->stopMotor()).touch();
        }
        //Awaiting for all STMs ready
        XTime time0 = XTime::now();
        for(;;) {
            if(XTime::now() - time0 > 1.0)
                throw XInterface::XInterfaceError(getLabel() +
                    i18n("Timeout while stopping motors."), __FILE__, __LINE__);
            bool not_ready = false;
            for(auto &stm: m_stms) {
                const shared_ptr<XMotorDriver> stm__ = shot_this[ *stm];
                if(!stm__)
                    continue;
                if( !Snapshot( *stm__)[ *stm__->ready()])
                    not_ready = true;
            }
            if( !not_ready)
                break;
        }
        //Go to Z = Z_HOME
        if(shot_this[ *currValue(Axis::Z)->value()] > HOME_Z)
            throw XInterface::XInterfaceError(getLabel() +
                i18n("Huh? z > home position."), __FILE__, __LINE__);
        setMaxSpeed(shot_this, Axis::Z);
        setTarget(shot_this, Axis::Z, HOME_Z);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while escaping, "));
        return;
    }
}

void XMicroCAM::onSetZeroTouched(const Snapshot &shot, XTouchableNode *) {
    Snapshot shot_this( *this);
    try {
        if(shot_this[ *running()])
            throw XInterface::XInterfaceError(getLabel() +
                i18n("Huh? still running."), __FILE__, __LINE__);
        for(auto &stm: m_stms) {
            const shared_ptr<XMotorDriver> stm__ = shot_this[ *stm];
            if(!stm__)
                continue;
            trans( *stm__->clear()).touch();
        }
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while setting zero, "));
        return;
    }
}
