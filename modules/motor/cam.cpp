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
#include <complex>
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
          create<XScalarEntry>("CurrZ", false, dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4f"),
          create<XScalarEntry>("CurrX", false, dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4f"),
          create<XScalarEntry>("CurrA", false, dynamic_pointer_cast<XDriver>(shared_from_this()), "%.2f"),
          },
      m_targetValues{
          create<XDoubleNode>("TargetValueZ", true),
          create<XDoubleNode>("TargetValueX", true),
          create<XDoubleNode>("TargetValueA", true),
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
      m_setMaxSpeeds{
          create<XTouchableNode>("SetMaxSpeedZ", true),
          create<XTouchableNode>("SetMaxSpeedX", true),
          create<XTouchableNode>("SetMaxSpeedA", true),
          },
      m_speedReturnPath(create<XDoubleNode>("SpeedReturnPath", false)),
      m_endmillRadius(create<XDoubleNode>("EndmillRadius", false)),
      m_offsetX(create<XDoubleNode>("OffsetX", false)),
      m_feedXY(create<XDoubleNode>("FeedXY", false)),
      m_feedZ(create<XDoubleNode>("FeedZ", false)),
      m_cutDepthXY(create<XDoubleNode>("CutDepthXY", false)),
      m_cutDepthZ(create<XDoubleNode>("CutDepthZ", false)),
      m_escapeToHome(create<XTouchableNode>("EscapeToHome", true)),
      m_setZeroPositions(create<XTouchableNode>("SetZeroPositions", true)),
      m_freeAllAxes(create<XTouchableNode>("FreeAllAxes", true)),

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
      m_abortAfter(create<XDoubleNode>("AbortAfter", false)),
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
        xqcon_create<XQLineEditConnector>(targetValue(Axis::Z), m_form->m_edTargetZ),
        xqcon_create<XQLineEditConnector>(targetValue(Axis::X), m_form->m_edTargetX),
        xqcon_create<XQLineEditConnector>(targetValue(Axis::A), m_form->m_edTargetA),
        xqcon_create<XQLineEditConnector>(maxSpeed(Axis::Z), m_form->m_edMaxSpeedZ),
        xqcon_create<XQLineEditConnector>(maxSpeed(Axis::X), m_form->m_edMaxSpeedX),
        xqcon_create<XQLineEditConnector>(maxSpeed(Axis::A), m_form->m_edMaxSpeedA),
        xqcon_create<XQLineEditConnector>(gearRatio(Axis::Z), m_form->m_edRatioZ),
        xqcon_create<XQLineEditConnector>(gearRatio(Axis::X), m_form->m_edRatioX),
        xqcon_create<XQLineEditConnector>(gearRatio(Axis::A), m_form->m_edRatioA),
        xqcon_create<XQLineEditConnector>(speedReturnPath(), m_form->m_edSpeedReturn, false),
//        xqcon_create<XQLineEditConnector>(endmillRadius(), m_form->m_edEndmillRadius, false),
        xqcon_create<XQLineEditConnector>(offsetX(), m_form->m_edOffsetX, false),
        xqcon_create<XQLineEditConnector>(feedZ(), m_form->m_edFeedZ, false),
        xqcon_create<XQLineEditConnector>(feedXY(), m_form->m_edFeedXY, false),
        xqcon_create<XQLineEditConnector>(cutDepthZ(), m_form->m_edCutDepthZ, false),
        xqcon_create<XQLineEditConnector>(cutDepthXY(), m_form->m_edCutDepthXY, false),
        xqcon_create<XQLineEditConnector>(pos1(Axis::Z), m_form->m_edZ1, false),
        xqcon_create<XQLineEditConnector>(pos1(Axis::X), m_form->m_edX1, false),
        xqcon_create<XQLineEditConnector>(pos1(Axis::A), m_form->m_edA1, false),
        xqcon_create<XQLineEditConnector>(pos2(Axis::Z), m_form->m_edZ2, false),
        xqcon_create<XQLineEditConnector>(pos2(Axis::X), m_form->m_edX2, false),
        xqcon_create<XQLineEditConnector>(pos2(Axis::A), m_form->m_edA2, false),
        xqcon_create<XQLineEditConnector>(startZ(), m_form->m_edZ0, false),
        xqcon_create<XQLineEditConnector>(roughness(), m_form->m_edRoughness, false),
        xqcon_create<XQLineEditConnector>(abortAfter(), m_form->m_edAbortAfter),
        xqcon_create<XQLedConnector>(m_slipping, m_form->m_ledSlipped),
        xqcon_create<XQLedConnector>(m_running, m_form->m_ledRunning),
        xqcon_create<XQButtonConnector>(m_setZeroPositions, m_form->m_btnSetZeroPos),
        xqcon_create<XQButtonConnector>(m_escapeToHome, m_form->m_btnEscape),
        xqcon_create<XQButtonConnector>(m_cutNow, m_form->m_btnCutNow),
        xqcon_create<XQButtonConnector>(m_appendToList, m_form->m_btnAppend),
        xqcon_create<XQButtonConnector>(m_execute, m_form->m_btnExec),
        xqcon_create<XQButtonConnector>(m_freeAllAxes, m_form->m_btnFreeAll),
        xqcon_create<XQLabelConnector>(runningStatus(), m_form->m_lblStatus),
        xqcon_create<XQButtonConnector>(setMaxSpeed(Axis::Z), m_form->m_btnSetMaxSpeedZ),
        xqcon_create<XQButtonConnector>(setMaxSpeed(Axis::X), m_form->m_btnSetMaxSpeedX),
        xqcon_create<XQButtonConnector>(setMaxSpeed(Axis::A), m_form->m_btnSetMaxSpeedA),
    };

    for(auto &c: m_stms)
        connect(c);

    iterate_commit([=](Transaction &tr){
        tr[ *m_slipping] = false;
        tr[ *m_running] = false;
        tr[ *gearRatio(Axis::Z)] = 360.0 / 1.5; // Linear Actuator LX20, thread pitch
        tr[ *gearRatio(Axis::X)] = 54.0/16.0 * 360.0 / 0.5; //reduction ratio of timing belt * micrometer
        tr[ *gearRatio(Axis::A)] = 72.0/18.0 * 360.0 / 10.0; //reduction ratio of timing belt * rotary table
        tr[ *maxSpeed(Axis::Z)] = 500*360 / 60.0 / tr[ *gearRatio(Axis::Z)]; //max of pushing mode
        tr[ *maxSpeed(Axis::X)] = 0.1;
        tr[ *maxSpeed(Axis::A)] = 30.0;
        tr[ *speedReturnPath()] = 3.0;
        tr[ *cutDepthZ()] = 0.2;
        tr[ *cutDepthXY()] = 0.2;
        tr[ *feedZ()] = 0.1;
        tr[ *feedXY()] = 0.1;
        tr[ *pos2(Axis::A)] = 360.0;
        tr[ *roughness()] = 10;
        tr[ *abortAfter()] = 5.0;
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
        m_lsnOnFreeAllTouched = tr[ *freeAllAxes()].onTouch().connectWeakly(
            shared_from_this(), &XMicroCAM::onFreeAllTouched);
        m_lsnOnTargetChanged = tr[ *targetValue(Axis::Z)].onValueChanged().connectWeakly(
            shared_from_this(), &XMicroCAM::onTargetChanged);
        tr[ *targetValue(Axis::X)].onValueChanged().connect(m_lsnOnTargetChanged);
        tr[ *targetValue(Axis::A)].onValueChanged().connect(m_lsnOnTargetChanged);
        m_lsnOnSetMaxSpeedTouched = tr[ *setMaxSpeed(Axis::Z)].onTouch().connectWeakly(
            shared_from_this(), &XMicroCAM::onSetMaxSpeedTouched);
        tr[ *setMaxSpeed(Axis::X)].onTouch().connect(m_lsnOnSetMaxSpeedTouched);
        tr[ *setMaxSpeed(Axis::A)].onTouch().connect(m_lsnOnSetMaxSpeedTouched);
    });
    pos1(Axis::A)->disable();
    pos2(Axis::A)->disable();
    endmillRadius()->disable();
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
    double deg_per_sec = (posToSTM(shot, axis, feed) - posToSTM(shot, axis, 0.0)) / 60.0;
    return deg_per_sec / 360.0 * Snapshot( *stm)[ *stm->stepMotor()];
}

void
XMicroCAM::setSpeed(const Snapshot &shot, Axis axis, double rate) {
    const shared_ptr<XMotorDriver> stm__ = shot[ *stm(axis)];
    if(!stm__)
        return;
    trans( *stm__->speed()) = getSpeed(shot, axis, rate);
}
double
XMicroCAM::getSpeed(const Snapshot &shot, Axis axis, double rate) {
    if(rate < 0)
        rate = shot[ *maxSpeed(axis)] * 60.0; //[mm/min]
    const shared_ptr<XMotorDriver> stm__ = shot[ *stm(axis)];
    if(!stm__)
        return -1;
    return feedToSTMHz(shot, axis, rate, stm__);
}

void
XMicroCAM::setTarget(const Snapshot &shot, Axis axis, double target) {
    const shared_ptr<XMotorDriver> stm__ = shot[ *stm(axis)];
    if(!stm__)
        return;
    trans( *stm__->target()) = posToSTM(shot, axis, target);

    iterate_commit([=](Transaction &tr){
        tr[ *targetValue(axis)] = target;
        tr.unmark(m_lsnOnTargetChanged);
    });
}
void XMicroCAM::onSetMaxSpeedTouched(const Snapshot &shot, XTouchableNode *node) {
    Snapshot shot_this( *this);
    try {
        for(auto axis: {Axis::Z, Axis::X, Axis::A}) {
            if(node == setMaxSpeed(axis).get())
                setSpeed(shot_this, axis);
        }
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while starting, "));
        return;
    }
}
std::deque<double>
XMicroCAM::divideFeed(const Snapshot &shot, const std::deque<Axis> &axes, const std::deque<double> &lengths, double feed) {
    std::deque<double> vav;
    double lsq = 0.0;
    for(double length: lengths) {
        lsq += length * length;
    }
    for(unsigned int i = 0; i < axes.size(); ++i) {
        const shared_ptr<XMotorDriver> stm__ = shot[ *stm(axes[i])];
        assert(stm__);
        Snapshot shot_stm( *stm__);
//        double acc = shot_stm[ *stm__->timeAcc()];
//        double dec = shot_stm[ *stm__->timeDec()];
        vav.push_back(feed * lengths[i] / sqrt(lsq));
    }
    return vav;
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
    if( !shot_this[ *this].isSlipping)
        tr[ *this].slipMark = XTime::now();
}

void XMicroCAM::visualize(const Snapshot &shot) {
    std::string line_to_do;
    iterate_commit([=, &line_to_do](Transaction &tr){
        tr[ *running()] = shot[ *this].isRunning;
        tr[ *slipping()] = shot[ *this].isSlipping;
        if(shot[ *this].isRunning) {
            if(shot[ *this].isAllReady &&
                (XTime::now() - shot[ *this].lineStartedTime > 1.0) && shot[ *this].codeLines) {
                //runs every 1sec.
                tr[ *this].lineStartedTime = XTime::now();
                std::stringstream ss;
                ss << *shot[ *this].codeLines;
                int lineno = 0;
                while(std::getline(ss, line_to_do)) {
                    if(line_to_do.empty()) break;
                    lineno++;
                    if(lineno == tr[ *this].codeLinePos + 1) {
                        tr[ *this].codeLinePos++;
                        break;
                    }
                    line_to_do.clear();
                }
                if(line_to_do.empty()) {
                    tr[ *runningStatus()] = "Code underflows. Confusing...";
                    tr[ *this].isRunning = false;
                }
                else {
                    tr[ *this].lastLine = std::make_shared<XString>(line_to_do);
                }
            }
            double rest = tr[ *this].estFinishTime - XTime::now();
            if(shot[ *this].lastLine)
                tr[ *runningStatus()] = formatString("%d min Left @Line %d: ", (int)lrint(rest/60.0), shot[ *this].codeLinePos) + *shot[ *this].lastLine;
        }
        else {
            if(XTime::now() - shot[ *this].labelMark > 1.0) {
                int min = (int)lrint(estimateTime(XString(m_form->m_txtCode->toPlainText())) / 60.0);
                if(min)
                    tr[ *runningStatus()] = formatString("Idle, estimating %d min:", min);
                else
                    tr[ *runningStatus()] = "Idle";
                tr[ *this].labelMark = XTime::now();
            }
        }
    });

    if( !shot[ *this].isRunning || line_to_do.empty())
        return;
    if(XTime::now() - shot[ *this].slipMark > shot[ *abortAfter()] + 0.5) {
        onEscapeTouched(shot, nullptr);
        return;
    }

    //execute one line.
    std::cerr << line_to_do << std::endl;

    parseCode(m_context, line_to_do);
    CodeBlock &blk(m_context);

    if(blk.scode > 0) {
        double f = blk.scode * 360.0; //deg/min
        setSpeed(shot, Axis::A, f);
    }
    switch(blk.mcode) {
    case -1:
        break;
    case 2:
    case 30:
        trans( *this).isRunning = false;
        break;
    case 3:
        {
           const shared_ptr<XMotorDriver> stm__ = shot[ *stm(Axis::A)];
           if(stm__)
               trans( *stm__->forwardMotor()).touch();
        }
        break;
    case 5:
        {
            const shared_ptr<XMotorDriver> stm__ = shot[ *stm(Axis::A)];
            if(!stm__)
                return;
            trans( *stm__->stopMotor()).touch();
        }
        break;
    default:
        throw XInterface::XInterfaceError(getLabel() +
            i18n(": Unsupported letter in ") + (XString)line_to_do, __FILE__, __LINE__);
    }
    switch(blk.gcode) {
    case -1:
        break;
    case 0:
    case 1:
        if(blk.axescount == 1) {
            setSpeed(shot, blk.axes[0], (blk.gcode == 0) ? -1.0 : blk.feed);
            setTarget(shot, blk.axes[0], blk.target[0]);
        }
        else if(blk.axescount == 2) {
            double dv1 = fabs(blk.target[0] - shot[ *this].values[static_cast<int>(blk.axes[0])]);
            double dv2 = fabs(blk.target[1] - shot[ *this].values[static_cast<int>(blk.axes[1])]);
            auto feeds = divideFeed(shot, {blk.axes[0], blk.axes[1]}, {dv1, dv2}, blk.feed);
//            setSpeed(shot, blk.axes[0], (blk.gcode == 0) ? -1.0 : feeds[0]);
//            setSpeed(shot, blk.axes[1], (blk.gcode == 0) ? -1.0 : feeds[1]);
//            setTarget(shot, blk.axes[0], blk.target[0]);
//            setTarget(shot, blk.axes[1], blk.target[1]);
            double t0 = posToSTM(shot, blk.axes[0], blk.target[0]);
            double t1 = posToSTM(shot, blk.axes[1], blk.target[1]);
            double hz0 = getSpeed(shot, blk.axes[0], (blk.gcode == 0) ? -1.0 : feeds[0]);
            double hz1 = getSpeed(shot, blk.axes[1], (blk.gcode == 0) ? -1.0 : feeds[1]);
            const shared_ptr<XMotorDriver> stm1 = shot[ *stm(blk.axes[0])];
            const shared_ptr<XMotorDriver> stm2 = shot[ *stm(blk.axes[1])];
            if(!stm1 || !stm2)
                return;
            stm1->runSequentially({{t0}, {t1}}, {{hz0}, {hz1}}, {stm2});
            iterate_commit([=](Transaction &tr){
                tr[ *targetValue(blk.axes[0])] = blk.target[0];
                tr[ *targetValue(blk.axes[1])] = blk.target[1];
                tr.unmark(m_lsnOnTargetChanged);
            });
        }
        break;
    case 2: //CW
    case 3: //CCW
        if(blk.axescount) {
            int divisions = 4; //max. seq. move for AR
            std::vector<std::vector<double>> arcpts(2);
            for(auto &x: arcpts)
                x.resize(divisions);
            fixCurveAngle(blk, shot[ *this].values, divisions, &arcpts[0]);
            std::vector<std::vector<double>> arcspeeds(2);
            for(auto &x: arcspeeds)
                x.resize(divisions);
            for(int i = divisions - 1; i >= 0 ; --i) {
                double dv1, dv2;
                if(i == 0) {
                    dv1 = fabs(arcpts[0][0] - shot[ *this].values[static_cast<int>(blk.axes[0])]);
                    dv2 = fabs(arcpts[1][0] - shot[ *this].values[static_cast<int>(blk.axes[1])]);
                }
                else {
                    dv1 = fabs(arcpts[0][i] - arcpts[0][i - 1]);
                    dv2 = fabs(arcpts[1][i] - arcpts[1][i - 1]);
                }
                auto feeds = divideFeed(shot, {blk.axes[0], blk.axes[1]}, {dv1, dv2}, blk.feed);
                arcspeeds[0][i] = getSpeed(shot, blk.axes[0], feeds[0]);
                arcspeeds[1][i] = getSpeed(shot, blk.axes[1], feeds[1]);
                arcpts[0][i] = posToSTM(shot, blk.axes[0], arcpts[0][i]);
                arcpts[1][i] = posToSTM(shot, blk.axes[1], arcpts[1][i]);
            }
            const shared_ptr<XMotorDriver> stm1 = shot[ *stm(blk.axes[0])];
            const shared_ptr<XMotorDriver> stm2 = shot[ *stm(blk.axes[1])];
            if(!stm1 || !stm2)
                return;
            stm1->runSequentially(arcpts, arcspeeds, {stm2});
            iterate_commit([=](Transaction &tr){
                tr[ *targetValue(blk.axes[0])] = blk.target[0];
                tr[ *targetValue(blk.axes[1])] = blk.target[1];
                tr.unmark(m_lsnOnTargetChanged);
            });
         }
        break;
    case 18: //G02/03 in XZ
        break;
    default:
        throw XInterface::XInterfaceError(getLabel() +
            i18n(": Unsupported letter in ") + (XString)line_to_do, __FILE__, __LINE__);
    }
}
void
XMicroCAM::parseCode(CodeBlock &context, std::string &line_to_do) {
    double v[9];
    char c[9];
    int conv = sscanf(line_to_do.c_str(), "%c%lf%c%lf%c%lf%c%lf%c%lf%c%lf%c%lf%c%lf%c%lf",
        &c[0], &v[0], &c[1], &v[1], &c[2], &v[2], &c[3], &v[3], &c[4], &v[4], &c[5], &v[5], &c[6], &v[6], &c[7], &v[7], &c[8], &v[8]);
    int pos = 0;

    CodeBlock blk;
    blk.gcode = context.gcode; //supported code all mordal.
    blk.feed = context.feed;

    for(; pos < conv / 2; ++pos) {
        switch(c[pos]) {
        case 'G':
            blk.gcode = lrint(v[pos]);
            break;
        case 'S':
            if(blk.scode >= 0)
                throw XInterface::XInterfaceError(getLabel() +
                    i18n(": Duplicated letter in ") + (XString)line_to_do, __FILE__, __LINE__);
            blk.scode = lrint(v[pos]);
            break;
        case 'M':
            if(blk.mcode >= 0)
                throw XInterface::XInterfaceError(getLabel() +
                    i18n(": Duplicated letter in ") + (XString)line_to_do, __FILE__, __LINE__);
            blk.mcode = lrint(v[pos]);
            break;
        case 'X':
//        case 'Y':
        case 'Z':
        case 'A':
            if(blk.axescount >= 2)
                throw XInterface::XInterfaceError(getLabel() +
                    i18n(": Too many axes in ") + (XString)line_to_do, __FILE__, __LINE__);
            blk.target[blk.axescount] = v[pos];
            blk.axes[blk.axescount] = letterToAxis(c[pos]);
            blk.axescount++;
            break;
        case 'F':
            blk.feed = v[pos];
            break;
        case 'I':
            blk.cx = v[pos];
            break;
        case 'J':
            blk.cy = v[pos];
            break;
        case 'K':
            blk.cz = v[pos];
            break;
        case 'R':
            blk.r = v[pos];
            break;
        case 'O':
        case 'N':
        case '%': //start
        case '/': //skip
        case '(': //comment
            if(pos == 0)
                break;
        default:
            throw XInterface::XInterfaceError(getLabel() +
                i18n(": Unknown letter in ") + (XString)line_to_do, __FILE__, __LINE__);
        }
    }
    context = blk;
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

double
XMicroCAM::fixCurveAngle(CodeBlock &blk, const double currPos[NUM_AXES],
    int division, std::vector<double> *pts) {
    auto z1 = std::complex<double>(currPos[static_cast<int>(Axis::Z)], currPos[static_cast<int>(Axis::X)]);
    auto fn_idx = [this,&blk](Axis axis){
      for(int i = 0; i < blk.axescount; ++i) {
          if(blk.axes[i] == axis)
              return i;
      }
      throw XInterface::XInterfaceError(getLabel() +
          i18n(" axis is not provided."), __FILE__, __LINE__);
    };
    auto z2 = std::complex<double>(blk.target[fn_idx(Axis::Z)], blk.target[fn_idx(Axis::X)]);
    if(blk.r != 0.0) {
    //determines cx,cz
        auto z0 = (z1 + z2) / 2.0;
        double r12 = std::abs(z1 - z0);
        double sgn = (blk.gcode == 2) ? 1 : -1; //CW or CCW
        sgn *= (blk.r > 0) ? 1 : -1;
        z0 -= sgn * std::complex<double>(0.0, 1.0) * (z2 - z1)/r12 * sqrt(blk.r * blk.r - r12*r12/4);
        blk.cz = z0.real();
        blk.cx = z0.imag();
    }
    auto z0 = std::complex<double>(blk.cz, blk.cx);
    double arg = std::arg((z2 - z0) / (z1 - z0));
    if(blk.gcode == 2)
        arg *= -1; //CW
    if(arg < 0)
        arg += 2 * M_PI;
    if(division) {
    //calculate points in the curve.
        double darg = arg / division;
        darg *= (blk.gcode == 2) ? -1 : 1;
        for(int i = 0; i < division; ++i) {
            auto z = z0 + (z1 - z0) * std::polar<double>(1.0, darg * (i + 1));
            pts[fn_idx(Axis::Z)][i] = z.real();
            pts[fn_idx(Axis::X)][i] = z.imag();
            if(i > 0){
                if(((pts[0][i] - pts[0][i - 1]) * (pts[0][0] - currPos[static_cast<int>(blk.axes[0])]) < 0) ||
                    ((pts[1][i] - pts[1][i - 1]) * (pts[1][0] - currPos[static_cast<int>(blk.axes[1])]) < 0))
                    throw XInterface::XInterfaceError(getLabel() +
                        i18n(" bidirectional move during G02/03 is not currently supported."), __FILE__, __LINE__);
            }
        }
    }
    return arg * std::abs(z1 - z0); //arc length
}
double XMicroCAM::estimateTime(const std::string &lines){
    double rest = 0.0;
    std::stringstream ss;
    ss << lines;
    std::string line;
    //Estimating total time.
    double currPos[NUM_AXES] = {};
    currPos[static_cast<int>(Axis::Z)] = HOME_Z;

    CodeBlock blk;
    while(std::getline(ss, line)) {
        if(line.empty()) break;
        double dist = -1;

        parseCode(blk, line);

        switch(blk.gcode) {
        case -1:
            break;
        case 0:
        case 1:
            if(blk.axescount == 1) {
                auto &p1 = currPos[static_cast<int>(blk.axes[0])];
                if(blk.gcode == 1)
                    dist = fabs(blk.target[0] - p1);
                p1 = blk.target[0];
            }
            else if(blk.axescount == 2) {
                auto &p1 = currPos[static_cast<int>(blk.axes[0])];
                auto &p2 = currPos[static_cast<int>(blk.axes[1])];
                if(blk.gcode == 1)
                    dist = sqrt(pow(blk.target[0] - p1, 2) + pow(blk.target[1] - p2, 2));
                p1 = blk.target[0]; p2 = blk.target[1];
            }
            break;
        case 2:
        case 3:
            if(blk.axescount) {
                dist = fixCurveAngle(blk, currPos);
                auto &p1 = currPos[static_cast<int>(blk.axes[0])];
                auto &p2 = currPos[static_cast<int>(blk.axes[1])];
                p1 = blk.target[0]; p2 = blk.target[1];
            }
            break;
        default:
            throw XInterface::XInterfaceError(getLabel() +
                i18n(": Unsupported letter in ") + (XString)line, __FILE__, __LINE__);
        }

        if(dist > 0)
            rest += dist / blk.feed * 60.0;
    }
    return rest;
}

void XMicroCAM::execCut() {
    iterate_commit([=](Transaction &tr){
        Snapshot &shot(tr);
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
        tr[ *this].slipMark = XTime::now();
        m_context = CodeBlock{}; //initialize context

        tr[ *this].estFinishTime += estimateTime( *shot[ *this].codeLines);
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
void XMicroCAM::onFreeAllTouched(const Snapshot &shot, XTouchableNode *) {
    Snapshot shot_this( *this);
    try {
        trans( *this).isRunning = false;
        bool activate = (m_form->m_btnFreeAll->text() == i18n("Activate"));
        if( !activate)
            trans( *this).isRunning = false;
        //Stops all STMs
        for(auto &stm: m_stms) {
            const shared_ptr<XMotorDriver> stm__ = shot_this[ *stm];
            if(!stm__)
                continue;
            trans( *stm__->active()) = activate;
        }
        m_form->m_btnFreeAll->setText(activate ? i18n("Free All") : i18n("Activate"));
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
void XMicroCAM::onTargetChanged(const Snapshot &shot, XValueNodeBase *node) {
    Snapshot shot_this( *this);
    try {
        for(int i = 0; i < NUM_AXES; ++i) {
            Axis axis = static_cast<Axis>(i);
            if(targetValue(axis).get() == node)
                setTarget(shot_this, axis, shot_this[ *dynamic_cast<XDoubleNode*>(node)]);
        }
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while starting, "));
        return;
    }
}
XString XMicroCAM::genCode(const Snapshot &shot_this) {
    if(shot_this[ *pos1(Axis::Z)] <= shot_this[ *startZ()])
        throw XInterface::XInterfaceError(getLabel() +
            i18n("Z0 must be < Z1."), __FILE__, __LINE__);
    if(shot_this[ *pos2(Axis::Z)] <= shot_this[ *startZ()])
        throw XInterface::XInterfaceError(getLabel() +
            i18n("Z0 must be < Z2."), __FILE__, __LINE__);

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
            fn_code_ln(currZ, x1, feed_xy * shot_this[ *speedReturnPath()]); //return path, already cut
            //z1' += z_cut, z2' += dz_cut, if z1' = z1 and z2' = z2, (r feed speed) = roughness
            currZ1 += cut_z;
            if(currZ1 > z1)
                currZ1 = z1;
            currZ2 += cut_z;
            if(currZ2 > z2)
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
    fn_code_ln(z0, currX, feed_z * shot_this[ *speedReturnPath()]); //return path, already cut
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

        //Go to Z = Z_HOME
        if(shot_this[ *currValue(Axis::Z)->value()] < HOME_Z)
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
