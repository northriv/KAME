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

#ifndef CAM_H_
#define CAM_H_

#include "motor.h"
#include "secondarydriver.h"
#include "xnodeconnector.h"
class XScalarEntry;
class Ui_FrmCAM;
typedef QForm<QMainWindow, Ui_FrmCAM> FrmCAM;

//Micro CAM with z, x, phi axes
class XMicroCAM : public XSecondaryDriver  {
public:
    XMicroCAM(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XMicroCAM() {}

    //! Shows all forms belonging to driver
    virtual void showForms();
protected:
    //! This function is called when a connected driver emit a signal
    virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter);
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot);
    //! Checks if the connected drivers have valid time stamps.
    //! \return true if dependency is resolved.
    //! This function must be reentrant unlike analyze().
    virtual bool checkDependency(const Snapshot &shot_this,
        const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const;

public:
    enum class Axis {Z = 0, X = 1, A = 2};
    constexpr static unsigned int NUM_AXES = 3;

    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &stm(Axis axis) const {return m_stms[static_cast<int>(axis)];}

    struct Payload : public XSecondaryDriver::Payload {
        double values[NUM_AXES];
        bool isSlipping;
        bool isRunning;
        bool isAllReady;
        int codeLinePos;
        XTime estFinishTime;
        XTime lineStartedTime;
        shared_ptr<XString> codeLines;
        shared_ptr<XString> lastLine;
        XTime slipMark;
        XTime labelMark = {};
    };

    const shared_ptr<XScalarEntry> &currValue(Axis axis) const {return m_currValues[static_cast<int>(axis)];} //!< [mm] or [deg]

    const shared_ptr<XDoubleNode> &targetValue(Axis axis) const {return m_targetValues[static_cast<int>(axis)];} //!< [mm] or [deg]
    const shared_ptr<XDoubleNode> &gearRatio(Axis axis) const {return m_gearRatios[static_cast<int>(axis)];} //!< [deg/mm] or Reduction ratio[deg/deg]
    const shared_ptr<XDoubleNode> &maxSpeed(Axis axis) const {return m_maxSpeeds[static_cast<int>(axis)];} //!< [mm/s] or [deg/s]

    const shared_ptr<XTouchableNode> &setMaxSpeed(Axis axis) const {return m_setMaxSpeeds[static_cast<int>(axis)];}

    //! cutting conditions:
    const shared_ptr<XDoubleNode> &endmillRadius() const {return m_endmillRadius;} //!< [mm]
    const shared_ptr<XDoubleNode> &offsetX() const {return m_offsetX;} //!< [mm] arbitrary offset to R reading
    const shared_ptr<XDoubleNode> &feedXY() const {return m_feedXY;}
    const shared_ptr<XDoubleNode> &feedZ() const {return m_feedZ;}
    const shared_ptr<XDoubleNode> &cutDepthXY() const {return m_cutDepthXY;}
    const shared_ptr<XDoubleNode> &cutDepthZ() const {return m_cutDepthZ;}
    const shared_ptr<XDoubleNode> &speedReturnPath() const {return m_speedReturnPath;} //!< [um]

    constexpr static double HOME_Z = -20; //!< [mm]
    const shared_ptr<XTouchableNode> &escapeToHome() const {return m_escapeToHome;}
    const shared_ptr<XTouchableNode> &setZeroPositions() const {return m_setZeroPositions;}
    const shared_ptr<XTouchableNode> &freeAllAxes() const {return m_freeAllAxes;}

    //! automatic cut
    const shared_ptr<XDoubleNode> &pos1(Axis axis) const {return m_pos1[static_cast<int>(axis)];} //!< [mm] or [deg]
    const shared_ptr<XDoubleNode> &pos2(Axis axis) const {return m_pos2[static_cast<int>(axis)];} //!< [mm] or [deg]
    const shared_ptr<XDoubleNode> &startZ() const {return m_startZ;}
    const shared_ptr<XDoubleNode> &roughness() const {return m_roughness;} //!< [um]
    const shared_ptr<XTouchableNode> &cutNow() const {return m_cutNow;}
    const shared_ptr<XTouchableNode> &appendToList() const {return m_appendToList;}

    const shared_ptr<XTouchableNode> &execute() const {return m_execute;}
    const shared_ptr<XBoolNode> &slipping() const {return  m_slipping;}
    const shared_ptr<XBoolNode> &running() const {return  m_running;}

    const shared_ptr<XDoubleNode> &abortAfter() const {return m_abortAfter;} //!< [s]

    const shared_ptr<XStringNode> &runningStatus() const {return  m_runningStatus;}

private:
    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > m_stms[NUM_AXES];
    const shared_ptr<XScalarEntry> m_currValues[NUM_AXES];

    const shared_ptr<XDoubleNode> m_targetValues[NUM_AXES];
    const shared_ptr<XDoubleNode> m_gearRatios[NUM_AXES];
    const shared_ptr<XDoubleNode> m_maxSpeeds[NUM_AXES];
    const shared_ptr<XTouchableNode> m_setMaxSpeeds[NUM_AXES];
    const shared_ptr<XDoubleNode> m_speedReturnPath;

    const shared_ptr<XDoubleNode> m_endmillRadius, m_offsetX, m_feedXY, m_feedZ, m_cutDepthXY, m_cutDepthZ;
    const shared_ptr<XTouchableNode> m_escapeToHome, m_setZeroPositions, m_freeAllAxes;

    const shared_ptr<XDoubleNode> m_pos1[NUM_AXES], m_pos2[NUM_AXES], m_startZ;
    const shared_ptr<XDoubleNode> m_roughness;

    const shared_ptr<XTouchableNode> m_execute, m_cutNow, m_appendToList;

    const shared_ptr<XBoolNode> m_slipping;
    const shared_ptr<XBoolNode> m_running;

    const shared_ptr<XDoubleNode> m_abortAfter;

    const shared_ptr<XStringNode> m_runningStatus;

    std::deque<xqcon_ptr> m_conUIs;

    shared_ptr<Listener> m_lsnOnCutNowTouched, m_lsnOnAppendToListTouched,
        m_lsnOnEscapeTouched, m_lsnOnFreeAllTouched, m_lsnOnTargetChanged,
        m_lsnOnSetZeroTouched, m_lsnOnExecuteTouched,
        m_lsnOnSetMaxSpeedTouched;

    const qshared_ptr<FrmCAM> m_form;

    void onCutNowTouched(const Snapshot &shot, XTouchableNode *);
    void onAppendToListTouched(const Snapshot &shot, XTouchableNode *);
    void onEscapeTouched(const Snapshot &shot, XTouchableNode *);
    void onFreeAllTouched(const Snapshot &shot, XTouchableNode *);
    void onSetZeroTouched(const Snapshot &shot, XTouchableNode *);
    void onExecuteTouched(const Snapshot &shot, XTouchableNode *);
    void onTargetChanged(const Snapshot &shot, XValueNodeBase *);
    void onSetMaxSpeedTouched(const Snapshot &shot, XTouchableNode *);

    XString genCode(const Snapshot &shot);
    void execCut();

    double posToSTM(const Snapshot &shot, Axis axis, double pos);
    double stmToPos(const Snapshot &shot, Axis axis, double stmpos);
    double feedToSTMHz(const Snapshot &shot, Axis axis, double feed, const shared_ptr<XMotorDriver> &stm);

    void setTarget(const Snapshot &shot, Axis axis, double target);
    double getSpeed(const Snapshot &shot, Axis axis, double rate = -1.0);
    void setSpeed(const Snapshot &shot, Axis axis, double rate = -1.0);
    void setMaxSpeed(const Snapshot &shot, Axis axis) {setSpeed(shot, axis);}

    constexpr Axis letterToAxis(char c) {
        switch(c) {case 'Z': return Axis::Z; case 'X': return Axis::X; case 'A': return Axis::A;
        default: throw XInterface::XInterfaceError(i18n("Unknown Axis Letter."), __FILE__, __LINE__);}}
    struct CodeBlock {
        int gcode = -1;
        double scode = -1;
        int mcode = -1;
        int axescount = 0;
        Axis axes[2] = {};
        double target[2] = {};
        double feed = -1;
        double cx = 0.0;
        double cy = 0.0;
        double cz = 0.0;
        double r = 0.0;
    };
    void parseCode(CodeBlock &context, std::string &line);
    CodeBlock m_context;

    double fixCurveAngle(CodeBlock &context, const double currPos[NUM_AXES],
        int division = 0, std::vector<double> *pts = nullptr);
    std::deque<double> divideFeed(const Snapshot &shot, const std::deque<Axis> &axes, const std::deque<double> &lengths, double feed);
    double estimateTime(const std::string &);
};


#endif /* CAM_H_ */
