/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef AUTOLCTUNER_H_
#define AUTOLCTUNER_H_
//---------------------------------------------------------------------------
#include "secondarydriver.h"
#include "motor.h"
#include "networkanalyzer.h"
//---------------------------------------------------------------------------
class Ui_FrmAutoLCTuner;
typedef QForm<QMainWindow, Ui_FrmAutoLCTuner> FrmAutoLCTuner;

class LCRFit;
class XLCRPlot;
/*
* Tunes the reflection at the target frequency to zero.
*/
class XAutoLCTuner : public XSecondaryDriver {
public:
	XAutoLCTuner(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XAutoLCTuner();

	//! Shows all forms belonging to driver
	virtual void showForms();
protected:

	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter,
		const Snapshot &shot_others,
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
	const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &stm1() const {return m_stm1;}
	const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &stm2() const {return m_stm2;}
	const shared_ptr<XItemNode<XDriverList, XNetworkAnalyzer> > &netana() const {return m_netana;}
    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > &relayDriver() const {return m_relayDriver;}

	/// Target frequency [MHz]
	const shared_ptr<XBoolNode> &tuning() const {return m_tuning;}
	const shared_ptr<XBoolNode> &succeeded() const {return m_succeeded;}
	const shared_ptr<XDoubleNode> &target() const {return m_target;}
	const shared_ptr<XDoubleNode> &reflectionTargeted() const {return m_reflectionTargeted;}
	const shared_ptr<XDoubleNode> &reflectionRequired() const {return m_reflectionRequired;}
	const shared_ptr<XBoolNode> &useSTM1() const {return m_useSTM1;}
	const shared_ptr<XBoolNode> &useSTM2() const {return m_useSTM2;}
	const shared_ptr<XTouchableNode> &abortTuning() const {return m_abortTuning;}
    const shared_ptr<XDoubleNode> &backlushMinusTh() const {return m_backlushMinusTh;}
    const shared_ptr<XDoubleNode> &backlushPlusTh() const {return m_backlushPlusTh;}
    const shared_ptr<XIntNode> &timeMax() const {return m_timeMax;}
    const shared_ptr<XIntNode> &origBackMax() const {return m_origBackMax;}
    const shared_ptr<XComboNode> &fitFunc() const {return m_fitFunc;}
    const shared_ptr<XDoubleNode> &backlashRecoveryFactor() const {return m_backlashRecoveryFactor;}



	class Payload : public XSecondaryDriver::Payload {
	public:
        void resetToFirstStage() {
            fitOrig.reset();
            fitRotated.reset();
            clearSTMDelta();
            deltaC1perDeltaSTM.fill(0.0);
            deltaC2perDeltaSTM.fill(0.0);
        }
        void clearSTMDelta() {
            for(int i: {0,1})
                stmDelta[i] = lastDirection(i) * 1e-10;
        }
        shared_ptr<LCRFit> fitOrig, fitRotated;
        std::array<double, 2> stmBacklash; //[deg]
        std::array<double, 2> stmTrustArea; //[deg]
        XTime timeSTMChanged; //STM positions will move to \a targetSTMValues.
        std::array<double, 2> targetSTMValues; //[deg]
        double smallestRLAtF0; //0 < RL < 1
        std::array<double, 2> bestSTMValues; //[deg]
        static constexpr double TestDeltaFirst = 10; //[deg]
        static constexpr double TestDeltaMax = 720; //[deg]
        static constexpr double DeltaMax = 6 * 360; //[deg]
        std::array<double, 2> stmDelta; //[deg], +:CW, -:CCW.
        int lastDirection(size_t i) const {return (stmDelta[i] > 0) ? 1 : -1;} //+:CW, -:CCW.
        std::array<double, 2> deltaC1perDeltaSTM; //[F/deg.]
        std::array<double, 2> deltaC2perDeltaSTM; //[F/deg.]

		XTime started;
        int iterationCount;
        bool isTargetAbondoned;
        int taintedCount;
        int residue_offset;
        double sor;
	};
private:
	const shared_ptr<XItemNode<XDriverList, XMotorDriver> > m_stm1, m_stm2;
	const shared_ptr<XItemNode<XDriverList, XNetworkAnalyzer> > m_netana;
    const shared_ptr<XItemNode<XDriverList, XMotorDriver> > m_relayDriver;

	const shared_ptr<XBoolNode> m_tuning;
	const shared_ptr<XBoolNode> m_succeeded;
	const shared_ptr<XDoubleNode> m_target;
	const shared_ptr<XDoubleNode> m_reflectionTargeted;
	const shared_ptr<XDoubleNode> m_reflectionRequired;
	const shared_ptr<XBoolNode> m_useSTM1, m_useSTM2;
	const shared_ptr<XTouchableNode> m_abortTuning;
    const shared_ptr<XStringNode> m_status;
    const shared_ptr<XDoubleNode> m_backlushMinusTh, m_backlushPlusTh;
    const shared_ptr<XIntNode> m_timeMax, m_origBackMax;
    const shared_ptr<XComboNode> m_fitFunc;
    const shared_ptr<XDoubleNode> m_backlashRecoveryFactor;
    const shared_ptr<XStringNode> m_l1, m_r1, m_r2, m_c1, m_c2;

    std::deque<xqcon_ptr> m_conUIs;

    shared_ptr<Listener> m_lsnOnTargetChanged, m_lsnOnAbortTouched, m_lsnOnStatusOut;
    shared_ptr<XLCRPlot> m_lcrPlot;

	const qshared_ptr<FrmAutoLCTuner> m_form;

	void onTargetChanged(const Snapshot &shot, XValueNodeBase *);
	void onAbortTuningTouched(const Snapshot &shot, XTouchableNode *);
    void onStatusChanged(const Snapshot &shot, XValueNodeBase *);

	void determineNextC(double &deltaC1, double &deltaC2,
		double x, double x_err,
		double y, double y_err,
		double dxdC1, double dxdC2,
		double dydC1, double dydC2);
    [[noreturn]] void abortTuningFromAnalyze(Transaction &tr, double rl_at_f0, XString &&message);
    [[noreturn]] void rollBack(Transaction &tr, XString &&message);
    void clearUIAndPlot(Transaction &tr);
};



#endif /* AUTOLCTUNER_H_ */
