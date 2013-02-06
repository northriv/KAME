/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef nmrrelaxH
#define nmrrelaxH

#include "secondarydriver.h"
#include "xnodeconnector.h"
//#include "pulserdriver.h"
//#include "nmrpulse.h"
//#include "nmrrelaxfit.h"
#include <complex>

#include "nmrspectrumsolver.h"

class XNMRPulseAnalyzer;
class XPulser;
class XRelaxFunc;
class XRelaxFuncList;
class XRelaxFuncPlot;
class XScalarEntry;

#include "xwavengraph.h"

class Ui_FrmNMRT1;
typedef QForm<QMainWindow, Ui_FrmNMRT1> FrmNMRT1;

//! Measure Relaxation Curve
class XNMRT1 : public XSecondaryDriver {
public:
	XNMRT1(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	~XNMRT1 () {}
  
	//! Shows all forms belonging to driver
	virtual void showForms();
protected:
	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) throw (XRecordError&);
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
	struct Payload : public XSecondaryDriver::Payload {
	private:
	friend class XNMRT1;
	friend class XRelaxFunc;
	friend class XRelaxFuncPlot;
		//! For fitting and display
		struct Pt {
			double var; /// auto-phase- or absolute value
			std::complex<double> c;
			double p1;
			double isigma; /// weight
			std::deque<std::complex<double> > value_by_cond;
		};
		struct ConvolutionCache {
			std::vector<std::complex<double> > wave;
			double windowwidth;
			FFT::twindowfunc windowfunc;
			int origin;
			double cfreq;
			double power;
		};
		//! Raw measured points
		struct RawPt {
			std::deque<std::complex<double> > value_by_cond;
			double p1;
		};
		std::deque<shared_ptr<ConvolutionCache> > m_convolutionCache;
		//! Stores all measured points.
		std::deque<RawPt> m_pts;
		//! Stores reduced points to manage fitting and display.
		std::deque<Pt> m_sumpts;
		double m_params[3]; //!< fitting parameters; 1/T1, c, a; ex. f(t) = c*exp(-t/T1) + a
		double m_errors[3]; //!< std. deviations

		XTime m_timeClearRequested;
	};

	//! Holds 1/T1 or 1/T2 and its std. deviation
	const shared_ptr<XScalarEntry> &t1inv() const {return m_t1inv;}
	const shared_ptr<XScalarEntry> &t1invErr() const {return m_t1invErr;}

	const shared_ptr<XItemNode < XDriverList, XPulser > > &pulser() const {return m_pulser;}
	const shared_ptr<XItemNode < XDriverList, XNMRPulseAnalyzer > > &pulse1() const {return m_pulse1;}
	const shared_ptr<XItemNode < XDriverList, XNMRPulseAnalyzer > > &pulse2() const {return m_pulse2;}

	//! If active, a control to Pulser is allowed
	const shared_ptr<XBoolNode> &active() const {return m_active;}
	//! Deduce phase from data
	const shared_ptr<XBoolNode> &autoPhase() const {return m_autoPhase;}
	//! Fit 3 parameters.
	const shared_ptr<XBoolNode> &mInftyFit() const {return m_mInftyFit;}
	//! Use absolute value, ignoring phase
	const shared_ptr<XBoolNode> &absFit() const {return m_absFit;}
	//! Region of P1 or 2tau for fitting, display, control of pulser [ms]
	const shared_ptr<XDoubleNode> &p1Min() const {return m_p1Min;}
	const shared_ptr<XDoubleNode> &p1Max() const {return m_p1Max;}
	//! (Deduced) phase of echoes [deg.]
	const shared_ptr<XDoubleNode> &phase() const {return m_phase;}
	//! Center freq of echoes [kHz].
	const shared_ptr<XDoubleNode> &freq() const {return m_freq;}
	/// FFT Window Function
	const shared_ptr<XComboNode> &windowFunc() const {return m_windowFunc;}
	/// FFT Window Length
	const shared_ptr<XComboNode> &windowWidth() const {return m_windowWidth;}
	//! Auto-select window.
	const shared_ptr<XBoolNode> &autoWindow() const {return m_autoWindow;}
	enum MEASMODE {MEAS_T1 = 0, MEAS_T2 = 1, MEAS_ST_E = 2};
	//! T1/T2/StE measurement
	const shared_ptr<XComboNode> &mode() const {return m_mode;}
	//! # of Samples for fitting and display
	const shared_ptr<XUIntNode> &smoothSamples() const {return m_smoothSamples;}
	//! Distribution of P1 or 2tau
	const shared_ptr<XComboNode> &p1Dist() const {return m_p1Dist;}
	//! Relaxation Function
	const shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > >  &relaxFunc() const {return m_relaxFunc;}

private:
	//! List of relaxation functions
	shared_ptr<XRelaxFuncList> m_relaxFuncs;
  
	friend class XRelaxFunc;
	friend class XRelaxFuncPlot;
 
	//! Holds 1/T1 or 1/T2 and its std. deviation
	const shared_ptr<XScalarEntry> m_t1inv;
	const shared_ptr<XScalarEntry> m_t1invErr;

	const shared_ptr<XItemNode < XDriverList, XPulser > > m_pulser;
	const shared_ptr<XItemNode < XDriverList, XNMRPulseAnalyzer > > m_pulse1;
	const shared_ptr<XItemNode < XDriverList, XNMRPulseAnalyzer > > m_pulse2;

	const shared_ptr<XBoolNode> m_active;
	const shared_ptr<XBoolNode> m_autoPhase;
	const shared_ptr<XBoolNode> m_mInftyFit;
	const shared_ptr<XBoolNode> m_absFit;
	const shared_ptr<XDoubleNode> m_p1Min;
	const shared_ptr<XDoubleNode> m_p1Max;
	const shared_ptr<XDoubleNode> m_phase;
	const shared_ptr<XDoubleNode> m_freq;
	const shared_ptr<XDoubleNode> m_bandWidth;
	const shared_ptr<XComboNode> m_windowFunc;
	const shared_ptr<XComboNode> m_windowWidth;
	const shared_ptr<XBoolNode> m_autoWindow;
	const shared_ptr<XComboNode> m_mode;
	const shared_ptr<XUIntNode> m_smoothSamples;
	const shared_ptr<XComboNode> m_p1Dist;
	shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > >  m_relaxFunc;
	const shared_ptr<XTouchableNode> m_resetFit, m_clearAll;
	const shared_ptr<XStringNode> m_fitStatus;

	//! for Non-Lenear-Least-Square fitting
	struct NLLS {
		std::deque<Payload::Pt> *pts; //pointer to data
		shared_ptr<XRelaxFunc> func; //pointer to the current relaxation function
		bool is_minftyfit; //3param fit or not.
		double fixed_minfty;
	};
 
	shared_ptr<XListener> m_lsnOnClearAll, m_lsnOnResetFit;
	shared_ptr<XListener> m_lsnOnActiveChanged;
	shared_ptr<XListener> m_lsnOnCondChanged;
	void onClearAll (const Snapshot &shot, XTouchableNode *);
	void onResetFit (const Snapshot &shot, XTouchableNode *);
	void onActiveChanged (const Snapshot &shot, XValueNodeBase *);
	void onCondChanged (const Snapshot &shot, XValueNodeBase *);
	xqcon_ptr m_conP1Min, m_conP1Max, m_conPhase, m_conFreq,
		m_conWindowFunc, m_conWindowWidth, m_conAutoWindow,
		m_conSmoothSamples, m_conASWClearance;
	xqcon_ptr m_conFitStatus;
	xqcon_ptr m_conP1Dist, m_conRelaxFunc;
	xqcon_ptr m_conClearAll, m_conResetFit;
	xqcon_ptr m_conActive, m_conAutoPhase, m_conMInftyFit, m_conAbsFit;
	xqcon_ptr m_conMode;
	xqcon_ptr m_conPulser, m_conPulse1, m_conPulse2;

	void analyzeSpectrum(Transaction &tr,
		const std::vector< std::complex<double> >&wave, int origin, double cf,
		std::deque<std::complex<double> > &value_by_cond);

	shared_ptr<SpectrumSolverWrapper> m_solver;

	const qshared_ptr<FrmNMRT1> m_form;
	const shared_ptr<XStatusPrinter> m_statusPrinter;
  
	//! Does fitting iterations \a itercnt times
	//! \param relax a pointer to a realaxation function
	//! \param itercnt counts 
	//! \param buf a message will be passed
	XString iterate(Transaction &tr, shared_ptr<XRelaxFunc> &relax, int itercnt);
 		      
	//! Store reduced points
	//! \sa m_pt, m_sumpts
	const shared_ptr<XWaveNGraph> m_wave;

	std::deque<double> m_windowWidthList;

	atomic<int> m_isPulserControlRequested;
};

//---------------------------------------------------------------------------
#endif
