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
#ifndef nmrrelaxH
#define nmrrelaxH

#include "secondarydriver.h"
#include "xnodeconnector.h"
//#include "pulserdriver.h"
//#include "nmrpulse.h"
//#include "nmrrelaxfit.h"
#include <complex>
#include "tikhonovreg.h"

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
			int isigma; /// weight
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
		std::vector<Pt> m_sumpts;
		double m_params[3]; //!< fitting parameters; 1/T1, c, a; ex. f(t) = c*exp(-t/T1) + a
		double m_errors[3]; //!< std. deviations        
        XTime m_timeClearRequested;

        //! Fields for Mapping via Tikhonov Regularization.
        double m_mapFreqRes; //!< [Hz]
        double m_mapBandWidth; //!< [Hz]
        long mapFreqCount() const {return lrint(m_mapBandWidth / m_mapFreqRes);}
        double mapStartFreq() const {return -(mapFreqCount() / 2) * m_mapFreqRes;} //!<[Hz]
        long m_mapTCount;
        struct Pulse {
            double p1;
            int avgCount = 0;
            Eigen::VectorXcd summedTrace; //!< time domain
            Eigen::VectorXcd ft;
            double ftOrigin = 0.0;
            double summedDarkPSDSq = 0.0; //! sum sq. of dark spectral power density.
        };
        std::vector<shared_ptr<Pulse>> m_allPulses;

        XTime m_timeMapFTCalcRequested; //!< updated if any condition for Pulse::ft has been changed.
        XTime m_timeMapClearRequested;
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
    //! Tracks peak freq. to accomodate a field decay.
    const shared_ptr<XBoolNode> &trackPeak() const {return m_trackPeak;}
    //! Region of P1 or 2tau for fitting, display, control of pulser [ms]
	const shared_ptr<XDoubleNode> &p1Min() const {return m_p1Min;}
	const shared_ptr<XDoubleNode> &p1Max() const {return m_p1Max;}
	//! Candidate for the next P1/2tau.
	const shared_ptr<XDoubleNode> &p1Next() const {return m_p1Next;}
	const shared_ptr<XDoubleNode> &p1AltNext() const {return m_p1AltNext;}
	//! (Deduced) phase of echoes [deg.]
	const shared_ptr<XDoubleNode> &phase() const {return m_phase;}
	//! Center freq of echoes [kHz].
	const shared_ptr<XDoubleNode> &freq() const {return m_freq;}
    //! FFT Window Function
	const shared_ptr<XComboNode> &windowFunc() const {return m_windowFunc;}
    //! FFT Window Length
	const shared_ptr<XComboNode> &windowWidth() const {return m_windowWidth;}
	//! Auto-select window.
	const shared_ptr<XBoolNode> &autoWindow() const {return m_autoWindow;}
    enum class MeasMode {T1 = 0, T2 = 1, ST_E = 2, T2_Multi = 3};
	//! T1/T2/StE measurement
	const shared_ptr<XComboNode> &mode() const {return m_mode;}
	//! # of Samples for fitting and display
	const shared_ptr<XUIntNode> &smoothSamples() const {return m_smoothSamples;}
	//! Strategy for distributing P1 or 2tau
	const shared_ptr<XComboNode> &p1Strategy() const {return m_p1Strategy;}
	//! Distribution of P1 or 2tau
	const shared_ptr<XComboNode> &p1Dist() const {return m_p1Dist;}
	//! Relaxation Function
	const shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > >  &relaxFunc() const {return m_relaxFunc;}

    //! Fields for Mapping via Tikhonov Regularization.
    enum class MapMode {Off = 0, NoiseAnalysis = 1, LCurve = 2, GCV = 3};
    const shared_ptr<XComboNode> &mapMode() const {return m_mapMode;}
    //! [kHz].
    const shared_ptr<XDoubleNode> &mapBandWidth() const {return m_mapBandWidth;}
    //! [kHz].
    const shared_ptr<XDoubleNode> &mapFreqRes() const {return m_mapFreqRes;}
    //! FFT Window Function for mapping
    const shared_ptr<XComboNode> &mapWindowFunc() const {return m_mapWindowFunc;}
    //! FFT Window Length for mapping [%]
    const shared_ptr<XDoubleNode> &mapWindowWidth() const {return m_mapWindowWidth;}

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
    const shared_ptr<XBoolNode> m_trackPeak;
    const shared_ptr<XDoubleNode> m_p1Min;
	const shared_ptr<XDoubleNode> m_p1Max;
	const shared_ptr<XDoubleNode> m_p1Next;
	const shared_ptr<XDoubleNode> m_p1AltNext;
	const shared_ptr<XDoubleNode> m_phase;
	const shared_ptr<XDoubleNode> m_freq;
	const shared_ptr<XDoubleNode> m_bandWidth;
	const shared_ptr<XComboNode> m_windowFunc;
	const shared_ptr<XComboNode> m_windowWidth;
	const shared_ptr<XBoolNode> m_autoWindow;
	const shared_ptr<XComboNode> m_mode;
	const shared_ptr<XUIntNode> m_smoothSamples;
	const shared_ptr<XComboNode> m_p1Strategy;
	const shared_ptr<XComboNode> m_p1Dist;
	shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > >  m_relaxFunc;
	const shared_ptr<XTouchableNode> m_resetFit, m_clearAll;
	const shared_ptr<XStringNode> m_fitStatus;

    //! Fields for Mapping via Tikhonov Regularization.
    const shared_ptr<XComboNode> m_mapMode;
    const shared_ptr<XDoubleNode> m_mapFreqRes;
    const shared_ptr<XDoubleNode> m_mapBandWidth;
    const shared_ptr<XComboNode> m_mapWindowFunc;
    const shared_ptr<XDoubleNode> m_mapWindowWidth;

	//! for Non-Lenear-Least-Square fitting
	struct NLLS {
		std::vector<Payload::Pt> *pts; //pointer to data
		shared_ptr<XRelaxFunc> func; //pointer to the current relaxation function
		bool is_minftyfit; //3param fit or not.
		double fixed_minfty;
	};
 
    shared_ptr<Listener> m_lsnOnClearAll, m_lsnOnResetFit;
    shared_ptr<Listener> m_lsnOnActiveChanged;
    shared_ptr<Listener> m_lsnOnCondChanged, m_lsnOnP1CondChanged, m_lsnOnMapCondChanged, m_lsnOnMapClearCondRequested;
    void onClearAll (const Snapshot &shot, XTouchableNode *);
	void onResetFit (const Snapshot &shot, XTouchableNode *);
    void onActiveChanged (const Snapshot &shot, XValueNodeBase *);
	void onCondChanged (const Snapshot &shot, XValueNodeBase *);
    void onMapCondChanged (const Snapshot &shot, XValueNodeBase *);
    void onMapClearCondRequested (const Snapshot &shot, XValueNodeBase *);
    void onP1CondChanged (const Snapshot &shot, XValueNodeBase *);
    std::deque<xqcon_ptr> m_conUIs;

	void analyzeSpectrum(Transaction &tr,
		const std::vector< std::complex<double> >&wave, int origin, double cf,
		std::deque<std::complex<double> > &value_by_cond);
    void storePulseForMapping(Transaction &tr, double p1_or_2tau,
        const std::vector< std::complex<double> >&wave, int origin,
        const std::vector<double>&darkpsd, double dfreq, double interval);
    void ZFFFT(Transaction &tr,
        std::vector< std::complex<double> >&bufin, std::vector< std::complex<double> >&bufout,
        shared_ptr<Payload::Pulse> p, double interval);

	shared_ptr<SpectrumSolverWrapper> m_solver;
    shared_ptr<SpectrumSolverWrapper> m_solverMapPulse;

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

	const static int CONVOLUTION_CACHE_SIZE = (3 * 10);

	const static char P1DIST_LINEAR[];
	const static char P1DIST_LOG[];
	const static char P1DIST_RECIPROCAL[];

	const static char P1STRATEGY_RANDOM[];
	const static char P1STRATEGY_FLATTEN[];

	double distributeP1(const Snapshot &shot, double uniform_x_0_to_1);
	void obtainNextP1(Transaction &tr);
    void setNextP1(const Snapshot &shot);

    const shared_ptr<XWaveNGraph> m_waveMap, m_waveAllRelaxCurves;
    unique_ptr<TikhonovRegular> m_regularization;
};

//---------------------------------------------------------------------------
#endif
