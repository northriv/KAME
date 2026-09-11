/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef nmrspectrumbaseH
#define nmrspectrumbaseH
//---------------------------------------------------------------------------
#include <secondarydriver.h>
#include <xnodeconnector.h>
#include <complex>
#include <deque>
#include <vector>
#include "nmrspectrumsolver.h"

class XNMRPulseAnalyzer;
class XWaveNGraph;
class XXYPlot;

template <class FRM>
class XNMRSpectrumBase : public XSecondaryDriver {
public:
	XNMRSpectrumBase(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! ususally nothing to do
	virtual ~XNMRSpectrumBase();
  
	//! Shows all forms belonging to driver
protected:
	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
         XDriver *emitter) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
	//! Checks if the connected drivers have valid time stamps.
	//! \return true if dependency is resolved.
	//! This function must be reentrant unlike analyze().
	virtual bool checkDependency(const Snapshot &shot_this,
		const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const override;
public:
	//! driver specific part below 
	struct Payload : public XSecondaryDriver::Payload {
		const std::vector<std::complex<double> > &wave() const {return m_wave;}
		//! Averaged weights.
		const std::vector<double> &weights() const {return m_weights;}
		//! Power spectrum density of dark. [V].
		const std::vector<double> &darkPSD() const {return m_darkPSD;}
		//! Resolution [Hz].
		double res() const {return m_res;}
		//! Value of the first point [Hz].
		double min() const {return m_min;}

		//! One time bin of a relaxation map: the sweep-axis accumulators of the
		//! records that share a time slot, on the same grid (\a min(), \a res(),
		//! same length) as the spectrum itself.  \a times carries every abscissa
		//! summed here -- more than one when the driver groups records, e.g. m
		//! CPMG echoes at a time, whose bin then sits at 2 tau n/m.
		struct MapBin {
			std::vector<double> times; //!< 2 tau, P1, ... of the summed records
			int avgCount = 0; //!< # of records summed
			std::deque<std::complex<double> > accum;
			std::deque<double> accum_weights;
			std::deque<double> accum_dark;
		};
		//! Empty unless the driver overrides mapBinning(); one entry per time bin.
		//! Shared by pointer-to-const with live Snapshots, so a bin is replaced,
		//! never written through.
		const std::vector<shared_ptr<const MapBin> > &mapBins() const {return m_mapBins;}
		//! Turns \a MapBin::accum_dark into a variance [V^2], as analyzeIFT() does
		//! for darkPSD(): the reciprocal of the period of the wave. 0 until a
		//! record has been binned.
		double mapPSDCoeff() const {return m_mapPSDCoeff;}
	private:
		template <class>
		friend class XNMRSpectrumBase;

		double m_res, m_min;

		std::vector<double> m_weights;
		std::vector<double> m_darkPSD;
		std::vector<std::complex<double> > m_wave;

		enum {ACCUM_BANKS = 3};
		std::deque<std::complex<double> > m_accum[ACCUM_BANKS];
		std::deque<double> m_accum_weights[ACCUM_BANKS];
		std::deque<double> m_accum_dark[ACCUM_BANKS]; //[V^2/Hz].

		std::deque<std::pair<double, double> > m_peaks;

		std::vector<shared_ptr<const MapBin> > m_mapBins;
		double m_mapPSDCoeff = 0.0;

		shared_ptr<const FFT> m_ift, m_preFFT;

		XTime m_timeClearRequested;
	};

	const shared_ptr<XItemNode<XDriverList, XNMRPulseAnalyzer> > &pulse() const {return m_pulse;}

	const shared_ptr<XDoubleNode> &bandWidth() const {return m_bandWidth;}
	//! Tune bandwidth to 50%/100%/200%.
	const shared_ptr<XComboNode> &bwList() const {return m_bwList;}
	//! Deduce phase from data
	const shared_ptr<XBoolNode> &autoPhase() const {return m_autoPhase;}
	//! (Deduced) phase of echoes [deg.]
	const shared_ptr<XDoubleNode> &phase() const {return m_phase;}
	//! Spectrum solvers.
	const shared_ptr<XComboNode> &solverList() const {return m_solverList;}
	///! FFT Window Function
	const shared_ptr<XComboNode> &windowFunc() const {return m_windowFunc;}
	//! Changing width of time-domain image [%]
	const shared_ptr<XDoubleNode> &windowWidth() const {return m_windowWidth;}
	//! Clears stored points.
	const shared_ptr<XTouchableNode> &clear() const {return m_clear;}
protected:
    shared_ptr<Listener> m_lsnOnClear, m_lsnOnCondChanged;
    
	//! \return true to be cleared.
    virtual bool onCondChangedImpl(const Snapshot &shot, XValueNodeBase *) = 0;
	//! [Hz]
	virtual double getFreqResHint(const Snapshot &shot_this) const = 0;
	//! [Hz]
	virtual double getMinFreq(const Snapshot &shot_this) const = 0;
	//! [Hz]
	virtual double getMaxFreq(const Snapshot &shot_this) const = 0;
	//! [Hz]
	virtual double getCurrentCenterFreq(const Snapshot &shot_this, const Snapshot &shot_others) const = 0;
    virtual void rearrangeInstrum(const Snapshot &) {}

	//! How the records of ONE acquisition are distributed over the time bins of
	//! a relaxation map.  A CPMG train gives one record per echo, grouped m at a
	//! time; a driver that has nothing to resolve in time gives none.
	struct MapBinning {
		int binCount = 0;
		std::vector<int> binOfRecord; //!< the bin each record belongs to
		std::vector<double> timeOfRecord; //!< its abscissa, 2 tau n, P1, ...
	};
	//! \return false (the default) to stay time-integrated: one record per
	//! acquisition, no map.  Called from analyze(), so read \a shot_this and
	//! \a shot_pulse only.
	virtual bool mapBinning(const Snapshot &shot_this, const Snapshot &shot_pulse,
		MapBinning &) const {return false;}
	//! \return the wave of record \a idx of one acquisition: an individual CPMG
	//! echo for a time-resolving driver, the (echo-averaged) wave otherwise.
	virtual const std::vector<std::complex<double> > &
		waveOfRecord(const Snapshot &shot_pulse, const XNMRPulseAnalyzer &pulse, int idx) const;
	//! \return by how much darkPSD() understates the noise of what
	//! waveOfRecord() hands over.  It is quoted for the echo-AVERAGED wave, so a
	//! driver binning individual echoes must undo that averaging or every
	//! criterion that reads an absolute noise level (Noise Analysis, and the
	//! weights on the curves) is optimistic by that factor.
	virtual double mapNoiseFactor(const Snapshot &shot_pulse,
		const XNMRPulseAnalyzer &pulse) const {return 1.0;}
	virtual void getValues(const Snapshot &shot_this, std::vector<double> &values) const = 0;
	virtual bool checkDependencyImpl(const Snapshot &shot_this,
		const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) const = 0;
private:
	//! Fourier Step Summation.
	void fssum(Transaction &tr, const Snapshot &shot_pulse, const Snapshot &shot_others);
	//! The same summation, one record at a time into the time bins of the map.
	void fssumTimeResolved(Transaction &tr, const Snapshot &shot_pulse,
		int len, double df, double cfreq, int bw_org);
	//! Creates, clears or shifts the map's time bins along with the sweep axis.
	void updateMapBins(Transaction &tr, const Snapshot &shot_pulse,
		bool axis_rebuilt, int head_shift, int length);
	void analyzeIFT(Transaction &tr, const Snapshot &shot_pulse);

	const shared_ptr<XItemNode<XDriverList, XNMRPulseAnalyzer> > m_pulse;
 
	const shared_ptr<XDoubleNode> m_bandWidth;
	const shared_ptr<XComboNode> m_bwList;
	const shared_ptr<XBoolNode> m_autoPhase;
	const shared_ptr<XDoubleNode> m_phase;
	const shared_ptr<XTouchableNode> m_clear;
	const shared_ptr<XComboNode> m_solverList;
	const shared_ptr<XComboNode> m_windowFunc;
	const shared_ptr<XDoubleNode> m_windowWidth;
	
    std::deque<xqcon_ptr> m_conBaseUIs;

	shared_ptr<SpectrumSolverWrapper> m_solver;
	shared_ptr<XXYPlot> m_peakPlot;

	void onCondChanged(const Snapshot &shot, XValueNodeBase *);

	atomic<int> m_isInstrumControlRequested;
	//! Set by analyze() instead of touching the pulse analyzer's avgClear()
	//! there: that touch commits at once and restarts the DSO through the
	//! listener chain, which analyze() must not do. visualize() performs it.
	atomic<int> m_isAvgClearRequested{0};
protected:
	const qshared_ptr<FRM> m_form;
	const shared_ptr<XStatusPrinter> m_statusPrinter;
	const shared_ptr<XWaveNGraph> m_spectrum;
	void onClear(const Snapshot &shot, XTouchableNode *);
};

#endif
