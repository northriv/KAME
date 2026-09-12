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
#ifndef nmrfspectrumH
#define nmrfspectrumH

#include "nmrspectrumbase.h"
#include "nmrrelaxmap.h"

class XSG;
class XPulser;
class XAutoLCTuner;
class XRelaxFunc;
class XRelaxFuncList;
class XWaveNGraph;
class QMainWindow;
class Ui_FrmNMRFSpectrum;
typedef QForm<QMainWindow, Ui_FrmNMRFSpectrum> FrmNMRFSpectrum;

class XNMRFSpectrum : public XNMRSpectrumBase<FrmNMRFSpectrum> {
public:
	XNMRFSpectrum(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! ususally nothing to do
    virtual ~XNMRFSpectrum() = default;
protected:
	//! \return true to be cleared.
    virtual bool onCondChangedImpl(const Snapshot &shot, XValueNodeBase *) override;
    virtual double getFreqResHint(const Snapshot &shot_this) const override;
    virtual double getMinFreq(const Snapshot &shot_this) const override;
    virtual double getMaxFreq(const Snapshot &shot_this) const override;
    virtual double getCurrentCenterFreq(const Snapshot &shot_this, const Snapshot &shot_others) const override;
    virtual void getValues(const Snapshot &shot_this, std::vector<double> &values) const override;

	virtual bool checkDependencyImpl(const Snapshot &shot_this,
		const Snapshot &shot_emitter, const Snapshot &shot_others,
        XDriver *emitter) const override;

    virtual void rearrangeInstrum(const Snapshot &shot) override;

    //! Inverts the time-resolved accumulation into a T2 map, on top of what the
    //! base class draws. \sa nmrrelaxmap.h
    virtual void visualize(const Snapshot &shot) override;
    //! One record per CPMG echo, grouped \a mapEchoesPerBin() at a time.
    virtual bool mapBinning(const Snapshot &shot_this, const Snapshot &shot_pulse,
        MapBinning &) const override;
    virtual const std::vector<std::complex<double> > &
        waveOfRecord(const Snapshot &shot_pulse, const XNMRPulseAnalyzer &pulse,
        int idx) const override;
    //! Every record binned here is one echo of the train, not their mean.
    virtual double mapNoiseFactor(const Snapshot &shot_pulse,
        const XNMRPulseAnalyzer &pulse) const override;
public:
	//! driver specific part below 
	const shared_ptr<XItemNode<XDriverList, XSG> > &sg1() const {return m_sg1;}
	const shared_ptr<XItemNode<XDriverList, XAutoLCTuner> > &autoTuner() const {return m_autoTuner;}
    const shared_ptr<XItemNode<XDriverList, XAutoLCTuner> > &autoTunerSecondary() const {return m_autoTunerSecondary;}
    const shared_ptr<XItemNode<XDriverList, XPulser> > &pulser() const {return m_pulser;}
	//! Offset for IF [MHz]
	const shared_ptr<XDoubleNode> &sg1FreqOffset() const {return m_sg1FreqOffset;}
	//! [MHz]
	const shared_ptr<XDoubleNode> &centerFreq() const {return m_centerFreq;}
	//! [kHz]
	const shared_ptr<XDoubleNode> &freqSpan() const {return m_freqSpan;}
	//! [kHz]
	const shared_ptr<XDoubleNode> &freqStep() const {return m_freqStep;}
	const shared_ptr<XBoolNode> &active() const {return m_active;}
	//! [MHz]
    const shared_ptr<XDoubleNode> &tuneCycleStep() const {return m_tuneCycleStep;}
    const shared_ptr<XComboNode> &tuneCycleStrategy() const {return m_tuneCycleStrategy;}
    enum class TuneCycleStrategy {ASIS = 0, TUNE_AWAIT = 1, AUTOTUNE = 2,
                            CYCLE_DBL = 3, CYCLE_QUAD = 4, CYCLE_OCT = 5};

    //! Relaxation map: a T2 distribution per frequency, out of the CPMG train
    //! the pulse analyzer stores echo by echo. \sa NMRRelaxMapMode
    const shared_ptr<XComboNode> &mapMode() const {return m_mapMode;}
    const shared_ptr<XComboNode> &mapTikhonovMatrix() const {return m_mapTikhonovMatrix;}
    //! # of consecutive echoes summed into one time bin, which then sits at the
    //! mean of their 2 tau n -- i.e. bins of 2 tau n/m rather than 2 tau n.
    const shared_ptr<XUIntNode> &mapEchoesPerBin() const {return m_mapEchoesPerBin;}
    //! Resolution of the map's frequency axis [kHz]; <= 0 takes the spectrum's.
    const shared_ptr<XDoubleNode> &mapFreqRes() const {return m_mapFreqRes;}
    //! What the inversion is fed at each frequency of the sweep. \sa MapPhaseMode
    const shared_ptr<XComboNode> &mapPhase() const {return m_mapPhase;}
    //! Decades of relaxation time to put on the grid BEYOND the echo train,
    //! for what has not finished decaying by its end.  Nothing out there is
    //! resolved -- every column past the last echo decays by less than 1/e
    //! across the whole train, so they are nearly the same column -- and only
    //! the weight that lands there means anything: "longer than the train".
    //! 0 keeps the grid to what was measured, and then such a component has
    //! nowhere to go but the last grid point, where it piles up.
    const shared_ptr<XDoubleNode> &mapTExtDecades() const {return m_mapTExtDecades;}
    //! A swept carrier does not keep one phase -- the probe, the cables and the
    //! synthesizer all turn it -- so the single phase() the spectrum carries
    //! cannot put every frequency in phase at once.  \a AutoPerFreq settles it
    //! frequency by frequency instead, from the phase of the sum over the time
    //! bins; the train at one frequency does share one phase, since relaxation
    //! is real.  \a Global keeps phase() for a rig whose sweep is coherent, and
    //! \a Absolute gives up the phase altogether -- noisier, and biased away
    //! from zero at long times, which is why it is not the default.
    enum class MapPhaseMode {AutoPerFreq = 0, Global = 1, Absolute = 2};
    //! Shape of the decay, e.g. multi-exponential for I > 1/2.
    const shared_ptr<XItemNode<XRelaxFuncList, XRelaxFunc> > &relaxFunc() const {return m_relaxFunc;}
private:
	const shared_ptr<XItemNode<XDriverList, XSG> > m_sg1;
    const shared_ptr<XItemNode<XDriverList, XAutoLCTuner> > m_autoTuner, m_autoTunerSecondary;
	const shared_ptr<XItemNode<XDriverList, XPulser> > m_pulser;
	const shared_ptr<XDoubleNode> m_sg1FreqOffset;

	const shared_ptr<XDoubleNode> m_centerFreq;
	const shared_ptr<XDoubleNode> m_freqSpan;
	const shared_ptr<XDoubleNode> m_freqStep;
	const shared_ptr<XBoolNode> m_active;
    const shared_ptr<XDoubleNode> m_tuneCycleStep;
    const shared_ptr<XComboNode> m_tuneCycleStrategy;

    const shared_ptr<XRelaxFuncList> m_relaxFuncs;
    const shared_ptr<XComboNode> m_mapMode;
    const shared_ptr<XComboNode> m_mapTikhonovMatrix;
    const shared_ptr<XUIntNode> m_mapEchoesPerBin;
    const shared_ptr<XDoubleNode> m_mapFreqRes;
    const shared_ptr<XComboNode> m_mapPhase;
    const shared_ptr<XDoubleNode> m_mapTExtDecades;
    const shared_ptr<XWaveNGraph> m_waveMapCurves, m_waveMap;
    shared_ptr<XItemNode<XRelaxFuncList, XRelaxFunc> > m_relaxFunc;
    //! Touched by visualize() only; analyze() may ask it to drop its kernel.
    NMRRelaxMapSolver m_mapSolver;

    //! Empties both map graphs, unless they are empty already.
    void clearRelaxMapGraphs();

    shared_ptr<Listener> m_lsnOnActiveChanged, m_lsnOnTuningChanged;
    
    std::deque<xqcon_ptr> m_conUIs;

	void onActiveChanged(const Snapshot &shot, XValueNodeBase *);
	void onTuningChanged(const Snapshot &shot, XValueNodeBase *); //!< receives signals from AutoLCTuner.
    void performTuning(const Snapshot &shot_this, double newf);

    double m_lastFreqAcquired; //!< to avoid inifite averaging after a sweep.
    double m_tunedFreq;
    int m_lastCycle; //!< 0-7
};


#endif
