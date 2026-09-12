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
//---------------------------------------------------------------------------
#include "ui_nmrfspectrumform.h"
#include "nmrfspectrum.h"
#include "signalgenerator.h"
#include "nmrspectrumbase_impl.h"
#include "autolctuner.h"
#include "pulserdriver.h"
#include "nmrpulse.h"
#include "nmrrelaxfit.h"
#include "graph.h"
#include "xwavengraph.h"

REGISTER_TYPE(XDriverList, NMRFSpectrum, "NMR frequency-swept spectrum measurement");

//---------------------------------------------------------------------------
XNMRFSpectrum::XNMRFSpectrum(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
	: XNMRSpectrumBase<FrmNMRFSpectrum>(name, runtime, ref(tr_meas), meas),
	  m_sg1(create<XItemNode<XDriverList, XSG> >(
		  "SG1", false, ref(tr_meas), meas->drivers(), true)),
	  m_autoTuner(create<XItemNode<XDriverList, XAutoLCTuner> >(
          "AutoTuner", false, ref(tr_meas), meas->drivers(), true)),
      m_autoTunerSecondary(create<XItemNode<XDriverList, XAutoLCTuner> >(
          "AutoTunerSecondary", false, ref(tr_meas), meas->drivers(), false)),
      m_pulser(create<XItemNode<XDriverList, XPulser> >(
		  "Pulser", false, ref(tr_meas), meas->drivers(), true)),
	  m_sg1FreqOffset(create<XDoubleNode>("SG1FreqOffset", false)),
	  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
	  m_freqSpan(create<XDoubleNode>("FreqSpan", false)),
	  m_freqStep(create<XDoubleNode>("FreqStep", false)),
	  m_active(create<XBoolNode>("Active", true)),
      m_tuneCycleStep(create<XDoubleNode>("TuneCycleStep", false)),
      m_tuneCycleStrategy(create<XComboNode>("TuneCycleStrategy", false, true)),
      m_relaxFuncs(create<XRelaxFuncList>("RelaxFuncs", true)),
      m_mapMode(create<XComboNode>("MapMode", false, true)),
      m_mapTikhonovMatrix(create<XComboNode>("MapTikhonovMatrix", false, true)),
      m_mapEchoesPerBin(create<XUIntNode>("MapEchoesPerBin", false)),
      m_mapFreqRes(create<XDoubleNode>("MapFreqRes", false, "%.4f")),
      m_mapPhase(create<XComboNode>("MapPhase", false, true)),
      m_mapTExtDecades(create<XDoubleNode>("MapTExtDecades", false, "%.2f")),
      m_waveMapCurves(create<XWaveNGraph>("RelaxCurves", false, m_form->m_graphMapCurves,
          m_form->m_edMapCurvesDump, m_form->m_tbMapCurvesDump, m_form->m_btnMapCurvesDump)),
      m_waveMap(create<XWaveNGraph>("RelaxMap", false, m_form->m_graphRelaxMap,
          m_form->m_edRelaxMapDump, m_form->m_tbRelaxMapDump, m_form->m_btnRelaxMapDump)) {

	connect(sg1());
//	connect(autoTuner());
//	connect(pulser());

	m_form->setWindowTitle(i18n("NMR Spectrum (Freq. Sweep) - ") + getLabel() );

	iterate_commit([=](Transaction &tr){
		//Inserted online: a node created outside tr would be invisible to it.
		m_relaxFunc = create<XItemNode<XRelaxFuncList, XRelaxFunc> >(
			tr, "RelaxFunc", false, tr, m_relaxFuncs, true);
    });

	iterate_commit([=](Transaction &tr){
		tr[ *m_spectrum].setLabel(0, "Freq [MHz]");
		tr[ *tr[ *m_spectrum].axisx()->label()] = i18n("Freq [MHz]");

		tr[ *centerFreq()] = 20;
        tr[ *sg1FreqOffset()] = 0;
		tr[ *freqSpan()] = 200;
		tr[ *freqStep()] = 1;

        for(auto &&x: {"As is", "Await Manual Tune", "Auto Tune", "Cyclic Avg. BPSK", "Cyclic Avg. QPSK", "Cyclic Avg. QPSKxP.I"})
            tr[ *tuneCycleStrategy()].add(x);
        tr[ *tuneCycleStrategy()] = (int)TuneCycleStrategy::ASIS;

        addRelaxMapModeItems(tr, mapMode());
        tr[ *mapMode()] = (int)NMRRelaxMapMode::Off;
        addTikhonovMatrixItems(tr, mapTikhonovMatrix());
        tr[ *mapTikhonovMatrix()] = (int)TikhonovRegular::TikhonovMatrix::I;
        tr[ *relaxFunc()].str(XString("NMR I=1/2"));
        tr[ *mapEchoesPerBin()] = 1;
        tr[ *mapFreqRes()] = 0.0;
        tr[ *mapPhase()].add({"Auto per Freq.", "Global", "Absolute"});
        tr[ *mapPhase()] = (int)MapPhaseMode::AutoPerFreq;
        tr[ *mapTExtDecades()] = 1.0;
        if( !setupRelaxCurvesGraph(tr, m_waveMapCurves, "Freq [MHz]", "2tau [us]")) return;
        if( !setupRelaxDensityMapGraph(tr, m_waveMap, "Freq [MHz]", "T2 [us]")) return;
    });

    //Ranges should be preset in prior to connectors.
    m_form->m_spbMapEchoesPerBin->setRange(1, 1024);
  
    m_conUIs = {
        xqcon_create<XQLineEditConnector>(m_sg1FreqOffset, m_form->m_edSG1FreqOffset),
        xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edCenterFreq),
        xqcon_create<XQLineEditConnector>(m_freqSpan, m_form->m_edFreqSpan),
        xqcon_create<XQLineEditConnector>(m_freqStep, m_form->m_edFreqStep),
        xqcon_create<XQComboBoxConnector>(m_sg1, m_form->m_cmbSG1, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(m_autoTuner, m_form->m_cmbAutoTuner, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(m_autoTunerSecondary, m_form->m_cmbAutoTunerSecondary, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(m_pulser, m_form->m_cmbPulser, ref(tr_meas)),
        xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive),
        xqcon_create<XQLineEditConnector>(m_tuneCycleStep, m_form->m_edTuneCycleStep),
        xqcon_create<XQComboBoxConnector>(m_tuneCycleStrategy, m_form->m_cmbTuneCycleStrategy, Snapshot( *m_tuneCycleStrategy)),
        xqcon_create<XQComboBoxConnector>(m_mapMode, m_form->m_cmbMapMode, Snapshot( *m_mapMode)),
        xqcon_create<XQComboBoxConnector>(m_mapTikhonovMatrix, m_form->m_cmbMapTikhonovMatrix, Snapshot( *m_mapTikhonovMatrix)),
        xqcon_create<XQComboBoxConnector>(m_relaxFunc, m_form->m_cmbMapRelaxFunc, Snapshot( *m_relaxFuncs)),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_mapEchoesPerBin, m_form->m_spbMapEchoesPerBin),
        xqcon_create<XQLineEditConnector>(m_mapFreqRes, m_form->m_edMapFreqRes),
        xqcon_create<XQComboBoxConnector>(m_mapPhase, m_form->m_cmbMapPhase, Snapshot( *m_mapPhase)),
        xqcon_create<XQLineEditConnector>(m_mapTExtDecades, m_form->m_edMapTExtDecades)
    };

	iterate_commit([=](Transaction &tr){
		m_lsnOnActiveChanged = tr[ *active()].onValueChanged().connectWeakly(
			shared_from_this(), &XNMRFSpectrum::onActiveChanged);
		tr[ *centerFreq()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *freqSpan()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *freqStep()].onValueChanged().connect(m_lsnOnCondChanged);
		//None of these clears the spectrum (onCondChangedImpl returns false):
		//regrouping the echoes or changing the criterion must not throw a sweep
		//away.  The bins themselves are rebuilt by updateMapBins() when the
		//binning no longer matches what was accumulated.
		for(auto &&x: std::vector<shared_ptr<XValueNodeBase>>(
			{mapMode(), mapTikhonovMatrix(), mapEchoesPerBin(), mapFreqRes(),
			relaxFunc(), mapPhase(), mapTExtDecades()}))
			tr[ *x].onValueChanged().connect(m_lsnOnCondChanged);
    });
}

void
XNMRFSpectrum::onActiveChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    if(shot_this[ *active()]) {
        switch((TuneCycleStrategy)(int)shot_this[ *tuneCycleStrategy()]) {
        case TuneCycleStrategy::ASIS:
            break;
        case TuneCycleStrategy::CYCLE_DBL:
        case TuneCycleStrategy::CYCLE_QUAD:
        case TuneCycleStrategy::CYCLE_OCT:
            {
                double x = shot_this[ *tuneCycleStep()] / shot_this[ *freqStep()];
                if((x < 0.9) || (fabs(x - lrint(x)) > 0.003 * x))
                    gErrPrint(i18n("Invalid cyclic step."));
            }
        case TuneCycleStrategy::TUNE_AWAIT:
        case TuneCycleStrategy::AUTOTUNE:
            {
                shared_ptr<XPulser> pulser__ = shot_this[ *pulser()];
                if( !pulser__)
                    gErrPrint(i18n("Pulser should be selected."));
                pulser__->iterate_commit([=](Transaction &tr){
                    tr[ *pulser__->firstPhase()] = 0;
                    tr[ *pulser__->invertPhase()] = false;
                });
            }
            if(shot_this[ *tuneCycleStep()]  <= 0.0)
                gErrPrint(i18n("Invalid tuning/cyclic step."));
            break;
        }

		onClear(shot_this, clear().get());
        m_lastFreqAcquired = -1000.0;
        m_tunedFreq = -1000.0;
        m_lastCycle = 0;
        double newf = getMinFreq(shot_this) * 1e-6; //MHz
        performTuning(shot_this, newf);
        newf += shot_this[ *sg1FreqOffset()];
		shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
		if(sg1__)
			trans( *sg1__->freq()) = newf;
	}
    else
    	m_lsnOnTuningChanged.reset();
}
bool
XNMRFSpectrum::onCondChangedImpl(const Snapshot &shot, XValueNodeBase *) {
    m_lastFreqAcquired = -1000.0;
    return false;
}
bool
XNMRFSpectrum::checkDependencyImpl(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
    if( !sg1__) return false;
    shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
    if(emitter != pulse__.get()) return false;
//    if(shot_emitter[ *pulse__].timeAwared() < shot_others[ *sg1__].time()) return false;
    double freq = getCurrentCenterFreq(shot_this, shot_others);
    if(m_lastFreqAcquired == freq) {
        return false; //skips for the same freq.
    }
    return true;
}
double
XNMRFSpectrum::getMinFreq(const Snapshot &shot_this) const{
	double cfreq = shot_this[ *centerFreq()]; //MHz
	double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
	return (cfreq - freq_span/2) * 1e6;
}
double
XNMRFSpectrum::getMaxFreq(const Snapshot &shot_this) const{
	double cfreq = shot_this[ *centerFreq()]; //MHz
	double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
	return (cfreq + freq_span/2) * 1e6;
}
double
XNMRFSpectrum::getFreqResHint(const Snapshot &shot_this) const {
	return 1e-6;
}
double
XNMRFSpectrum::getCurrentCenterFreq(const Snapshot &shot_this, const Snapshot &shot_others) const {
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
	assert( sg1__ );
	assert(shot_others[ *sg1__].time().isSet() );
    double freq = shot_others[ *sg1__].freq() - shot_this[ *sg1FreqOffset()]; //MHz
	return freq * 1e6;
}
void
XNMRFSpectrum::performTuning(const Snapshot &shot_this, double newf) {
    if((shot_this[ *tuneCycleStrategy()] != (int)TuneCycleStrategy::AUTOTUNE) &&
         (shot_this[ *tuneCycleStrategy()] != (int)TuneCycleStrategy::TUNE_AWAIT))
        return; //tuning is declined by user.

    if(fabs(m_tunedFreq - newf) <= shot_this[ *tuneCycleStep()] / 2 * 1e-3)
        return; //not needed yet

    newf += shot_this[ *tuneCycleStep()] / 2 * 1e-3; //to be tuned to

    shared_ptr<XPulser> pulser__ = shot_this[ *pulser()];
    if( !pulser__) {
        gWarnPrint(i18n("Pulser should be selected."));
        return;
    }
    //Tunes Capacitors.
    trans( *pulser__->output()) = false; // Pulse off.
    shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
    if(pulse__)
        trans( *pulse__->avgClear()).touch();

    if((shot_this[ *tuneCycleStrategy()] == (int)TuneCycleStrategy::TUNE_AWAIT)) {
        g_statusPrinter->printMessage(getLabel() + " " + i18n("Tune it by yourself to") +
            formatString(" %.3f~MHz", newf) + i18n(", and turn pulse on again."), true, __FILE__, __LINE__, true);
    }
    if((shot_this[ *tuneCycleStrategy()] == (int)TuneCycleStrategy::AUTOTUNE)) {
        shared_ptr<XAutoLCTuner> autotuner = shot_this[ *autoTuner()];
        if( !autotuner) {
            gWarnPrint(i18n("AutoTuner should be selected."));
            return;
        }
        autotuner->iterate_commit([=](Transaction &tr){
            m_lsnOnTuningChanged = tr[ *autotuner->tuning()].onValueChanged().connectWeakly(
                shared_from_this(), &XNMRFSpectrum::onTuningChanged);
            tr[ *autotuner->target()] = newf;
        });
        if(shared_ptr<XAutoLCTuner> autotuner2 = shot_this[ *autoTunerSecondary()]) {
            autotuner2->iterate_commit([=](Transaction &tr){
                tr[ *autotuner2->tuning()].onValueChanged().connect(m_lsnOnTuningChanged);
            });
        }
    }
    m_tunedFreq = newf;
}
void
XNMRFSpectrum::rearrangeInstrum(const Snapshot &shot_this) {
    //Never touch m_lsnOnTuningChanged while an auto-tuner is running: a stray
    //pulse record analyzed during tuning (typically the acquisition already in
    //flight when performTuning() turned the pulser off, attributed to the
    //already-moved SG frequency and thus passing the m_lastFreqAcquired guard)
    //can complete a step and reach here.  The unconditional reset below would
    //then steal the tuning-finished event, and the pulser would never be
    //turned back on after "succeeded" (2026-07 bug report).  Skipping is
    //correct: with the pulser off such a record is noise, and instrument
    //control resumes via onTuningChanged() once tuning finishes.
    const shared_ptr<XAutoLCTuner> tuners[] = {
        shot_this[ *autoTuner()], shot_this[ *autoTunerSecondary()]};
    for(auto &&tuner: tuners) {
        if(tuner && Snapshot( *tuner)[ *tuner->tuning()])
            return;
    }
    m_lsnOnTuningChanged.reset();
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
    if( ! sg1__)
        return;
    Snapshot shot_sg( *sg1__);
    if( !shot_sg[ *sg1__].time())
        return;
    double freq = getCurrentCenterFreq(shot_this, shot_sg);
    m_lastFreqAcquired = freq; //suppresses double accumulation.
    freq *= 1e-6; //MHz
    //sets new freq
	if(shot_this[ *active()]) {
	    double cfreq = shot_this[ *centerFreq()]; //MHz
		double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
		if(cfreq <= freq_span / 2) {
			throw XRecordError(i18n("Invalid center freq."), __FILE__, __LINE__);
		}

        double freq_step = shot_this[ *tuneCycleStep()] * 1e-3; //MHz
        int num_psk_cycles = 0;
        switch((TuneCycleStrategy)(int)shot_this[ *tuneCycleStrategy()]) {
        default:
            freq_step = shot_this[ *freqStep()] * 1e-3; //MHz
            break;
        case TuneCycleStrategy::CYCLE_DBL:
            num_psk_cycles = 2; break;
        case TuneCycleStrategy::CYCLE_QUAD:
            num_psk_cycles = 4; break;
        case TuneCycleStrategy::CYCLE_OCT:
            num_psk_cycles = 8; break;
        }
        if(freq_span < freq_step * 1.5) {
			throw XRecordError(i18n("Too large freq. step."), __FILE__, __LINE__);
		}
	  
		double newf = freq; //MHz
		newf += freq_step;
		
        if(newf >= getMaxFreq(shot_this) * 1e-6) {
            double x = (newf - getMinFreq(shot_this) * 1e-6) / freq_step;
            newf -= floor(x + 0.01) * freq_step; //the first freq of this cycle.
            ++m_lastCycle;
            if(m_lastCycle >= num_psk_cycles) {
                m_lastCycle = 0;
                newf += shot_this[ *freqStep()] * 1e-3; //shifted.
                if((newf - getMinFreq(shot_this) * 1e-6) / freq_step > 0.99) {
                    trans( *active()) = false; //finish
                    return;
                }
            }
            shared_ptr<XPulser> pulser__ = shot_this[ *pulser()];
            if( !pulser__)
                throw XRecordError(i18n("Pulser should be selected."), __FILE__, __LINE__);
            if( ***pulser__->rtime() < 1000)
                throw XRecordError(i18n("Too short repetition period."), __FILE__, __LINE__);
            pulser__->iterate_commit([=](Transaction &tr){
                tr[ *pulser__->firstPhase()] = m_lastCycle % 4;
                tr[ *pulser__->invertPhase()] = (m_lastCycle / 4) ? true : false;
            });
        }

        newf = round(newf * 1e8) / 1e8; //rounds

        performTuning(shot_this, newf); //tunes a circuit if needed.

        newf += shot_this[ *sg1FreqOffset()]; //modifies SG freq.
        if(sg1__)
            trans( *sg1__->freq()) = newf;
    }
}
void
XNMRFSpectrum::onTuningChanged(const Snapshot &shot, XValueNodeBase *node) {
    Snapshot shot_this( *this);
    shared_ptr<XPulser> pulser__ = shot_this[ *pulser()];
    if( !pulser__) return;
//    if(shot_this[ *tuneStrategy()] != TUNEAUTOTUNER) return;
    {
        shared_ptr<XAutoLCTuner> autotuner = shot_this[ *autoTuner()];
        if(autotuner && (autotuner->tuning().get() == node)) {
            Snapshot shot_tuner( *autotuner);
            if(shot_tuner[ *autotuner->tuning()])
                return; //still tuner is running.
            if( !shot_tuner[ *autotuner->succeeded()])
                return; //awaiting manual tuning.
            //Primary tuning just finished.
            if(shared_ptr<XAutoLCTuner> autotuner2 = shot_this[ *autoTunerSecondary()]) {
                if(autotuner2 != autotuner) {
                    //Starting the secondary tuner puts the RF relay back on the
                    //tuning path, so the pulser must stay off until the secondary
                    //has finished too.  Keep m_lsnOnTuningChanged connected and
                    //leave: its own tuning-finished event resumes the sweep.
                    //Falling through to "Pulse on." below would fire into the
                    //VNA path while the capacitors are still moving.
                    trans( *autotuner2->target()) = (double)shot_tuner[ *autotuner->target()];
                    if(Snapshot( *autotuner2)[ *autotuner2->tuning()])
                        return;
                }
                //Same driver as the primary: the write above would be a no-op
                //(onTargetChanged only fires on an actual value change), so there
                //is no second tuning to wait for.
            }
        }
        else if(shared_ptr<XAutoLCTuner> autotuner2 = shot_this[ *autoTunerSecondary()]) {
            Snapshot shot_tuner( *autotuner2);
            if(shot_tuner[ *autotuner2->tuning()])
                return; //still tuner is running.
            if( !shot_tuner[ *autotuner2->succeeded()])
                return; //awaiting manual tuning.
        }
    }
    m_lsnOnTuningChanged.reset();
    if(shot_this[ *active()]) {
        //Tuning has succeeded, go on.
        trans( *pulser__->output()) = true; // Pulse on.
        shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
        if(pulse__)
            trans( *pulse__->avgClear()).touch();
    }
}
void
XNMRFSpectrum::getValues(const Snapshot &shot_this, std::vector<double> &values) const {
	int wave_size = shot_this[ *this].wave().size();
	double min__ = shot_this[ *this].min();
	double res = shot_this[ *this].res();
	values.resize(wave_size);
	for(unsigned int i = 0; i < wave_size; i++) {
		double freq = min__ + i * res;
		values[i] = freq * 1e-6;
	}
}

bool
XNMRFSpectrum::mapBinning(const Snapshot &shot_this, const Snapshot &shot_pulse,
    MapBinning &binning) const {
    if((NMRRelaxMapMode)(int)shot_this[ *mapMode()] == NMRRelaxMapMode::Off)
        return false;
    shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
    if( !pulse__)
        return false;
    //The train as the pulse analyzer stored it: one record per echo, the n-th
    //of them at 2 tau n.  Summing m of them into a bin puts that bin at the
    //mean of their times, i.e. at 2 tau n/m -- the kernel row of the bin is
    //then the mean of the rows of its members, which is exact (\sa
    //NMRRelaxMapData).  It buys S/N and a smaller problem at the price of time
    //resolution, and m = 1 leaves every echo its own bin.
    int nechoes = (int)shot_pulse[ *pulse__].echoesT2().size();
    double twotau = shot_pulse[ *pulse__].echoPeriod() * 1e3; //[ms] -> [us]
    if((nechoes < 2) || (twotau <= 0.0))
        return false;
    int m = (int)std::max(1u, (unsigned int)shot_this[ *mapEchoesPerBin()]);
    //A remainder goes into the LAST bin rather than into a short one of its
    //own: that bin already carries the weakest signal of the train, and the
    //inversion weights every bin alike, so a bin of m/2 echoes would be the
    //noisiest point of the curve and still count as much as the first.  Its
    //kernel row is the mean over the echoes it really holds, so nothing is
    //biased by being fed more of them.
    binning.binCount = std::max(1, nechoes / m);
    binning.binOfRecord.resize(nechoes);
    binning.timeOfRecord.resize(nechoes);
    for(int i = 0; i < nechoes; ++i) {
        binning.binOfRecord[i] = std::min(i / m, binning.binCount - 1);
        binning.timeOfRecord[i] = twotau * (i + 1);
    }
    return true;
}
const std::vector<std::complex<double> > &
XNMRFSpectrum::waveOfRecord(const Snapshot &shot_pulse, const XNMRPulseAnalyzer &pulse,
    int idx) const {
    const std::vector<std::vector<std::complex<double> > > &echoes(shot_pulse[pulse].echoesT2());
    if((idx >= 0) && (idx < (int)echoes.size()))
        return echoes[idx];
    return shot_pulse[pulse].wave();
}
double
XNMRFSpectrum::mapNoiseFactor(const Snapshot &shot_pulse, const XNMRPulseAnalyzer &pulse) const {
    return shot_pulse[pulse].darkPSDFactorPerEcho();
}
void
XNMRFSpectrum::clearRelaxMapGraphs() {
    for(auto &&graph: {m_waveMapCurves, m_waveMap}) {
        if( !Snapshot( *graph)[ *graph].rowCount())
            continue; //already empty; spares a commit per record while off.
        graph->iterate_commit([&](Transaction &tr){
            tr[ *graph->graph()->onScreenStrings()] = "";
            tr[ *graph].clearPoints();
            graph->drawGraph(tr);
        });
    }
}
void
XNMRFSpectrum::visualize(const Snapshot &shot) {
    XNMRSpectrumBase<FrmNMRFSpectrum>::visualize(shot);

    auto mapmode = (NMRRelaxMapMode)(int)shot[ *mapMode()];
    const std::vector<shared_ptr<const Payload::MapBin> > &bins(shot[ *this].mapBins());
    int nbin = (int)bins.size();
    int len = (int)shot[ *this].wave().size();
    if( !shot[ *this].time() || (mapmode == NMRRelaxMapMode::Off) || (nbin < 2) || (len < 2)) {
        clearRelaxMapGraphs();
        return;
    }
    //The grid of relaxation times covers what was measured, 2 tau to 2 tau x n,
    //and -- by mapTExtDecades() -- however far beyond the user is prepared to
    //read as "did not finish decaying" or "was over before we looked".  Out
    //there nothing is resolved: past the last echo every column decays by less
    //than 1/e across the whole train, so they are nearly one column, and only
    //the total weight that lands there carries meaning.  With no extension at
    //all, though, such a component has nowhere to go but the end grid point,
    //and piles up on it.
    double tfirst = 0.0, tlast = 0.0;
    for(auto &&bin: bins) {
        for(double t: bin->times) {
            if((tfirst == 0.0) || (t < tfirst)) tfirst = t;
            if(t > tlast) tlast = t;
        }
    }
    if(tlast <= tfirst)
        return;
    double ext = std::max(0.0, std::min(3.0, (double)shot[ *mapTExtDecades()]));
    //Half as far below as above: below the first echo the columns do not merely
    //resemble one another, they vanish.  A component at 2 tau / 3 still leaves
    //5% of itself in the first echo; one a decade down leaves nothing in any of
    //them, and an unknown the data cannot touch buys nothing -- while the
    //negative lobes it invites are answered with more smoothing, map-wide.
    double tmax = tlast * pow(10.0, ext);
    double tmin = tfirst * pow(10.0, -0.5 * ext);
    int ntcount = std::min(200, nbin * 10);

    double res = shot[ *this].res();
    double min__ = shot[ *this].min();
    //The map's frequency axis merges adjacent points of the sweep axis: fewer
    //unknowns, better S/N per curve, and -- with the cap below, as in XNMRT1 --
    //a plot that cannot grow without bound when the span or the resolution does.
    int decim = 1;
    if(shot[ *mapFreqRes()] > 0.0)
        decim = std::max(1L, lrint(shot[ *mapFreqRes()] * 1e3 / res));
    int nx = len / decim;
    constexpr long MAX_MAP_POINTS = 50000;
    while((nx > 1) && ((long)nx * std::max(nbin, ntcount) > MAX_MAP_POINTS)) {
        decim *= 2;
        nx = len / decim;
    }
    if(nx < 1)
        return;

    NMRRelaxMapData data;
    data.resize(nx, nbin);
    for(int i = 0; i < nx; ++i)
        data.xvalues[i] = (min__ + (i * decim + 0.5 * (decim - 1)) * res) * 1e-6; //[MHz]
    auto phmode = (MapPhaseMode)(int)shot[ *mapPhase()];
    auto cph = std::polar(1.0, -(double)shot[ *phase()] / 180.0 * M_PI);
    //The dark power accumulated alongside the signal, as a variance per point.
    //It is the one the pulse analyzer quotes for the echo-AVERAGED wave, so it
    //underestimates that of a single echo; only the KnownError criterion (Noise
    //Analysis) reads it as an absolute, the others use the curves themselves.
    double psdcoeff = shot[ *this].mapPSDCoeff();
    double th = FFT::windowFuncHamming(0.1);
    double noisesq = 0.0;
    int noisecnt = 0;
    for(int b = 0; b < nbin; ++b) {
        const Payload::MapBin &bin( *bins[b]);
        data.timesOfBin[b] = bin.times;
        int size = (int)bin.accum.size();
        for(int i = 0; i < nx; ++i) {
            std::complex<double> sum(0.0);
            double w = 0.0, dark = 0.0;
            for(int k = i * decim; (k < (i + 1) * decim) && (k < size); ++k) {
                sum += bin.accum[k];
                w += bin.accum_weights[k];
                dark += bin.accum_dark[k];
            }
            if(w <= th)
                continue; //never swept here, or too far off the excitation.
            //Still unrotated: the phase is settled per frequency below.
            std::complex<double> z = sum / w;
            data.y.coeffRef(i, b) = std::real(z);
            data.yimag.coeffRef(i, b) = std::imag(z);
            double sigmasq = dark / (w * w) * psdcoeff;
            if(sigmasq > 0.0) {
                data.isigma.coeffRef(i, b) = 1.0 / sqrt(sigmasq);
                noisesq += sigmasq;
                ++noisecnt;
            }
        }
    }
    data.noiseSq = noisecnt ? (noisesq / noisecnt) : 0.0;

    //A swept carrier does not keep one phase: the probe's tuning, the cable
    //delay and the synthesizer all turn it as the sweep moves, so the one
    //phase() the spectrum carries cannot put every frequency in phase at once
    //-- which is the difference from XNMRT1, where every point is acquired at
    //the same carrier.  The train at ONE frequency does share a phase, though,
    //since the relaxation behind it is real and positive.  So the phase is
    //settled frequency by frequency, from the sum over that frequency's bins,
    //and the inversion is left with a real signal that decays to zero: what
    //the kernel says it does, and what makes the non-negativity of the density
    //meaningful.  Each bin enters the sum weighted by its own magnitude, so
    //bins whose signal has already decayed contribute noise, not direction.
    for(int i = 0; i < nx; ++i) {
        std::complex<double> rot(1.0, 0.0);
        switch(phmode) {
        case MapPhaseMode::Global:
        default:
            rot = cph;
            break;
        case MapPhaseMode::AutoPerFreq: {
            std::complex<double> sum(0.0, 0.0);
            for(int b = 0; b < nbin; ++b) {
                std::complex<double> z(data.y.coeff(i, b), data.yimag.coeff(i, b));
                sum += z * std::abs(z);
            }
            double a = std::abs(sum);
            if(a > 0.0)
                rot = std::conj(sum) / a; //exp(-i arg(sum))
            break;
            }
        case MapPhaseMode::Absolute:
            //The magnitude has no phase to get wrong, but it does not decay to
            //zero either: it settles on the noise floor, which the kernel has
            //no term for and the inversion could only explain by inventing a
            //component that never decays.  Taking its own variance back out
            //restores the zero asymptote, in expectation.
            for(int b = 0; b < nbin; ++b) {
                double re = data.y.coeff(i, b), im = data.yimag.coeff(i, b);
                double isig = data.isigma.coeff(i, b);
                double sq = re * re + im * im - ((isig > 0.0) ? 1.0 / (isig * isig) : 0.0);
                data.y.coeffRef(i, b) = (sq > 0.0) ? sqrt(sq) : 0.0;
                data.yimag.coeffRef(i, b) = 0.0;
            }
            continue;
        }
        for(int b = 0; b < nbin; ++b) {
            std::complex<double> z(data.y.coeff(i, b), data.yimag.coeff(i, b));
            z *= rot;
            data.y.coeffRef(i, b) = std::real(z);
            data.yimag.coeffRef(i, b) = std::imag(z);
        }
    }

    const char *phname = "global";
    switch(phmode) {
    case MapPhaseMode::AutoPerFreq:
        phname = "auto";
        break;
    case MapPhaseMode::Absolute:
        phname = "abs";
        break;
    default:
        break;
    }
    //What it took to acquire these curves; the inversion's own settings go on
    //the density map instead, and neither line holds much text.
    drawRelaxCurves(m_waveMapCurves, data, "2tau [us]",
        formatString("2tau=%.4gus x%u/bin df=%.4gkHz ph=%s", tmin,
            std::max(1u, (unsigned int)shot[ *mapEchoesPerBin()]),
            decim * res * 1e-3, phname));

    shared_ptr<XRelaxFunc> relax_fn = shot[ *relaxFunc()];
    if( !relax_fn)
        return;
    std::vector<double> tgrid = NMRRelaxMapData::makeTGrid(tmin, tmax, ntcount);
    //An echo train decays: -f + 1 turns the recovery XRelaxFunc quotes into it.
    //Unlike the T1 map, no fit feeds the kernel, so it stands on its own.
    Eigen::MatrixXd density = m_mapSolver.exec(data, tgrid, relax_fn, -1.0,
        (TikhonovRegular::TikhonovMatrix)(int)shot[ *mapTikhonovMatrix()],
        tikhonovMethodOf(mapmode), data.strongestRow());
    //The extension is a setting of the inversion, so it belongs on that graph's
    //line, next to the grid it widened -- with the window it widened away from,
    //since the solver's own T=... is the grid, not the measurement.
    XString note = m_mapSolver.status();
    if(ext > 0.0)
        note += formatString(" ext=%.2gdec meas=%.4g-%.4g", ext, tfirst, tlast);
    drawRelaxDensityMap(m_waveMap, data, tgrid, density, "T2 [us]", note);
}
