/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "nmrrelax.h"
#include "nmrrelaxfit.h"
#include "ui_nmrrelaxform.h"
#include "nmrpulse.h"
#include "pulserdriver.h"
#include "analyzer.h"
#include "graph.h"
#include "graphwidget.h"
#include "rand.h"

REGISTER_TYPE(XDriverList, NMRT1, "NMR relaxation measurement");

#include <qpushbutton.h>
#include <qcombobox.h>
#include <qcheckbox.h>
#include <knuminput.h>
#include <kiconloader.h>

#define CONVOLUTION_CACHE_SIZE (3 * 10)

#define P1DIST_LINEAR "Linear"
#define P1DIST_LOG "Log"
#define P1DIST_RECIPROCAL "Reciprocal"

#define f(x) ((shot[ *p1Dist()].to_str() == P1DIST_LINEAR) ? (1-(x)) * p1min + (x) * p1max : \
	      ((shot[ *p1Dist()].to_str() == P1DIST_LOG) ? p1min * exp((x) * log(p1max/p1min)) : \
	       1/((1-(x))/p1min + (x)/p1max)))

class XRelaxFuncPlot : public XFuncPlot {
public:
	XRelaxFuncPlot(const char *name, bool runtime, Transaction &tr, const shared_ptr<XGraph> &graph
				   , const shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > >  &item,
				   const shared_ptr<XNMRT1> &owner)
		: XFuncPlot(name, runtime, tr, graph), m_item(item), m_owner(owner)
	{}
	~XRelaxFuncPlot() {}
	virtual double func(double t) const {
		shared_ptr<XNMRT1> owner = m_owner.lock();
		if( !owner) return 0;
		Snapshot shot( *owner);
		shared_ptr<XRelaxFunc> func1 = shot[ *m_item];
		if( !func1) return 0;
		double f, df;
		double it1 = shot[ *owner].m_params[0];
		double c = shot[ *owner].m_params[1];
		double a = shot[ *owner].m_params[2];
		func1->relax( &f, &df, t, it1);
		return c * f + a;
	}
private:
	shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > > m_item;
	weak_ptr<XNMRT1> m_owner;
};

XNMRT1::XNMRT1(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
	: XSecondaryDriver(name, runtime, ref(tr_meas), meas),
	  m_relaxFuncs(create<XRelaxFuncList>("RelaxFuncs", true)),
	  m_t1inv(create<XScalarEntry>("T1inv", false,
								   dynamic_pointer_cast<XDriver>(shared_from_this()))),
	  m_t1invErr(create<XScalarEntry>("T1invErr", false,
									  dynamic_pointer_cast<XDriver>(shared_from_this()))),
	  m_pulser(create<XItemNode < XDriverList, XPulser > >(
		  "Pulser", false, ref(tr_meas), meas->drivers(), true)),
	  m_pulse1(create<XItemNode < XDriverList, XNMRPulseAnalyzer > >(
		  "NMRPulseAnalyzer1", false, ref(tr_meas), meas->drivers(), true)),
	  m_pulse2(create<XItemNode < XDriverList, XNMRPulseAnalyzer > >(
		  "NMRPulseAnalyzer2", false, ref(tr_meas), meas->drivers())),
	  m_active(create<XBoolNode>("Active", false)),
	  m_autoPhase(create<XBoolNode>("AutoPhase", false)),
	  m_mInftyFit(create<XBoolNode>("MInftyFit", false)),
	  m_absFit(create<XBoolNode>("AbsFit", false)),
	  m_p1Min(create<XDoubleNode>("P1Min", false)),
	  m_p1Max(create<XDoubleNode>("P1Max", false)),
	  m_phase(create<XDoubleNode>("Phase", false, "%.2f")),
	  m_freq(create<XDoubleNode>("Freq", false)),
	  m_autoWindow(create<XBoolNode>("AutoWindow", false)),
	  m_windowFunc(create<XComboNode>("WindowFunc", false, true)),
	  m_windowWidth(create<XComboNode>("WindowWidth", false, true)),
	  m_mode(create<XComboNode>("Mode", false, true)),
	  m_smoothSamples(create<XUIntNode>("SmoothSamples", false)),
	  m_p1Dist(create<XComboNode>("P1Dist", false, true)),
	  m_resetFit(create<XTouchableNode>("ResetFit", true)),
	  m_clearAll(create<XTouchableNode>("ClearAll", true)),
	  m_fitStatus(create<XStringNode>("FitStatus", true)),
	  m_solver(create<SpectrumSolverWrapper>("SpectrumSolver", true, shared_ptr<XComboNode>(), m_windowFunc, shared_ptr<XDoubleNode>())),
	  m_form(new FrmNMRT1(g_pFrmMain)),
	  m_statusPrinter(XStatusPrinter::create(m_form.get())),
	  m_wave(create<XWaveNGraph>("Wave", true, m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)) {

	for(Transaction tr( *this);; ++tr) {
		m_relaxFunc = create<XItemNode < XRelaxFuncList, XRelaxFunc > >(
						  tr, "RelaxFunc", false, ref(tr), m_relaxFuncs, true);
		if(tr.commit())
			break;
	}

	m_form->m_btnClear->setIcon(
    	KIconLoader::global()->loadIcon("editdelete",
																KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
    m_form->m_btnResetFit->setIcon(
    	KIconLoader::global()->loadIcon("reload",
																KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );

    m_form->setWindowTitle(i18n("NMR Relax. Meas. - ") + getLabel() );

    meas->scalarEntries()->insert(tr_meas, t1inv());
    meas->scalarEntries()->insert(tr_meas, t1invErr());

    connect(pulser());
    connect(pulse1());
    connect(pulse2());

    m_windowWidthList.push_back(0.25);
    m_windowWidthList.push_back(0.5);
//    m_windowWidthList.push_back(0.75);
    m_windowWidthList.push_back(1.0);
//    m_windowWidthList.push_back(1.25);
    m_windowWidthList.push_back(1.5);
    m_windowWidthList.push_back(2.0);
	for(Transaction tr( *this);; ++tr) {
	    tr[ *m_windowFunc] = SpectrumSolverWrapper::WINDOW_FUNC_HAMMING;
	    tr[ *m_windowWidth].add("25%");
	    tr[ *m_windowWidth].add("50%");
	//    tr[ *m_windowWidth].add("75%");
	    tr[ *m_windowWidth].add("100%");
	//    tr[ *m_windowWidth].add("125%");
	    tr[ *m_windowWidth].add("150%");
	    tr[ *m_windowWidth].add("200%");
	    tr[ *m_windowWidth] = 2;

	    const char *labels[] = {"P1 [ms] or 2Tau [us]", "Intens [V]",
								"Weight [1/V]", "Abs [V]", "Re [V]", "Im [V]"};
		tr[ *m_wave].setColCount(6, labels);
		tr[ *m_wave].insertPlot(i18n("Relaxation"), 0, 1, -1, 2);
		tr[ *m_wave].insertPlot(i18n("Out-of-Phase"), 0, 5, -1, 2);
		shared_ptr<XAxis> axisx = tr[ *m_wave].axisx();
		shared_ptr<XAxis> axisy = tr[ *m_wave].axisy();
		tr[ *axisx->logScale()] = true;
		tr[ *axisy->label()] = i18n("Relaxation");
		tr[ *tr[ *m_wave].plot(0)->drawLines()] = false;
		tr[ *tr[ *m_wave].plot(1)->drawLines()] = false;
		tr[ *tr[ *m_wave].plot(1)->intensity()] = 0.6;
		shared_ptr<XFuncPlot> plot3 = m_wave->graph()->plots()->create<XRelaxFuncPlot>(
			tr, "FittedCurve", true, ref(tr), m_wave->graph(),
			relaxFunc(), static_pointer_cast<XNMRT1>(shared_from_this()));
		tr[ *plot3->label()] = i18n("Fitted Curve");
		tr[ *plot3->axisX()] = axisx;
		tr[ *plot3->axisY()] = axisy;
		tr[ *plot3->drawPoints()] = false;
		tr[ *plot3->pointColor()] = clBlue;
		tr[ *plot3->lineColor()] = clBlue;
		tr[ *plot3->drawBars()].setUIEnabled(false);
		tr[ *plot3->barColor()].setUIEnabled(false);
		tr[ *plot3->clearPoints()].setUIEnabled(false);
		tr[ *m_wave].clearPoints();

		tr[ *mode()].add("T1 Measurement");
		tr[ *mode()].add("T2 Measurement");
		tr[ *mode()].add("St.E. Measurement");
		tr[ *mode()] = MEAS_T1;

		tr[ *p1Dist()].add(P1DIST_LINEAR);
		tr[ *p1Dist()].add(P1DIST_LOG);
		tr[ *p1Dist()].add(P1DIST_RECIPROCAL);
		tr[ *p1Dist()] = 1;

		tr[ *relaxFunc()].str(XString("NMR I=1/2"));
		tr[ *p1Min()] = 1.0;
		tr[ *p1Max()] = 100.0;
		tr[ *autoPhase()] = true;
		tr[ *autoWindow()] = true;
		tr[ *mInftyFit()] = true;
		tr[ *smoothSamples()] = 200;

		if(tr.commit())
			break;
	}

	m_form->m_numPhase->setRange( -360.0, 360.0, 10.0, true);

	m_conP1Min = xqcon_create<XQLineEditConnector>(m_p1Min, m_form->m_edP1Min);
	m_conP1Max = xqcon_create<XQLineEditConnector>(m_p1Max, m_form->m_edP1Max);
	m_conPhase = xqcon_create<XKDoubleNumInputConnector>(m_phase, m_form->m_numPhase);
	m_conFreq = xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq);
	m_conAutoWindow = xqcon_create<XQToggleButtonConnector>(m_autoWindow, m_form->m_ckbAutoWindow);
	m_conWindowFunc = xqcon_create<XQComboBoxConnector>(m_windowFunc, m_form->m_cmbWindowFunc, Snapshot( *m_windowFunc));
	m_conWindowWidth = xqcon_create<XQComboBoxConnector>(m_windowWidth, m_form->m_cmbWindowWidth, Snapshot( *m_windowWidth));
	m_conSmoothSamples = xqcon_create<XQLineEditConnector>(m_smoothSamples, m_form->m_edSmoothSamples);
	m_conP1Dist = xqcon_create<XQComboBoxConnector>(m_p1Dist, m_form->m_cmbP1Dist, Snapshot( *m_p1Dist));
	m_conClearAll = xqcon_create<XQButtonConnector>(m_clearAll, m_form->m_btnClear);
	m_conResetFit = xqcon_create<XQButtonConnector>(m_resetFit, m_form->m_btnResetFit);
	m_conActive = xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive);
	m_conAutoPhase = xqcon_create<XQToggleButtonConnector>(m_autoPhase, m_form->m_ckbAutoPhase);
	m_conMInftyFit = xqcon_create<XQToggleButtonConnector>(m_mInftyFit, m_form->m_ckbMInftyFit);
	m_conAbsFit = xqcon_create<XQToggleButtonConnector>(m_absFit, m_form->m_ckbAbsFit);
	m_conFitStatus = xqcon_create<XQTextBrowserConnector>(m_fitStatus, m_form->m_txtFitStatus);
	m_conRelaxFunc = xqcon_create<XQComboBoxConnector>(m_relaxFunc, m_form->m_cmbRelaxFunc, Snapshot( *m_relaxFuncs));
	m_conMode = xqcon_create<XQComboBoxConnector>(m_mode, m_form->m_cmbMode, Snapshot( *m_mode));

	m_conPulser = xqcon_create<XQComboBoxConnector>(m_pulser, m_form->m_cmbPulser, ref(tr_meas));
	m_conPulse1 = xqcon_create<XQComboBoxConnector>(m_pulse1, m_form->m_cmbPulse1, ref(tr_meas));
	m_conPulse2 = xqcon_create<XQComboBoxConnector>(m_pulse2, m_form->m_cmbPulse2, ref(tr_meas));

	for(Transaction tr( *this);; ++tr) {
		m_lsnOnActiveChanged = tr[ *active()].onValueChanged().connectWeakly(
			shared_from_this(), &XNMRT1::onActiveChanged);
		m_lsnOnCondChanged = tr[ *p1Max()].onValueChanged().connectWeakly(
			shared_from_this(), &XNMRT1::onCondChanged);
		tr[ *p1Min()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *phase()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *smoothSamples()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *mInftyFit()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *absFit()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *relaxFunc()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *autoPhase()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *freq()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *autoWindow()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *windowFunc()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *windowWidth()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *mode()].onValueChanged().connect(m_lsnOnCondChanged);

		m_lsnOnClearAll = tr[ *m_clearAll].onTouch().connectWeakly(
			shared_from_this(), &XNMRT1::onClearAll);
		m_lsnOnResetFit = tr[ *m_resetFit].onTouch().connectWeakly(
			shared_from_this(), &XNMRT1::onResetFit);
		if(tr.commit())
			break;
	}
}
void
XNMRT1::showForms() {
	m_form->show();
	m_form->raise();
}

void
XNMRT1::onClearAll(const Snapshot &shot, XTouchableNode *) {
    trans( *this).m_timeClearRequested = XTime::now();
    requestAnalysis();
}
void
XNMRT1::onResetFit(const Snapshot &shot, XTouchableNode *) {
	for(Transaction tr( *this);; ++tr) {
		const Snapshot &shot(tr);
		double x = randMT19937();
		double p1min = shot[ *p1Min()];
		double p1max = shot[ *p1Max()];
		if((p1min <= 0) || (p1min >= p1max)) {
	      	gErrPrint(i18n("Invalid P1Min or P1Max."));
	      	return;
		}
		tr[ *this].m_params[0] = 1.0 / f(x);
		tr[ *this].m_params[1] = 0.1;
		tr[ *this].m_params[2] = 0.0;
		if(tr.commit())
			break;
	}
	requestAnalysis();
}
void
XNMRT1::onCondChanged(const Snapshot &shot, XValueNodeBase *node) {
//    if((node == phase()) && *autoPhase()) return;
//    if(((node == windowWidth()) || (node == windowFunc())) && *autoWindow()) return;
    if(
		(node == mode().get()) ||
		(node == freq().get())) {
        trans( *this).m_timeClearRequested = XTime::now();
    }
	requestAnalysis();
}
void
XNMRT1::analyzeSpectrum(Transaction &tr,
	const std::vector< std::complex<double> >&wave, int origin, double cf,
	std::deque<std::complex<double> > &value_by_cond) {
	const Snapshot &shot_this(tr);

	value_by_cond.clear();
	std::deque<FFT::twindowfunc> funcs;
	m_solver->windowFuncs(funcs);

	int idx = 0;
	for(std::deque<double>::const_iterator wit = m_windowWidthList.begin(); wit != m_windowWidthList.end(); wit++) {
		for(std::deque<FFT::twindowfunc>::iterator fit = funcs.begin(); fit != funcs.end(); fit++) {
			if(shot_this[ *this].m_convolutionCache.size() <= idx) {
				tr[ *this].m_convolutionCache.push_back(
					shared_ptr<Payload::ConvolutionCache>(new Payload::ConvolutionCache));
			}
			shared_ptr<Payload::ConvolutionCache> cache = tr[ *this].m_convolutionCache[idx];
			if((cache->windowwidth != *wit) || (cache->origin != origin) ||
				(cache->windowfunc != *fit) || (cache->cfreq != cf) ||
				(cache->wave.size() != wave.size())) {
				cache->windowwidth = *wit;
				cache->origin = origin;
				cache->windowfunc = *fit;
				cache->cfreq = cf;
				cache->power = 0.0;
				cache->wave.resize(wave.size());
				double wk = 1.0 / FFTSolver::windowLength(wave.size(), -origin, *wit);
				for(int i = 0; i < (int)wave.size(); i++) {
					double w = ( *fit)((i - origin) * wk) / (double)wave.size();
					cache->wave[i] = std::polar(w, -2.0*M_PI*cf*(i - origin));
					cache->power += w*w;
				}
			}

			std::complex<double> z(0.0);
			for(int i = 0; i < (int)cache->wave.size(); i++) {
				z += wave[i] * cache->wave[i];
			}

//			m_solver->solver()->exec(wave, fftout, -origin, 0.0, *fit, *wit);
//			value_by_cond.push_back(fftout[(cf + fftlen) % fftlen]);
			value_by_cond.push_back(z);
			idx++;
		}
	}
}
bool
XNMRT1::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
    shared_ptr<XPulser> _pulser = shot_this[ *pulser()];
    shared_ptr<XNMRPulseAnalyzer> _pulse1 = shot_this[ *pulse1()];
    shared_ptr<XNMRPulseAnalyzer> _pulse2 = shot_this[ *pulse2()];
    if( !_pulser || !_pulse1) return false;
    if(emitter == this) return true;
    if(emitter == _pulser.get()) return false;
	ASSERT((emitter == _pulse1.get()) || (emitter == _pulse2.get()));

    const Snapshot &shot_pulse1((emitter == _pulse1.get()) ? shot_emitter : shot_others);
    const Snapshot &shot_pulse2((emitter == _pulse2.get()) ? shot_emitter : shot_others);

    if(shot_others[ *_pulser].time() > shot_pulse1[ *_pulse1].time()) return false;

	switch(shot_others[ *_pulser].combMode()) {
	default:
		if(emitter == _pulse2.get())
			return false;
		return true;
	case XPulser::N_COMB_MODE_COMB_ALT:
	case XPulser::N_COMB_MODE_P1_ALT:
		if( !_pulse2) {
			m_statusPrinter->printError(i18n("2 Pulse Analyzers needed."));
			return false;
		}
	    if(shot_pulse1[ *_pulse1].time() != shot_pulse2[ *_pulse2].time()) return false;
		return true;
	}
//    return (_pulser->time() < _pulse1->timeAwared()) && (_pulser->time() < _pulse1->time());
}

void
XNMRT1::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) throw (XRecordError&) {
	Snapshot &shot_this(tr);

	double p1min = shot_this[ *p1Min()];
	double p1max = shot_this[ *p1Max()];

	if((p1min <= 0) || (p1min >= p1max)) {
		throw XRecordError(i18n("Invalid P1Min or P1Max."), __FILE__, __LINE__);
	}

	const int samples = shot_this[ *smoothSamples()];
	if(samples <= 10) {
		throw XRecordError(i18n("Invalid # of Samples."), __FILE__, __LINE__);
	}
	if(samples >= 100000) {
		m_statusPrinter->printWarning(i18n("Too many Samples."), true);
	}

	int _mode = shot_this[ *mode()];
	shared_ptr<XNMRPulseAnalyzer> _pulse1 = shot_this[ *pulse1()];
	shared_ptr<XNMRPulseAnalyzer> _pulse2 = shot_this[ *pulse2()];
    const Snapshot &shot_pulse1((emitter == _pulse1.get()) ? shot_emitter : shot_others);
    const Snapshot &shot_pulse2((emitter == _pulse2.get()) ? shot_emitter : shot_others);

	shared_ptr<XPulser> _pulser = shot_this[ *pulser()];
	const Snapshot &shot_pulser(shot_others);
	ASSERT( _pulser );
	if(shot_pulser[ *_pulser].time()) {
		//Check consitency.
		switch (_mode) {
		case MEAS_T1:
			break;
		case MEAS_T2:
			break;
		case MEAS_ST_E:
			if((shot_pulser[ *_pulser].tau() != shot_pulser[ *_pulser].combPT()) ||
					(shot_pulser[ *_pulser].combNum() != 2) ||
					( !shot_pulser[ *_pulser->conserveStEPhase()]) ||
					(shot_pulser[ *_pulser].pw1() != 0.0) ||
					(shot_pulser[ *_pulser].pw2() != shot_pulser[ *_pulser].combPW()))
				m_statusPrinter->printWarning(i18n("Strange St.E. settings."));
			break;
		}
	}

	// read spectra from NMRPulseAnalyzers
	if( emitter != this) {
		ASSERT( _pulse1 );
		ASSERT( shot_pulse1[ *_pulse1].time() );
		ASSERT( shot_pulser[ *_pulser].time() );
		ASSERT( emitter != _pulser.get() );

		if(shot_pulse1[ *_pulse1->exAvgIncr()]) {
			m_statusPrinter->printWarning(i18n("Do NOT use incremental avg. Skipping."));
			throw XSkippedRecordError(__FILE__, __LINE__);
		}

		std::deque<std::complex<double> > cmp1, cmp2;
		double cfreq = shot_this[ *freq()] * 1e3 * shot_pulse1[ *_pulse1].interval();
		analyzeSpectrum(tr, shot_pulse1[ *_pulse1].wave(),
			shot_pulse1[ *_pulse1].waveFTPos(), cfreq, cmp1);
		Payload::RawPt pt1, pt2;
		if(_pulse2) {
			analyzeSpectrum(tr, shot_pulse2[ *_pulse2].wave(),
				shot_pulse2[ *_pulse2].waveFTPos(), cfreq, cmp2);
			pt2.value_by_cond.resize(cmp2.size());
		}
		pt1.value_by_cond.resize(cmp1.size());
		switch(shot_pulser[ *_pulser].combMode()) {
        default:
			throw XRecordError(i18n("Unknown Comb Mode!"), __FILE__, __LINE__);
        case XPulser::N_COMB_MODE_COMB_ALT:
			if(_mode != MEAS_T1) throw XRecordError(i18n("Use T1 mode!"), __FILE__, __LINE__);
			ASSERT(_pulse2);
            pt1.p1 = shot_pulser[ *_pulser].combP1();
            for(int i = 0; i < cmp1.size(); i++)
            	pt1.value_by_cond[i] = (cmp1[i] - cmp2[i]) / cmp1[i];
            tr[ *this].m_pts.push_back(pt1);
            break;
        case XPulser::N_COMB_MODE_P1_ALT:
			if(_mode == MEAS_T2)
                throw XRecordError(i18n("Do not use T2 mode!"), __FILE__, __LINE__);
			ASSERT(_pulse2);
            pt1.p1 = shot_pulser[ *_pulser].combP1();
            std::copy(cmp1.begin(), cmp1.end(), pt1.value_by_cond.begin());
            tr[ *this].m_pts.push_back(pt1);
            pt2.p1 = shot_pulser[ *_pulser].combP1Alt();
            std::copy(cmp2.begin(), cmp2.end(), pt2.value_by_cond.begin());
            tr[ *this].m_pts.push_back(pt2);
            break;
        case XPulser::N_COMB_MODE_ON:
			if(_mode != MEAS_T2) {
                pt1.p1 = shot_pulser[ *_pulser].combP1();
                std::copy(cmp1.begin(), cmp1.end(), pt1.value_by_cond.begin());
                tr[ *this].m_pts.push_back(pt1);
                break;
			}
			m_statusPrinter->printWarning(i18n("T2 mode with comb pulse!"));
        case XPulser::N_COMB_MODE_OFF:
			if(_mode != MEAS_T2) {
				m_statusPrinter->printWarning(i18n("Do not use T1 mode! Skipping."));
				throw XSkippedRecordError(__FILE__, __LINE__);
			}
			//T2 measurement
            pt1.p1 = 2.0 * shot_pulser[ *_pulser].tau();
            std::copy(cmp1.begin(), cmp1.end(), pt1.value_by_cond.begin());
            tr[ *this].m_pts.push_back(pt1);
            break;
        }
    }

	tr[ *this].m_sumpts.clear();

	if(shot_this[ *this].m_timeClearRequested > shot_pulse1[ *_pulse1].timeAwared()) {
		tr[ *this].m_pts.clear();
		tr[ *m_wave].clearPoints();
		tr[ *m_fitStatus] = "";
		trans( *_pulse1->avgClear()).touch();
		if(_pulse2)
			trans( *_pulse2->avgClear()).touch();
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

	shared_ptr<XRelaxFunc> func = shot_this[ *relaxFunc()];
	if( !func) {
		throw XRecordError(i18n("Please select relaxation func."), __FILE__, __LINE__);
	}

	tr[ *this].m_sumpts.resize(samples);
	std::deque<Payload::Pt> &sumpts(tr[ *this].m_sumpts);

	{
		Payload::Pt dummy;
		dummy.c = 0; dummy.p1 = 0; dummy.isigma = 0;
		dummy.value_by_cond.resize(shot_this[ *this].m_convolutionCache.size());
		std::fill(tr[ *this].m_sumpts.begin(), tr[ *this].m_sumpts.end(), dummy);
		double k = shot_this[ *this].m_sumpts.size() / log(p1max/p1min);
		std::deque<Payload::RawPt>::const_iterator pts_begin(shot_this[ *this].m_pts.begin());
		std::deque<Payload::RawPt>::const_iterator pts_end(shot_this[ *this].m_pts.end());
		int sum_size = (int)shot_this[ *this].m_sumpts.size();
		for(std::deque<Payload::RawPt>::const_iterator it = pts_begin; it != pts_end; it++) {
			int idx = lrint(log(it->p1 / p1min) * k);
			if((idx < 0) || (idx >= sum_size)) continue;
			double p1 = it->p1;
			//For St.E., T+tau = P1+3*tau.
			if(_mode == MEAS_ST_E)
				p1 += 3 * shot_pulser[ *_pulser].tau() * 1e-3;
			sumpts[idx].isigma += 1.0;
			sumpts[idx].p1 += p1;
			for(unsigned int i = 0; i < it->value_by_cond.size(); i++)
				sumpts[idx].value_by_cond[i] += it->value_by_cond[i];
		}
	}

	std::deque<std::complex<double> > sum_c(
		shot_this[ *this].m_convolutionCache.size()), corr(shot_this[ *this].m_convolutionCache.size());
	double sum_t = 0.0;
	int n = 0;
	for(std::deque<Payload::Pt>::const_iterator it = shot_this[ *this].m_sumpts.begin();
		it != shot_this[ *this].m_sumpts.end(); it++) {
		if(it->isigma == 0) continue;
		double t = log10(it->p1 / it->isigma);
		for(unsigned int i = 0; i < it->value_by_cond.size(); i++) {
			sum_c[i] += it->value_by_cond[i];
			corr[i] += it->value_by_cond[i] * t;
		}
		sum_t += t;
		n++;
	}
	if(n == 0)
		throw XSkippedRecordError(__FILE__, __LINE__);

	for(unsigned int i = 0; i < corr.size(); i++) {
		corr[i] -= sum_c[i]*sum_t/(double)n;
		corr[i] *= ((_mode == MEAS_T1) ? 1 : -1);
	}

	bool _absfit = shot_this[ *absFit()];

	std::deque<double> phase_by_cond(corr.size(), shot_this[ *phase()] / 180.0 * M_PI);
	int cond = -1;
	double maxsn2 = 0.0;
	for(unsigned int i = 0; i < corr.size(); i++) {
		if(shot_this[ *autoPhase()]) {
			phase_by_cond[i] = std::arg(corr[i]);
		}
		if(shot_this[ *autoWindow()]) {
			double sn2 = _absfit ? std::abs(corr[i]) : std::real(corr[i] * std::polar(1.0, -phase_by_cond[i]));
			sn2 = sn2 * sn2 / shot_this[ *this].m_convolutionCache[i]->power;
			if(maxsn2 < sn2) {
				maxsn2 = sn2;
				cond = i;
			}
		}
	}
	if(cond < 0) {
		cond = 0;
		for(std::deque<shared_ptr<Payload::ConvolutionCache> >::const_iterator it
			= shot_this[ *this].m_convolutionCache.begin();
			it != shot_this[ *this].m_convolutionCache.end(); ++it) {
			if((m_windowWidthList[std::max(0, (int)shot_this[ *windowWidth()])] == ( *it)->windowwidth) &&
				(m_solver->windowFunc(shot_this) == ( *it)->windowfunc)) {
				break;
			}
			cond++;
		}
	}
	if(cond >= (shot_this[ *this].m_convolutionCache.size())) {
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	double ph = phase_by_cond[cond];
	if(shot_this[ *autoPhase()]) {
		tr[ *phase()] = ph / M_PI * 180;
		tr.unmark(m_lsnOnCondChanged); //avoiding recursive signaling.
    }
	if(shot_this[ *autoWindow()]) {
		for(unsigned int i = 0; i < m_windowWidthList.size(); i++) {
			if(m_windowWidthList[i] == shot_this[ *this].m_convolutionCache[cond]->windowwidth)
				tr[ *windowWidth()] = i;
		}
		std::deque<FFT::twindowfunc> funcs;
		m_solver->windowFuncs(funcs);
		for(unsigned int i = 0; i < funcs.size(); i++) {
			if(funcs[i] == shot_this[ *this].m_convolutionCache[cond]->windowfunc)
				tr[ *windowFunc()] = i;
		}
		tr.unmark(m_lsnOnCondChanged); //avoiding recursive signaling.
	}
	std::complex<double> cph(std::polar(1.0, -phase_by_cond[cond]));
	for(std::deque<Payload::Pt>::iterator it = sumpts.begin(); it != sumpts.end(); it++) {
		if(it->isigma == 0) continue;
		it->p1 = it->p1 / it->isigma;
		it->c =  it->value_by_cond[cond] * cph / it->isigma;
		it->var = (_absfit) ? std::abs(it->c) : std::real(it->c);
		it->isigma = sqrt(it->isigma);
    }

	tr[ *m_fitStatus] = iterate(tr, func, 4);

	t1inv()->value(tr, 1000.0 * shot_this[ *this].m_params[0]);
	t1invErr()->value(tr, 1000.0 * shot_this[ *this].m_errors[0]);

	m_isPulserControlRequested = (emitter != this);
}

void
XNMRT1::visualize(const Snapshot &shot) {
	if( !shot[ *this].time()) {
		for(Transaction tr( *this);; ++tr) {
			tr[ *m_wave].clearPoints();
			if(tr.commit())
				break;
		}
		return;
	}

	//set new P1s
	if(shot[ *active()] && m_isPulserControlRequested.compareAndSet((int)true, (int)false)) {
		shared_ptr<XPulser> _pulser = shot[ *pulser()];
		if(_pulser) {
			for(Transaction tr( *_pulser);; ++tr) {
				double p1min = shot[ *p1Min()];
				double p1max = shot[ *p1Max()];
				double x = randMT19937();
				double np1, np2;
				np1 = f(x);
				np2 = f(1-x);
				switch(shot[ *mode()]) {
				case MEAS_T1:
				case MEAS_ST_E:
					tr[ *_pulser->combP1()] = np1;
					tr[ *_pulser->combP1Alt()] = np2;
					break;
				case MEAS_T2:
					tr[ *_pulser->tau()] = np1 / 2.0;
					break;
				}
				if(tr.commit())
					break;
			}
		}
	}

	for(Transaction tr( *m_wave);; ++tr) {
		XString label;
		switch(shot[ *mode()]) {
		case MEAS_T1:
			label = "P1 [ms]";
			break;
		case MEAS_T2:
			label = "2tau [us]";
			break;
		case MEAS_ST_E:
			label = "T+tau [ms]";
			break;
		}
		tr[ *m_wave].setLabel(0, label.c_str());
		tr[ *tr[ *m_wave].axisx()->label()] = label;
		tr[ *m_wave].setRowCount(shot[ *this].m_sumpts.size());
		double *colp1 = tr[ *m_wave].cols(0);
		double *colval = tr[ *m_wave].cols(1);
		double *colabs = tr[ *m_wave].cols(3);
		double *colre = tr[ *m_wave].cols(4);
		double *colim = tr[ *m_wave].cols(5);
		double *colisigma = tr[ *m_wave].cols(2);
		int i = 0;
		for(std::deque<Payload::Pt>::const_iterator it = shot[ *this].m_sumpts.begin();
			it != shot[ *this].m_sumpts.end(); it++) {
			if(it->isigma == 0) {
				colval[i] = 0;
				colabs[i] = 0;
				colre[i] = 0;
				colim[i] = 0;
				colp1[i] = 0;
			}
			else {
				colval[i] = it->var;
				colabs[i] = std::abs(it->c);
				colre[i] = std::real(it->c);
				colim[i] = std::imag(it->c);
				colp1[i] = it->p1;
			}
			colisigma[i] = it->isigma;
			i++;
		}
		m_wave->drawGraph(tr);
		if(tr.commit()) {
			break;
		}
	}
}

void
XNMRT1::onActiveChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
	if(shot_this[ *active()]) {
		const shared_ptr<XPulser> _pulser = shot_this[ *pulser()];
		const shared_ptr<XNMRPulseAnalyzer> _pulse1 = shot_this[ *pulse1()];
		const shared_ptr<XNMRPulseAnalyzer> _pulse2 = shot_this[ *pulse2()];

		onClearAll(shot_this, m_clearAll.get());
		if( !_pulser || !_pulse1) {
			gErrPrint(getLabel() + ": " + i18n("No pulser or No NMR Pulse Analyzer."));
			return;
		}

		Snapshot shot_pulse1( *_pulse1);
		Snapshot shot_pulser( *_pulser);
		if(_pulse2 &&
		   ((shot_pulser[ *_pulser->combMode()] == XPulser::N_COMB_MODE_COMB_ALT) ||
			(shot_pulser[ *_pulser->combMode()] == XPulser::N_COMB_MODE_P1_ALT))) {
			for(Transaction tr( *_pulse2);; ++tr) {
				tr[ *_pulse2->fromTrig()] =
					shot_pulse1[ *_pulse1->fromTrig()] + shot_pulser[ *_pulser->altSep()];
				tr[ *_pulse2->width()] = (double)shot_pulse1[ *_pulse1->width()];
				tr[ *_pulse2->phaseAdv()] = (double)shot_pulse1[ *_pulse1->phaseAdv()];
				tr[ *_pulse2->bgPos()] =
					shot_pulse1[ *_pulse1->bgPos()] + shot_pulser[ *_pulser->altSep()];
				tr[ *_pulse2->bgWidth()] = (double)shot_pulse1[ *_pulse1->bgWidth()];
				tr[ *_pulse2->fftPos()] =
					shot_pulse1[ *_pulse1->fftPos()] + shot_pulser[ *_pulser->altSep()];
				tr[ *_pulse2->fftLen()] = (int)shot_pulse1[ *_pulse1->fftLen()];
				tr[ *_pulse2->windowFunc()] = (int)shot_pulse1[ *_pulse1->windowFunc()];
				tr[ *_pulse2->usePNR()] = (bool)shot_pulse1[ *_pulse1->usePNR()];
				tr[ *_pulse2->pnrSolverList()] = (int)shot_pulse1[ *_pulse1->pnrSolverList()];
				tr[ *_pulse2->solverList()] = (int)shot_pulse1[ *_pulse1->solverList()];
				tr[ *_pulse2->numEcho()] = (int)shot_pulse1[ *_pulse1->numEcho()];
				tr[ *_pulse2->echoPeriod()] = (double)shot_pulse1[ *_pulse1->echoPeriod()];
				if(tr.commit())
					break;
			}
		}
	}
}


