/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
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
#include "nmrrelaxform.h"
#include "nmrpulse.h"
#include "pulserdriver.h"
#include "analyzer.h"
#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"
#include "rand.h"

REGISTER_TYPE(XDriverList, NMRT1, "NMR relaxation measurement");

#include <klocale.h>
#include <qpushbutton.h>
#include <qcombobox.h>
#include <qcheckbox.h>
#include <knuminput.h>
#include <kapplication.h>
#include <kiconloader.h>

#define CONVOLUTION_CACHE_SIZE (3 * 10)

#define P1DIST_LINEAR "Linear"
#define P1DIST_LOG "Log"
#define P1DIST_RECIPROCAL "Reciprocal"

#define f(x) ((p1Dist()->to_str() == P1DIST_LINEAR) ? (1-(x)) * p1min + (x) * p1max : \
	      ((p1Dist()->to_str() == P1DIST_LOG) ? p1min * exp((x) * log(p1max/p1min)) : \
	       1/((1-(x))/p1min + (x)/p1max)))
         
class XRelaxFuncPlot : public XFuncPlot
{
	XNODE_OBJECT
protected:
	XRelaxFuncPlot(const char *name, bool runtime, const shared_ptr<XGraph> &graph
				   , const shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > >  &item,
				   const shared_ptr<XNMRT1> &owner) 
		: XFuncPlot(name, runtime, graph), m_item(item), m_owner(owner)
	{}
public:
	~XRelaxFuncPlot() {}
	virtual double func(double t) const
	{
		shared_ptr<XRelaxFunc> func1 = *m_item;
		if(!func1) return 0;
		shared_ptr<XNMRT1> owner = m_owner.lock();
		if(!owner) return 0;
		double f, df;
		double it1 = owner->m_params[0];
		double c = owner->m_params[1];
		double a = owner->m_params[2];
		func1->relax(&f, &df, t, it1);
		return c * f + a;
	}
private:
	shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > > m_item;
	weak_ptr<XNMRT1> m_owner;
};
	       
XNMRT1::XNMRT1(const char *name, bool runtime,
			   const shared_ptr<XScalarEntryList> &scalarentries,
			   const shared_ptr<XInterfaceList> &interfaces,
			   const shared_ptr<XThermometerList> &thermometers,
			   const shared_ptr<XDriverList> &drivers)
	: XSecondaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	  m_relaxFuncs(create<XRelaxFuncList>("RelaxFuncs", true)),
	  m_t1inv(create<XScalarEntry>("T1inv", false,
								   dynamic_pointer_cast<XDriver>(shared_from_this()))),
	  m_t1invErr(create<XScalarEntry>("T1invErr", false, 
									  dynamic_pointer_cast<XDriver>(shared_from_this()))),
	  m_pulser(create<XItemNode < XDriverList, XPulser > >("Pulser", false, drivers, true)),
	  m_pulse1(create<XItemNode < XDriverList, XNMRPulseAnalyzer > >("NMRPulseAnalyzer1", false, drivers, true)),
	  m_pulse2(create<XItemNode < XDriverList, XNMRPulseAnalyzer > >("NMRPulseAnalyzer2", false, drivers)),
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
	  m_relaxFunc(create<XItemNode < XRelaxFuncList, XRelaxFunc > >(
					  "RelaxFunc", false, m_relaxFuncs, true)),
	  m_resetFit(create<XNode>("ResetFit", true)),
	  m_clearAll(create<XNode>("ClearAll", true)),
	  m_fitStatus(create<XStringNode>("FitStatus", true)),
	  m_solver(create<SpectrumSolverWrapper>("SpectrumSolver", true, shared_ptr<XComboNode>(), m_windowFunc, shared_ptr<XDoubleNode>())),
	  m_form(new FrmNMRT1(g_pFrmMain)),
	  m_statusPrinter(XStatusPrinter::create(m_form.get())),
	  m_wave(create<XWaveNGraph>("Wave", true, m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)) {
    m_form->m_btnClear->setIconSet(
		KApplication::kApplication()->iconLoader()->loadIconSet("editdelete", 
																KIcon::Toolbar, KIcon::SizeSmall, true ) );  
    m_form->m_btnResetFit->setIconSet(
		KApplication::kApplication()->iconLoader()->loadIconSet("reload", 
																KIcon::Toolbar, KIcon::SizeSmall, true ) );  
    
    m_form->setCaption(KAME::i18n("NMR Relax. Meas. - ") + getLabel() );
  
    scalarentries->insert(t1inv());
    scalarentries->insert(t1invErr());
    
    connect(pulser(), true);
    connect(pulse1(), false);
    connect(pulse2(), false);

    m_windowFunc->value(SpectrumSolverWrapper::WINDOW_FUNC_HAMMING);
    m_windowWidthList.push_back(0.25);
    m_windowWidthList.push_back(0.5);
//    m_windowWidthList.push_back(0.75);
    m_windowWidthList.push_back(1.0);
//    m_windowWidthList.push_back(1.25);
    m_windowWidthList.push_back(1.5);
    m_windowWidthList.push_back(2.0);
    m_windowWidth->add("25%");
    m_windowWidth->add("50%");
//    m_windowWidth->add("75%");
    m_windowWidth->add("100%");
//    m_windowWidth->add("125%");
    m_windowWidth->add("150%");
    m_windowWidth->add("200%");
    m_windowWidth->value(2);
	{
		const char *labels[] = {"P1 [ms] or 2Tau [us]", "Intens [V]",
								"Weight [1/V]", "Abs [V]", "Re [V]", "Im [V]"};
		m_wave->setColCount(6, labels);
		m_wave->insertPlot(KAME::i18n("Relaxation"), 0, 1, -1, 2);
		m_wave->insertPlot(KAME::i18n("Out-of-Phase"), 0, 5, -1, 2);
		shared_ptr<XAxis> axisx = m_wave->axisx();
		shared_ptr<XAxis> axisy = m_wave->axisy();
		axisx->logScale()->value(true);
		axisy->label()->value(KAME::i18n("Relaxation"));
		m_wave->plot(0)->drawLines()->value(false);
		m_wave->plot(1)->drawLines()->value(false);
		m_wave->plot(1)->intensity()->value(0.6);
		shared_ptr<XFuncPlot> plot3 = create<XRelaxFuncPlot>(
			"FittedCurve", true, m_wave->graph(),
			relaxFunc(), dynamic_pointer_cast<XNMRT1>(shared_from_this()));
		m_wave->graph()->plots()->insert(plot3);
		plot3->label()->value(KAME::i18n("Fitted Curve"));
		plot3->axisX()->value(axisx);
		plot3->axisY()->value(axisy);
		plot3->drawPoints()->value(false);
		plot3->pointColor()->value(clBlue);
		plot3->lineColor()->value(clBlue);
		plot3->drawBars()->setUIEnabled(false);
		plot3->barColor()->setUIEnabled(false);
		plot3->clearPoints()->setUIEnabled(false);
		m_wave->clear();
	}

	mode()->add("T1 Measurement");
	mode()->add("T2 Measurement");
	mode()->add("St.E. Measurement");
	mode()->value(MEAS_T1);
	
	p1Dist()->add(P1DIST_LINEAR);
	p1Dist()->add(P1DIST_LOG);
	p1Dist()->add(P1DIST_RECIPROCAL);
	p1Dist()->value(1);

	try {
		relaxFunc()->str(std::string("NMR I=1/2"));
	}
	catch (XKameError &e) {
		e.print();
	}
	p1Min()->value(1.0);
	p1Max()->value(100.0);
	autoPhase()->value(true);
	autoWindow()->value(true);
	mInftyFit()->value(true);
	smoothSamples()->value(200);

	m_conP1Min = xqcon_create<XQLineEditConnector>(m_p1Min, m_form->m_edP1Min);
	m_conP1Max = xqcon_create<XQLineEditConnector>(m_p1Max, m_form->m_edP1Max);
	m_conPhase = xqcon_create<XKDoubleNumInputConnector>(m_phase, m_form->m_numPhase);
	m_conFreq = xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq);
	m_conAutoWindow = xqcon_create<XQToggleButtonConnector>(m_autoWindow, m_form->m_ckbAutoWindow);
	m_conWindowFunc = xqcon_create<XQComboBoxConnector>(m_windowFunc, m_form->m_cmbWindowFunc);
	m_conWindowWidth = xqcon_create<XQComboBoxConnector>(m_windowWidth, m_form->m_cmbWindowWidth);
	m_conSmoothSamples = xqcon_create<XQLineEditConnector>(m_smoothSamples, m_form->m_edSmoothSamples);
	m_conP1Dist = xqcon_create<XQComboBoxConnector>(m_p1Dist, m_form->m_cmbP1Dist);
	m_conClearAll = xqcon_create<XQButtonConnector>(m_clearAll, m_form->m_btnClear);
	m_conResetFit = xqcon_create<XQButtonConnector>(m_resetFit, m_form->m_btnResetFit);
	m_conActive = xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive);
	m_conAutoPhase = xqcon_create<XQToggleButtonConnector>(m_autoPhase, m_form->m_ckbAutoPhase);
	m_conMInftyFit = xqcon_create<XQToggleButtonConnector>(m_mInftyFit, m_form->m_ckbMInftyFit);
	m_conAbsFit = xqcon_create<XQToggleButtonConnector>(m_absFit, m_form->m_ckbAbsFit);
	m_conFitStatus = xqcon_create<XQTextBrowserConnector>(m_fitStatus, m_form->m_txtFitStatus);
	m_conRelaxFunc = xqcon_create<XQComboBoxConnector>(m_relaxFunc, m_form->m_cmbRelaxFunc);
	m_conMode = xqcon_create<XQComboBoxConnector>(m_mode, m_form->m_cmbMode);

	m_conPulser = xqcon_create<XQComboBoxConnector>(m_pulser, m_form->m_cmbPulser);
	m_conPulse1 = xqcon_create<XQComboBoxConnector>(m_pulse1, m_form->m_cmbPulse1);
	m_conPulse2 = xqcon_create<XQComboBoxConnector>(m_pulse2, m_form->m_cmbPulse2);
        
	m_lsnOnActiveChanged = active()->onValueChanged().connectWeak(
		shared_from_this(), &XNMRT1::onActiveChanged);
	m_lsnOnCondChanged = p1Max()->onValueChanged().connectWeak(
		shared_from_this(), &XNMRT1::onCondChanged);
	p1Min()->onValueChanged().connect(m_lsnOnCondChanged);
	phase()->onValueChanged().connect(m_lsnOnCondChanged);
	smoothSamples()->onValueChanged().connect(m_lsnOnCondChanged);
	mInftyFit()->onValueChanged().connect(m_lsnOnCondChanged);
	absFit()->onValueChanged().connect(m_lsnOnCondChanged);
	relaxFunc()->onValueChanged().connect(m_lsnOnCondChanged);
	autoPhase()->onValueChanged().connect(m_lsnOnCondChanged);
	freq()->onValueChanged().connect(m_lsnOnCondChanged);
	autoWindow()->onValueChanged().connect(m_lsnOnCondChanged);
	windowFunc()->onValueChanged().connect(m_lsnOnCondChanged);
	windowWidth()->onValueChanged().connect(m_lsnOnCondChanged);
	mode()->onValueChanged().connect(m_lsnOnCondChanged);
  
	m_lsnOnClearAll = m_clearAll->onTouch().connectWeak(
		shared_from_this(), &XNMRT1::onClearAll);
	m_lsnOnResetFit = m_resetFit->onTouch().connectWeak(
		shared_from_this(), &XNMRT1::onResetFit);
}
void
XNMRT1::showForms()
{
	m_form->show();
	m_form->raise();
}

void
XNMRT1::onClearAll(const shared_ptr<XNode> &)
{
    m_timeClearRequested = XTime::now();
    requestAnalysis();
}
void
XNMRT1::onResetFit(const shared_ptr<XNode> &)
{
	const double x = randMT19937();
	const double p1min = *p1Min();
	const double p1max = *p1Max();
	if((p1min <= 0) || (p1min >= p1max)) {
      	gErrPrint(KAME::i18n("Invalid P1Min or P1Max."));  
      	return;
	}
	m_params[0] = 1.0 / f(x);
	m_params[1] = 0.1;
	m_params[2] = 0.0;
	requestAnalysis();
}
void
XNMRT1::onCondChanged(const shared_ptr<XValueNodeBase> &node)
{
    if((node == phase()) && *autoPhase()) return;
    if(((node == windowWidth()) || (node == windowFunc())) && *autoWindow()) return;
    if(
		(node == mode()) ||
		(node == freq())) {
        m_timeClearRequested = XTime::now();
    }
	requestAnalysis();
}
void
XNMRT1::analyzeSpectrum(
	const std::vector< std::complex<double> >&wave, int origin, double cf,
	std::deque<std::complex<double> > &value_by_cond) {
	value_by_cond.clear();
	std::deque<FFT::twindowfunc> funcs;
	m_solver->windowFuncs(funcs);

	int idx = 0;
	for(std::deque<double>::iterator wit = m_windowWidthList.begin(); wit != m_windowWidthList.end(); wit++) {
		for(std::deque<FFT::twindowfunc>::iterator fit = funcs.begin(); fit != funcs.end(); fit++) {
			if(m_convolutionCache.size() <= idx) {
				m_convolutionCache.push_back(shared_ptr<ConvolutionCache>(new ConvolutionCache));
			}
			shared_ptr<ConvolutionCache> cache = m_convolutionCache[idx];
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
					double w = (*fit)((i - origin) * wk) / (double)wave.size();
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
XNMRT1::checkDependency(const shared_ptr<XDriver> &emitter) const {
    shared_ptr<XPulser> _pulser = *pulser();
    shared_ptr<XNMRPulseAnalyzer> _pulse1 = *pulse1();
    shared_ptr<XNMRPulseAnalyzer> _pulse2 = *pulse2();
    if(!_pulser || !_pulse1) return false;
    if(emitter == shared_from_this()) return true;
    if(emitter == _pulser) return false;
    if(_pulser->time() > _pulse1->time()) return false;
    
//	if (_pulser->time() > _pulse1->time())
//		return false;
		
	switch(_pulser->combModeRecorded()) {
	default:
		return true;
	case XPulser::N_COMB_MODE_COMB_ALT:
	case XPulser::N_COMB_MODE_P1_ALT:
		if(!_pulse2) {
			m_statusPrinter->printError(KAME::i18n("2 Pulse Analyzers needed."));
			return false;
		}
		if(_pulse1->time() != _pulse2->time()) return false;
		return true;
	}    
//    return (_pulser->time() < _pulse1->timeAwared()) && (_pulser->time() < _pulse1->time());
}

void
XNMRT1::analyze(const shared_ptr<XDriver> &emitter) throw (XRecordError&)
{
	const double p1min = *p1Min();
	const double p1max = *p1Max();
    
	if((p1min <= 0) || (p1min >= p1max)) {
		throw XRecordError(KAME::i18n("Invalid P1Min or P1Max."), __FILE__, __LINE__);  
	}

	const int samples = *smoothSamples();
	if(samples <= 10) {
		throw XRecordError(KAME::i18n("Invalid # of Samples."), __FILE__, __LINE__);  
	}
	if(samples >= 100000) {
		m_statusPrinter->printWarning(KAME::i18n("Too many Samples."), true);
	}

	int _mode = *mode();
	const shared_ptr<XNMRPulseAnalyzer> _pulse1 = *pulse1();
    
	shared_ptr<XPulser> _pulser = *pulser();
	ASSERT( _pulser );
	if(_pulser->time()) {
		//Check consitency.
		switch (_mode) {
		case MEAS_T1:
			break;
		case MEAS_T2:
			break;
		case MEAS_ST_E:
			if((_pulser->tauRecorded() != _pulser->combPTRecorded()) ||
					(_pulser->combNumRecorded() != 2) ||
					(!*_pulser->conserveStEPhase()) ||
					(_pulser->pw1Recorded() != 0.0) ||
					(_pulser->pw2Recorded() != _pulser->combPWRecorded()))
				m_statusPrinter->printWarning(KAME::i18n("Strange St.E. settings."));
			break;
		}
	}
	
	// read spectra from NMRPulseAnalyzers
	if( emitter != shared_from_this() ) {
		ASSERT( _pulse1 );
		ASSERT( _pulse1->time() );
		shared_ptr<XNMRPulseAnalyzer> _pulse2 = *pulse2();
		ASSERT( _pulser->time() );
		ASSERT( emitter != _pulser );
        
		bool _active = *active();
      
		std::deque<std::complex<double> > cmp1, cmp2;
		double cfreq = *freq() * 1e3 * _pulse1->interval();
		analyzeSpectrum(_pulse1->wave(), _pulse1->waveFTPos(), cfreq, cmp1);
		RawPt pt1, pt2;
		if(_pulse2) {
			analyzeSpectrum(_pulse2->wave(), _pulse2->waveFTPos(), cfreq, cmp2);
			pt2.value_by_cond.resize(cmp2.size());
		}
		pt1.value_by_cond.resize(cmp1.size());
		switch(_pulser->combModeRecorded()) {
        default:
			throw XRecordError(KAME::i18n("Unknown Comb Mode!"), __FILE__, __LINE__);
        case XPulser::N_COMB_MODE_COMB_ALT:
			if(_mode != MEAS_T1) throw XRecordError(KAME::i18n("Use T1 mode!"), __FILE__, __LINE__);
			ASSERT(_pulse2);
            pt1.p1 = _pulser->combP1Recorded();
            for(int i = 0; i < cmp1.size(); i++)
            	pt1.value_by_cond[i] = (cmp1[i] - cmp2[i]) / cmp1[i];
            m_pts.push_back(pt1);
            break;
        case XPulser::N_COMB_MODE_P1_ALT:
			if(_mode == MEAS_T2) 
                throw XRecordError(KAME::i18n("Do not use T2 mode!"), __FILE__, __LINE__);
			ASSERT(_pulse2);
            pt1.p1 = _pulser->combP1Recorded();
            std::copy(cmp1.begin(), cmp1.end(), pt1.value_by_cond.begin());
            m_pts.push_back(pt1);
            pt2.p1 = _pulser->combP1AltRecorded();
            std::copy(cmp2.begin(), cmp2.end(), pt2.value_by_cond.begin());
            m_pts.push_back(pt2);
            break;
        case XPulser::N_COMB_MODE_ON:
			if(_mode != MEAS_T2) {
                pt1.p1 = _pulser->combP1Recorded();
                std::copy(cmp1.begin(), cmp1.end(), pt1.value_by_cond.begin());
                m_pts.push_back(pt1);
                break;
			}
			m_statusPrinter->printWarning(KAME::i18n("T2 mode with comb pulse!"));
        case XPulser::N_COMB_MODE_OFF:
			if(_mode != MEAS_T2) {
				m_statusPrinter->printWarning(KAME::i18n("Do not use T1 mode! Skipping."));
				throw XSkippedRecordError(__FILE__, __LINE__);
			}
			//T2 measurement
            pt1.p1 = 2.0 * _pulser->tauRecorded();
            std::copy(cmp1.begin(), cmp1.end(), pt1.value_by_cond.begin());
            m_pts.push_back(pt1);
            break;
        }

		//set new P1s
		if(_active) {
			unlockConnection(_pulser);
			double x = randMT19937();
			double np1, np2;
			np1 = f(x);
			np2 = f(1-x);
			_pulser->output()->value(false);
			switch (_mode) {
			case MEAS_T1:
			case MEAS_ST_E:
				_pulser->combP1()->value(np1);
				_pulser->combP1Alt()->value(np2);
				break;
			case MEAS_T2:
				_pulser->tau()->value(np1 / 2.0);
				break;
			}
			_pulser->output()->value(true);
        }
    }
  
	m_sumpts.clear();
  
	if(m_timeClearRequested > _pulse1->timeAwared()) {
		m_pts.clear();
		m_wave->clear();
		m_fitStatus->value("");
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
  
	shared_ptr<XRelaxFunc> func = *relaxFunc();
	if(!func) {
		throw XRecordError(KAME::i18n("Please select relaxation func."), __FILE__, __LINE__);  
	}
  
	m_sumpts.resize(samples);

	Pt dummy;
	dummy.c = 0; dummy.p1 = 0; dummy.isigma = 0;
	dummy.value_by_cond.resize(m_convolutionCache.size());
	std::fill(m_sumpts.begin(), m_sumpts.end(), dummy);
	double k = m_sumpts.size() / log(p1max/p1min);
	for(std::deque<RawPt>::iterator it = m_pts.begin(); it != m_pts.end(); it++) {
		int idx = lrint(log(it->p1 / p1min) * k);
		if((idx < 0) || (idx >= (int)m_sumpts.size())) continue;
		double p1 = it->p1;
		//For St.E., T+tau = P1+3*tau. 
		if(_mode == MEAS_ST_E)
			p1 += 3 * _pulser->tauRecorded() * 1e-3;
		m_sumpts[idx].isigma += 1.0;
		m_sumpts[idx].p1 += p1;
		for(unsigned int i = 0; i < it->value_by_cond.size(); i++)
			m_sumpts[idx].value_by_cond[i] += it->value_by_cond[i];
    }

	std::deque<std::complex<double> > sum_c(m_convolutionCache.size()), corr(m_convolutionCache.size());
	double sum_t = 0.0;
	int n = 0;
	for(std::deque<Pt>::iterator it = m_sumpts.begin(); it != m_sumpts.end(); it++) {
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
	
	bool _absfit = *absFit();
	
	std::deque<double> phase_by_cond(corr.size(), *phase() / 180.0 * M_PI);
	int cond = -1;
	double maxsn2 = 0.0;
	for(unsigned int i = 0; i < corr.size(); i++) {
		if(*autoPhase()) {
			phase_by_cond[i] = std::arg(corr[i]);
		}
		if(*autoWindow()) {
			double sn2 = _absfit ? std::abs(corr[i]) : std::real(corr[i] * std::polar(1.0, -phase_by_cond[i]));
			sn2 = sn2 * sn2 / m_convolutionCache[i]->power;
			if(maxsn2 < sn2) {
				maxsn2 = sn2;
				cond = i;
			}
		}
	}
	if(cond < 0) {
		cond = 0;
		for(std::deque<shared_ptr<ConvolutionCache> >::iterator it = m_convolutionCache.begin();
			it != m_convolutionCache.end(); it++) {
			if((m_windowWidthList[std::max(0, (int)*windowWidth())] == (*it)->windowwidth) &&
				(m_solver->windowFunc() == (*it)->windowfunc)) {
				break;
			}
			cond++;
		}
	}
	if(cond >= (m_convolutionCache.size())) {
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	double ph = phase_by_cond[cond];
	if(*autoPhase()) {
		phase()->value(ph / M_PI * 180);
    }
	if(*autoWindow()) {
		for(unsigned int i = 0; i < m_windowWidthList.size(); i++) {
			if(m_windowWidthList[i] == m_convolutionCache[cond]->windowwidth)
				windowWidth()->value(i);
		}
		std::deque<FFT::twindowfunc> funcs;
		m_solver->windowFuncs(funcs);
		for(unsigned int i = 0; i < funcs.size(); i++) {
			if(funcs[i] == m_convolutionCache[cond]->windowfunc)
				windowFunc()->value(i);
		}
	}
	std::complex<double> cph(std::polar(1.0, -phase_by_cond[cond]));
	for(std::deque<Pt>::iterator it = m_sumpts.begin(); it != m_sumpts.end(); it++) {
		if(it->isigma == 0) continue;
		it->p1 = it->p1 / it->isigma;
		it->c =  it->value_by_cond[cond] * cph / it->isigma;
		it->var = (_absfit) ? std::abs(it->c) : std::real(it->c);
		it->isigma = sqrt(it->isigma);
    }

	m_fitStatus->value(iterate(func, 4));

	t1inv()->value(1000.0 * m_params[0]);
	t1invErr()->value(1000.0 * m_errors[0]);
}

void
XNMRT1::visualize()
{
	if(!time()) {
		m_wave->clear();
		return;
	}

	{   XScopedWriteLock<XWaveNGraph> lock(*m_wave);
	std::string label;
	switch (*mode()) {
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
	m_wave->setLabel(0, label.c_str());
	m_wave->axisx()->label()->value(label);
	m_wave->setRowCount(m_sumpts.size());
	double *colp1 = m_wave->cols(0);
	double *colval = m_wave->cols(1);
	double *colabs = m_wave->cols(3);
	double *colre = m_wave->cols(4);
	double *colim = m_wave->cols(5);
	double *colisigma = m_wave->cols(2);
	int i = 0;
	for(std::deque<Pt>::iterator it = m_sumpts.begin(); it != m_sumpts.end(); it++) {
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
	}
}
  
void
XNMRT1::onActiveChanged(const shared_ptr<XValueNodeBase> &)
{
	if(*active() == true)
	{
		const shared_ptr<XPulser> _pulser = *pulser();
		const shared_ptr<XNMRPulseAnalyzer> _pulse1 = *pulse1();
		const shared_ptr<XNMRPulseAnalyzer> _pulse2 = *pulse2();

		onClearAll(shared_from_this());
		if(!_pulser || !_pulse1) {
			gErrPrint(getLabel() + ": " + KAME::i18n("No pulser or No NMR Pulse Analyzer."));  
			return;
		}
      
		if(_pulse2 && 
		   ((*_pulser->combMode() == XPulser::N_COMB_MODE_COMB_ALT) ||
			(*_pulser->combMode() == XPulser::N_COMB_MODE_P1_ALT))) {
			_pulse2->fromTrig()->value(
				*_pulse1->fromTrig() + *_pulser->altSep());
			_pulse2->width()->value(*_pulse1->width());
			_pulse2->phaseAdv()->value(*_pulse1->phaseAdv());
			_pulse2->bgPos()->value(
				*_pulse1->bgPos() + *_pulser->altSep());
			_pulse2->bgWidth()->value(*_pulse1->bgWidth());
			_pulse2->fftPos()->value(
				*_pulse1->fftPos() + *_pulser->altSep());
			_pulse2->fftLen()->value(*_pulse1->fftLen());
			_pulse2->windowFunc()->value(*_pulse1->windowFunc());
			_pulse2->useDNR()->value(*_pulse1->useDNR());
			_pulse2->solverList()->value(*_pulse1->solverList());
			_pulse2->numEcho()->value(*_pulse1->numEcho());
			_pulse2->echoPeriod()->value(*_pulse1->echoPeriod());
		}
	}
}


