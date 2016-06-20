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
#include "nmrspectrumsolver.h"
#include "ar.h"
#ifdef USE_FREQ_ESTM
    #include "freqest.h"
#endif
#include "freqestleastsquare.h"

const char SpectrumSolverWrapper::SPECTRUM_SOLVER_ZF_FFT[] = "ZF-FFT";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_STRICT[] = "Strict MEM";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_STRICT_BURG[] = "Burg+Strict MEM";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_BURG_AICc[] = "Burg's MEM AICc";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_BURG_MDL[] = "Burg's MEM MDL";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_AR_YW_AICc[] = "Yule-Walker AR AICc";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_AR_YW_MDL[] = "Yule-Walker AR MDL";
#ifdef USE_FREQ_ESTM
    const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_STRICT_EV[] = "EV+Strict MEM";
    const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MUSIC_AIC[] = "MUSIC AIC";
    const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MUSIC_MDL[] = "MUSIC MDL";
    const char SpectrumSolverWrapper::SPECTRUM_SOLVER_EV_AIC[] = "Eigenvector AIC";
    const char SpectrumSolverWrapper::SPECTRUM_SOLVER_EV_MDL[] = "Eigenvector MDL";
    const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MVDL[] = "Capon's MVDL(MLM)";
#endif
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_LS_HQ[] = "LeastSquare HQ";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_LS_AICc[] = "LeastSquare AICc";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_LS_MDL[] = "LeastSquare MDL";

const char SpectrumSolverWrapper::WINDOW_FUNC_DEFAULT[] = "Rect";
const char SpectrumSolverWrapper::WINDOW_FUNC_HANNING[] = "Hanning";
const char SpectrumSolverWrapper::WINDOW_FUNC_HAMMING[] = "Hamming";
const char SpectrumSolverWrapper::WINDOW_FUNC_FLATTOP[] = "Flat-Top";
const char SpectrumSolverWrapper::WINDOW_FUNC_BLACKMAN[] = "Blackman";
const char SpectrumSolverWrapper::WINDOW_FUNC_BLACKMAN_HARRIS[] = "Blackman-Harris";
const char SpectrumSolverWrapper::WINDOW_FUNC_KAISER_1[] = "Kaiser a=3";
const char SpectrumSolverWrapper::WINDOW_FUNC_KAISER_2[] = "Kaiser a=7.2";
const char SpectrumSolverWrapper::WINDOW_FUNC_KAISER_3[] = "Kaiser a=15";

SpectrumSolverWrapper::SpectrumSolverWrapper(const char *name, bool runtime,
	const shared_ptr<XComboNode> selector, const shared_ptr<XComboNode> windowfunc,
	const shared_ptr<XDoubleNode> windowlength, bool leastsquareonly)
	: XNode(name, runtime), m_selector(selector), m_windowfunc(windowfunc), m_windowlength(windowlength) {
	if(windowfunc) {
		for(Transaction tr( *windowfunc);; ++tr) {
			tr[ *windowfunc].add(WINDOW_FUNC_DEFAULT);
			tr[ *windowfunc].add(WINDOW_FUNC_HANNING);
			tr[ *windowfunc].add(WINDOW_FUNC_HAMMING);
			tr[ *windowfunc].add(WINDOW_FUNC_BLACKMAN);
			tr[ *windowfunc].add(WINDOW_FUNC_BLACKMAN_HARRIS);
			tr[ *windowfunc].add(WINDOW_FUNC_FLATTOP);
			tr[ *windowfunc].add(WINDOW_FUNC_KAISER_1);
			tr[ *windowfunc].add(WINDOW_FUNC_KAISER_2);
			tr[ *windowfunc].add(WINDOW_FUNC_KAISER_3);
			if(tr.commit())
				break;
		}
	}
	if(selector) {
		for(Transaction tr( *selector);; ++tr) {
			if( !leastsquareonly) {
				tr[ *selector].add(SPECTRUM_SOLVER_ZF_FFT);
				tr[ *selector].add(SPECTRUM_SOLVER_MEM_STRICT);
		//		tr[ *selector].add(SPECTRUM_SOLVER_MEM_STRICT_EV);
#ifdef USE_FREQ_ESTM
                tr[ *selector].add(SPECTRUM_SOLVER_MVDL);
				tr[ *selector].add(SPECTRUM_SOLVER_EV_MDL);
				tr[ *selector].add(SPECTRUM_SOLVER_MUSIC_MDL);
#endif
				tr[ *selector].add(SPECTRUM_SOLVER_MEM_BURG_AICc);
				tr[ *selector].add(SPECTRUM_SOLVER_MEM_BURG_MDL);
				tr[ *selector].add(SPECTRUM_SOLVER_AR_YW_AICc);
				tr[ *selector].add(SPECTRUM_SOLVER_AR_YW_MDL);
		//		tr[ *selector].add(SPECTRUM_SOLVER_MEM_STRICT_BURG);
			}
			tr[ *selector].add(SPECTRUM_SOLVER_LS_HQ);
			tr[ *selector].add(SPECTRUM_SOLVER_LS_AICc);
			tr[ *selector].add(SPECTRUM_SOLVER_LS_MDL);

			m_lsnOnChanged = tr[ *selector].onValueChanged().connectWeakly(
				shared_from_this(), &SpectrumSolverWrapper::onSolverChanged);
			if(tr.commit()) {
				onSolverChanged(tr, selector.get());
				break;
			}
		}
	}
}
SpectrumSolverWrapper::~SpectrumSolverWrapper() {
	if(m_windowfunc) {
		trans( *m_windowfunc).clear();
	}
	if(m_selector) {
		trans( *m_selector).clear();
	}
}
FFT::twindowfunc
SpectrumSolverWrapper::windowFunc(const Snapshot &shot) const {
	FFT::twindowfunc func = &FFT::windowFuncRect;
	if(shot[ *m_windowfunc]) {
		if(shot[ *m_windowfunc].to_str() == WINDOW_FUNC_HANNING) func = &FFT::windowFuncHanning;
		if(shot[ *m_windowfunc].to_str() == WINDOW_FUNC_HAMMING) func = &FFT::windowFuncHamming;
		if(shot[ *m_windowfunc].to_str() == WINDOW_FUNC_FLATTOP) func = &FFT::windowFuncFlatTop;
		if(shot[ *m_windowfunc].to_str() == WINDOW_FUNC_BLACKMAN) func = &FFT::windowFuncBlackman;
		if(shot[ *m_windowfunc].to_str() == WINDOW_FUNC_BLACKMAN_HARRIS) func = &FFT::windowFuncBlackmanHarris;
		if(shot[ *m_windowfunc].to_str() == WINDOW_FUNC_KAISER_1) func = &FFT::windowFuncKaiser1;
		if(shot[ *m_windowfunc].to_str() == WINDOW_FUNC_KAISER_2) func = &FFT::windowFuncKaiser2;
		if(shot[ *m_windowfunc].to_str() == WINDOW_FUNC_KAISER_3) func = &FFT::windowFuncKaiser3;
	}
	return func;
}
void
SpectrumSolverWrapper::windowFuncs(std::deque<FFT::twindowfunc> &funcs) const {
	funcs.clear();
	funcs.push_back(&FFT::windowFuncRect);
	funcs.push_back(&FFT::windowFuncHanning);
	funcs.push_back(&FFT::windowFuncHamming);
	funcs.push_back(&FFT::windowFuncFlatTop);
	funcs.push_back(&FFT::windowFuncBlackman);
	funcs.push_back(&FFT::windowFuncBlackmanHarris);
	funcs.push_back(&FFT::windowFuncKaiser1);
	funcs.push_back(&FFT::windowFuncKaiser2);
	funcs.push_back(&FFT::windowFuncKaiser3);
}

void
SpectrumSolverWrapper::onSolverChanged(const Snapshot &shot, XValueNodeBase *) {
    shared_ptr<Payload::WrapperBase> wrapper;
	bool has_window = true;
	bool has_length = true;
	if(m_selector) {
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_MEM_BURG_AICc) {
			wrapper.reset(new Payload::Wrapper<MEMBurg>(new MEMBurg( &SpectrumSolver::icAICc)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_MEM_BURG_MDL) {
			wrapper.reset(new Payload::Wrapper<MEMBurg>(new MEMBurg( &SpectrumSolver::icMDL)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_AR_YW_AICc) {
			wrapper.reset(new Payload::Wrapper<YuleWalkerAR>(new YuleWalkerAR( &SpectrumSolver::icAICc)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_AR_YW_MDL) {
			wrapper.reset(new Payload::Wrapper<YuleWalkerAR>(new YuleWalkerAR( &SpectrumSolver::icMDL)));
		}
#ifdef USE_FREQ_ESTM
        if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_MUSIC_AIC) {
			wrapper.reset(new Payload::Wrapper<MUSIC>(new MUSIC( &SpectrumSolver::icAIC)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_MUSIC_MDL) {
			wrapper.reset(new Payload::Wrapper<MUSIC>(new MUSIC( &SpectrumSolver::icMDL)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_EV_AIC) {
			wrapper.reset(new Payload::Wrapper<EigenVectorMethod>(new EigenVectorMethod( &SpectrumSolver::icAIC)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_EV_MDL) {
			wrapper.reset(new Payload::Wrapper<EigenVectorMethod>(new EigenVectorMethod( &SpectrumSolver::icMDL)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_MVDL) {
			wrapper.reset(new Payload::Wrapper<MVDL>(new MVDL));
		}
        if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_MEM_STRICT_EV) {
            wrapper.reset(new Payload::Wrapper<CompositeSpectrumSolver<MEMStrict, EigenVectorMethod> >(
                new CompositeSpectrumSolver<MEMStrict, EigenVectorMethod>()));
        }
#endif
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_MEM_STRICT) {
			wrapper.reset(new Payload::Wrapper<MEMStrict>(new MEMStrict));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_MEM_STRICT_BURG) {
			wrapper.reset(new Payload::Wrapper<CompositeSpectrumSolver<MEMStrict, MEMBurg> >(
				new CompositeSpectrumSolver<MEMStrict, MEMBurg>()));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_LS_HQ) {
			wrapper.reset(new Payload::Wrapper<FreqEstLeastSquare>(new FreqEstLeastSquare( &SpectrumSolver::icHQ)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_LS_AICc) {
			wrapper.reset(new Payload::Wrapper<FreqEstLeastSquare>(new FreqEstLeastSquare( &SpectrumSolver::icAICc)));
		}
		if(shot[ *m_selector].to_str() == SPECTRUM_SOLVER_LS_MDL) {
			wrapper.reset(new Payload::Wrapper<FreqEstLeastSquare>(new FreqEstLeastSquare( &SpectrumSolver::icMDL)));
		}
	}
	if( !wrapper) {
		wrapper.reset(new Payload::Wrapper<FFTSolver>(new FFTSolver));
	}
	if(m_windowfunc)
		m_windowfunc->setUIEnabled(has_window);
	if(m_windowlength)
		m_windowlength->setUIEnabled(has_length);
    iterate_commit([=](Transaction &tr){
        tr[ *this].m_wrapper = wrapper;
    });
}
