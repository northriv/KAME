/***************************************************************************
 Copyright (C) 2002-2008 Kentaro Kitagawa
 kitag@issp.u-tokyo.ac.jp

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
#include "freqest.h"
#include "freqestleastsquare.h"

const char SpectrumSolverWrapper::SPECTRUM_SOLVER_ZF_FFT[] = "ZF-FFT";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_STRICT[] = "Strict MEM";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_STRICT_EV[] = "EV+Strict MEM";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_STRICT_BURG[] = "Burg+Strict MEM";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_BURG_AICc[] = "Burg's MEM AICc";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_BURG_MDL[] = "Burg's MEM MDL";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_AR_YW_AICc[] = "Yule-Walker AR AICc";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_AR_YW_MDL[] = "Yule-Walker AR MDL";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MUSIC_AIC[] = "MUSIC AIC";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MUSIC_MDL[] = "MUSIC MDL";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_EV_AIC[] = "Eigenvector AIC";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_EV_MDL[] = "Eigenvector MDL";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MVDL[] = "Capon's MVDL(MLM)";
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
	const shared_ptr<XDoubleNode> windowlength)
	: XNode(name, runtime), m_selector(selector), m_windowfunc(windowfunc), m_windowlength(windowlength) {
	if(windowfunc) {
		windowfunc->add(WINDOW_FUNC_DEFAULT);
		windowfunc->add(WINDOW_FUNC_HANNING);
		windowfunc->add(WINDOW_FUNC_HAMMING);
		windowfunc->add(WINDOW_FUNC_BLACKMAN);
		windowfunc->add(WINDOW_FUNC_BLACKMAN_HARRIS);
		windowfunc->add(WINDOW_FUNC_FLATTOP);
		windowfunc->add(WINDOW_FUNC_KAISER_1);
		windowfunc->add(WINDOW_FUNC_KAISER_2);
		windowfunc->add(WINDOW_FUNC_KAISER_3);
	}
	if(selector) {
		selector->add(SPECTRUM_SOLVER_ZF_FFT);
		selector->add(SPECTRUM_SOLVER_MEM_STRICT);
//		selector->add(SPECTRUM_SOLVER_MEM_STRICT_EV);
		selector->add(SPECTRUM_SOLVER_LS_AICc);
		selector->add(SPECTRUM_SOLVER_LS_MDL);
		selector->add(SPECTRUM_SOLVER_MVDL);
		selector->add(SPECTRUM_SOLVER_EV_MDL);
		selector->add(SPECTRUM_SOLVER_MUSIC_MDL);
		selector->add(SPECTRUM_SOLVER_MEM_BURG_AICc);
		selector->add(SPECTRUM_SOLVER_MEM_BURG_MDL);
		selector->add(SPECTRUM_SOLVER_AR_YW_AICc);
		selector->add(SPECTRUM_SOLVER_AR_YW_MDL);
//		selector->add(SPECTRUM_SOLVER_MEM_STRICT_BURG);
		m_lsnOnChanged = selector->onValueChanged().connectWeak(shared_from_this(), &SpectrumSolverWrapper::onSolverChanged);
	}
	onSolverChanged(selector);
}
SpectrumSolverWrapper::~SpectrumSolverWrapper() {
	if(m_windowfunc) {
		m_windowfunc->clear();
	}
	if(m_selector) {
		m_selector->clear();
	}
}
FFT::twindowfunc
SpectrumSolverWrapper::windowFunc() const {
	FFT::twindowfunc func = &FFT::windowFuncRect;
	if(m_windowfunc) {
		if(m_windowfunc->to_str() == WINDOW_FUNC_HANNING) func = &FFT::windowFuncHanning;
		if(m_windowfunc->to_str() == WINDOW_FUNC_HAMMING) func = &FFT::windowFuncHamming;
		if(m_windowfunc->to_str() == WINDOW_FUNC_FLATTOP) func = &FFT::windowFuncFlatTop;
		if(m_windowfunc->to_str() == WINDOW_FUNC_BLACKMAN) func = &FFT::windowFuncBlackman;
		if(m_windowfunc->to_str() == WINDOW_FUNC_BLACKMAN_HARRIS) func = &FFT::windowFuncBlackmanHarris;
		if(m_windowfunc->to_str() == WINDOW_FUNC_KAISER_1) func = &FFT::windowFuncKaiser1;
		if(m_windowfunc->to_str() == WINDOW_FUNC_KAISER_2) func = &FFT::windowFuncKaiser2;
		if(m_windowfunc->to_str() == WINDOW_FUNC_KAISER_3) func = &FFT::windowFuncKaiser3;
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
SpectrumSolverWrapper::onSolverChanged(const shared_ptr<XValueNodeBase> &) {
	shared_ptr<SpectrumSolver> solver;
	bool has_window = true;
	bool has_length = true;
	if(m_selector) {
		if(m_selector->to_str() == SPECTRUM_SOLVER_MEM_BURG_AICc) {
			solver.reset(new MEMBurg(&SpectrumSolver::icAICc));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_MEM_BURG_MDL) {
			solver.reset(new MEMBurg(&SpectrumSolver::icMDL));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_AR_YW_AICc) {
			solver.reset(new YuleWalkerAR(&SpectrumSolver::icAICc));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_AR_YW_MDL) {
			solver.reset(new YuleWalkerAR(&SpectrumSolver::icMDL));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_MUSIC_AIC) {
			solver.reset(new MUSIC(&SpectrumSolver::icAIC));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_MUSIC_MDL) {
			solver.reset(new MUSIC(&SpectrumSolver::icMDL));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_EV_AIC) {
			solver.reset(new EigenVectorMethod(&SpectrumSolver::icAIC));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_EV_MDL) {
			solver.reset(new EigenVectorMethod(&SpectrumSolver::icMDL));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_MVDL) {
			solver.reset(new MVDL);
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_MEM_STRICT) {
			solver.reset(new MEMStrict);
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_MEM_STRICT_EV) {
			solver.reset(new CompositeSpectrumSolver<MEMStrict, EigenVectorMethod>());
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_MEM_STRICT_BURG) {
			solver.reset(new CompositeSpectrumSolver<MEMStrict, MEMBurg>());
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_LS_AICc) {
			solver.reset(new FreqEstLeastSquare(&SpectrumSolver::icAICc));
		}
		if(m_selector->to_str() == SPECTRUM_SOLVER_LS_MDL) {
			solver.reset(new FreqEstLeastSquare(&SpectrumSolver::icMDL));
		}
	}
	if(!solver) {
		solver.reset(new FFTSolver);
	}
	if(m_windowfunc)
		m_windowfunc->setUIEnabled(has_window);
	if(m_windowlength)
		m_windowlength->setUIEnabled(has_length);
	m_solver = solver;	
}
