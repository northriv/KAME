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
#include "nmrspectrumsolver.h"

const char SpectrumSolverWrapper::SPECTRUM_SOLVER_ZF_FFT[] = "ZF-FFT";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_BURG[] = "Burg's MEM";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_MEM_STRICT[] = "Strict MEM";
const char SpectrumSolverWrapper::SPECTRUM_SOLVER_AR_YW[] = "Yule-Walker AR";

const char SpectrumSolverWrapper::WINDOW_FUNC_DEFAULT[] = "Rect/Tri(AR)";
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
	selector->add(SPECTRUM_SOLVER_ZF_FFT);
	selector->add(SPECTRUM_SOLVER_MEM_BURG);
	selector->add(SPECTRUM_SOLVER_AR_YW);
	selector->add(SPECTRUM_SOLVER_MEM_STRICT);
	windowfunc->add(WINDOW_FUNC_DEFAULT);
	windowfunc->add(WINDOW_FUNC_HANNING);
	windowfunc->add(WINDOW_FUNC_HAMMING);
	windowfunc->add(WINDOW_FUNC_BLACKMAN);
	windowfunc->add(WINDOW_FUNC_BLACKMAN_HARRIS);
	windowfunc->add(WINDOW_FUNC_FLATTOP);
	windowfunc->add(WINDOW_FUNC_KAISER_1);
	windowfunc->add(WINDOW_FUNC_KAISER_2);
	windowfunc->add(WINDOW_FUNC_KAISER_3);
	m_lsnOnChanged = selector->onValueChanged().connectWeak(shared_from_this(), &SpectrumSolverWrapper::onSolverChanged);
	onSolverChanged(selector);
}
SpectrumSolverWrapper::~SpectrumSolverWrapper() {
	m_selector->clear();
}
SpectrumSolver::twindowfunc
SpectrumSolverWrapper::windowFunc() const {
	SpectrumSolver::twindowfunc func = &SpectrumSolver::windowFuncRect;
	if(m_selector->to_str() == SPECTRUM_SOLVER_AR_YW)
		func = &SpectrumSolver::windowFuncTri;
	if(m_windowfunc->to_str() == WINDOW_FUNC_HANNING) func = &SpectrumSolver::windowFuncHanning;
	if(m_windowfunc->to_str() == WINDOW_FUNC_HAMMING) func = &SpectrumSolver::windowFuncHamming;
	if(m_windowfunc->to_str() == WINDOW_FUNC_FLATTOP) func = &SpectrumSolver::windowFuncFlatTop;
	if(m_windowfunc->to_str() == WINDOW_FUNC_BLACKMAN) func = &SpectrumSolver::windowFuncBlackman;
	if(m_windowfunc->to_str() == WINDOW_FUNC_BLACKMAN_HARRIS) func = &SpectrumSolver::windowFuncBlackmanHarris;
	if(m_windowfunc->to_str() == WINDOW_FUNC_KAISER_1) func = &SpectrumSolver::windowFuncKaiser1;
	if(m_windowfunc->to_str() == WINDOW_FUNC_KAISER_2) func = &SpectrumSolver::windowFuncKaiser2;
	if(m_windowfunc->to_str() == WINDOW_FUNC_KAISER_3) func = &SpectrumSolver::windowFuncKaiser3;
	return func;
}

void
SpectrumSolverWrapper::onSolverChanged(const shared_ptr<XValueNodeBase> &) {
	shared_ptr<SpectrumSolver> solver;
	if(m_selector->to_str() == SPECTRUM_SOLVER_MEM_BURG) {
		m_windowfunc->setUIEnabled(false);		
		solver.reset(new MEMBurg);
	}
	if(m_selector->to_str() == SPECTRUM_SOLVER_MEM_STRICT) {
		m_windowfunc->setUIEnabled(false);		
		solver.reset(new MEMStrict);
	}
	if(m_selector->to_str() == SPECTRUM_SOLVER_AR_YW) {
		m_windowfunc->setUIEnabled(true);
		solver.reset(new YuleWalkerAR);
	}
	if(!solver) {
		m_windowfunc->setUIEnabled(true);
		solver.reset(new FFTSolver);
	}
	m_solver = solver;	
}
