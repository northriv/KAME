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
#ifndef NMRSPECTRUMSOLVER_H_
#define NMRSPECTRUMSOLVER_H_
//---------------------------------------------------------------------------
#include "support.h"
#include "nmrmem.h"
#include "xitemnode.h"

class SpectrumSolverWrapper : public XNode {
	XNODE_OBJECT
public:
	SpectrumSolverWrapper(const char *name, bool runtime,
		const shared_ptr<XComboNode> selector, const shared_ptr<XComboNode> windowfunc,
		const shared_ptr<XDoubleNode> windowlength);
	~SpectrumSolverWrapper();
	shared_ptr<SpectrumSolver> solver() {return m_solver;}
	  
	static const char SPECTRUM_SOLVER_ZF_FFT[];
	static const char SPECTRUM_SOLVER_MEM_STRICT[];
	static const char SPECTRUM_SOLVER_MEM_BURG_AICc[];
	static const char SPECTRUM_SOLVER_MEM_BURG_MDL[];
	static const char SPECTRUM_SOLVER_AR_YW_AICc[];
	static const char SPECTRUM_SOLVER_AR_YW_MDL[];

	static const char WINDOW_FUNC_DEFAULT[];
	static const char WINDOW_FUNC_HANNING[];
	static const char WINDOW_FUNC_HAMMING[];
	static const char WINDOW_FUNC_FLATTOP[];
	static const char WINDOW_FUNC_BLACKMAN[];
	static const char WINDOW_FUNC_BLACKMAN_HARRIS[];
	static const char WINDOW_FUNC_KAISER_1[];
	static const char WINDOW_FUNC_KAISER_2[];
	static const char WINDOW_FUNC_KAISER_3[];
	
	SpectrumSolver::twindowfunc windowFunc() const;
private:
	const shared_ptr<XComboNode> m_selector, m_windowfunc;
	const shared_ptr<XDoubleNode> m_windowlength;
	shared_ptr<SpectrumSolver> m_solver;
	shared_ptr<XListener> m_lsnOnChanged;
	void onSolverChanged(const shared_ptr<XValueNodeBase> &);
};

#endif /*NMRSPECTRUMSOLVER_H_*/
