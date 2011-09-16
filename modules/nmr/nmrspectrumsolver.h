/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
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
#include "spectrumsolver.h"
#include "xitemnode.h"

class SpectrumSolverWrapper : public XNode {
public:
	SpectrumSolverWrapper(const char *name, bool runtime,
		const shared_ptr<XComboNode> selector, const shared_ptr<XComboNode> windowfunc,
		const shared_ptr<XDoubleNode> windowlength, bool leastsqureonly = false);
	~SpectrumSolverWrapper();

	struct Payload : public XNode::Payload {
		Payload() : XNode::Payload() {}
		Payload(const Payload &x) : XNode::Payload(x) {
			if(x.m_wrapper)
				m_wrapper.reset(x.m_wrapper->clone());
		}
		SpectrumSolver &solver() {return m_wrapper->solver();}
		const SpectrumSolver &solver() const {return m_wrapper->solver();}
	private:
		friend class SpectrumSolverWrapper;
		struct WrapperBase {
			virtual ~WrapperBase() {}
			virtual WrapperBase *clone() = 0;
			virtual SpectrumSolver &solver() = 0;
			virtual const SpectrumSolver &solver() const = 0;
		};
		template <class T>
		struct Wrapper : public WrapperBase {
			Wrapper(T *p) : m_solver(p) {}
			virtual Wrapper* clone() { return new Wrapper(new T( *m_solver)); }
			virtual T &solver() {return *m_solver; }
			virtual const T &solver() const {return *m_solver; }
		private:
			Wrapper();
			unique_ptr<T> m_solver;
		};
		unique_ptr<WrapperBase> m_wrapper;
	};
	  
	static const char SPECTRUM_SOLVER_ZF_FFT[];
	static const char SPECTRUM_SOLVER_MEM_STRICT[];
	static const char SPECTRUM_SOLVER_MEM_STRICT_EV[];
	static const char SPECTRUM_SOLVER_MEM_STRICT_BURG[];
	static const char SPECTRUM_SOLVER_MEM_BURG_AICc[];
	static const char SPECTRUM_SOLVER_MEM_BURG_MDL[];
	static const char SPECTRUM_SOLVER_AR_YW_AICc[];
	static const char SPECTRUM_SOLVER_AR_YW_MDL[];
	static const char SPECTRUM_SOLVER_MUSIC_AIC[];
	static const char SPECTRUM_SOLVER_MUSIC_MDL[];
	static const char SPECTRUM_SOLVER_EV_AIC[];
	static const char SPECTRUM_SOLVER_EV_MDL[];
	static const char SPECTRUM_SOLVER_MVDL[];
	static const char SPECTRUM_SOLVER_LS_HQ[];
	static const char SPECTRUM_SOLVER_LS_AICc[];
	static const char SPECTRUM_SOLVER_LS_MDL[];

	static const char WINDOW_FUNC_DEFAULT[];
	static const char WINDOW_FUNC_HANNING[];
	static const char WINDOW_FUNC_HAMMING[];
	static const char WINDOW_FUNC_FLATTOP[];
	static const char WINDOW_FUNC_BLACKMAN[];
	static const char WINDOW_FUNC_BLACKMAN_HARRIS[];
	static const char WINDOW_FUNC_KAISER_1[];
	static const char WINDOW_FUNC_KAISER_2[];
	static const char WINDOW_FUNC_KAISER_3[];
	
	FFT::twindowfunc windowFunc(const Snapshot &shot) const;
	void windowFuncs(std::deque<FFT::twindowfunc> &funcs) const;
private:
	const shared_ptr<XComboNode> m_selector, m_windowfunc;
	const shared_ptr<XDoubleNode> m_windowlength;
	shared_ptr<XListener> m_lsnOnChanged;
	void onSolverChanged(const Snapshot &shot, XValueNodeBase *);
};

#endif /*NMRSPECTRUMSOLVER_H_*/
