/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef nmrpulseH
#define nmrpulseH
//---------------------------------------------------------------------------
#include <vector>
#include "secondarydriver.h"
#include "dso.h"
#include "pulserdriver.h"
#include <complex>
//---------------------------------------------------------------------------
#include "nmrspectrumsolver.h"
#include "xwavengraph.h"

class Ui_FrmNMRPulse;
typedef QForm<QMainWindow, Ui_FrmNMRPulse> FrmNMRPulse;

class XNMRPulseAnalyzer : public XSecondaryDriver {
public:
	XNMRPulseAnalyzer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XNMRPulseAnalyzer();
  
	//! Shows all forms belonging to driver
	virtual void showForms();
protected:

	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter,
		const Snapshot &shot_others,
		XDriver *emitter) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
	//! Checks if the connected drivers have valid time stamps.
	//! \return true if dependency is resolved.
	//! This function must be reentrant unlike analyze().
	virtual bool checkDependency(const Snapshot &shot_this,
		const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) const;
 
public:
	//! driver specific part below 
	struct Payload : public XSecondaryDriver::Payload {
		//! Time-domain Wave.
		const std::vector<std::complex<double> > &wave() const {return m_wave;}
		//! Power spectrum of the noise estimated from the background. [V^2/Hz].
		const std::vector<double> &darkPSD() const {return m_darkPSD;}
		//! freq. resolution [Hz]
		double dFreq() const {return m_dFreq;}
		//! time resolution [sec.]
		double interval() const {return m_interval;}
		//! time diff. of the first point from trigger [sec.]
		double startTime() const {return m_startTime;}
		//! Length of the wave().
		int waveWidth() const {return m_waveWidth;}
		//! Position of the origin of FT.
		int waveFTPos() const {return m_waveFTPos;}
		//! Length of FT.
		int ftWidth() const {return m_ftWave.size();}
	private:
		friend class XNMRPulseAnalyzer;
		std::vector<std::complex<double> > m_wave;
		std::vector<double> m_darkPSD;
		/// FFT Wave
		const std::vector<std::complex<double> > &ftWave() const {return m_ftWave;}
		double m_ftWavePSDCoeff;
		std::vector<std::complex<double> > m_ftWave;
		std::vector<std::complex<double> > m_dsoWave;
		int m_dsoWaveStartPos, m_waveFTPos, m_waveWidth;
		double m_dFreq;  ///< Hz per point
		//! # of summations.
		int m_avcount;
		//! Stored Waves for avg.
		std::vector<std::complex<double> > m_waveSum;
		std::vector<double> m_darkPSDSum;
		//! time resolution
		double m_interval;
		//! time diff. of the first point from trigger
		double m_startTime;

		//for FFT/MEM.
		shared_ptr<FFT> m_ftDark;

		XTime m_timeClearRequested;
	};
  
	//! Entry storing the power of the strogest peak.
	const shared_ptr<XScalarEntry> &entryPeakAbs() const {return m_entryPeakAbs;}
	//! Entry storing the freq. of the strongest peak.
	const shared_ptr<XScalarEntry> &entryPeakFreq() const {return m_entryPeakFreq;}

	const shared_ptr<XItemNode<XDriverList, XDSO> > &dso() const {return m_dso;}
	const shared_ptr<XItemNode<XDriverList, XPulser> > &pulser() const {return m_pulser;}

	void acquire();

	//! Origin from trigger [ms]
	const shared_ptr<XDoubleNode> &fromTrig() const {return m_fromTrig;}
	//! length of data points [ms]
	const shared_ptr<XDoubleNode> &width() const {return m_width;}

	const shared_ptr<XDoubleNode> &phaseAdv() const {return m_phaseAdv;}   ///< [deg]
	/// Periodic-Noise Reduction
	const shared_ptr<XBoolNode> &usePNR() const {return m_usePNR;}
	/// Select spectrum solver.
	const shared_ptr<XComboNode> &pnrSolverList() const {return m_pnrSolverList;}
	/// Select spectrum solver. FFT/AR/MEM.
	const shared_ptr<XComboNode> &solverList() const {return m_solverList;}
	/// Position from trigger, for background subtraction or PNR [ms]
	const shared_ptr<XDoubleNode> &bgPos() const {return m_bgPos;}
	/// length for background subtraction or PNR [ms]  
	const shared_ptr<XDoubleNode> &bgWidth() const {return m_bgWidth;}
	/// Phase 0 deg. position of FT component from trigger [ms]
	const shared_ptr<XDoubleNode> &fftPos() const {return m_fftPos;}
	/// If exceeding Width, do zerofilling
	const shared_ptr<XUIntNode> &fftLen() const {return m_fftLen;}
	/// FFT/AR Window Function
	const shared_ptr<XComboNode> &windowFunc() const {return m_windowFunc;}
	/// FFT/AR/MEM Window Length / Taps
	const shared_ptr<XDoubleNode> &windowWidth() const {return m_windowWidth;}
	/// Set Digital IF frequency
	const shared_ptr<XDoubleNode> &difFreq() const {return m_difFreq;}

	/// Extra Average with infinite steps
	const shared_ptr<XBoolNode> &exAvgIncr() const {return m_exAvgIncr;}
	/// Extra Average Steps
	const shared_ptr<XUIntNode> &extraAvg() const {return m_extraAvg;}
	/// Clear averaging results
	const shared_ptr<XTouchableNode> &avgClear() const {return m_avgClear;}

	/// # of echoes
	const shared_ptr<XUIntNode> &numEcho() const {return m_numEcho;}
	/// If NumEcho > 1, need periodic term of echoes [ms]
	const shared_ptr<XDoubleNode> &echoPeriod() const {return m_echoPeriod;}

private:
	/// Stored Wave for display.
	const shared_ptr<XWaveNGraph> &waveGraph() const {return m_waveGraph;}
	/// Stored FFT Wave for display.
	const shared_ptr<XWaveNGraph> &ftWaveGraph() const {return m_ftWaveGraph;}

	const shared_ptr<XScalarEntry> m_entryPeakAbs; 
	const shared_ptr<XScalarEntry> m_entryPeakFreq; 

	const shared_ptr<XItemNode<XDriverList, XDSO> > m_dso;
 
	const shared_ptr<XDoubleNode> m_fromTrig;
	const shared_ptr<XDoubleNode> m_width;

	const shared_ptr<XDoubleNode> m_phaseAdv;   ///< [deg]
	const shared_ptr<XBoolNode> m_usePNR;
	const shared_ptr<XComboNode> m_pnrSolverList;
	const shared_ptr<XComboNode> m_solverList;
	const shared_ptr<XDoubleNode> m_bgPos;
	const shared_ptr<XDoubleNode> m_bgWidth;
	const shared_ptr<XDoubleNode> m_fftPos;
	const shared_ptr<XUIntNode> m_fftLen;
	const shared_ptr<XComboNode> m_windowFunc;
	const shared_ptr<XDoubleNode> m_windowWidth;
	const shared_ptr<XDoubleNode> m_difFreq;

	const shared_ptr<XBoolNode> m_exAvgIncr;
	const shared_ptr<XUIntNode> m_extraAvg;

	const shared_ptr<XUIntNode> m_numEcho;
	const shared_ptr<XDoubleNode> m_echoPeriod;

	const shared_ptr<XTouchableNode> m_spectrumShow;
	const shared_ptr<XTouchableNode> m_avgClear;

	//! Phase Inversion Cycling
	const shared_ptr<XBoolNode> m_picEnabled;
	const shared_ptr<XItemNode<XDriverList, XPulser> > m_pulser;
	atomic<int> m_isPulseInversionRequested;

	xqcon_ptr m_conFromTrig, m_conWidth, m_conPhaseAdv, m_conBGPos,
		m_conUsePNR, m_conPNRSolverList, m_conSolverList, 
		m_conBGWidth, m_conFFTPos, m_conFFTLen, m_conExtraAv;
	xqcon_ptr m_conExAvgIncr;
	xqcon_ptr m_conAvgClear, m_conSpectrumShow, m_conWindowFunc, m_conWindowWidth, m_conDIFFreq;
	xqcon_ptr m_conNumEcho, m_conEchoPeriod;
	xqcon_ptr m_conDSO;
	xqcon_ptr m_conPulser, m_conPICEnabled;

	shared_ptr<XListener> m_lsnOnSpectrumShow, m_lsnOnAvgClear;
	shared_ptr<XListener> m_lsnOnCondChanged;
    
	const qshared_ptr<FrmNMRPulse> m_form;
	const shared_ptr<XStatusPrinter> m_statusPrinter;
	const qshared_ptr<FrmGraphNURL> m_spectrumForm;

	const shared_ptr<XWaveNGraph> m_waveGraph;
	const shared_ptr<XWaveNGraph> m_ftWaveGraph;
  
	//for FFT/MEM.
	shared_ptr<SpectrumSolverWrapper> m_solver;
	shared_ptr<SpectrumSolverWrapper> m_solverPNR;
	shared_ptr<XXYPlot> m_peakPlot;

	void onCondChanged(const Snapshot &shot, XValueNodeBase *);
	void onSpectrumShow(const Snapshot &shot, XTouchableNode *);
	void onAvgClear(const Snapshot &shot, XTouchableNode *);
  
	void backgroundSub(Transaction &tr,
		std::vector<std::complex<double> > &wave, int pos, int length, int bgpos, int bglength);
  
	void rotNFFT(Transaction &tr, int ftpos, double ph,
				 std::vector<std::complex<double> > &wave, std::vector<std::complex<double> > &ftwave);
};

#endif
