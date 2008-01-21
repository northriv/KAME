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

class FrmNMRPulse;
class XWaveNGraph;
class XXYPlot;
class FrmGraphNURL;

class XNMRPulseAnalyzer : public XSecondaryDriver
{
	XNODE_OBJECT
protected:
	XNMRPulseAnalyzer(const char *name, bool runtime,
					  const shared_ptr<XScalarEntryList> &scalarentries,
					  const shared_ptr<XInterfaceList> &interfaces,
					  const shared_ptr<XThermometerList> &thermometers,
					  const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XNMRPulseAnalyzer();
  
	//! show all forms belonging to driver
	virtual void showForms();
protected:

	//! this is called when connected driver emit a signal
	//! unless dependency is broken
	//! all connected drivers are readLocked
	virtual void analyze(const shared_ptr<XDriver> &emitter) throw (XRecordError&);
	//! this is called after analyze() or analyzeRaw()
	//! record is readLocked
	virtual void visualize();
	//! check connected drivers have valid time
	//! \return true if dependency is resolved
	virtual bool checkDependency(const shared_ptr<XDriver> &emitter) const;
 
public:
	//! driver specific part below 
  
	//! Entry storing the power of the strogest peak.
	const shared_ptr<XScalarEntry> &entryPeakPow() const {return m_entryPeakPow;}
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
	/// Dynamic Noise Reduction
	const shared_ptr<XBoolNode> &useDNR() const {return m_useDNR;}
	/// Select spectrum solver. FFT/AR/MEM.
	const shared_ptr<XComboNode> &solverList() const {return m_solverList;}
	/// Position from trigger, for background subtraction or DNR [ms]
	const shared_ptr<XDoubleNode> &bgPos() const {return m_bgPos;}
	/// length for background subtraction or DNR [ms]  
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

	/// # of echoes
	const shared_ptr<XUIntNode> &numEcho() const {return m_numEcho;}
	/// If NumEcho > 1, need periodic term of echoes [ms]
	const shared_ptr<XDoubleNode> &echoPeriod() const {return m_echoPeriod;}

	//! records below.

	/// Time-domain Wave.
	const std::vector<std::complex<double> > &wave() const {return m_wave;}
	//! freq. resolution [Hz]
	double dFreq() const {return m_dFreq;}
	//! time resolution [sec.]
	double interval() const {return m_interval;}
	//! time diff. of the first point from trigger [sec.]
	double startTime() const {return m_startTime;}
	//! Length of analyzed wave.
	int waveWidth() const {return m_waveWidth;}
	//! Position of the origin of FT.
	int waveFTPos() const {return m_waveFTPos;}
	//! Length of FT.
	int ftWidth() const {return m_ftWave.size();}
	
private:
	/// Stored Wave for display.
	const shared_ptr<XWaveNGraph> &waveGraph() const {return m_waveGraph;}
	/// Stored FFT Wave for display.
	const shared_ptr<XWaveNGraph> &ftWaveGraph() const {return m_ftWaveGraph;}

	const shared_ptr<XScalarEntry> m_entryPeakPow; 
	const shared_ptr<XScalarEntry> m_entryPeakFreq; 

	const shared_ptr<XItemNode<XDriverList, XDSO> > m_dso;
 
	const shared_ptr<XDoubleNode> m_fromTrig;
	const shared_ptr<XDoubleNode> m_width;

	const shared_ptr<XDoubleNode> m_phaseAdv;   ///< [deg]
	const shared_ptr<XBoolNode> m_useDNR;
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

	const shared_ptr<XNode> m_spectrumShow;
	const shared_ptr<XNode> m_avgClear;

	//! Phase Inversion Cycling
	const shared_ptr<XBoolNode> m_picEnabled;
	const shared_ptr<XItemNode<XDriverList, XPulser> > m_pulser;
  
	//! Records
	std::vector<std::complex<double> > m_wave;
	/// FFT Wave starting with -fmax/2.
	const std::vector<std::complex<double> > &ftWave() const {return m_ftWave;}
	std::vector<std::complex<double> > m_ftWave;
	std::vector<std::complex<double> > m_dsoWave;
	int m_dsoWaveStartPos, m_waveFTPos, m_waveWidth;
	double m_dFreq;  ///< Hz per point
	//! # of summations.
	int m_avcount;
	//! Stored Waves for avg.
	std::vector<std::complex<double> > m_waveSum;
	//! time resolution
	double m_interval;
	//! time diff. of the first point from trigger
	double m_startTime;
  
	xqcon_ptr m_conFromTrig, m_conWidth, m_conPhaseAdv, m_conBGPos, m_conUseDNR, m_conSolverList, 
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
  
	void onCondChanged(const shared_ptr<XValueNodeBase> &);
	void onSpectrumShow(const shared_ptr<XNode> &);
	void onAvgClear(const shared_ptr<XNode> &);
  
	void backgroundSub(std::vector<std::complex<double> > &wave, int pos, int length, int bgpos, int bglength);
  
	//for FFT/MEM.
	shared_ptr<SpectrumSolverWrapper> m_solver;
	shared_ptr<SpectrumSolver> m_solverRecorded;
	shared_ptr<XXYPlot> m_peakPlot;
	MEMStrict m_memDNR;
	
	void rotNFFT(int ftpos, double ph, 
				 std::vector<std::complex<double> > &wave, std::vector<std::complex<double> > &ftwave, int diffreq);
    
	XTime m_timeClearRequested;
};



#endif
