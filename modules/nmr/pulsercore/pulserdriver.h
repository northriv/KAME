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
//---------------------------------------------------------------------------
#ifndef pulserdriverH
#define pulserdriverH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xitemnode.h"
#include "xnodeconnector.h"
#include "softtrigger.h"
#include <complex>
#include "fft.h"

class QMainWindow;
class Ui_FrmPulser;
typedef QForm<QMainWindow, Ui_FrmPulser> FrmPulser;
class Ui_FrmPulserMore;
typedef QForm<QMainWindow, Ui_FrmPulserMore> FrmPulserMore;

class XQPulserDriverConnector;

//! Base class of NMR Pulsers
class DECLSPEC_SHARED XPulser : public XPrimaryDriver {
public:
	XPulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
;
	virtual ~XPulser() {}
	//! shows all forms belonging to driver
	virtual void showForms();

	//! driver specific part below

	//! \sa pulseAnalyzerMode()
	enum {N_MODE_NMR_PULSER = 0, N_MODE_PULSE_ANALYZER = 1};
    //! \sa combMode(), Payload::combMode().
	enum {N_COMB_MODE_OFF = 0, N_COMB_MODE_ON = 1, N_COMB_MODE_P1_ALT = 2, N_COMB_MODE_COMB_ALT = 3};
    //! \sa rtMode(), Payload::rtMode().
    enum {N_RT_MODE_FIXREP = 0, N_RT_MODE_FIXREST = 1};
	//! \sa numPhaseCycle(), Payload::numPhaseCycle().
	enum {MAX_NUM_PHASE_CYCLE = 16};
	//! # of digital-pulse ports.
	enum {NUM_DO_PORTS= 16};
  	//! for RelPatList patterns. \sa Payload::RelPatList.
	enum {PAT_DO_MASK = (1 << NUM_DO_PORTS) - 1,
		  PAT_QAM_PHASE = (1 << NUM_DO_PORTS),
		  PAT_QAM_PHASE_MASK = PAT_QAM_PHASE * 3,
		  PAT_QAM_PULSE_IDX = PAT_QAM_PHASE * 4,
		  PAT_QAM_PULSE_IDX_P1 = PAT_QAM_PULSE_IDX * 1,
		  PAT_QAM_PULSE_IDX_P2 = PAT_QAM_PULSE_IDX * 2,
		  PAT_QAM_PULSE_IDX_PCOMB = PAT_QAM_PULSE_IDX * 3,
		  PAT_QAM_PULSE_IDX_INDUCE_EMISSION = PAT_QAM_PULSE_IDX * 4,
		  PAT_QAM_PULSE_IDX_MASK = PAT_QAM_PULSE_IDX * 15,
		  PAT_QAM_MASK = PAT_QAM_PHASE_MASK | PAT_QAM_PULSE_IDX_MASK,
  	};

	struct Payload : public XPrimaryDriver::Payload {
	    //! ver 1 records.
	    int16_t combMode() const {return m_combMode;}
	    double rtime() const {return m_rtime;}
	    double tau() const {return m_tau;}
	    double pw1() const {return m_pw1;}
	    double pw2() const {return m_pw2;}
	    double combP1() const {return m_combP1;}
	    double altSep() const {return m_altSep;}
	    double combP1Alt() const {return m_combP1Alt;}
	    double aswSetup() const {return m_aswSetup;}
	    double aswHold() const {return m_aswHold;}
	    //! ver 2 records.
	    double difFreq() const {return m_difFreq;}
	    double combPW() const {return m_combPW;}
	    double combPT() const {return m_combPT;}
	    uint16_t echoNum() const {return m_echoNum;}
	    uint16_t combNum() const {return m_combNum;}
	    int16_t rtMode() const {return m_rtMode;}
	    uint16_t numPhaseCycle() const {return m_numPhaseCycle;}
	    //! ver 3 records.
	    bool invertPhase() const {return m_invertPhase;}
	    //! ver 4 records.
	    int16_t p1Func() const {return m_p1Func;}
	    int16_t p2Func() const {return m_p2Func;}
	    int16_t combFunc() const {return m_combFunc;}
	    double p1Level() const {return m_p1Level;}
	    double p2Level() const {return m_p2Level;}
	    double combLevel() const {return m_combLevel;}
	    double masterLevel() const {return m_masterLevel;}
	    double combOffRes() const {return m_combOffRes;}
	    bool conserveStEPhase() const {return m_conserveStEPhase;}

	    bool isPulseAnalyzerMode() const {return  m_paPulseBW > 0;}
	    double paPulseRept() const {return m_rtime;}
	    double paPulseBW() const {return m_paPulseBW;}
	    double paPulseOrigin() const {return m_paPulseOrigin;}

	    //! periodic term of one cycle [ms].
	    double periodicTerm() const;

		struct RelPat {
			RelPat(uint32_t pat, uint64_t t, uint64_t toapp) :
				pattern(pat), time(t), toappear(toapp) {}
			uint32_t pattern;
			uint64_t time; //!< using unit of resolution().
			uint64_t toappear; //!< term between this pattern and the previous. unit of resolution().
		};

		typedef std::deque<RelPat> RelPatList;
		RelPatList &relPatList() {return m_relPatList;}
		const RelPatList &relPatList() const {return m_relPatList;}

	  	const std::vector<std::complex<double> > &qamWaveForm(unsigned int idx) const {
	  		return m_qamWaveForm[idx];
	  	}
	private:
		friend class XPulser;

	    //! ver 1 records
	    int16_t m_combMode;
	    int16_t m_pulserMode;
	    double m_rtime;
	    double m_tau;
	    double m_pw1;
	    double m_pw2;
	    double m_combP1;
	    double m_altSep;
	    double m_combP1Alt;
	    double m_aswSetup;
	    double m_aswHold;
	    //! ver 2 records
	    double m_difFreq;
	    double m_combPW;
	    double m_combPT;
	    uint16_t m_echoNum;
	    uint16_t m_combNum;
	    int16_t m_rtMode;
	    uint16_t m_numPhaseCycle;
	    //! ver 3 records
	    bool m_invertPhase;
	    //! ver 4 records
	    int16_t m_p1Func, m_p2Func, m_combFunc;
	    double m_p1Level, m_p2Level, m_combLevel, m_masterLevel;
	    double m_combOffRes;
	    bool m_conserveStEPhase;

	    //! PA mode
	    double m_paPulseBW;
	    double m_paPulseOrigin; //!< [us]

	    //! Patterns.
	    RelPatList m_relPatList;
		std::vector<std::complex<double> >
            m_qamWaveForm[XPulser::PAT_QAM_PULSE_IDX_MASK / XPulser::PAT_QAM_PULSE_IDX];
	};
	
	const shared_ptr<XBoolNode> &output() const {return m_output;}
 	const shared_ptr<XComboNode> &combMode() const {return m_combMode;} //!< see above definitions in header file
	//! Control period to next pulse sequence
	//! Fix Repetition Time or Fix Rest Time which means time between pulse sequences 
 	const shared_ptr<XComboNode> &rtMode() const {return m_rtMode;}
    const shared_ptr<XComboNode> &numPhaseCycle() const {return m_numPhaseCycle;} //!< How many cycles in phase cycling
    const shared_ptr<XDoubleNode> &rtime() const {return m_rt;} //!< Repetition/Rest Time [ms]
    const shared_ptr<XDoubleNode> &tau() const {return m_tau;}  //!< [us]
    const shared_ptr<XDoubleNode> &combPW() const {return m_combPW;} //!< PulseWidths [us]
    const shared_ptr<XDoubleNode> &pw1() const {return m_pw1;} //!< PulseWidths [us]
    const shared_ptr<XDoubleNode> &pw2() const {return m_pw2;} //!< PulseWidths [us]
    const shared_ptr<XDoubleNode> &combPT() const {return m_combPT;} //!< Comb pulse periodic term [us]
    const shared_ptr<XDoubleNode> &combP1() const {return m_combP1;} //!< P1 and P1 alternative
    const shared_ptr<XDoubleNode> &combP1Alt() const {return m_combP1Alt;} //!< P1 and P1 alternative
    const shared_ptr<XDoubleNode> &aswSetup() const {return m_aswSetup;} //!< Analog switch setting, setup(proceeding) time before the first spin echo
    const shared_ptr<XDoubleNode> &aswHold() const {return m_aswHold;}  //!< Analog switch setting, hold time after the last spin echo
    const shared_ptr<XDoubleNode> &altSep() const {return m_altSep;} //!< Separation time in DSO record, cause a shift of trigger of DSO in alternatively mode
    const shared_ptr<XDoubleNode> &g2Setup() const {return m_g2Setup;} //!< Setup time of pre-gating port and QPSK
    const shared_ptr<XUIntNode> &combNum() const {return m_combNum;} //!< # of comb pulses
    const shared_ptr<XUIntNode> &echoNum() const {return m_echoNum;} //!< # of Spin echoes (i.e. pi pulses)
    const shared_ptr<XBoolNode> &drivenEquilibrium() const {return m_drivenEquilibrium;} //!< polarize spins after pulse sequence or not
    const shared_ptr<XDoubleNode> &combOffRes() const {return m_combOffRes;} //!< off-resonance comb pulses
    const shared_ptr<XComboNode> &combFunc() const {return m_combFunc;} //!< Pulse Modulation
    const shared_ptr<XComboNode> &p1Func() const {return m_p1Func;} //!< Pulse Modulation
    const shared_ptr<XComboNode> &p2Func() const {return m_p2Func;} //!< Pulse Modulation
    const shared_ptr<XDoubleNode> &combLevel() const {return m_combLevel;} //!< [dB], Pulse Modulation
    const shared_ptr<XDoubleNode> &p1Level() const {return m_p1Level;} //!< [dB], Pulse Modulation
    const shared_ptr<XDoubleNode> &p2Level() const {return m_p2Level;} //!< [dB], Pulse Modulation
    const shared_ptr<XDoubleNode> &masterLevel() const {return m_masterLevel;} //!< [dB]
    const shared_ptr<XBoolNode> &induceEmission() const {return m_induceEmission;}
    const shared_ptr<XDoubleNode> &induceEmissionPhase() const {return m_induceEmissionPhase;}
    const shared_ptr<XDoubleNode> &qamOffset1() const {return m_qamOffset1;}
    const shared_ptr<XDoubleNode> &qamOffset2() const {return m_qamOffset2;} //!< [%F.S.]
    const shared_ptr<XDoubleNode> &qamLevel1() const {return m_qamLevel1;} //! < Quadrature Amplitude Modulation. Amplitude compensation factor.
    const shared_ptr<XDoubleNode> &qamLevel2() const {return m_qamLevel2;}
    const shared_ptr<XDoubleNode> &qamDelay1() const {return m_qamDelay1;} //! < Delaying compensation [us].
    const shared_ptr<XDoubleNode> &qamDelay2() const {return m_qamDelay2;} //!< [us]
    const shared_ptr<XDoubleNode> &difFreq() const {return m_difFreq;} //!< [MHz]
    const shared_ptr<XDoubleNode> &qswDelay() const {return m_qswDelay;} //!< Q-switch setting, period after the end-edge of pulses [us].
    const shared_ptr<XDoubleNode> &qswWidth() const {return m_qswWidth;} //!< Q-switch setting, width of suppression [us].
    const shared_ptr<XDoubleNode> &qswSoftSWOff() const {return m_qswSoftSWOff;} //!< Q-switch setting, second pulse [us].
    const shared_ptr<XBoolNode> &qswPiPulseOnly() const {return m_qswPiPulseOnly;} //!< Q-switch setting, use QSW only for pi pulses.
    const shared_ptr<XBoolNode> &invertPhase() const {return m_invertPhase;}
    const shared_ptr<XBoolNode> &conserveStEPhase() const {return m_conserveStEPhase;}
    const shared_ptr<XComboNode> &portSel(unsigned int port) const {
    	assert(port < NUM_DO_PORTS);
    	return m_portSel[port];
    }
    const shared_ptr<XBoolNode> &pulseAnalyzerMode() const {return m_pulseAnalyzerMode;}
    const shared_ptr<XDoubleNode> &paPulseRept() const {return m_paPulseRept;}
    const shared_ptr<XDoubleNode> &paPulseBW() const {return m_paPulseBW;}
    const shared_ptr<XUIntNode> &firstPhase() const {return m_firstPhase;} //!< 0-3, selects the first phase of QPSK.

    //! time resolution [ms]
    virtual double resolution() const = 0;
protected:
	//! indice for return values of portSel().
	enum {PORTSEL_UNSEL = -1,
		  PORTSEL_GATE = 0, PORTSEL_PREGATE = 1, PORTSEL_GATE3 = 2,
		  PORTSEL_TRIG1 = 3, PORTSEL_TRIG2 = 4, PORTSEL_ASW = 5, PORTSEL_QSW = 6,
		  PORTSEL_PULSE1 = 7, PORTSEL_PULSE2 = 8, 
		  PORTSEL_COMB = 9, PORTSEL_COMB_FM = 10,
		  PORTSEL_QPSK_A = 11, PORTSEL_QPSK_B = 12,
		  PORTSEL_QPSK_OLD_NONINV = 13, PORTSEL_QPSK_OLD_INV = 14,
		  PORTSEL_QPSK_OLD_PSGATE = 15,
		  PORTSEL_PULSE_ANALYZER_GATE = 16,
		  /*PORTSEL_PAUSING = 17*/};
	//! \param func e.g. PORTSEL_GATE.
	//! \return bit mask.
	unsigned int selectedPorts(const Snapshot &shot, int func) const;
 
	//! Starts up your threads, connects GUI, and activates signals.
	virtual void start();
	//! Shuts down your threads, unconnects GUI, and deactivates signals
	//! This function may be called even if driver has already stopped.
	virtual void stop();
  
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! This function is called inside analyze() or analyzeRaw()
	//! This might be called even if the record is broken (time() == false).
	virtual void visualize(const Snapshot &shot);
  
	typedef FFT::twindowfunc tpulsefunc;
	tpulsefunc pulseFunc(int func_no) const;
	int pulseFuncNo(const XString &str) const;

    //! Sends patterns to pulser or turns off.
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) = 0;
    //! Converts RelPatList to native patterns
    virtual void createNativePatterns(Transaction &tr) = 0;
    virtual double resolutionQAM() const = 0;
    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const = 0;
    //! existence of AO ports.
    virtual bool hasQAMPorts() const = 0;

    bool hasSoftwareTrigger() const {return !!softwareTrigger();}
    shared_ptr<SoftwareTrigger> softwareTrigger() const {return m_softwareTrigger;}
    //! \sa SoftwareTriggerManager::create()
    shared_ptr<SoftwareTrigger> m_softwareTrigger;
private:
    const shared_ptr<XBoolNode> m_output;
    const shared_ptr<XComboNode> m_combMode; //!< see above definitions in header file
	//! Control period to next pulse sequence
	//! Fix Repetition Time or Fix Rest Time which means time between pulse sequences 
    const shared_ptr<XComboNode> m_rtMode;
    const shared_ptr<XDoubleNode> m_rt; //!< Repetition/Rest Time [ms]
    const shared_ptr<XDoubleNode> m_tau;  //!< [us]
    const shared_ptr<XDoubleNode> m_combPW, m_pw1, m_pw2; //!< PulseWidths [us]
    const shared_ptr<XUIntNode> m_combNum; //!< # of comb pulses
    const shared_ptr<XDoubleNode> m_combPT; //!< Comb pulse periodic term [us]
    const shared_ptr<XDoubleNode> m_combP1, m_combP1Alt; //!< P1 and P1 alternative
    const shared_ptr<XDoubleNode> m_aswSetup; //!< Analog switch setting, setup(proceeding) time before the first spin echo
    const shared_ptr<XDoubleNode> m_aswHold;  //!< Analog switch setting, hold time after the last spin echo
    const shared_ptr<XDoubleNode> m_altSep; //!< Separation time in DSO record, cause a shift of trigger of DSO in alternatively mode
    const shared_ptr<XDoubleNode> m_g2Setup; //!< Setup time of pre-gating port and QPSK
    const shared_ptr<XUIntNode> m_echoNum; //!< # of Spin echoes (i.e. pi pulses)
    const shared_ptr<XDoubleNode> m_combOffRes; //!< off-resonance comb pulses
    const shared_ptr<XBoolNode> m_drivenEquilibrium; //!< polarize spins after pulse sequence or not
    const shared_ptr<XComboNode> m_numPhaseCycle; //!< How many cycles in phase cycling
    const shared_ptr<XComboNode> m_p1Func, m_p2Func, m_combFunc; //!< Pulse Modulation
    const shared_ptr<XDoubleNode> m_p1Level, m_p2Level, m_combLevel; //!< [dB], Pulse Modulation
    const shared_ptr<XDoubleNode> m_masterLevel; //!< [dB]
    const shared_ptr<XDoubleNode> m_qamOffset1;
    const shared_ptr<XDoubleNode> m_qamOffset2; //!< [%F.S.]
    const shared_ptr<XDoubleNode> m_qamLevel1;
    const shared_ptr<XDoubleNode> m_qamLevel2;
    const shared_ptr<XDoubleNode> m_qamDelay1;
    const shared_ptr<XDoubleNode> m_qamDelay2; //!< [us]
    const shared_ptr<XDoubleNode> m_difFreq; //!< [MHz]
    const shared_ptr<XBoolNode> m_induceEmission; 
    const shared_ptr<XDoubleNode> m_induceEmissionPhase; 
    const shared_ptr<XDoubleNode> m_qswDelay;
    const shared_ptr<XDoubleNode> m_qswWidth;
    const shared_ptr<XDoubleNode> m_qswSoftSWOff;
    const shared_ptr<XBoolNode> m_invertPhase;
    const shared_ptr<XBoolNode> m_conserveStEPhase; 
    const shared_ptr<XBoolNode> m_qswPiPulseOnly;
    shared_ptr<XComboNode> m_portSel[NUM_DO_PORTS];
    const shared_ptr<XBoolNode> m_pulseAnalyzerMode;
    const shared_ptr<XDoubleNode> m_paPulseRept; //!< [ms]
    const shared_ptr<XDoubleNode> m_paPulseBW; //!< [kHz]
    const shared_ptr<XUIntNode> m_firstPhase; //!< 0-3, selects QPSK for the first cycle.

	const shared_ptr<XTouchableNode> m_moreConfigShow;
    std::deque<xqcon_ptr> m_conUIs;
    shared_ptr<Listener> m_lsnOnPulseChanged;
    shared_ptr<Listener> m_lsnOnMoreConfigShow;
	void onMoreConfigShow(const Snapshot &shot, XTouchableNode *);

	const qshared_ptr<FrmPulser> m_form;
	const qshared_ptr<FrmPulserMore> m_formMore;
  
	xqcon_ptr m_conPulserDriver;
    
	void onPulseChanged(const Snapshot &shot, XValueNodeBase *node);

	//! creates \a RelPatList
	void createRelPatListNMRPulser(Transaction &tr) throw (XRecordError&);
	void createRelPatListPulseAnalyzer(Transaction &tr) throw (XRecordError&);
	//! \return maskbits for QPSK ports.
	unsigned int bitpatternsOfQPSK(const Snapshot &shot, unsigned int qpsk[4], unsigned int qpskinv[4], bool invert);

	//! prepares waveforms for QAM.
	void makeWaveForm(Transaction &tr, unsigned int pnum_minus_1,
					  double pw, unsigned int to_center,
					  tpulsefunc func, double dB, double freq = 0.0, double phase = 0.0);
  
	//! truncates time by resolution().
	inline double rintTermMilliSec(double msec) const;
	inline double rintTermMicroSec(double usec) const;
	inline uint64_t ceilSampsMicroSec(double us) const;
	inline uint64_t rintSampsMicroSec(double us) const;
	inline uint64_t rintSampsMilliSec(double ms) const;

	void changeUIStatus(bool nmrmode, bool state);

    //! \sa SoftwareTrigger::onTriggerRequested()
    void onTriggerRequested(uint64_t threshold);
    shared_ptr<Listener> m_lsnOnTriggerRequested;
    int m_lastIdxFreeRun;
    uint32_t m_lastPatFreeRun;
    uint64_t m_totalSampsOfFreeRun;
};

inline double
XPulser::rintTermMilliSec(double msec) const {
	double res = resolution();
	return rint(msec / res) * res;
}
inline double
XPulser::rintTermMicroSec(double usec) const {
	double res = resolution() * 1e3;
	return rint(usec / res) * res;
}
inline uint64_t
XPulser::ceilSampsMicroSec(double usec) const {
	double res = resolution() * 1e3;
	return llrint(usec / res + 0.499);
}
inline uint64_t
XPulser::rintSampsMicroSec(double usec) const {
	double res = resolution() * 1e3;
	return llrint(usec / res);
}
inline uint64_t
XPulser::rintSampsMilliSec(double msec) const {
	double res = resolution();
	return llrint(msec / res);
}

#endif
