/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef nmrfspectrumH
#define nmrfspectrumH
#include "secondarydriver.h"
#include "xnodeconnector.h"
#include <complex>

class XSG;
class XNMRPulseAnalyzer;
class FrmNMRFSpectrum;
class XWaveNGraph;

class FrmNMRSGControl;

class XNMRFSpectrum : public XSecondaryDriver
{
 XNODE_OBJECT
 protected:
  XNMRFSpectrum(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  //! ususally nothing to do
  ~XNMRFSpectrum() {}
  
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
  
  const shared_ptr<XItemNode<XDriverList, XNMRPulseAnalyzer> > &pulse() const {return m_pulse;}
  const shared_ptr<XItemNode<XDriverList, XSG> > &sg1() const {return m_sg1;}
  const shared_ptr<XItemNode<XDriverList, XSG> > &sg2() const {return m_sg2;}

  //! [MHz]
  const shared_ptr<XDoubleNode> &sg1FreqOffset() const {return m_sg1FreqOffset;}
  //! [MHz]
  const shared_ptr<XDoubleNode> &sg2FreqOffset() const {return m_sg2FreqOffset;}
  //! [MHz]
  const shared_ptr<XDoubleNode> &centerFreq() const {return m_centerFreq;}
  //! [kHz]
  const shared_ptr<XDoubleNode> &bandWidth() const {return m_bandWidth;}
  //! [kHz]
  const shared_ptr<XDoubleNode> &freqSpan() const {return m_freqSpan;}
  //! [kHz]
  const shared_ptr<XDoubleNode> &freqStep() const {return m_freqStep;}
  const shared_ptr<XBoolNode> &active() const {return m_active;}

  //! records below.
  const std::deque<std::complex<double> > &wave() const {return m_wave;}
  //! averaged count
  const std::deque<int> &counts() const {return m_counts;}
  //! freq resolution [MHz]
  double df() const {return m_df;}
  //! freq of the first point [MHz]
  double fMin() const {return m_fMin;}
 private:
  const shared_ptr<XItemNode<XDriverList, XNMRPulseAnalyzer> > m_pulse;
  const shared_ptr<XItemNode<XDriverList, XSG> > m_sg1, m_sg2;
 
  const shared_ptr<XDoubleNode> m_sg1FreqOffset;
  const shared_ptr<XDoubleNode> m_sg2FreqOffset;
  const shared_ptr<XDoubleNode> m_centerFreq;
  const shared_ptr<XDoubleNode> m_bandWidth;
  const shared_ptr<XDoubleNode> m_freqSpan;
  const shared_ptr<XDoubleNode> m_freqStep;
  const shared_ptr<XBoolNode> m_active;
  const shared_ptr<XNode> m_clear;
  
  //! Records
  std::deque<int> m_counts;
  double m_df, m_fMin;
  std::deque<std::complex<double> > m_wave;

  shared_ptr<XListener> m_lsnOnClear, m_lsnOnCondChanged, m_lsnOnActiveChanged;
    
  const qshared_ptr<FrmNMRFSpectrum> m_form;

  const shared_ptr<XWaveNGraph> m_spectrum;

  xqcon_ptr m_conCenterFreq, m_conBandWidth,
   m_conFreqSpan, m_conFreqStep;
  xqcon_ptr m_conSG1FreqOffset, m_conSG2FreqOffset;
  xqcon_ptr m_conClear, m_conActive;
  xqcon_ptr m_conSG1, m_conSG2, m_conPulse;

  void onCondChanged(const shared_ptr<XValueNodeBase> &);
  void onClear(const shared_ptr<XNode> &);
  void add(double freq, std::complex<double> c);
  void onActiveChanged(const shared_ptr<XValueNodeBase> &);
        
  XTime m_timeClearRequested;
};


#endif
