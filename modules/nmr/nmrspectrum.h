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
#ifndef nmrspectrumH
#define nmrspectrumH
//---------------------------------------------------------------------------
#include <secondarydriver.h>
#include <xnodeconnector.h>
#include <complex>

class XMagnetPS;
class XNMRPulseAnalyzer;
class FrmNMRSpectrum;
class XWaveNGraph;

class XNMRSpectrum : public XSecondaryDriver
{
	XNODE_OBJECT
protected:
	XNMRSpectrum(const char *name, bool runtime,
				 const shared_ptr<XScalarEntryList> &scalarentries,
				 const shared_ptr<XInterfaceList> &interfaces,
				 const shared_ptr<XThermometerList> &thermometers,
				 const shared_ptr<XDriverList> &drivers);
public:
	//! ususally nothing to do
	~XNMRSpectrum() {}
  
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
	const shared_ptr<XItemNode<XDriverList, XMagnetPS> > &magnet() const {return m_magnet;}

	const shared_ptr<XDoubleNode> &centerFreq() const {return m_centerFreq;}
	const shared_ptr<XDoubleNode> &bandWidth() const {return m_bandWidth;}
	const shared_ptr<XDoubleNode> &resolution() const {return m_resolution;}
	const shared_ptr<XDoubleNode> &fieldFactor() const {return m_fieldFactor;}
	const shared_ptr<XDoubleNode> &residualField() const {return m_residualField;}
	const shared_ptr<XDoubleNode> &fieldMin() const {return m_fieldMin;}
	const shared_ptr<XDoubleNode> &fieldMax() const {return m_fieldMax;}

	//! records below.
	const std::deque<std::complex<double> > &wave() const {return m_wave;}
	//! averaged count
	const std::deque<int> &counts() const {return m_counts;}
	//! field resolution
	double dH() const {return m_dH;}
	//! field of the first point
	double hMin() const {return m_hMin;}
private:
	const shared_ptr<XItemNode<XDriverList, XNMRPulseAnalyzer> > m_pulse;
	const shared_ptr<XItemNode<XDriverList, XMagnetPS> > m_magnet;
 
	const shared_ptr<XDoubleNode> m_centerFreq;
	const shared_ptr<XDoubleNode> m_bandWidth;
	const shared_ptr<XDoubleNode> m_resolution;
	const shared_ptr<XDoubleNode> m_fieldFactor;
	const shared_ptr<XDoubleNode> m_residualField;
	const shared_ptr<XDoubleNode> m_fieldMin;
	const shared_ptr<XDoubleNode> m_fieldMax;

	const shared_ptr<XNode> m_clear;
  
	//! Records
	std::deque<int> m_counts;
	double m_dH, m_hMin;
	std::deque<std::complex<double> > m_wave;

	shared_ptr<XListener> m_lsnOnClear, m_lsnOnCondChanged;
    
	const qshared_ptr<FrmNMRSpectrum> m_form;

	const shared_ptr<XWaveNGraph> m_spectrum;

	xqcon_ptr m_conCenterFreq, m_conBandWidth, m_conResolution,
		m_conFieldFactor, m_conResidualField, m_conFieldMin, m_conFieldMax;
	xqcon_ptr m_conMagnet, m_conPulse;
	xqcon_ptr m_conClear;

	void onCondChanged(const shared_ptr<XValueNodeBase> &);
	void onClear(const shared_ptr<XNode> &);
	void add(double field, std::complex<double> c);
        
	XTime m_timeClearRequested;
};

#endif
