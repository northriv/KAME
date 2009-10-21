/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
#include <nmrspectrumbase.h>

class XMagnetPS;
class XDMM;

class Ui_FrmNMRSpectrum;
typedef QForm<QMainWindow, Ui_FrmNMRSpectrum> FrmNMRSpectrum;

class XNMRSpectrum : public XNMRSpectrumBase<FrmNMRSpectrum>
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
protected:
	//! \return true to be cleared.
	virtual bool onCondChangedImpl(const shared_ptr<XValueNodeBase> &) const;
	virtual double getFreqResHint() const;
	virtual double getMinFreq() const;
	virtual double getMaxFreq() const;
	virtual double getCurrentCenterFreq() const;
	virtual void getValues(std::vector<double> &values) const;

	virtual bool checkDependencyImpl(const shared_ptr<XDriver> &emitter) const;
public:
	const shared_ptr<XItemNode<XDriverList, XMagnetPS, XDMM> > &magnet() const {return m_magnet;}

	const shared_ptr<XDoubleNode> &centerFreq() const {return m_centerFreq;}
	const shared_ptr<XDoubleNode> &resolution() const {return m_resolution;}
	const shared_ptr<XDoubleNode> &minValue() const {return m_minValue;}
	const shared_ptr<XDoubleNode> &maxValue() const {return m_maxValue;}
	const shared_ptr<XDoubleNode> &fieldFactor() const {return m_fieldFactor;}
	const shared_ptr<XDoubleNode> &residualField() const {return m_residualField;}
private:
	const shared_ptr<XItemNode<XDriverList, XMagnetPS, XDMM> > m_magnet;
 
	const shared_ptr<XDoubleNode> m_centerFreq;
	const shared_ptr<XDoubleNode> m_resolution;
	const shared_ptr<XDoubleNode> m_minValue, m_maxValue;
	const shared_ptr<XDoubleNode> m_fieldFactor;
	const shared_ptr<XDoubleNode> m_residualField;
	xqcon_ptr m_conCenterFreq, m_conResolution, m_conMin, m_conMax,
		m_conFieldFactor, m_conResidualField;
	xqcon_ptr m_conMagnet;
};

#endif
