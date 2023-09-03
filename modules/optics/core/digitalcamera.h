/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
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

#ifndef digitalcameraH
#define digitalcameraH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmDigitalCamera;
typedef QForm<QMainWindow, Ui_FrmDigitalCamera> FrmDigitalCamera;

class XGraph;
class XWaveNGraph;
class X2DImagePlot;

//! Base class for scientific/machine vision digital camera.
class DECLSPEC_SHARED XDigitalCamera : public XPrimaryDriverWithThread {
public:
    XDigitalCamera(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
    virtual ~XDigitalCamera() {}
	//! Shows all forms belonging to driver.
    virtual void showForms() override;

    const shared_ptr<XGraph> &graph() const {return m_graph;}

	struct Payload : public XPrimaryDriver::Payload {
        double exposure() const {return m_exposure;} //! [s]

        double m_exposure;
        unsigned int m_accumulated;
        double m_electric_dark;
    private:
        friend class XDigitalCamera;
    };
protected:

	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;
  
	//! driver specific part below
    const shared_ptr<XDoubleNode> &exposure() const {return m_exposure;}
	const shared_ptr<XUIntNode> &average() const {return m_average;}
    const shared_ptr<XTouchableNode> &storeDark() const {return m_storeDark;}
    const shared_ptr<XBoolNode> &subtractDark() const {return m_subtractDark;}
protected:
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    virtual void onExposureChanged(const Snapshot &shot, XValueNodeBase *) = 0;
    void onStoreDarkTouched(const Snapshot &shot, XTouchableNode *);

    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    virtual void convertRawAndAccum(RawDataReader &reader, Transaction &tr) = 0;

    virtual void acquireSpectrum(shared_ptr<RawData> &) = 0;

private:
	const shared_ptr<XWaveNGraph> &waveForm() const {return m_waveForm;}
    const shared_ptr<XDoubleNode> m_exposure;
	const shared_ptr<XUIntNode> m_average;
    const shared_ptr<XTouchableNode> m_storeDark;
    const shared_ptr<XBoolNode> m_subtractDark;

    const qshared_ptr<FrmDigitalCamera> m_form;
	const shared_ptr<XWaveNGraph> m_waveForm;

	shared_ptr<Listener> m_lsnOnAverageChanged;
    shared_ptr<Listener> m_lsnOnExposureChanged;
    shared_ptr<Listener> m_lsnOnStoreDarkTouched;

    std::deque<xqcon_ptr> m_conUIs;

	shared_ptr<XGraph> m_graph;
    shared_ptr<X2DImagePlot> m_imagePlot;
	
    bool m_storeDarkInvoked;

    void *execute(const atomic<bool> &) override;
};

//---------------------------------------------------------------------------

#endif
