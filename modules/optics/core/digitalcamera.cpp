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
#include "digitalcamera.h"
#include "ui_digitalcameraform.h"
#include "x2dimage.h"
#include "graph.h"
#include "graphwidget.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XDigitalCamera::XDigitalCamera(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_brightness(create<XUIntNode>("Brightness", true)),
    m_shutter(create<XUIntNode>("Shutter", true)),
    m_average(create<XUIntNode>("Average", false)),
    m_storeDark(create<XTouchableNode>("StoreDark", true)),
    m_subtractDark(create<XBoolNode>("SubtractDark", false)),
    m_videoMode(create<XComboNode>("VideoMode", true)),
    m_frameRate(create<XComboNode>("FrameRate", true)),
    m_autoGainForAverage(create<XBoolNode>("AutoGainForAverage", false)),
    m_gainForAverage(create<XDoubleNode>("GainForAverage", false)),
    m_status(create<XStringNode>("Status", true)),
    m_form(new FrmDigitalCamera),
    m_liveImage(create<X2DImage>("LiveImage", false,
                                   m_form->m_graphwidget, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump)),
    m_processedImage(create<X2DImage>("ProcessedImage", false,
                                   m_form->m_graphwidget, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump)) {

    m_conUIs = {
        xqcon_create<XQSpinBoxUnsignedConnector>(average(), m_form->m_spbAverage),
//        xqcon_create<XQLineEditConnector>((), m_form->m_edIntegrationTime),
        xqcon_create<XQButtonConnector>(storeDark(), m_form->m_btnStoreDark),
        xqcon_create<XQToggleButtonConnector>(subtractDark(), m_form->m_ckbSubtractDark)
    };

    std::vector<shared_ptr<XNode>> runtime_ui{
        storeDark(),
        brightness(),
        shutter(),
        videoMode(),
        frameRate(),
        status(),
    };
    iterate_commit([=](Transaction &tr){
        tr[ *average()] = 1;
        tr[ *autoGainForAverage()] = true;
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });
}
void
XDigitalCamera::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XDigitalCamera::onStoreDarkTouched(const Snapshot &shot, XTouchableNode *) {
    m_storeDarkInvoked = true;
}
void
XDigitalCamera::onVideoModeChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setVideoMode(shot[ *videoMode()]);
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}

void
XDigitalCamera::onBrightnessChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setBrightness(shot[ *brightness()]);
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XDigitalCamera::onShutterChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setShutter(shot[ *shutter()]);
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}

void
XDigitalCamera::visualize(const Snapshot &shot) {
	  if( !shot[ *this].time()) {
		return;
	  }
      //            iterate_commit([&](Transaction &tr){
      //                m_liveImage->setImage(tr, std::move( *image));
      //            });

}

void *
XDigitalCamera::execute(const atomic<bool> &terminated) {

    std::vector<shared_ptr<XNode>> runtime_ui{
        storeDark(),
        brightness(),
        shutter(),
        videoMode(),
        frameRate(),
        status(),
    };

    m_storeDarkInvoked = false;

	iterate_commit([=](Transaction &tr){
        m_lsnOnBrightnessChanged = tr[ *brightness()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onBrightnessChanged);
        m_lsnOnShutterChanged = tr[ *shutter()].onValueChanged().connectWeakly(
                    shared_from_this(), &XDigitalCamera::onShutterChanged);
        m_lsnOnStoreDarkTouched = tr[ *storeDark()].onTouch().connectWeakly(
            shared_from_this(), &XDigitalCamera::onStoreDarkTouched, Listener::FLAG_MAIN_THREAD_CALL);
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(true);
    });

	while( !terminated) {
		XTime time_awared = XTime::now();
		auto writer = std::make_shared<RawData>();
		// try/catch exception of communication errors
		try {
            acquireRaw(writer);
        }
		catch (XDriver::XSkippedRecordError&) {
			msecsleep(10);
			continue;
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}
		finishWritingRaw(writer, time_awared, XTime::now());
    }

    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_lsnOnBrightnessChanged.reset();
    m_lsnOnShutterChanged.reset();
    m_lsnOnStoreDarkTouched.reset();
	return NULL;
}
