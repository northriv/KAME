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
    m_clearAverage(create<XTouchableNode>("ClearAverage", true)),
    m_subtractDark(create<XBoolNode>("SubtractDark", false)),
    m_videoMode(create<XComboNode>("VideoMode", true)),
    m_frameRate(create<XComboNode>("FrameRate", true)),
    m_autoGainForAverage(create<XBoolNode>("AutoGainForAverage", false)),
    m_gainForAverage(create<XDoubleNode>("GainForAverage", false)),
    m_status(create<XStringNode>("Status", true)),
    m_form(new FrmDigitalCamera),
    m_liveImage(create<X2DImage>("LiveImage", false,
                                   m_form->m_graphwidgetLive)),
    m_processedImage(create<X2DImage>("ProcessedImage", false,
                                   m_form->m_graphwidgetProcessed, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump)) {

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(videoMode(), m_form->m_cmbVideomode, Snapshot( *videoMode())),
        xqcon_create<XQComboBoxConnector>(frameRate(), m_form->m_cmbFrameRate, Snapshot( *frameRate())),
        xqcon_create<XQSpinBoxUnsignedConnector>(average(), m_form->m_spbAverage),
        xqcon_create<XQSpinBoxUnsignedConnector>(shutter(), m_form->m_spbShutter),
        xqcon_create<XQSpinBoxUnsignedConnector>(brightness(), m_form->m_spbBrightness),
        xqcon_create<XQDoubleSpinBoxConnector>(gainForAverage(), m_form->m_dblGain),
//        xqcon_create<XQLineEditConnector>((), m_form->m_edIntegrationTime),
        xqcon_create<XQButtonConnector>(m_clearAverage, m_form->m_btnClearAverage),
        xqcon_create<XQButtonConnector>(storeDark(), m_form->m_btnStoreDark),
        xqcon_create<XQToggleButtonConnector>(subtractDark(), m_form->m_ckbSubtractDark),
        xqcon_create<XQToggleButtonConnector>(m_autoGainForAverage, m_form->m_ckbAutoGain)
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
XDigitalCamera::onClearAverageTouched(const Snapshot &shot, XTouchableNode *) {
    m_clearAverageInvoked = true;
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
      iterate_commit([&](Transaction &tr){
        m_liveImage->setImage(tr, tr[ *this].liveImage());
      });
}

local_shared_ptr<std::vector<uint32_t>>
XDigitalCamera::summedCountsFromPool(int imagesize) {
    local_shared_ptr<std::vector<uint32_t>> summedCountsNext, p;
    for(int i = 0; i < NumSummedCountsPool; ++i) {
        if( !m_summedCountsPool[i])
            m_summedCountsPool[i] = make_local_shared<std::vector<uint32_t>>(imagesize);
        p = m_summedCountsPool[i];
        if(p.use_count() == 2) { //not owned by other threads.
            summedCountsNext = p;
            p->resize(imagesize);
        }
    }
    if( !summedCountsNext)
        summedCountsNext = make_local_shared<std::vector<uint32_t>>(imagesize);
    return summedCountsNext;
}

void
XDigitalCamera::setGray16Image(RawDataReader &reader, Transaction &tr, uint32_t width, uint32_t height, bool big_endian) {
    auto summedCountsNext = summedCountsFromPool(width * height);
    if(m_clearAverageInvoked || !tr[ *this].m_summedCounts || (tr[ *this].m_summedCounts->size() != width * height)) {
        tr[ *this].m_summedCounts = make_local_shared<std::vector<uint32_t>>(width * height, 0);
        m_clearAverageInvoked = false;
        tr[ *this].m_accumulated = 0;
    }
    tr[ *this].m_electric_dark = 0.0; //unsupported
    uint32_t *summedNext = &summedCountsNext->at(0);
    const uint32_t *summed = &tr[ *this].m_summedCounts->at(0);
    auto liveimage = std::make_shared<QImage>(width, height, QImage::Format_Grayscale16);
    auto processedimage = std::make_shared<QImage>(width, height, QImage::Format_Grayscale16);
    uint16_t *words = reinterpret_cast<uint16_t*>(liveimage->bits());
    uint32_t vmin = 0xffffffffu;
    uint32_t vmax = 0u;
    uint32_t gain_disp = lrint(0x100u * tr[ *gainForAverage()]);
    for(unsigned int i  = 0; i < width * height; ++i) {
         auto gray = reader.pop<uint16_t>();
#ifdef __BIG_ENDIAN__
         if( !big_endian) {
#else
         if(big_endian) {
#endif
             uint8_t *b = reinterpret_cast<uint8_t *>( &gray);
             uint8_t b0 = b[0];
             b[0] = b[1];
             b[1] = b0;
         }
         uint32_t w = gray * gain_disp / 0x100u;
         if(w >= 0x10000u)
             w = 0xffffu;
         *words++ = w;
         uint64_t v = gray + *summed++;
         if(v > 0x100000000uL)
             v = 0xffffffffu;
         if(v > vmax)
             vmax = v;
         if(v < vmin)
             vmin = v;
         *summedNext++ = v;
    }
    tr[ *this].m_accumulated++;
    tr[ *this].m_summedCounts = summedCountsNext;
    tr[ *this].m_avMax = vmax / tr[ *this].accumulated();
    tr[ *this].m_avMin = vmin / tr[ *this].accumulated();
    tr[ *this].m_liveImage = liveimage;
    tr[ *this].m_processedImage = processedimage;
    if(m_storeDarkInvoked) {
        tr[ *this].m_darkCounts = summedCountsNext;
        m_storeDarkInvoked = false;
    }
    if(tr[ *m_autoGainForAverage]) {
        tr[ *gainForAverage()]  = (double)0xffffu / (vmax / tr[ *this].m_accumulated);
    }
    uint64_t gain_av = lrint(0x100000000uL * tr[ *gainForAverage()] / tr[ *this].m_accumulated);
    uint16_t *processed = reinterpret_cast<uint16_t*>(processedimage->bits());
    summed = &tr[ *this].m_summedCounts->at(0);
    if( !tr[ *subtractDark()] || !tr[ *this].m_darkCounts || (tr[ *this].m_darkCounts->size() != width * height)) {
        for(unsigned int i  = 0; i < width * height; ++i) {
            *processed++ = (*summed++ * gain_av)  / 0x100000000uL;
        }
    }
    else {
        uint32_t *dark = &tr[ *this].m_darkCounts->at(0);
        uint64_t gain_dark = lrint(0x100000000uL * tr[ *gainForAverage()] / tr[ *this].m_darkAccumulated);
        for(unsigned int i  = 0; i < width * height; ++i) {
            *processed++ = (*summed++ * gain_av - *dark++ * gain_dark)  / 0x100000000uL;
        }
    }
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
        m_lsnOnVideoModeChanged = tr[ *videoMode()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onVideoModeChanged);
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
