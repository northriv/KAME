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
    m_exposureTime(create<XDoubleNode>("ExposureTime", true)),
    m_average(create<XUIntNode>("Average", false)),
    m_storeDark(create<XTouchableNode>("StoreDark", true)),
    m_clearAverage(create<XTouchableNode>("ClearAverage", true)),
    m_roiSelectionTool(create<XTouchableNode>("ROISelectionTool", true)),
    m_subtractDark(create<XBoolNode>("SubtractDark", false)),
    m_videoMode(create<XComboNode>("VideoMode", true)),
    m_triggerMode(create<XComboNode>("TriggerMode", true)),
    m_frameRate(create<XComboNode>("FrameRate", true)),
    m_coloringMethod(create<XComboNode>("ColoringMethod", false, true)),
    m_autoGainForAverage(create<XBoolNode>("AutoGainForAverage", false)),
    m_incrementalAverage(create<XBoolNode>("IncrementalAverage", false)),
    m_colorIndex(create<XUIntNode>("ColorIndex", true)),
    m_gainForAverage(create<XDoubleNode>("GainForAverage", false)),
    m_form(new FrmDigitalCamera),
    m_liveImage(create<X2DImage>("LiveImage", false,
                                   m_form->m_graphwidgetLive)),
    m_processedImage(create<X2DImage>("ProcessedImage", false,
                                   m_form->m_graphwidgetProcessed, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
                                   MAX_COLORS + 1, //w/dark
                                   m_form->m_tbMathMenu, meas, static_pointer_cast<XDriver>(shared_from_this()))) {

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(videoMode(), m_form->m_cmbVideomode, Snapshot( *videoMode())),
        xqcon_create<XQComboBoxConnector>(frameRate(), m_form->m_cmbFrameRate, Snapshot( *frameRate())),
        xqcon_create<XQComboBoxConnector>(triggerMode(), m_form->m_cmbTrigger, Snapshot( *triggerMode())),
        xqcon_create<XQComboBoxConnector>(coloringMethod(), m_form->m_cmbColoringMethod, Snapshot( *coloringMethod())),
        xqcon_create<XQSpinBoxUnsignedConnector>(average(), m_form->m_spbAverage),
        xqcon_create<XQLineEditConnector>(exposureTime(), m_form->m_edExposure),
        xqcon_create<XQSpinBoxUnsignedConnector>(brightness(), m_form->m_spbBrightness),
        xqcon_create<XQSpinBoxUnsignedConnector>(colorIndex(), m_form->m_spbColorIndex),
        xqcon_create<XQDoubleSpinBoxConnector>(gainForAverage(), m_form->m_dblGainProcessed),
//        xqcon_create<XQLineEditConnector>((), m_form->m_edIntegrationTime),
        xqcon_create<XQButtonConnector>(m_clearAverage, m_form->m_btnClearAverage),
        xqcon_create<XQButtonConnector>(storeDark(), m_form->m_btnStoreDark),
        xqcon_create<XQButtonConnector>(m_roiSelectionTool, m_form->m_tbROI),
        xqcon_create<XQToggleButtonConnector>(subtractDark(), m_form->m_ckbSubtractDark),
        xqcon_create<XQToggleButtonConnector>(m_incrementalAverage, m_form->m_ckbIncrementalAverage),
        xqcon_create<XQToggleButtonConnector>(m_autoGainForAverage, m_form->m_ckbAutoGainProcessed)
    };

    std::vector<shared_ptr<XNode>> runtime_ui{
        brightness(),
        exposureTime(),
        videoMode(),
        triggerMode(),
        colorIndex(),
        frameRate(),
        m_roiSelectionTool,
    };
    iterate_commit([=](Transaction &tr){
        tr[ *triggerMode()].add({"Continueous", "Single-shot", "Ext. Pos. Edge", "Ext. Neg. Edge", "Ext. Pos. Exposure", "Ext. Neg. Exposure"});
        tr[ *coloringMethod()].add({"Monochrome", "RGB Wheel", "DeltaPL/PL"});
        tr[ *coloringMethod()] = 0;
        tr[ *average()] = 1;
        tr[ *autoGainForAverage()] = true;
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
        m_lsnOnClearAverageTouched = tr[ *clearAverage()].onTouch().connectWeakly(
            shared_from_this(), &XDigitalCamera::onClearAverageTouched, Listener::FLAG_MAIN_THREAD_CALL);
        m_lsnOnStoreDarkTouched = tr[ *storeDark()].onTouch().connectWeakly(
            shared_from_this(), &XDigitalCamera::onStoreDarkTouched, Listener::FLAG_MAIN_THREAD_CALL);
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
XDigitalCamera::onTriggerModeChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setTriggerMode(static_cast<TriggerMode>((unsigned int)shot[ *triggerMode()]));
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
XDigitalCamera::onExposureTimeChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setExposureTime(shot[ *exposureTime()]);
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XDigitalCamera::onROISelectionToolTouched(const Snapshot &shot, XTouchableNode *) {
    m_lsnOnROISelectionToolFinished = m_form->m_graphwidgetLive->onPlaneSelectedByTool().connectWeakly(
        shared_from_this(), &XDigitalCamera::onROISelectionToolFinished);
    m_form->m_graphwidgetLive->activatePlaneSelectionTool(XAxis::AxisDirection::X, XAxis::AxisDirection::Y,
        "ROI");
}
void
XDigitalCamera::onROISelectionToolFinished(const Snapshot &shot, const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint>&res) {
    auto label = std::get<0>(res);
    auto src = std::get<1>(res);
    auto dst = std::get<2>(res);
    m_lsnOnROISelectionToolFinished.reset();
    try {
        setVideoMode(Snapshot( *this)[ *videoMode()], std::min(src.x, dst.x), std::min(src.y, dst.y),
                abs(src.x - dst.x), abs(src.y - dst.y));
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
          unsigned int cidx = shot[ *this].m_colorIndex + 1u;
          tr[ *m_colorIndex] = (cidx > shot[ *this].m_maxColorIndex) ? 0 : cidx;

          tr[ *m_liveImage->graph()->osdStrings()] = shot[ *this].m_status;
          m_liveImage->updateImage(tr, shot[ *this].liveImage());

          if(cidx > shot[ *this].m_maxColorIndex) {
              std::vector<double> coeffs;
              std::vector<const uint32_t *> rawimages;
              for(unsigned int cidx = 0; cidx <= shot[ *this].m_maxColorIndex; ++cidx) {
                  coeffs.push_back(shot[ *this].m_coefficients[cidx]);
                  rawimages.push_back( &shot[ *this].m_summedCounts[cidx]->at(0));
              }
              if(shot[ *this].m_darkCounts) {
                  coeffs.push_back(shot[ *this].m_darkCoefficient);
                  rawimages.push_back( &shot[ *this].m_darkCounts->at(0));
              }
              tr[ *m_processedImage->graph()->osdStrings()] = formatString("Avg:%u", (unsigned int)shot[ *this].accumulated());
              m_processedImage->updateImage(tr, shot[ *this].processedImage(), rawimages, coeffs);
          }
      });
}

local_shared_ptr<std::vector<uint32_t>>
XDigitalCamera::summedCountsFromPool(int imagesize) {
    local_shared_ptr<std::vector<uint32_t>> summedCountsNext, p;
//    for(int i = 0; i < NumSummedCountsPool; ++i) {
//        if( !m_summedCountsPool[i])
//            m_summedCountsPool[i] = make_local_shared<std::vector<uint32_t>>(imagesize);
//        p = m_summedCountsPool[i];
//        if(p.use_count() == 2) { //not owned by other threads.
//            summedCountsNext = p;
//            p->resize(imagesize);
//        }
//    }
    if( !summedCountsNext)
        summedCountsNext = make_local_shared<std::vector<uint32_t>>(imagesize);
    return summedCountsNext;
}

void
XDigitalCamera::setGrayImage(RawDataReader &reader, Transaction &tr, uint32_t width, uint32_t height, bool big_endian, bool mono16) {
    auto summedCountsNext = summedCountsFromPool(width * height);
    unsigned int cidx = tr[ *colorIndex()];
    unsigned int max_cidx = std::map<ColoringMethod, unsigned int>{{ColoringMethod::MONO, 0}, {ColoringMethod::RGBWHEEL, 2}, {ColoringMethod::DPL_PL, 1}}
        .at(static_cast<ColoringMethod>((unsigned int)tr[ *coloringMethod()]));
    tr[ *this].m_maxColorIndex = max_cidx;
    if(cidx > max_cidx) {
        cidx = 0;
        tr[ *this].m_colorIndex = cidx;
        m_clearAverageInvoked = true;
        throw XSkippedRecordError(__FILE__, __LINE__);
    }
    tr[ *this].m_colorIndex = cidx;
    if( !tr[ *incrementalAverage()] && !m_clearAverageInvoked)
        for(unsigned int i = 0; i <= tr[ *this].m_maxColorIndex; ++i) {
            m_clearAverageInvoked = true;
            if(tr[ *average()] > tr[ *this].accumulated(i))
                m_clearAverageInvoked = false;
        }
    for(unsigned int i = 0; i <= tr[ *this].m_maxColorIndex; ++i) {
        if( !tr[ *this].m_summedCounts[i] || (tr[ *this].m_summedCounts[i]->size() != width * height)) {
            tr[ *this].m_summedCounts[i] = make_local_shared<std::vector<uint32_t>>(width * height, 0);
            m_clearAverageInvoked = true;
        }
    }
    if(m_clearAverageInvoked) {
        m_clearAverageInvoked = false;
        for(unsigned int i = 0; i <= tr[ *this].m_maxColorIndex; ++i) {
            std::fill(tr[ *this].m_summedCounts[i]->begin(), tr[ *this].m_summedCounts[i]->end(), 0);
            tr[ *this].m_accumulated[i] = 0;
        }
    }
    tr[ *this].m_electric_dark = 0.0; //unsupported
    uint32_t *summedNext = &summedCountsNext->at(0);
    const uint32_t *summed = &tr[ *this].m_summedCounts[cidx]->at(0);
    auto liveimage = std::make_shared<QImage>(width, height, QImage::Format_Grayscale16);
    uint16_t *words = reinterpret_cast<uint16_t*>(liveimage->bits());
    uint32_t vmin = 0xffffffffu;
    uint32_t vmax = 0u;
    uint32_t gain_disp = lrint(0x100u * tr[ *gainForAverage()]);
    if(mono16) {
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
             uint32_t w = gray * gain_disp / 0x100u; //16bitFS if G=1
             if(w >= 0x10000u)
                 w = 0xffffu;
             *words++ = w;
             uint64_t v = gray + *summed++; //raw values w/o gain
             if(v > 0x100000000uLL)
                 v = 0xffffffffuL;
             if(v > vmax)
                 vmax = v;
             if(v < vmin)
                 vmin = v;
             *summedNext++ = v;
        }
    }
    else {
        for(unsigned int i  = 0; i < width * height; ++i) {
             auto gray = reader.pop<uint8_t>();
             uint32_t w = gray * gain_disp / 0x100u;
             if(w >= 0x10000u)
                 w = 0xffffu;
             *words++ = w;
             uint64_t v = gray + *summed++;
             if(v > 0x100000000uLL)
                 v = 0xffffffffuL;
             if(v > vmax)
                 vmax = v;
             if(v < vmin)
                 vmin = v;
             *summedNext++ = v;
        }
    }
    assert(words == reinterpret_cast<uint16_t*>(liveimage->bits()) + width * height);
    assert(summedNext == &summedCountsNext->at(0) + width * height);
    assert(summed == &tr[ *this].m_summedCounts[cidx]->at(0) + width * height);
//    fprintf(stderr, "gain %u vmin:vmax %u %u\n", gain_disp, vmin, vmax);
    (tr[ *this].m_accumulated[cidx])++;
    tr[ *this].m_summedCounts[cidx] = summedCountsNext;
    tr[ *this].m_liveImage = liveimage;

    if(m_storeDarkInvoked) { // && (cidx == 0)
        tr[ *this].m_darkCounts = make_local_shared<std::vector<uint32_t>>( *summedCountsNext);
        tr[ *this].m_darkAccumulated = tr[ *this].accumulated(cidx);
        m_storeDarkInvoked = false;
    }

    //Waiting for all colors
    if(cidx != max_cidx)
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.

    auto processedimage = std::make_shared<QImage>(width, height,
        (tr[ *coloringMethod()] == (unsigned int)ColoringMethod::MONO) ?  QImage::Format_Grayscale16 : QImage::Format_RGBA8888);
    tr[ *this].m_processedImage = processedimage;
    if( !tr[ *subtractDark()] || !tr[ *this].m_darkCounts || (tr[ *this].m_darkCounts->size() != width * height)) {
        tr[ *this].m_darkCounts.reset();
    }
    if(tr[ *m_autoGainForAverage]) {
        tr[ *gainForAverage()]  = lrint((double)0xffffu / (vmax / tr[ *this].m_accumulated[cidx]));
    }
    switch((unsigned int)tr[ *coloringMethod()]) {
    case (unsigned int)ColoringMethod::MONO: {
        uint16_t *processed = reinterpret_cast<uint16_t*>(processedimage->bits());
        tr[ *this].m_coefficients[0] = 1.0 / tr[ *this].m_accumulated[cidx];
        uint64_t gain_av = lrint(0x100000000uLL * tr[ *gainForAverage()] * tr[ *this].m_coefficients[cidx]);
        summed = &tr[ *this].m_summedCounts[cidx]->at(0);
        if( !tr[ *this].m_darkCounts) {
            for(unsigned int i  = 0; i < width * height; ++i) {
                uint64_t v = (*summed++ * gain_av)  / 0x100000000uLL;
                *processed++ = std::min(v, 0xffffuLL);
            }
        }
        else {
            const uint32_t *dark = &tr[ *this].m_darkCounts->at(0);
            tr[ *this].m_darkCoefficient = 1.0 / tr[ *this].m_darkAccumulated;
            uint64_t gain_dark = lrint(0x100000000uLL * tr[ *gainForAverage()] * tr[ *this].m_darkCoefficient);
            for(unsigned int i  = 0; i < width * height; ++i) {
                int64_t v = (int64_t)(*summed++ * gain_av - *dark++ * gain_dark)  / 0x100000000LL;
                *processed++ = std::max((int64_t)0, std::min(v, 0xffffLL));
            }
            assert(dark == &tr[ *this].m_darkCounts->at(0) + width * height);
        }
        assert(summed == &tr[ *this].m_summedCounts[0]->at(0) + width * height);
        assert((uint8_t*)processed == processedimage->constBits() + width * height * 2);
        break;
    }
    case (unsigned int)ColoringMethod::RGBWHEEL: {
        uint8_t *processed = reinterpret_cast<uint8_t*>(processedimage->bits());
        uint64_t gain_av[3];
        const uint32_t *summed[3];
        for(unsigned int cidx: {0,1,2}) {
            tr[ *this].m_coefficients[cidx] = 1.0 / tr[ *this].m_accumulated[cidx];
            gain_av[cidx] = lrint(0x100000000uLL * tr[ *gainForAverage()] / 256.0 * tr[ *this].m_coefficients[cidx]);
            summed[cidx] = &tr[ *this].m_summedCounts[cidx]->at(0);
        }
        if( !tr[ *this].m_darkCounts) {
            for(unsigned int i  = 0; i < width * height; ++i) {
                for(unsigned int cidx: {0,1,2}) {
                    uint64_t v = (*(summed[cidx])++ * gain_av[cidx])  / 0x100000000uLL;
                    *processed++ = std::min(v, 0xffuLL);
                }
                *processed++ = 0xffu;
            }
        }
        else {
            const uint32_t *dark = &tr[ *this].m_darkCounts->at(0);
            tr[ *this].m_darkCoefficient = 1.0 / tr[ *this].m_darkAccumulated;
            uint64_t gain_dark = lrint(0x100000000uLL * tr[ *gainForAverage()] / 256.0 * tr[ *this].m_darkCoefficient);
            for(unsigned int i  = 0; i < width * height; ++i) {
                for(unsigned int cidx: {0,1,2}) {
                    int64_t v = (int64_t)(*(summed[cidx])++ * gain_av[cidx] - *dark * gain_dark)  / 0x100000000LL;
                    *processed++ = std::max((int64_t)0, std::min(v, 0xffLL));
                }
                *processed++ = 0xffu;
                dark++;
            }
            assert(dark == &tr[ *this].m_darkCounts->at(0) + width * height);
        }
        assert(processed == processedimage->constBits() + width * height * 4);
        assert(summed[0] == &tr[ *this].m_summedCounts[0]->at(0) + width * height);
        assert(summed[1] == &tr[ *this].m_summedCounts[1]->at(0) + width * height);
        assert(summed[2] == &tr[ *this].m_summedCounts[2]->at(0) + width * height);
        break;
    }
    case (unsigned int)ColoringMethod::DPL_PL: {
        uint8_t *processed = reinterpret_cast<uint8_t*>(processedimage->bits());
        uint64_t gain_av[2];
        const uint32_t *summed[2];
        for(unsigned int cidx: {0,1}) {
            tr[ *this].m_coefficients[cidx] = 1.0 / tr[ *this].m_accumulated[cidx];
            summed[cidx] = &tr[ *this].m_summedCounts[cidx]->at(0);
        }
        const unsigned int *accum = tr[ *this].m_accumulated;
        int64_t dpl_min = 0x100000000000LL, dpl_max = 0;
        for(unsigned int i  = 0; i < width * height; ++i) {
            int64_t dpl = (int64_t)(*(summed[1])++ * accum[0] - *(summed[0])++ * accum[1]);
            if(dpl > dpl_max)
                dpl_max = dpl;
            if(dpl < dpl_min)
                dpl_min = dpl;
        }
        assert(summed[0] == &tr[ *this].m_summedCounts[0]->at(0) + width * height);
        assert(summed[1] == &tr[ *this].m_summedCounts[1]->at(0) + width * height);
//        dpl_max /= accum[0] * accum[1];// max(PL[1] - PL[0])
//        dpl_min /= accum[0] * accum[1];
        for(unsigned int cidx: {0,1}) {
            gain_av[cidx] = lrint(0x100000000uLL * tr[ *gainForAverage()] / 256.0 * tr[ *this].m_coefficients[cidx]);
            gain_av[cidx] /= 2;
        }
        for(unsigned int cidx: {0,1})
            summed[cidx] = &tr[ *this].m_summedCounts[cidx]->at(0);
        int64_t dpl_gain[3] = {};
        if(std::max(dpl_max, -dpl_min)) {
            dpl_gain[0] = lrint(0x100000000LL * 1.0 / std::max(dpl_max, -dpl_min));
//            dpl_gain[0] = lrint(128.0 / std::max(dpl_max, -dpl_min) / accum[0] / accum[1]);
            dpl_gain[1] = 0;
            dpl_gain[2] = -dpl_gain[0];
        }
        if( !tr[ *this].m_darkCounts) {
            for(unsigned int i  = 0; i < width * height; ++i) {
                int64_t dpl = (int64_t)*summed[1] * accum[0] - (int64_t)*summed[0] * accum[1];
                for(unsigned int cidx: {0,1,2}) {
                    int64_t v = (*summed[0] * gain_av[0] + dpl * dpl_gain[cidx])  / 0x100000000LL;
                    *processed++ = std::min(v, 0xffLL);
                }
                *processed++ = 0xffu;
                for(unsigned int cidx: {0,1})
                    (summed[cidx])++;
            }
        }
        else {
            const uint32_t *dark = &tr[ *this].m_darkCounts->at(0);
            tr[ *this].m_darkCoefficient = 1.0 / tr[ *this].m_darkAccumulated;
            uint64_t gain_dark = lrint(0x100000000uLL * tr[ *gainForAverage()] / 256.0 * tr[ *this].m_darkCoefficient);
            gain_dark /= 2;
            for(unsigned int i  = 0; i < width * height; ++i) {
                int64_t dpl = (int64_t)(*summed[1] * accum[0] - *summed[0] * accum[1]);
                int64_t green = (int64_t)(*summed[0] * gain_av[0] - *dark * gain_dark)  / 0x100000000LL;
                for(unsigned int cidx: {0,1,2}) {
                    int64_t v = green + (green * dpl * dpl_gain[cidx]) / 0x100000000LL;
                    *processed++ = std::max((int64_t)0, std::min(v, 0xffLL));
                }
                *processed++ = 0xffu;
                for(unsigned int cidx: {0,1})
                    (summed[cidx])++;
                dark++;
            }
            assert(dark == &tr[ *this].m_darkCounts->at(0) + width * height);
        }
        assert(processed == processedimage->constBits() + width * height * 4);
        assert(summed[0] == &tr[ *this].m_summedCounts[0]->at(0) + width * height);
        assert(summed[1] == &tr[ *this].m_summedCounts[1]->at(0) + width * height);
        break;
    }
    }

    if(tr[ *incrementalAverage()]) {
        tr[ *average()] = tr[ *this].m_accumulated[cidx];
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
    else {
        if(tr[ *average()] > tr[ *this].m_accumulated[cidx])
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
}

void *
XDigitalCamera::execute(const atomic<bool> &terminated) {

    std::vector<shared_ptr<XNode>> runtime_ui{
        brightness(),
        exposureTime(),
        videoMode(),
        triggerMode(),
        colorIndex(),
        frameRate(),
        m_roiSelectionTool,
    };

    m_storeDarkInvoked = false;

	iterate_commit([=](Transaction &tr){
        m_lsnOnVideoModeChanged = tr[ *videoMode()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onVideoModeChanged);
        m_lsnOnTriggerModeChanged = tr[ *triggerMode()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onTriggerModeChanged);
        m_lsnOnBrightnessChanged = tr[ *brightness()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onBrightnessChanged);
        m_lsnOnExposureTimeChanged = tr[ *exposureTime()].onValueChanged().connectWeakly(
                    shared_from_this(), &XDigitalCamera::onExposureTimeChanged);
        m_lsnOnROISelectionToolTouched = tr[ *m_roiSelectionTool].onTouch().connectWeakly(
            shared_from_this(), &XDigitalCamera::onROISelectionToolTouched, Listener::FLAG_MAIN_THREAD_CALL);
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(true);
    });

    XTime time_awared = XTime::now();
    XTime time;
    while( !terminated) {
		auto writer = std::make_shared<RawData>();
		// try/catch exception of communication errors
		try {
            time = acquireRaw(writer);
        }
		catch (XDriver::XSkippedRecordError&) {
			msecsleep(10);
			continue;
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}
        finishWritingRaw(writer, time_awared, time);
        time_awared = time;
    }

    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_lsnOnBrightnessChanged.reset();
    m_lsnOnExposureTimeChanged.reset();
    m_lsnOnTriggerModeChanged.reset();
    m_lsnOnVideoModeChanged.reset();
    m_lsnOnROISelectionToolTouched.reset();
    return NULL;
}
