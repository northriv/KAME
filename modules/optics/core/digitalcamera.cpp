/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
#include "graphpainter.h"
#include "xnodeconnector.h"
#include "graphmathtool.h"
#include <QColorSpace>

XDigitalCamera::XDigitalCamera(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_cameraGain(create<XDoubleNode>("CameraGain", true)),
    m_emGain(create<XDoubleNode>("CameraGain", true)),
    m_blackLvlOffset(create<XUIntNode>("BlackLevelOffset", true)),
    m_exposureTime(create<XDoubleNode>("ExposureTime", true)),
    m_storeDark(create<XTouchableNode>("StoreDark", true)),
    m_roiSelectionTool(create<XTouchableNode>("ROISelectionTool", true)),
    m_antiShakePixels(create<XUIntNode>("AntiShakePixels", false)),
    m_subtractDark(create<XBoolNode>("SubtractDark", false)),
    m_videoMode(create<XComboNode>("VideoMode", true)),
    m_triggerMode(create<XComboNode>("TriggerMode", true)),
    m_triggerSrc(create<XComboNode>("TriggerSrc", true)),
    m_frameRate(create<XComboNode>("FrameRate", true)),
    m_autoGainForDisp(create<XBoolNode>("AutoGainForDisp", false)),
    m_gainForDisp(create<XDoubleNode>("GainForDisp", false)),
    m_form(new FrmDigitalCamera),
    m_liveImage(create<X2DImage>("LiveImage", false,
                                   m_form->m_graphwidgetLive,
                                   m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
                                   2, m_form->m_dblGamma,//w/dark
                                   m_form->m_tbMathMenu, meas, static_pointer_cast<XDriver>(shared_from_this()))),
    m_waveHist(create<XWaveNGraph>("WaveHistogram", true, m_form->m_graphwidgetHist, m_form->m_edHistDump, m_form->m_tbHistDump, m_form->m_btnHistDump)) {

    m_form->setWindowTitle(i18n("Digital Camera - ") + getLabel() );
    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(videoMode(), m_form->m_cmbVideomode, Snapshot( *videoMode())),
        xqcon_create<XQComboBoxConnector>(frameRate(), m_form->m_cmbFrameRate, Snapshot( *frameRate())),
        xqcon_create<XQComboBoxConnector>(triggerMode(), m_form->m_cmbTrigger, Snapshot( *triggerMode())),
        xqcon_create<XQComboBoxConnector>(triggerSrc(), m_form->m_cmbTriggerSrc, Snapshot( *triggerSrc())),
        xqcon_create<XQLineEditConnector>(exposureTime(), m_form->m_edExposure),
        xqcon_create<XQDoubleSpinBoxConnector>(cameraGain(), m_form->m_dblCameraGain),
        xqcon_create<XQDoubleSpinBoxConnector>(emGain(), m_form->m_dblEMGain),
        xqcon_create<XQSpinBoxUnsignedConnector>(blackLvlOffset(), m_form->m_spbBrightness),
        xqcon_create<XQSpinBoxUnsignedConnector>(antiShakePixels(), m_form->m_spbAntiVibrationPixels),
        xqcon_create<XQDoubleSpinBoxConnector>(gainForDisp(), m_form->m_dblGainProcessed),
        xqcon_create<XQButtonConnector>(storeDark(), m_form->m_btnStoreDark),
        xqcon_create<XQButtonConnector>(m_roiSelectionTool, m_form->m_tbROI),
        xqcon_create<XQToggleButtonConnector>(subtractDark(), m_form->m_ckbSubtractDark),
        xqcon_create<XQToggleButtonConnector>(m_autoGainForDisp, m_form->m_ckbAutoGainProcessed)
    };

    std::vector<shared_ptr<XNode>> runtime_ui{
        cameraGain(),
        emGain(),
        blackLvlOffset(),
        exposureTime(),
        videoMode(),
        triggerMode(),
        triggerSrc(),
        frameRate(),
        m_roiSelectionTool,
    };
    iterate_commit([=](Transaction &tr){
        tr[ *triggerMode()].add({"Continueous", "Single-shot", "Ext. Pos. Edge", "Ext. Neg. Edge", "Ext. Pos. Exposure", "Ext. Neg. Exposure"});
        tr[ *autoGainForDisp()] = true;
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
        m_lsnOnStoreDarkTouched = tr[ *storeDark()].onTouch().connectWeakly(
            shared_from_this(), &XDigitalCamera::onStoreDarkTouched);
        m_lsnOnAntiShakeChanged = tr[ *antiShakePixels()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onAntiShakeChanged);

        const char *labels[] = {"Intens.", "Counts"};
        tr[ *m_waveHist].setColCount(2, labels);
        if( !tr[ *m_waveHist].insertPlot(tr, i18n("Histogram"), 0, 1, -1)) return;
        shared_ptr<XAxis> axisx = tr[ *m_waveHist].axisx();
        shared_ptr<XAxis> axisy = tr[ *m_waveHist].axisy();
        tr[ *m_waveHist->graph()->drawLegends()] = false;
        tr[ *axisx->label()] = i18n("Intens.");
        tr[ *axisy->label()] = i18n("Counts");
        tr[ *axisy->displayTicLabels()] = false;
        tr[ *axisy->displayLabel()] = false;
        tr[ *axisx->displayLabel()] = false;
        tr[ *axisx->displayMinorTics()] = false;
        tr[ *axisy->displayMinorTics()] = false;
        tr[ *axisx->displayMajorTics()] = false;
        tr[ *axisy->displayMajorTics()] = false;
        tr[ *tr[ *m_waveHist].plot(0)->drawBars()] = true;
        tr[ *tr[ *m_waveHist].plot(0)->drawPoints()] = false;
        tr[ *tr[ *m_waveHist].plot(0)->drawLines()] = false;
    });
}
void
XDigitalCamera::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XDigitalCamera::onAntiShakeChanged(const Snapshot &shot, XValueNodeBase *) {
    trans( *this).m_storeAntiShakeInvoked = true;
}
void
XDigitalCamera::onStoreDarkTouched(const Snapshot &shot, XTouchableNode *) {
    trans( *this).m_storeDarkInvoked = true;
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
XDigitalCamera::onTriggerSrcChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setTriggerSrc(Snapshot( *this));
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XDigitalCamera::onGainChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    try {
        setGain(shot[ *cameraGain()], shot[ *emGain()]);
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XDigitalCamera::onBlackLevelOffsetChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setBlackLevelOffset(shot[ *blackLvlOffset()]);
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
XDigitalCamera::onROISelectionToolFinished(const Snapshot &shot,
    const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint, XQGraph*>&res) {
    auto label = std::get<0>(res);
    auto src = std::get<1>(res);
    auto dst = std::get<2>(res);
    auto widget = std::get<3>(res);
    m_lsnOnROISelectionToolFinished.reset();

    //upside down for y axis. //todo fix yscale.
//    const XNode::NodeList &axes_list( *shot.list(widget->graph()->axes()));
//    auto axisy = static_pointer_cast<XAxis>(axes_list.at(1));
//    src.y = axisy->axisToVal(1.0 - axisy->valToAxis(src.y));
//    dst.y = axisy->axisToVal(1.0 - axisy->valToAxis(dst.y));

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
    if( !shot[ *this].time())
        return;
    uint32_t gain_disp = lrint(0x100u * shot[ *gainForDisp()]);
    const uint32_t *raw = &shot[ *this].m_rawCounts->at(shot[ *this].firstPixel());
    unsigned int stride = shot[ *this].stride();
    unsigned int width = shot[ *this].width();
    unsigned int height = shot[ *this].height();
    unsigned int edark = floor(shot[ *this].m_electric_dark);
    auto liveimage = std::make_shared<QImage>(width, height, QImage::Format_Grayscale16);
    liveimage->setColorSpace(QColorSpace::SRgbLinear);
    uint16_t *words = reinterpret_cast<uint16_t*>(liveimage->bits());
    if( !shot[ *this].m_darkCounts) {
        for(unsigned int y  = 0; y < height; ++y) {
            for(unsigned int x  = 0; x < width; ++x) {
                 uint32_t w = (*raw++ - edark) * gain_disp / 0x100u; //16bitFS if G=1
                 if(w >= 0x10000u)
                     w = 0xffffu;
                 *words++ = w;
            }
            raw += stride - width;
        }
    }
    else {
        const uint32_t *dark = &shot[ *this].m_darkCounts->at(shot[ *this].firstPixel());
        for(unsigned int y  = 0; y < height; ++y) {
            for(unsigned int x  = 0; x < width; ++x) {
                int32_t v = *raw++ - *dark++;
                if(v < 0) v = 0;
                uint32_t w = v * gain_disp / 0x100u; //16bitFS if G=1
                if(w >= 0x10000u)
                    w = 0xffffu;
                *words++ = w;
            }
            raw += stride - width;
            dark += stride - width;
        }
    }
    assert(words == reinterpret_cast<uint16_t*>(liveimage->bits()) + width * height);
    std::vector<double> coeffs = {1.0};
    std::vector<const uint32_t *> rawimages = { &shot[ *this].m_rawCounts->at(shot[ *this].firstPixel())};
    if(shot[ *this].m_darkCounts) {
        coeffs.push_back(1.0);
        rawimages.push_back(&shot[ *this].m_darkCounts->at(shot[ *this].firstPixel()));
    }
    iterate_commit([&](Transaction &tr){
      tr[ *m_liveImage->graph()->onScreenStrings()] = shot[ *this].m_status;
      tr[ *this].m_qimage = liveimage;
      m_liveImage->updateRawImages(tr, width, height, rawimages, stride, coeffs);
      m_liveImage->updateQImage(tr, liveimage);

      tr[ *m_waveHist].setLabel(0, "Histogram");
      double vmax = shot[ *this].m_maxIntensity;
      double vmin = shot[ *this].m_minIntensity;
      double vmode = shot[ *this].m_modeIntensity;
      tr[ *m_waveHist->graph()->onScreenStrings()] = formatString("Mode=%u, Min=%u, Max=%u",
        (unsigned int)vmode, (unsigned int)vmin, (unsigned int)vmax);
      size_t hist_length = shot[ *this].m_histogram.size();
      tr[ *m_waveHist].setRowCount(hist_length);
      std::vector<float> hist_x(hist_length), hist_y(hist_length);
      const auto &hist = shot[ *this].m_histogram;
      for(unsigned int i = 0; i < hist_length; ++i) {
          hist_x[i] = i * (vmax - vmin) / (hist_length - 1) + vmin;
          hist_y[i] = hist[i];
      }
      tr[ *m_waveHist].setColumn(0, std::move(hist_x));
      tr[ *m_waveHist].setColumn(1, std::move(hist_y));
      m_waveHist->drawGraph(tr);
    });
}

void
XDigitalCamera::setGrayImage(RawDataReader &reader, Transaction &tr, uint32_t width, uint32_t height, bool big_endian, bool mono16) {
    auto rawCountsNext = m_pool.allocate(width * height);
    if( !tr[ *this].m_rawCounts || (tr[ *this].m_rawCounts->size() != width * height)) {
        tr[ *this].m_rawCounts = make_local_shared<std::vector<uint32_t>>(width * height, 0);
    }
//    std::fill(tr[ *this].m_rawCounts->begin(), tr[ *this].m_rawCounts->end(), 0);

    uint32_t *rawNext = &rawCountsNext->at(0);
    uint16_t vmin = 0xffffu;
    uint16_t vmax = 0u;
    uint16_t vignore;
    constexpr unsigned int num_hist = 32;
    std::vector<uint32_t> hist_fullrange;
    constexpr unsigned int num_conv = 3;
    std::vector<uint32_t> dummy_lines(width + num_conv, 0); //for y < num_comv. results will not be used.
    uint32_t *raw_lines[num_conv];
    constexpr unsigned int kernel_len = 3; //cubic

    auto fn_prepare_prevlines = [&raw_lines, &dummy_lines, &rawCountsNext, width](unsigned int y) {
        if(y >= num_conv + 1) {
            for(unsigned int i = 0; i < num_conv; ++i)
                raw_lines[i] = &rawCountsNext->at(width * (y - num_conv + i + 1));
        }
        else {
            for(auto &&x : raw_lines)
                x = &dummy_lines[num_conv]; //for useless calc. and to avoid violation.
        }
    };
    auto fn_take_histogram = [&hist_fullrange, &vmin, &vmax](uint16_t v, uint64_t vsat) {
        if(v > vmax)
            vmax = v;
        if(v < vmin)
            vmin = v;
        ++hist_fullrange[(uint64_t)v * hist_fullrange.size() / vsat];
    };
    //rearrange histogram in range of [vmin, vmax].
    auto fn_finalize_histogram = [&](uint64_t vsat) {
        auto &hist = tr[ *this].m_histogram;
        hist.resize(num_hist);
        std::fill(hist.begin(), hist.end(), 0);
        for(unsigned int i = 0; i < hist_fullrange.size(); ++i) {
            size_t j = lrint(((double)i * vsat / hist_fullrange.size() - vmin) / (vmax - vmin) * hist.size());
            hist[std::min(hist.size() - 1, j)] += hist_fullrange[i];
        }
        uint64_t vmode = std::distance(hist_fullrange.begin(), std::max_element(hist_fullrange.begin(), hist_fullrange.end()));
        vmode = vmode * vsat / hist_fullrange.size();

        //Determines the threshold for anti-shake CoG.
        vignore = std::min(vmode + (vmax - vmode) / 3u, vmode * 3u);
//        uint32_t tot = 0;
//        for(int i = hist_fullrange.size() - 1; i >= 0; i--) {
//            tot += hist_fullrange[i];
//            if(tot > width * height / 10) {
//                vignore = (uint64_t)i * vsat / hist_fullrange.size(); //intensity at top 10%.
//                break;
//            }
//        }

        tr[ *this].m_modeIntensity = vmode;
        tr[ *this].m_maxIntensity = vmax;
        tr[ *this].m_minIntensity = vmin;
    };
    if(mono16) {
        hist_fullrange.resize(0x10000u, 0);
        for(unsigned int y  = 0; y < height; ++y) {
            fn_prepare_prevlines(y);
            for(unsigned int x  = 0; x < width; ++x) {
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
                 *rawNext++ = gray;
                 fn_take_histogram(gray, 0x10000u);
            }
        }
        fn_finalize_histogram(0x10000u);
    }
    else {
        hist_fullrange.resize(0x100u, 0);
        for(unsigned int y  = 0; y < height; ++y) {
            fn_prepare_prevlines(y);
            for(unsigned int x  = 0; x < width; ++x) {
                 auto gray = reader.pop<uint8_t>();
                 *rawNext++ = gray;
                 fn_take_histogram(gray, 0x100u);
            }
        }
        fn_finalize_histogram(0x100u);
    }

    assert(rawNext == &rawCountsNext->at(0) + width * height);
//    fprintf(stderr, "gain %u vmin:vmax %u %u\n", gain_disp, vmin, vmax);
    tr[ *this].m_rawCounts = rawCountsNext;
    tr[ *this].m_electric_dark = 0;
    if(tr[ *this].m_storeDarkInvoked) { // && (cidx == 0)
        tr[ *this].m_storeDarkInvoked = false;
        tr[ *this].m_darkCounts = make_local_shared<std::vector<uint32_t>>( *rawCountsNext);
    }
    else if(tr[ *subtractDark()]) {
        tr[ *this].m_electric_dark = vmin;
    }
    if( !tr[ *subtractDark()] || !tr[ *this].m_darkCounts || (tr[ *this].m_darkCounts->size() != width * height)) {
        tr[ *this].m_darkCounts.reset();
    }
    if(tr[ *m_autoGainForDisp]) {
        //todo auto gain using stored dark
        tr[ *gainForDisp()]  = lrint((double)0xffffu / (vmax - floor(tr[ *this].m_electric_dark)));
    }
    int32_t antishake_pixels = tr[ *m_antiShakePixels];
    if(tr[ *this].m_storeAntiShakeInvoked) { // && (cidx == 0)
        //Setting has been changed by user.
        tr[ *this].m_antishake_pixels = antishake_pixels;
        if(antishake_pixels > 0) {
            if(std::min(width, height) < tr[ *m_antiShakePixels] * 16)
                throw XSkippedRecordError(i18n("Too many pixels for antishaking."), __FILE__, __LINE__);
        }
    }
    tr[ *this].m_stride = width;
    tr[ *this].m_width = width;
    tr[ *this].m_height = height;
    if(antishake_pixels) {
        uint64_t cogx = 0u, cogy = 0u, toti = 0u;
        {
            //ignores pixels darker than vignore value.
            uint32_t *raw = &tr[ *this].m_rawCounts->at(0);
            raw = &tr[ *this].m_rawCounts->at(0);
            for(unsigned int y = 0; y < height; ++y) {
                for(unsigned int x  = 0; x < width; ++x) {
                   uint64_t v = *raw++;
                   v = std::max((int64_t)0, (int64_t)v - (int64_t)vignore);
                   cogx += v * x;
                   cogy += v * y;
                   toti += v;
                }
            }
        }
        if(tr[ *this].m_storeAntiShakeInvoked) {
            //stores original image info. before shake.
            tr[ *this].m_cogXOrig = (double)cogx / toti;
            tr[ *this].m_cogYOrig = (double)cogy / toti;
            tr[ *this].m_storeAntiShakeInvoked = false;
        }

        const int max_pixelshift_bycog = antishake_pixels;
        tr[ *this].m_stride = width;
        const int pixels_skip = max_pixelshift_bycog + (kernel_len - 1) / 2;
        tr[ *this].m_width = width - 2 * pixels_skip;
        tr[ *this].m_height = height - 2 * pixels_skip;

        //finds the pixels shifts
        double shift_dx = (double)cogx / toti - tr[ *this].m_cogXOrig;
        int shift_x = lrint(shift_dx);
        double shift_dy = (double)cogy / toti - tr[ *this].m_cogYOrig;
        int shift_y = lrint(shift_dy);
        shift_dx -= shift_x;
        shift_dy -= shift_y;
        shift_x = std::min(shift_x, max_pixelshift_bycog);
        shift_x = std::max(shift_x, -max_pixelshift_bycog);
        shift_y = std::min(shift_y, max_pixelshift_bycog);
        shift_y = std::max(shift_y, -max_pixelshift_bycog);
        tr[ *this].m_status += formatString(" anti-shake (%+5.1f, %+5.1f)", shift_x + shift_dx, shift_y + shift_dy);
        tr[ *this].m_firstPixel = (shift_y + pixels_skip) * width + shift_x + pixels_skip; //(0,0) origin for the secondary drivers.
//        const auto &edge_orig = tr[ *this].m_edgesOrig;
        {
            //bi-cubic interpolation
            std::vector<int64_t> kernel(kernel_len * kernel_len);
            constexpr double a = -0.2;
            for(unsigned int ky = 0; ky < kernel_len; ++ky) {
                for(unsigned int kx = 0; kx < kernel_len; ++kx) {
                    double x = pow(kx - shift_dx - (kernel_len - 1)/2, 2.0) + pow(ky - shift_dy - (kernel_len - 1)/2, 2.0);
                    x = sqrt(x);
                    if(x < 1.0)
                        x = (a+2)*x*x*x - (a+3)*x*x + 1;
                    else if(x < 2)
                        x = a*x*x*x -5*a*x*x + 8*a*x - 4*a;
                    else
                        x = 0.0;
                    kernel[ky * kernel_len + kx] = llrint(x * 0x100000000LL * 0x100LL);
                }
            }
            double frac = (double)0x100000000LL / std::accumulate(kernel.begin(), kernel.end(), 0LL);
            for(auto &x: kernel)
                x = llrint(x * frac);
            {
                //convolution.
                uint32_t *raw = &tr[ *this].m_rawCounts->at(tr[ *this].firstPixel());
                unsigned int stride = tr[ *this].stride();
                unsigned int width = tr[ *this].width();
                unsigned int height = tr[ *this].height();
                //copies (-N/2, -N/2) to (width + N/2, N/2) pixels for convolution, not to be overwritten by new values.
                std::vector<uint32_t> cache_orig_lines[kernel_len];
                for(int k = 0; k < kernel_len; ++k) {
                    cache_orig_lines[k].resize(width + kernel_len - 1);
                    const uint32_t *bg = raw - (kernel_len - 1)/2 + (k - (int)(kernel_len - 1)/2) * (int)stride;
                    assert(bg >= &tr[ *this].m_rawCounts->at(0));
                    std::copy(bg, bg + cache_orig_lines[k].size(), &cache_orig_lines[k][0]);
                }
                int64_t sum = 0;
                uint32_t *raw_x0 = raw;
                for(unsigned int y = 0; y < height; ++y) {
                    for(unsigned int x  = 0; x < width; ++x) {
                        const int64_t *pk = &kernel[0];
                        for(unsigned int ky = 0; ky < kernel_len; ++ky) {
                            const uint32_t *v = &cache_orig_lines[ky][x];
                            for(unsigned int kx = 0; kx < kernel_len; ++kx) {
                                sum += (uint64_t)*v++ * *pk++;
                            }
                        }
                        uint32_t w = sum / 0x100000000uLL;
                        sum = sum % 0x100000000uLL; //leaves residual.
                        *raw++ = w;
                    }
                    raw_x0 += stride;
                    raw = raw_x0;
                    if(y < height - 1) { //no thank you at the last line.
                        //slides cached values and retrieves original values.
                        for(unsigned int k = 0; k < kernel_len - 1; ++k) {
                            std::copy(cache_orig_lines[k + 1].begin(), cache_orig_lines[k + 1].end(), &cache_orig_lines[k][0]);
                        }
                        const uint32_t *bg = raw - (kernel_len - 1)/2 + (kernel_len - 1 - (int)(kernel_len - 1)/2) * (int)stride;
                        assert(bg + cache_orig_lines[0].size() <= &tr[ *this].m_rawCounts->at(0) + (height + 2*pixels_skip) * stride);
                        std::copy(bg, bg + cache_orig_lines[0].size(), &cache_orig_lines[kernel_len - 1][0]);
                    }
                }
            }
        }
    }
    else {
         tr[ *this].m_firstPixel = 0;
         tr[ *this].m_storeAntiShakeInvoked = false;
    }
//    case (unsigned int)ColoringMethod::RGBWHEEL: {
//        uint8_t *processed = reinterpret_cast<uint8_t*>(processedimage->bits());
//        uint64_t gain_av[3];
//        const uint32_t *summed[3];
//        for(unsigned int cidx: {0,1,2}) {
//            tr[ *this].m_coefficients[cidx] = 1.0 / tr[ *this].m_accumulated[cidx];
//            gain_av[cidx] = lrint(0x100000000uLL * tr[ *gainForDisp()] / 256.0 * tr[ *this].m_coefficients[cidx]);
//            summed[cidx] = &tr[ *this].m_summedCounts[cidx]->at(0);
//        }
//        if( !tr[ *this].m_darkCounts) {
//            for(unsigned int i  = 0; i < width * height; ++i) {
//                for(unsigned int cidx: {0,1,2}) {
//                    uint64_t v = (*(summed[cidx])++ * gain_av[cidx])  / 0x100000000uLL;
//                    *processed++ = std::min(v, 0xffuLL);
//                }
//                *processed++ = 0xffu;
//            }
//        }
//        else {
//            const uint32_t *dark = &tr[ *this].m_darkCounts->at(0);
//            tr[ *this].m_darkCoefficient = 1.0 / tr[ *this].m_darkAccumulated;
//            uint64_t gain_dark = lrint(0x100000000uLL * tr[ *gainForDisp()] / 256.0 * tr[ *this].m_darkCoefficient);
//            for(unsigned int i  = 0; i < width * height; ++i) {
//                for(unsigned int cidx: {0,1,2}) {
//                    int64_t v = (int64_t)(*(summed[cidx])++ * gain_av[cidx] - *dark * gain_dark)  / 0x100000000LL;
//                    *processed++ = std::max((int64_t)0, std::min(v, 0xffLL));
//                }
//                *processed++ = 0xffu;
//                dark++;
//            }
//            assert(dark == &tr[ *this].m_darkCounts->at(0) + width * height);
//        }
//        assert(processed == processedimage->constBits() + width * height * 4);
//        assert(summed[0] == &tr[ *this].m_summedCounts[0]->at(0) + width * height);
//        assert(summed[1] == &tr[ *this].m_summedCounts[1]->at(0) + width * height);
//        assert(summed[2] == &tr[ *this].m_summedCounts[2]->at(0) + width * height);
//        break;
//    }

}

void *
XDigitalCamera::execute(const atomic<bool> &terminated) {

    std::vector<shared_ptr<XNode>> runtime_ui{
        cameraGain(),
        emGain(),
        blackLvlOffset(),
        exposureTime(),
        videoMode(),
        triggerMode(),
        triggerSrc(),
        frameRate(),
        m_roiSelectionTool,
    };

    trans( *this).m_storeDarkInvoked = false;
    trans( *this).m_storeAntiShakeInvoked = false;

    afterOpen();

	iterate_commit([=](Transaction &tr){
        m_lsnOnVideoModeChanged = tr[ *videoMode()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onVideoModeChanged);
        m_lsnOnTriggerModeChanged = tr[ *triggerMode()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onTriggerModeChanged);
        m_lsnOnTriggerSrcChanged = tr[ *triggerSrc()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onTriggerSrcChanged);
        m_lsnOnGainChanged = tr[ *cameraGain()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onGainChanged);
        tr[ *emGain()].onValueChanged().connect(m_lsnOnGainChanged);
        m_lsnOnBlackLevelOffsetChanged = tr[ *blackLvlOffset()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onBlackLevelOffsetChanged);
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
        if(time.diff_sec(time_awared) > ***exposureTime() + 0.1) {
            //hack for ext trigger.
            time_awared = time;
            time_awared -= ***exposureTime() + 0.1;
        }
        finishWritingRaw(writer, time_awared, time);
        time_awared = time;
    }

    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_lsnOnGainChanged.reset();
    m_lsnOnBlackLevelOffsetChanged.reset();
    m_lsnOnExposureTimeChanged.reset();
    m_lsnOnTriggerModeChanged.reset();
    m_lsnOnTriggerSrcChanged.reset();
    m_lsnOnVideoModeChanged.reset();
    m_lsnOnROISelectionToolTouched.reset();
    return NULL;
}
