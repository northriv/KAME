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
#include "graphpainter.h"
#include "xnodeconnector.h"
#include "graphmathtool.h"
#include "tikhonovreg.h"

XDigitalCamera::XDigitalCamera(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_cameraGain(create<XDoubleNode>("CameraGain", true)),
    m_brightness(create<XUIntNode>("Brightness", true)),
    m_exposureTime(create<XDoubleNode>("ExposureTime", true)),
    m_storeDark(create<XTouchableNode>("StoreDark", true)),
    m_roiSelectionTool(create<XTouchableNode>("ROISelectionTool", true)),
    m_antiShakePixels(create<XUIntNode>("AntiShakePixels", false)),
    m_subtractDark(create<XBoolNode>("SubtractDark", false)),
    m_videoMode(create<XComboNode>("VideoMode", true)),
    m_triggerMode(create<XComboNode>("TriggerMode", true)),
    m_frameRate(create<XComboNode>("FrameRate", true)),
    m_autoGainForDisp(create<XBoolNode>("AutoGainForDisp", false)),
    m_colorIndex(create<XUIntNode>("ColorIndex", true)),
    m_gainForDisp(create<XDoubleNode>("GainForDisp", false)),
    m_form(new FrmDigitalCamera),
    m_liveImage(create<X2DImage>("LiveImage", false,
                                   m_form->m_graphwidgetLive,
                                   m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
                                   2, //w/dark
                                   m_form->m_tbMathMenu, meas, static_pointer_cast<XDriver>(shared_from_this()))) {


    m_form->setWindowTitle(i18n("Digital Camera - ") + getLabel() );
    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(videoMode(), m_form->m_cmbVideomode, Snapshot( *videoMode())),
        xqcon_create<XQComboBoxConnector>(frameRate(), m_form->m_cmbFrameRate, Snapshot( *frameRate())),
        xqcon_create<XQComboBoxConnector>(triggerMode(), m_form->m_cmbTrigger, Snapshot( *triggerMode())),
        xqcon_create<XQLineEditConnector>(exposureTime(), m_form->m_edExposure),
        xqcon_create<XQDoubleSpinBoxConnector>(cameraGain(), m_form->m_dblCameraGain),
        xqcon_create<XQSpinBoxUnsignedConnector>(brightness(), m_form->m_spbBrightness),
        xqcon_create<XQSpinBoxUnsignedConnector>(colorIndex(), m_form->m_spbColorIndex),
        xqcon_create<XQSpinBoxUnsignedConnector>(antiShakePixels(), m_form->m_spbAntiVibrationPixels),
        xqcon_create<XQDoubleSpinBoxConnector>(gainForDisp(), m_form->m_dblGainProcessed),
        xqcon_create<XQButtonConnector>(storeDark(), m_form->m_btnStoreDark),
        xqcon_create<XQButtonConnector>(m_roiSelectionTool, m_form->m_tbROI),
        xqcon_create<XQToggleButtonConnector>(subtractDark(), m_form->m_ckbSubtractDark),
        xqcon_create<XQToggleButtonConnector>(m_autoGainForDisp, m_form->m_ckbAutoGainProcessed)
    };

    std::vector<shared_ptr<XNode>> runtime_ui{
        cameraGain(),
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
        tr[ *autoGainForDisp()] = true;
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
        m_lsnOnStoreDarkTouched = tr[ *storeDark()].onTouch().connectWeakly(
            shared_from_this(), &XDigitalCamera::onStoreDarkTouched);
        m_lsnOnAntiShakeChanged = tr[ *antiShakePixels()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onAntiShakeChanged);
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
    if(shot[ *m_antiShakePixels] > 0)
        m_storeAntiShakeInvoked = true;
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
XDigitalCamera::onTriggerModeChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setTriggerMode(static_cast<TriggerMode>((unsigned int)shot[ *triggerMode()]));
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
void
XDigitalCamera::onCameraGainChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setCameraGain(shot[ *cameraGain()]);
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
XDigitalCamera::onROISelectionToolFinished(const Snapshot &shot,
    const std::tuple<XString, XGraph::ValPoint, XGraph::ValPoint, XQGraph*>&res) {
    auto label = std::get<0>(res);
    auto src = std::get<1>(res);
    auto dst = std::get<2>(res);
    auto widget = std::get<3>(res);
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
      iterate_commit([&](Transaction &tr){
          tr[ *m_liveImage->graph()->onScreenStrings()] = shot[ *this].m_status;
          if( !!shot[ *this].time()) {
              uint32_t gain_disp = lrint(0x100u * shot[ *gainForDisp()]);
              const uint32_t *raw = &shot[ *this].m_rawCounts->at(shot[ *this].firstPixel());
              unsigned int stride = shot[ *this].stride();
              unsigned int width = shot[ *this].width();
              unsigned int height = shot[ *this].height();
              auto liveimage = std::make_shared<QImage>(width, height, QImage::Format_Grayscale16);
              uint16_t *words = reinterpret_cast<uint16_t*>(liveimage->bits());
              if( !tr[ *this].m_darkCounts) {
                  for(unsigned int y  = 0; y < height; ++y) {
                      for(unsigned int x  = 0; x < width; ++x) {
                           uint32_t w = *raw++ * gain_disp / 0x100u; //16bitFS if G=1
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
              tr[ *this].m_qimage = liveimage;

              std::vector<double> coeffs = {1.0};
              std::vector<const uint32_t *> rawimages = { &shot[ *this].m_rawCounts->at(shot[ *this].firstPixel())};
              if(tr[ *this].m_darkCounts) {
                  coeffs.push_back(1.0);
                  rawimages.push_back(&shot[ *this].m_darkCounts->at(shot[ *this].firstPixel()));
              }
              m_liveImage->updateImage(tr, liveimage, rawimages, stride, coeffs);
          }
      });
}

local_shared_ptr<std::vector<uint32_t>>
XDigitalCamera::rawCountsFromPool(int imagesize) {
    local_shared_ptr<std::vector<uint32_t>> summedCountsNext, p;
//    for(int i = 0; i < NumSummedCountsPool; ++i) {
//        if( !m_summedCountsPool[i])
//            m_summedCountsPool[i] = make_local_shared<std::vector<uint16_t>>(imagesize);
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
    int32_t antishake_pixels = tr[ *m_antiShakePixels];
    if(antishake_pixels > 0) {
        if(std::min(width, height) < tr[ *m_antiShakePixels] * 10)
            throw XSkippedRecordError(i18n("Too many pixels for antishaking."), __FILE__, __LINE__);
    }

    auto rawCountsNext = rawCountsFromPool(width * height);
    unsigned int cidx = tr[ *colorIndex()];
    tr[ *this].m_colorIndex = cidx;
    if( !tr[ *this].m_rawCounts || (tr[ *this].m_rawCounts->size() != width * height)) {
        tr[ *this].m_rawCounts = make_local_shared<std::vector<uint32_t>>(width * height, 0);
    }
//    std::fill(tr[ *this].m_rawCounts->begin(), tr[ *this].m_rawCounts->end(), 0);

    tr[ *this].m_electric_dark = 0.0; //unsupported
    uint32_t *rawNext = &rawCountsNext->at(0);
    uint16_t vmin = 0xffffu;
    uint16_t vmax = 0u;
    uint64_t cogx = 0u, cogy = 0u, toti = 0u;
    constexpr unsigned int num_conv = 3;
    std::vector<uint32_t> dummy_lines(width + num_conv - 1, 0); //for y < num_comv. results will not be used.
    uint32_t *raw_prevlines[num_conv - 1];
    constexpr unsigned int num_conv_tsvd = 5;
    constexpr unsigned int num_edges = 20;
    struct Edge {
        unsigned int x, y; //center position.
        uint64_t sobel_norm; //norm2 of sobel filter.
    };
    std::deque<Edge> edges = {{0,0,0}};
    const Edge *edge_min = &edges.front();
    //stores prominent edge, which does not overwraps each other.
    auto fn_detect_edge = [&edges, &edge_min, &rawNext, &raw_prevlines, antishake_pixels](unsigned int x, unsigned int y) {
// kernel
//        sobel_x = [-1 0 1; -2 0 2; -1 0 1];
//        sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
        int64_t sobel_x = -(int64_t)*(raw_prevlines[0] - 2) + (int64_t)*raw_prevlines[0];
        sobel_x += 2 * (-(int64_t)*(raw_prevlines[1] - 2) + (int64_t)*raw_prevlines[1]);
        sobel_x += -(int64_t)*(rawNext - 2) + (int64_t)*rawNext;
        int64_t sobel_y = -(int64_t)*(raw_prevlines[0] - 2) - 2 * (int64_t)*(raw_prevlines[0] - 1) - (int64_t)*raw_prevlines[0];
        sobel_y += *(rawNext - 2) + 2 * *(rawNext - 1) + *rawNext;
//        //For kernel calc., caches x - 2 values into previous lines.
//        *(raw_prevlines[0] - 2) = *(raw_prevlines[1] - 2);
//        *(raw_prevlines[1] - 2) = *(rawNext - 2);
        uint64_t sobel_norm = sobel_x * sobel_x + sobel_y * sobel_y;
        if((edge_min->sobel_norm < sobel_norm) &&
            (x >= num_conv - 1 + antishake_pixels) && (y >= num_conv + 1 + antishake_pixels)) {
            if(edge_min->sobel_norm == 0)
                edges.clear(); //erases dummy edge.
            //abandon overwrapping edges.
            auto it = std::find_if(edges.begin(), edges.end(), [x, y](const auto &v){
                return ((v.x - (x - 1)) <= num_conv_tsvd) && ((v.y - (y - 1)) <= num_conv_tsvd);});
            bool isvalid = true;
            if(it != edges.end()) {
                if(it->sobel_norm < sobel_norm)
                    edges.erase(it);
                else
                    isvalid = false; //new edge is overwrapped and useless.
            }
            if(isvalid) {
                //edges be ascending in terms of sobel_norm.
                auto it = std::find_if(edges.begin(), edges.end(), [sobel_norm](const auto &x){return x.sobel_norm > sobel_norm;});
                edges.insert(it, {x - 1, y - 1, sobel_norm});
            }
            if(edges.size() > num_edges)
                edges.pop_front();
            edge_min = &edges.front();
        }
    };
    auto fn_prepare_prevlines = [&raw_prevlines, &dummy_lines, &rawCountsNext, width](unsigned int y) {
        if(y >= num_conv + 1) {
            raw_prevlines[0] = &rawCountsNext->at(width * (y - 2));
            raw_prevlines[1] = &rawCountsNext->at(width * (y - 1));
        }
        else {
            for(auto &&x : raw_prevlines)
                x = &dummy_lines[num_conv]; //for useless calc. and to avoid violation.
        }
    };
    auto fn_detect_min_max_cog = [&cogx, &cogy, &vmin, &vmax, &toti](uint16_t v, unsigned int x, unsigned int y) {
        if(v > vmax)
            vmax = v;
        if(v < vmin)
            vmin = v;
        toti += v;
        cogx += v * x;
        cogy += v * y;
    };
    if(mono16) {
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
                 *rawNext = gray;
                 fn_detect_min_max_cog(gray, x, y);
                 fn_detect_edge(x, y);
                 rawNext++;
            }
        }
    }
    else {
        for(unsigned int y  = 0; y < height; ++y) {
            fn_prepare_prevlines(y);
            for(unsigned int x  = 0; x < width; ++x) {
                 auto gray = reader.pop<uint8_t>();
                 *rawNext = gray;
                 fn_detect_min_max_cog(gray, x, y);
                 fn_detect_edge(x, y);
                 rawNext++;
            }
        }
    }
    if(tr[ *m_autoGainForDisp]) {
        tr[ *gainForDisp()]  = lrint((double)0xffffu / vmax);
    }
    assert(rawNext == &rawCountsNext->at(0) + width * height);
//    fprintf(stderr, "gain %u vmin:vmax %u %u\n", gain_disp, vmin, vmax);
    tr[ *this].m_rawCounts = rawCountsNext;
    if(m_storeDarkInvoked.compare_set_strong(true, false)) { // && (cidx == 0)
        tr[ *this].m_darkCounts = make_local_shared<std::vector<uint32_t>>( *rawCountsNext);
    }
    if( !tr[ *subtractDark()] || !tr[ *this].m_darkCounts || (tr[ *this].m_darkCounts->size() != width * height)) {
        tr[ *this].m_darkCounts.reset();
    }
    if(m_storeAntiShakeInvoked.compare_set_strong(true, false)) { // && (cidx == 0)
        //stores original image info. before shake.
        tr[ *this].m_cogXOrig = (double)cogx / toti;
        tr[ *this].m_cogYOrig = (double)cogy / toti;

//        N x N kernel, p-th M x M image around the edge.
//        I^p(m, n) = I^p_orig(m + a - N/2 - M/2, n + b - N/2 - M/2) kab
//        y_i = I^{i / M^2}((i % M^2) / M - M/2, (i % M^2) % M - M/2), i < M^2
//        x_j = k(j / N, j % N), j < N^2
//        A_ij = I^{i / M^2}_orig((i % M^2) / M + j / N - N/2 - M/2, (i % M^2) % M + j % N - N/2 - M/2)
        constexpr unsigned int max_rank = num_conv_tsvd;
        constexpr unsigned int n = num_conv_tsvd;
        unsigned int m = 2 * antishake_pixels;
        Eigen::MatrixXd matA = Eigen::MatrixXd::Zero(m * m * edges.size(), n * n);
        const uint32_t *raw = &rawCountsNext->at(0);
        for(unsigned int i = 0; i < matA.cols(); ++i) {
            unsigned int p = i / m / m;
            for(unsigned int j = 0; j < matA.rows(); ++j) {
                unsigned int y = (i % (m * m)) / m + j / n + antishake_pixels + edges[p].x - n / 2 - m / 2;
                unsigned int x = (i % (m * m)) % m + j % n + antishake_pixels + edges[p].y - n / 2 - m / 2;
                matA(i, j) = (double)raw[y * width + x];
            }
        }
        tr[ *this].m_tsvd = std::make_shared<TikhonovRegular>(matA, TikhonovRegular::TikhonovMatrix::NONE, 10000, max_rank);
    }
    tr[ *this].m_stride = width;
    tr[ *this].m_width = width - antishake_pixels * 2;
    tr[ *this].m_height = height - antishake_pixels * 2;
    if(antishake_pixels) {
        //finds the pixels shifts
        double shift_dx = (double)cogx / toti - tr[ *this].m_cogXOrig;
        int shift_x = floor(shift_dx);
        shift_dx -= shift_x;
        double shift_dy = (double)cogy / toti - tr[ *this].m_cogYOrig;
        int shift_y = floor(shift_dy);
        shift_dy -= shift_y;
        shift_x = std::min(shift_x, antishake_pixels - 1);
        shift_x = std::max(shift_x, -antishake_pixels);
        shift_y = std::min(shift_y, antishake_pixels - 1);
        shift_y = std::max(shift_y, -antishake_pixels);
//        fprintf(stderr, "shift: (%d, %d)\n", shift_x, shift_y);
        tr[ *this].m_firstPixel = (shift_y + antishake_pixels) * width + shift_x + antishake_pixels;
        {
            constexpr unsigned int n = num_conv_tsvd;
            unsigned int m = 2 * antishake_pixels;
            Eigen::VectorXd vecY = Eigen::VectorXd::Zero(m * m * edges.size());
            const uint32_t *raw = &tr[ *this].m_rawCounts->at(tr[ *this].firstPixel());
            for(unsigned int i = 0; i < vecY.size(); ++i) {
                unsigned int p = i / m / m;
                unsigned int y = (i % (m * m)) / m + edges[p].x - n / 2 - m / 2;
                unsigned int x = (i % (m * m)) % m + edges[p].y - n / 2 - m / 2;
                vecY(i) = (double)raw[y * width + x];
            }
            Eigen::VectorXd vecX = tr[ *this].m_tsvd->solve(vecY); //kernel
            vecX.size();
        }
        {
            //bi-linear interpolation.
            uint32_t *raw = &tr[ *this].m_rawCounts->at(tr[ *this].firstPixel());
            unsigned int stride = tr[ *this].stride();
            unsigned int width = tr[ *this].width();
            unsigned int height = tr[ *this].height();
            std::vector<uint32_t> line(width);
            std::copy(raw, raw + width, &line[0]);
            raw += stride;
            uint64_t a00 = lrint(0x100000000uLL * (1 - shift_dx) * (1 - shift_dy));
            uint64_t a10 = lrint(0x100000000uLL * shift_dx * (1 - shift_dy));
            uint64_t a01 = lrint(0x100000000uLL * (1 - shift_dx) * shift_dy);
            uint64_t a11 = 0x100000000uLL - a00 - a10 - a01;
            for(unsigned int y  = 1; y < height + 1; ++y) {
                uint32_t *raw_ym1 = &line[0];
                uint32_t w_xym1 = *raw_ym1++;
                uint32_t w_xm1 = *raw++;
                for(unsigned int x  = 1; x < width + 1; ++x) {
                     uint32_t w_ym1 = *raw_ym1;
                     uint32_t w = *raw;
                     *raw_ym1++ = w; //stores original value.
                     *raw++ = (w * a11 + w_ym1 * a10 + w_xm1 * a01 + w_xym1 * a00) / 0x100000000uLL;
                     w_xym1 = w_ym1;
                     w_xm1 = w;
                }
                raw += stride - width;
            }
        }
    }
    else {
         tr[ *this].m_firstPixel = 0;
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
        m_lsnOnCameraGainChanged = tr[ *cameraGain()].onValueChanged().connectWeakly(
            shared_from_this(), &XDigitalCamera::onCameraGainChanged);
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

    m_lsnOnCameraGainChanged.reset();
    m_lsnOnBrightnessChanged.reset();
    m_lsnOnExposureTimeChanged.reset();
    m_lsnOnTriggerModeChanged.reset();
    m_lsnOnVideoModeChanged.reset();
    m_lsnOnROISelectionToolTouched.reset();
    return NULL;
}
