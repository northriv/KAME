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
#include "filterwheel.h"
#include "imageprocessor.h"
#include "ui_imageprocessorform.h"
#include "x2dimage.h"
#include "graph.h"
#include "graphwidget.h"
#include "xnodeconnector.h"
#include "graphmathtool.h"
#include <QToolButton>
#include <QColorSpace>
#include "graphmathtoolconnector.h"

REGISTER_TYPE(XDriverList, ImageProcessor, "RGB Image Processor for camera");

XImageProcessor::XImageProcessor(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSecondaryDriver(name, runtime, ref(tr_meas), meas),
    m_camera(create<XItemNode<XDriverList, XDigitalCamera> >(
          "DigitalCamera", false, ref(tr_meas), meas->drivers(), true)),
    m_filterWheel(create<XItemNode<XDriverList, XFilterWheel> >(
          "FilterWheel", false, ref(tr_meas), meas->drivers(), true)),
    m_average(create<XUIntNode>("Average", false)),
    m_clearAverage(create<XTouchableNode>("ClearAverage", true)),
    m_autoGain(create<XBoolNode>("AutoGain", false)),
    m_incrementalAverage(create<XBoolNode>("IncrementalAverage", false)),
    m_filterIndexR(create<XUIntNode>("filterIndexR", false)),
    m_filterIndexG(create<XUIntNode>("filterIndexG", false)),
    m_filterIndexB(create<XUIntNode>("filterIndexB", false)),
    m_colorGainR(create<XDoubleNode>("ColorGainR", false)),
    m_colorGainG(create<XDoubleNode>("ColorGainG", false)),
    m_colorGainB(create<XDoubleNode>("ColorGainB", false)),
    m_gainForDisp(create<XDoubleNode>("GainForDisp", false)),
    m_form(new FrmImageProcessor),
    m_rgbImage(create<X2DImage>("RGBImage", false,
                                   m_form->m_graphwidget, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
                                   2, m_form->m_dblGamma,
                                   m_form->m_tbMathMenu, meas, static_pointer_cast<XDriver>(shared_from_this()))) {

    auto plot = m_rgbImage->plot();

    connect(camera());
    connect(filterWheel());

    m_form->setWindowTitle(i18n("RGB Image Processor - ") + getLabel() );

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(m_camera, m_form->m_cmbCamera, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(m_filterWheel, m_form->m_cmbFilterWheel, ref(tr_meas)),
        xqcon_create<XQSpinBoxUnsignedConnector>(average(), m_form->m_spbAverage),
        xqcon_create<XQDoubleSpinBoxConnector>(colorGainR(), m_form->m_dblGainR),
        xqcon_create<XQDoubleSpinBoxConnector>(colorGainG(), m_form->m_dblGainG),
        xqcon_create<XQDoubleSpinBoxConnector>(colorGainB(), m_form->m_dblGainB),
        xqcon_create<XQSpinBoxUnsignedConnector>(filterIndexR(), m_form->m_spbFilterIdxR),
        xqcon_create<XQSpinBoxUnsignedConnector>(filterIndexG(), m_form->m_spbFilterIdxG),
        xqcon_create<XQSpinBoxUnsignedConnector>(filterIndexB(), m_form->m_spbFilterIdxB),
        xqcon_create<XQDoubleSpinBoxConnector>(gainForDisp(), m_form->m_dblGain),
//        xqcon_create<XQLineEditConnector>((), m_form->m_edIntegrationTime),
        xqcon_create<XQButtonConnector>(m_clearAverage, m_form->m_btnClearAverage),
        xqcon_create<XQToggleButtonConnector>(m_incrementalAverage, m_form->m_ckbIncrementalAverage),
        xqcon_create<XQToggleButtonConnector>(m_autoGain, m_form->m_ckbAutoGain),
    };

    iterate_commit([=](Transaction &tr){
        tr[ *average()] = 1;
        tr[ *autoGain()] = true;
        tr[ *colorGainR()] = 1.0;
        tr[ *colorGainG()] = 1.0;
        tr[ *colorGainB()] = 1.0;
        tr[ *filterIndexR()] = 0;
        tr[ *filterIndexG()] = 1;
        tr[ *filterIndexB()] = 2;
    });

    iterate_commit([=](Transaction &tr){
        m_lsnOnClearAverageTouched = tr[ *clearAverage()].onTouch().connectWeakly(
            shared_from_this(), &XImageProcessor::onClearAverageTouched);
        m_lsnOnCondChanged = tr[ *average()].onValueChanged().connectWeakly(
            shared_from_this(), &XImageProcessor::onCondChanged);
        tr[ *filterIndexR()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *filterIndexG()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *filterIndexB()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *colorGainR()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *colorGainG()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *colorGainB()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *gainForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *incrementalAverage()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *autoGain()].onValueChanged().connect(m_lsnOnCondChanged);
    });
}
XImageProcessor::~XImageProcessor() {
}
void
XImageProcessor::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}
void
XImageProcessor::onCondChanged(const Snapshot &shot, XValueNodeBase *node) {
    if(node == incrementalAverage().get())
        trans( *average()) = 0;
    if(node == incrementalAverage().get())
        onClearAverageTouched(shot, clearAverage().get());
    else
        requestAnalysis();
}
void
XImageProcessor::onClearAverageTouched(const Snapshot &shot, XTouchableNode *) {
    trans( *this).m_timeClearRequested = XTime::now();
    requestAnalysis();
}

bool
XImageProcessor::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) const {
    shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
    if( !camera__) return false;
    if(emitter == this) return true;
    if(emitter != camera__.get())
        return false;
    //ignores old camera frames
    if((shot_emitter[ *camera__].time() < shot_this[ *this].m_timeClearRequested) &&
        shot_this[ *this].m_timeClearRequested - shot_emitter[ *camera__].time() < 60.0) //not reading raw binary
        return false;
    return true;
}
void
XImageProcessor::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) {
    const Snapshot &shot_this(tr);

    shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
    const Snapshot &shot_camera((emitter == camera__.get()) ? shot_emitter : shot_others);

    bool clear = (shot_this[ *this].m_timeClearRequested.isSet());
    tr[ *this].m_timeClearRequested = {};

    const auto rawimage = shot_camera[ *camera__].rawCounts();
    unsigned int width = shot_camera[ *camera__].width();
    unsigned int height = shot_camera[ *camera__].height();
    unsigned int raw_stride = shot_camera[ *camera__].stride();

    std::array<unsigned int, 3> rgb_filterIndices = {shot_this[ *filterIndexR()], shot_this[ *filterIndexG()], shot_this[ *filterIndexB()]};
    unsigned int seq_len = 3; //RGB
    if( !tr[ *incrementalAverage()] && !clear && (emitter == camera__.get())) {
        clear = true;
        for(unsigned int i = 0; i < seq_len; ++i) {
            if(std::max(1u, (unsigned int)tr[ *average()]) > tr[ *this].m_accumulated[i])
                clear = false;
        }
    }
    for(unsigned int i = 0; i < seq_len; ++i) {
        if( !tr[ *this].m_summedCounts[i] || (tr[ *this].m_summedCounts[i]->size() != width * height)) {
            clear = true;
        }
        if(tr[ *this].m_accumulated[i] && (tr[ *this].m_filterIndice[i] != rgb_filterIndices[i])) {
            //filter index mismatches during averaging.
            clear = true;
        }
    }
    tr[ *this].m_width = width;
    tr[ *this].m_height = height;
    if(clear) {
        for(unsigned int i = 0; i < seq_len; ++i) {
            tr[ *this].m_summedCounts[i] = summedCountsFromPool(width * height);
            std::fill(tr[ *this].m_summedCounts[i]->begin(), tr[ *this].m_summedCounts[i]->end(), 0);
            tr[ *this].m_accumulated[i] = 0;
        }
    }
    if(emitter == camera__.get()) {
        shared_ptr<XFilterWheel> wheel__ = shot_this[ *filterWheel()];
        if( !wheel__)
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
//            throw XDriver::XRecordError(i18n("Filter wheel is not specified."), __FILE__, __LINE__);
        int wheelidx = shot_others[ *wheel__].wheelIndexOfFrame(
            shot_camera[ *camera__].time(), shot_camera[ *camera__].timeAwared());
        auto it = std::find(rgb_filterIndices.begin(), rgb_filterIndices.end(),  wheelidx);
        unsigned int cidx = std::distance(rgb_filterIndices.begin(), it);
        if(cidx >= 3) //non-RGB filter
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.

        tr[ *this].m_filterIndice[cidx] = rgb_filterIndices[cidx]; // = wheelidx
        tr[ *this].m_colorGains[cidx] =
            std::array<double, 3>{shot_this[ *colorGainR()], shot_this[ *colorGainG()], shot_this[ *colorGainB()]}
                [cidx];

        auto summedCountsNext = summedCountsFromPool(width * height);
        uint32_t *summedNext = &summedCountsNext->at(0);
        const uint32_t *summed = &tr[ *this].m_summedCounts[cidx]->at(0);

        const uint32_t *raw = &rawimage->at(shot_camera[ *camera__].firstPixel());
        for(unsigned int y  = 0; y < height; ++y) {
            for(unsigned int x  = 0; x < width; ++x) {
                uint64_t v = *summed++ + *raw++;
                if(v > 0x100000000uLL)
                    v = 0xffffffffuL;
                *summedNext++ = v;
            }
            raw += raw_stride - width;
        }
        assert(summedNext == &summedCountsNext->at(0) + width * height);
        assert(summed == &tr[ *this].m_summedCounts[cidx]->at(0) + width * height);
        (tr[ *this].m_accumulated[cidx])++;
        tr[ *this].m_summedCounts[cidx] = summedCountsNext; // = summed + live image
    }

    if(shot_this[ *this].accumulatedCountRGB() == 0)
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.

    for(unsigned int i = 0; i < seq_len; ++i)
        tr[ *this].m_coefficients[i] = tr[ *this].m_colorGains[i] / tr[ *this].m_accumulated[i]; //for math tools

    if(tr[ *m_autoGain]) {
        double gain = 0xffff;
        for(unsigned int i = 0; i < seq_len; ++i) {
            const uint32_t *summed = &tr[ *this].m_summedCounts[i]->at(0);
            uint32_t vmin = 0xffffffffu;
            uint32_t vmax = 0u;
            for(unsigned int i  = 0; i < width * height; ++i) {
                uint32_t v = *summed++;
                if(v > vmax)
                    vmax = v;
                if(v < vmin)
                    vmin = v;
            }
            if(vmax > 0) {
                gain = std::min(gain, (double)0xffffu / (vmax * tr[ *this].m_coefficients[i]));
            }
        }
        {
            tr[ *gainForDisp()]  = gain;
            tr.unmark(m_lsnOnCondChanged);
        }
    }
    if(tr[ *incrementalAverage()]) {
        tr[ *average()] = shot_this[ *this].accumulatedCountRGB();
        tr.unmark(m_lsnOnCondChanged);
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
    else {
        if(tr[ *average()] > shot_this[ *this].accumulatedCountRGB())
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
}
void
XImageProcessor::visualize(const Snapshot &shot) {
    if(shot[ *this].accumulatedCountRGB() == 0)
        return;

    unsigned int width = shot[ *this].width();
    unsigned int height = shot[ *this].height();
    auto qimage = std::make_shared<QImage>(width, height, QImage::Format_RGBA64);
    qimage->setColorSpace(QColorSpace::SRgbLinear);

    unsigned int seq_len = 3; //RGB

    {
        uint64_t gain_av[3];
        for(unsigned int cidx = 0; cidx < seq_len; ++cidx)
            gain_av[cidx] = lrint(0x100000000uLL * shot[ *gainForDisp()] * shot[ *this].m_coefficients[cidx]);
        uint16_t *processed = reinterpret_cast<uint16_t*>(qimage->bits()); //16bit color
        const uint32_t *summed[3];
        for(unsigned int cidx = 0; cidx < seq_len; ++cidx)
            summed[cidx] = &shot[ *this].m_summedCounts[cidx]->at(0);

        for(unsigned int i  = 0; i < width * height; ++i) {
            for(unsigned int cidx = 0; cidx < seq_len; ++cidx) {
                int64_t v = ((int64_t)(*summed[cidx] * gain_av[cidx]))  / 0x100000000LL;
                *processed++ = std::max(0LL, std::min(v, 0xffffLL));
                (summed[cidx])++;
            }
            *processed++ = 0xffffu;
        }
    }

    std::vector<double> coeffs;
    std::vector<const uint32_t *> rawimages;
    for(unsigned int cidx: {0,1}) {
        coeffs.push_back(shot[ *this].m_coefficients[seq_len - 2 + cidx]);
        rawimages.push_back( &shot[ *this].m_summedCounts[seq_len - 2 + cidx]->at(0));
    }
        iterate_commit([&](Transaction &tr){
        tr[ *this].m_qimage = qimage;
        tr[ *m_rgbImage->graph()->onScreenStrings()] = formatString("Avg:%u", (unsigned int)shot[ *this].m_accumulated[0]);
        m_rgbImage->updateImage(tr, qimage, rawimages, width, coeffs);
    });

//    shared_ptr<XFilterWheel> wheel__ = shot[ *filterWheel()];
//    if(wheel__) {
//        wheel__->goAround();
//    }
}

local_shared_ptr<std::vector<uint32_t>>
XImageProcessor::summedCountsFromPool(int imagesize) {
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


