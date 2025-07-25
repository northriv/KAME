/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "ui_odmr2danalysisform.h"
#include "odmrfspectrum.h"
#include "odmrimaging.h"
#include "odmr2danalysis.h"
#include "signalgenerator.h"

#include "x2dimage.h"
#include <QColorSpace>


REGISTER_TYPE(XDriverList, ODMR2DAnalysis, "ODMR image analysis on frequency sweep");


//---------------------------------------------------------------------------
XODMR2DAnalysis::XODMR2DAnalysis(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XSecondaryDriver(name, runtime, ref(tr_meas), meas),
    m_form(new FrmODMR2DAnalysis),
    m_odmrFSpectrum(create<XItemNode<XDriverList, XODMRFSpectrum> >(
        "ODMRFSpectrum", false, ref(tr_meas), meas->drivers(), true)),
    m_regionSelection(create<XComboNode>("RegionSelection", false, true)),
    m_analysisMethod(create<XComboNode>("AnalysisMethod", false, true)),
    m_average(create<XUIntNode>("Average", false)),
    m_clearAverage(create<XTouchableNode>("ClearAverage", true)),
    m_incrementalAverage(create<XBoolNode>("IncrementalAverage", false)),
    m_autoMinMaxForColorMap(create<XBoolNode>("AutoMinMaxForColorMap", false)),
    m_minForColorMap(create<XDoubleNode>("MinForColorMap", false)),
    m_maxForColorMap(create<XDoubleNode>("MaxForColorMap", false)),
    m_colorMapMethod(create<XComboNode>("AnalysisMethod", false, true)),
    m_processedImage(create<X2DImage>("ProcessedImage", false,
                                      m_form->m_graphwidgetProcessed, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
                                      2, m_form->m_dblGamma,
                                      m_form->m_tbMathMenu, meas, static_pointer_cast<XDriver>(shared_from_this()),
                                      true)) {

    connect(odmrFSpectrum());

    m_form->setWindowTitle(i18n("ODMR Image Analysis on Freq. Sweep - ") + getLabel() );

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(m_odmrFSpectrum, m_form->m_cmbODMRFSpectrum, ref(tr_meas)),
        xqcon_create<XQSpinBoxUnsignedConnector>(average(), m_form->m_spbAverage),
        xqcon_create<XQDoubleSpinBoxConnector>(m_minForColorMap, m_form->m_dblMinForColorMap),
        xqcon_create<XQDoubleSpinBoxConnector>(m_maxForColorMap, m_form->m_dblMaxForColorMap),
        xqcon_create<XQButtonConnector>(m_clearAverage, m_form->m_btnClearAverage),
        xqcon_create<XQToggleButtonConnector>(m_incrementalAverage, m_form->m_ckbIncrementalAverage),
        xqcon_create<XQToggleButtonConnector>(m_autoMinMaxForColorMap, m_form->m_ckbAutoMinMapForColorMap),
        xqcon_create<XQComboBoxConnector>(m_analysisMethod, m_form->m_cmbAnalysisMethod, Snapshot( *m_analysisMethod)),
        xqcon_create<XQComboBoxConnector>(m_colorMapMethod, m_form->m_cmbColorMapMethod, Snapshot( *m_colorMapMethod)),
        xqcon_create<XQComboBoxConnector>(m_regionSelection, m_form->m_cmbSelRegion, Snapshot( *m_regionSelection)),
    };

    iterate_commit([=](Transaction &tr){
        tr[ *average()] = 1;
        tr[ *m_autoMinMaxForColorMap] = true;
        tr[ *m_colorMapMethod].add({"RedWhiteBlue", "YellowGreenBlue"});
        tr[ *m_analysisMethod].add({"CoG", "2nd Moment"});
        tr[ *m_regionSelection].add({"All", "Sub Region"});
    });

    iterate_commit([=](Transaction &tr){
        m_lsnOnClearAverageTouched = tr[ *clearAverage()].onTouch().connectWeakly(
            shared_from_this(), &XODMR2DAnalysis::onClearAverageTouched);
        m_lsnOnCondChanged = tr[ *average()].onValueChanged().connectWeakly(
            shared_from_this(), &XODMR2DAnalysis::onCondChanged);
        tr[ *m_minForColorMap].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *m_maxForColorMap].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *incrementalAverage()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *colorMapMethod()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *regionSelection()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *analysisMethod()].onValueChanged().connect(m_lsnOnCondChanged);
    });
}
//---------------------------------------------------------------------------

void
XODMR2DAnalysis::showForms() {
    m_form->showNormal();
    m_form->raise();
}

void
XODMR2DAnalysis::onCondChanged(const Snapshot &shot, XValueNodeBase *node) {
    if(node == incrementalAverage().get())
        trans( *average()) = 0;
    if(node == incrementalAverage().get())
        onClearAverageTouched(shot, clearAverage().get());
    else
        requestAnalysis();
}
void
XODMR2DAnalysis::onClearAverageTouched(const Snapshot &shot, XTouchableNode *) {
    trans( *this).m_timeClearRequested = XTime::now();
    requestAnalysis();
}
bool
XODMR2DAnalysis::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) const {
    shared_ptr<XODMRFSpectrum> fspectrum__ = shot_this[ *odmrFSpectrum()];
    if( !fspectrum__) return false;
    if((emitter != fspectrum__.get()) && (emitter != this))
        return false;
    const Snapshot &shot_fspectrum((emitter == fspectrum__.get()) ? shot_emitter : shot_others);
    shared_ptr<XODMRImaging> odmr__ = shot_fspectrum[ *fspectrum__->odmr()];
    if( !odmr__) return false;
    shared_ptr<XSG> sg1__ = shot_fspectrum[ *fspectrum__->sg1()];
    if( !sg1__) return false;
    return true;
}
void
XODMR2DAnalysis::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) {
    const Snapshot &shot_this(tr);

    shared_ptr<XODMRFSpectrum> fspectrum__ = shot_this[ *odmrFSpectrum()];
    const Snapshot &shot_fspectrum((emitter == fspectrum__.get()) ? shot_emitter : shot_others);
    const Snapshot &shot_odmr = shot_others;
    shared_ptr<XODMRImaging> odmr__ = shot_fspectrum[ *fspectrum__->odmr()];

    if(shot_odmr[ *odmr__->incrementalAverage()]) {
        gWarnPrint(i18n("Do NOT use incremental avg. Skipping."));
        throw XSkippedRecordError(__FILE__, __LINE__);
    }

    bool clear = (shot_this[ *this].m_timeClearRequested.isSet());
    tr[ *this].m_timeClearRequested = {};

    double freq = shot_fspectrum[ *fspectrum__].lastFreqAcquiredInHz() * 1e-6; //M/Hz

    double max__; //MHz
    double min__; //MHz

    if(shot_this[ *regionSelection()] == 0) {
        //ALL
        double cfreq = shot_fspectrum[ *fspectrum__->centerFreq()]; //MHz
        double freq_span = shot_fspectrum[ *fspectrum__->freqSpan()] * 1e-3; //MHz
        max__ = (cfreq + freq_span/2);
        min__ = (cfreq - freq_span/2);
    }
    else {
        double sub_max = shot_fspectrum[ *fspectrum__->subRegionMaxFreq()]; //MHz
        double sub_min = shot_fspectrum[ *fspectrum__->subRegionMinFreq()]; //MHz
        max__ = sub_max;
        min__ = sub_min;
    }
    if(max__ <= min__) {
        throw XSkippedRecordError(i18n("Invalid min. and max."), __FILE__, __LINE__);
    }
    if((freq < min__) || (freq > max__))
        throw XSkippedRecordError(__FILE__, __LINE__);

    double coeff_freqidx = 1.0 / (shot_fspectrum[ *fspectrum__->freqStep()] * 1e-3); //1/MHz
    if(tr[ *this].m_dfreq != 1.0 / coeff_freqidx)
        clear = true;
    tr[ *this].m_dfreq = 1.0 / coeff_freqidx;
    uint32_t freqidx = lrint(coeff_freqidx * (freq - min__));

    unsigned int width = shot_odmr[ *odmr__].width();
    unsigned int height = shot_odmr[ *odmr__].height();

    if(tr[ *this].m_minValue != min__)
        clear = true;
    tr[ *this].m_minValue = min__;

    if( !tr[ *incrementalAverage()] && !clear && (emitter == fspectrum__.get())) {
        clear = true;
        if(std::max(1u, (unsigned int)tr[ *average()]) > tr[ *this].m_accumulated[0])
            clear = false;
    }
    unsigned int num_summed_frames = tr[ *this].secondMoment() ? 3 : 2;
    for(unsigned int i = 0; i < num_summed_frames; ++i) {
        if( !tr[ *this].m_summedCounts[i] || (tr[ *this].m_summedCounts[i]->size() != width * height)) {
            clear = true;
        }
    }

    tr[ *this].m_width = width;
    tr[ *this].m_height = height;

    if(clear) {
        tr[ *this].m_summedCounts[2].reset();
        for(unsigned int i = 0; i < num_summed_frames; ++i) {
            tr[ *this].m_summedCounts[i] = summedCountsFromPool(width * height);
            std::fill(tr[ *this].m_summedCounts[i]->begin(), tr[ *this].m_summedCounts[i]->end(), 0);
            tr[ *this].m_accumulated[i] = 0;
        }
    }

    constexpr int64_t coeff_PLOn_o_Off = 0x8000LL;

    if(emitter == fspectrum__.get()) {
        auto rawCountsPLOff = shot_odmr[ *odmr__].rawCountsPLOff();
        auto rawCountsPLOn = shot_odmr[ *odmr__].rawCountsPLOn();

        const uint32_t *summed_on_o_off = &tr[ *this].m_summedCounts[0]->at(0),
            *summed_f_on_o_off = &tr[ *this].m_summedCounts[1]->at(0);
        auto summedCountsNext_on_o_off = summedCountsFromPool(width * height);
        auto summedCountsNext_f_on_o_off = summedCountsFromPool(width * height);
        uint32_t *summedNext_on_o_off = &summedCountsNext_on_o_off->at(0);
        uint32_t *summedNext_f_on_o_off = &summedCountsNext_f_on_o_off->at(0);

        const uint32_t *pploff = &rawCountsPLOff->at(0);
        const uint32_t *pplon = &rawCountsPLOn->at(0);

        for(unsigned int y  = 0; y < height; ++y) {
            for(unsigned int x  = 0; x < width; ++x) {
                uint32_t ploff = *pploff++;
                uint32_t plon = *pplon++;
                int32_t plon_o_off_s32 = plon * coeff_PLOn_o_Off / (int64_t)ploff;
                *summedNext_on_o_off++ = *summed_on_o_off++ + (uint32_t)plon_o_off_s32;
                *summedNext_f_on_o_off++ = *summed_f_on_o_off++ + (uint32_t)(freqidx * plon_o_off_s32);
            }
        }
        // CoG = <f dPL/PL> / <dPL/PL> = <f (on/off - 1)> / <on/off - 1> = (<f on/off> - <f>) / (<on/off> -  1)
        // = (sum (f on/off C) - sum f C) / (sum on/off C -  N C)
        tr[ *this].m_accumulated[0]++;
        tr[ *this].m_accumulated[1] += freqidx;
        tr[ *this].m_summedCounts[0] = summedCountsNext_on_o_off;
        tr[ *this].m_summedCounts[1] = summedCountsNext_f_on_o_off;
        // tr[ *this].m_accumulated[2] += freqsq_u16;

        // const uint32_t *summed_fsq_on_o_off = &tr[ *this].m_summedCounts[2]->at(0);
        // auto summedCountsNext_fsq_on_o_off = summedCountsFromPool(width * height);
    }

    if( !shot_this[ *this].m_accumulated[0])
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.

    tr[ *this].m_coefficients[0] = 1.0 / coeff_PLOn_o_Off / tr[ *this].m_accumulated[0]; // 1/C/N
    tr[ *this].m_coefficients[1] = tr[ *this].m_coefficients[0] / coeff_freqidx; // 1/C/N [MHz]
    // tr[ *this].m_coefficients[2] = tr[ *this].m_coefficients[0] / coeff_freqsq_u16; //for math tools

    int64_t offsets[3];
    offsets[0] = llrint(-1.0 / tr[ *this].m_coefficients[0]); //-N C
    offsets[1] = llrint(-(double)tr[ *this].m_accumulated[1] / (tr[ *this].m_coefficients[0] * tr[ *this].m_accumulated[0])); //- sum f C

    tr[ *this].m_offsets[0] = offsets[0] * tr[ *this].m_coefficients[0]; // = -1
    tr[ *this].m_offsets[1] = offsets[1] * tr[ *this].m_coefficients[1]; // = -sum f/N = -<f> [MHz]

    if(tr[ *m_autoMinMaxForColorMap]) {
        const uint32_t *summed_on_o_off = &tr[ *this].m_summedCounts[0]->at(0),
            *summed_f_on_o_off = &tr[ *this].m_summedCounts[1]->at(0);

        constexpr int64_t coeff_dCoG = 0x10000LL;
        int64_t dcogmin = 0x7fffffffffffffffLL;
        int64_t dcogmax = -0x7fffffffffffffffLL;
        for(unsigned int i  = 0; i < width * height; ++i) {
            int64_t dplopl = (int64_t)*summed_on_o_off++ + offsets[0];
            int64_t f_dplopl = (int64_t)*summed_f_on_o_off++ + offsets[1];
            if(abs(dplopl) < coeff_PLOn_o_Off / 5000) continue; //ignore |dPL/PL| < 0.02%
            int64_t dcog = f_dplopl * coeff_dCoG / dplopl;
            if(dcog > dcogmax)
                dcogmax = dcog;
            if(dcog < dcogmin)
                dcogmin = dcog;
        }
        tr[ *minForColorMap()]  = std::max(min__, (double)dcogmin / coeff_dCoG / coeff_freqidx + min__);
        tr[ *maxForColorMap()]  = std::min(max__, (double)dcogmax / coeff_dCoG / coeff_freqidx + min__);
        tr.unmark(m_lsnOnCondChanged);
    }
    if(tr[ *incrementalAverage()]) {
        tr[ *average()] = tr[ *this].m_accumulated[0];
        tr.unmark(m_lsnOnCondChanged);
//        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
    else {
        if(tr[ *average()] > tr[ *this].m_accumulated[0])
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }

}

void
XODMR2DAnalysis::visualize(const Snapshot &shot) {
    if( !shot[ *this].m_accumulated[0])
        return;

    unsigned int width = shot[ *this].width();
    unsigned int height = shot[ *this].height();
    auto qimage = std::make_shared<QImage>(width, height, QImage::Format_RGBA64);
    qimage->setColorSpace(QColorSpace::SRgbLinear);
    auto cbimage = std::make_shared<QImage>(width, 1, QImage::Format_RGBA64);
    qimage->setColorSpace(QColorSpace::SRgbLinear);

    uint16_t *processed = reinterpret_cast<uint16_t*>(qimage->bits());
    const uint32_t *summed_on_o_off = &shot[ *this].m_summedCounts[0]->at(0),
        *summed_f_on_o_off = &shot[ *this].m_summedCounts[1]->at(0);

    int64_t offsets[3];
    offsets[0] = llrint(-1.0 / shot[ *this].m_coefficients[0]); //-N C
    offsets[1] = llrint(-(double)shot[ *this].m_accumulated[1] / (shot[ *this].m_coefficients[0] * shot[ *this].m_accumulated[0])); //- sum f C

    double cogmin = shot[ *minForColorMap()];
    double cogmax = shot[ *maxForColorMap()];

    constexpr int64_t coeff_dCoG = 0x10000LL;
    double dfreq = shot[ *this].dfreq(); //MHz
    double fmin = shot[ *this].minValue();
    int64_t dcog_offset = coeff_dCoG * llrint((fmin - cogmin) / dfreq);

    std::array<int64_t, 3> coloroffsets_low = {};
    std::array<int64_t, 3> coloroffsets_high = {};
    int64_t colorgain = llrint(0x100000000uLL * 0xffffuLL * dfreq / (cogmax - cogmin) / coeff_dCoG); // /256 is needed for RGBA8888 format
    std::array<int64_t, 3> dcog_gain_low = {};
    std::array<int64_t, 3> dcog_gain_high = {};
    switch((unsigned int)shot[ *m_colorMapMethod]) {
    case 0:
    default:
        //RedWhiteBlue
        dcog_gain_low[0] = 2 * colorgain;
        dcog_gain_low[1] = 2 * colorgain;
        dcog_gain_low[2] = 0;
        coloroffsets_low[0] = 0;
        coloroffsets_low[1] = 0;
        coloroffsets_low[2] = 0x100000000LL * 0xffffLL; // or 0xff
        dcog_gain_high[0] = 0;
        dcog_gain_high[1] = -2 * colorgain;
        dcog_gain_high[2] = -2 * colorgain;
        break;
    case 1:
        //YellowGreenBlue
        dcog_gain_low[0] = 0;
        dcog_gain_low[1] = 2 * colorgain;
        dcog_gain_low[2] = -2 * colorgain;
        coloroffsets_low[0] = 0;
        coloroffsets_low[1] = 0;
        coloroffsets_low[2] = 0x100000000LL * 0xffffLL;// or 0xff
        dcog_gain_high[0] = 2 * colorgain;
        dcog_gain_high[1] = 0;
        dcog_gain_high[2] = 0;
        break;
    }
    int64_t thres = 0x7fff * 0x100000000LL / colorgain;
    for(unsigned int cidx: {0,1,2})
        coloroffsets_high[cidx] = coloroffsets_low[cidx] + thres * (dcog_gain_low[cidx] - dcog_gain_high[cidx]);

    for(unsigned int i  = 0; i < width * height; ++i) {
        int64_t dplopl = (int64_t)*summed_on_o_off++ + offsets[0];
        int64_t f_dplopl = (int64_t)*summed_f_on_o_off++ + offsets[1];
        if( !dplopl) {
            *processed++ = 0; *processed++ = 0; *processed++ = 0; *processed++ = 0xffffu;
            continue;
        }
        int64_t dcog = f_dplopl * coeff_dCoG / dplopl + dcog_offset;
        const auto &dcog_gain = (dcog > thres) ? dcog_gain_high : dcog_gain_low;
        const auto &coloroffsets = (dcog > thres) ? coloroffsets_high : coloroffsets_low;
        for(unsigned int cidx: {0,1,2}) {
            int64_t v = (dcog * dcog_gain[cidx] + coloroffsets[cidx]) / 0x100000000LL;
            *processed++ = std::max(0LL, std::min(v, 0xffffLL));
        }
        *processed++ = 0xffffu;
    }
    //colorbar
    processed = reinterpret_cast<uint16_t*>(cbimage->bits());
    for(unsigned int i  = 0; i < cbimage->width(); ++i) {
        int64_t dcog = (double)i / (cbimage->width() - 1) * (cogmax - cogmin) / dfreq * coeff_dCoG;
        const auto &dcog_gain = (dcog > thres) ? dcog_gain_high : dcog_gain_low;
        const auto &coloroffsets = (dcog > thres) ? coloroffsets_high : coloroffsets_low;
        for(unsigned int cidx: {0,1,2}) {
            int64_t v = (dcog * dcog_gain[cidx] + coloroffsets[cidx]) / 0x100000000LL;
            *processed++ = std::max(0LL, std::min(v, 0xffffLL));
        }
        *processed++ = 0xffffu;
    }

    std::vector<double> coeffs, offsets_image;
    std::vector<const uint32_t *> rawimages;
    for(unsigned int cidx: {0,1}) {
        coeffs.push_back(shot[ *this].m_coefficients[cidx]);
        offsets_image.push_back(shot[ *this].m_offsets[cidx]);
        rawimages.push_back( &shot[ *this].m_summedCounts[cidx]->at(0));
    }
    iterate_commit([&](Transaction &tr){
        tr[ *this].m_qimage = qimage;
        tr[ *m_processedImage->graph()->onScreenStrings()] = formatString("Avg:%u", (unsigned int)shot[ *this].m_accumulated[0]);
        m_processedImage->updateImage(tr, qimage, rawimages, width, coeffs, offsets_image);
        m_processedImage->updateColorBarImage(tr, cogmin, cogmax, cbimage);
    });
}

local_shared_ptr<std::vector<uint32_t>>
XODMR2DAnalysis::summedCountsFromPool(int imagesize) {
    local_shared_ptr<std::vector<uint32_t>> summedCountsNext, p;
    for(int i = 0; i < NumSummedCountsPool; ++i) {
        if( !m_summedCountsPool[i])
            m_summedCountsPool[i] = make_local_shared<std::vector<uint32_t>>(imagesize);
        p.swap(m_summedCountsPool[i]); //atomic swap
        if(p && p.unique()) { //confirmed uniquness.
            m_summedCountsPool[i].compareAndSet({}, p); //sharing me for later use.
            summedCountsNext = p;
            p->resize(imagesize);
            break;
        }
        m_summedCountsPool[i].compareAndSet({}, p); //restoring busy one for later use.
    }
    if( !summedCountsNext)
        summedCountsNext = make_local_shared<std::vector<uint32_t>>(imagesize);
    return summedCountsNext;
}



