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
                                      4, m_form->m_dblGamma,
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
        tr[ *m_colorMapMethod].add({"RedWhiteBlue", "YellowGreenBlue", "ByDialog"});
        tr[ *m_analysisMethod].add({"CoG", "2nd Moment", "MeanDev", "MeanFreqSplit"});
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
    // if(node == incrementalAverage().get())
    //     trans( *average()) = 0;
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
    tr[ *this].m_dfreq = 1.0 / coeff_freqidx; //MHz
    int32_t freqidx = lrint(coeff_freqidx * (freq - min__));
    int32_t freqidx_mid = lrint(coeff_freqidx * ((max__ + min__) / 2 - min__));
    int32_t freqidx_max = lrint(coeff_freqidx * (max__ - min__));
    tr[ *this].m_freq_mid = freqidx_mid / coeff_freqidx + min__; //MHz
    int32_t freqminusmid_sq_idx = (freqidx - freqidx_mid) * (freqidx - freqidx_mid);

    unsigned int width = shot_odmr[ *odmr__].width();
    unsigned int height = shot_odmr[ *odmr__].height();

    if(tr[ *this].m_freq_min != min__)
        clear = true;
    tr[ *this].m_freq_min = min__;
    if((int)tr[ *this].method() != tr[ *m_analysisMethod])
        clear = true;
    tr[ *this].m_method = (Method)(int)tr[ *m_analysisMethod];
    bool secondmom = tr[ *this].method() == Method::SecondMom;
    bool meansplit = tr[ *this].method() == Method::MeanFreqSplit;
    bool meandev = (tr[ *this].method() == Method::MeanDev) || meansplit;

    unsigned int num_summed_frames = tr[ *this].numSummedFrames();
    unsigned int num_total_frames = tr[ *this].numTotalFrames();

    if(clear) {
        for(unsigned int i = 0; i < MaxNumFrames; ++i)
            tr[ *this].m_summedCounts[i].reset(); //all clear incl. past frames
    }

    bool copy_prev = false;
    if( !tr[ *incrementalAverage()] && !clear && (emitter == fspectrum__.get())) {
        clear = true;
        if(std::max(1u, (unsigned int)tr[ *average()]) > tr[ *this].m_accumulatedCount)
            clear = false;
        if(clear)
            copy_prev = true;
    }
    if(tr[ *incrementalAverage()] && (tr[ *this].m_accumulatedCount > 20))
        copy_prev = true; //assuming accuracy of CoG/MeanDev is enough good.
    for(unsigned int i = num_summed_frames; i < num_total_frames; ++i) {
        if( !tr[ *this].m_summedCounts[i])
            copy_prev = true; //copy please anyway.
    }
    if(copy_prev) {
        //copies previous results for MeanDev and MeanFreqSplit.
        for(unsigned int i = 0; i < num_total_frames - num_summed_frames; ++i)
            tr[ *this].m_summedCounts[num_summed_frames + i] = tr[ *this].m_summedCounts[i];
    }

    for(unsigned int i = 0; i < num_summed_frames; ++i) {
        if( !tr[ *this].m_summedCounts[i])
            clear = true; //not yet allocated properly.
    }
    for(unsigned int i = 0; i < num_total_frames; ++i) {
        if(tr[ *this].m_summedCounts[i] && (tr[ *this].m_summedCounts[i]->size() != width * height))
            clear = true; //image size has been changed.
    }

    tr[ *this].m_width = width;
    tr[ *this].m_height = height;

    if(clear) {
        for(unsigned int i = 0; i < num_summed_frames; ++i) {
            tr[ *this].m_summedCounts[i] = m_pool.allocate(width * height);
            std::fill(tr[ *this].m_summedCounts[i]->begin(), tr[ *this].m_summedCounts[i]->end(), BaseOffset);
            tr[ *this].m_accumulatedCount = 0;
        }
        bool has_prev_frame = false;
        for(unsigned int i = num_summed_frames; i < num_total_frames; ++i) {
            if(tr[ *this].m_summedCounts[i])
                has_prev_frame = true; //C should NOT be changed.
        }
        if( !has_prev_frame) {
            //preset value for C, accepting at least 10 avg counts w/o rounding.
            tr[ *this].m_coeff_PLOn_o_Off = BaseOffset / 2 / 10;
            if(secondmom)
                tr[ *this].m_coeff_PLOn_o_Off /= freqidx_max * freqidx_max / 4;
            else
                tr[ *this].m_coeff_PLOn_o_Off /= freqidx_max / 2;
        }
    }

    int64_t coeff_PLOn_o_Off = tr[ *this].m_coeff_PLOn_o_Off; // = "C" in the following formula.

    if(emitter == fspectrum__.get()) {
        auto rawCountsPLOff = shot_odmr[ *odmr__].rawCountsPLOff();
        auto rawCountsPLOn = shot_odmr[ *odmr__].rawCountsPLOn();

        const uint32_t *pSummed[MaxNumFrames];
        decltype(m_pool.allocate(width * height)) summedCountsNext[4];
        uint32_t *pSummedNext[4];
        for(unsigned int i = 0; i < num_summed_frames; ++i) {
            pSummed[i] = &tr[ *this].m_summedCounts[i]->at(0);
            summedCountsNext[i] = m_pool.allocate(width * height);
            pSummedNext[i] = &summedCountsNext[i]->at(0);
        }
        for(unsigned int i = num_summed_frames; i < num_total_frames; ++i) {
            if( !tr[ *this].m_summedCounts[i])
                meandev = false; //no prev CoG results.
            else
                pSummed[i] = &tr[ *this].m_summedCounts[i]->at(0);
        }

        const uint32_t *pploff = &rawCountsPLOff->at(0);
        const uint32_t *pplon = &rawCountsPLOn->at(0);

        //Accumulation of results into m_suumedCounts[num_summed_frames], for CoG/second moment calc/meandev/meanfreqsplit.
        //! C*PLon/PLoff-C, freq(unit of df)*C*on/off-fC, (freq(unit of df) - fmid)^2*C*on/off-f^2 C
        //!     or |f-fcog|*C*on/off-|f-fcog|*C, ||f-fcog|-fmd|*C*on/off-||f-fcog|-fmd|*C,
        //! (prev)C*PLon/PLoff-C, (prev)fcog C, (prev)fmd C
        int32_t max_v = 0;
        for(unsigned int y  = 0; y < height; ++y) {
            for(unsigned int x  = 0; x < width; ++x) {
                uint32_t ploff = *pploff++;
                uint32_t plon = *pplon++;
                //PLon/PLoff mult. by "C", integer calc. expecting nearly 16bit resolution for PL contrast.
                //avoiding slow floating point calc.
                // CoG = <f dPL/PL> / <dPL/PL>
                //   = <f (on/off - 1)> / <on/off - 1>
                //   = (<f on/off> - <f>) / (<on/off> -  1)
                //   = (sum (f on/off C) - sum f C) / (sum on/off C -  N C)
                int32_t dpl_o_off_s32 = ploff ? (plon * coeff_PLOn_o_Off / ploff - coeff_PLOn_o_Off) : coeff_PLOn_o_Off;
                *pSummedNext[0]++ = *pSummed[0]++ + (uint32_t)dpl_o_off_s32;
                uint32_t v = *pSummed[1]++ + (uint32_t)(freqidx * dpl_o_off_s32);
                max_v = std::max(max_v, std::abs((int32_t)(v - BaseOffset)));;
                *pSummedNext[1]++ = v;
                if(secondmom) {
                    // 2ndMoment = <(f - CoG)^2 dPL/PL> / <dPL/PL>
                    //  = <f^2 dPL/PL> / <dPL/PL> - CoG^2=
                    //  = <f^2 (on/off - 1)> / <on/off - 1> - CoG^2
                    //  = (<f^2 on/off> - <f^2>) / (<on/off> -  1) - CoG^2
                    //  = (sum (f^2 on/off C) - sum f^2 C) / (sum on/off C -  N C) - CoG^2
                    //  (here CoG's origin is fmid).
                    v = *pSummed[2]++ + (uint32_t)(freqminusmid_sq_idx * dpl_o_off_s32);
                    max_v = std::max(max_v, std::abs((int32_t)(v - BaseOffset)));;
                    *pSummedNext[2]++ = v;
                }
                else if(meandev) {
                    //MeanDev = <|f - fcog| dPL/PL>/<dPL/PL> = (sum |f - fcog| on/off C - sum |f - fcog| C) / (sum on/off C -  N C)
                    // =  (sum |f C - fcog C| on/off C / C - sum |f C - fcog C|) / (sum on/off C -  N C)
                    int64_t dplopl_prev = *pSummed[num_summed_frames]++ - (int64_t)BaseOffset; //sum (on/off C) - N C
                    int64_t f_dplopl_prev = *pSummed[num_summed_frames + 1]++ - (int64_t)BaseOffset; // sum (f on/off C) - sum f C
                    int64_t fcogC_prev = dplopl_prev ? (f_dplopl_prev * coeff_PLOn_o_Off / dplopl_prev) : 0;
                    int64_t fdevC = std::abs((int64_t)freqidx * coeff_PLOn_o_Off - fcogC_prev);
                    v = *pSummed[2]++ + (int32_t)(fdevC * dpl_o_off_s32 / coeff_PLOn_o_Off);
                    max_v = std::max(max_v, std::abs((int32_t)(v - BaseOffset)));;
                    *pSummedNext[2]++ = v;
                    if(meansplit) {
                        //MeanFreqSplit = <||f - fcog| - fmd| dPL/PL>/<dPL/PL> = (sum ||f - fcog| - fmd| on/off C - sum ||f - fcog| - fmd| C) / (sum on/off C -  N C)
                        // =  (sum ||f C - fcog C| - fmd C| on/off C / C - sum ||f C - fcog C| - fmd C|) / (sum on/off C -  N C)
                        v = *pSummed[3]++ + (int32_t)(std::abs(fdevC - (*pSummed[num_summed_frames + 2]++ - (int64_t)BaseOffset)) * dpl_o_off_s32 / coeff_PLOn_o_Off);
                        max_v = std::max(max_v, std::abs((int32_t)(v - BaseOffset)));;
                        *pSummedNext[3]++ = v;
                    }
                }
            }
        }
        tr[ *this].m_accumulatedCount++;  //N
        for(unsigned int i = 0; i < num_summed_frames; ++i) {
            tr[ *this].m_summedCounts[i] = summedCountsNext[i]; //updating.
        }
        if(max_v > BaseOffset / 2) {
            //rounding by 2 to avoid overflow.
            coeff_PLOn_o_Off /= 2;
            tr[ *this].m_coeff_PLOn_o_Off = coeff_PLOn_o_Off; //for later accumulation.
            for(auto &&summed: tr[ *this].m_summedCounts) {
                if(auto v = summed)
                    for(auto &x: *v)
                        x = (uint32_t)((int32_t)(x - BaseOffset) / 2) + BaseOffset;
            }
        }
    }

    if( !shot_this[ *this].m_accumulatedCount)
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.

    tr[ *this].m_coefficients[0] = 1.0 / coeff_PLOn_o_Off / tr[ *this].m_accumulatedCount; // 1/C/N
    for(unsigned int i = 1; i < num_summed_frames; ++i) {
        tr[ *this].m_coefficients[i] = tr[ *this].m_coefficients[0] / coeff_freqidx; // 1/C/N [MHz]
    }
    if(secondmom)
        tr[ *this].m_coefficients[2] = tr[ *this].m_coefficients[0] / coeff_freqidx / coeff_freqidx; // 1/C/N [MHz^2]

    if(tr[ *m_autoMinMaxForColorMap]) {
        switch(tr[ *this].method()) {
        default:
        case Method::CoG:
            tr[ *minForColorMap()] = min__;
            tr[ *maxForColorMap()] = max__;
            break;
        case Method::SecondMom:
            tr[ *minForColorMap()] = 0;
            tr[ *maxForColorMap()] = (max__ - min__) * (max__ - min__) / 2;
        break;
        case Method::MeanDev:
        case Method::MeanFreqSplit:
            tr[ *minForColorMap()] = 0;
            tr[ *maxForColorMap()] = (max__ - min__) / 2;
            break;
        }
        tr.unmark(m_lsnOnCondChanged);
    }
    if(tr[ *incrementalAverage()]) {
        tr[ *average()] = tr[ *this].m_accumulatedCount;
        tr.unmark(m_lsnOnCondChanged);
//        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
    else {
        if(tr[ *average()] > tr[ *this].m_accumulatedCount)
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }

}

void
XODMR2DAnalysis::visualize(const Snapshot &shot) {
    if( !shot[ *this].m_accumulatedCount)
        return;

    unsigned int width = shot[ *this].width();
    unsigned int height = shot[ *this].height();
    auto qimage = std::make_shared<QImage>(width, height, QImage::Format_RGBA64);
    qimage->setColorSpace(QColorSpace::SRgbLinear); //for colormap plot.
    auto cbimage = std::make_shared<QImage>(width, 1, QImage::Format_RGBA64);
    qimage->setColorSpace(QColorSpace::SRgbLinear); //for colorbar.

    uint16_t *processed = reinterpret_cast<uint16_t*>(qimage->bits());

    unsigned int num_summed_frames = shot[ *this].numSummedFrames();

    const uint32_t *pSummed[MaxNumFrames];
    for(unsigned int i = 0; i < num_summed_frames; ++i) {
        pSummed[i] = &shot[ *this].m_summedCounts[i]->at(0);
    }

    Method method = shot[ *this].method();
    bool secondmoment = method == Method::SecondMom;

    double cmap_min = shot[ *minForColorMap()]; //MHz or MHz^2
    double cmap_max = shot[ *maxForColorMap()];

    //64bit integer calc. for CoG freq. or 2nd moment, with fullscale = 0x10000.
    constexpr int64_t coeff_vstep = 0x10000LL;
    double vstep = shot[ *this].dfreq(); //MHz
    double fmin = shot[ *this].m_freq_min; //MHz
    int64_t fmid_idx = llrint((shot[ *this].m_freq_mid - fmin) / shot[ *this].dfreq());
    int64_t value_offset;
    switch(method) {
    default:
    case Method::CoG:
        value_offset = coeff_vstep * llrint((fmin - cmap_min) / vstep);
        break;
    case Method::SecondMom:
        vstep = vstep * vstep; //MHz^2
    case Method::MeanDev:
    case Method::MeanFreqSplit:
        value_offset = coeff_vstep * llrint((-cmap_min) / vstep);
        break;
    }

    std::array<int64_t, 3> coloroffsets_low = {};
    std::array<int64_t, 3> coloroffsets_high = {};
    //64bit integer calc. for color map intensities, with fullscale = 0x100000000uLL * 0xffffuLL.
    int64_t colorgain = llrint(0x100000000uLL * 0xffffuLL * vstep / (cmap_max - cmap_min) / coeff_vstep); // /256 is needed for RGBA8888 format

    std::array<int64_t, 3> colorgain_low = {};
    std::array<int64_t, 3> colorgain_high = {};

    std::array<uint32_t, 3> colors; //low, middle, high; x - (1 - x) * (1 - alpha)

    switch((unsigned int)shot[ *m_colorMapMethod]) {
    case 0:
    default:
        //RedWhiteBlue
        // colors = {0x000080u, 0xffffffu, 0x800000u};
        colors = {0x000000bfu, 0xffffffffu, 0x00bf0000u};
        break;
    case 1:
        //YellowGreenBlue
        colors = {0xff0000ffu, 0xff00ff00u, 0xffffff00u};
        break;
    case 2:
        //By colorbar/line settings
        colors[0] = shot[ *m_processedImage->colorBarPlot()->colorPlotColorLow()];
        colors[1] = shot[ *m_processedImage->graph()->titleColor()];
        colors[2] = shot[ *m_processedImage->colorBarPlot()->colorPlotColorHigh()];
        break;
    }
    for(auto cidx: {0,1,2}) {
        std::array<int64_t, 3> intens;
        for(auto cbidx: {0,1,2}) {
            intens[cbidx] = (colors[cbidx] >> ((2 - cidx) * 8)) % 0x100u;
            intens[cbidx] -= lrint((0xff - intens[cbidx]) * (1.0 - colors[cbidx] / 0x1000000u / 255.0));
        }
        coloroffsets_low[cidx] = 0x100000000LL * 0xffffLL / 0xffLL * intens[0];
        colorgain_low[cidx] = 2 * colorgain / 0xff * (intens[1] - intens[0]);
        colorgain_high[cidx] = 2 * colorgain / 0xff * (intens[2] - intens[1]);
    }

    int64_t thres = 0x7fff * 0x100000000LL / colorgain;
    for(unsigned int cidx: {0,1,2})
        coloroffsets_high[cidx] = coloroffsets_low[cidx] + thres * (colorgain_low[cidx] - colorgain_high[cidx]);

    //constructing colormap plot.
    for(unsigned int i  = 0; i < width * height; ++i) {
        int64_t dplopl = *pSummed[0]++ - (int64_t)BaseOffset; //sum (on/off C) - N C
        int64_t f_dplopl = *pSummed[1]++ - (int64_t)BaseOffset; // sum (f on/off C) - sum f C
        if( !dplopl) {
            //avoiding division by zero. Maybe black or saturated pixel.
            *processed++ = 0; *processed++ = 0; *processed++ = 0; *processed++ = 0xffffu;
            for(unsigned int i = 2; i < num_summed_frames; ++i)
                pSummed[i]++; //for next loop.
            continue;
        }
        int64_t cmvalue;
        switch(method) {
        default:
        case Method::CoG:
            //CoG = (sum (f on/off C) - sum f C) / (sum on/off C -  N C)
            cmvalue = f_dplopl * coeff_vstep / dplopl + value_offset;
            break;
        case Method::SecondMom: {
            //  2nd mom = (sum (f^2 on/off C) - sum f^2 C) / (sum on/off C -  N C) - CoG^2
            //  (here CoG's origin is fmid).
            // = ((sum (f^2 on/off C) - sum f^2 C)*(sum on/off C -  N C) - (sum (f on/off C) - sum f C)^2) / (sum on/off C -  N C)^2
            int64_t fsq_dplopl = *pSummed[2]++ - (int64_t)BaseOffset; //sum (f^2 on/off C) - sum f^2 C
            f_dplopl -= dplopl * fmid_idx; //shifting origin to fmid.
            cmvalue = (fsq_dplopl * dplopl - f_dplopl * f_dplopl) * coeff_vstep / (dplopl * dplopl) + value_offset;
            }
            break;
        case Method::MeanDev:
            cmvalue = (*pSummed[2]++ - (int64_t)BaseOffset) * coeff_vstep / dplopl + value_offset;
            break;
        case Method::MeanFreqSplit:
            cmvalue = (*pSummed[3]++ - (int64_t)BaseOffset) * coeff_vstep / dplopl + value_offset;
            break;
        }
        const auto &color_gain = (cmvalue > thres) ? colorgain_high : colorgain_low;
        const auto &coloroffsets = (cmvalue > thres) ? coloroffsets_high : coloroffsets_low;
        for(unsigned int cidx: {0,1,2}) {
            int64_t v = (cmvalue * color_gain[cidx] + coloroffsets[cidx]) / 0x100000000LL;
            *processed++ = std::max(0LL, std::min(v, 0xffffLL));
        }
        *processed++ = 0xffffu;
    }
    //colorbar
    processed = reinterpret_cast<uint16_t*>(cbimage->bits());
    for(unsigned int i  = 0; i < cbimage->width(); ++i) {
        int64_t dcog = (double)i / (cbimage->width() - 1) * (cmap_max - cmap_min) / vstep * coeff_vstep;
        const auto &color_gain = (dcog > thres) ? colorgain_high : colorgain_low;
        const auto &coloroffsets = (dcog > thres) ? coloroffsets_high : coloroffsets_low;
        for(unsigned int cidx: {0,1,2}) {
            int64_t v = (dcog * color_gain[cidx] + coloroffsets[cidx]) / 0x100000000LL;
            *processed++ = std::max(0LL, std::min(v, 0xffffLL));
        }
        *processed++ = 0xffffu;
    }

    std::vector<double> coeffs, offsets_image;
    std::vector<const uint32_t *> rawimages;
    for(unsigned int cidx = 0; cidx < num_summed_frames; ++cidx) {
        coeffs.push_back(shot[ *this].m_coefficients[cidx]);
        rawimages.push_back( &shot[ *this].m_summedCounts[cidx]->at(0));
        offsets_image.push_back(-(double)BaseOffset * shot[ *this].m_coefficients[cidx]);
    }
    iterate_commit([&](Transaction &tr){
        tr[ *this].m_qimage = qimage;
        tr[ *m_processedImage->graph()->onScreenStrings()] = formatString("Avg:%u", (unsigned int)shot[ *this].m_accumulatedCount);
        m_processedImage->updateImage(tr, qimage, rawimages, width, coeffs, offsets_image);
        m_processedImage->updateColorBarImage(tr, cmap_min, cmap_max, cbimage);
    });
}



