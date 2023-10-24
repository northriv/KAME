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
#include "odmrimaging.h"
#include "ui_odmrimagingform.h"
#include "x2dimage.h"
#include "graph.h"0
#include "graphwidget.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include "graphmathtool.h"
#include <QToolButton>
#include "graphmathtoolconnector.h"

//REGISTER_TYPE(XDriverList, ODMRImaging, "ODMR postprocessor for camera");

XODMRImaging::XODMRImaging(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSecondaryDriver(name, runtime, ref(tr_meas), meas),
    m_camera(create<XItemNode<XDriverList, XDigitalCamera> >(
          "DigitalCamera", false, ref(tr_meas), meas->drivers(), true)),
    m_average(create<XUIntNode>("Average", false)),
    m_clearAverage(create<XTouchableNode>("ClearAverage", true)),
    m_autoGainForDisp(create<XBoolNode>("AutoGainForDisp", false)),
    m_incrementalAverage(create<XBoolNode>("IncrementalAverage", false)),
    m_wheelIndex(create<XUIntNode>("WheelIndex", true)),
    m_gainForDisp(create<XDoubleNode>("GainForDisp", false)),
    m_minDPLoPLForDisp(create<XDoubleNode>("MinDPLoPLForDisp", false)),
    m_maxDPLoPLForDisp(create<XDoubleNode>("MaxDPLoPLForDisp", false)),
    m_dispMethod(create<XComboNode>("DispMethod", false, true)),
    m_refIntensFrames(create<XUIntNode>("RefIntensFrames", false)),
    m_sampleToolLists({
        create<XGraph2DMathToolList>("SmplPL", false, meas, static_pointer_cast<XDriver>(shared_from_this())),
        create<XGraph2DMathToolList>("SmplPLMWOn", false, meas, static_pointer_cast<XDriver>(shared_from_this())),
                      }),
    m_referenceToolLists({
        create<XGraph2DMathToolList>("Reference", false, meas, static_pointer_cast<XDriver>(shared_from_this())),
        create<XGraph2DMathToolList>("ReferenceMWOn", false, meas, static_pointer_cast<XDriver>(shared_from_this())),
                       }),
    m_darkToolLists({
        create<XGraph2DMathToolList>("Dark", false, meas, static_pointer_cast<XDriver>(shared_from_this())),
        create<XGraph2DMathToolList>("DarkMWOn", false, meas, static_pointer_cast<XDriver>(shared_from_this())),
                        }),
    m_form(new FrmODMRImaging),
    m_processedImage(create<X2DImage>("ProcessedImage", false,
                                   m_form->m_graphwidgetProcessed, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
                                   2,
                                   m_form->m_tbMathMenu, meas, static_pointer_cast<XDriver>(shared_from_this()))) {

    connect(camera());
    m_entries = meas->scalarEntries();

    m_form->setWindowTitle(i18n("ODMR Imaging - ") + getLabel() );

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(m_camera, m_form->m_cmbCamera, ref(tr_meas)),
        xqcon_create<XQSpinBoxUnsignedConnector>(average(), m_form->m_spbAverage),
        xqcon_create<XQSpinBoxUnsignedConnector>(wheelIndex(), m_form->m_spbWheelIndex),
        xqcon_create<XQSpinBoxUnsignedConnector>(refIntensFrames(), m_form->m_spbRefIntensFrames),
        xqcon_create<XQDoubleSpinBoxConnector>(gainForDisp(), m_form->m_dblGainForDisp),
        xqcon_create<XQDoubleSpinBoxConnector>(minDPLoPLForDisp(), m_form->m_dblMinDPL),
        xqcon_create<XQDoubleSpinBoxConnector>(maxDPLoPLForDisp(), m_form->m_dblMaxDPL),
//        xqcon_create<XQLineEditConnector>((), m_form->m_edIntegrationTime),
        xqcon_create<XQButtonConnector>(m_clearAverage, m_form->m_btnClearAverage),
        xqcon_create<XQToggleButtonConnector>(m_incrementalAverage, m_form->m_ckbIncrementalAverage),
        xqcon_create<XQToggleButtonConnector>(m_autoGainForDisp, m_form->m_ckbAutoGainForDisp),
        xqcon_create<XQComboBoxConnector>(m_dispMethod, m_form->m_cmbDispMethod, Snapshot( *m_dispMethod)),
    };

    m_conTools = {
        std::make_shared<XQGraph2DMathToolConnector>(m_sampleToolLists, m_form->m_tbSmplObjMenu, m_form->m_graphwidgetProcessed),
        std::make_shared<XQGraph2DMathToolConnector>(m_referenceToolLists, m_form->m_tbRefObjMenu, m_form->m_graphwidgetProcessed),
        std::make_shared<XQGraph2DMathToolConnector>(m_darkToolLists, m_form->m_tbDarkObjMenu, m_form->m_graphwidgetProcessed),
    };

    iterate_commit([=](Transaction &tr){
        tr[ *average()] = 1;
        tr[ *autoGainForDisp()] = true;
        tr[ *dispMethod()].add({"PL colored by dPL/PL", "dPL"});
    });

    iterate_commit([=](Transaction &tr){
        m_lsnOnClearAverageTouched = tr[ *clearAverage()].onTouch().connectWeakly(
            shared_from_this(), &XODMRImaging::onClearAverageTouched, Listener::FLAG_MAIN_THREAD_CALL);
        m_lsnOnCondChanged = tr[ *average()].onValueChanged().connectWeakly(
            shared_from_this(), &XODMRImaging::onCondChanged);
        tr[ *gainForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *minDPLoPLForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *maxDPLoPLForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *incrementalAverage()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *autoGainForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *dispMethod()].onValueChanged().connect(m_lsnOnCondChanged);
    });
}
XODMRImaging::~XODMRImaging() {
}
void
XODMRImaging::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}
void
XODMRImaging::onCondChanged(const Snapshot &shot, XValueNodeBase *node) {
    if(node == incrementalAverage().get())
        trans( *average()) = 0;
    if(node == incrementalAverage().get())
        onClearAverageTouched(shot, clearAverage().get());
    else
        requestAnalysis();
}
void
XODMRImaging::onClearAverageTouched(const Snapshot &shot, XTouchableNode *) {
    trans( *this).m_timeClearRequested = XTime::now();
    requestAnalysis();
}

bool
XODMRImaging::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) const {
    shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
    if( !camera__) return false;
    if(emitter == this) return true;
    if(emitter != camera__.get())
        return false;
//    if(shot_emitter[ *camera__->colorIndex()] != shot_this[ *wheelIndex()])
//        return false;
    return true;
}
void
XODMRImaging::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) {
    const Snapshot &shot_this(tr);

    shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
    const Snapshot &shot_camera((emitter == camera__.get()) ? shot_emitter : shot_others);

    bool clear = (shot_this[ *this].m_timeClearRequested.isSet());
    tr[ *this].m_timeClearRequested = {};

    const auto rawimage = shot_camera[ *camera__].rawCounts();
    unsigned int width = shot_camera[ *camera__].width();
    unsigned int height = shot_camera[ *camera__].height();
    if( !tr[ *incrementalAverage()] && !clear && (emitter == camera__.get())) {
        clear = true;
        for(unsigned int i: {0, 1}) {
            if(std::max(1u, (unsigned int)tr[ *average()]) > tr[ *this].m_accumulated[i])
                clear = false;
        }
    }
    for(unsigned int i: {0, 1}) {
        if( !tr[ *this].m_summedCounts[i] || (tr[ *this].m_summedCounts[i]->size() != width * height)) {
            tr[ *this].m_summedCounts[i] = make_local_shared<std::vector<uint32_t>>(width * height, 0);
            clear = true;
        }
    }
    tr[ *this].m_width = width;
    tr[ *this].m_height = height;
    if(clear) {
        for(unsigned int i: {0, 1}) {
            std::fill(tr[ *this].m_summedCounts[i]->begin(), tr[ *this].m_summedCounts[i]->end(), 0);
            tr[ *this].m_accumulated[i] = 0;
        }
    }
    if(emitter == camera__.get()) {
        unsigned int cidx = tr[ *this].currentIndex(); //MW off: 0, on: 1
        auto summedCountsNext = summedCountsFromPool(width * height);
        uint32_t *summedNext = &summedCountsNext->at(0);
        const uint32_t *summed = &tr[ *this].m_summedCounts[cidx]->at(0);

        const uint32_t *raw = &rawimage->at(0);
        for(unsigned int i  = 0; i < width * height; ++i) {
            uint64_t v = *summed++ + *raw++;
            if(v > 0x100000000uLL)
                v = 0xffffffffuL;
            *summedNext++ = v;
        }
        assert(raw == &rawimage->at(0) + width * height);
        assert(summedNext == &summedCountsNext->at(0) + width * height);
        assert(summed == &tr[ *this].m_summedCounts[cidx]->at(0) + width * height);
        (tr[ *this].m_accumulated[cidx])++;
        tr[ *this].m_summedCounts[cidx] = summedCountsNext; // = summed + live image
    }

    if( !tr[ *this].m_accumulated[0] || (tr[ *this].m_accumulated[0] != tr[ *this].m_accumulated[1]))
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.

    for(unsigned int cidx: {0,1})
        tr[ *this].m_coefficients[cidx] = 1.0 / tr[ *this].m_accumulated[cidx]; //for math tools

    if(tr[ *m_autoGainForDisp]) {
        const uint32_t *summed[2];
        for(unsigned int cidx: {0,1}) {
            summed[cidx] = &tr[ *this].m_summedCounts[cidx]->at(0);
        }
        int32_t dpl_min = 0x7fffffffL, dpl_max = -0x7fffffffL;
        uint32_t vmin = 0xffffffffu;
        uint32_t vmax = 0u;
        double dplopl_min, dplopl_max;
        for(unsigned int i  = 0; i < width * height; ++i) {
            uint32_t v = *summed[0]++;
            if(v > vmax)
                vmax = v;
            if(v < vmin)
                vmin = v;
            int32_t dpl = (int32_t)(*summed[1]++ - v);
            if(dpl > dpl_max) {
                dpl_max = dpl;
                dplopl_max = (double)dpl / v;
            }
            if(dpl < dpl_min) {
                dpl_min = dpl;
                dplopl_min = (double)dpl / v;
            }
        }
        assert(summed[0] == &tr[ *this].m_summedCounts[0]->at(0) + width * height);
        assert(summed[1] == &tr[ *this].m_summedCounts[1]->at(0) + width * height);

        if(vmax > 0) {
            tr[ *gainForDisp()]  = lrint((double)0xffffu / (vmax / tr[ *this].m_accumulated[0]));
            tr[ *minDPLoPLForDisp()]  = 100.0 * dplopl_min;
            tr[ *maxDPLoPLForDisp()]  = 100.0 * dplopl_max;
            tr.unmark(m_lsnOnCondChanged);
        }
    }

    {
        const uint32_t *summed[2];
        for(unsigned int cidx: {0,1}) {
            summed[cidx] = &tr[ *this].m_summedCounts[cidx]->at(0);
        }
        unsigned int stride = width;
        for(unsigned int cidx: {0,1}) {
            m_sampleToolLists[cidx]->update(tr, summed[cidx], stride, stride, height, shot_this[ *this].m_coefficients[cidx]);
            m_referenceToolLists[cidx]->update(tr, summed[cidx], stride, stride, height, shot_this[ *this].m_coefficients[cidx]);
            m_darkToolLists[cidx]->update(tr, summed[cidx], stride, stride, height, shot_this[ *this].m_coefficients[cidx]);
        }
        auto fn_tool_to_vector = [&](std::vector<double>&vec, const shared_ptr<XGraph2DMathToolList> &toollist, double dark = 0) {
            vec.clear();
            auto &list = *shot_this.list(toollist);
            unsigned int tot_pixels = 0;
            for(unsigned int i = 0; i < shot_this.size(toollist); ++ i) {
                auto sumtool = dynamic_pointer_cast<XGraph2DMathToolSum>(list.at(i));
                if(sumtool) {
                    unsigned int pix = sumtool->pixels(shot_this);
                    vec.push_back(shot_this[ *sumtool->entry()->value()] - dark * pix);
                    tot_pixels += pix;
                }
                auto avgtool = dynamic_pointer_cast<XGraph2DMathToolAverage>(list.at(i));
                if(avgtool) {
                    unsigned int pix = avgtool->pixels(shot_this);
                    vec.push_back((shot_this[ *avgtool->entry()->value()] - dark) * pix); //converts avg to sum.
                    tot_pixels += pix;
                }
            }
            return tot_pixels;
        };
        for(auto *x: {&tr[ *this].m_referenceIntensities, &tr[ *this].m_sampleIntensities}) {
            for(unsigned int cidx: {0,1}) {
                (*x)[cidx].clear();
            }
        }
        double darks[2] = {};
        if(shot_this.size(m_darkToolLists[0]) &&
                shot_this.size(m_darkToolLists[0]) == shot_this.size(m_darkToolLists[1])) {
            for(unsigned int cidx: {0,1}) {
                std::vector<double> vec;
                unsigned int pixels = fn_tool_to_vector(vec, m_darkToolLists[cidx]);
                darks[cidx] = std::accumulate(vec.begin(), vec.end(), 0) / pixels;
            }
        }
        if(shot_this.size(m_referenceToolLists[0]) &&
            shot_this.size(m_referenceToolLists[0]) == shot_this.size(m_referenceToolLists[1])) {
            for(unsigned int cidx: {0,1}) {
                fn_tool_to_vector(tr[ *this].m_referenceIntensities[cidx], m_referenceToolLists[cidx], darks[cidx]);
            }
        }
        if(shot_this.size(m_sampleToolLists[0]) &&
                shot_this.size(m_sampleToolLists[0]) == shot_this.size(m_sampleToolLists[1])) {
            for(unsigned int cidx: {0,1}) {
                fn_tool_to_vector(tr[ *this].m_sampleIntensities[cidx], m_sampleToolLists[cidx], darks[cidx]);
            }

            tr[ *this].m_pl0 = shot_this[ *this].m_sampleIntensities[0]; //stores orig.
            analyzeIntensities(tr);

            if(auto entries = m_entries.lock()) {
                auto &list = *shot_this.list(m_sampleToolLists[0]);
                unsigned int j = 0;
                for(unsigned int i = 0; i < list.size(); ++ i) {
                    shared_ptr<XScalarEntry> entryPL, entryDPLoPL;
                    shared_ptr<XNode> tool;
                    if(auto sumtool = dynamic_pointer_cast<XGraph2DMathToolSum>(list.at(i)))
                        tool = sumtool;
                    if(auto avgtool = dynamic_pointer_cast<XGraph2DMathToolAverage>(list.at(i)))
                        tool = avgtool;
                    if( !tool)
                        continue;
                    entryPL = m_samplePLEntries[tool.get()];
                    if( !entryPL) {
                        entryPL = create<XScalarEntry>(ref(tr), formatString("Smpl%u,PL", i).c_str(), true,
                            dynamic_pointer_cast<XDriver>(shared_from_this()));
                        m_samplePLEntries[tool.get()] = entryPL;
                    }
                    else
//                        entryPL->value(ref(tr), shot_this[ *this].m_pl0[j]);
                        entryPL->value(ref(tr), shot_this[ *this].m_sampleIntensities[0][j]);
                    entryDPLoPL = m_sampleDPLoPLEntries[tool.get()];
                    if( !entryDPLoPL) {
                        entryDPLoPL = create<XScalarEntry>(ref(tr), formatString("Smpl%u,dPL/PL", i).c_str(), true,
                            dynamic_pointer_cast<XDriver>(shared_from_this()));
                        m_sampleDPLoPLEntries[tool.get()] = entryDPLoPL;
                    }
                    else
                        entryDPLoPL->value(ref(tr), shot_this[ *this].dPLoPL(j));
                    j++;
                }
            }
        }
    }

    if(tr[ *incrementalAverage()]) {
        tr[ *average()] = tr[ *this].m_accumulated[1];
        tr.unmark(m_lsnOnCondChanged);
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
    else {
        if(tr[ *average()] > tr[ *this].m_accumulated[1])
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
}
void
XODMRImaging::visualize(const Snapshot &shot) {
    if(auto entries = m_entries.lock()) {
        //inserts new entries
        Snapshot shot_entries(*entries);
        if(shot_entries.size()) {
            auto &list = *shot_entries.list();
            for(auto &&x: m_samplePLEntries) {
                if(std::find(list.begin(), list.end(), x.second) == list.end()) {
                    entries->insert(x.second);
                }
            }
            for(auto &&x: m_sampleDPLoPLEntries) {
                if(std::find(list.begin(), list.end(), x.second) == list.end()) {
                    entries->insert(x.second);
                }
            }
        }
        //releases unused entries.
        if(shot.size(m_sampleToolLists[0])) {
            auto &list = *shot.list(m_sampleToolLists[0]);
            for(auto it = m_samplePLEntries.begin(); it != m_samplePLEntries.end();) {
                auto tool_it = std::find_if(list.begin(), list.end(), [&it](const shared_ptr<XNode> &x){return it->first == x.get();});
                if(tool_it == list.end()) {
                    entries->release(it->second);
                    release(it->second);
                    it = m_samplePLEntries.erase(it); //not existing anymore.
                }
                else {
                    it++;
                }
            }
            for(auto it = m_sampleDPLoPLEntries.begin(); it != m_sampleDPLoPLEntries.end();) {
                auto tool_it = std::find_if(list.begin(), list.end(), [&it](const shared_ptr<XNode> &x){return it->first == x.get();});
                if(tool_it == list.end()) {
                    entries->release(it->second);
                    release(it->second);
                    it = m_sampleDPLoPLEntries.erase(it); //not existing anymore.
                }
                else {
                    it++;
                }
            }
        }
        else {
            for(auto &&x: m_samplePLEntries) {
                entries->release(x.second);
                release(x.second);
            }
            m_samplePLEntries.clear();
            for(auto &&x: m_sampleDPLoPLEntries) {
                entries->release(x.second);
                release(x.second);
            }
            m_sampleDPLoPLEntries.clear();
        }
    }

    if( !shot[ *this].m_accumulated[0] || (shot[ *this].m_accumulated[0] != shot[ *this].m_accumulated[1]))
        return;
    unsigned int width = shot[ *this].width();
    unsigned int height = shot[ *this].height();
    auto qimage = std::make_shared<QImage>(width, height, QImage::Format_RGBA8888);

    {
        uint64_t gain_av = lrint(0x100000000uLL * shot[ *gainForDisp()] / 256.0 * shot[ *this].m_coefficients[0]);
        double dpl_min = shot[ *minDPLoPLForDisp()] / 100.0;
        double dpl_max = shot[ *maxDPLoPLForDisp()] / 100.0;
        int64_t dpl_gain_pos[3] = {};
        int64_t dpl_gain_neg[3] = {};
        switch((unsigned int)shot[ *m_dispMethod]) {
        case 0:
        default:
            //Colored by DPL/PL
            gain_av /= 2; //max. 128 * 0x100000000uLL for autogain.
            dpl_gain_pos[0] = lrint(gain_av / dpl_max);
            dpl_gain_pos[1] = -dpl_gain_pos[0];
            dpl_gain_pos[2] = -dpl_gain_pos[0];
            dpl_gain_neg[2] = lrint(gain_av / dpl_min); //negative
            dpl_gain_neg[0] = -dpl_gain_neg[2];
            dpl_gain_neg[1] = -dpl_gain_neg[2];
            break;
        case 1:
            //DPL red for positive, blue for negative
            dpl_gain_pos[0] = lrint(gain_av / dpl_max);
            dpl_gain_neg[2] = lrint(gain_av / dpl_min); //negative
            gain_av = 0;
            break;
        }
        uint8_t *processed = reinterpret_cast<uint8_t*>(qimage->bits());
        const uint32_t *summed[2];
        for(unsigned int cidx: {0,1})
            summed[cidx] = &shot[ *this].m_summedCounts[cidx]->at(0);

        for(unsigned int i  = 0; i < width * height; ++i) {
            int32_t dpl = *summed[1] - *summed[0];
            const int64_t *dpl_gain = (dpl > 0) ? dpl_gain_pos : dpl_gain_neg;
            for(unsigned int cidx: {0,1,2}) {
                int64_t v = ((int64_t)(*summed[0] * gain_av) + dpl * dpl_gain[cidx])  / 0x100000000LL;
                *processed++ = std::max(0LL, std::min(v, 0xffLL));
            }
            *processed++ = 0xffu;
            for(unsigned int cidx: {0,1})
                (summed[cidx])++;
        }
        assert(processed == qimage->constBits() + width * height * 4);
        assert(summed[0] == &shot[ *this].m_summedCounts[0]->at(0) + width * height);
        assert(summed[1] == &shot[ *this].m_summedCounts[1]->at(0) + width * height);
    }


    std::vector<double> coeffs;
    std::vector<const uint32_t *> rawimages;
    for(unsigned int cidx: {0,1}) {
        coeffs.push_back(shot[ *this].m_coefficients[cidx]);
        rawimages.push_back( &shot[ *this].m_summedCounts[cidx]->at(0));
    }
    iterate_commit([&](Transaction &tr){
        tr[ *this].m_qimage = qimage;
        tr[ *m_processedImage->graph()->onScreenStrings()] = formatString("Avg:%u", (unsigned int)shot[ *this].m_accumulated[0]);
        m_processedImage->updateImage(tr, qimage, rawimages, coeffs);
    });
}

local_shared_ptr<std::vector<uint32_t>>
XODMRImaging::summedCountsFromPool(int imagesize) {
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


