/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
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
#include "odmrimaging.h"
#include "ui_odmrimagingform.h"
#include "x2dimage.h"
#include "graph.h"
#include "graphwidget.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include "graphmathtool.h"
#include <QToolButton>
#include "graphmathtoolconnector.h"
#include <QColorSpace>

XODMRImaging::XODMRImaging(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSecondaryDriver(name, runtime, ref(tr_meas), meas),
    m_camera(create<XItemNode<XDriverList, XDigitalCamera> >(
          "DigitalCamera", false, ref(tr_meas), meas->drivers(), true)),
    m_filterWheel(create<XItemNode<XDriverList, XFilterWheel> >(
          "FilterWheel", false, ref(tr_meas), meas->drivers(), false)),
    m_average(create<XUIntNode>("Average", false)),
    m_precedingSkips(create<XUIntNode>("PrecedingSkips", false)),
    m_clearAverage(create<XTouchableNode>("ClearAverage", true)),
    m_autoGainForDisp(create<XBoolNode>("AutoGainForDisp", false)),
    m_incrementalAverage(create<XBoolNode>("IncrementalAverage", false)),
    m_filterIndex(create<XUIntNode>("FilterIndex", false)),
    m_gainForDisp(create<XDoubleNode>("GainForDisp", false)),
    m_minDPLoPLForDisp(create<XDoubleNode>("MinDPLoPLForDisp", false)),
    m_maxDPLoPLForDisp(create<XDoubleNode>("MaxDPLoPLForDisp", false)),
    m_dispMethod(create<XComboNode>("DispMethod", false, true)),
    m_refIntensFrames(create<XUIntNode>("RefIntensFrames", false)),
    m_sequence(create<XComboNode>("Sequence", false, true)),
    m_binning(create<XUIntNode>("Binning", false)),
    m_form(new FrmODMRImaging),
    m_processedImage(create<X2DImage>("ProcessedImage", false,
                                   m_form->m_graphwidgetProcessed, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
                                   2, m_form->m_dblGamma,
                                   m_form->m_tbMathMenu, meas, static_pointer_cast<XDriver>(shared_from_this()),
                                   true)) {

    auto plot = m_processedImage->plot();
    m_sampleToolLists = {
        create<XGraph2DMathToolList>("SmplPLMWOFF-2", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("SmplPLMWOFF-1", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("SmplPLMWOFF", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("SmplPLMWOn", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
                      };
    m_referenceToolLists = {
        create<XGraph2DMathToolList>("RefMWOFF-2", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("RefMWOFF-1", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("RefMWOFF", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("ReferenceMWOn", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
                       };
    m_darkToolLists = {
        create<XGraph2DMathToolList>("DarkMWOFF-2", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("DarkMWOFF-1", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("DarkMWOFF", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
        create<XGraph2DMathToolList>("DarkMWOn", false, meas, static_pointer_cast<XDriver>(shared_from_this()), plot),
                        };

    connect(camera());
    connect(filterWheel());
    m_entries = meas->scalarEntries();

    m_form->setWindowTitle(i18n("ODMR Imaging - ") + getLabel() );

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(m_camera, m_form->m_cmbCamera, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(m_filterWheel, m_form->m_cmbFilterWheel, ref(tr_meas)),
        xqcon_create<XQSpinBoxUnsignedConnector>(average(), m_form->m_spbAverage),
        xqcon_create<XQSpinBoxUnsignedConnector>(precedingSkips(), m_form->m_spbSkipPreceding),
        xqcon_create<XQSpinBoxUnsignedConnector>(filterIndex(), m_form->m_spbFilterIndex),
        xqcon_create<XQSpinBoxUnsignedConnector>(refIntensFrames(), m_form->m_spbRefIntensFrames),
        xqcon_create<XQDoubleSpinBoxConnector>(gainForDisp(), m_form->m_dblGainForDisp),
        xqcon_create<XQDoubleSpinBoxConnector>(minDPLoPLForDisp(), m_form->m_dblMinDPL),
        xqcon_create<XQDoubleSpinBoxConnector>(maxDPLoPLForDisp(), m_form->m_dblMaxDPL),
//        xqcon_create<XQLineEditConnector>((), m_form->m_edIntegrationTime),
        xqcon_create<XQButtonConnector>(m_clearAverage, m_form->m_btnClearAverage),
        xqcon_create<XQToggleButtonConnector>(m_incrementalAverage, m_form->m_ckbIncrementalAverage),
        xqcon_create<XQToggleButtonConnector>(m_autoGainForDisp, m_form->m_ckbAutoGainForDisp),
        xqcon_create<XQComboBoxConnector>(m_dispMethod, m_form->m_cmbDispMethod, Snapshot( *m_dispMethod)),
        xqcon_create<XQComboBoxConnector>(m_sequence, m_form->m_cmbSequence, Snapshot( *m_sequence)),
        xqcon_create<XQSpinBoxUnsignedConnector>(binning(), m_form->m_spbBinning),
    };

    m_conTools = {
        std::make_shared<XQGraph2DMathToolConnector>(m_sampleToolLists, m_form->m_tbSmplObjMenu, m_form->m_graphwidgetProcessed),
        std::make_shared<XQGraph2DMathToolConnector>(m_referenceToolLists, m_form->m_tbRefObjMenu, m_form->m_graphwidgetProcessed),
        std::make_shared<XQGraph2DMathToolConnector>(m_darkToolLists, m_form->m_tbDarkObjMenu, m_form->m_graphwidgetProcessed),
    };

    iterate_commit([=](Transaction &tr){
        for(auto &&x: m_sampleToolLists)
            x->setBaseColor(0xffff00u);
        for(auto &&x: m_referenceToolLists)
            x->setBaseColor(0x00ffffu);
        tr[ *average()] = 1;
        tr[ *precedingSkips()] = 0;
        tr[ *binning()] = 1;
        tr[ *autoGainForDisp()] = true;
        tr[ *dispMethod()].add({"dPL(RedWhiteBlue)", "dPL(YellowGreenBlue)",
            "dPL/PL(RedWhiteBlue)", "dPL/PL(YellowGreenBlue)",
            "dPL(By Dialog)", "dPL/PL(By Dialog)"});
        tr[ *sequence()].add({"OFF,ON", "OFF,OFF,ON", "OFF,OFF,OFF,ON"});
    });

    iterate_commit([=](Transaction &tr){
        m_lsnOnClearAverageTouched = tr[ *clearAverage()].onTouch().connectWeakly(
            shared_from_this(), &XODMRImaging::onClearAverageTouched);
        m_lsnOnCondChanged = tr[ *average()].onValueChanged().connectWeakly(
            shared_from_this(), &XODMRImaging::onCondChanged);
        tr[ *gainForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *minDPLoPLForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *maxDPLoPLForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *incrementalAverage()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *autoGainForDisp()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *dispMethod()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *sequence()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *binning()].onValueChanged().connect(m_lsnOnCondChanged);
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
    if((node == incrementalAverage().get()) || (node == binning().get()))
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
    //ignores old camera frames
    if((shot_emitter[ *camera__].time() < shot_this[ *this].m_timeClearRequested) &&
        shot_this[ *this].m_timeClearRequested - shot_emitter[ *camera__].time() < 60.0) //not reading raw binary
        return false;
//    shared_ptr<XFilterWheel> wheel__ = shot_this[ *filterWheel()];
//    if(wheel__)
//        if(shot_others[ *wheel__].wheelIndex() != shot_this[ *wheelIndex()])
//            return false;
    return true;
}
void
XODMRImaging::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) {

    tr[ *this].m_releasedEntries.clear();

    const Snapshot &shot_this(tr);

    shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
    const Snapshot &shot_camera((emitter == camera__.get()) ? shot_emitter : shot_others);

    bool clear = (shot_this[ *this].m_timeClearRequested.isSet());
    tr[ *this].m_timeClearRequested = {};

    const auto rawimage = shot_camera[ *camera__].rawCounts();
    unsigned int binning__ = std::max((unsigned int)shot_this[ *binning()], 1u);
    unsigned int width_camera = shot_camera[ *camera__].width();
    unsigned int width = width_camera / binning__;
    unsigned int height_camera = shot_camera[ *camera__].height();
    unsigned int height = height_camera / binning__;
    unsigned int raw_stride = shot_camera[ *camera__].stride();
    auto seq = std::map<unsigned int, Sequence>
        {{0, Sequence::OFF_ON},{1, Sequence::OFF_OFF_ON},{2, Sequence::OFF_OFF_OFF_ON}
        }.at(shot_this[ *sequence()]);
    if(tr[ *this].sequence() != seq) {
        clear = true;
        tr[ *this].m_sequence = seq;
    }
    unsigned int seq_len = shot_this[ *this].sequenceLength();
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
    }
    tr[ *this].m_width = width;
    tr[ *this].m_height = height;
    if(clear) {
        for(unsigned int i = 0; i < seq_len; ++i) {
            tr[ *this].m_summedCounts[i] = m_pool.allocate(width * height);
            std::fill(tr[ *this].m_summedCounts[i]->begin(), tr[ *this].m_summedCounts[i]->end(), 0);
            tr[ *this].m_accumulated[i] = 0;
        }
        tr[ *this].m_skippedFrames = 0;
    }
    if(emitter == camera__.get()) {
        shared_ptr<XFilterWheel> wheel__ = shot_this[ *filterWheel()];
        if(wheel__) {
            int wheelidx = shot_others[ *wheel__].wheelIndexOfFrame(
                shot_camera[ *camera__].time(), shot_camera[ *camera__].timeAwared());
            if(wheelidx != shot_this[ *filterIndex()])
                throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
        }

        if(shot_this[ *this].m_skippedFrames < shot_this[ *precedingSkips()] * seq_len) {
            tr[ *this].m_skippedFrames++;
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
        }
        unsigned int cidx = shot_this[ *this].currentIndex();
        auto summedCountsNext = m_pool.allocate(width * height);
        uint32_t *summedNext = &summedCountsNext->at(0);
        const uint32_t *summed = &tr[ *this].m_summedCounts[cidx]->at(0);

        const uint32_t *raw = &rawimage->at(shot_camera[ *camera__].firstPixel());
        for(unsigned int y  = 0; y < height; ++y) {
            for(unsigned int x  = 0; x < width; ++x) {
                uint64_t v = *summed++;
                for(unsigned int bin_dy = 0; bin_dy < binning__; ++bin_dy)
                    for(unsigned int bin_dx = 0; bin_dx < binning__; ++bin_dx)
                        v += raw[bin_dy * raw_stride + bin_dx];
                if(v > 0x100000000uLL)
                    v = 0xffffffffuL;
                *summedNext++ = v;
                raw += binning__;
            }
            raw += raw_stride * binning__ - width * binning__;
        }
        assert(summedNext == &summedCountsNext->at(0) + width * height);
        assert(summed == &tr[ *this].m_summedCounts[cidx]->at(0) + width * height);
        (tr[ *this].m_accumulated[cidx])++;
        tr[ *this].m_summedCounts[cidx] = summedCountsNext; // = summed + live image
    }

    if( !shot_this[ *this].m_accumulated[0] || (shot_this[ *this].currentIndex() > 0))
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.

    for(unsigned int i = 0; i < seq_len; ++i)
        tr[ *this].m_coefficients[i] = 1.0 / tr[ *this].m_accumulated[i]; //for math tools

    if(tr[ *m_autoGainForDisp]) {
        const uint32_t *summed[2];
        for(unsigned int cidx: {0,1}) {
            summed[cidx] = &tr[ *this].m_summedCounts[seq_len - 2 + cidx]->at(0);
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
        assert(summed[0] == &tr[ *this].m_summedCounts[seq_len - 2 + 0]->at(0) + width * height);
        assert(summed[1] == &tr[ *this].m_summedCounts[seq_len - 2 + 1]->at(0) + width * height);

        if(vmax > 0) {
            tr[ *gainForDisp()]  = (double)0xffffu / (vmax / tr[ *this].m_accumulated[0]);
            tr[ *minDPLoPLForDisp()]  = 100.0 * dplopl_min;
            tr[ *maxDPLoPLForDisp()]  = 100.0 * dplopl_max;
            tr.unmark(m_lsnOnCondChanged);
        }
    }
    if(tr[ *incrementalAverage()]) {
        tr[ *average()] = tr[ *this].m_accumulated[seq_len - 1];
        tr.unmark(m_lsnOnCondChanged);
        throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }
    else {
        if(tr[ *average()] > tr[ *this].m_accumulated[seq_len - 1])
            throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
    }

    {
        const uint32_t *summed[4];
        for(unsigned int i = 0; i < seq_len; ++i) {
            summed[i] = &tr[ *this].m_summedCounts[i]->at(0);
        }
        unsigned int stride = width;
        for(unsigned int i = 0; i < seq_len; ++i) {
            unsigned int tidx = 4 - seq_len + i;
            m_sampleToolLists[tidx]->update(ref(tr), m_form->m_graphwidgetProcessed, summed[i], stride, stride, height, shot_this[ *this].m_coefficients[i]);
            m_referenceToolLists[tidx]->update(ref(tr), m_form->m_graphwidgetProcessed, summed[i], stride, stride, height, shot_this[ *this].m_coefficients[i]);
            m_darkToolLists[tidx]->update(ref(tr), m_form->m_graphwidgetProcessed, summed[i], stride, stride, height, shot_this[ *this].m_coefficients[i]);
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
            for(unsigned int i = 0; i < seq_len; ++i) {
                (*x)[i].clear();
            }
        }
        double darks[4] = {};
        if(shot_this.size(m_darkToolLists[0])) {
            for(unsigned int i = 0; i < seq_len; ++i) {
                unsigned int tidx = 4 - seq_len + i;
                if(shot_this.size(m_darkToolLists[0]) != shot_this.size(m_darkToolLists[tidx]))
                    throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
                std::vector<double> vec;
                unsigned int pixels = fn_tool_to_vector(vec, m_darkToolLists[tidx]);
                darks[i] = std::accumulate(vec.begin(), vec.end(), 0) / pixels;
            }
        }
        if(shot_this.size(m_referenceToolLists[0])) {
            for(unsigned int i = 0; i < seq_len; ++i) {
                unsigned int tidx = 4 - seq_len + i;
                if(shot_this.size(m_referenceToolLists[0]) != shot_this.size(m_referenceToolLists[tidx]))
                    throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
                fn_tool_to_vector(tr[ *this].m_referenceIntensities[i], m_referenceToolLists[tidx], darks[i]);
            }
        }
        if(shot_this.size(m_sampleToolLists[0])) {
            for(unsigned int i = 0; i < seq_len; ++i) {
                unsigned int tidx = 4 - seq_len + i;
                if(shot_this.size(m_sampleToolLists[0]) != shot_this.size(m_sampleToolLists[tidx]))
                    throw XSkippedRecordError(__FILE__, __LINE__); //visualize() will be called.
                fn_tool_to_vector(tr[ *this].m_sampleIntensities[i], m_sampleToolLists[tidx], darks[i]);
                tr[ *this].m_sampleIntensitiesCorrected[i] = shot_this[ *this].m_sampleIntensities[i]; //copy
            }
            analyzeIntensities(tr);

//            if(auto entries = m_entries.lock()) {
            {
                //creats new entry for each math tool which is not listed in the map.
//                Snapshot shot_entries(*entries);
//                auto &entry_list = *shot_entries.list();
                auto &list = *shot_this.list(m_sampleToolLists[0]);
                unsigned int j = 0;
                for(unsigned int i = 0; i < list.size(); ++i) {
                    shared_ptr<XScalarEntry> entryPL, entryDPLoPL;
                    shared_ptr<XNode> tool;
                    if(auto sumtool = dynamic_pointer_cast<XGraph2DMathToolSum>(list.at(i)))
                        tool = sumtool;
                    if(auto avgtool = dynamic_pointer_cast<XGraph2DMathToolAverage>(list.at(i)))
                        tool = avgtool;
                    if( !tool)
                        continue;
                    entryPL = tr[ *this].m_samplePLEntries[tool.get()];
                    if( !entryPL) {
                        entryPL = create<XScalarEntry>(ref(tr), formatString("Smpl%u,PL", i).c_str(), true,
                            dynamic_pointer_cast<XDriver>(shared_from_this()));
                        if( !entryPL)
                            return;
                        tr[ *this].m_samplePLEntries[tool.get()] = entryPL;
                    }
                    if(shot_this[ *this].m_sampleIntensities[seq_len - 2].size() > j) //confirms a number of mathtools is uniform.
                        entryPL->value(ref(tr), shot_this[ *this].pl0(j)); //update value

                    entryDPLoPL = tr[ *this].m_sampleDPLoPLEntries[tool.get()];
                    if( !entryDPLoPL) {
                        entryDPLoPL = create<XScalarEntry>(ref(tr), formatString("Smpl%u,dPL/PL", i).c_str(), true,
                            dynamic_pointer_cast<XDriver>(shared_from_this()));
                        if( !entryDPLoPL)
                            return;
                        tr[ *this].m_sampleDPLoPLEntries[tool.get()] = entryDPLoPL;
                    }
                    if(shot_this[ *this].m_sampleIntensities[seq_len - 2].size() > j)
                        if(shot_this[ *this].m_sampleIntensities[seq_len - 1].size() > j) //confirms a number of mathtools is uniform.
                            entryDPLoPL->value(ref(tr), shot_this[ *this].dPLoPL(j)); //update value

                    j++;
                }

                //removes entry in the map which no more exists in the math tool list.
                for(auto &&e :
                    {ref(tr[ *this].m_samplePLEntries), ref(tr[ *this].m_sampleDPLoPLEntries)}) {
                    for(auto it = e.get().begin(); it != e.get().end();) {
                        auto tool_it = std::find_if(list.begin(), list.end(), [&it](const shared_ptr<XNode> &x){return it->first == x.get();});
                        if(tool_it == list.end()) {
                            tr[ *this].m_releasedEntries.push_back(it->second);
                            it = e.get().erase(it); //not existing anymore.
                        }
                        else
                            it++;
                    }
                }
            }
        }
        else {
            //no sample. releases all the entries.
            for(auto &&x: tr[ *this].m_samplePLEntries) {
                tr[ *this].m_releasedEntries.push_back(x.second);
            }
            tr[ *this].m_samplePLEntries.clear();
            for(auto &&x: tr[ *this].m_sampleDPLoPLEntries) {
                tr[ *this].m_releasedEntries.push_back(x.second);
            }
            tr[ *this].m_sampleDPLoPLEntries.clear();
        }
        //iterating/erasing map should be separated.
        for(auto &&x: tr[ *this].m_releasedEntries)
            if( !release(tr, x)) //map may be copy-constructed after release()., losing iterator.
                return;
    }
}
void
XODMRImaging::visualize(const Snapshot &shot) {
    if(auto entries = m_entries.lock()) {
        //inserts new entries
        Snapshot shot_entries(*entries);
        if(shot_entries.size()) {
            auto &list = *shot_entries.list();
            for(auto &&x: shot[ *this].m_samplePLEntries) {
                if(std::find(list.begin(), list.end(), x.second) == list.end()) {
                    entries->insert(x.second);
                }
            }
            for(auto &&x: shot[ *this].m_sampleDPLoPLEntries) {
                if(std::find(list.begin(), list.end(), x.second) == list.end()) {
                    entries->insert(x.second);
                }
            }
            for(auto &&x: shot[ *this].m_releasedEntries) {
                entries->release(x);
            }
        }
    }

    if( !shot[ *this].m_accumulated[0] || (shot[ *this].currentIndex() > 0))
        return;

    unsigned int width = shot[ *this].width();
    unsigned int height = shot[ *this].height();
    auto qimage = std::make_shared<QImage>(width, height, QImage::Format_RGBA64);
    qimage->setColorSpace(QColorSpace::SRgbLinear);
    auto cbimage = std::make_shared<QImage>(width, 1, QImage::Format_RGBA64);
    cbimage->setColorSpace(QColorSpace::SRgbLinear);

    unsigned int seq_len = shot[ *this].sequenceLength();

    double dpl_min = shot[ *minDPLoPLForDisp()] / 100.0;
    double dpl_max = shot[ *maxDPLoPLForDisp()] / 100.0;
    {
        uint64_t coeff_dpl = 0x100000000uLL * 0xffffuLL; // /256 is needed for RGBA8888 format
        uint64_t gain_av = llrint(coeff_dpl / 0xffffuLL
            * std::max((double)shot[ *gainForDisp()], 0.1) * shot[ *this].m_coefficients[0]);
        std::array<uint64_t, 3> gains = {}; //offsets for dPL/PL
        std::array<int64_t, 3> dpl_gain_pos = {};
        std::array<int64_t, 3> dpl_gain_neg = {};
        int64_t denom_coeff_PL = 0;

        std::array<uint32_t, 3> colors; //neg, zero, pos; x - (1 - x) * (1 - alpha)

        switch((unsigned int)shot[ *m_dispMethod]) {
        case 2:
            //"dPL/PL(RedWhiteBlue)"
            denom_coeff_PL = coeff_dpl / gain_av;
            // colors = {0x0000ffu, 0xffffffu, 0xff0000u};
            colors = {0x000000bfu, 0xffffffffu, 0x00bf0000u};
            break;
        case 0:
            //"dPL/(RedWhiteBlue)"
            //Colored by DPL/PL
            colors = {0xff0000ffu, 0xffffffffu, 0xffff0000u};
            break;
        default:
        case 3:
            //"dPL/PL(YellowGreenBlue)"
            //Colored by DPL/PL
            denom_coeff_PL = coeff_dpl / gain_av;
        case 1:
            //"dPL(YellowGreenBlue)"
            //DPL yellow for positive, blue for negative
            colors = {0xff0000ffu, 0xff00ff00u, 0xffffff00u};
            break;
        case 5:
            //"dPL/PL(By Dialog)"
            //Colored by DPL/PL
            denom_coeff_PL = coeff_dpl / gain_av;
        case 4:
            //"dPL(By Dialog)"
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
            gains[cidx] = gain_av / 0xffLL * intens[1];//max. 0x7fffuL( or 0x7fu) * 0x100000000uLL for autogain.
            dpl_gain_pos[cidx] = llrint(gain_av / dpl_max / 0xffLL * (intens[2] - intens[1]));
            dpl_gain_neg[cidx] = llrint(gain_av / dpl_min / 0xffLL * (intens[0] - intens[1]));
        }

        uint16_t *processed = reinterpret_cast<uint16_t*>(qimage->bits());
        const uint32_t *summed[2];
        for(unsigned int cidx: {0,1})
            summed[cidx] = &shot[ *this].m_summedCounts[seq_len - 2 + cidx]->at(0);

        for(unsigned int i  = 0; i < width * height; ++i) {
            int64_t pl0 = *summed[0]++;
            int64_t dpl = (int64_t)*summed[1]++ - pl0;
            if(denom_coeff_PL) {
                if( !pl0) {
                    *processed++ = 0; *processed++ = 0; *processed++ = 0; *processed++ = 0xffffu;
                    continue;
                }
                dpl = dpl * denom_coeff_PL / pl0;
                pl0 = coeff_dpl / gain_av;
            }
            const auto &dpl_gain = (dpl > 0) ? dpl_gain_pos : dpl_gain_neg;
            for(unsigned int cidx: {0,1,2}) {
                int64_t v = ((int64_t)(pl0 * gains[cidx]) + dpl * dpl_gain[cidx])  / 0x100000000LL;
//                *processed++ = std::max(0LL, std::min(v, 0xffLL));
                *processed++ = std::max(0LL, std::min(v, 0xffffLL));
            }
            *processed++ = 0xffffu;
        }
        assert(processed == (uint16_t *)qimage->constBits() + width * height * 4);
        assert(summed[0] == &shot[ *this].m_summedCounts[seq_len - 2]->at(0) + width * height);
        assert(summed[1] == &shot[ *this].m_summedCounts[seq_len - 1]->at(0) + width * height);

        //colorbar
        processed = reinterpret_cast<uint16_t*>(cbimage->bits());
        for(unsigned int i  = 0; i < cbimage->width(); ++i) {
            int64_t pl0 = coeff_dpl / gain_av;
            int64_t dpl = ((double)i / (cbimage->width() - 1) * (dpl_max - dpl_min) + dpl_min) * pl0;
            const auto &dpl_gain = (dpl > 0) ? dpl_gain_pos : dpl_gain_neg;
            for(unsigned int cidx: {0,1,2}) {
                int64_t v = ((int64_t)(pl0 * gains[cidx]) + dpl * dpl_gain[cidx])  / 0x100000000LL;
//                *processed++ = std::max(0LL, std::min(v, 0xffLL));
                *processed++ = std::max(0LL, std::min(v, 0xffffLL));
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
        tr[ *m_processedImage->graph()->onScreenStrings()] = formatString("Avg:%u", (unsigned int)shot[ *this].m_accumulated[0]);
        m_processedImage->updateImage(tr, qimage, rawimages, width, coeffs);
        m_processedImage->updateColorBarImage(tr, dpl_min * 100.0, dpl_max * 100.0, cbimage);
    });
}


