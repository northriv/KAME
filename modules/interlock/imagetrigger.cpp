/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "imagetrigger.h"
#include "ui_imagetriggerform.h"
#include "digitalcamera.h"
#include "graphntoolbox.h"
#include "x2dimage.h"
#include "graph.h"
#include <QColorSpace>
#include <QStatusBar>
#include <QImage>
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include <algorithm>

REGISTER_TYPE(XDriverList, ImageTrigger, "Image Trigger (saves camera frames)");

XImageTrigger::XImageTrigger(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_camera(create<tCamera>("Camera", false, ref(tr_meas), meas->drivers())),
    m_consecutive(create<XUIntNode>("Consecutive", false)),
    m_holdOff(create<XDoubleNode>("HoldOff", false)),
    m_framesBefore(create<XUIntNode>("FramesBefore", false)),
    m_framesAfter(create<XUIntNode>("FramesAfter", false)),
    m_interval(create<XDoubleNode>("Interval", false)),
    m_fileName(create<XStringNode>("FileName", false)),
    m_maxFiles(create<XUIntNode>("MaxFiles", false)),
    m_status(create<XStringNode>("Status", true)),
    m_entryEvents(create<XScalarEntry>("Events", false,
        dynamic_pointer_cast<XDriver>(shared_from_this()), "%.0f")),
    m_form(new FrmImageTrigger) {

    //What was saved last, shown beside the settings: a display, not a record --
    //it is set directly, never through a raw record, so a replay leaves it be.
    m_lastShot = create<X2DImage>("LastShot", false, m_form->m_graphwidget,
        nullptr, nullptr, nullptr, m_form->m_dblGamma);
    for(unsigned int i = 0; i < NumConditions; ++i)
        m_conditions.push_back(create<XInterlockCondition>(
            formatString("Condition%u", i + 1).c_str(), false, ref(tr_meas), meas->scalarEntries()));
    m_armed = create<XBoolNode>("Armed", false);

    meas->scalarEntries()->insert(tr_meas, m_entryEvents);

    m_form->statusBar()->hide();
    m_form->setWindowTitle(i18n("Image Trigger - ") + getLabel());

    QComboBox *cmb_entries[NumConditions] = {m_form->m_cmbEntry1, m_form->m_cmbEntry2};
    QComboBox *cmb_modes[NumConditions] = {m_form->m_cmbMode1, m_form->m_cmbMode2};
    QLineEdit *ed_thresholds[NumConditions] = {m_form->m_edThreshold1, m_form->m_edThreshold2};
    m_conUIs = {
        xqcon_create<XQToggleButtonConnector>(m_armed, m_form->m_ckbArmed),
        xqcon_create<XQLabelConnector>(m_status, m_form->m_lblStatus),
        xqcon_create<XQComboBoxConnector>(m_camera, m_form->m_cmbCamera, ref(tr_meas)),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_consecutive, m_form->m_spbConsecutive),
        xqcon_create<XQLineEditConnector>(m_holdOff, m_form->m_edHoldOff),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_framesBefore, m_form->m_spbFramesBefore),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_framesAfter, m_form->m_spbFramesAfter),
        xqcon_create<XQLineEditConnector>(m_interval, m_form->m_edInterval),
        xqcon_create<XFilePathConnector>(m_fileName, m_form->m_edFileName, m_form->m_tbFileName,
            "JPEG images (*.jpg);;PNG images, lossless (*.png);;All files (*.*)", true),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_maxFiles, m_form->m_spbMaxFiles),
    };
    for(unsigned int i = 0; i < NumConditions; ++i) {
        auto &c = m_conditions[i];
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(c->entry(), cmb_entries[i], ref(tr_meas)));
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(c->mode(), cmb_modes[i], Snapshot( *c->mode())));
        m_conUIs.push_back(xqcon_create<XQLineEditConnector>(c->threshold(), ed_thresholds[i]));
    }

    iterate_commit([=](Transaction &tr){
        tr[ *m_consecutive] = 3;
        tr[ *m_holdOff] = 10.0;
        tr[ *m_framesBefore] = 3;
        tr[ *m_framesAfter] = 5;
        tr[ *m_interval] = 1.0;
        tr[ *m_maxFiles] = 0;
        tr[ *m_status] = i18n("Disarmed");
        m_lsnOnArmed = tr[ *m_armed].onValueChanged().connectWeakly(
            shared_from_this(), &XImageTrigger::onArmedChanged);
        m_lsnOnCamera = tr[ *m_camera].onValueChanged().connectWeakly(
            shared_from_this(), &XImageTrigger::onCameraChanged);
        m_lsnOnSetting = tr[ *m_interval].onValueChanged().connectWeakly(
            shared_from_this(), &XImageTrigger::onSettingChanged);
        tr[ *m_framesBefore].onValueChanged().connect(m_lsnOnSetting);
        tr[ *m_framesAfter].onValueChanged().connect(m_lsnOnSetting);
    });
    onSettingChanged(Snapshot( *this), nullptr);
}

void
XImageTrigger::onArmedChanged(const Snapshot &shot, XValueNodeBase *) {
    if(shot[ *m_armed] && !m_running.exchange(true))
        start();
}

void
XImageTrigger::onSettingChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    m_intervalCache = std::max(0.02, (double)shot[ *m_interval]);
    //Enough for the frames before an event and those after it, which are
    //taken from the ring as they arrive.
    m_ringCapacity = (unsigned int)shot[ *m_framesBefore] + (unsigned int)shot[ *m_framesAfter] + 2;
}

void
XImageTrigger::onCameraChanged(const Snapshot &, XValueNodeBase *) {
    shared_ptr<XDigitalCamera> cam = Snapshot( *this)[ *m_camera];
    m_lsnOnCameraRecord.reset();
    {
        XScopedLock<XMutex> lock(m_ringMutex);
        m_ring.clear();
        m_lastKept = XTime();
    }
    if( !cam)
        return;
    cam->iterate_commit([=](Transaction &tr){
        m_lsnOnCameraRecord = tr[ *cam].onRecord().connectWeakly(
            shared_from_this(), &XImageTrigger::onCameraRecord);
    });
}

void
XImageTrigger::onCameraRecord(const Snapshot &shot, XDriver *driver) {
    //Inside the camera's commit, on its acquisition thread: a pointer copy
    //under a short lock, and nothing else.
    auto cam = dynamic_cast<XDigitalCamera *>(driver);
    if( !cam)
        return;
    const auto &p = shot[ *cam];
    auto counts = p.rawCounts();
    const XTime t = p.time();
    if( !counts || !t.isSet() || !p.width() || !p.height())
        return;
    XScopedLock<XMutex> lock(m_ringMutex);
    //One frame per Interval: holding every frame of a 30 fps camera for a few
    //seconds would be hundreds of megabytes.
    //diff_usec, not diff_sec: the latter is whole seconds, and kept the
    //spacing from going below one.
    if(m_lastKept.isSet() && (t.diff_usec(m_lastKept) * 1e-6 < m_intervalCache * 0.95))
        return;
    Frame f;
    f.counts = counts;
    f.width = p.width();
    f.height = p.height();
    f.stride = p.stride();
    f.firstPixel = p.firstPixel();
    f.time = t;
    f.seq = ++m_lastSeq;
    m_ring.push_back(std::move(f));
    m_lastKept = t;
    while(m_ring.size() > m_ringCapacity)
        m_ring.pop_front();
}

QImage
XImageTrigger::frameImage(const Frame &frame, bool lossless) {
    const size_t needed = frame.firstPixel + (size_t)(frame.height - 1) * frame.stride + frame.width;
    if( !frame.counts || (frame.counts->size() < needed))
        return {};
    uint32_t vmax = 0;
    for(unsigned int y = 0; y < frame.height; ++y) {
        const uint32_t *src = &( *frame.counts)[frame.firstPixel + (size_t)y * frame.stride];
        for(unsigned int x = 0; x < frame.width; ++x)
            vmax = std::max(vmax, src[x]);
    }
    //PNG keeps the counts themselves, losslessly: 8-bit when they fit (a
    //webcam's luminance), 16-bit otherwise (a scientific camera's linear
    //counts).  JPEG is 8-bit and lossy, a tenth of the size for watching a
    //room: counts above 255 are scaled down to fit, for looking at, not for
    //measuring.
    const bool eight = !lossless || (vmax <= 0xffu);
    const uint32_t shift = (eight && (vmax > 0xffu)) ? [vmax]{
            uint32_t s = 0;
            while((vmax >> s) > 0xffu) ++s;
            return s;
        }() : 0;
    QImage img(frame.width, frame.height, eight ? QImage::Format_Grayscale8 : QImage::Format_Grayscale16);
    for(unsigned int y = 0; y < frame.height; ++y) {
        const uint32_t *src = &( *frame.counts)[frame.firstPixel + (size_t)y * frame.stride];
        if(eight) {
            uchar *dst = img.scanLine(y);
            for(unsigned int x = 0; x < frame.width; ++x)
                dst[x] = (uchar)(src[x] >> shift);
        }
        else {
            auto dst = reinterpret_cast<uint16_t *>(img.scanLine(y));
            for(unsigned int x = 0; x < frame.width; ++x)
                dst[x] = (uint16_t)std::min(src[x], 0xffffu);
        }
    }
    //8 bits as they came are the camera's own encoding (a webcam's, gamma
    //already applied); anything else is linear counts, which the display's
    //Gamma then encodes.
    img.setColorSpace((eight && !shift) ? QColorSpace::SRgb : QColorSpace::SRgbLinear);
    return img;
}

XString
XImageTrigger::save(const QImage &img, const XTime &time, const XString &templ) {
    //The format is FileName's extension (frameImage() was told whether it is PNG).
    const QString suffix = QFileInfo(QString(templ.c_str())).suffix().toLower();
    const QString path = XGraphNToolBox::nextNumberedPath(QString(templ.c_str()), m_fileSeq, time);
    return XGraphNToolBox::writeAtomically(path, [&](const QString &tmp) {
        //Qt guesses the format from the name; the temporary's ends in ".part".
        return (suffix == "png") ? img.save(tmp, "PNG") : img.save(tmp, suffix.toLatin1().constData(), 90);
    });
}

void
XImageTrigger::showLastShot(const QImage &img, const XTime &time, unsigned int event) {
    auto image = std::make_shared<QImage>(img); //shares the pixels; nothing is copied.
    const XString caption = formatString("Event #%u  ", event) + time.getTimeFmtStr("%Y-%m-%d %H:%M:%S", false);
    m_lastShot->iterate_commit([&](Transaction &tr){
        m_lastShot->updateQImage(tr, image);
        tr[ *m_lastShot->graph()->onScreenStrings()] = caption;
    });
}

void
XImageTrigger::enforceMaxFiles(const XString &templ, unsigned int max_files) {
    //Only this series' own files: "<stem>_<number>_..." with its extension.
    const QFileInfo t(QString(templ.c_str()));
    const QString stem = t.completeBaseName(), suffix = t.suffix();
    QDir dir(t.absolutePath());
    const QRegularExpression re("^" + QRegularExpression::escape(stem) + "_(\\d+)_");
    std::vector<std::pair<unsigned int, QString>> files;
    for(auto &&name: dir.entryList({stem + "_*." + suffix}, QDir::Files)) {
        auto m = re.match(name);
        if(m.hasMatch())
            files.emplace_back(m.captured(1).toUInt(), name);
    }
    if(files.size() <= max_files)
        return;
    std::sort(files.begin(), files.end());
    for(size_t i = 0; i < files.size() - max_files; ++i)
        dir.remove(files[i].second);
}

void
XImageTrigger::writeRecord(State state, unsigned int events) {
    auto writer = std::make_shared<RawData>();
    writer->push((uint32_t)state);
    writer->push((uint32_t)events);
    const XTime now = XTime::now();
    finishWritingRaw(writer, now, now);
}

void
XImageTrigger::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    const uint32_t state = reader.pop<uint32_t>();
    const uint32_t events = reader.pop<uint32_t>();
    if(state > (uint32_t)State::HoldOff)
        throw XRecordError(i18n("Unknown trigger state."), __FILE__, __LINE__);
    tr[ *this].m_state = (State)state;
    tr[ *this].m_events = events;
    m_entryEvents->value(tr, events);
}

void *
XImageTrigger::execute(const atomic<bool> &terminated) {
    constexpr int64_t TICK_MS = 50;
    constexpr double SILENT_S = 2.0; //!< a source quieter than this is reported, never acted on.
    constexpr int64_t HEARTBEAT_NS = 1'000'000'000;

    bool was_armed = false;
    State state = State::Disarmed, last_state = State::Disarmed;
    unsigned int events = Snapshot( *this)[ *this].events();
    int64_t holdoff_until = 0, last_record_ns = 0;
    std::vector<Frame> pending;
    uint64_t last_taken = 0;
    unsigned int after_left = 0;
    XString shown, last_error;

    auto show = [&](const XString &status) {
        if(status == shown)
            return;
        trans( *m_status) = status;
        shown = status;
    };

    while( !terminated) {
        const int64_t now = XInterlockCondition::steadyNS();
        Snapshot shot( *this);
        if( !shot[ *m_armed]) {
            was_armed = false;
            state = State::Disarmed;
            pending.clear();
            show(i18n("Disarmed"));
        }
        else {
            if( !was_armed) {
                for(auto &c: m_conditions)
                    c->restartWatchdog(now);
                state = State::Watching;
                holdoff_until = 0;
                last_error.clear();
                was_armed = true;
            }
            const unsigned int consecutive = std::max(1u, (unsigned int)shot[ *m_consecutive]);
            unsigned int watched = 0;
            bool hit = false;
            XString silent;
            for(unsigned int i = 0; i < NumConditions; ++i) {
                auto &c = m_conditions[i];
                if( !c->isEnabled())
                    continue;
                ++watched;
                if(c->isSilent(now, SILENT_S)) {
                    if(silent.empty())
                        silent = formatString("Condition%u", i + 1);
                }
                else if(c->hitStreak() >= consecutive)
                    hit = true;
            }
            const XString templ = shot[ *m_fileName].to_str();
            if(templ != m_fileTemplate) {
                m_fileTemplate = templ;
                m_fileSeq = 0; //another series: look its numbers up afresh.
            }

            if(state != State::Capturing) {
                state = (now < holdoff_until) ? State::HoldOff : State::Watching;
                if((state == State::Watching) && hit) {
                    ++events;
                    {
                        XScopedLock<XMutex> lock(m_ringMutex);
                        const size_t n = std::min((size_t)(unsigned int)shot[ *m_framesBefore], m_ring.size());
                        pending.assign(m_ring.end() - n, m_ring.end());
                        last_taken = m_lastSeq;
                    }
                    after_left = shot[ *m_framesAfter];
                    state = State::Capturing;
                }
            }
            if(state == State::Capturing) {
                {
                    XScopedLock<XMutex> lock(m_ringMutex);
                    for(auto &f: m_ring) {
                        if( !after_left)
                            break;
                        if(f.seq > last_taken) {
                            pending.push_back(f);
                            last_taken = f.seq;
                            --after_left;
                        }
                    }
                }
                //Outside the lock and any transaction: encoding and writing take a while.
                const bool png = (QFileInfo(QString(templ.c_str())).suffix().toLower() == "png");
                QImage shown;
                XTime shown_time;
                for(auto &f: pending) {
                    QImage img = frameImage(f, png);
                    if(img.isNull()) {
                        last_error = i18n("frame size mismatch");
                        continue;
                    }
                    if(templ.length()) {
                        XString err = save(img, f.time, templ);
                        if(err.length())
                            last_error = err;
                    }
                    shown = img;
                    shown_time = f.time;
                }
                if(templ.length() && pending.size() && (unsigned int)shot[ *m_maxFiles])
                    enforceMaxFiles(templ, shot[ *m_maxFiles]);
                if( !shown.isNull())
                    showLastShot(shown, shown_time, events);
                pending.clear();
                if( !after_left) {
                    state = State::HoldOff;
                    holdoff_until = now + (int64_t)(std::max(0.0, (double)shot[ *m_holdOff]) * 1e9);
                }
            }

            //formatString() translates its format itself.
            XString status;
            switch(state) {
            case State::Capturing:
                status = formatString("Event #%u: %u frame(s) to go", events, after_left);
                break;
            case State::HoldOff:
                status = formatString("Event #%u saved; watching again in %.0f s", events,
                    (holdoff_until - now) * 1e-9);
                break;
            default:
                if( !watched)
                    status = i18n("Armed, but no condition is on");
                else if(silent.length())
                    status = formatString("Watching; no data from %s", silent.c_str());
                else
                    status = formatString("Watching: %u condition(s), %u event(s) so far", watched, events);
                break;
            }
            shared_ptr<XDigitalCamera> cam = shot[ *m_camera];
            if( !cam)
                status += XString(" -- ") + XString(i18n("no camera chosen"));
            else if(templ.empty())
                status += XString(" -- ") + XString(i18n("no FileName: nothing is saved"));
            if(last_error.length())
                status += " -- " + last_error;
            show(status);
        }
        if((state != last_state) || (now - last_record_ns > HEARTBEAT_NS)) {
            writeRecord(state, events);
            last_state = state;
            last_record_ns = now;
        }
        msecsleep(TICK_MS);
    }
    //Stopped from outside (Measurement > Stop, the driver released): untick
    //Armed rather than leave it ticked with nothing watching.
    iterate_commit([&](Transaction &tr){
        tr[ *m_armed] = false;
        tr[ *m_status] = i18n("Disarmed (measurement stopped)");
    });
    writeRecord(State::Disarmed, events);
    m_running = false;
    return nullptr;
}
