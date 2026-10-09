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
#include "webcamera.h"
#include "analyzer.h"
#include <map>

REGISTER_TYPE(XDriverList, WebCamera, "USB/Built-in Video Camera (Webcam)");

namespace {

//! (label, id) pairs; identical models get " #2", " #3"... so each label
//! names one camera.  Labels are what the Device combo (and a .kam) holds.
std::vector<std::pair<XString, std::string>>
labelledDevices() {
    std::vector<std::pair<XString, std::string>> list;
    std::map<XString, unsigned int> seen;
    for(auto &&d: WebCam::enumerateDevices()) {
        XString label = d.name;
        if(unsigned int n = ++seen[label]; n > 1)
            label += formatString(" #%u", n);
        list.emplace_back(label, d.id);
    }
    return list;
}

//! The largest format up to 1920x1080 that reaches 15 fps: a camera's first
//! or largest format is often a 4K MJPEG mode it delivers at a few fps.
int
defaultFormat(const std::vector<WebCam::Format> &formats) {
    int best = 0;
    uint64_t best_area = 0;
    double best_fps = 0;
    for(unsigned int i = 0; i < formats.size(); ++i) {
        const auto &f = formats[i];
        uint64_t area = (uint64_t)f.width * f.height;
        if((area > 1920u * 1080u) || ((f.maxFPS > 0) && (f.maxFPS < 15)))
            continue;
        if((area > best_area) || ((area == best_area) && (f.maxFPS > best_fps))) {
            best = i;
            best_area = area;
            best_fps = f.maxFPS;
        }
    }
    return best;
}

//! The format's own maximum first, then round rates below it.  Rates under
//! the format's minimum are still offered: XWebCamera thins frames in software.
std::vector<double>
frameRates(const WebCam::Format &f) {
    const double fmax = (f.maxFPS > 0) ? f.maxFPS : 30.0;
    std::vector<double> rates = {fmax};
    for(double r: {120.0, 60.0, 50.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1})
        if(r < fmax * 0.99)
            rates.push_back(r);
    return rates;
}

XString
formatLabel(const WebCam::Format &f) {
    XString s = formatString("%ux%u %s", f.width, f.height, f.pixelFormat.c_str());
    if(f.maxFPS > 0)
        s += (f.minFPS > 0) && (f.minFPS < f.maxFPS - 1e-3) ?
            formatString(" %.4g-%.4g fps", f.minFPS, f.maxFPS) : formatString(" %.4g fps", f.maxFPS);
    return s;
}

} //namespace

XWebCamInterface::XWebCamInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
    XInterface(name, runtime, driver) {
    auto devices = labelledDevices();
    iterate_commit([=](Transaction &tr){
        for(auto &&d: devices)
            tr[ *device()].add(d.first);
    });
}

void
XWebCamInterface::open() {
    //Enumerated again: the id, not the label, opens a camera, and the list
    //may have changed since the combo was filled.
    const XString selected = Snapshot( *this)[ *device()].to_str();
    std::string id;
    for(auto &&d: labelledDevices())
        if(d.first == selected)
            id = d.second;
    if(id.empty())
        throw XInterfaceError(i18n("Camera not found: ") + selected, __FILE__, __LINE__);
    shared_ptr<WebCam::Camera> cam;
    try {
        cam = WebCam::openCamera(id);
    }
    catch(WebCam::Error &e) {
        throw XInterfaceError(e.what(), __FILE__, __LINE__);
    }
    XScopedLock<XMutex> lock(m_cameraMutex);
    m_camera = std::move(cam);
}

void
XWebCamInterface::close() {
    shared_ptr<WebCam::Camera> cam;
    {
        XScopedLock<XMutex> lock(m_cameraMutex);
        cam.swap(m_camera);
    }
    //Streaming stops when the last copy goes: here, or in the acquisition
    //thread if it is still holding one.
}

XWebCamera::XWebCamera(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDigitalCamera(name, runtime, ref(tr_meas), meas),
    m_interface(XNode::create<XWebCamInterface>("Interface", false,
        dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_interface);
    m_minFrameInterval = 0;
    m_singleShot = false;
    m_singleShotPending = false;
    //No webcam API offers these.
    emGain()->disable();
    triggerSrc()->disable();
    blackLvlOffset()->disable();
    if( !WebCam::backendInfo().canSetGain)
        cameraGain()->disable();
    iterate_commit([=](Transaction &tr){
        //The first two TriggerMode values; the external ones cannot be served.
        tr[ *triggerMode()].clear();
        tr[ *triggerMode()].add({"Continueous", "Single-shot"});
        m_lsnOnOpen = tr[ *interface()].onOpen().connectWeakly(
            shared_from_this(), &XWebCamera::onOpen);
        m_lsnOnClose = tr[ *interface()].onClose().connectWeakly(
            shared_from_this(), &XWebCamera::onClose);
        //XDigitalCamera applies FrameRate only with a VideoMode change;
        //here it takes effect by itself.
        m_lsnOnFrameRateChanged = tr[ *frameRate()].onValueChanged().connectWeakly(
            shared_from_this(), &XWebCamera::onFrameRateChanged);
    });
}

void
XWebCamera::onOpen(const Snapshot &shot, XInterface *) {
    try {
        open();
    }
    catch (XInterface::XInterfaceError& e) {
        e.print(getLabel() + i18n(": Opening driver failed, because "));
        onClose(shot, NULL);
    }
}
void
XWebCamera::onClose(const Snapshot &, XInterface *) {
    try {
        stop();
    }
    catch (XInterface::XInterfaceError& e) {
        e.print(getLabel() + i18n(": Stopping driver failed, because "));
        closeInterface();
    }
}
void
XWebCamera::closeInterface() {
    try {
        interface()->stop();
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}

shared_ptr<WebCam::Camera>
XWebCamera::camera() const {
    auto cam = interface()->camera();
    if( !cam)
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("The camera is not open."), __FILE__, __LINE__);
    return cam;
}

void
XWebCamera::setFrameRateItems(Transaction &tr, const std::vector<double> &rates) {
    tr[ *frameRate()].clear();
    for(double r: rates)
        tr[ *frameRate()].add(formatString("%.4g fps", r));
    tr[ *frameRate()] = 0;
    tr.unmark(m_lsnOnFrameRateChanged);
}

void
XWebCamera::open() {
    auto cam = camera();
    std::vector<WebCam::Format> formats;
    int index;
    try {
        formats = cam->formats();
        index = cam->activeFormat();
    }
    catch(WebCam::Error &e) {
        throw XInterface::XInterfaceError(getLabel() + " " + e.what(), __FILE__, __LINE__);
    }
    if(formats.empty())
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("The camera reports no video format."), __FILE__, __LINE__);
    if((index < 0) || (index >= (int)formats.size()))
        index = defaultFormat(formats);
    std::vector<XString> labels;
    for(auto &&f: formats)
        labels.push_back(formatLabel(f));
    const auto rates = frameRates(formats[index]);
    {
        XScopedLock<XMutex> lock(m_mutex);
        m_formats = formats;
        m_formatIndex = index;
        m_rates = rates;
        m_roi = {};
    }
    m_minFrameInterval = 0;
    m_singleShot = false;
    m_singleShotPending = false;
    m_lastDelivered_us = 0;
    m_fpsMeasured = 0;

    //XDigitalCamera connects its VideoMode/TriggerMode listeners in execute(),
    //i.e. after this: filling the combos here does not call back.
    iterate_commit([=](Transaction &tr){
        tr[ *videoMode()].clear();
        for(auto &&s: labels)
            tr[ *videoMode()].add(s);
        tr[ *videoMode()] = index;
        setFrameRateItems(tr, rates);
        tr[ *triggerMode()] = (unsigned int)TriggerMode::CONTINUEOUS;
    });

    try {
        cam->setFormat(index, rates[0]);
    }
    catch(WebCam::Error &e) {
        throw XInterface::XInterfaceError(getLabel() + " " + e.what(), __FILE__, __LINE__);
    }
    start();
}

void
XWebCamera::setVideoMode(unsigned int mode, unsigned int roix, unsigned int roiy, unsigned int roiw, unsigned int roih) {
    auto cam = camera();
    bool format_changed;
    std::vector<double> rates;
    {
        XScopedLock<XMutex> lock(m_mutex);
        if(mode >= m_formats.size())
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("No such video mode."), __FILE__, __LINE__);
        //A VideoMode change arrives with no ROI, so it also clears the crop.
        //A ROI is picked on the displayed, already cropped image, hence relative to the crop in force.
        m_roi = (roiw && roih) ? ROI{m_roi.x + roix, m_roi.y + roiy, roiw, roih} : ROI{};
        format_changed = ((int)mode != m_formatIndex);
        if(format_changed) {
            m_formatIndex = mode;
            m_rates = frameRates(m_formats[mode]);
            rates = m_rates;
        }
    }
    if( !format_changed)
        return;
    //A new format brings its own rates, and starts at its maximum.
    m_minFrameInterval = 0;
    iterate_commit([=](Transaction &tr){
        setFrameRateItems(tr, rates);
    });
    try {
        cam->setFormat(mode, rates[0]);
    }
    catch(WebCam::Error &e) {
        throw XInterface::XInterfaceError(getLabel() + " " + e.what(), __FILE__, __LINE__);
    }
}

void
XWebCamera::onFrameRateChanged(const Snapshot &shot, XValueNodeBase *) {
    const int idx = shot[ *frameRate()];
    double fps;
    int format;
    {
        XScopedLock<XMutex> lock(m_mutex);
        if((idx < 0) || (idx >= (int)m_rates.size()) || (m_formatIndex < 0))
            return;
        fps = m_rates[idx];
        format = m_formatIndex;
    }
    m_minFrameInterval = (idx == 0) ? 0.0 : 1.0 / fps;
    auto cam = interface()->camera();
    if( !cam)
        return; //open() applies the rate.
    try {
        cam->setFormat(format, fps); //hardware rate where the backend can; the thinning above covers the rest.
    }
    catch(WebCam::Error &e) {
        gErrPrint(getLabel() + " " + e.what());
    }
}

void
XWebCamera::setTriggerMode(TriggerMode mode) {
    switch(mode) {
    case TriggerMode::CONTINUEOUS:
        m_singleShot = false;
        break;
    case TriggerMode::SINGLE:
        m_singleShotPending = true;
        m_singleShot = true;
        break;
    default:
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Webcams have no trigger input."), __FILE__, __LINE__);
    }
}

void
XWebCamera::setGain(unsigned int g, unsigned int) {
    bool done;
    try {
        done = camera()->setGain(g);
    }
    catch(WebCam::Error &e) {
        throw XInterface::XInterfaceError(getLabel() + " " + e.what(), __FILE__, __LINE__);
    }
    if( !done)
        throw XInterface::XInterfaceError(getLabel() + " " +
            i18n("This camera has no adjustable gain; 0 selects auto."), __FILE__, __LINE__);
}

void
XWebCamera::setExposureTime(double time) {
    bool done;
    try {
        done = camera()->setExposureTime(time);
    }
    catch(WebCam::Error &e) {
        throw XInterface::XInterfaceError(getLabel() + " " + e.what(), __FILE__, __LINE__);
    }
    if( !done)
        throw XInterface::XInterfaceError(getLabel() + " " +
            i18n("This camera does not take a manual exposure time; 0 selects auto exposure."), __FILE__, __LINE__);
}

XTime
XWebCamera::acquireRaw(shared_ptr<RawData> &writer) {
    auto cam = interface()->camera();
    if( !cam)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    if( !cam->waitFrame(m_frame, 200)) //bounded, so the loop sees termination.
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    if(m_frame.luma.size() < (size_t)m_frame.width * m_frame.height)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    const int64_t ts = m_frame.timestamp_us;
    if(m_singleShot) {
        if( !m_singleShotPending.exchange(false))
            throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    }
    else {
        //5% slack, so thinning 30 fps to 10 keeps every third frame even when one arrives a little early.
        const double interval = m_minFrameInterval;
        if((interval > 0) && m_lastDelivered_us && (ts - m_lastDelivered_us < llrint(interval * 0.95e6)))
            throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    }
    if(m_lastDelivered_us && (ts > m_lastDelivered_us)) {
        const double fps = 1e6 / (ts - m_lastDelivered_us);
        m_fpsMeasured = m_fpsMeasured ? (0.8 * m_fpsMeasured + 0.2 * fps) : fps;
    }
    m_lastDelivered_us = ts;

    unsigned int x0 = 0, y0 = 0, width = m_frame.width, height = m_frame.height;
    {
        XScopedLock<XMutex> lock(m_mutex);
        if(m_roi.w && m_roi.h && (m_roi.x < width) && (m_roi.y < height)) {
            x0 = m_roi.x;
            y0 = m_roi.y;
            width = std::min(m_roi.w, width - x0);
            height = std::min(m_roi.h, height - y0);
        }
    }
    writer->push(RAW_MONO8);
    writer->push((uint32_t)width);
    writer->push((uint32_t)height);
    writer->push((uint32_t)x0);
    writer->push((uint32_t)y0);
    writer->push((uint32_t)cam->droppedFrames());
    writer->push((float)m_fpsMeasured);
    writer->reserve(writer->size() + (size_t)width * height);
    for(unsigned int y = 0; y < height; ++y) {
        auto row = reinterpret_cast<const char *>( &m_frame.luma[(size_t)(y0 + y) * m_frame.width + x0]);
        writer->insert(writer->end(), row, row + width);
    }
    return XTime{(long)(ts / 1000000), (long)(ts % 1000000)};
}

void
XWebCamera::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    if(reader.pop<uint32_t>() != RAW_MONO8)
        throw XRecordError(i18n("Unknown webcam record format."), __FILE__, __LINE__);
    const uint32_t width = reader.pop<uint32_t>();
    const uint32_t height = reader.pop<uint32_t>();
    const uint32_t x0 = reader.pop<uint32_t>();
    const uint32_t y0 = reader.pop<uint32_t>();
    const uint32_t dropped = reader.pop<uint32_t>();
    const float fps = reader.pop<float>();
    tr[ *this].m_status = formatString("%ux%u", width, height);
    if(x0 || y0)
        tr[ *this].m_status += formatString(" @(%u,%u)", x0, y0);
    tr[ *this].m_status += formatString(" %.1f fps dropped:%u", fps, dropped);
    setGrayImage(reader, tr, width, height);
}
