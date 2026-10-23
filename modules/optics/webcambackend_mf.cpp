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
// WebCam backend over Media Foundation (Windows).
//
// Native, as AVFoundation is on macOS: no Qt Multimedia module to install,
// no FFmpeg to ship, and a YUV frame's luma plane is read as it comes.
//
// Threading: each camera owns one worker thread, in the multithreaded COM
// apartment, which creates the source reader and alone calls it: it reads
// frames in synchronous mode and, between frames, runs what the other
// threads ask (format, exposure, gain) through a queue, with a bounded wait
// on their side.  Closing tells the worker to stop and waits for it to
// finish the frame it is reading.  Shutting the media source down from the
// closing thread does not end a synchronous ReadSample() -- it hangs it for
// good -- so a camera that sends nothing more has its worker let go, with
// the state it shares, rather than block the caller.
//
// Exposure and gain go through the UVC controls the capture source exposes
// as IAMCameraControl / IAMVideoProcAmp.  Their exposure unit is log2
// seconds, so a manual exposure time is rounded to a power of two.
#include "webcambackend.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <strmif.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <thread>
#include <tuple>

namespace {

int64_t unixMicrosecondsNow() {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

//! Owning COM pointer, enough for this file.
template <class T>
class ComPtr {
public:
    ComPtr() = default;
    ComPtr(const ComPtr &x) : m_p(x.m_p) {if(m_p) m_p->AddRef();}
    ~ComPtr() {reset();}
    ComPtr &operator=(const ComPtr &x) {
        if(x.m_p) x.m_p->AddRef();
        reset();
        m_p = x.m_p;
        return *this;
    }
    void reset() {if(m_p) m_p->Release(); m_p = nullptr;}
    T **put() {reset(); return &m_p;}
    T *get() const {return m_p;}
    T *operator->() const {return m_p;}
    explicit operator bool() const {return m_p != nullptr;}
    template <class U>
    ComPtr<U> as(REFIID iid) const {
        ComPtr<U> u;
        if(m_p) m_p->QueryInterface(iid, reinterpret_cast<void**>(u.put()));
        return u;
    }
private:
    T *m_p = nullptr;
};

std::string hresultText(HRESULT hr) {
    char buf[32];
    snprintf(buf, sizeof(buf), "0x%08lx", (unsigned long)hr);
    std::string s = buf;
    if(hr == E_ACCESSDENIED)
        s += ": camera access is denied. Allow desktop apps to use the camera in"
             " Windows Settings > Privacy & security > Camera.";
    else if(hr == MF_E_HW_MFT_FAILED_START_STREAMING)
        s += ": the camera is in use by another application.";
    else if(hr == MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED)
        s += ": the camera has been disconnected.";
    return s;
}
void check(HRESULT hr, const char *what) {
    if(FAILED(hr))
        throw WebCam::Error(std::string(what) + " failed, " + hresultText(hr));
}

std::string toUtf8(const wchar_t *w) {
    if( !w)
        return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if(n <= 1)
        return {};
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}
std::wstring toWide(const std::string &s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if(n <= 1)
        return {};
    std::wstring w(n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

//! The calling thread in the multithreaded apartment, for as long as it lives.
//! A thread already in a single-threaded one (Qt's main thread) is left as it
//! is: Media Foundation's objects are free-threaded.
class ComScope {
public:
    ComScope() : m_hr(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
    ~ComScope() {if(SUCCEEDED(m_hr)) CoUninitialize();}
private:
    HRESULT m_hr;
};

//! MFStartup once per process.  Never shut down: a camera torn down at exit
//! may still be in flight.
void startMediaFoundation() {
    static const HRESULT hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
    check(hr, "MFStartup");
}

//! Runs \a f on a fresh MTA thread and returns its result (or rethrows).
template <typename F>
auto inMTA(F f) -> decltype(f()) {
    using R = decltype(f());
    std::packaged_task<R()> task([&f]{ComScope com; return f();});
    auto result = task.get_future();
    std::thread(std::move(task)).join();
    return result.get();
}

struct Subtype {
    GUID guid;
    const char *name;
    enum class Luma {Plane8, Y0Packed, Y1Packed, None} luma;
};
//! Uncompressed formats whose luma is read in place; anything else is decoded to NV12.
const Subtype kSubtypes[] = {
    {MFVideoFormat_NV12, "NV12", Subtype::Luma::Plane8},
    {MFVideoFormat_I420, "I420", Subtype::Luma::Plane8},
    {MFVideoFormat_IYUV, "IYUV", Subtype::Luma::Plane8},
    {MFVideoFormat_YV12, "YV12", Subtype::Luma::Plane8},
    {MFVideoFormat_L8, "L8", Subtype::Luma::Plane8},
    {MFVideoFormat_YUY2, "YUY2", Subtype::Luma::Y0Packed},
    {MFVideoFormat_YVYU, "YVYU", Subtype::Luma::Y0Packed},
    {MFVideoFormat_UYVY, "UYVY", Subtype::Luma::Y1Packed},
    {MFVideoFormat_MJPG, "MJPG", Subtype::Luma::None},
    {MFVideoFormat_H264, "H264", Subtype::Luma::None},
    {MFVideoFormat_RGB24, "RGB24", Subtype::Luma::None},
    {MFVideoFormat_RGB32, "RGB32", Subtype::Luma::None},
};
const Subtype *findSubtype(const GUID &g) {
    for(auto &&s: kSubtypes)
        if(IsEqualGUID(s.guid, g))
            return &s;
    return nullptr;
}
std::string subtypeName(const GUID &g) {
    if(auto s = findSubtype(g))
        return s->name;
    //A FOURCC subtype keeps its code in Data1.
    char fcc[5] = {(char)(g.Data1 & 0xff), (char)((g.Data1 >> 8) & 0xff),
        (char)((g.Data1 >> 16) & 0xff), (char)((g.Data1 >> 24) & 0xff), 0};
    for(int i = 0; i < 4; ++i)
        if((fcc[i] < 0x20) || (fcc[i] > 0x7e))
            return "?";
    return fcc;
}

double frameRateOf(IMFMediaType *t, const GUID &key) {
    UINT32 num = 0, den = 0;
    if(FAILED(MFGetAttributeRatio(t, key, &num, &den)) || !den)
        return 0;
    return (double)num / den;
}

//! A device format as offered to XWebCamera: one size and subtype, with the
//! native media types that give it at each frame rate.
struct NativeFormat {
    WebCam::Format format;
    GUID subtype;
    std::vector<std::pair<double, DWORD>> rates; //!< (fps, native type index), ascending
};

//! The opened device and the thread that alone drives it.  Shared between
//! the Camera and that thread, so that a thread which will not end (a
//! ReadSample() waiting on a stalled camera, which shutting the source down
//! does not end in synchronous mode) can be let go without its state.
class Device {
public:
    explicit Device(std::shared_ptr<WebCam::FrameSlot> slot) : m_slot(std::move(slot)) {}
    void run(std::string symlink, std::promise<void> ready);
    //! Runs  f on the worker and waits for it, up to 10 s.
    template <typename R>
    R call(std::function<R()> f);
    void stop() {
        m_stop = true;
        m_queueCond.notify_all();
    }

    //Worker thread only:
    void apply(int index, double fps);
    bool exposure(double sec);
    bool gain(double g);

    std::vector<WebCam::Format> m_formats; //!< set before run() reports ready, read-only after.
    std::atomic<int> m_active{-1};
private:
    void open(const std::string &symlink);
    HRESULT readOne();
    bool extractLuma(IMFSample *sample);

    const std::shared_ptr<WebCam::FrameSlot> m_slot;
    std::atomic<bool> m_stop{false};
    std::mutex m_queueMutex;
    std::condition_variable m_queueCond;
    std::deque<std::function<void()>> m_queue;

    //Worker-only.
    ComPtr<IMFMediaSource> m_source;
    ComPtr<IMFSourceReader> m_reader;
    ComPtr<IAMCameraControl> m_cameraControl;
    ComPtr<IAMVideoProcAmp> m_procAmp;
    std::vector<NativeFormat> m_native;
    bool m_streaming = false;
    Subtype::Luma m_outLuma = Subtype::Luma::None;
    unsigned int m_outWidth = 0, m_outHeight = 0;
    WebCam::Frame m_buf; //!< recycled through FrameSlot::post().
    unsigned int m_failures = 0;
};

class MFCamera : public WebCam::Camera {
public:
    explicit MFCamera(const std::string &symlink);
    ~MFCamera() override;

    std::vector<WebCam::Format> formats() const override {return m_dev->m_formats;}
    int activeFormat() const override {return m_dev->m_active;}
    void setFormat(int index, double fps) override;
    bool setExposureTime(double sec) override;
    bool setGain(double gain) override;
private:
    const std::shared_ptr<Device> m_dev;
    std::thread m_thread;
    std::future<void> m_ended;
};

MFCamera::MFCamera(const std::string &symlink) : m_dev(std::make_shared<Device>(m_slot)) {
    std::promise<void> ready, ended;
    auto opened = ready.get_future();
    m_ended = ended.get_future();
    m_thread = std::thread([dev = m_dev, symlink](std::promise<void> ready, std::promise<void> ended) {
        dev->run(symlink, std::move(ready));
        ended.set_value();
    }, std::move(ready), std::move(ended));
    //Not bounded: opening a device waits on nothing of ours.
    try {
        opened.get();
    }
    catch(...) {
        m_thread.join();
        throw;
    }
}

MFCamera::~MFCamera() {
    m_dev->stop();
    //The worker ends after the frame it is waiting for: a frame interval, or
    //a manual exposure.  A camera that sends nothing more never lets it; the
    //thread keeps the device then, and leaks it rather than block the caller.
    if(m_ended.wait_for(std::chrono::seconds(5)) == std::future_status::ready)
        m_thread.join();
    else {
        fprintf(stderr, "WebCam: the camera did not stop; leaving its thread.\n");
        m_thread.detach();
    }
}

void
MFCamera::setFormat(int index, double fps) {
    if((index < 0) || (index >= (int)m_dev->m_formats.size()))
        throw WebCam::Error("No such video format.");
    Device *dev = m_dev.get();
    m_dev->call<void>([dev, index, fps]{dev->apply(index, fps);});
}
bool
MFCamera::setExposureTime(double sec) {
    Device *dev = m_dev.get();
    return m_dev->call<bool>([dev, sec]{return dev->exposure(sec);});
}
bool
MFCamera::setGain(double gain) {
    Device *dev = m_dev.get();
    return m_dev->call<bool>([dev, gain]{return dev->gain(gain);});
}

void
Device::run(std::string symlink, std::promise<void> ready) {
    ComScope com;
    try {
        startMediaFoundation();
        open(symlink);
        ready.set_value();
    }
    catch(...) {
        ready.set_exception(std::current_exception());
        return;
    }
    while( !m_stop) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            if(m_queue.empty() && !m_streaming)
                m_queueCond.wait_for(lock, std::chrono::milliseconds(200));
            if( !m_queue.empty()) {
                job = std::move(m_queue.front());
                m_queue.pop_front();
            }
        }
        if(job) {
            job();
            continue;
        }
        if(m_streaming && !m_stop)
            readOne();
    }
    m_reader.reset();
    m_cameraControl.reset();
    m_procAmp.reset();
    if(m_source)
        m_source->Shutdown();
    m_source.reset();
}

template <typename R>
R
Device::call(std::function<R()> f) {
    auto task = std::make_shared<std::packaged_task<R()>>(std::move(f));
    auto result = task->get_future();
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_queue.push_back([task]{(*task)();});
    }
    m_queueCond.notify_all();
    //A ReadSample() in progress delays the job by up to a frame; long manual
    //exposures make frames slow, hence the margin.
    if(result.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
        throw WebCam::Error("The camera thread did not respond within 10 s.");
    return result.get();
}

void
Device::open(const std::string &symlink) {
    ComPtr<IMFAttributes> attr;
    check(MFCreateAttributes(attr.put(), 2), "MFCreateAttributes");
    check(attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID), "SetGUID");
    const std::wstring wlink = toWide(symlink);
    check(attr->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, wlink.c_str()), "SetString");
    check(MFCreateDeviceSource(attr.get(), m_source.put()), "Opening the camera (MFCreateDeviceSource)");

    ComPtr<IMFAttributes> rattr;
    check(MFCreateAttributes(rattr.put(), 1), "MFCreateAttributes");
    //Lets the reader convert what a decoder hands out into the output type asked.
    rattr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    check(MFCreateSourceReaderFromMediaSource(m_source.get(), rattr.get(), m_reader.put()),
        "MFCreateSourceReaderFromMediaSource");
    m_reader->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE);
    m_reader->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);

    m_cameraControl = m_source.as<IAMCameraControl>(IID_IAMCameraControl);
    m_procAmp = m_source.as<IAMVideoProcAmp>(IID_IAMVideoProcAmp);

    //Native types, grouped by size and subtype; each fixed rate (or range) is one type.
    std::map<std::tuple<UINT32, UINT32, std::string>, size_t> index;
    for(DWORD i = 0; ; ++i) {
        ComPtr<IMFMediaType> t;
        if(FAILED(m_reader->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, i, t.put())))
            break;
        GUID major = {}, sub = {};
        if(FAILED(t->GetGUID(MF_MT_MAJOR_TYPE, &major)) || !IsEqualGUID(major, MFMediaType_Video))
            continue;
        if(FAILED(t->GetGUID(MF_MT_SUBTYPE, &sub)))
            continue;
        UINT32 w = 0, h = 0;
        if(FAILED(MFGetAttributeSize(t.get(), MF_MT_FRAME_SIZE, &w, &h)) || !w || !h)
            continue;
        double fps = frameRateOf(t.get(), MF_MT_FRAME_RATE);
        double fmin = frameRateOf(t.get(), MF_MT_FRAME_RATE_RANGE_MIN);
        double fmax = frameRateOf(t.get(), MF_MT_FRAME_RATE_RANGE_MAX);
        if(fmax <= 0) fmax = fps;
        if(fmin <= 0) fmin = fps;
        const std::string name = subtypeName(sub);
        auto key = std::make_tuple(w, h, name);
        auto it = index.find(key);
        if(it == index.end()) {
            NativeFormat nf;
            nf.format.width = w;
            nf.format.height = h;
            nf.format.pixelFormat = name;
            nf.format.minFPS = fmin;
            nf.format.maxFPS = fmax;
            nf.subtype = sub;
            it = index.emplace(key, m_native.size()).first;
            m_native.push_back(nf);
        }
        NativeFormat &nf = m_native[it->second];
        nf.format.minFPS = (nf.format.minFPS > 0) ? std::min(nf.format.minFPS, fmin) : fmin;
        nf.format.maxFPS = std::max(nf.format.maxFPS, fmax);
        nf.rates.emplace_back(fps > 0 ? fps : fmax, i);
    }
    for(auto &&nf: m_native)
        std::sort(nf.rates.begin(), nf.rates.end());
    if(m_native.empty())
        throw WebCam::Error("The camera reports no video format.");
    for(auto &&nf: m_native)
        m_formats.push_back(nf.format);
}

void
Device::apply(int index, double fps) {
    const NativeFormat &nf = m_native[index];
    //The slowest native rate that still reaches fps; else the fastest there is.
    DWORD type_index = nf.rates.back().second;
    for(auto &&r: nf.rates)
        if(r.first >= fps * 0.99) {
            type_index = r.second;
            break;
        }
    m_streaming = false;
    m_reader->Flush((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM);

    ComPtr<IMFMediaType> native;
    check(m_reader->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, type_index, native.put()),
        "GetNativeMediaType");
    //The device's own type first, so that the size, subtype and rate are the
    //ones chosen; then, for a compressed one, an NV12 output the reader
    //decodes into.
    check(m_reader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, native.get()),
        "Setting the video format");
    const Subtype *st = findSubtype(nf.subtype);
    m_outLuma = st ? st->luma : Subtype::Luma::None;
    if(m_outLuma == Subtype::Luma::None) {
        ComPtr<IMFMediaType> out;
        check(MFCreateMediaType(out.put()), "MFCreateMediaType");
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        out->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
        MFSetAttributeSize(out.get(), MF_MT_FRAME_SIZE, nf.format.width, nf.format.height);
        UINT32 num = 0, den = 0;
        if(SUCCEEDED(MFGetAttributeRatio(native.get(), MF_MT_FRAME_RATE, &num, &den)))
            MFSetAttributeRatio(out.get(), MF_MT_FRAME_RATE, num, den);
        check(m_reader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, out.get()),
            ("Decoding " + nf.format.pixelFormat + " to NV12").c_str());
        m_outLuma = Subtype::Luma::Plane8;
    }
    ComPtr<IMFMediaType> current;
    check(m_reader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, current.put()),
        "GetCurrentMediaType");
    UINT32 w = 0, h = 0;
    check(MFGetAttributeSize(current.get(), MF_MT_FRAME_SIZE, &w, &h), "MF_MT_FRAME_SIZE");
    m_outWidth = w;
    m_outHeight = h;
    m_failures = 0;
    m_streaming = true;
    m_active = index;
    //The first frame here, so that a camera another application holds, or
    //one the privacy settings refuse, fails the format change -- and so the
    //opening of the interface -- with the reason, instead of later as frames
    //that never come.
    const HRESULT hr = readOne();
    if(FAILED(hr)) {
        m_streaming = false;
        m_active = -1;
        check(hr, "Starting the video stream");
    }
}

//! One frame into the slot.  \return the failure, or S_OK.
HRESULT
Device::readOne() {
    DWORD stream = 0, flags = 0;
    LONGLONG ts = 0;
    ComPtr<IMFSample> sample;
    HRESULT hr = m_reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0,
        &stream, &flags, &ts, sample.put());
    if(m_stop)
        return S_OK;
    if(SUCCEEDED(hr) && (flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM)))
        hr = MF_E_END_OF_STREAM;
    if(FAILED(hr)) {
        //A disconnected or seized camera keeps failing; stop asking after a few.
        if(++m_failures >= 5) {
            fprintf(stderr, "WebCam: streaming stopped, %s\n", hresultText(hr).c_str());
            m_streaming = false;
        }
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return hr;
    }
    if( !(flags & MF_SOURCE_READERF_STREAMTICK) && sample) {
        m_failures = 0;
        if(extractLuma(sample.get()))
            m_slot->post(m_buf);
    }
    return S_OK;
}

bool
Device::extractLuma(IMFSample *sample) {
    ComPtr<IMFMediaBuffer> buffer;
    if(FAILED(sample->ConvertToContiguousBuffer(buffer.put())))
        return false;
    const unsigned int w = m_outWidth, h = m_outHeight;
    if( !w || !h)
        return false;
    BYTE *scan0 = nullptr;
    LONG pitch = 0;
    auto buf2d = buffer.as<IMF2DBuffer>(IID_IMF2DBuffer);
    bool locked2d = buf2d && SUCCEEDED(buf2d->Lock2D(&scan0, &pitch));
    BYTE *data = nullptr;
    DWORD length = 0;
    if( !locked2d) {
        if(FAILED(buffer->Lock(&data, nullptr, &length)))
            return false;
        scan0 = data;
        pitch = (m_outLuma == Subtype::Luma::Plane8) ? (LONG)w : (LONG)(2 * w);
        if((size_t)std::abs(pitch) * h > length) {
            buffer->Unlock();
            return false;
        }
    }
    m_buf.width = w;
    m_buf.height = h;
    m_buf.timestamp_us = unixMicrosecondsNow();
    m_buf.luma.resize((size_t)w * h);
    uint8_t *dst = m_buf.luma.data();
    for(unsigned int y = 0; y < h; ++y) {
        const BYTE *s = scan0 + (ptrdiff_t)y * pitch;
        switch(m_outLuma) {
        case Subtype::Luma::Plane8:
            memcpy(dst, s, w); dst += w; break;
        case Subtype::Luma::Y0Packed:
            for(unsigned int x = 0; x < w; ++x) *dst++ = s[2 * x]; break;
        case Subtype::Luma::Y1Packed:
            for(unsigned int x = 0; x < w; ++x) *dst++ = s[2 * x + 1]; break;
        case Subtype::Luma::None:
            break;
        }
    }
    if(locked2d)
        buf2d->Unlock2D();
    else
        buffer->Unlock();
    return true;
}

bool
Device::exposure(double sec) {
    if( !m_cameraControl)
        return false;
    long vmin, vmax, step, vdef, caps;
    if(FAILED(m_cameraControl->GetRange(CameraControl_Exposure, &vmin, &vmax, &step, &vdef, &caps)))
        return false;
    if(sec <= 0) {
        if( !(caps & CameraControl_Flags_Auto))
            return false;
        long v = vdef, flags = 0;
        m_cameraControl->Get(CameraControl_Exposure, &v, &flags);
        return SUCCEEDED(m_cameraControl->Set(CameraControl_Exposure, v, CameraControl_Flags_Auto));
    }
    if( !(caps & CameraControl_Flags_Manual))
        return false;
    //UVC through DirectShow/MF: the value is log2 of the time in seconds.
    long v = lrint(std::log2(sec));
    v = std::min(std::max(v, vmin), vmax);
    return SUCCEEDED(m_cameraControl->Set(CameraControl_Exposure, v, CameraControl_Flags_Manual));
}

bool
Device::gain(double g) {
    if( !m_procAmp)
        return false;
    long vmin, vmax, step, vdef, caps;
    if(FAILED(m_procAmp->GetRange(VideoProcAmp_Gain, &vmin, &vmax, &step, &vdef, &caps)))
        return false;
    if(g <= 0) {
        //No automatic gain on many UVC cameras (auto exposure moves it
        //instead): the device's default is the nearest thing.
        if(caps & VideoProcAmp_Flags_Auto)
            return SUCCEEDED(m_procAmp->Set(VideoProcAmp_Gain, vdef, VideoProcAmp_Flags_Auto));
        return SUCCEEDED(m_procAmp->Set(VideoProcAmp_Gain, vdef, VideoProcAmp_Flags_Manual));
    }
    long v = lrint(g);
    v = std::min(std::max(v, vmin), vmax);
    return SUCCEEDED(m_procAmp->Set(VideoProcAmp_Gain, v, VideoProcAmp_Flags_Manual));
}

} //namespace

namespace WebCam {

const BackendInfo &
backendInfo() {
    static const BackendInfo info = {"Media Foundation", true};
    return info;
}

std::vector<DeviceInfo>
enumerateDevices() {
    try {
        return inMTA([]() {
            startMediaFoundation();
            std::vector<DeviceInfo> list;
            ComPtr<IMFAttributes> attr;
            check(MFCreateAttributes(attr.put(), 1), "MFCreateAttributes");
            check(attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID), "SetGUID");
            IMFActivate **devices = nullptr;
            UINT32 count = 0;
            check(MFEnumDeviceSources(attr.get(), &devices, &count), "MFEnumDeviceSources");
            for(UINT32 i = 0; i < count; ++i) {
                WCHAR *name = nullptr, *link = nullptr;
                UINT32 len = 0;
                devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, &len);
                devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &link, &len);
                if(link)
                    list.push_back({toUtf8(link), name ? toUtf8(name) : toUtf8(link)});
                CoTaskMemFree(name);
                CoTaskMemFree(link);
                devices[i]->Release();
            }
            CoTaskMemFree(devices);
            return list;
        });
    }
    catch(Error &e) {
        fprintf(stderr, "WebCam: device enumeration failed, %s\n", e.what());
        return {};
    }
}

std::unique_ptr<Camera>
openCamera(const std::string &id) {
    //Windows asks no permission of a desktop app; a denial in the privacy
    //settings comes back as E_ACCESSDENIED, which check() spells out.
    return std::unique_ptr<Camera>(new MFCamera(id));
}

} //namespace WebCam
