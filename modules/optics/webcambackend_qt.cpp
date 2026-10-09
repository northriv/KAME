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
// WebCam backend over Qt Multimedia (Windows, Linux; macOS on request).
//
// Threading: every Qt Multimedia object here is created, used and deleted on
// the main thread, which is the only thread the platform plugins are written
// for.  Calls from other threads are queued there by onMainThread() with a
// bounded wait.  Frames are taken in whatever thread QVideoSink emits from
// and handed to the driver through the FrameSlot.
#include "webcambackend.h"

#include <QCoreApplication>
#include <QThread>
#include <QPointer>
#include <QCamera>
#include <QCameraDevice>
#include <QMediaDevices>
#include <QMediaCaptureSession>
#include <QVideoSink>
#include <QVideoFrame>
#include <QImage>
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
#include <QPermissions>
#if QT_CONFIG(permissions)
#define WEBCAM_QT_PERMISSIONS
#endif
#endif

#include <cstring>
#include <future>

namespace {

int64_t unixMicrosecondsNow() {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

//! Runs \a f on the main thread and returns its result (or rethrows).
//! Bounded, because a main thread blocked on something the caller holds
//! would otherwise deadlock both; \a f must therefore capture nothing by
//! reference, as it may still run after the caller has given up.
template <typename F>
auto onMainThread(F f) -> decltype(f()) {
    using R = decltype(f());
    QCoreApplication *app = QCoreApplication::instance();
    if( !app)
        throw WebCam::Error("Qt Multimedia needs a running application.");
    if(QThread::currentThread() == app->thread())
        return f();
    auto task = std::make_shared<std::packaged_task<R()>>(std::move(f));
    auto result = task->get_future();
    QMetaObject::invokeMethod(app, [task]{(*task)();}, Qt::QueuedConnection);
    if(result.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
        throw WebCam::Error("The main thread did not respond within 10 s.");
    return result.get();
}

//! Fills out.luma from \a frame. Formats with a luma plane (or packed
//! YUV 4:2:2) are read directly; anything else, MJPEG included, goes through
//! QVideoFrame::toImage().
bool extractLuma(QVideoFrame frame, WebCam::Frame &out) {
    const int w = frame.width(), h = frame.height();
    if((w <= 0) || (h <= 0))
        return false;
    out.width = w;
    out.height = h;
    out.luma.resize((size_t)w * h);
    uint8_t *dst = out.luma.data();

    enum class Layout {Luma8, Luma16, Packed422Y0, Packed422Y1, Other};
    Layout layout = Layout::Other;
    switch(frame.pixelFormat()) {
    case QVideoFrameFormat::Format_NV12:
    case QVideoFrameFormat::Format_NV21:
    case QVideoFrameFormat::Format_YUV420P:
    case QVideoFrameFormat::Format_YUV422P:
    case QVideoFrameFormat::Format_YV12:
    case QVideoFrameFormat::Format_Y8:
        layout = Layout::Luma8; break;
    case QVideoFrameFormat::Format_P010:
    case QVideoFrameFormat::Format_P016:
    case QVideoFrameFormat::Format_Y16:
        layout = Layout::Luma16; break; //MSB-aligned 16-bit little endian: the high byte is the 8-bit value.
    case QVideoFrameFormat::Format_YUYV:
        layout = Layout::Packed422Y0; break;
    case QVideoFrameFormat::Format_UYVY:
        layout = Layout::Packed422Y1; break;
    default:
        break;
    }
    if(layout == Layout::Other) {
        const QImage img = frame.toImage().convertToFormat(QImage::Format_Grayscale8);
        if((img.width() != w) || (img.height() != h))
            return false;
        for(int y = 0; y < h; ++y)
            memcpy(dst + (size_t)y * w, img.constScanLine(y), w);
        return true;
    }

    if( !frame.map(QVideoFrame::ReadOnly)) //still QVideoFrame's own enum as of 6.10.
        return false;
    const uint8_t *src = frame.bits(0);
    const int bpl = frame.bytesPerLine(0);
    if(src) {
        for(int y = 0; y < h; ++y) {
            const uint8_t *s = src + (size_t)y * bpl;
            switch(layout) {
            case Layout::Luma8:
                memcpy(dst, s, w); dst += w; break;
            case Layout::Luma16:
                for(int x = 0; x < w; ++x) *dst++ = s[2 * x + 1]; break;
            case Layout::Packed422Y0:
                for(int x = 0; x < w; ++x) *dst++ = s[2 * x]; break;
            case Layout::Packed422Y1:
                for(int x = 0; x < w; ++x) *dst++ = s[2 * x + 1]; break;
            case Layout::Other:
                break;
            }
        }
    }
    frame.unmap();
    return src != nullptr;
}

class QtCamera : public WebCam::Camera {
public:
    //! Main thread only.
    explicit QtCamera(const QCameraDevice &device);
    ~QtCamera() override;

    std::vector<WebCam::Format> formats() const override {return m_formats;}
    int activeFormat() const override;
    void setFormat(int index, double fps) override;
    bool setExposureTime(double sec) override;
    bool setGain(double gain) override;
private:
    QObject *m_holder; //!< parent of the objects below; deleted on the main thread.
    QPointer<QCamera> m_camera;
    QPointer<QMediaCaptureSession> m_session;
    QList<QCameraFormat> m_qformats;
    std::vector<WebCam::Format> m_formats;
};

QtCamera::QtCamera(const QCameraDevice &device) :
    m_holder(new QObject) {
    auto session = new QMediaCaptureSession(m_holder);
    auto camera = new QCamera(device, m_holder);
    auto sink = new QVideoSink(m_holder);
    session->setCamera(camera);
    session->setVideoSink(sink);
    m_camera = camera;
    m_session = session;
    m_qformats = device.videoFormats();
    for(const QCameraFormat &f: m_qformats) {
        WebCam::Format fmt;
        fmt.width = f.resolution().width();
        fmt.height = f.resolution().height();
        fmt.pixelFormat = QVideoFrameFormat::pixelFormatToString(f.pixelFormat()).toStdString();
        fmt.minFPS = f.minFrameRate();
        fmt.maxFPS = f.maxFrameRate();
        m_formats.push_back(fmt);
    }
    //Direct: converts in the emitting thread, which is not necessarily main.
    auto slot = m_slot;
    auto buf = std::make_shared<WebCam::Frame>(); //recycled through FrameSlot::post().
    QObject::connect(sink, &QVideoSink::videoFrameChanged, sink, [slot, buf](const QVideoFrame &frame) {
        buf->timestamp_us = unixMicrosecondsNow();
        if(extractLuma(frame, *buf))
            slot->post( *buf);
    }, Qt::DirectConnection);
    QObject::connect(camera, &QCamera::errorOccurred, camera, [](QCamera::Error, const QString &msg) {
        fprintf(stderr, "WebCam: %s\n", msg.toLocal8Bit().constData());
    });
}

QtCamera::~QtCamera() {
    QObject *holder = m_holder;
    QPointer<QMediaCaptureSession> session = m_session;
    QPointer<QCamera> camera = m_camera;
    auto teardown = [holder, session, camera]() {
        if(session)
            session->setVideoSink(nullptr); //no further frames into the slot.
        if(camera)
            camera->stop();
        delete holder;
    };
    QCoreApplication *app = QCoreApplication::instance();
    if( !app)
        return; //at exit, with no main loop left to run it: leaked.
    if(QThread::currentThread() == app->thread())
        teardown();
    else
        QMetaObject::invokeMethod(app, teardown, Qt::QueuedConnection); //not waited for.
}

int
QtCamera::activeFormat() const {
    QPointer<QCamera> camera = m_camera;
    QList<QCameraFormat> formats = m_qformats;
    return onMainThread([camera, formats]() -> int {
        if( !camera)
            return -1;
        return (int)formats.indexOf(camera->cameraFormat()); //null format (never set) is not in the list.
    });
}

void
QtCamera::setFormat(int index, double) {
    //Qt Multimedia has no frame-rate control apart from the format itself;
    //XWebCamera thins the frames out to the requested rate.
    if((index < 0) || (index >= m_qformats.size()))
        throw WebCam::Error("No such video format.");
    QPointer<QCamera> camera = m_camera;
    QCameraFormat format = m_qformats.at(index);
    onMainThread([camera, format]() {
        if( !camera)
            throw WebCam::Error("The camera has gone.");
        camera->stop();
        camera->setCameraFormat(format);
        camera->start();
        if(camera->error() != QCamera::NoError)
            throw WebCam::Error(camera->errorString().toStdString());
    });
}

bool
QtCamera::setExposureTime(double sec) {
    QPointer<QCamera> camera = m_camera;
    return onMainThread([camera, sec]() -> bool {
        if( !camera)
            return false;
        if(sec <= 0) {
            if(camera->isExposureModeSupported(QCamera::ExposureAuto))
                camera->setExposureMode(QCamera::ExposureAuto);
            camera->setAutoExposureTime();
            return true;
        }
        if( !(camera->supportedFeatures() & QCamera::Feature::ManualExposureTime))
            return false;
        if(camera->isExposureModeSupported(QCamera::ExposureManual))
            camera->setExposureMode(QCamera::ExposureManual);
        camera->setManualExposureTime((float)sec);
        return true;
    });
}

bool
QtCamera::setGain(double gain) {
    QPointer<QCamera> camera = m_camera;
    return onMainThread([camera, gain]() -> bool {
        if( !camera)
            return false;
        if(gain <= 0) {
            camera->setAutoIsoSensitivity();
            return true;
        }
        if( !(camera->supportedFeatures() & QCamera::Feature::IsoSensitivity))
            return false;
        camera->setManualIsoSensitivity((int)lrint(gain));
        return true;
    });
}

void
requestPermission() {
#ifdef WEBCAM_QT_PERMISSIONS
    const char *denied = "Camera access is denied. Allow KAME in the system's privacy settings for the camera.";
    const QCameraPermission perm;
    Qt::PermissionStatus status = onMainThread([perm]() {
        return QCoreApplication::instance()->checkPermission(perm);
    });
    if(status == Qt::PermissionStatus::Undetermined) {
        auto answer = std::make_shared<std::promise<Qt::PermissionStatus>>();
        auto future = answer->get_future();
        onMainThread([perm, answer]() {
            QCoreApplication *app = QCoreApplication::instance();
            app->requestPermission(perm, app, [answer](const QPermission &p) {
                try {answer->set_value(p.status());} catch(std::future_error &) {}
            });
        });
        QCoreApplication *app = QCoreApplication::instance();
        if(QThread::currentThread() == app->thread())
            throw WebCam::Error("Camera permission has been requested; answer the dialog and turn the interface on again.");
        if(future.wait_for(std::chrono::seconds(120)) != std::future_status::ready)
            throw WebCam::Error("The camera permission dialog was not answered.");
        status = future.get();
    }
    if(status != Qt::PermissionStatus::Granted)
        throw WebCam::Error(denied);
#endif
}

} //namespace

namespace WebCam {

const BackendInfo &
backendInfo() {
    static const BackendInfo info = {"Qt Multimedia", true};
    return info;
}

std::vector<DeviceInfo>
enumerateDevices() {
    try {
        return onMainThread([]() {
            std::vector<DeviceInfo> list;
            for(const QCameraDevice &d: QMediaDevices::videoInputs()) {
                const QByteArray id = d.id();
                list.push_back({std::string(id.constData(), id.size()), d.description().toStdString()});
            }
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
    requestPermission();
    return onMainThread([id]() -> std::unique_ptr<Camera> {
        const QByteArray qid(id.data(), (int)id.size());
        for(const QCameraDevice &d: QMediaDevices::videoInputs())
            if(d.id() == qid)
                return std::unique_ptr<Camera>(new QtCamera(d));
        throw Error("The camera is not connected.");
    });
}

} //namespace WebCam
