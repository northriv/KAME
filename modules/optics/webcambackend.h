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
//---------------------------------------------------------------------------

#ifndef webcambackendH
#define webcambackendH
//---------------------------------------------------------------------------
// Platform face of a webcam (built-in or USB video class) for XWebCamera.
// Exactly one implementation is linked, chosen by optics.pro:
//   webcambackend_avf.mm  AVFoundation (macOS)
//   webcambackend_qt.cpp  Qt Multimedia (elsewhere, or CONFIG+=webcam_qtmultimedia)
// Deliberately std-only: the AVFoundation side is Objective-C++, and keeping
// KAME and Qt headers out of it avoids their macros (slots, signals, emit)
// meeting Objective-C's keywords.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <stdexcept>

namespace WebCam {

struct DeviceInfo {
    std::string id;   //!< stable platform identifier, used to open
    std::string name; //!< human-readable, shown in the interface's Device combo
};

struct Format {
    unsigned int width = 0, height = 0;
    std::string pixelFormat; //!< as the platform names it, e.g. "420v", "NV12", "Jpeg"
    double minFPS = 0, maxFPS = 0;
};

//! One 8-bit luminance frame. Rows are tightly packed (stride == width).
struct Frame {
    unsigned int width = 0, height = 0;
    std::vector<uint8_t> luma;
    int64_t timestamp_us = 0; //!< capture time, microseconds since the Unix epoch
};

//! Whatever the platform API refuses; XWebCamera turns it into XInterfaceError.
struct Error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

//! Hand-off from the platform's capture callback to the acquisition thread.
//! Keeps only the newest frame: a camera runs at its own pace, and a frame
//! the driver has not taken yet is worth less than the one replacing it.
class FrameSlot {
public:
    //! Swaps buffers: \a frame.luma comes back holding a recycled one, so a
    //! producer that keeps its Frame allocates only until sizes settle.
    void post(Frame &frame) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if(m_fresh)
            ++m_dropped;
        m_frame.width = frame.width;
        m_frame.height = frame.height;
        m_frame.timestamp_us = frame.timestamp_us;
        m_frame.luma.swap(frame.luma);
        m_fresh = true;
        m_cond.notify_one();
    }
    //! \return false on timeout.
    bool wait(Frame &frame, unsigned int timeout_ms) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if( !m_cond.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]{return m_fresh;}))
            return false;
        frame.width = m_frame.width;
        frame.height = m_frame.height;
        frame.timestamp_us = m_frame.timestamp_us;
        frame.luma.swap(m_frame.luma); //hands the caller's old buffer back for reuse.
        m_fresh = false;
        return true;
    }
    //! Frames replaced before the driver took them, since the camera was opened.
    uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_dropped;
    }
    void addDropped(uint64_t n) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_dropped += n;
    }
private:
    mutable std::mutex m_mutex;
    std::condition_variable m_cond;
    Frame m_frame;
    bool m_fresh = false;
    uint64_t m_dropped = 0;
};

//! An opened camera. Methods may be called from any thread; each
//! implementation serializes them itself (a dispatch queue for AVFoundation,
//! the main thread for Qt Multimedia), so callers need not hold a lock.
class Camera {
public:
    virtual ~Camera() = default;
    virtual std::vector<Format> formats() const = 0;
    //! Index into formats() of the format the device is in now, or -1 when unknown.
    virtual int activeFormat() const = 0;
    //! (Re)starts streaming in formats()[index]. \a fps is a hint: a backend
    //! that cannot set the rate in hardware streams at the format's own rate,
    //! and XWebCamera thins the frames out.
    virtual void setFormat(int index, double fps) = 0;
    //! Exposure time [s]; <= 0 returns to auto exposure.
    //! \return false when the device cannot do it.
    virtual bool setExposureTime(double sec) = 0;
    //! Sensor gain (ISO where the platform calls it that); <= 0 returns to auto.
    //! \return false when the device cannot do it.
    virtual bool setGain(double gain) = 0;

    //! \return false on timeout.
    bool waitFrame(Frame &frame, unsigned int timeout_ms) {return m_slot->wait(frame, timeout_ms);}
    uint64_t droppedFrames() const {return m_slot->dropped();}
protected:
    //! shared_ptr: the platform callback may outlive the Camera by one frame.
    const std::shared_ptr<FrameSlot> m_slot = std::make_shared<FrameSlot>();
};

struct BackendInfo {
    const char *name;
    bool canSetGain; //!< false: no device on this backend can, so XWebCamera disables the control.
};
const BackendInfo &backendInfo();

std::vector<DeviceInfo> enumerateDevices();
//! Asks for camera permission first where the OS requires it; may block until the user answers.
std::unique_ptr<Camera> openCamera(const std::string &id);

} //namespace WebCam

#endif
