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
#include "euresyscamera.h"
#include "analyzer.h"
#include <QImage>

#if defined USE_EURESYS_EGRABBER

unique_ptr<Euresys::EGenTL> XEGrabberInterface::s_gentl;
unique_ptr<Euresys::EGrabberDiscovery> XEGrabberInterface::s_discovery;
int XEGrabberInterface::s_refcnt = 0;
XRecursiveMutex XEGrabberInterface::s_mutex;

REGISTER_TYPE(XDriverList, EGrabberCamera, "Camera via Euresys eGrabber");

//---------------------------------------------------------------------------
XEGrabberInterface::XEGrabberInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
    XInterface(name, runtime, driver) {
    XScopedLock<XEGrabberInterface> lock( *this);
    if(s_refcnt++ == 0) {
        try {
            s_gentl = std::make_unique<Euresys::EGenTL>();
            s_discovery = std::make_unique<Euresys::EGrabberDiscovery>( *s_gentl);
            for (int i = 0; i < s_discovery->cameraCount(); ++i) {
                 Euresys::EGrabber<> grabber(s_discovery->cameras(i));
                 trans( *device()).add(
                    formatString("%i:",
                    (int)grabber.getInteger<Euresys::InterfaceModule>("InterfaceIndex"))
                    + grabber.getString<Euresys::DeviceModule>("DeviceModelName"));
            }
        }
        catch (const std::exception &e) {                                                 // 7
            gErrPrint(XString("error: ") + e.what());
        }
    }
}
XEGrabberInterface::~XEGrabberInterface() {
    XScopedLock<XEGrabberInterface> lock( *this);
    if(--s_refcnt == 0) {
        s_discovery.reset();
        s_gentl.reset();
    }

}
void
XEGrabberInterface::open() {
    XScopedLock<XEGrabberInterface> lock( *this);
    Snapshot shot( *this);
    try {
        for (int i = 0; i < s_discovery->cameraCount(); ++i) {
             Euresys::EGrabber<> grabber(s_discovery->cameras(i));
             if(shot[ *device()].to_str() ==
                 formatString("%i:",
                 (int)grabber.getInteger<Euresys::InterfaceModule>("InterfaceIndex"))
                 + grabber.getString<Euresys::DeviceModule>("DeviceModelName")) {
                 m_camera = std::make_shared<Euresys::EGrabber<>>
                    (grabber.getGenTL(),
                     (int)grabber.getInteger<Euresys::InterfaceModule>("InterfaceIndex"),
                     (int)grabber.getInteger<Euresys::InterfaceModule>("DeviceIndex"));
                 break;
             }
        }
        std::string str_intf = m_camera->getString<Euresys::InterfaceModule>("InterfaceID");
        std::string str_dev = m_camera->getString<Euresys::DeviceModule>("DeviceID");
        int width = m_camera->getInteger<Euresys::RemoteModule>("Width");
        int height = m_camera->getInteger<Euresys::RemoteModule>("Height");
        fprintf(stderr, "%s:%s %ix%i", str_intf.c_str(), str_dev.c_str(), width, height);
    }
    catch (const std::exception &e) {
        gErrPrint(XString("error: ") + e.what());
        m_camera.reset();
        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    }
}
void
XEGrabberInterface::close() {
    if(m_camera) {
        m_camera->stop();
        m_camera.reset();
    }
}

XEGrabberCamera::XEGrabberCamera(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XEGrabberDriver<XDigitalCamera>(name, runtime, ref(tr_meas), meas) {
//    startWavelen()->disable();
//    stopWavelen()->disable();
}

void
XEGrabberCamera::open() {
    auto camera = interface()->camera();
    if( !camera)
        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    m_isTrasmitting = false;

    using namespace Euresys;

    try {
        auto videomodes = camera->getStringList<RemoteModule>(
                    query::enumEntries("PixelFormat"));
        auto triggersources = camera->getStringList<RemoteModule>(
                    query::enumEntries("TriggerSource"));
        bool trigon = camera->getString<RemoteModule>("TriggerMode") == "On";
        TriggerMode trigmode = TriggerMode::CONTINUEOUS;
        std::string expmode = camera->getString<RemoteModule>("ExposureMode");
        std::string trigact = camera->getString<RemoteModule>("TriggerActivation");
        if(expmode == "Timed") {
            if(trigon) {
                if(trigact == "RisingEdge")
                    trigmode = TriggerMode::EXT_POS_EDGE;
                else if(trigact == "FallingEdge")
                    trigmode = TriggerMode::EXT_NEG_EDGE;
            }
        }
        else if(expmode == "TriggerWidth") {
            if(trigon) {
                if(trigact == "RisingEdge")
                    trigmode = TriggerMode::EXT_POS_EXPOSURE;
                else if(trigact == "FallingEdge")
                    trigmode = TriggerMode::EXT_NEG_EXPOSURE;
            }
        }
        else if(expmode == "TriggerControlled") {
        }

//        grabber.setString<Euresys::DeviceModule>("CameraControlMethod", "RG");              // 4
//        grabber.setString<Euresys::DeviceModule>("CycleTriggerSource", "Immediate");        // 5
//        grabber.setFloat<Euresys::DeviceModule>("CycleTargetPeriod", 1e6 / FPS);

        iterate_commit([=](Transaction &tr){
            tr[ *videoMode()] = -1;
            tr[ *videoMode()].clear();
            for(auto &s: videomodes)
                tr[ *videoMode()].add(s);
            tr[ *triggerMode()] = (unsigned int)trigmode;
            tr[ *frameRate()].clear();
            for(double rate = 240.0; rate > 1.7; rate /= 2) {
                tr[ *frameRate()].add(formatString("%f fps", rate));
            }
    //        for(double rate = 1.0; rate > 0.001; rate /= 2) {
    //            tr[ *frameRate()].add(formatString("%f fps", rate));
    //        }
        });
    }
    catch (const std::exception &e) {                                                 // 7
        gErrPrint(XString("error: ") + e.what());
        return;
    }


    start();
}

void
XEGrabberCamera::stopTransmission() {
    XScopedLock<XEGrabberInterface> lock( *interface());
    if(m_isTrasmitting) {
        m_isTrasmitting = false;
        auto camera = interface()->camera();
        camera->stop();
    }
}
void
XEGrabberCamera::setVideoMode(unsigned int mode, unsigned int roix, unsigned int roiy, unsigned int roiw, unsigned int roih) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    stopTransmission();
//    if(dc1394_video_set_transmission(interface()->camera(), DC1394_OFF))
//        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not stop transmission."), __FILE__, __LINE__);
    Snapshot shot( *this);


    setTriggerMode(static_cast<TriggerMode>((unsigned int)shot[ *triggerMode()]));
}
void
XEGrabberCamera::setTriggerMode(TriggerMode mode) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    stopTransmission();


    m_isTrasmitting = true;
}
void
XEGrabberCamera::setBrightness(unsigned int brightness) {
    XScopedLock<XEGrabberInterface> lock( *interface());
//    stopTransmission();
}
void
XEGrabberCamera::setExposureTime(double shutter) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    stopTransmission();
    setTriggerMode(static_cast<TriggerMode>((unsigned int)Snapshot( *this)[ *triggerMode()]));
}
void
XEGrabberCamera::setCameraGain(double db) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    int gain = lrint(255.0 * db / 20.0);
    gain = std::min(std::max(0, gain), 255);
}

void
XEGrabberCamera::analyzeRaw(RawDataReader &reader, Transaction &tr) {

}

XTime
XEGrabberCamera::acquireRaw(shared_ptr<RawData> &writer) {
    if( !m_isTrasmitting)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    XScopedLock<XEGrabberInterface> lock( *interface());
    Snapshot shot( *this);


    return XTime::now(); //time stamp is invalid for win + libdc1394.

}

#endif // USE_LIBDC1394
