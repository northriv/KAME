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
#include <QImage>

#if defined USE_EURESYS_EGRABBER

unique_ptr<Euresys::EGenTL> XEGrabberInterface::s_gentl;
unique_ptr<Euresys::EGrabberDiscovery> XEGrabberInterface::s_discovery;
int XEGrabberInterface::s_refcnt = 0;
XRecursiveMutex XEGrabberInterface::s_mutex;

REGISTER_TYPE(XDriverList, EGrabberCamera, "Coaxpress Camera via Euresys eGrabber");
REGISTER_TYPE(XDriverList, GrablinkCamera, "Cameralink Camera via Euresys eGrabber");
REGISTER_TYPE(XDriverList, HamamatsuCameraOverGrablink, "Hamamatsu Camera via Euresys grablink");
REGISTER_TYPE(XDriverList, JAICameraOverGrablink, "JAI Camera via Euresys grablink");

//---------------------------------------------------------------------------
XEGrabberInterface::XEGrabberInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver, bool grablink) :
    XCustomCharInterface(name, runtime, driver) {
    XScopedLock<XEGrabberInterface> lock( *this);
    XScopedLock<XRecursiveMutex> slock( s_mutex);
    if(s_refcnt++ == 0) {
        try {
            using namespace Euresys;
            s_gentl = grablink ?
                        std::make_unique<EGenTL>(Grablink()) :
                        std::make_unique<EGenTL>(Coaxlink());
            s_discovery = std::make_unique<EGrabberDiscovery>( *s_gentl);
            s_discovery->discover();
            fprintf(stderr, "eGrabber count:%i; camera count:%i\n", s_discovery->egrabberCount(), s_discovery->cameraCount());
            for (int i = 0; i < s_discovery->cameraCount(); ++i) {
                 EGrabberCameraInfo info = s_discovery->cameras(i);
                 EGrabberInfo grabber = info.grabbers[0];
                 if(grabber.isRemoteAvailable) {
                     trans( *device()).add(
                        formatString("%i:", grabber.deviceIndex) + grabber.deviceModelName);
                 }
            }
        }
        catch (const std::exception &e) {                                                 // 7
            gErrPrint(XString("error: ") + e.what());
        }
    }
}
XEGrabberInterface::~XEGrabberInterface() {
    XScopedLock<XEGrabberInterface> lock( *this);
    XScopedLock<XRecursiveMutex> slock( s_mutex);
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
        using namespace Euresys;
        s_discovery->discover();
        for (int i = 0; i < s_discovery->cameraCount(); ++i) {
             EGrabberCameraInfo info = s_discovery->cameras(i);
             EGrabberInfo grabber = info.grabbers[0];
             if(grabber.isRemoteAvailable) {
                 if(shot[ *device()].to_str() ==
                         formatString("%i:", grabber.deviceIndex) + grabber.deviceModelName) {
                     m_camera = std::make_shared<Camera>(s_discovery->cameras(i));
                     break;
                 }
             }
        }
        if( !m_camera)
            throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);

        std::string str_intf = m_camera->getString<Euresys::InterfaceModule>("InterfaceID");
        std::string str_dev = m_camera->getString<Euresys::DeviceModule>("DeviceID");
        int width = m_camera->getInteger<Euresys::RemoteModule>("Width");
        int height = m_camera->getInteger<Euresys::RemoteModule>("Height");
        fprintf(stderr, "%s:%s %ix%i\n", str_intf.c_str(), str_dev.c_str(), width, height);

        m_bIsSerialPortOpened = false;
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
        closeSerialPort();
//        m_camera->stop();
        m_camera.reset();
    }
}
void
XEGrabberInterface::lock() {
    s_mutex.lock();
}
void
XEGrabberInterface::unlock() {
    closeSerialPort();
    s_mutex.unlock();
}
void
XEGrabberInterface::checkAndOpenSerialPort() {
    assert(isLocked());
    if(m_bIsSerialPortOpened)
        return;
    using namespace Euresys;
    //Serial comm.
    m_camera->setString<DeviceModule>("SerialOperationSelector", "Open");
    m_camera->execute<DeviceModule>("SerialOperationExecute");
    std::string status = m_camera->getString<DeviceModule>("SerialOperationStatus");
    if(status != "Success")
        throw std::runtime_error(status);

    std::vector<std::string> serialBaudRateCand =
            m_camera->getStringList<DeviceModule>(query::enumEntries("SerialBaudRate"));
    std::string bratestr = formatString("%u", m_serialBaudRate);
    auto baudit = std::find_if(serialBaudRateCand.begin(), serialBaudRateCand.end(), [&bratestr](auto &x){return (x.rfind(bratestr) != std::string::npos);});
    if(baudit == serialBaudRateCand.end())
        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    m_camera->setString<DeviceModule>("SerialBaudRate", *baudit);

    m_serialEOS = "\r";
    m_camera->setInteger<DeviceModule>("SerialTimeout", 500);
    m_camera->setInteger<DeviceModule>("SerialAccessBufferLength", 4096);

    m_bIsSerialPortOpened = true;
}
void
XEGrabberInterface::closeSerialPort() {
    if( !m_bIsSerialPortOpened)
        return;
    using namespace Euresys;
    m_camera->setString<DeviceModule>("SerialOperationSelector", "Close");
    m_camera->execute<DeviceModule>("SerialOperationExecute");
    m_bIsSerialPortOpened = false;
}

void
XEGrabberInterface::flush() {
    try {
        using namespace Euresys;
        m_camera->setString<DeviceModule>("SerialOperationSelector", "Flush");
        m_camera->execute<DeviceModule>("SerialOperationExecute");
        std::string status = m_camera->getString<DeviceModule>("SerialOperationStatus");
        if(status != "Success")
            throw std::runtime_error(status);
    }
    catch (const std::exception &e) {
        throw XInterface::XInterfaceError(XString("error: ") + e.what(), __FILE__, __LINE__);
    }
}
void
XEGrabberInterface::send(const char *buf) {
    checkAndOpenSerialPort(); //eGrabber does not allow sharing serialoperations by threads.
    XString str = buf + m_serialEOS;
    try {
        using namespace Euresys;
        m_camera->setString<DeviceModule>("SerialOperationSelector", "Write");
        m_camera->setInteger<DeviceModule>("SerialAccessLength", str.size());
        m_camera->setRegister<DeviceModule>("SerialAccessBuffer", str.data(), str.size());
        m_camera->execute<DeviceModule>("SerialOperationExecute");
        std::string status = m_camera->getString<DeviceModule>("SerialOperationStatus");
        if(status != "Success")
            throw std::runtime_error(status);
        if(m_camera->getInteger<DeviceModule>("SerialOperationResult") != (int)str.size())
            throw XInterface::XInterfaceError(i18n("Serial packet size mismatch."), __FILE__, __LINE__);
    }
    catch (const std::exception &e) {
        throw XInterface::XInterfaceError(XString("error: ") + e.what(), __FILE__, __LINE__);
    }
}
void
XEGrabberInterface::receive() {
    checkAndOpenSerialPort(); //eGrabber does not allow sharing serialoperations by threads.
    try {
        using namespace Euresys;
        buffer_receive().clear();
        XTime time = XTime::now();
        while(1) {
            if(XTime::now().diff_sec(time) > 1.0)
                throw XInterface::XCommError(i18n("Serial comm. timed out."), __FILE__, __LINE__);

            char buf[256];
            uint64_t rsize = m_camera->getInteger<DeviceModule>("SerialReadQueueSize");
            if( !rsize) {
                msecsleep(10);
                continue;
            }
            m_camera->setString<DeviceModule>("SerialOperationSelector", "Read");
            m_camera->setInteger<DeviceModule>("SerialAccessLength", std::min((int)rsize, (int)sizeof(buf) - 1));
            m_camera->execute<DeviceModule>("SerialOperationExecute");
            std::string status = m_camera->getString<DeviceModule>("SerialOperationStatus");
            if(status != "Success")
                throw std::runtime_error(status);
            rsize = m_camera->getInteger<DeviceModule>("SerialOperationResult");
            m_camera->getRegister<DeviceModule>("SerialAccessBuffer", buf, rsize);
            buffer_receive().insert(buffer_receive().end(), buf, buf + rsize);
            buf[rsize] = '\0';
            auto pos = std::string(buf).find(m_serialEOS);
            if(pos != std::string::npos) {
                buffer_receive().resize(buffer_receive().size() - rsize + pos + m_serialEOS.size());
                buffer_receive().push_back('\0');
                return;
            }
        }
    }
    catch (const std::exception &e) {
        throw XInterface::XInterfaceError(XString("error: ") + e.what(), __FILE__, __LINE__);
    }
}

XEGrabberCamera::XEGrabberCamera(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas, bool grablink) :
    XEGrabberDriver<XDigitalCamera>(name, runtime, ref(tr_meas), meas, grablink) {
}
XGrablinkCamera::XGrablinkCamera(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XEGrabberCamera(name, runtime, ref(tr_meas), meas, true) {
}

void
XEGrabberCamera::open() {
    auto camera = interface()->camera();
    if( !camera)
        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    m_isTrasmitting = false;

    try {
        using namespace Euresys;

        m_featuresInRemoteModule = camera->getStringList<RemoteModule>(query::features());
        fprintf(stderr, "Features:\n");
        for(auto &f: m_featuresInRemoteModule)
            fprintf(stderr, "%s ", f.c_str());
        fprintf(stderr, "\n");
//        std::string dvn = camera->getString<DeviceModule>(query::enumEntries("DeviceVendorName"));
//        std::string dmn = camera->getString<DeviceModule>(query::enumEntries("DeviceModelName"));
//        fprintf(stderr, "%s %s\n", dvn.c_str(), dmn.c_str());

        auto videomodes = camera->getStringList<RemoteModule>(
                    query::enumEntries("PixelFormat"));
        if(isFeatureAvailableInRemoteModule("TriggerSource")) {
            auto triggersources = camera->getStringList<RemoteModule>(
                        query::enumEntries("TriggerSource"));
        }

        bool feat_avail = true;
        for(auto &f: {"TriggerMode", "ExposureMode", "TriggerActivation"}) {
            if( !isFeatureAvailableInRemoteModule(f))
                feat_avail = false;
        }
        if(feat_avail) {
            bool trigon = camera->getString<RemoteModule>("TriggerMode") == "On";
            std::string expmode = camera->getString<RemoteModule>("ExposureMode");
            std::string trigact = camera->getString<RemoteModule>("TriggerActivation");
            std::map<TriggerMode, std::pair<std::string, std::string>> modes = {
                {TriggerMode::EXT_POS_EDGE, {"Timed", "RisingEdge"}},
                {TriggerMode::EXT_NEG_EDGE, {"Timed", "FallingEdge"}},
                {TriggerMode::EXT_POS_EXPOSURE, {"TriggerWidth", "RisingEdge"}},
                {TriggerMode::EXT_NEG_EXPOSURE, {"TriggerWidth", "FallingEdge"}},
            };
            auto tmit = std::find_if(modes.begin(), modes.end(),
                [&](auto&x){return (x.second.first == expmode) && (x.second.second == trigact);});
            TriggerMode trigmode = TriggerMode::CONTINUEOUS;
            if(trigon && (tmit != modes.end())) {
                trigmode = tmit->first;
            }
            iterate_commit([=](Transaction &tr){
                tr[ *triggerMode()] = (unsigned int)trigmode;
            });
        }

        if(isFeatureAvailableInRemoteModule("Blacklevel")) {
            double blacklvl = camera->getFloat<RemoteModule>("Blacklevel");
            iterate_commit([=](Transaction &tr){
                tr[ *blackLvlOffset()] = blacklvl;
            });
        }
        if(isFeatureAvailableInRemoteModule("ExposureTime")) {
            double exp_time = camera->getFloat<RemoteModule>("ExposureTime") * 1e-3; //to ms
            iterate_commit([=](Transaction &tr){
                tr[ *exposureTime()] = exp_time;
            });
        }
        if(isFeatureAvailableInRemoteModule("Gain")) {
            double gain_db = camera->getFloat<RemoteModule>("Gain"); //dB
            iterate_commit([=](Transaction &tr){
                tr[ *cameraGain()] = gain_db;
            });
        }

//        grabber.setString<Euresys::DeviceModule>("CameraControlMethod", "RG");              // 4
//        grabber.setFloat<Euresys::DeviceModule>("CycleTargetPeriod", 1e6 / FPS);

        iterate_commit([=](Transaction &tr){
            tr[ *videoMode()] = -1;
            tr[ *videoMode()].clear();
            for(auto &s: videomodes)
                tr[ *videoMode()].add(s);
            tr[ *frameRate()].clear();
            for(double rate = 240.0; rate > 1.7; rate /= 2) {
                tr[ *frameRate()].add(formatString("%f fps", rate));
            }
        });
    }
    catch (const std::exception &e) {
        throw XInterface::XInterfaceError(XString("error: ") + e.what(), __FILE__, __LINE__);
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
    Snapshot shot( *this);

    auto camera = interface()->camera();
    try {
        using namespace Euresys;
        camera->setString<RemoteModule>("PixelFormat", shot[ *videoMode()].to_str());

        if(isFeatureAvailableInRemoteModule("OffsetX")) {
            unsigned int w = camera->getInteger<RemoteModule>("WidthMax");
            unsigned int h = camera->getInteger<RemoteModule>("HeightMax");
            roix = roix / 4 * 4;
            roiy = roiy / 4 * 4;
            roiw = (roiw + 3) / 4 * 4;
            roih = (roih + 3) / 4 * 4;
            if( !roiw || !roih || (roix + roiw >= w) || (roiy + roih >= h) || (roiw > w) || (roih > h)) {
                roix = 0; roiy = 0; roiw = w; roih = h;
            }
            camera->setInteger<RemoteModule>("Width", roiw);
            camera->setInteger<RemoteModule>("Height", roih);
            camera->setInteger<RemoteModule>("OffsetX", roix);
            camera->setInteger<RemoteModule>("OffsetY", roiy);
        }
        else {
         //using serial commands.
            auto v = setVideoModeViaSerial(roix, roiy, roiw, roih);
            camera->setInteger<RemoteModule>("Width", v.first);
            camera->setInteger<RemoteModule>("Height", v.second);
        }
    }
    catch (const std::exception &e) {
        throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
    }
    setTriggerMode(static_cast<TriggerMode>((unsigned int)shot[ *triggerMode()]));
}
void
XEGrabberCamera::setTriggerMode(TriggerMode mode) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    stopTransmission();

    if(isFeatureAvailableInRemoteModule("TriggerMode")) {
        auto camera = interface()->camera();
        if(mode != TriggerMode::CONTINUEOUS) {
             std::map<TriggerMode, std::pair<std::string, std::string>> modes = {
                 {TriggerMode::EXT_POS_EDGE, {"Timed", "RisingEdge"}},
                 {TriggerMode::EXT_NEG_EDGE, {"Timed", "FallingEdge"}},
                 {TriggerMode::EXT_POS_EXPOSURE, {"TriggerWidth", "RisingEdge"}},
                 {TriggerMode::EXT_NEG_EXPOSURE, {"TriggerWidth", "FallingEdge"}},
             };
             try {
                 using namespace Euresys;
                 camera->setString<RemoteModule>("TriggerMode", "ON");
                 camera->setString<RemoteModule>("ExposureMode", modes.at(mode).first);
                 camera->setString<RemoteModule>("TriggerActivation", modes.at(mode).second);
             }
             catch (const std::exception &e) {
                 throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
             }
        }
        else {
            try {
                using namespace Euresys;
                camera->setString<RemoteModule>("TriggerMode", "OFF");
            }
            catch (const std::exception &e) {
                throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
            }
        }
        m_isTrasmitting = true;
    }
    else {
        setTriggerModeViaSerial(mode);
    }

    auto camera = interface()->camera();
    try {
        using namespace Euresys;
        camera->reallocBuffers(3);
        camera->start((mode == TriggerMode::SINGLE) ? 1 : GENTL_INFINITE);
    }
        catch (const std::exception &e) {
        throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
    }
    m_isTrasmitting = true;
    if(mode == TriggerMode::SINGLE){
        try {
            using namespace Euresys;
            camera->execute<Euresys::RemoteModule>("TriggerSoftware");
        }
        catch (const std::exception &e) {
            throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
        }
    }
}
void
XEGrabberCamera::setBlackLevelOffset(unsigned int blackLvlOffset) {
    XScopedLock<XEGrabberInterface> lock( *interface());
//    stopTransmission();
    auto camera = interface()->camera();
    try {
        using namespace Euresys;
        camera->setFloat<RemoteModule>("Blacklevel", blackLvlOffset);
    }
    catch (const std::exception &e) {
        throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
    }
}
void
XEGrabberCamera::setExposureTime(double shutter) {
    XScopedLock<XEGrabberInterface> lock( *interface());
//    stopTransmission();
//    setTriggerMode(static_cast<TriggerMode>((unsigned int)Snapshot( *this)[ *triggerMode()]));
    auto camera = interface()->camera();
    try {
        using namespace Euresys;
        camera->setFloat<RemoteModule>("ExposureTime", shutter * 1e3); //us
    }
    catch (const std::exception &e) {
        throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
    }
}
void
XEGrabberCamera::setGain(unsigned int g, unsigned int emgain) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    auto camera = interface()->camera();
    try {
        using namespace Euresys;
        camera->setFloat<RemoteModule>("Gain", g);
    }
    catch (const std::exception &e) {
        throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
    }
}

void
XEGrabberCamera::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    union {
        char shortname[8];
        uint64_t v;
    } feat;
    int feat_size = reader.pop<uint32_t>();
    feat.v = reader.pop<uint64_t>();
    int64_t width = reader.pop<int64_t>();
    feat.v = reader.pop<uint64_t>();
    int64_t height = reader.pop<int64_t>();
//    feat.v = reader.pop<uint64_t>();
    int64_t xpos = 0 ;//reader.pop<int64_t>();
//    feat.v = reader.pop<uint64_t>();
    int64_t ypos = 0 ;//reader.pop<int64_t>();
    feat.v = reader.pop<uint64_t>();
    int64_t payloadsize = reader.pop<int64_t>();
    feat_size -= 3;
    for(int i = 0; i < feat_size; ++i) {
        feat.v = reader.pop<uint64_t>();
        reader.pop<int64_t>(); //remaining Integer features.
    }
    feat_size = reader.pop<uint32_t>();
    for(int i = 0; i < feat_size; ++i) {
        feat.v = reader.pop<uint64_t>();
        reader.pop<int32_t>(); //remaining enum features.
    }
    feat_size = reader.pop<uint32_t>();
    for(int i = 0; i < feat_size; ++i) {
        feat.v = reader.pop<uint64_t>();
        reader.pop<int16_t>(); //remaining Boolean features.
    }
    feat_size = reader.pop<uint32_t>();
    for(int i = 0; i < feat_size; ++i) {
        feat.v = reader.pop<uint64_t>();
        reader.pop<double>(); //remaining Float features.
    }

    reader.popIterator() += reader.pop<int64_t>(); //for future use.

    uint64_t timestamp = reader.pop<uint64_t>();
    uint64_t image_bytes = reader.pop<uint64_t>();
    XTime time = {(long)(timestamp / 1000000uLL), (long)(timestamp % 1000000uLL)};
    uint64_t pixelformat = reader.pop<uint64_t>();
    width = reader.pop<uint64_t>();
    height = reader.pop<uint64_t>();
    size_t imagesize = reader.pop<uint64_t>();
    size_t imgpitch = reader.pop<uint64_t>();
    size_t delivered = reader.pop<uint64_t>();
    uint64_t fr = reader.pop<uint64_t>();
    uint64_t dr = reader.pop<uint64_t>();
    reader.pop<uint64_t>();
    reader.pop<uint64_t>();

    unsigned int bpp = image_bytes / (width * height);
    unsigned int padding_bytes = image_bytes - bpp * (width * height);
    setGrayImage(reader, tr, width, height, false, bpp == 2);
    reader.popIterator() += padding_bytes;

    tr[ *this].m_status = formatString("%ux%u @(%u,%u), %u MB/s, %u fps",
        (unsigned int)width, (unsigned int)height, (unsigned int)xpos, (unsigned int)ypos,
        (unsigned int)dr, (unsigned int)fr);
}

XTime
XEGrabberCamera::acquireRaw(shared_ptr<RawData> &writer) {
    if( !m_isTrasmitting)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    XScopedLock<XEGrabberInterface> lock( *interface());
    Snapshot shot( *this);
    auto camera = interface()->camera();
//DeviceVendorName DeviceModelName Width Height PixelFormat PixelSize PayloadSize DeviceTapGeometry AcquisitionMode AcquisitionStart AcquisitionStop DeviceClockSelector DeviceClockFrequency
    auto int_features = {
        "Width", "Height", "OffsetX", "OffsetY",
        "PayloadSize",
        "SensorWidth", "SensorHeight", "BinningHorizontal", "BinningVertical",
        "DecimationHorizontal", "DecimationVertical",
        "AcquisitionFrameCount",
        "GainRaw"};
    auto enum_features = {"SensorDigitizationTaps",
        "PixelFormat", "PixelColorFilter",
        "AcquisitionMode",
        "TriggerSelector", "TriggerMode", "TriggerSource", "TriggerActivation",
        "ExposureMode",
        "GainSelector",
        "BalanceRatioSelector"};
    auto bool_features = {"ReverseX", "ReverseY"};
    auto float_features = {"AcquisitionFrameRate", "TriggerDelay",
        "ExposureTime", //[us]
        "Gain",
        "Balcklevel", "Gamma"
        };
    try {
        using namespace Euresys;

        writer->push<uint32_t>(int_features.size());
        for(auto &feat: int_features) {
            for(unsigned int i = 0; i < 8; ++i)
                writer->push<char>((strlen(feat) > i) ? feat[i] : '\0'); //first 8 letters
            if(isFeatureAvailableInRemoteModule(feat)) {
                int64_t v = camera->getInteger<RemoteModule>(feat);
                writer->push<int64_t>(v);
            }
            else if( !pushFeatureSerialCommand(writer, feat))
                    writer->push<int64_t>(0);
        }
        writer->push<uint32_t>(enum_features.size());
        for(auto &feat: enum_features) {
            for(unsigned int i = 0; i < 8; ++i)
                writer->push<char>((strlen(feat) > i) ? feat[i] : '\0'); //first 8 letters
            if(isFeatureAvailableInRemoteModule(feat)) {
                int32_t v = camera->getInteger<RemoteModule>(feat);
                writer->push<int32_t>(v);
            }
            else if( !pushFeatureSerialCommand(writer, feat))
                    writer->push<int32_t>(0);
        }
        writer->push<uint32_t>(bool_features.size());
        for(auto &feat: bool_features) {
            for(unsigned int i = 0; i < 8; ++i)
                writer->push<char>((strlen(feat) > i) ? feat[i] : '\0'); //first 8 letters
            if(isFeatureAvailableInRemoteModule(feat)) {
                int16_t v = camera->getInteger<RemoteModule>(feat);
                writer->push<int16_t>(v);
            }
            else if( !pushFeatureSerialCommand(writer, feat))
                    writer->push<int16_t>(0);
        }
        writer->push<uint32_t>(float_features.size());
        for(auto &feat: float_features) {
            for(unsigned int i = 0; i < 8; ++i)
                writer->push<char>((strlen(feat) > i) ? feat[i] : '\0'); //first 8 letters
            if(isFeatureAvailableInRemoteModule(feat)) {
                double v = camera->getFloat<RemoteModule>(feat);
                writer->push<double>(v);
            }
            else if( !pushFeatureSerialCommand(writer, feat))
                    writer->push<double>(0.0);
        }
        writer->push<int64_t>(0); //for future use.

        ScopedBuffer buf( *camera, 10); //10 ms timeout

        void *ptr = buf.getInfo<void *>(GenTL::BUFFER_INFO_BASE);
        uint64_t ts = buf.getInfo<uint64_t>(GenTL::BUFFER_INFO_TIMESTAMP);
        size_t size = buf.getInfo<size_t>(GenTL::BUFFER_INFO_SIZE);
        uint64_t pixelFormat = buf.getInfo<uint64_t>(gc::BUFFER_INFO_PIXELFORMAT);
        size_t width = buf.getInfo<size_t>(gc::BUFFER_INFO_WIDTH);
        size_t height = buf.getInfo<size_t>(gc::BUFFER_INFO_DELIVERED_IMAGEHEIGHT);
        size_t imagesize = buf.getInfo<size_t>(ge::BUFFER_INFO_CUSTOM_PART_SIZE);
        size_t imgpitch = buf.getInfo<size_t>(ge::BUFFER_INFO_CUSTOM_LINE_PITCH);
        size_t delivered = buf.getInfo<size_t>(ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS);
        uint64_t fr = camera->getInteger<StreamModule>("StatisticsFrameRate");
        uint64_t dr = camera->getInteger<StreamModule>("StatisticsDataRate");

        writer->push<uint64_t>(ts);
        writer->push<uint64_t>(size);
        writer->push<uint64_t>(pixelFormat);
        writer->push<uint64_t>(width);
        writer->push<uint64_t>(height);
        writer->push<uint64_t>(imagesize);
        writer->push<uint64_t>(imgpitch);
        writer->push<uint64_t>(delivered);
        writer->push<uint64_t>(fr);
        writer->push<uint64_t>(dr);
        writer->push<int64_t>(0); //for future use.
        writer->push<int64_t>(0); //for future use.

        writer->insert(writer->end(), (char*)ptr, (char*)ptr + size);
    #if defined __WIN32__ || defined WINDOWS || defined _WIN32
        return XTime::now(); //time stamp is invalid for win
    #else
        return XTime(ts / 1000000uLL, ts % 1000000uLL);
    #endif
    }
    catch(const Euresys::gentl_error &e) {
        if(e.gc_err == GenTL::GC_ERR_TIMEOUT)
            throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
        throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
    }
    catch(const std::exception &e) {
        throw XInterface::XInterfaceError(e.what(), __FILE__, __LINE__);
    }
}

XHamamatsuCameraOverGrablink::XHamamatsuCameraOverGrablink(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XEGrabberCamera(name, runtime, ref(tr_meas), meas, true) {
    interface()->setSerialBaudRate(9600);
    interface()->setSerialEOS("\r");
}
bool
XHamamatsuCameraOverGrablink::pushFeatureSerialCommand(shared_ptr<RawData> &writer, const std::string &featname) {
    if(featname == "OffsetX") {
        writer->push<int64_t>(m_offsetx);
        return true;
    }
    if(featname == "OffsetY") {
        writer->push<int64_t>(m_offsety);
        return true;
    }
    if(featname == "SensorWidth") {
        writer->push<int64_t>(m_xdatapx);
        return true;
    }
    if(featname == "SensorHeight") {
        writer->push<int64_t>(m_ydatapx);
        return true;
    }
    if(featname == "Gain") {
        writer->push<double>(m_ceg);
        return true;
    }
    if(featname == "ExposureTime") {
        writer->push<double>(m_rat * 1e6);
        return true;
    }
    return false;
}
std::pair<unsigned int, unsigned int>
XHamamatsuCameraOverGrablink::setVideoModeViaSerial(unsigned int roix, unsigned int roiw, unsigned int roiy, unsigned int roih) {
    roix = roix / 8 * 8;
    roiy = roiy / 8 * 8;
    roiw = (roiw + 7) / 8 * 8;
    roih = (roih + 7) / 8 * 8;
    unsigned int w = m_xdatapx;
    unsigned int h = m_ydatapx;
    if( !roiw || !roih || (roix + roiw >= w) || (roiy + roih >= h) || (roiw > w) || (roih > h)) {
        roix = 0; roiy = 0; roiw = w; roih = h;
    }

    unsigned int binning = 1;
    char smd = 'N';
    if(binning)
        smd = 'S';
    else if(roix || (roiw < w) || roiy || (roih < h))
            smd = 'A';
    interface()->queryf("SMD %c", smd);
    checkSerialError(__FILE__, __LINE__);
    if(smd == 'S') {
        interface()->queryf("SPX %u", binning);
        checkSerialError(__FILE__, __LINE__);
    }
    else if(smd == 'A') {
        interface()->queryf("SHO %u", roix);
        checkSerialError(__FILE__, __LINE__);
        interface()->queryf("SHW %u", roiw);
        checkSerialError(__FILE__, __LINE__);
        interface()->queryf("SVO %u", roiy);
        checkSerialError(__FILE__, __LINE__);
        interface()->queryf("SVW %u", roih);
        checkSerialError(__FILE__, __LINE__);
    }
    m_offsetx = roix;
    m_offsety = roiy;
    return {roiw, roih};
}

void
XHamamatsuCameraOverGrablink::setTriggerModeViaSerial(TriggerMode mode) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    char em = 'T'; //by AET
    char pol = 'P';
    char amd = 'N';
    switch(mode) {
    case TriggerMode::SINGLE:
        em = 'E';
        break;
    case TriggerMode::CONTINUEOUS:
        break;
    case TriggerMode::EXT_POS_EDGE:
        amd = 'E';
        break;
    case TriggerMode::EXT_POS_EXPOSURE:
        amd = 'E';
        em = 'L';
        break;
    case TriggerMode::EXT_NEG_EDGE:
        amd = 'E';
        pol = 'N';
        break;
    case TriggerMode::EXT_NEG_EXPOSURE:
        amd = 'E';
        pol = 'N';
        em = 'L';
        break;
    }
    interface()->queryf("AMD %c", amd);
    checkSerialError(__FILE__, __LINE__);
    if(amd == 'E') {
        interface()->queryf("ATP %c", pol);
        checkSerialError(__FILE__, __LINE__);
        interface()->queryf("EMD %c", em);
        checkSerialError(__FILE__, __LINE__);
    }
//    if(mode == TriggerMode::SINGLE) {
//        interface()->send("EST");
//        checkSerialError(__FILE__, __LINE__);
//    }
}
void
XHamamatsuCameraOverGrablink::setBlackLevelOffset(unsigned int v) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    interface()->queryf("CEO %u", v);
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
}
void
XHamamatsuCameraOverGrablink::setExposureTime(double shutter) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    unsigned int mm = shutter / 60;
    shutter -= mm * 60;
    if(mm)
        interface()->queryf("AET %u:%.3f", mm, shutter);
    else
        interface()->queryf("AET %.3f", shutter);
    checkSerialError(__FILE__, __LINE__);
    interface()->query("?TMP");
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?RAT");
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->scanf("RAT %lf", &m_rat);
}
void
XHamamatsuCameraOverGrablink::setGain(unsigned int v, unsigned int emgain) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    if(m_bIsEM)
        interface()->queryf("EMG %u", emgain);
    interface()->queryf("CEG %u", v);
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?EMG");
    if(interface()->scanf("EMG%d", &m_emg) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    interface()->query("?CEG");
    if(interface()->scanf("CEG%d", &m_ceg) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
}

void
XHamamatsuCameraOverGrablink::afterOpen() {
    XScopedLock<XEGrabberInterface> lock( *interface()); //for serial opening.
    interface()->query("RES Y");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?CAI T");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?VER");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?INF");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());

//    interface()->query("?CAI H");
//    checkSerialError(__FILE__, __LINE__);
//    fprintf(stderr, "%s\n", interface()->toStr().c_str());
//    interface()->query("?CAI V");
//    checkSerialError(__FILE__, __LINE__);
//    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?CAI A");
    checkSerialError(__FILE__, __LINE__);
    unsigned int x, y;
    if(interface()->scanf("CAI A%u,%u,%u,%u,%u,%u", &m_xdummypx, &x, &m_xdatapx, &m_ydummypx, &y, &m_ydatapx) != 6)
        throw XInterface::XConvError(__FILE__, __LINE__);
    m_xdummypx += x; m_ydummypx += y;
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?CAI I");
    checkSerialError(__FILE__, __LINE__);
    if(interface()->scanf("CAI I%u", &m_maxBitsFast) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());

    interface()->query("?CAI S");
    if(interface()->scanf("CAI S%u", &m_maxBitsSlow) == 1)
        m_bHasSlowScan = true;
    fprintf(stderr, "%s\n", interface()->toStr().c_str());

    interface()->query("?CAI O");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());

    double v;
    interface()->query("?TMP");
    if(interface()->scanf("TMP %lf", &v) == 1)
        m_bIsCooled = true;
    fprintf(stderr, "%s\n", interface()->toStr().c_str());

    interface()->query("?RAT");
    if(interface()->scanf("RAT%lf", &v) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    trans( *exposureTime()) = v;
    m_rat = v;
    fprintf(stderr, "%s\n", interface()->toStr().c_str());

    y = 0;
    interface()->query("?EMG");
    if(interface()->scanf("EMG%d", &y) == 1) {
        m_bIsEM = true;
        m_emg = y;
        trans( *emGain()) = y;
    }
    else
        emGain()->disable();
    fprintf(stderr, "%s\n", interface()->toStr().c_str());

    interface()->query("?CEG");
    if(interface()->scanf("CEG%d", &x) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    m_ceg = x;
    trans( *cameraGain()) = x;
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?CEO");
    if(interface()->scanf("CEO%d", &x) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    trans( *blackLvlOffset()) = x;
    fprintf(stderr, "%s\n", interface()->toStr().c_str());

    interface()->query("?ATP");
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?AMD");
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?SMD");
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?EMD");
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?ESC");
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("?SPX");
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
}

void
XHamamatsuCameraOverGrablink::checkSerialError(const char *file, unsigned int line) {
    XString str = interface()->toStrSimplified();
    if(str.empty())
        throw XInterface::XConvError(file, line);
    unsigned int code;
    if(interface()->scanf("E%u", &code) == 1) {
        switch(code) {
        case 1:
            throw XInterface::XInterfaceError(i18n("Comm. error."), file, line);
        case 2:
            throw XInterface::XInterfaceError(i18n("Comm. buffer overflows."), file, line);
        case 3:
            throw XInterface::XInterfaceError(i18n("Unknown command or parameter."), file, line);
        case 4:
            throw XInterface::XInterfaceError(i18n("Command unsuitable for current condition."), file, line);
        case 5:
            throw XInterface::XInterfaceError(i18n("Wrong parameter"), file, line);
        case 6:
            throw XInterface::XInterfaceError(i18n("Parameter unsuitable for current condition."), file, line);
        default:
            throw XInterface::XInterfaceError(i18n("Unknown error code."), file, line);
        }
    }
}


XJAICameraOverGrablink::XJAICameraOverGrablink(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XEGrabberCamera(name, runtime, ref(tr_meas), meas, true) {
    interface()->setSerialBaudRate(9600);
    interface()->setSerialEOS("\r\n");
    emGain()->disable();
}

std::pair<unsigned int, unsigned int>
XJAICameraOverGrablink::setVideoModeViaSerial(unsigned int roix, unsigned int roiw, unsigned int roiy, unsigned int roih) {
    roix = roix / 8 * 8;
    roiy = roiy / 8 * 8;
    roiw = (roiw + 7) / 8 * 8;
    roih = (roih + 7) / 8 * 8;
    unsigned int w = m_sensorWidth;
    unsigned int h = m_sensorHeight;
    if( !roiw || !roih || (roix + roiw >= w) || (roiy + roih >= h) || (roiw > w) || (roih > h)) {
        roix = 0; roiy = 0; roiw = w; roih = h;
    }

    interface()->queryf("OFC=%u", roix);
    checkSerialError(__FILE__, __LINE__);
    interface()->queryf("WTC=%u", roiw);
    checkSerialError(__FILE__, __LINE__);
    interface()->queryf("OFL=%u", roiy);
    checkSerialError(__FILE__, __LINE__);
    interface()->queryf("HTL=%u", roih);
    checkSerialError(__FILE__, __LINE__);

    Snapshot shot( *this);
    unsigned int mode = shot[ *videoMode()];
    interface()->queryf("VPB=%u", (mode >= 2) ? 0 : 1); //12 bit
    checkSerialError(__FILE__, __LINE__);
    interface()->queryf("BA=%u", mode);
    checkSerialError(__FILE__, __LINE__);

    return {roiw, roih};
}

void
XJAICameraOverGrablink::setTriggerModeViaSerial(TriggerMode mode) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    unsigned int em = 1, act = 0, asc = 0;
    switch(mode) {
    case TriggerMode::SINGLE:
        break;
    case TriggerMode::CONTINUEOUS:
        asc = 1;
        break;
    case TriggerMode::EXT_POS_EDGE:
        act = 0;
        break;
    case TriggerMode::EXT_POS_EXPOSURE:
        act = 2;
        em = 2;
        break;
    case TriggerMode::EXT_NEG_EDGE:
        act = 1;
        break;
    case TriggerMode::EXT_NEG_EXPOSURE:
        act = 3;
        em = 2;
        break;
    }
    //For Base configuration, 1CL
    interface()->queryf("TAGM=%u", 1u); //DeviceTapGeometry, Geometry_1X2_1Y
    checkSerialError(__FILE__, __LINE__);
    interface()->camera()->setString<Euresys::StreamModule>("DeviceTapGeometrySource", "DataStream");
    interface()->camera()->setString<Euresys::StreamModule>("StripeArrangement", "Geometry_1X2_1Y");

    interface()->queryf("EM=%u", em); //ExposureMode
    checkSerialError(__FILE__, __LINE__);
    interface()->queryf("ASC=%u", asc); //ExposureAuto
    checkSerialError(__FILE__, __LINE__);
    interface()->queryf("TA=%u", act); //FrameStartTrigActivation
    checkSerialError(__FILE__, __LINE__);
    if(mode == TriggerMode::SINGLE) {
        interface()->query("STRG=0"); //TriggerSoftware
        checkSerialError(__FILE__, __LINE__);
    }
}
void
XJAICameraOverGrablink::setBlackLevelOffset(unsigned int v) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    interface()->queryf("BL=%u", v);
    checkSerialError(__FILE__, __LINE__);
}
void
XJAICameraOverGrablink::setExposureTime(double shutter) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    interface()->queryf("PE=%lu", lrint(shutter * 1e6)); //ExposureTimeRaw
    checkSerialError(__FILE__, __LINE__);
    interface()->query("TMP0?");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
}
void
XJAICameraOverGrablink::setGain(unsigned int g, unsigned int emgain) {
    XScopedLock<XEGrabberInterface> lock( *interface());
    interface()->queryf("FGA=%u", g);
    checkSerialError(__FILE__, __LINE__);
}
void
XJAICameraOverGrablink::afterOpen() {
    XScopedLock<XEGrabberInterface> lock( *interface()); //for serial opening.
    interface()->query("DVN?");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("MD?");
    checkSerialError(__FILE__, __LINE__);
    if(interface()->toStrSimplified().find("GO-24") != std::string::npos) {
        m_sensorWidth = 1936;
        m_sensorHeight = 1216;
    }
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("DV?");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
    interface()->query("VN?");
    checkSerialError(__FILE__, __LINE__);
    fprintf(stderr, "%s\n", interface()->toStr().c_str());
}
void
XJAICameraOverGrablink::checkSerialError(const char *file, unsigned int line) {
    XString str = interface()->toStrSimplified();
    if(str.empty())
        throw XInterface::XConvError(file, line);
    if(str[0] == '0') {
        unsigned int code;
        if(interface()->scanf("%2u", &code) != 1)
            throw XInterface::XConvError(file, line);
        throw XInterface::XInterfaceError(str, file, line);
    }
}

#endif //USE_EURESYS_EGRABBER
