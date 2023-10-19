/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "iidccamera.h"
#include "analyzer.h"
#include <QImage>

#if defined USE_LIBDC1394

REGISTER_TYPE(XDriverList, IIDCCamera, "IEEE1394 IIDC Camera");

dc1394_t *XDC1394Interface::s_dc1394 = nullptr;
int XDC1394Interface::s_refcnt = 0;
XRecursiveMutex XDC1394Interface::s_mutex;

//---------------------------------------------------------------------------
XDC1394Interface::XDC1394Interface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
    XInterface(name, runtime, driver) {
    XScopedLock<XDC1394Interface> lock( *this);
    if(s_refcnt++ == 0) {
        s_dc1394 = dc1394_new();
    }
    dc1394camera_list_t *list;
    dc1394error_t err = dc1394_camera_enumerate(s_dc1394, &list);
    if( !err) {
        for(unsigned int i = 0; i < list->num; ++i) {
            dc1394camera_t *camera = dc1394_camera_new(s_dc1394, list->ids[i].guid);
            if(camera) {
                err = dc1394_camera_print_info(camera, stdout);
                if( !err) {
                    trans( *device()).add(camera->model);
                }
                dc1394_camera_free(camera);
            }
        }
        dc1394_camera_free_list(list);
    }
}
XDC1394Interface::~XDC1394Interface() {
    XScopedLock<XDC1394Interface> lock( *this);
    if(--s_refcnt == 0) {
        dc1394_free(s_dc1394);
    }

}
void
XDC1394Interface::open() {
    Snapshot shot( *this);
    dc1394camera_list_t *list;
    dc1394error_t err = dc1394_camera_enumerate(s_dc1394, &list);
    if( !err) {
        for(unsigned int i = 0; i < list->num; ++i) {
            dc1394camera_t *camera = dc1394_camera_new(s_dc1394, list->ids[i].guid);
            if(camera) {
                if( !err) {
                    if(shot[ *device()].to_str() == camera->model) {
                        m_devname = camera->model;
                        m_camera = camera;
                    }
                    else
                        dc1394_camera_free(camera);
                }
            }
        }
        dc1394_camera_free_list(list);
    }

}
void
XDC1394Interface::close() {
    if(m_camera) {
        dc1394_video_set_transmission(m_camera, DC1394_OFF);
//        if(m_camera->has_vmode_error_status != DC1394_TRUE)
        dc1394_capture_stop(m_camera);
        msecsleep(200); //some waits needed when buffer is not empty!!!!!
        dc1394_camera_free(m_camera);
    }
    m_camera = nullptr;
}


XIIDCCamera::XIIDCCamera(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDC1394Driver<XDigitalCamera>(name, runtime, ref(tr_meas), meas) {
//    startWavelen()->disable();
//    stopWavelen()->disable();
}

void
XIIDCCamera::open() {
    if( !interface()->camera())
        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    m_isTrasmitting = false;
    // get video modes:
    dc1394video_modes_t video_modes;
    if(dc1394_video_get_supported_modes(interface()->camera(),&video_modes)) {
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get video modes."), __FILE__, __LINE__);
    }

    const std::map<dc1394video_mode_t, const char *> iidcVideoModes = {
        {DC1394_VIDEO_MODE_160x120_YUV444, "160x120 YUV444"},
        {DC1394_VIDEO_MODE_320x240_YUV422, "320x240 YUV422"},
        {DC1394_VIDEO_MODE_640x480_YUV411, "640x480 YUV411"},
        {DC1394_VIDEO_MODE_640x480_YUV422, "640x480 YUV422"},
        {DC1394_VIDEO_MODE_640x480_RGB8, "640x480 RGB8"},
        {DC1394_VIDEO_MODE_640x480_MONO8, "640x480 MONO8"},
        {DC1394_VIDEO_MODE_640x480_MONO16, "640x480 MONO16"},
        //                DC1394_VIDEO_MODE_800x600_YUV422,
        //                DC1394_VIDEO_MODE_800x600_RGB8,
        {DC1394_VIDEO_MODE_800x600_MONO8, "800x600_MONO8"},
        //                DC1394_VIDEO_MODE_1024x768_YUV422,
        //                DC1394_VIDEO_MODE_1024x768_RGB8,
        {DC1394_VIDEO_MODE_1024x768_MONO8, "1024x768_MONO8"},
        {DC1394_VIDEO_MODE_800x600_MONO16, "800x600_MONO16"},
        {DC1394_VIDEO_MODE_1024x768_MONO16, "1024x768_MONO16"},
        //                DC1394_VIDEO_MODE_1280x960_YUV422,
        //                DC1394_VIDEO_MODE_1280x960_RGB8,
        {DC1394_VIDEO_MODE_1280x960_MONO8, "1280x960_MONO8"},
        //                DC1394_VIDEO_MODE_1600x1200_YUV422,
        //                DC1394_VIDEO_MODE_1600x1200_RGB8,
        {DC1394_VIDEO_MODE_1600x1200_MONO8, "1600x1200_MONO8"},
        {DC1394_VIDEO_MODE_1280x960_MONO16, "1280x960_MONO16"},
        {DC1394_VIDEO_MODE_1600x1200_MONO16, "1600x1200_MONO16"},
        //                DC1394_VIDEO_MODE_EXIF,
        {DC1394_VIDEO_MODE_FORMAT7_0, "FORMAT7 0"},
        {DC1394_VIDEO_MODE_FORMAT7_1, "FORMAT7 1"},
        {DC1394_VIDEO_MODE_FORMAT7_2, "FORMAT7 2"},
        {DC1394_VIDEO_MODE_FORMAT7_3, "FORMAT7 3"},
        {DC1394_VIDEO_MODE_FORMAT7_4, "FORMAT7 4"},
        {DC1394_VIDEO_MODE_FORMAT7_5, "FORMAT7 5"},
        {DC1394_VIDEO_MODE_FORMAT7_6, "FORMAT7 6"},
        {DC1394_VIDEO_MODE_FORMAT7_7, "FORMAT7 7"},
    };

    const std::map<dc1394color_coding_t, const char *> iidcColorCodings = {
        {DC1394_COLOR_CODING_MONO8, "MONO8"},
        {DC1394_COLOR_CODING_YUV411, "YUV411"},
        {DC1394_COLOR_CODING_YUV422, "YUV422"},
        {DC1394_COLOR_CODING_YUV444, "YUV444"},
        {DC1394_COLOR_CODING_RGB8, "RGB8"},
        {DC1394_COLOR_CODING_MONO16, "MONO16"},
        {DC1394_COLOR_CODING_RGB16, "RGB16"},
        {DC1394_COLOR_CODING_MONO16S, "MONO16S"},
        {DC1394_COLOR_CODING_RGB16S, "RGB16S"},
        {DC1394_COLOR_CODING_RAW8, "RAW8"},
        {DC1394_COLOR_CODING_RAW16, "RAW16"}
    };

    m_availableVideoModes.clear();
    std::vector<XString> modestrings;
    for(unsigned int i = 0; i < video_modes.num; ++i) {
        dc1394video_mode_t video_mode=video_modes.modes[i];
        if( !dc1394_is_video_mode_scalable(video_mode)) {
            dc1394color_coding_t coding;
            dc1394_get_color_coding_from_video_mode(interface()->camera(),video_mode, &coding);
            try {
                modestrings.push_back(iidcVideoModes.at(video_mode));
                m_availableVideoModes.push_back({video_mode, coding});
            }
            catch(std::out_of_range &) {
                modestrings.push_back(formatString("Mode %u", (unsigned int)video_mode));
                m_availableVideoModes.push_back({video_mode, coding});
            }
        }
        else {
            unsigned int w, h;
            dc1394_format7_get_max_image_size(interface()->camera(), video_mode, &w, &h);
            dc1394color_codings_t codings;
            dc1394_format7_get_color_codings(interface()->camera(), video_mode, &codings);
            for(unsigned int j = 0; j < codings.num; ++j) {
                modestrings.push_back(formatString("%ux%u ", w, h) + iidcColorCodings.at(codings.codings[j]));
                m_availableVideoModes.push_back({video_mode, codings.codings[j]});
            }
        }
    }

    TriggerMode trigmode = TriggerMode::CONTINUEOUS;
    dc1394switch_t powered;
    if(dc1394_external_trigger_get_power(interface()->camera(), &powered))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
    if(powered) {
        dc1394trigger_mode_t mode;
        if(dc1394_external_trigger_get_mode(interface()->camera(), &mode))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
        dc1394trigger_polarity_t pl;
        if(dc1394_external_trigger_get_polarity(interface()->camera(), &pl))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
        if(pl ==  DC1394_TRIGGER_ACTIVE_HIGH) {
            if(mode == DC1394_TRIGGER_MODE_0)
                trigmode = TriggerMode::EXT_POS_EDGE;
            else
                trigmode = TriggerMode::EXT_POS_EXPOSURE;
        }
        else {
            if(mode == DC1394_TRIGGER_MODE_0)
                trigmode = TriggerMode::EXT_NEG_EDGE;
            else
                trigmode = TriggerMode::EXT_NEG_EXPOSURE;
        }
    }
    else {
        dc1394bool_t is_on;
        if(dc1394_video_get_one_shot(interface()->camera(), &is_on))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
        if(is_on == DC1394_TRUE)
            trigmode = TriggerMode::SINGLE;
    }

    if(dc1394_video_set_iso_speed(interface()->camera(), DC1394_ISO_SPEED_400))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set 400 Msps."), __FILE__, __LINE__);

    auto fn_is_feature_present = [&](dc1394feature_t feature){
        dc1394bool_t v;
        if(dc1394_feature_is_present(interface()->camera(), feature, &v))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
        return v;
    };
    auto fn_get_feature_values = [&](dc1394feature_t feature){
        unsigned int vmin, vmax, v;
        if(dc1394_feature_get_boundaries(interface()->camera(), feature, &vmin, &vmax))
                    throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
        if(dc1394_feature_get_value(interface()->camera(), feature, &v))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
        return std::tuple<unsigned int, unsigned int, unsigned int>{v, vmin, vmax};
    };
    auto fn_get_feature_absolute_values = [&](dc1394feature_t feature){
        float vmin, vmax, v;
        if(dc1394_feature_get_absolute_boundaries(interface()->camera(), feature, &vmin, &vmax))
                    throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
        if(dc1394_feature_get_absolute_value(interface()->camera(), feature, &v))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
        return std::tuple<float, float, float>{v, vmin, vmax};
    };
    if(fn_is_feature_present(DC1394_FEATURE_BRIGHTNESS)) {
        auto [v, vmin, vmax] = fn_get_feature_values(DC1394_FEATURE_BRIGHTNESS);
        trans( *brightness()) = v;
    }
    if(fn_is_feature_present(DC1394_FEATURE_SHUTTER)) {
        auto [v, vmin, vmax] = fn_get_feature_absolute_values(DC1394_FEATURE_SHUTTER);
        trans( *exposureTime()) = v;
    }
    if(fn_is_feature_present(DC1394_FEATURE_GAIN)) {
        auto [v, vmin, vmax] = fn_get_feature_values(DC1394_FEATURE_GAIN);
        //Hamamatsu ORCA
        v = lrint(20 * v / 255.0);
        trans( *cameraGain()) = v;
    }

    iterate_commit([=](Transaction &tr){
        tr[ *videoMode()] = -1;
        tr[ *videoMode()].clear();
        for(auto &s: modestrings)
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

    start();
}

void
XIIDCCamera::stopTransmission() {
    XScopedLock<XDC1394Interface> lock( *interface());
    m_isTrasmitting = false;
    if(dc1394_video_set_transmission(interface()->camera(), DC1394_OFF))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not stop transmission."), __FILE__, __LINE__);
    msecsleep(100);
//    if(interface()->camera()->has_vmode_error_status != DC1394_TRUE)
    dc1394_capture_stop(interface()->camera());
}
void
XIIDCCamera::setVideoMode(unsigned int mode, unsigned int roix, unsigned int roiy, unsigned int roiw, unsigned int roih) {
    XScopedLock<XDC1394Interface> lock( *interface());
    stopTransmission();
    Snapshot shot( *this);
    dc1394video_mode_t video_mode;
    dc1394color_coding_t coding;
    try {
        auto p = m_availableVideoModes.at(mode);
        video_mode = p.first;
        coding = p.second;
    }
    catch(std::out_of_range &) {
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set video modes."), __FILE__, __LINE__);
    }
    if(dc1394_video_set_mode(interface()->camera(), video_mode))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set video modes."), __FILE__, __LINE__);

    if(dc1394_is_video_mode_scalable(video_mode)) {
        unsigned int w, h;
        dc1394_format7_get_max_image_size(interface()->camera(), video_mode, &w, &h);
//        if(dc1394_format7_set_color_coding(interface()->camera(), video_mode, coding))
        roix = roix / 4 * 4;
        roiy = roiy / 4 * 4;
        roiw = (roiw + 3) / 4 * 4;
        roih = (roih + 3) / 4 * 4;
        if( !roiw || !roih || (roix + roiw >= w) || (roiy + roih >= h) || (roiw > w) || (roih > h)) {
            roix = 0; roiy = 0; roiw = w; roih = h;
        }
        if(dc1394_format7_set_roi(interface()->camera(), video_mode, coding,
                                     DC1394_USE_MAX_AVAIL, roix, roiy, roiw, roih))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set video modes."), __FILE__, __LINE__);

//        double rate = 5;
//        sscanf(shot[ *frameRate()].to_str().c_str(), "%lf", &rate);
//        uint32_t bits;
//        dc1394_get_color_coding_bit_size(coding, &bits);
//        uint32_t bytepersec = rate * w * h * (bits/8) * 125e-6;

//        if(dc1394_format7_set_packet_size(interface()->camera(), video_mode, bytepersec))
//            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set framerate."), __FILE__, __LINE__);
    }
    else {
//        // get highest framerate
//        dc1394framerates_t framerates;
//        if(dc1394_video_get_supported_framerates(interface()->camera(), video_mode,&framerates))
//            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get framerates."), __FILE__, __LINE__);
        dc1394framerate_t rate = (dc1394framerate_t)(DC1394_FRAMERATE_240 - (int)shot[ *frameRate()]);
        if(rate < DC1394_FRAMERATE_1_875)
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set framerate."), __FILE__, __LINE__);
        if(dc1394_video_set_framerate(interface()->camera(), rate))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set framerate."), __FILE__, __LINE__);
    }

    dc1394featureset_t features;
    if(dc1394_feature_get_all(interface()->camera(), &features) == DC1394_SUCCESS)
        dc1394_feature_print_all(&features, stdout);

    setTriggerMode(static_cast<TriggerMode>((unsigned int)shot[ *triggerMode()]));
}
void
XIIDCCamera::setTriggerMode(TriggerMode mode) {
    XScopedLock<XDC1394Interface> lock( *interface());
    stopTransmission();
    if(dc1394_software_trigger_set_power(interface()->camera(), DC1394_OFF))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not stop transmission."), __FILE__, __LINE__);
    if(dc1394_external_trigger_set_power(interface()->camera(), DC1394_OFF))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set info.."), __FILE__, __LINE__);

    if(mode == TriggerMode::SINGLE){
        if(dc1394_software_trigger_set_power(interface()->camera(), DC1394_ON))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not start transmission."), __FILE__, __LINE__);
        if(dc1394_capture_setup(interface()->camera(), 4, DC1394_CAPTURE_FLAGS_DEFAULT))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not setup capture."), __FILE__, __LINE__);
        msecsleep(50); //exposure
        if(dc1394_video_set_one_shot(interface()->camera(), DC1394_ON))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not take a shot."), __FILE__, __LINE__);

        m_isTrasmitting = true;
        return;
    }

    if(mode != TriggerMode::CONTINUEOUS) {
         std::map<TriggerMode, std::pair<dc1394trigger_mode_t, dc1394trigger_polarity_t>> modes = {
             {TriggerMode::EXT_POS_EDGE, {DC1394_TRIGGER_MODE_0, DC1394_TRIGGER_ACTIVE_HIGH}},
             {TriggerMode::EXT_NEG_EDGE, {DC1394_TRIGGER_MODE_0, DC1394_TRIGGER_ACTIVE_LOW}},
             {TriggerMode::EXT_POS_EXPOSURE, {DC1394_TRIGGER_MODE_1, DC1394_TRIGGER_ACTIVE_HIGH}},
             {TriggerMode::EXT_NEG_EXPOSURE, {DC1394_TRIGGER_MODE_1, DC1394_TRIGGER_ACTIVE_LOW}},
         };
        if(dc1394_external_trigger_set_mode(interface()->camera(), modes.at(mode).first))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set info.."), __FILE__, __LINE__);
        if(dc1394_external_trigger_set_polarity(interface()->camera(), modes.at(mode).second))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set info.."), __FILE__, __LINE__);
        if(dc1394_external_trigger_set_power(interface()->camera(), DC1394_ON))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set info.."), __FILE__, __LINE__);
    }
    if(dc1394_capture_setup(interface()->camera(), 6, DC1394_CAPTURE_FLAGS_DEFAULT))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not setup capture."), __FILE__, __LINE__);

    if(dc1394_video_set_transmission(interface()->camera(), DC1394_ON))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not start transmission."), __FILE__, __LINE__);

    m_isTrasmitting = true;
}
void
XIIDCCamera::setBrightness(unsigned int brightness) {
    XScopedLock<XDC1394Interface> lock( *interface());
//    stopTransmission();
    if(dc1394_feature_set_value(interface()->camera(), DC1394_FEATURE_BRIGHTNESS, brightness))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
}
void
XIIDCCamera::setExposureTime(double shutter) {
    XScopedLock<XDC1394Interface> lock( *interface());
//    stopTransmission();
    if(dc1394_feature_set_absolute_value(interface()->camera(), DC1394_FEATURE_SHUTTER, shutter))
//    if(dc1394_feature_set_value(interface()->camera(), DC1394_FEATURE_SHUTTER, shutter))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
}
void
XIIDCCamera::setCameraGain(double db) {
    XScopedLock<XDC1394Interface> lock( *interface());
//    stopTransmission();
    //Hamamatsu ORCA
    int gain = lrint(255.0 * db / 20.0);
    gain = std::min(std::max(0, gain), 255);
    if(dc1394_feature_set_value(interface()->camera(), DC1394_FEATURE_GAIN, gain))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
}

void
XIIDCCamera::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    uint32_t width = reader.pop<uint32_t>();
    uint32_t height = reader.pop<uint32_t>();
    auto xpos = reader.pop<uint32_t>();
    auto ypos = reader.pop<uint32_t>();
    auto color_coding = static_cast<dc1394color_coding_t>(reader.pop<uint32_t>());          /* the color coding used. This field is valid for all video modes. */
    auto color_filter = static_cast<dc1394color_filter_t>(reader.pop<uint32_t>());          /* the color filter used. This field is valid only for RAW modes and IIDC 1.31 */
    uint32_t                 yuv_byte_order = reader.pop<uint32_t>();        /* the order of the fields for 422 formats: YUYV or UYVY */
    uint32_t                 data_depth = reader.pop<uint32_t>();            /* the number of bits per pixel. The number of grayscale levels is 2^(this_number).
                                                       This is independent from the colour coding */
    uint32_t                 stride = reader.pop<uint32_t>();                /* the number of bytes per image line */
    auto video_mode = static_cast<dc1394video_mode_t>(reader.pop<uint32_t>());            /* the video mode used for capturing this frame */
    uint64_t                 total_bytes = reader.pop<uint64_t>();           /* the total size of the frame buffer in bytes. May include packet-
                                                       multiple padding and intentional padding (vendor specific) */
    uint32_t                 image_bytes = reader.pop<uint32_t>();          /* the number of bytes used for the image (image data only, no padding) */
    uint32_t                 padding_bytes = reader.pop<uint32_t>();         /* the number of extra bytes, i.e. total_bytes-image_bytes.  */
    uint32_t                 packet_size = reader.pop<uint32_t>();           /* the size of a packet in bytes. (IIDC data) */
    uint32_t                 packets_per_frame = reader.pop<uint32_t>();     /* the number of packets per frame. (IIDC data) */
    uint64_t                 timestamp = reader.pop<uint64_t>();             /* the unix time [microseconds] at which the frame was captured in
                                                       the video1394 ringbuffer */
    uint32_t                 frames_behind = reader.pop<uint32_t>();         /* the number of frames in the ring buffer that are yet to be accessed by the user */
    uint32_t                 id = reader.pop<uint32_t>();                    /* the frame position in the ring buffer */
    uint64_t                 allocated_image_bytes = reader.pop<uint64_t>(); /* amount of memory allocated in for the *image field. */
    auto little_endian = static_cast<dc1394bool_t>(reader.pop<uint32_t>());         /* DC1394_TRUE if little endian (16bpp modes only),
                                                       DC1394_FALSE otherwise */
    auto data_in_padding = static_cast<dc1394bool_t>(reader.pop<uint32_t>());       /* DC1394_TRUE if data is present in the padding bytes in IIDC 1.32 format,
                                                       DC1394_FALSE otherwise */
    unsigned int bpp = image_bytes / (width * height);
    tr[ *this].m_status = formatString("%ux%u @(%u,%u)", width, height, xpos, ypos)+  tr[ *this].time().getTimeStr() + formatString(" behind:%u", frames_behind);

    XTime time = {(long)(timestamp / 1000000uLL), (long)(timestamp % 1000000uLL)};

    setGrayImage(reader, tr, width, height, little_endian != DC1394_TRUE, bpp == 2);
    reader.popIterator() += padding_bytes;
}

XTime
XIIDCCamera::acquireRaw(shared_ptr<RawData> &writer) {
    XScopedLock<XDC1394Interface> lock( *interface());
    Snapshot shot( *this);
    if( !m_isTrasmitting)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

    dc1394video_frame_t *frame;
    auto ret = dc1394_capture_dequeue(interface()->camera(), DC1394_CAPTURE_POLICY_POLL, &frame);
    if((ret < 0) || !frame)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    if(ret)
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not capture."), __FILE__, __LINE__);

    writer->push((uint32_t)frame->size[0]);
    writer->push((uint32_t)frame->size[1]);
    writer->push((uint32_t)frame->position[0]);
    writer->push((uint32_t)frame->position[1]);
    writer->push((uint32_t)frame->color_coding);
    writer->push((uint32_t)frame->color_filter);
    writer->push((uint32_t)frame->yuv_byte_order);
    writer->push((uint32_t)frame->data_depth);
    writer->push((uint32_t)frame->stride);
    writer->push((uint32_t)frame->video_mode);
    writer->push((uint64_t)frame->total_bytes);
    writer->push((uint32_t)frame->image_bytes);
    writer->push((uint32_t)frame->padding_bytes);
    writer->push((uint32_t)frame->packet_size);
    writer->push((uint32_t)frame->packets_per_frame);
    writer->push((uint64_t)frame->timestamp);
    writer->push((uint32_t)frame->frames_behind);
    writer->push((uint32_t)frame->id);
    writer->push((uint64_t)frame->allocated_image_bytes);
    writer->push((uint32_t)frame->little_endian);
    writer->push((uint32_t)frame->data_in_padding);
    writer->insert(writer->end(), (char*)frame->image, (char*)frame->image + frame->padding_bytes + frame->image_bytes);

    if(dc1394_capture_enqueue(interface()->camera(), frame))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not release frame."), __FILE__, __LINE__);

    return XTime{(long)(frame->timestamp / 1000000uLL), (long)(frame->timestamp % 1000000uLL)};
}

#endif // USE_LIBDC1394
