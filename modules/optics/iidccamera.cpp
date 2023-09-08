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

const std::map<dc1394video_mode_t, const char *> XIIDCCamera::s_iidcVideoModes = {
    //                DC1394_VIDEO_MODE_160x120_YUV444= 64,
    //                DC1394_VIDEO_MODE_320x240_YUV422,
    //                DC1394_VIDEO_MODE_640x480_YUV411,
    //                DC1394_VIDEO_MODE_640x480_YUV422,
    //                DC1394_VIDEO_MODE_640x480_RGB8,
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
        dc1394_capture_stop(m_camera);
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
    // get video modes:
    dc1394video_modes_t video_modes;
    if(dc1394_video_get_supported_modes(interface()->camera(),&video_modes)) {
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get video modes."), __FILE__, __LINE__);
    }


    std::vector<XString> modestrings;
    for(int i = video_modes.num - 1; i >= 0; --i) {
        if( !dc1394_is_video_mode_scalable(video_modes.modes[i])) {
            dc1394color_coding_t coding;
            dc1394_get_color_coding_from_video_mode(interface()->camera(),video_modes.modes[i], &coding);
            dc1394video_mode_t video_mode=video_modes.modes[i];
            try {
                modestrings.push_back(s_iidcVideoModes.at(video_mode));
            }
            catch(std::out_of_range &) {
                modestrings.push_back(formatString("Mode %u", (unsigned int)video_mode));
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
    if(fn_is_feature_present(DC1394_FEATURE_GAIN)) {
        auto [v, vmin, vmax] = fn_get_feature_values(DC1394_FEATURE_GAIN);
        trans( *gain()) = v;
    }
    if(fn_is_feature_present(DC1394_FEATURE_SHUTTER)) {
        auto [v, vmin, vmax] = fn_get_feature_values(DC1394_FEATURE_SHUTTER);
        trans( *shutter()) = v;
    }


    iterate_commit([=](Transaction &tr){
        tr[ *videoMode()].clear();
        for(auto &s: modestrings)
            tr[ *videoMode()].add(s);
        tr[ *triggerMode()] = (unsigned int)trigmode;
    });

    start();
}

void
XIIDCCamera::setVideoMode(unsigned int mode) {
    XScopedLock<XDC1394Interface> lock( *interface());
    Snapshot shot( *this);
    dc1394video_mode_t video_mode = {};
    for(auto &s: s_iidcVideoModes) {
        if(s.second == shot[ *videoMode()].to_str()) {
            video_mode = s.first;
        }
    }
    if(video_mode ==  dc1394video_mode_t{}) {
        unsigned int m;
        if(sscanf(shot[ *videoMode()].to_str().c_str(), "Mode %u", &m) != 1)
            return;
        video_mode = static_cast<dc1394video_mode_t>(m);
    }
    if(dc1394_video_set_mode(interface()->camera(), video_mode))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set video modes."), __FILE__, __LINE__);
    // get highest framerate
    dc1394framerates_t framerates;
    if(dc1394_video_get_supported_framerates(interface()->camera(),video_mode,&framerates))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get framerates."), __FILE__, __LINE__);
    dc1394framerate_t framerate=framerates.framerates[framerates.num - 1];
    if(dc1394_video_set_framerate(interface()->camera(), framerate))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set framerate."), __FILE__, __LINE__);

    dc1394featureset_t features;
    if(dc1394_feature_get_all(interface()->camera(), &features) == DC1394_SUCCESS)
        dc1394_feature_print_all(&features, stdout);

    setTriggerMode(static_cast<TriggerMode>((unsigned int)shot[ *triggerMode()]));
}
void
XIIDCCamera::setTriggerMode(TriggerMode mode) {
    XScopedLock<XDC1394Interface> lock( *interface());
    if(dc1394_video_set_transmission(interface()->camera(), DC1394_OFF))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not stop transmission."), __FILE__, __LINE__);
    if(dc1394_software_trigger_set_power(interface()->camera(), DC1394_OFF))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not stop transmission."), __FILE__, __LINE__);

    if(mode == TriggerMode::SINGLE){
        if(dc1394_software_trigger_set_power(interface()->camera(), DC1394_ON))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not start transmission."), __FILE__, __LINE__);
        if(dc1394_capture_setup(interface()->camera(), 4, DC1394_CAPTURE_FLAGS_DEFAULT))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not setup capture."), __FILE__, __LINE__);
        msecsleep(500); //exposure
        if(dc1394_video_set_one_shot(interface()->camera(), DC1394_ON))
            throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not take a shot."), __FILE__, __LINE__);
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
    }

    if(dc1394_capture_setup(interface()->camera(), 6, DC1394_CAPTURE_FLAGS_DEFAULT))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not setup capture."), __FILE__, __LINE__);

    if(dc1394_video_set_transmission(interface()->camera(), DC1394_ON))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not start transmission."), __FILE__, __LINE__);
}
void
XIIDCCamera::setGain(unsigned int gain) {
    XScopedLock<XDC1394Interface> lock( *interface());
    if(dc1394_feature_set_value(interface()->camera(), DC1394_FEATURE_GAIN, gain))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get info.."), __FILE__, __LINE__);
}
void
XIIDCCamera::setShutter(unsigned int shutter) {
    XScopedLock<XDC1394Interface> lock( *interface());
    if(dc1394_feature_set_value(interface()->camera(), DC1394_FEATURE_SHUTTER, shutter))
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
    unsigned int bpp = (image_bytes - padding_bytes) / (width * height);
    tr[ *this].m_status = formatString("%ux%u ", width, height) + s_iidcVideoModes.at(video_mode) + formatString(" stamp:%llu behind:%u", timestamp, frames_behind);

    setGrayImage(reader, tr, width, height, little_endian != DC1394_TRUE, bpp == 2);
}

XTime
XIIDCCamera::acquireRaw(shared_ptr<RawData> &writer) {
    XScopedLock<XDC1394Interface> lock( *interface());
    Snapshot shot( *this);
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
    writer->insert(writer->end(), (char*)frame->image + frame->padding_bytes, (char*)frame->image + frame->image_bytes);

    if(dc1394_capture_enqueue(interface()->camera(), frame))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not release frame."), __FILE__, __LINE__);
    return XTime{(long)(frame->timestamp / 1000000uLL), (long)(frame->timestamp % 1000000uLL)};
}

#endif // USE_LIBDC1394
//camera_status = CAMERA_ON;
//res=DC1394_VIDEO_MODE_FORMAT7_2; // 2x2 binning
//dc1394_video_set_operation_mode(camera, DC1394_OPERATION_MODE_LEGACY);
//dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400);
//dc1394_video_set_mode(camera,res);
//dc1394_format7_set_color_coding(camera, res, DC1394_COLOR_CODING_MONO8);
//dc1394_avt_set_timebase(camera,9); // 9=1ms, 8=500us, 7=200us,
//6=100us, 5=50us, 4=20us
//dc1394_avt_set_extented_shutter(camera, extexp);
//dc1394switch_t pwr = DC1394_ON;
//dc1394_feature_set_value(camera, DC1394_FEATURE_GAIN, gain);
//// The line below needs to be commented if we use an external trigger.
//dc1394_software_trigger_set_power(camera, DC1394_ON);
////dc1394_external_trigger_set_mode(camera, 0);
//62APPENDIX B. SOFTWARE TO PROGRAM AND CONTROL THE CAMERA
////dc1394_external_trigger_set_power(camera, pwr);
//// The two lines above are used when we got an external trigger
//// rigged to the system.
//if (dc1394_capture_setup(camera, 4,DC1394_CAPTURE_FLAGS_DEFAULT)
//!= DC1394_SUCCESS) {
//fprintf(stderr, "unable to setup camera- check line %d of %s to make
//sure\n",__LINE__,__FILE__);
//perror("that the video mode,framerate and format are supported\n");
//printf("is one supported by your camera\n");
//cleanup();
//exit(-1);
//}
////If we use an external trigger we need to comment out the
//// ve lines below marked with: "//comment"
///* main event loop */
//gettimeofday(&start, NULL); //comment
//loop = 0;
//while(loop<imageloop){
//while(t<100000*y){ //comment
//gettimeofday(&end, NULL); //comment
//t = (end.tv_sec*1000000 + end.tv_usec)-(start.tv_sec*1000000
//+ start.tv_usec); //comment
//} //comment
//y=y++;
//// The line below starts the exposure in the case of a
//// software trigger, if else it waits here for the trigger
//// signal in the case of an external trigger.
//shotresult = dc1394_video_set_one_shot(camera, DC1394_ON);
//if (shotresult == DC1394_SUCCESS) {
//if (dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT,
//&Videoframe)!=DC1394_SUCCESS)
//printf("Error: Failed to capture from GUPPY\n");
//if (Videoframe) {
//63
//sprintf(datal, "\n", loop);
//skriv_datal(Videoframe, datal); // here we write to le.
//dc1394_capture_enqueue (camera, Videoframe);
//}
}
