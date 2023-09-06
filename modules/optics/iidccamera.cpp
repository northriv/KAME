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
//            DC1394_COLOR_CODING_MONO8= 352,
//            DC1394_COLOR_CODING_YUV411,
//            DC1394_COLOR_CODING_YUV422,
//            DC1394_COLOR_CODING_YUV444,
//            DC1394_COLOR_CODING_RGB8,
//            DC1394_COLOR_CODING_MONO16,
//            DC1394_COLOR_CODING_RGB16,
//            DC1394_COLOR_CODING_MONO16S,
//            DC1394_COLOR_CODING_RGB16S,
//            DC1394_COLOR_CODING_RAW8,
//            DC1394_COLOR_CODING_RAW16
            if(coding == DC1394_COLOR_CODING_MONO16) {
                dc1394video_mode_t video_mode=video_modes.modes[i];
                modestrings.push_back(s_iidcVideoModes.at(video_mode));
            }
        }
    }

    if(dc1394_video_set_iso_speed(interface()->camera(), DC1394_ISO_SPEED_400))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set 400 Msps."), __FILE__, __LINE__);

    iterate_commit([=](Transaction &tr){
        tr[ *videoMode()].clear();
        for(auto &s: modestrings)
            tr[ *videoMode()].add(s);
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
    if(video_mode ==  dc1394video_mode_t{})
        return;
    if(dc1394_video_set_mode(interface()->camera(), video_mode))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set video modes."), __FILE__, __LINE__);
    // get highest framerate
    dc1394framerates_t framerates;
    if(dc1394_video_get_supported_framerates(interface()->camera(),video_mode,&framerates))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get framerates."), __FILE__, __LINE__);
    dc1394framerate_t framerate=framerates.framerates[framerates.num - 1];
    if(dc1394_video_set_framerate(interface()->camera(), framerate))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not set framerate."), __FILE__, __LINE__);

    if(dc1394_capture_setup(interface()->camera(), 4, DC1394_CAPTURE_FLAGS_DEFAULT))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not setup capture."), __FILE__, __LINE__);

    dc1394featureset_t features;
    if(dc1394_feature_get_all(interface()->camera(), &features) == DC1394_SUCCESS)
        dc1394_feature_print_all(&features, stdout);
}
void
XIIDCCamera::setBrightness(unsigned int brightness) {
    XScopedLock<XDC1394Interface> lock( *interface());

//    unsigned int avg = shot[ *average()];
}
void
XIIDCCamera::setShutter(unsigned int shutter) {
    XScopedLock<XDC1394Interface> lock( *interface());
}

void
XIIDCCamera::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    uint32_t width = reader->pop<uint32_t>();
    tr[ *this].m_width = width;
    uint32_t height = reader->pop<uint32_t>();
    tr[ *this].m_height = height;
    tr[ *this].m_xpos = reader->pop<uint32_t>();
    tr[ *this].m_ypos = reader->pop<uint32_t>();
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
    tr[ *this].m_status = formatString("%ux%u", width, height)
    writer->insert(writer->end(), (char*)frame->image + frame->padding_bytes, (char*)frame + frame->image_bytes);
}

void
XIIDCCamera::acquireRaw(shared_ptr<RawData> &writer) {
    unsigned int width, height;

    Snapshot shot( *this);
    dc1394video_mode_t video_mode = {};
    for(auto &s: s_iidcVideoModes) {
        if(s.second == shot[ *videoMode()].to_str()) {
            video_mode = s.first;
        }
    }
    if(video_mode ==  dc1394video_mode_t{})
        throw XPrimaryDriver::XSkippedRecordError(__FILE__, __LINE__);

//    if(dc1394_get_color_coding_from_video_mode(interface()->camera(), video_mode, &coding))
//        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get coding."), __FILE__, __LINE__);


    /*-----------------------------------------------------------------------
     *  have the interface()->camera() start sending us data
     *-----------------------------------------------------------------------*/
    if(dc1394_video_set_transmission(interface()->camera(), DC1394_ON))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not start ISO transmission."), __FILE__, __LINE__);

    /*-----------------------------------------------------------------------
     *  capture one frame
     *-----------------------------------------------------------------------*/
    dc1394video_frame_t *frame;
    if(dc1394_capture_dequeue(interface()->camera(), DC1394_CAPTURE_POLICY_WAIT, &frame))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not capture."), __FILE__, __LINE__);

    /*-----------------------------------------------------------------------
     *  stop data transmission
     *-----------------------------------------------------------------------*/
    if(dc1394_video_set_transmission(interface()->camera(),DC1394_OFF))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not stop."), __FILE__, __LINE__);

    if(dc1394_get_image_size_from_video_mode(interface()->camera(), video_mode, &width, &height))
        throw XInterface::XInterfaceError(getLabel() + " " + i18n("Could not get geometry."), __FILE__, __LINE__);

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
    writer->insert(writer->end(), (char*)frame->image + frame->padding_bytes, (char*)frame + frame->image_bytes);
}
//unsigned char          * image;                 /* the image. May contain padding data too (vendor specific). Read/write allowed. Free NOT allowed if
//                           returned by dc1394_capture_dequeue() */
//uint32_t                 size[2];               /* the image size [width, height] */
//uint32_t                 position[2];           /* the WOI/ROI position [horizontal, vertical] == [0,0] for full frame */
//dc1394color_coding_t     color_coding;          /* the color coding used. This field is valid for all video modes. */
//dc1394color_filter_t     color_filter;          /* the color filter used. This field is valid only for RAW modes and IIDC 1.31 */
//uint32_t                 yuv_byte_order;        /* the order of the fields for 422 formats: YUYV or UYVY */
//uint32_t                 data_depth;            /* the number of bits per pixel. The number of grayscale levels is 2^(this_number).
//                                                   This is independent from the colour coding */
//uint32_t                 stride;                /* the number of bytes per image line */
//dc1394video_mode_t       video_mode;            /* the video mode used for capturing this frame */
//uint64_t                 total_bytes;           /* the total size of the frame buffer in bytes. May include packet-
//                                                   multiple padding and intentional padding (vendor specific) */
//uint32_t                 image_bytes;           /* the number of bytes used for the image (image data only, no padding) */
//uint32_t                 padding_bytes;         /* the number of extra bytes, i.e. total_bytes-image_bytes.  */
//uint32_t                 packet_size;           /* the size of a packet in bytes. (IIDC data) */
//uint32_t                 packets_per_frame;     /* the number of packets per frame. (IIDC data) */
//uint64_t                 timestamp;             /* the unix time [microseconds] at which the frame was captured in
//                                                   the video1394 ringbuffer */
//uint32_t                 frames_behind;         /* the number of frames in the ring buffer that are yet to be accessed by the user */
//dc1394camera_t           *camera;               /* the parent camera of this frame */
//uint32_t                 id;                    /* the frame position in the ring buffer */
//uint64_t                 allocated_image_bytes; /* amount of memory allocated in for the *image field. */
//dc1394bool_t             little_endian;         /* DC1394_TRUE if little endian (16bpp modes only),
//                                                   DC1394_FALSE otherwise */
//dc1394bool_t             data_in_padding;       /* DC1394_TRUE if data is present in the padding bytes in IIDC 1.32 format,
//                                                   DC1394_FALSE otherwise */
#endif // USE_LIBDC1394
