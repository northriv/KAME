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
unique_ptr<QImage>
XIIDCCamera::acquireRaw(shared_ptr<RawData> &) {
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

    /*-----------------------------------------------------------------------
     *  save image as 'Image.pgm'
     *-----------------------------------------------------------------------*/
//    imagefile=fopen(IMAGE_FILE_NAME, "wb");

//    if( imagefile == NULL) {
//        perror( "Can't create '" IMAGE_FILE_NAME "'");
//        cleanup_and_exit(interface()->camera());
//    }

//    fprintf(imagefile,"P5\n%u %u 255\n", width, height);
//    fwrite(frame->image, 1, height*width, imagefile);
//    fclose(imagefile);
//    printf("wrote: " IMAGE_FILE_NAME "\n");



    QImage image(300, 300, QImage::Format_RGB32);
    QRgb value;

    value = qRgb(189, 149, 39); // 0xffbd9527
    image.setPixel(1, 1, value);

    value = qRgb(122, 163, 39); // 0xff7aa327
    image.setPixel(0, 1, value);
    image.setPixel(1, 0, value);

    value = qRgb(237, 187, 51); // 0xffedba31
    image.setPixel(2, 1, value);
    return std::make_unique<QImage>(std::move(image));
}

#endif // USE_LIBDC1394
