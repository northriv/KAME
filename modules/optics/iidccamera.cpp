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

REGISTER_TYPE(XDriverList, IIDCCamera, "IEEE1394 IIDC camera");
#include "dc1394/dc1394.h"

//---------------------------------------------------------------------------
XDC1394Interface::XDC1394Interface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
    XInterface(name, runtime, driver) {
    dc1394featureset_t features;
    dc1394framerates_t framerates;
    dc1394video_modes_t video_modes;
    dc1394color_coding_t coding;
    unsigned int width, height;
    dc1394video_frame_t *frame;

    dc1394_t *dc1394 = dc1394_new();
    if(dc1394) {
        dc1394camera_list_t *list;
        dc1394error_t err = dc1394_camera_enumerate(dc1394, &list);
        if( !err) {
            for(unsigned int i = 0; i < dc1394->num_cameras; ++i) {
                list[i].model;

            }
        }
        dc1394_free(dc1394);
    }

    camera = dc1394_camera_new (dc1394, list->ids[0].guid);
    if (!camera) {
        dc1394_log_error("Failed to initialize camera with guid %"PRIx64, list->ids[0].guid);
        return 1;
    }
    dc1394_camera_free_list (list);

    printf("Using camera with GUID %"PRIx64"\n", camera->guid);

}
void
XDC1394Interface::open() {

}
void
XDC1394Interface::close() {

}


XIIDCCamera::XIIDCCamera(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDC1394Driver<XDigitalCamera>(name, runtime, ref(tr_meas), meas) {
//    startWavelen()->disable();
//    stopWavelen()->disable();
}

void
XIIDCCamera::open() {

    start();
}
void
XIIDCCamera::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
    unsigned int avg = shot[ *average()];
}
void
XIIDCCamera::onExposureChanged(const Snapshot &shot, XValueNodeBase *) {
//    XScopedLock<XDC1394Interface> lock( *interface());
    try {
//        interface()->setIntegrationTime(lrint(shot[ *integrationTime()] * 1e6));
    }
    catch (XKameError &e) {
        e.print(getLabel() + " " + i18n(" Error"));
    }
}
unique_ptr<QImage>
XIIDCCamera::acquireRaw(shared_ptr<RawData> &) {
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
