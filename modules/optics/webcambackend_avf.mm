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
// WebCam backend over AVFoundation (macOS).
//
// Manual retain/release, not ARC: qmake compiles .mm with the target's
// CXXFLAGS, so -fobjc-arc could only be had by imposing it on every C++ file
// of this module.  support_osx.mm is MRC for the same reason.
//
// Threading: AVCaptureSession calls go through m_sessionQueue (startRunning
// blocks, and Apple asks for it off the main thread); frames arrive on
// m_frameQueue and are handed to the driver through the FrameSlot.
#include "webcambackend.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

#if __has_feature(objc_arc)
#error "webcambackend_avf.mm is written for manual retain/release."
#endif

namespace {

std::string toStd(NSString *s) {
    const char *p = s ? [s UTF8String] : nullptr;
    return p ? std::string(p) : std::string();
}
std::string describe(NSError *e) {
    return e ? toStd([e localizedDescription]) : std::string("unknown error");
}
std::string describe(NSException *e) {
    return toStd([e name]) + ": " + toStd([e reason]);
}
std::string fourCC(FourCharCode c) {
    char s[5] = {(char)(c >> 24), (char)(c >> 16), (char)(c >> 8), (char)c, 0};
    for(int i = 0; i < 4; ++i)
        if( !isprint((unsigned char)s[i])) {
            char hex[16];
            snprintf(hex, sizeof(hex), "0x%08x", (unsigned int)c);
            return hex;
        }
    return s;
}
int64_t unixMicrosecondsNow() {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

//! Fills frame.luma from an 8-bit YCbCr or BGRA buffer. Only formats
//! pickPixelFormat() can choose need handling here.
bool extractLuma(CVPixelBufferRef pb, WebCam::Frame &frame) {
    if(CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess)
        return false;
    bool ok = true;
    const OSType fmt = CVPixelBufferGetPixelFormatType(pb);
    const size_t w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb);
    frame.width = (unsigned int)w;
    frame.height = (unsigned int)h;
    frame.luma.resize(w * h);
    uint8_t *dst = frame.luma.data();
    if(CVPixelBufferIsPlanar(pb)) {
        //420f / 420v: luma is plane 0, one byte per pixel.
        auto src = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddressOfPlane(pb, 0));
        const size_t bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
        if( !src || (CVPixelBufferGetWidthOfPlane(pb, 0) != w) || (CVPixelBufferGetHeightOfPlane(pb, 0) != h))
            ok = false;
        else
            for(size_t y = 0; y < h; ++y)
                memcpy(dst + y * w, src + y * bpr, w);
    }
    else {
        auto src = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(pb));
        const size_t bpr = CVPixelBufferGetBytesPerRow(pb);
        if( !src)
            ok = false;
        else switch(fmt) {
        case kCVPixelFormatType_422YpCbCr8: //'2vuy' = UYVY: luma at odd bytes
        case kCVPixelFormatType_422YpCbCr8_yuvs: { //'yuvs' = YUYV: luma at even bytes
            const size_t off = (fmt == kCVPixelFormatType_422YpCbCr8) ? 1 : 0;
            for(size_t y = 0; y < h; ++y) {
                const uint8_t *s = src + y * bpr + off;
                for(size_t x = 0; x < w; ++x, s += 2)
                    *dst++ = *s;
            }
            break;
        }
        case kCVPixelFormatType_32BGRA:
            for(size_t y = 0; y < h; ++y) {
                const uint8_t *s = src + y * bpr;
                for(size_t x = 0; x < w; ++x, s += 4)
                    *dst++ = (uint8_t)((29u * s[0] + 150u * s[1] + 77u * s[2]) >> 8); //BT.601 weights
            }
            break;
        default:
            ok = false;
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    return ok;
}

//! Asks AVFoundation for the first of these it can deliver: the bi-planar
//! 4:2:0 formats carry luma as a separate plane, so a frame costs one memcpy
//! per row; the rest need a pass over every pixel.
OSType pickPixelFormat(AVCaptureVideoDataOutput *output) {
    const OSType preferred[] = {
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        kCVPixelFormatType_422YpCbCr8,
        kCVPixelFormatType_422YpCbCr8_yuvs,
        kCVPixelFormatType_32BGRA,
    };
    NSArray<NSNumber *> *avail = [output availableVideoCVPixelFormatTypes];
    for(OSType f: preferred)
        if([avail containsObject:@(f)])
            return f;
    return kCVPixelFormatType_32BGRA; //every output can convert to it.
}

void authorize() {
    if( ![[NSBundle mainBundle] objectForInfoDictionaryKey:@"NSCameraUsageDescription"])
        throw WebCam::Error("This build's Info.plist has no NSCameraUsageDescription, and macOS "
            "terminates a process that touches the camera without one. Rebuild kame.app with "
            "the Info.plist from the source tree.");
    const char *denied = "Camera access is denied. Allow KAME in System Settings > Privacy & Security > Camera. "
        "(When KAME is launched from another app, such as Qt Creator or Terminal, macOS may ask "
        "about that app instead.)";
    switch([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo]) {
    case AVAuthorizationStatusAuthorized:
        return;
    case AVAuthorizationStatusNotDetermined: {
        if([NSThread isMainThread])
            throw WebCam::Error("Camera permission must be requested off the main thread.");
        //The handler block retains the semaphore, so releasing ours after a
        //timeout cannot leave it signalling a dead object.
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block BOOL granted = NO;
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL g) {
            granted = g;
            dispatch_semaphore_signal(sem);
        }];
        long timedout = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 120 * (int64_t)NSEC_PER_SEC));
        dispatch_release(sem);
        if(timedout)
            throw WebCam::Error("The camera permission dialog was not answered.");
        if( !granted)
            throw WebCam::Error(denied);
        return;
    }
    default:
        throw WebCam::Error(denied);
    }
}

} //namespace

@interface KAMEWebCamDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
- (instancetype)initWithSlot:(std::shared_ptr<WebCam::FrameSlot>)slot;
@end

@implementation KAMEWebCamDelegate {
    std::shared_ptr<WebCam::FrameSlot> _slot;
    WebCam::Frame _frame; //touched only on the (serial) frame queue; its buffer is recycled by FrameSlot::post().
}
- (instancetype)initWithSlot:(std::shared_ptr<WebCam::FrameSlot>)slot {
    if((self = [super init]))
        _slot = slot;
    return self;
}
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    CVImageBufferRef pb = CMSampleBufferGetImageBuffer(sampleBuffer);
    if( !pb)
        return;
    //The PTS is on the host clock; date the frame by its age rather than by
    //when this callback happened to run.  Anything implausible (another
    //clock, NaN) falls back to now.
    const CMTime pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
    double age = 0.0;
    if(CMTIME_IS_NUMERIC(pts))
        age = CMTimeGetSeconds(CMTimeSubtract(CMClockGetTime(CMClockGetHostTimeClock()), pts));
    if( !(age >= 0.0 && age < 10.0))
        age = 0.0;
    _frame.timestamp_us = unixMicrosecondsNow() - llrint(age * 1e6);
    if(extractLuma(pb, _frame))
        _slot->post(_frame);
}
- (void)captureOutput:(AVCaptureOutput *)output didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    _slot->addDropped(1); //late frames AVFoundation discarded before we saw them.
}
@end

namespace {

class AVFCamera : public WebCam::Camera {
public:
    explicit AVFCamera(AVCaptureDevice *device);
    ~AVFCamera() override {release();}

    std::vector<WebCam::Format> formats() const override {return m_formats;}
    int activeFormat() const override;
    void setFormat(int index, double fps) override;
    bool setExposureTime(double sec) override;
    bool setGain(double gain) override {return gain <= 0;} //macOS has no ISO control (iOS-only API).
private:
    void release();
    //! Runs \a f on the session queue, turning an NSException into WebCam::Error.
    void onSessionQueue(void (^f)(void)) const;

    AVCaptureDevice *m_device = nil;
    NSArray<AVCaptureDeviceFormat *> *m_avFormats = nil;
    AVCaptureDeviceInput *m_input = nil;
    AVCaptureVideoDataOutput *m_output = nil;
    AVCaptureSession *m_session = nil;
    KAMEWebCamDelegate *m_delegate = nil;
    dispatch_queue_t m_sessionQueue = nullptr, m_frameQueue = nullptr;
    std::vector<WebCam::Format> m_formats;
};

AVFCamera::AVFCamera(AVCaptureDevice *device) {
    @autoreleasepool {
        m_device = [device retain];
        m_sessionQueue = dispatch_queue_create("kame.webcam.session", DISPATCH_QUEUE_SERIAL);
        m_frameQueue = dispatch_queue_create("kame.webcam.frames", DISPATCH_QUEUE_SERIAL);
        std::string err;
        @try {
            m_avFormats = [[device formats] copy];
            for(AVCaptureDeviceFormat *f in m_avFormats) {
                WebCam::Format fmt;
                CMVideoDimensions dim = CMVideoFormatDescriptionGetDimensions([f formatDescription]);
                fmt.width = dim.width;
                fmt.height = dim.height;
                fmt.pixelFormat = fourCC(CMFormatDescriptionGetMediaSubType([f formatDescription]));
                for(AVFrameRateRange *r in [f videoSupportedFrameRateRanges]) {
                    if((fmt.minFPS == 0) || ([r minFrameRate] < fmt.minFPS))
                        fmt.minFPS = [r minFrameRate];
                    fmt.maxFPS = std::max(fmt.maxFPS, [r maxFrameRate]);
                }
                m_formats.push_back(fmt);
            }
            NSError *e = nil;
            m_input = [[AVCaptureDeviceInput alloc] initWithDevice:device error:&e];
            if( !m_input)
                err = describe(e);
            else {
                m_output = [[AVCaptureVideoDataOutput alloc] init];
                [m_output setVideoSettings:@{(id)kCVPixelBufferPixelFormatTypeKey: @(pickPixelFormat(m_output))}];
                [m_output setAlwaysDiscardsLateVideoFrames:YES];
                m_delegate = [[KAMEWebCamDelegate alloc] initWithSlot:m_slot];
                [m_output setSampleBufferDelegate:m_delegate queue:m_frameQueue];
                m_session = [[AVCaptureSession alloc] init];
                [m_session beginConfiguration];
                if( ![m_session canAddInput:m_input])
                    err = "The camera cannot be added to a capture session (in use by another app?).";
                else {
                    [m_session addInput:m_input];
                    if( ![m_session canAddOutput:m_output])
                        err = "The capture session refused a video data output.";
                    else
                        [m_session addOutput:m_output];
                }
                [m_session commitConfiguration];
            }
        }
        @catch(NSException *e) {
            err = describe(e);
        }
        if( !err.empty()) {
            release(); //the destructor does not run for a throwing constructor.
            throw WebCam::Error(err);
        }
    }
}

void
AVFCamera::release() {
    @autoreleasepool {
        if(m_session && m_sessionQueue) {
            AVCaptureSession *session = m_session;
            dispatch_sync(m_sessionQueue, ^{
                @try {
                    if([session isRunning])
                        [session stopRunning];
                }
                @catch(NSException *) {}
            });
        }
        if(m_output)
            [m_output setSampleBufferDelegate:nil queue:nullptr];
        if(m_frameQueue)
            dispatch_sync(m_frameQueue, ^{}); //drains a callback already in flight before its delegate goes.
        [m_session release]; m_session = nil;
        [m_output release]; m_output = nil;
        [m_input release]; m_input = nil;
        [m_delegate release]; m_delegate = nil;
        [m_avFormats release]; m_avFormats = nil;
        [m_device release]; m_device = nil;
        if(m_frameQueue) {dispatch_release(m_frameQueue); m_frameQueue = nullptr;}
        if(m_sessionQueue) {dispatch_release(m_sessionQueue); m_sessionQueue = nullptr;}
    }
}

void
AVFCamera::onSessionQueue(void (^f)(void)) const {
    __block std::string err;
    @autoreleasepool {
        dispatch_sync(m_sessionQueue, ^{
            @try {
                f();
            }
            @catch(NSException *e) {
                err = describe(e);
            }
        });
    }
    if( !err.empty())
        throw WebCam::Error(err);
}

int
AVFCamera::activeFormat() const {
    __block NSUInteger idx = NSNotFound;
    AVCaptureDevice *device = m_device;
    NSArray *formats = m_avFormats;
    onSessionQueue(^{
        idx = [formats indexOfObject:[device activeFormat]];
    });
    return (idx == NSNotFound) ? -1 : (int)idx;
}

void
AVFCamera::setFormat(int index, double fps) {
    if((index < 0) || (index >= (int)m_formats.size()))
        throw WebCam::Error("No such video format.");
    __block std::string err;
    AVCaptureDevice *device = m_device;
    AVCaptureSession *session = m_session;
    AVCaptureDeviceFormat *format = [m_avFormats objectAtIndex:index];
    onSessionQueue(^{
        //activeFormat must be set with the input already in the session,
        //which then switches itself to AVCaptureSessionPresetInputPriority;
        //set before, the session's preset overrides it at startRunning.
        [session beginConfiguration];
        NSError *e = nil;
        if([device lockForConfiguration:&e]) {
            [device setActiveFormat:format];
            if(fps > 0) {
                for(AVFrameRateRange *r in [format videoSupportedFrameRateRanges]) {
                    if((fps < [r minFrameRate] - 1e-3) || (fps > [r maxFrameRate] + 1e-3))
                        continue;
                    //Caps the rate only; a camera may still slow down in low light.
                    [device setActiveVideoMinFrameDuration:(fps >= [r maxFrameRate] - 1e-3) ?
                        [r minFrameDuration] : CMTimeMakeWithSeconds(1.0 / fps, 1000000)];
                    break;
                }
            }
            [device unlockForConfiguration];
        }
        else
            err = describe(e);
        [session commitConfiguration];
        if(err.empty() && ![session isRunning])
            [session startRunning];
    });
    if( !err.empty())
        throw WebCam::Error(err);
}

bool
AVFCamera::setExposureTime(double sec) {
    //setExposureModeCustomWithDuration:ISO: is iOS-only; on macOS all there
    //is to choose is automatic exposure.
    if(sec > 0)
        return false;
    __block std::string err;
    AVCaptureDevice *device = m_device;
    onSessionQueue(^{
        if( ![device isExposureModeSupported:AVCaptureExposureModeContinuousAutoExposure])
            return;
        NSError *e = nil;
        if([device lockForConfiguration:&e]) {
            [device setExposureMode:AVCaptureExposureModeContinuousAutoExposure];
            [device unlockForConfiguration];
        }
        else
            err = describe(e);
    });
    if( !err.empty())
        throw WebCam::Error(err);
    return true;
}

} //namespace

namespace WebCam {

const BackendInfo &
backendInfo() {
    static const BackendInfo info = {"AVFoundation", false};
    return info;
}

std::vector<DeviceInfo>
enumerateDevices() {
    std::vector<DeviceInfo> list;
    @autoreleasepool {
        @try {
            NSMutableArray<AVCaptureDeviceType> *types =
                [NSMutableArray arrayWithObject:AVCaptureDeviceTypeBuiltInWideAngleCamera];
            if(@available(macOS 14.0, *)) {
                [types addObject:AVCaptureDeviceTypeExternal];
            }
            else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                [types addObject:AVCaptureDeviceTypeExternalUnknown];
#pragma clang diagnostic pop
            }
            AVCaptureDeviceDiscoverySession *ds = [AVCaptureDeviceDiscoverySession
                discoverySessionWithDeviceTypes:types mediaType:AVMediaTypeVideo
                position:AVCaptureDevicePositionUnspecified];
            for(AVCaptureDevice *d in [ds devices])
                list.push_back({toStd([d uniqueID]), toStd([d localizedName])});
        }
        @catch(NSException *e) {
            fprintf(stderr, "WebCam: device enumeration failed, %s\n", describe(e).c_str());
        }
    }
    return list;
}

std::unique_ptr<Camera>
openCamera(const std::string &id) {
    @autoreleasepool {
        authorize();
        AVCaptureDevice *device = [AVCaptureDevice deviceWithUniqueID:[NSString stringWithUTF8String:id.c_str()]];
        if( !device)
            throw Error("The camera is not connected.");
        return std::unique_ptr<Camera>(new AVFCamera(device));
    }
}

} //namespace WebCam
