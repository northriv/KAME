/***************************************************************************
        Copyright (C) 2002-2017 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "cyfxusb.h"
#include <libusb-1.0/libusb.h>

struct CyFXLibUSBDevice : public CyFXUSBDevice {
    CyFXLibUSBDevice(libusb_device_handle *h, const XString &n) : handle(h) {}

    virtual int initialize() final;
    virtual void finalize() final;

    XString virtual getString(int descid) final;

    virtual AsyncIO asyncBulkWrite(int pipe, const uint8_t *buf, int len) final;
    virtual AsyncIO asyncBulkRead(int pipe, uint8_t *buf, int len) final;

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) final;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) final;

private:
    libusb_device_handle *handle;
    struct USBList {
        USBList() noexcept {
            size = libusb_get_device_list(NULL, &list);
            if(size < 0 ) {
                fprintf(stderr, "Error during dev. enum. of libusb: %s\n", libusb_error_name(size));
            }
        }
        ~USBList() {
            if(size >= 0)
                libusb_free_device_list(list, 1);
        }
        libusb_device *operator[](ssize_t i) const noexcept {
            if((i >= size) || (i < 0))
                return nullptr;
            return list[i];
        }
        libusb_device **list;
        int size;
    };
};

struct CyFXUSBDevice::AsyncIO::Transfer {
};

CyFXUSBDevice::AsyncIO::AsyncIO(unique_ptr<Transfer>&& t) :
    m_transfer(std::forward<unique_ptr<Transfer>>(t)) {
}
bool
CyFXUSBDevice::AsyncIO::hasFinished() const {
}
int64_t
CyFXUSBDevice::AsyncIO::waitFor() {
}


CyFXUSBDevice::List
CyFXUSBDevice::enumerateDevices() {
    CyFXUSBDevice::List list;
    for(int idx = 0; idx < 9; ++idx) {

        list.push_back(std::make_shared<CyFXEasyUSBDevice>(h, name));
    }

    return std::move(list);
}
void
CyFXLibUSBDevice::open() {
    if( !handle) {
        libusb_device_descriptor desc;
        int ret = libusb_get_device_descriptor(pdev, &desc);
        if(ret) {
            fprintf(stderr, "Error obtaining dev. desc. in libusb: %s\n", libusb_error_name(ret));
            return -1;
        }

        int bus_num = libusb_get_bus_number(pdev);
        int addr = libusb_get_device_address(pdev);
    //    fprintf(stderr, "USB %d: PID=%d,VID=%d,BUS#%d,ADDR=%d.\n",
    //        n, desc.idProduct, desc.idVendor, bus_num, addr);

        if(((desc.idProduct != FX2_DEF_PID) || (desc.idVendor != FX2_DEF_VID))
            && ((desc.idProduct != THAMWAY_PID) || (desc.idVendor != THAMWAY_VID))) {
            return -1;
        }
        ret = libusb_open(pdev, h);
        if(ret) {
            fprintf(stderr, "Error opening dev. in libusb: %s\n", libusb_error_name(ret));
           return -1;
        }

        unsigned char manu[256] = {}, prod[256] = {}, serial[256] = {};
        libusb_get_string_descriptor_ascii( *h, desc.iManufacturer, manu, 255);
        libusb_get_string_descriptor_ascii( *h, desc.iProduct, prod, 255);
        libusb_get_string_descriptor_ascii( *h, desc.iSerialNumber, serial, 255);
        fprintf(stderr, "USB: VID=0x%x, PID=0x%x,BUS#%d,ADDR=%d;%s;%s;%s.\n",
            desc.idVendor, desc.idProduct, bus_num, addr, manu, prod, serial);

    //    ret = libusb_set_auto_detach_kernel_driver( *h, 1);
    //    if(ret) {
    //        fprintf(stderr, "USB %d: Warning auto detach is not supported: %s\n", n, libusb_error_name(ret));
    //    }
        ret = libusb_kernel_driver_active( *h, 0);
        if(ret < 0) {
            libusb_close( *h);
            fprintf(stderr, "USB: Error on libusb: %s\n", libusb_error_name(ret));
            return -1;
        }
        if(ret == 1) {
            fprintf(stderr, "USB: kernel driver is active, detaching...\n");
            ret = libusb_detach_kernel_driver( *h, 0);
            if(ret < 0) {
                libusb_close( *h);
                fprintf(stderr, "USB: Error on libusb: %s\n", libusb_error_name(ret));
                return -1;
            }
        }
    //    ret = libusb_set_configuration( *h, 1);
        ret = libusb_claim_interface( *h, 0);
        if(ret) {
            libusb_close( *h);
            fprintf(stderr, "USB: Error claiming interface: %s\n", libusb_error_name(ret));
            return -1;
        }
        ret = libusb_set_interface_alt_setting( *h, 0 , 0 );
        if(ret) {
            libusb_release_interface( *h,0);
            libusb_close( *h);
            fprintf(stderr, "USB: Error ALT setting for interface: %s\n", libusb_error_name(ret));
            return -1;
        }
        return 0;
    }
    //obtains device descriptor
    struct DevDesc {
        uint8_t bLength;
        uint8_t bDescriptorType;
        uint16_t bcdUSB;
        uint8_t bDeviceClass;
        uint8_t bDeviceSubClass;
        uint8_t bDeviceProtocol;
        uint8_t bMaxPacketSize0;
        uint16_t idVendor;
        uint16_t idProduct;
        uint16_t bcdDevice;
        uint8_t iManufacturer;
        uint8_t iProduct;
        uint8_t iSerialNumber;
        uint8_t bNumConfigurations;
    } dev_desc;
    auto buf = reinterpret_cast<uint8_t*>( &dev_desc);
    //Reads common descriptor.
    controlRead(CtrlReq::GET_DESC,
        CtrlReqType::STD, USB_DEVICE_DESCRIPTOR_TYPE * 0x100u,
        0, buf, sizeof(DevDesc));

    m_productID = dev_desc.idProduct;
    m_vendorID = dev_desc.idVendor;
}

void
CyFXUSBDevice::close() {
    //hacks for stability.
    libusb_clear_halt( *h, 0x2);
    libusb_clear_halt( *h, 0x6);
    libusb_clear_halt( *h, 0x8);
    libusb_reset_device( *h);
    libusb_release_interface( *h,0);
    libusb_close( *h);
    handle = nullptr;
}

int
CyFXLibUSBDevice::initialize() {
    int ret = libusb_init(NULL);
    if(ret) {
        fprintf(stderr, "Error during initialization of libusb: %s\n", libusb_error_name(ret));
        return -1;
    }
    return ret;
}

void
CyFXLibUSBDevice::finalize() {
    libusb_exit(NULL);
}

int
CyFXLibUSBDevice::controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, const uint8_t *wbuf, int len) {
    int ret = libusb_control_transfer(handle,
        LIBUSB_ENDPOINT_OUT | type,
        request,
        value, index, wbuf, len, USB_TIMEOUT);
    if(ret < 0) {
        throw XInterfaceError(formatString("USB: %s.", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    return ret;
}

int
CyFXLibUSBDevice::controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, uint8_t *rdbuf, int len) {
    int ret = libusb_control_transfer(handle,
        LIBUSB_ENDPOINT_IN | type,
        request,
        value, index, rdbuf, len, USB_TIMEOUT);
    if(ret < 0) {
        throw XInterfaceError(formatString("USB: %s.", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    return ret;
}


XString
CyFXLibUSBDevice::getString(int descid) {
    unsigned char s[128];
    int ret = libusb_get_string_descriptor_ascii(handle, descid, s, 127);
    if(ret < 0) {
         throw XInterfaceError(formatString("Error during USB get string desc.: %s\n", libusb_error_name(ret)));
    }
    s[ret] = '\0';
    return s;
}

CyFXUSBDevice::AsyncIO
CyFXLibUSBDevice::asyncBulkWrite(int pipe, const uint8_t *buf, int len) {
    int ep;
    switch(pipe) {
    case TFIFO:
        ep = 0x2;
        break;
    case CPIPE:
        ep = 0x8;
        break;
    default:
        return -1;
    }

    int cnt = 0;
    for(int i = 0; len > 0;){
        int transferred;
        int ret = libusb_bulk_transfer( *h, LIBUSB_ENDPOINT_OUT | ep, buf, len, &transferred, USB_TIMEOUT);
        if(ret) {
            fprintf(stderr, "Error during USB Bulk writing: %s\n", libusb_error_name(ret));
            return -1;
        }
        buf += transferred;
        len -= transferred;
        cnt += transferred;
    }
    return 0;
}

CyFXUSBDevice::AsyncIO
CyFXLibUSBDevice::asyncBulkRead(int pipe, uint8_t* buf, int len) {
    int ep;
    switch(pipe) {
    case RFIFO:
        ep = 0x6;
        break;
    default:
        return -1;
    }
    int cnt = 0;
    for(int i = 0; len > 0;){
        int l = std::min(len, 0x8000);
        int transferred;
        int ret = libusb_bulk_transfer( *h, LIBUSB_ENDPOINT_IN | ep, buf, l, &transferred, USB_TIMEOUT);
        if(ret) {
            fprintf(stderr, "Error during USB Bulk reading: %s\n", libusb_error_name(ret));
            return -1;
        }
        buf += transferred;
        len -= transferred;
        cnt += transferred;
    }
    return cnt;
}


