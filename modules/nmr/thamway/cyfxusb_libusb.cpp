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
#include "interface.h"
#include <libusb-1.0/libusb.h>

static constexpr int USB_TIMEOUT = 1000; //ms

struct CyFXLibUSBDevice : public CyFXUSBDevice {
    CyFXLibUSBDevice(libusb_device *d) : handle(nullptr), dev(d) {
        libusb_device_descriptor desc;
        int ret = libusb_get_device_descriptor(dev, &desc);
        if(ret) {
            throw XInterface::XInterfaceError(formatString("Error obtaining dev. desc. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
        m_productID = desc.idProduct;
        m_vendorID = desc.idVendor;

        libusb_ref_device(dev);
    }
    ~CyFXLibUSBDevice() {
        libusb_unref_device(dev);
    }

    virtual void finalize() final;

    virtual void open() final;
    virtual void close() final;

    XString virtual getString(int descid) final;

    virtual AsyncIO asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len) final;
    virtual AsyncIO asyncBulkRead(uint8_t ep, uint8_t *buf, int len) final;

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) final;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) final;

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
    static libusb_context *context;
private:
    friend struct AsyncIO;
    libusb_device_handle *handle;
    libusb_device *dev;
};

libusb_context *CyFXLibUSBDevice::context = nullptr;

class CyFXUSBDevice::AsyncIO::Transfer {
public:
    Transfer() {
        transfer = libusb_alloc_transfer(0);
    }
    ~Transfer() {
        libusb_free_transfer(transfer);
    }
    static void cb_fn(struct libusb_transfer *transfer) {
//        switch(transfer->status) {
//        case LIBUSB_TRANSFER_COMPLETED:
//            break;
//        case LIBUSB_TRANSFER_CANCELLED:
//        case LIBUSB_TRANSFER_NO_DEVICE:
//        case LIBUSB_TRANSFER_TIMED_OUT:
//        case LIBUSB_TRANSFER_ERROR:
//        case LIBUSB_TRANSFER_STALL:
//        case LIBUSB_TRANSFER_OVERFLOW:
//        default:
//            break;
//        }
        reinterpret_cast<Transfer*>(transfer->user_data)->completed = 1;
    }
    unique_ptr<std::vector<uint8_t>> buf;
    libusb_transfer *transfer;
    uint8_t *rdbuf = nullptr;
    int completed = 0;
};

CyFXUSBDevice::AsyncIO::AsyncIO() :
    m_transfer(new Transfer) {
}
bool
CyFXUSBDevice::AsyncIO::hasFinished() const {
    return ptr()->completed;
}
int64_t
CyFXUSBDevice::AsyncIO::waitFor() {
    while( !ptr()->completed) {
        int ret = libusb_handle_events_completed(CyFXLibUSBDevice::context, &ptr()->completed);
        if(ret)
            throw XInterface::XInterfaceError(formatString("Error during transfer in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
    }
    if(ptr()->transfer->status != LIBUSB_TRANSFER_COMPLETED)
        throw XInterface::XInterfaceError(formatString("Error during transfer in libusb: %s\n", libusb_error_name(ptr()->transfer->status)).c_str(), __FILE__, __LINE__);
    if(ptr()->rdbuf) {
        std::copy(ptr()->buf->begin(), ptr()->buf->begin() + ptr()->transfer->actual_length, ptr()->rdbuf);
    }
    return ptr()->transfer->actual_length;
}


CyFXUSBDevice::List
CyFXUSBDevice::enumerateDevices(bool initialization) {
    if(initialization) {
        int ret = libusb_init( &CyFXLibUSBDevice::context);
        if(ret) {
            throw XInterface::XInterfaceError(formatString("Error during initialization of libusb libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
    }
    CyFXUSBDevice::List list;
    CyFXLibUSBDevice::USBList devlist;
    for(int n = 0; n < devlist.size; ++n) {
        list.push_back(std::make_shared<CyFXLibUSBDevice>(devlist[n]));
    }
    return list;
}
void
CyFXLibUSBDevice::open() {
    if( !handle) {
        libusb_device_descriptor desc;
        int ret = libusb_get_device_descriptor(dev, &desc);
        if(ret) {
            throw XInterface::XInterfaceError(formatString("Error obtaining dev. desc. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }

        int bus_num = libusb_get_bus_number(dev);
        int addr = libusb_get_device_address(dev);
    //    fprintf(stderr, "USB %d: PID=%d,VID=%d,BUS#%d,ADDR=%d.\n",
    //        n, desc.idProduct, desc.idVendor, bus_num, addr);

        ret = libusb_open(dev, &handle);
        if(ret) {
            throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }

        unsigned char manu[256] = {}, prod[256] = {}, serial[256] = {};
        libusb_get_string_descriptor_ascii( handle, desc.iManufacturer, manu, 255);
        libusb_get_string_descriptor_ascii( handle, desc.iProduct, prod, 255);
        libusb_get_string_descriptor_ascii( handle, desc.iSerialNumber, serial, 255);
        fprintf(stderr, "USB: VID=0x%x, PID=0x%x,BUS#%d,ADDR=%d;%s;%s;%s.\n",
            desc.idVendor, desc.idProduct, bus_num, addr, manu, prod, serial);

    //    ret = libusb_set_auto_detach_kernel_driver( *h, 1);
    //    if(ret) {
    //        fprintf(stderr, "USB %d: Warning auto detach is not supported: %s\n", n, libusb_error_name(ret));
    //    }
        ret = libusb_kernel_driver_active(handle, 0);
        if(ret < 0) {
            libusb_close(handle); handle = nullptr;
            throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
        if(ret == 1) {
            fprintf(stderr, "USB: kernel driver is active, detaching...\n");
            ret = libusb_detach_kernel_driver(handle, 0);
            if(ret < 0) {
                libusb_close(handle); handle = nullptr;
                throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
            }
        }
    //    ret = libusb_set_configuration( *h, 1);
        ret = libusb_claim_interface(handle, 0);
        if(ret) {
            libusb_close(handle); handle = nullptr;
            throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
        ret = libusb_set_interface_alt_setting(handle, 0 , 0 );
        if(ret) {
            libusb_release_interface(handle,0);
            libusb_close(handle); handle = nullptr;
            throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
    }
}

void
CyFXLibUSBDevice::close() {
    //hacks for stability.
    libusb_clear_halt(handle, 0x2);
    libusb_clear_halt(handle, 0x6);
    libusb_clear_halt(handle, 0x8);
    libusb_reset_device(handle);
    libusb_release_interface(handle,0);
    libusb_close(handle); handle = nullptr;
}

void
CyFXLibUSBDevice::finalize() {
    libusb_exit(CyFXLibUSBDevice::context);
}

int
CyFXLibUSBDevice::controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, const uint8_t *wbuf, int len) {
    std::vector<uint8_t> buf(len);
    std::copy(wbuf, wbuf + len, buf.begin());
    int ret = libusb_control_transfer(handle,
        LIBUSB_ENDPOINT_OUT | (uint8_t)type,
        (uint8_t)request,
        value, index, &buf[0], len, USB_TIMEOUT);
    if(ret < 0) {
        throw XInterface::XInterfaceError(formatString("USB: %s.", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    return ret;
}

int
CyFXLibUSBDevice::controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, uint8_t *rdbuf, int len) {
    int ret = libusb_control_transfer(handle,
        LIBUSB_ENDPOINT_IN | (int8_t)type,
        (uint8_t)request,
        value, index, rdbuf, len, USB_TIMEOUT);
    if(ret < 0) {
        throw XInterface::XInterfaceError(formatString("USB: %s.", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    return ret;
}


XString
CyFXLibUSBDevice::getString(int descid) {
    char s[128];
    int ret = libusb_get_string_descriptor_ascii(handle, descid, (uint8_t*)s, 127);
    if(ret < 0) {
         throw XInterface::XInterfaceError(formatString("Error during USB get string desc.: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    s[ret] = '\0';
    return s;
}

CyFXUSBDevice::AsyncIO
CyFXLibUSBDevice::asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len) {
    AsyncIO async;
    async.ptr()->buf.reset(new std::vector<uint8_t>(len));
    std::copy(buf, buf + len, async.ptr()->buf->begin());
    libusb_fill_bulk_transfer(async.ptr()->transfer, handle,
            LIBUSB_ENDPOINT_OUT | ep, &async.ptr()->buf->at(0), len,
            &AsyncIO::Transfer::cb_fn, &async, USB_TIMEOUT);
    int ret = libusb_submit_transfer(async.ptr()->transfer);
    if(ret != 0) {
         throw XInterface::XInterfaceError(formatString("USB Error during submitting a transfer: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    return async;
}

CyFXUSBDevice::AsyncIO
CyFXLibUSBDevice::asyncBulkRead(uint8_t ep, uint8_t* buf, int len) {
    AsyncIO async;
    async.ptr()->buf.reset(new std::vector<uint8_t>(len));
    async.ptr()->rdbuf = buf;
    libusb_fill_bulk_transfer(async.ptr()->transfer, handle,
            LIBUSB_ENDPOINT_IN | ep, &async.ptr()->buf->at(0), len,
            &AsyncIO::Transfer::cb_fn, &async, USB_TIMEOUT);
    int ret = libusb_submit_transfer(async.ptr()->transfer);
    if(ret != 0) {
         throw XInterface::XInterfaceError(formatString("USB Error during submitting a transfer: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    return async;
}


