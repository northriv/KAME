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
#include <cstring>

static constexpr int USB_TIMEOUT = 4000; //ms

struct CyFXLibUSBDevice : public CyFXUSBDevice {
    CyFXLibUSBDevice(libusb_device *d) : handle(nullptr), dev(d) {
        libusb_device_descriptor desc;
        int ret = libusb_get_device_descriptor(dev, &desc);
        if(ret) {
            throw XInterface::XInterfaceError(formatString("Error obtaining dev. desc. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
        m_productID = desc.idProduct;
        m_vendorID = desc.idVendor;
        fprintf(stderr, "USB dev, %x:%x\n", m_vendorID, m_productID);
        libusb_ref_device(dev);
    }
    ~CyFXLibUSBDevice() {
        libusb_unref_device(dev);
    }

    virtual void open() override;
    virtual void close() override;

    XString virtual getString(int descid) override;

//    virtual int64_t bulkWrite(uint8_t ep, const uint8_t *buf, int len) {
//        msecsleep(5);
//        int actual_length;
//        int ret = libusb_bulk_transfer(handle,
//                                       LIBUSB_ENDPOINT_OUT | ep, const_cast<uint8_t*>(buf), len, &actual_length, USB_TIMEOUT);
//        if(ret != 0)
//            throw XInterface::XInterfaceError(formatString("USB Error during a transfer: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
//        return actual_length;
//    }
//    virtual int64_t bulkRead(uint8_t ep, uint8_t* buf, int len) {
//        msecsleep(5);
//        int actual_length;
//        int ret = libusb_bulk_transfer(handle,
//                                       LIBUSB_ENDPOINT_IN | ep, buf, len, &actual_length, USB_TIMEOUT);
//        if(ret != 0)
//            throw XInterface::XInterfaceError(formatString("USB Error during a transfer: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
//        return actual_length;
//    }

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) override;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) override;

    virtual unique_ptr<AsyncIO> asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms = 0) override;
    virtual unique_ptr<AsyncIO> asyncBulkRead(uint8_t ep, uint8_t *buf, int len, unsigned int timeout_ms = 0) override;

    struct AsyncIO : public CyFXUSBDevice::AsyncIO {
        AsyncIO() {
            transfer = libusb_alloc_transfer(0);
            stl_bufferGarbage->swap(buf);
        }
        AsyncIO(AsyncIO&&) noexcept = default;
        virtual ~AsyncIO() {
            readBarrier();
            if( !completed) {
                if(abort()) {
                    try {
                        waitFor(); //wait for cb_fn() completion.
                    }
                    catch(XInterface::XInterfaceError &e) {
                        fprintf(stderr, "Error during aborting USB asyncIO: %s\n", e.msg().c_str());
                    }
                }
                else {
                    readBarrier();
                    if( !completed)
                        fprintf(stderr, "Error during aborting USB asyncIO, aborted twice!\n");
                }
            }
            libusb_free_transfer(transfer);
            if(buf.size() > stl_bufferGarbage->size())
                stl_bufferGarbage->swap(buf);
        }

        virtual bool hasFinished() const noexcept override;
        virtual int64_t waitFor() override;
        virtual bool abort() noexcept override;

        static void cb_fn(struct libusb_transfer *transfer) {
//            switch(transfer->status) {
//            case LIBUSB_TRANSFER_COMPLETED:
//                break;
//            case LIBUSB_TRANSFER_CANCELLED:
//            case LIBUSB_TRANSFER_NO_DEVICE:
//            case LIBUSB_TRANSFER_TIMED_OUT:
//            case LIBUSB_TRANSFER_ERROR:
//            case LIBUSB_TRANSFER_STALL:
//            case LIBUSB_TRANSFER_OVERFLOW:
//            default:
//                break;
//            }
            writeBarrier();
            *reinterpret_cast<int*>(transfer->user_data) = 1; //completed = 1
            writeBarrier();
        }
        std::vector<uint8_t> buf;
        libusb_transfer *transfer;
        uint8_t *rdbuf = nullptr;
        int completed = 0;
    };

    struct USBList {
        USBList() noexcept;
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
private:
    static struct Context {
        Context() {
            int ret = libusb_init( &context);
            if(ret)
                fprintf(stderr, "Error during initialization of libusb libusb: %s\n", libusb_error_name(ret));
        }
        ~Context() {
            libusb_exit(context);
        }
        libusb_context *context;
    } s_context;

    friend struct AsyncIO;
    libusb_device_handle *handle;
    libusb_device *dev;
};

CyFXLibUSBDevice::Context CyFXLibUSBDevice::s_context;

CyFXLibUSBDevice::USBList::USBList() noexcept {
    size = libusb_get_device_list(s_context.context, &list);
    if(size < 0 ) {
        fprintf(stderr, "Error during dev. enum. of libusb: %s\n", libusb_error_name(size));
    }
}


bool
CyFXLibUSBDevice::AsyncIO::hasFinished() const noexcept {
    if(completed)
        return true;
    auto start = XTime::now();
    while( !completed) {
        struct timeval tv = {};
        readBarrier();
        int ret = libusb_handle_events_timeout_completed(s_context.context, &tv, (int*)&completed); //returns immediately.
        if(ret)
            fprintf(stderr, "Error during checking status in libusb: %s\n", libusb_error_name(ret));
        if( !completed && (XTime::now() - start > 0.02)) {
            break;
        }
        //handles events within 20 ms.
        readBarrier();
    }
    return completed;
}

int64_t
CyFXLibUSBDevice::AsyncIO::waitFor() {
    auto start = XTime::now();
    while( !completed) {
        struct timeval tv;
        tv.tv_sec = USB_TIMEOUT / 1000;
        tv.tv_usec = (USB_TIMEOUT % 1000) * 1000;
        int ret = libusb_handle_events_timeout_completed(s_context.context, &tv, &completed);
        if(ret)
            throw XInterface::XInterfaceError(formatString("Error during completing transfer in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        if( !completed && (XTime::now() - start > USB_TIMEOUT * 1e-3)) {
            fprintf(stderr, "Libusb async transfer aborting due to timeout.\n");
            abort();
        }
        readBarrier();
    }
    if(completed && (transfer->status != LIBUSB_TRANSFER_COMPLETED)) {
        if(transfer->status == LIBUSB_TRANSFER_CANCELLED)
            return 0;
        if(transfer->status != LIBUSB_TRANSFER_TIMED_OUT)
            throw XInterface::XInterfaceError(formatString("Error, unhandled complete status in libusb: %s\n", libusb_error_name(transfer->status)).c_str(), __FILE__, __LINE__);
    }
    if(rdbuf) {
        readBarrier();
        assert(buf.size() >= transfer->actual_length);
        std::memcpy(rdbuf, &buf[0], transfer->actual_length);
    }
    return transfer->actual_length;
}

bool
CyFXLibUSBDevice::AsyncIO::abort() noexcept {
    int ret = libusb_cancel_transfer(transfer);
    if(ret) {
        readBarrier();
        if(completed && (ret == LIBUSB_ERROR_NOT_FOUND))
            return false; //already completed.
        gErrPrint(formatString("Error during cancelling transfer in libusb: %s\n", libusb_error_name(ret)).c_str());
        return false;
    }
    fprintf(stderr, "Libusb async transfer aborted.\n");
    return true;
}

CyFXUSBDevice::List
CyFXUSBDevice::enumerateDevices() {
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
    //    fprintf(stderr, "USB %d: PID=0x%x,VID=0x%x,BUS#%d,ADDR=%d.\n",
    //        n, desc.idProduct, desc.idVendor, bus_num, addr);

        ret = libusb_open(dev, &handle);
        if(ret) {
            handle = nullptr;
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
//        ret = libusb_kernel_driver_active(handle, 0);
//        if(ret < 0) {
////            libusb_close(handle); handle = nullptr;
////            throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
//        }
//        if(ret == 1) {
//            fprintf(stderr, "USB: kernel driver is active, detaching...\n");
//            ret = libusb_detach_kernel_driver(handle, 0);
//            if(ret < 0) {
//                libusb_close(handle); handle = nullptr;
//                throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
//            }
//        }
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
    if(handle) {
//        libusb_clear_halt(handle, 0x2);
//        libusb_clear_halt(handle, 0x6);
//        libusb_clear_halt(handle, 0x8);
        libusb_reset_device(handle);
        libusb_release_interface(handle,0);
        libusb_close(handle);
        fprintf(stderr, "USB: closed.\n");
    }
    handle = nullptr;
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

unique_ptr<CyFXUSBDevice::AsyncIO>
CyFXLibUSBDevice::asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms) {
    unique_ptr<AsyncIO> async(new AsyncIO);
    async->buf.resize(len);
    std::memcpy( &async->buf[0], buf, len);
    libusb_fill_bulk_transfer(async->transfer, handle,
            LIBUSB_ENDPOINT_OUT | ep, &async->buf.at(0), len,
            &AsyncIO::cb_fn, &async->completed, timeout_ms);
    int ret = libusb_submit_transfer(async->transfer);
    if(ret != 0) {
         async->completed = true; //not to abort() in the destructor.
         throw XInterface::XInterfaceError(formatString("USB Error during submitting a transfer: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    return std::move(async);
}

unique_ptr<CyFXUSBDevice::AsyncIO>
CyFXLibUSBDevice::asyncBulkRead(uint8_t ep, uint8_t* buf, int len, unsigned int timeout_ms) {
    unique_ptr<AsyncIO> async(new AsyncIO);
    async->buf.resize(len);
    async->rdbuf = buf;
    libusb_fill_bulk_transfer(async->transfer, handle,
            LIBUSB_ENDPOINT_IN | ep, &async->buf.at(0), len,
            &AsyncIO::cb_fn, &async->completed, timeout_ms);
    int ret = libusb_submit_transfer(async->transfer);
    if(ret != 0) {
         async->completed = true; //not to abort() in the destructor.
         throw XInterface::XInterfaceError(formatString("USB Error during submitting a transfer: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    return std::move(async);
}


