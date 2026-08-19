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
#include "cyfxusb.h"
#include "interface.h"
#include <libusb-1.0/libusb.h>
#include <cstring>

static constexpr int USB_TIMEOUT = 6000; //ms

struct CyFXLibUSBDevice : public CyFXUSBDevice {
    CyFXLibUSBDevice(libusb_device *d) : handle(nullptr), dev(d) {
        libusb_device_descriptor desc;
        int ret = libusb_get_device_descriptor(dev, &desc);
        if(ret) {
            throw XInterface::XInterfaceError(formatString("Error obtaining dev. desc. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
        m_productID = desc.idProduct;
        m_vendorID = desc.idVendor;
        // m_serialNo = desc.iSerialNumber;
        fprintf(stderr, "USB dev, %x:%x\n", m_vendorID, m_productID);
        m_bus = libusb_get_bus_number(dev);
        uint8_t port_num[7] = {};
        int len = libusb_get_port_numbers(dev, port_num, sizeof(port_num));
        for(int i = 0; i < len; ++i)
            m_portPath.push_back(port_num[i]);
        libusb_ref_device(dev);
    }
    ~CyFXLibUSBDevice() {
        libusb_unref_device(dev);
    }

    virtual void open() override;
    virtual void close() override;

    XString virtual getString(int descid) override;

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
   virtual int64_t bulkWrite(uint8_t ep, const uint8_t *buf, int len) override {
       if( !handle)
           //Device was closed concurrently; a null dev_handle aborts inside libusb.
           throw XInterface::XInterfaceError("USB: bulk write attempted on a closed device handle.\n", __FILE__, __LINE__);
       msecsleep(5);
       int actual_length;
       int ret = libusb_bulk_transfer(handle,
                                      LIBUSB_ENDPOINT_OUT | ep, const_cast<uint8_t*>(buf), len, &actual_length, USB_TIMEOUT);
       if(ret != 0)
           throw XInterface::XInterfaceError(formatString("USB Error during a transfer: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
       //Flushes buffer by ZLP in WinUSB.
       int zlp_transferred;
       unsigned char zlp_buf[1];
       ret = libusb_bulk_transfer(handle,
                                    LIBUSB_ENDPOINT_OUT | ep, zlp_buf, 0, &zlp_transferred, 100);
       if(ret != 0)
           throw XInterface::XInterfaceError(formatString("USB Error during a transfer: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
       return actual_length;
   }
#endif
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
            if( !transfer)
                //Every path below fills and submits this unconditionally.
                throw XInterface::XInterfaceError("USB: libusb_alloc_transfer() failed.\n", __FILE__, __LINE__);
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
                    if( !completed) {
                        //Comes here after one of threads cancels the transfer and "Libusb async transfer is going to be aborted." is shown.
                        //, when multiple async reads are waiting at the same endpoint, in OSX.
                        fprintf(stderr, "Error during aborting USB asyncIO, aborted twice!\n");
                        msecsleep(100); //expecting cb_fn() might be called.
                    }
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

        vector_u8 buf;
        libusb_transfer *transfer;
        uint8_t *rdbuf = nullptr;
        int completed = 0;
        //! The device this transfer was submitted on, so abort() can re-check
        //! that it is still open before asking libusb to cancel.  A raw pointer
        //! is sound here: an AsyncIO is only ever produced by that device's own
        //! asyncBulkWrite()/asyncBulkRead() and handed to a caller that reached
        //! the device through a shared_ptr, so the device outlives the transfer.
        CyFXLibUSBDevice *m_owner = nullptr;
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

    unsigned int busNo() const {return m_bus;}
    const std::vector<uint8_t> &portPath() const {return m_portPath;}

    virtual XString name() const override {
        XString n = formatString(":%u", busNo());
        for(auto &&port: portPath())
            n += formatString(":%u", port);
        return n;
    }
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

    uint8_t m_bus;
    std::vector<uint8_t> m_portPath;
};

CyFXLibUSBDevice::Context CyFXLibUSBDevice::s_context;

CyFXUSBDevice::List
#ifdef USE_LIBUSB_WITH_WINCYFX
enumerateDevicesByLibUSB() {
#else
CyFXUSBDevice::enumerateDevices() {
#endif
    CyFXUSBDevice::List list;
    CyFXLibUSBDevice::USBList devlist;
    for(int n = 0; n < devlist.size; ++n) {
        list.push_back(std::make_shared<CyFXLibUSBDevice>(devlist[n]));
    }
    return list;
}

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
    // This watchdog must OUTLAST the timeout libusb itself was given for the
    // transfer, otherwise libusb can never complete it as
    // LIBUSB_TRANSFER_TIMED_OUT -- the clean outcome the status check further
    // down already handles -- and every stalled transfer is forced down the far
    // more fragile cancel path instead.  It was the other way round: this used
    // USB_TIMEOUT (6 s) while bulkWrite()/bulkRead() pass
    // TIMEOUT_MS_LONG_ENOUGH (10 s) as timeout_ms, so the TIMED_OUT branch was
    // dead code.  transfer->timeout == 0 means "no timeout" to libusb, and there
    // this watchdog is legitimately the only bound.
    const double deadline = transfer->timeout ?
        transfer->timeout * 1e-3 + 2.0 : USB_TIMEOUT * 1e-3;
    auto start = XTime::now();
    bool cancel_issued = false;
    while( !completed) {
        struct timeval tv;
        tv.tv_sec = USB_TIMEOUT / 1000;
        tv.tv_usec = (USB_TIMEOUT % 1000) * 1000;
        int ret = libusb_handle_events_timeout_completed(s_context.context, &tv, &completed);
        if(ret)
            throw XInterface::XInterfaceError(formatString("Error during completing transfer in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        if( !completed && (XTime::now() - start > deadline)) {
            if(cancel_issued)
                //The cancellation produced no completion callback within another
                //full deadline.  Waiting longer cannot help, and looping here
                //would re-issue libusb_cancel_transfer() on every iteration.
                throw XInterface::XInterfaceError("USB: async transfer neither completed nor cancelled.\n", __FILE__, __LINE__);
            fprintf(stderr, "Libusb async transfer aborting due to timeout.\n");
            cancel_issued = true;
            if( !abort())
                throw XInterface::XInterfaceError("USB: async transfer timed out and could not be cancelled.\n", __FILE__, __LINE__);
            start = XTime::now(); //grants the cancellation its own grace period.
        }
        readBarrier();
    }
    if(completed && (transfer->status != LIBUSB_TRANSFER_COMPLETED)) {
        if(transfer->status == LIBUSB_TRANSFER_CANCELLED)
            return 0;
        if(transfer->status != LIBUSB_TRANSFER_TIMED_OUT) {
            //added because HR4000 never recovers after LIBUSB_TRANSFER_OVERFLOW in osx.
            libusb_clear_halt(transfer->dev_handle, transfer->endpoint);
            throw XInterface::XInterfaceError(formatString("Error, unhandled complete status in libusb: %s\n", libusb_error_name(transfer->status)).c_str(), __FILE__, __LINE__);
        }
    }
    if(rdbuf) {
        readBarrier();
//        if(transfer->actual_length == 0) {
//            // //added because HR4000 never recovers after LIBUSB_TRANSFER_OVERFLOW in osx.
//            // libusb_clear_halt(transfer->dev_handle, transfer->endpoint);
//            throw XInterface::XInterfaceError(formatString("Error, zero-length return with complete status in libusb: %s\n", libusb_error_name(transfer->status)).c_str(), __FILE__, __LINE__);
//        }
        assert(buf.size() >= transfer->actual_length);
        std::memcpy(rdbuf, &buf[0], transfer->actual_length);
    }
    return transfer->actual_length;
}

bool
CyFXLibUSBDevice::AsyncIO::abort() noexcept {
    if(m_owner && !m_owner->handle) {
        //The device was closed while this transfer was still in flight, so
        //libusb_close() has already freed the dev_handle the transfer points at
        //and libusb_cancel_transfer() would dereference it.  That is the SIGSEGV
        //seen on the first Linux hardware run: XCyFXUSBInterface::initialize()
        //examines the shared s_devices entries -- open()/close() -- from its own
        //thread under s_mutex, while a driver thread does I/O under the device's
        //own mutex, and the two locks do not exclude each other.  Guarding here
        //keeps the crash away; the underlying race still needs fixing in the
        //interface layer's lock ordering.
        fprintf(stderr, "Libusb async transfer cannot be cancelled: device already closed.\n");
        return false;
    }
    //According to man of libusb, in osx, all the transfer for the same ep will be cancelled.
    int ret = libusb_cancel_transfer(transfer);
    if(ret) {
        readBarrier();
        if(ret == LIBUSB_ERROR_NOT_FOUND) {
            if(completed)
                return false; //already completed.
            //already canceled.
            fprintf(stderr, "Libusb async transfer is already canceled.\n");
            return true;
        }
        dbgPrint(formatString("Error during cancelling transfer in libusb: %s\n", libusb_error_name(ret)).c_str());
        return false;
    }
    fprintf(stderr, "Libusb async transfer is going to be aborted.\n");
    return true;
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
            if(ret == LIBUSB_ERROR_ACCESS) {
                // The common Linux first-run failure: /dev/bus/usb/BBB/DDD is
                // root-only, and libusb opens it O_RDWR.  This fires before the
                // claim_interface hint below ever gets a chance, so repeat the
                // advice here with the concrete device and the exact fix.
                fprintf(stderr, "USB: permission denied for %04x:%04x (bus %d, addr %d).\n"
                    "  On Linux /dev/bus/usb must be writable by this user; install the shipped"
                    " udev rule (see INSTALL.linux):\n"
                    "    sudo install -m 644 kame/70-kame.rules /etc/udev/rules.d/70-kame.rules\n"
                    "    sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                    "  then re-plug the device.  On macOS, check Privacy settings.\n",
                    desc.idVendor, desc.idProduct, bus_num, addr);
            }
            throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }

        unsigned char manu[256] = {}, prod[256] = {}, serial[256] = {};
        libusb_get_string_descriptor_ascii( handle, desc.iManufacturer, manu, 255);
        libusb_get_string_descriptor_ascii( handle, desc.iProduct, prod, 255);
        libusb_get_string_descriptor_ascii( handle, desc.iSerialNumber, serial, 255);
        fprintf(stderr, "USB: VID=0x%x, PID=0x%x,BUS#%d,ADDR=%d;%s;%s;%s.\n",
            desc.idVendor, desc.idProduct, bus_num, addr, manu, prod, serial);

        // Linux binds a kernel driver (usbserial/ftdi_sio/cdc_acm, depending
        // on the device's descriptors) to the interface as soon as it is
        // plugged in, and libusb_claim_interface() then fails with
        // LIBUSB_ERROR_BUSY.  This detach step is a no-op on macOS and
        // Windows — which is why it has been commented out since the port —
        // but on Linux it is mandatory.  Ask libusb to do it around
        // claim/release; if the backend cannot, say so and carry on so the
        // claim below still produces the real error.
        ret = libusb_set_auto_detach_kernel_driver(handle, 1);
        if(ret && (ret != LIBUSB_ERROR_NOT_SUPPORTED)) {
            fprintf(stderr, "USB: warning, auto detach of kernel driver failed: %s\n",
                libusb_error_name(ret));
        }
    //    ret = libusb_set_configuration( *h, 1);
        ret = libusb_claim_interface(handle, 0);
        if(ret) {
            if(ret == LIBUSB_ERROR_ACCESS) {
                fprintf(stderr, "USB: permission denied.  On Linux, install a udev rule granting"
                    " access to this device (see INSTALL.linux); on macOS, check Privacy settings.\n");
            }
            if(ret == LIBUSB_ERROR_BUSY) {
                fprintf(stderr, "USB: interface is claimed by a kernel driver that could not be"
                    " detached.  Unbind it, or blacklist the module.\n");
            }
            libusb_close(handle); handle = nullptr;
            throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
        ret = libusb_set_interface_alt_setting(handle, 0 , 0 );
        if(ret) {
            libusb_release_interface(handle,0);
            libusb_close(handle); handle = nullptr;
            throw XInterface::XInterfaceError(formatString("Error opening dev. in libusb: %s\n", libusb_error_name(ret)).c_str(), __FILE__, __LINE__);
        }
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
        libusb_clear_halt(handle, 0x2);
        libusb_clear_halt(handle, 0x6);
        libusb_clear_halt(handle, 0x8);
#endif
    }
}

void
CyFXLibUSBDevice::close() {
    //Clear `handle` FIRST.  Every "was the device closed concurrently?" guard in
    //this file tests it, and while the teardown below ran with `handle` still
    //set those guards had a window in which they let a caller through to an
    //already-freed dev_handle.  Publishing the null first makes them meaningful.
    libusb_device_handle *h = handle;
    handle = nullptr;
    if(h) {
        libusb_reset_device(h);
        libusb_release_interface(h, 0);
        libusb_close(h);
        fprintf(stderr, "USB: closed.\n");
    }
}

int
CyFXLibUSBDevice::controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, const uint8_t *wbuf, int len) {
    if( !handle)
        //Device was closed concurrently; a null dev_handle aborts inside libusb.
        throw XInterface::XInterfaceError("USB: control write attempted on a closed device handle.\n", __FILE__, __LINE__);
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
    if( !handle)
        //Device was closed concurrently; a null dev_handle aborts inside libusb.
        throw XInterface::XInterfaceError("USB: control read attempted on a closed device handle.\n", __FILE__, __LINE__);
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
    if( !handle)
        //Device was closed concurrently; a null dev_handle aborts inside libusb.
        throw XInterface::XInterfaceError("USB: get string desc. attempted on a closed device handle.\n", __FILE__, __LINE__);
    int ret = libusb_get_string_descriptor_ascii(handle, descid, (uint8_t*)s, sizeof(s) - 1);
    if(ret < 0) {
         throw XInterface::XInterfaceError(formatString("Error during USB get string desc.: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    s[std::min(ret, (int)(sizeof(s) - 1))] = '\0';
    return s;
}

unique_ptr<CyFXUSBDevice::AsyncIO>
CyFXLibUSBDevice::asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms) {
    if( !handle)
        //Device was closed (e.g. USB link lost / interface stopped) concurrently with a write.
        //Throw a catchable error here instead of submitting a transfer on a null dev_handle,
        //which makes libusb_submit_transfer() hit an internal assertion and abort() the whole
        //process (observed via the XThamwayPROT status-poll thread).
        throw XInterface::XInterfaceError("USB: bulk write attempted on a closed device handle.\n", __FILE__, __LINE__);
    unique_ptr<AsyncIO> async(new AsyncIO);
    async->m_owner = this;
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
    if( !handle)
        //Device was closed (e.g. USB link lost / interface stopped) concurrently with a read.
        //Throw a catchable error here instead of submitting a transfer on a null dev_handle,
        //which makes libusb_submit_transfer() hit an internal assertion and abort() the whole
        //process (observed via the XThamwayPROT status-poll thread).
        throw XInterface::XInterfaceError("USB: bulk read attempted on a closed device handle.\n", __FILE__, __LINE__);
    unique_ptr<AsyncIO> async(new AsyncIO);
    async->m_owner = this;
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


