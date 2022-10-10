/***************************************************************************
        Copyright (C) 2002-2022 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "oceanopticsusb.h"
#include "interface.h"
#include <libusb-1.0/libusb.h>
#include <cstring>

//!todo sharing codes with cyfxusb.cpp cyfxusb_libusb.cpp


static constexpr int USB_TIMEOUT = 4000; //ms

struct OceanOpticsLibUSBDevice : public OceanOpticsUSBDevice {
    OceanOpticsLibUSBDevice(libusb_device *d) : handle(nullptr), dev(d) {
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
    ~OceanOpticsLibUSBDevice() {
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

    struct AsyncIO : public OceanOpticsUSBDevice::AsyncIO {
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

OceanOpticsLibUSBDevice::Context OceanOpticsLibUSBDevice::s_context;

OceanOpticsLibUSBDevice::USBList::USBList() noexcept {
    size = libusb_get_device_list(s_context.context, &list);
    if(size < 0 ) {
        fprintf(stderr, "Error during dev. enum. of libusb: %s\n", libusb_error_name(size));
    }
}


bool
OceanOpticsLibUSBDevice::AsyncIO::hasFinished() const noexcept {
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
OceanOpticsLibUSBDevice::AsyncIO::waitFor() {
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
OceanOpticsLibUSBDevice::AsyncIO::abort() noexcept {
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

constexpr unsigned int TIMEOUT_MS = 500;

XThreadLocal<std::vector<uint8_t>>
OceanOpticsUSBDevice::AsyncIO::stl_bufferGarbage;

XMutex XOceanOpticsUSBInterface::s_mutex;
int XOceanOpticsUSBInterface::s_refcnt = 0;
typename XOceanOpticsUSBInterface::USBDevice::List XOceanOpticsUSBInterface::s_devices;

static constexpr unsigned int OCEANOPTICS_VENDOR_ID = 0x2457;
static const std::map<unsigned int, std::string> cs_oceanOpticsModels = {
    {0x1012, "HR4000"},
    {0x1016, "HR2000+"},
    {0x101e, "USB2000+"},
};

XOceanOpticsUSBInterface::XOceanOpticsUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
 XCustomCharInterface(name, runtime, driver) {
    control()->setUIEnabled(false);
    device()->setUIEnabled(false);
    m_threadInit.reset(new XThread{shared_from_this(), [this](const atomic<bool>&) {
        XScopedLock<XMutex> slock(s_mutex);
        try {
            if( !(s_refcnt++)) {
                openAllUSBdevices();
            }

            for(auto &&x : s_devices) {
                if( !x) continue;
                std::string name = cs_oceanOpticsModels.at(x->productID());
                if(name.length()) {
                    auto shot = iterate_commit([=](Transaction &tr){
                        tr[ *device()].add(name);
                    });
                    m_candidates.emplace(name, x);
                }
            }
        }
        catch (XInterface::XInterfaceError &e) {
            e.print();
        }
        control()->setUIEnabled(true);
        device()->setUIEnabled(true);
    }});
}

XOceanOpticsUSBInterface::~XOceanOpticsUSBInterface() {
    this->m_threadInit.reset();
    XScopedLock<XMutex> slock(s_mutex);
    m_usbDevice.reset();
    m_candidates.clear();
    s_refcnt--;
    if( !s_refcnt)
        closeAllUSBdevices();
}

void
XOceanOpticsUSBInterface::openAllUSBdevices() {
    gMessagePrint("USB: Initializing/opening all the devices");
    s_devices = USBDevice::enumerateDevices();

    {
        for(auto &&x : s_devices) {
            if( !x) continue;
            try {
                if(x->vendorID() == OCEANOPTICS_VENDOR_ID) {
                    cs_oceanOpticsModels.at(x->productID());
                    continue;
                }
            }
            catch (std::out_of_range &e) {
            }
            x.reset();
            continue;
        }
    }
    gMessagePrint("USB: initialization done.");
}

void
XOceanOpticsUSBInterface::closeAllUSBdevices() {
    s_devices.clear();
}

void
XOceanOpticsUSBInterface::open() {
    Snapshot shot( *this);
    auto it = m_candidates.find(shot[ *device()].to_str());
    if(it != m_candidates.end()) {
        m_usbDevice = it->second;
        usb()->open();
    }
    else {
        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    }
}

void
XOceanOpticsUSBInterface::close() {
    if(usb())
        usb()->close();
    m_usbDevice.reset();
}

int64_t
OceanOpticsUSBDevice::bulkWrite(uint8_t ep, const uint8_t *buf, int len) {
    auto async = asyncBulkWrite(ep, buf, len, TIMEOUT_MS);
    auto ret = async->waitFor();
    return ret;
}

int64_t
OceanOpticsUSBDevice::bulkRead(uint8_t ep, uint8_t* buf, int len) {
    auto async = asyncBulkRead(ep, buf, len, TIMEOUT_MS);
    return async->waitFor();
}

OceanOpticsUSBDevice::List
OceanOpticsUSBDevice::enumerateDevices() {
    OceanOpticsUSBDevice::List list;
    OceanOpticsLibUSBDevice::USBList devlist;
    for(int n = 0; n < devlist.size; ++n) {
        list.push_back(std::make_shared<OceanOpticsLibUSBDevice>(devlist[n]));
    }
    return list;
}
void
OceanOpticsLibUSBDevice::open() {
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
OceanOpticsLibUSBDevice::close() {
    if(handle) {
//        libusb_clear_halt(handle, 0x2);
//        libusb_clear_halt(handle, 0x6);
//        libusb_clear_halt(handle, 0x8);
//        libusb_reset_device(handle);
        libusb_release_interface(handle,0);
        libusb_close(handle);
    }
    handle = nullptr;
}

int
OceanOpticsLibUSBDevice::controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
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
OceanOpticsLibUSBDevice::controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
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
OceanOpticsLibUSBDevice::getString(int descid) {
    char s[128];
    int ret = libusb_get_string_descriptor_ascii(handle, descid, (uint8_t*)s, 127);
    if(ret < 0) {
         throw XInterface::XInterfaceError(formatString("Error during USB get string desc.: %s\n", libusb_error_name(ret)), __FILE__, __LINE__);
    }
    s[ret] = '\0';
    return s;
}

unique_ptr<OceanOpticsUSBDevice::AsyncIO>
OceanOpticsLibUSBDevice::asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms) {
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

unique_ptr<OceanOpticsUSBDevice::AsyncIO>
OceanOpticsLibUSBDevice::asyncBulkRead(uint8_t ep, uint8_t* buf, int len, unsigned int timeout_ms) {
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
void
XOceanOpticsUSBInterface::initDevice() {
    uint8_t cmds[] = {(uint8_t)CMD::INIT};
    usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));
}

void
XOceanOpticsUSBInterface::setIntegrationTime(unsigned int us) {
    uint8_t hh = us / 0x1000000uL;
    uint8_t hl = (us / 0x10000uL) % 0x100uL;
    uint8_t lh = (us / 0x100uL) % 0x100uL;
    uint8_t ll = us % 0x100uL;
    uint8_t cmds[] = {(uint8_t)CMD::SET_INTEGRATION_TIME, hh, hl, lh, ll}; //littleendian
    usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));
}

std::vector<uint8_t>
XOceanOpticsUSBInterface::readInstrumStatus() {
    XScopedLock<XOceanOpticsUSBInterface> lock( *this);
    uint8_t cmds[] = {(uint8_t)CMD::QUERY_OP_INFO};
    usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));
    std::vector<uint8_t> stat(16);
    usb()->bulkRead(m_ep_in_others, (uint8_t*)&stat[0], stat.size());
    return stat;
}

XOceanOpticsUSBInterface::InstrumConfig
XOceanOpticsUSBInterface::readConfigurations() {
    InstrumConfig config;
    auto fn_query_conf = [this](uint8_t no){
        XScopedLock<XOceanOpticsUSBInterface> lock( *this);
        uint8_t cmds[] = {(uint8_t)CMD::QUERY_INFO, no};
        usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));
        uint8_t buf[CMD_READ_SIZE + 1];
        buf[CMD_READ_SIZE] = '\0';
        int size = usb()->bulkRead(m_ep_in_others, buf, CMD_READ_SIZE);
        if((buf[0] != cmds[0]) || (buf[1] != cmds[1]))
            throw XInterface::XConvError(__FILE__, __LINE__);
        return std::string((char*)&buf[2]);
    };
    config.serialNo = fn_query_conf(0);
    for(unsigned int i = 0; i < 4; ++i)
        config.wavelenCalib[i] = fn_query_conf(i + 1);
    config.strayLightConst = fn_query_conf(5);
    for(unsigned int i = 0; i < 8; ++i)
        config.nonlinCorr[i] = fn_query_conf(i + 6);
    config.nlpoly = fn_query_conf(14);
    config.opticalBenchConfig = fn_query_conf(15);
    config.spectrometerConfig = fn_query_conf(16);

    return config;
}

int
XOceanOpticsUSBInterface::readSpectrum(std::vector<uint8_t> &buf) {
    XScopedLock<XOceanOpticsUSBInterface> lock( *this);
    uint8_t cmds[] = {(uint8_t)CMD::REQUEST_SPECTRA};
    usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));

    buf.resize(m_bytesInSpec);
    int len = usb()->bulkRead(m_ep_in_spec, &buf[0], buf.size());

    return len;
}

