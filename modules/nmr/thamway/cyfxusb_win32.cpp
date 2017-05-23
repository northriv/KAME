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
#include "cyfxusb_win32.h"

#include <setupapi.h>
#include "fx2fw.h"
#include <cstring>

constexpr uint32_t IOCTL_EZUSB_GET_DEVICE_DESCRIPTOR =  0x222004;
constexpr uint32_t IOCTL_EZUSB_GET_STRING_DESCRIPTOR = 0x222044;
constexpr uint32_t IOCTL_EZUSB_ANCHOR_DOWNLOAD = 0x22201c;
constexpr uint32_t IOCTL_EZUSB_VENDOR_REQUEST = 0x222014;
constexpr uint32_t IOCTL_EZUSB_BULK_WRITE = 0x222051;
constexpr uint32_t IOCTL_EZUSB_BULK_READ = 0x22204e;

constexpr uint32_t IOCTL_ADAPT_GET_FRIENDLY_NAME = 0x220040;
constexpr uint32_t IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER = 0x220020;
constexpr uint32_t IOCTL_ADAPT_SEND_NON_EP0_TRANSFER = 0x220024;
constexpr uint32_t IOCTL_ADAPT_SEND_NON_EP0_DIRECT = 0x22004b;

CyFXWin32USBDevice::AsyncIO::~AsyncIO() {
    if( !hasFinished())
        abort();
}

bool
CyFXWin32USBDevice::AsyncIO::hasFinished() const noexcept {
    return HasOverlappedIoCompleted( &overlap);
}
int64_t
CyFXWin32USBDevice::AsyncIO::waitFor() {
    if(m_count_imm < 0) {
        ssize_t offset = ioctlbuf_rdpos ? (ioctlbuf_rdpos - &ioctlbuf[prepadding]) : 0;
        DWORD num;
        if( !GetOverlappedResult(handle, &overlap, &num, true)) {
            auto e = GetLastError();
//            if(e == ERROR_OPERATION_ABORTED)
//                return 0; //IO has been canceled.
            if(e == ERROR_BUSY)
                throw XInterface::XInterfaceError(i18n("USB device is busy."), __FILE__, __LINE__);
            throw XInterface::XInterfaceError(formatString("Error during USB tranfer:%d.", (int)e), __FILE__, __LINE__);
        }
        if(num < offset) {
            throw XInterface::XInterfaceError(i18n("Too short return packet during USB tranfer."), __FILE__, __LINE__);
        }
        finalize(num - offset);
    }
    if(rdbuf) {
        std::memcpy(rdbuf, ioctlbuf_rdpos, m_count_imm);
    }
    if(ioctlbuf.size() > AsyncIO::stl_bufferGarbage->size())
        stl_bufferGarbage->swap(ioctlbuf);
    return m_count_imm;
}
bool
CyFXWin32USBDevice::AsyncIO::abort() noexcept {
    return CancelIoEx(handle, &overlap);
}
unique_ptr<CyFXWin32USBDevice::AsyncIO>
CyFXWin32USBDevice::asyncIOCtrl(uint64_t code, const void *in, ssize_t size_in, void *out, ssize_t size_out) {
    DWORD nbyte;
    unique_ptr<AsyncIO> async(new AsyncIO);
    async->handle = handle;
    if( !DeviceIoControl(handle, code, (void*)in, size_in, out, size_out, &nbyte, &async->overlap)) {
        auto e = GetLastError();
        if(e == ERROR_IO_PENDING)
            return std::move(async); //async IO has been submitted.
        throw XInterface::XInterfaceError(formatString("IOCTL error:%d.", (int)e), __FILE__, __LINE__);
    }
    async->finalize(nbyte); //IO has been synchronously performed.
    return std::move(async);
}

CyFXWin32USBDevice::CyFXWin32USBDevice(HANDLE h, const XString &n) : CyFXUSBDevice(),
    handle(h), name(n) {
}

int64_t
CyFXWin32USBDevice::ioCtrl(uint64_t code, const void *in, ssize_t size_in, void *out, ssize_t size_out) {
    auto async = asyncIOCtrl(code, in, size_in, out, size_out);
    return async->waitFor();
}

void
CyFXWin32USBDevice::setIDs() {
    struct DeviceDescriptor {
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
    };
    static_assert(sizeof(DeviceDescriptor)== 18, "");
//    static_assert(sizeof(USB_DEVICE_DESCRIPTOR)== 18, "");
    //obtains device descriptor
    DeviceDescriptor dev_desc;
    auto buf = reinterpret_cast<uint8_t*>( &dev_desc);
    //Reads common descriptor.
    controlRead(CtrlReq::GET_DESCRIPTOR,
        CtrlReqType::STANDARD, USB_DEVICE_DESCRIPTOR_TYPE * 0x100u,
        0, buf, sizeof(dev_desc));

    m_productID = dev_desc.idProduct;
    m_vendorID = dev_desc.idVendor;
}

CyFXUSBDevice::List
CyFXUSBDevice::enumerateDevices() {
    CyFXUSBDevice::List list;
    //Searching for ezusb.sys devices, having symbolic files, first.
    for(int idx = 0; idx < 10; ++idx) {
        XString name = formatString("\\\\.\\Ezusb-%d",idx);
        fprintf(stderr, "EZUSB: opening device: %s\n", name.c_str());
        HANDLE h = CreateFileA(name.c_str(),
           GENERIC_WRITE | GENERIC_READ,
           FILE_SHARE_WRITE | FILE_SHARE_READ,	0,
           OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
        if(h == INVALID_HANDLE_VALUE) {
            int e = (int)GetLastError();
            if(e != ERROR_FILE_NOT_FOUND)
                throw XInterface::XInterfaceError(formatString("INVALID HANDLE %d for %s\n", e, name.c_str()), __FILE__, __LINE__);
            if(idx == 0) break;
            return std::move(list);
        }
        auto dev = std::make_shared<CyFXEzUSBDevice>(h, name);
        dev->setIDs();
        list.push_back(dev);
    }

    //Standard scheme with CyUSB3.sys devices.
    //GUID: AE18AA60-7F6A-11d4-97DD-00010229B959
    constexpr GUID guid = {0xae18aa60, 0x7f6a, 0x11d4, 0x97, 0xdd, 0x0, 0x1, 0x2, 0x29, 0xb9, 0x59};

    HDEVINFO hdev = SetupDiGetClassDevs(&guid, NULL, NULL, DIGCF_PRESENT | DIGCF_INTERFACEDEVICE);
    if(hdev == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "No CyUSB3 device\n");
        return list;
    }
    SP_INTERFACE_DEVICE_DATA info;
    info.cbSize = sizeof(SP_INTERFACE_DEVICE_DATA);
    for(int idx = 0; idx < 50; ++idx) {
        if(! SetupDiEnumDeviceInterfaces(hdev, 0, &guid, idx, &info))
            break;
        DWORD size = 0;
        SetupDiGetDeviceInterfaceDetail(hdev, &info, NULL, 0, &size, NULL);
        std::vector<char> buf(size, 0);
        auto detail = reinterpret_cast<PSP_INTERFACE_DEVICE_DETAIL_DATA>(&buf[0]);
        detail->cbSize = sizeof(SP_INTERFACE_DEVICE_DETAIL_DATA);
        ULONG len;
        if(SetupDiGetInterfaceDeviceDetail(hdev, &info, detail, size, &len, NULL)){
            char str[1024] = {};
            int ret = WideCharToMultiByte(CP_UTF8, 0, detail->DevicePath,
                    -1, str, sizeof(str) - 1, NULL, NULL);
            if(ret < 0) {
                int e = (int)GetLastError();
                fprintf(stderr, "Unicode conversion failed: %d\n", e);
                continue;
            }
            XString name(str);
            HANDLE h = CreateFile(detail->DevicePath,
                GENERIC_WRITE | GENERIC_READ,
                FILE_SHARE_WRITE | FILE_SHARE_READ,	0,
                OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
            if(h == INVALID_HANDLE_VALUE) {
                int e = (int)GetLastError();
                fprintf(stderr, "INVALID HANDLE %d for %s\n", e, name.c_str());
                continue;
            }
            auto dev = std::make_shared<CyUSB3Device>(h, name);
            fprintf(stderr, "CyUSB3 device found: %s\n", dev->friendlyName().c_str());
            dev->setIDs();
            list.push_back(dev);
        }
    }
    SetupDiDestroyDeviceInfoList(hdev);
    return std::move(list);
}
void
CyFXWin32USBDevice::open() {
    if( !handle) {
        handle = CreateFileA(name.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            0,
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED,
            NULL);
        if(handle == INVALID_HANDLE_VALUE) {
            handle = nullptr;
            int e = (int)GetLastError();
            throw XInterface::XInterfaceError(formatString("INVALID HANDLE %d for %s\n", e, name.c_str()), __FILE__, __LINE__);
        }
    }
}

void
CyFXWin32USBDevice::close() {
    if(handle) CloseHandle(handle);
    handle = nullptr;
}

XString
CyFXEzUSBDevice::getString(int descid) {
    uint8_t  buf[130] = {};
    StringDescCtrl sin;
    sin.Index = descid;
    sin.LanguageId = 27;
    //Reads common descriptor.
    ioCtrl(IOCTL_EZUSB_GET_STRING_DESCRIPTOR, &sin, sizeof(sin), buf, sizeof(buf));

    int len = buf[0];
    {
        //Reads string descriptor.
        char str[len / 2 + 1] = {};
        uint8_t desc_type = buf[1];
        if(desc_type != USB_STRING_DESCRIPTOR_TYPE)
            throw XInterface::XInterfaceError(i18n("Size mismatch during control transfer."), __FILE__, __LINE__);
        if(len >= sizeof(buf))
            throw XInterface::XInterfaceError(i18n("Size mismatch during control transfer."), __FILE__, __LINE__);
        char *s = str;
        for(int i = 0; i < buf[0]/2 - 1; i++){
            *(s++) = (char)buf[2 * i + 2];
        }
        return {str};
    }
}

unique_ptr<CyFXUSBDevice::AsyncIO>
CyFXEzUSBDevice::asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms) {
    std::vector<uint8_t> ioctlbuf(sizeof(BulkTransferCtrl));
    auto tr = reinterpret_cast<BulkTransferCtrl *>(&ioctlbuf[0]);
    //FX2FW specific
    switch(ep) {
    case 2:
        tr->pipeNum = TFIFO;
        break;
    case 8:
        tr->pipeNum = CPIPE;
        break;
    default:
        throw XInterface::XInterfaceError("Unknown pipe", __FILE__, __LINE__);
    }

    auto ret = asyncIOCtrl(IOCTL_EZUSB_BULK_WRITE,
        &ioctlbuf[0], ioctlbuf.size(), const_cast<uint8_t*>(buf), len);
    ret->ioctlbuf.swap(ioctlbuf); //buffer shouldn't be freed in this scope.
    return std::move(ret);
}

unique_ptr<CyFXUSBDevice::AsyncIO>
CyFXEzUSBDevice::asyncBulkRead(uint8_t ep, uint8_t* buf, int len, unsigned int timeout_ms) {
    std::vector<uint8_t> ioctlbuf(sizeof(BulkTransferCtrl));
    auto tr = reinterpret_cast<BulkTransferCtrl *>(&ioctlbuf[0]);
    //FX2FW specific
    switch(ep) {
    case 6:
        tr->pipeNum = RFIFO;
        break;
    default:
        throw XInterface::XInterfaceError("Unknown pipe", __FILE__, __LINE__);
    }

    auto ret = asyncIOCtrl(IOCTL_EZUSB_BULK_READ,
        &ioctlbuf[0], ioctlbuf.size(), buf, len);
    ret->ioctlbuf.swap(ioctlbuf); //buffer shouldn't be freed in this scope.
    return std::move(ret);
}


int
CyFXEzUSBDevice::controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, const uint8_t *wbuf, int len) {
    if(type == CtrlReqType::VENDOR) {
        if((len > 1) && (value == 0) && (index == 0)) {
            ioCtrl(IOCTL_EZUSB_ANCHOR_DOWNLOAD, wbuf, len, NULL, 0);
            return len;
        }
        else {
            static_assert(sizeof(VendorRequestCtrl) == 10, "");
            std::vector<uint8_t> buf(9 + len);
            auto tr = reinterpret_cast<VendorRequestCtrl *>(&buf[0]);
            *tr = VendorRequestCtrl{}; //0 fill.
            std::copy(wbuf, wbuf + len, &tr->bData);
            tr->bRequest = (uint8_t)request;
            tr->wValue = value;
            tr->wIndex = index;
            tr->wLength = len;
            tr->direction = 0;
            ioCtrl(IOCTL_EZUSB_VENDOR_REQUEST, &buf[0], buf.size(), NULL, 0);
            return len;
        }
    }
    throw XInterface::XInterfaceError("Unknown type.", __FILE__, __LINE__);
}

int
CyFXEzUSBDevice::controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, uint8_t *rdbuf, int len) {
    switch(type) {
    case CtrlReqType::VENDOR:
        {
           std::vector<uint8_t> buf(sizeof(VendorRequestCtrl));
           auto tr = reinterpret_cast<VendorRequestCtrl *>(&buf[0]);
            *tr = VendorRequestCtrl{}; //0 fill.
            tr->bRequest = (uint8_t)request;
            tr->wValue = value;
            tr->wIndex = index;
            tr->wLength = len;
            tr->direction = 1;
            return ioCtrl(IOCTL_EZUSB_VENDOR_REQUEST, &buf[0], buf.size(), rdbuf, len);
        }
    case CtrlReqType::STANDARD:
        if(request == CtrlReq::GET_DESCRIPTOR) {
            if(value / 0x100u == USB_DEVICE_DESCRIPTOR_TYPE) {
                return ioCtrl(IOCTL_EZUSB_GET_DEVICE_DESCRIPTOR, NULL, 0, rdbuf, len);
            }
        }
    default:
        throw XInterface::XInterfaceError("Unknown request type", __FILE__, __LINE__);
    }
}

//void
//CyUSB3Device::setIDs() {
//    unsigned int vid, pid;
//    if(sscanf(name.c_str(), "\\\\?\\usb#vid_%x&pid_%x", &vid, &pid) != 2)
//        throw XInterface::XInterfaceError("Unknown USB handle.", __FILE__, __LINE__);
//    m_vendorID = vid;
//    m_productID = pid;
//}

std::vector<uint8_t>
CyUSB3Device::setupSingleTransfer(uint8_t ep, CtrlReq request,
    CtrlReqType type, uint16_t value, uint16_t index, int len, uint32_t timeout_ms) {
    std::vector<uint8_t> buf;
    AsyncIO::stl_bufferGarbage->swap(buf); //recycles large vector from TLS.
    buf.reserve(SIZEOF_SINGLE_TRANSFER + len + PAD_BEFORE); //for async. transfer to prevent size doubling.
    buf.resize(SIZEOF_SINGLE_TRANSFER + len);
    struct SetupPacket {
        uint8_t bmRequest, bRequest;
        uint16_t wValue, wIndex, wLength;
        uint32_t timeOut;
    };//end of setup packet., 12 bytes.
    auto tr = reinterpret_cast<SetupPacket *>(&buf[0]);
    *tr = {}; //zero clear.
    tr->bmRequest = (uint8_t)type;
    tr->bRequest = (uint8_t)request;
    tr->wValue = value;
    tr->wIndex = index;
    tr->wLength = len;
    tr->timeOut = timeout_ms ? timeout_ms / 1000u : 0xffffffffu; //sec. or infinite
    struct Packet1 {
        uint8_t bReserved2, ucEndpointAddress;
    };
    auto tr1 = reinterpret_cast<Packet1*>(&buf[sizeof(SetupPacket)]);
    *tr1 = {}; //zero clear;
    tr1->ucEndpointAddress = ep;
    struct Packet2 {
        uint32_t ntStatus, usbdStatus, isoPacketOffset, isoPacketLength;
        uint32_t bufferOffset, bufferLength;
    } packet2 = {0, 0, 0, 0, SIZEOF_SINGLE_TRANSFER, (uint32_t)len};
    std::copy( (uint8_t*)(&packet2), (uint8_t*)(&packet2 + 1),
        &buf[sizeof(SetupPacket) + sizeof(Packet1)]); //SSE2 will crash w/ mal-alignment.
    static_assert(sizeof(SetupPacket) + sizeof(Packet1) +
        ((uint8_t*)(&packet2 + 1) - (uint8_t*)(&packet2)) == SIZEOF_SINGLE_TRANSFER, "");
    return std::move(buf);
}

int
CyUSB3Device::controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, const uint8_t *wbuf, int len) {
    auto buf = setupSingleTransfer(0, request, type, value, index, len, 1000);
    std::copy(wbuf, wbuf + len, &buf[SIZEOF_SINGLE_TRANSFER]);
    ioCtrl(IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER, &buf[0], buf.size(), &buf[0], buf.size());
    return len;
}

int
CyUSB3Device::controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, uint8_t *rdbuf, int len) {
    auto buf = setupSingleTransfer(0, request, (CtrlReqType)(0x80u | (uint8_t)type), value, index, len, 1000);
    int ret = ioCtrl(IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER, &buf[0], buf.size(), &buf[0], buf.size());
    if((ret < SIZEOF_SINGLE_TRANSFER) || (ret > SIZEOF_SINGLE_TRANSFER + len))
        throw XInterface::XInterfaceError(i18n("Size mismatch during control transfer."), __FILE__, __LINE__);
    std::copy( &buf[SIZEOF_SINGLE_TRANSFER], &buf[ret], rdbuf);
    return ret - SIZEOF_SINGLE_TRANSFER;
}


XString
CyUSB3Device::getString(int descid) {
    uint8_t buf[4];
    //Reads supported LangID.
    controlRead(CtrlReq::GET_DESCRIPTOR,
        CtrlReqType::STANDARD, USB_STRING_DESCRIPTOR_TYPE * 0x100u,
        0, buf, sizeof(buf));
    uint16_t langid = buf[2] + buf[3] * 0x100u; //0x0409 (english)
    //Reads common descriptor.
    controlRead(CtrlReq::GET_DESCRIPTOR,
        CtrlReqType::STANDARD, USB_STRING_DESCRIPTOR_TYPE * 0x100u + descid,
        langid, buf, 2);
    int len = buf[0];
    if(len) {
        //Reads string descriptor.
        uint8_t buf[len] = {};
        char str[len / 2 + 1] = {};
        int ret = controlRead(CtrlReq::GET_DESCRIPTOR,
            CtrlReqType::STANDARD, USB_STRING_DESCRIPTOR_TYPE * 0x100u + descid,
            27, buf, len);
        if(ret <= 2)
            throw XInterface::XInterfaceError(i18n("Size mismatch during control transfer."), __FILE__, __LINE__);
        uint8_t desc_type = buf[1];
        if(desc_type != USB_STRING_DESCRIPTOR_TYPE)
            throw XInterface::XInterfaceError(i18n("Size mismatch during control transfer."), __FILE__, __LINE__);
        char *s = str;
        for(int i = 0; i < buf[0]/2 - 1; i++){
            *(s++) = (char)buf[2 * i + 2];
        }
        return {str};
    }
    throw XInterface::XInterfaceError(i18n("Could not obtain string desc.."), __FILE__, __LINE__);
}

unique_ptr<CyFXUSBDevice::AsyncIO>
CyUSB3Device::asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms) {
    auto ioctlbuf = setupSingleTransfer(ep, (CtrlReq)0, (CtrlReqType)0, 0, 0, len, timeout_ms);
    //shifts by PAD_BEFORE=2 to align the user data with 8bytes.
    ioctlbuf.resize(len + SIZEOF_SINGLE_TRANSFER + PAD_BEFORE);
    std::memmove( &ioctlbuf[PAD_BEFORE], &ioctlbuf[0], SIZEOF_SINGLE_TRANSFER);
    std::memcpy(&ioctlbuf[SIZEOF_SINGLE_TRANSFER + PAD_BEFORE], buf, len);
    auto ret = asyncIOCtrl(IOCTL_ADAPT_SEND_NON_EP0_TRANSFER,
       &ioctlbuf[PAD_BEFORE], ioctlbuf.size() - PAD_BEFORE,
       &ioctlbuf[PAD_BEFORE], ioctlbuf.size() - PAD_BEFORE);
    ret->ioctlbuf.swap(ioctlbuf); //buffer shouldn't be freed in this scope.
    return std::move(ret);
}

unique_ptr<CyFXUSBDevice::AsyncIO>
CyUSB3Device::asyncBulkRead(uint8_t ep, uint8_t* buf, int len, unsigned int timeout_ms) {
    auto ioctlbuf = setupSingleTransfer(0x80u | ep, (CtrlReq)0, (CtrlReqType)0, 0, 0, len, timeout_ms);
    //shifts by PAD_BEFORE=2 to align the user data with 8bytes.
    ioctlbuf.resize(len + SIZEOF_SINGLE_TRANSFER + PAD_BEFORE);
    std::memmove( &ioctlbuf[PAD_BEFORE], &ioctlbuf[0], SIZEOF_SINGLE_TRANSFER);
    auto ret = asyncIOCtrl(IOCTL_ADAPT_SEND_NON_EP0_TRANSFER,
        &ioctlbuf[PAD_BEFORE], ioctlbuf.size() - PAD_BEFORE,
        &ioctlbuf[PAD_BEFORE], ioctlbuf.size() - PAD_BEFORE);
    ret->ioctlbuf.swap(ioctlbuf); //buffer shouldn't be freed in this scope.
    ret->rdbuf = buf;
    ret->setBufferOffset( &ret->ioctlbuf[SIZEOF_SINGLE_TRANSFER + PAD_BEFORE], PAD_BEFORE);
    return std::move(ret);
}

XString
CyUSB3Device::friendlyName() {
    char friendlyname[256] = {};
    ioCtrl(IOCTL_ADAPT_GET_FRIENDLY_NAME,
          friendlyname, sizeof(friendlyname), friendlyname, sizeof(friendlyname));
    return friendlyname;
}

