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

#define NOMINMAX
#include <windows.h>

constexpr uint32_t IOCTL_EZUSB_GET_STRING_DESCRIPTOR = 0x222044;
//constexpr uint32_t IOCTL_EZUSB_ANCHOR_DOWNLOAD = 0x22201c;
constexpr uint32_t IOCTL_EZUSB_VENDOR_REQUEST = 0x222014;
constexpr uint32_t IOCTL_EZUSB_BULK_WRITE = 0x222051;
constexpr uint32_t IOCTL_EZUSB_BULK_READ = 0x22204e;

constexpr uint32_t IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER = 0x220020;
constexpr uint32_t IOCTL_ADAPT_SEND_NON_EP0_TRANSFER = 0x220024;
constexpr uint32_t IOCTL_ADAPT_SEND_NON_EP0_DIRECT = 0x22004b;

typedef struct _VENDOR_OR_CLASS_REQUEST_CONTROL {
   UCHAR direction;
   UCHAR requestType;
   UCHAR recepient;
   UCHAR requestTypeReservedBits;
   UCHAR request;
   USHORT value;
   USHORT index;
} VENDOR_OR_CLASS_REQUEST_CONTROL;

typedef struct _GET_STRING_DESCRIPTOR_IN {
   UCHAR    Index;
   USHORT   LanguageId;
} GET_STRING_DESCRIPTOR_IN;

typedef struct _VENDOR_REQUEST_IN {
    BYTE    bRequest;
    WORD    wValue;
    WORD    wIndex;
    WORD    wLength;
    BYTE    direction;
    BYTE    bData;
} VENDOR_REQUEST_IN;

typedef struct _BULK_TRANSFER_CONTROL {
   ULONG pipeNum;
} BULK_TRANSFER_CONTROL;

typedef struct _BULK_LATENCY_CONTROL {
   ULONG bulkPipeNum;
   ULONG intPipeNum;
   ULONG loops;
} BULK_LATENCY_CONTROL;

struct CyFXUSBDevice::AsyncIO::Transfer {
    OVERLAPPED overlap;
    HANDLE handle;
    unique_ptr<std::vector<uint8_t>> ioctlbuf;
    void *ioctlbuf_rdpos = nullptr;
    uint8_t *rdbuf = nullptr;
};

CyFXUSBDevice::AsyncIO::AsyncIO() : m_transfer(new Transfer) {
}
bool
CyFXUSBDevice::AsyncIO::hasFinished() {
    return HasOverlappedIoCompleted( &ptr()->overlap);
}
int64_t
CyFXUSBDevice::AsyncIO::waitFor() {
    if( !m_count_imm) {
        DWORD num;
        GetOverlappedResult(ptr()->handle, &ptr()->overlap, &num, true);
        finalize(num);
    }
    if(ptr()->rdbuf) {
        std::copy(ptr()->ioctlbuf_rdpos, &ptr()->ioctlbuf->at(0) + m_count_num, ptr()->rdbuf);
    }
    return m_count_imm;
}
CyFXUSBDevice::AsyncIO
CyFXWin32USBDevice::async_ioctl(uint64_t code, const void *in, ssize_t size_in, void *out, ssize_t size_out) {
    DWORD nbyte;
    AsyncIO async(tr);
    CyFXUSBDevice::AsyncIO::Transfer& tr = *async.ptr();
    tr.handle = handle;
    if( !DeviceIoControl(handle, code, in, size_in, out, size_out, &nbyte, &async.ptr()->tr)) {
        auto e = GetLastError();
        if(e == ERROR_IO_PENDING)
            return std::move(async);
        throw XInterfaceError("IOCTL error:%s.", e);
    }
    async.finanlize(nbyte);
    return std::move(async);
}
int64_t
CyFXWin32USBDevice::ioctl(uint64_t code, const void *in, ssize_t size_in, void *out, ssize_t size_out) {
    auto async = async_ioctl(code, in, size_in, out, size_out);
    return async.waitFor();
}

CyFXUSBDevice::List
CyFXUSBDevice::enumerateDevices() {
    CyFXUSBDevice::List list;
    for(int idx = 0; idx < 9; ++idx) {
        XString name = formatString("\\\\.\\Ezusb-%d",n);
        fprintf(stderr, "cusb: opening device: %s\n", name.c_str());
        HANDLE h = CreateFileA(name.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            0,
            OPEN_EXISTING,
           FILE_FLAG_OVERLAPPED,
           NULL);
        if(h == INVALID_HANDLE_VALUE) {
            int e = (int)GetLastError();
            if(e != ERROR_FILE_NOT_FOUND)
                throw XInterfaceError(formatString("INVALID HANDLE %d for %s\n", e, name.c_str()));
            if(idx == 0) break;
            return std::move(list);
        }
        list.push_back(std::make_shared<CyFXEasyUSBDevice>(h, name));
    }

    return std::move(list);
}
void
CyFXUSBDevice::open() {
    if( !handle) {
        handle = CreateFileA(name.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            0,
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED,
            NULL);
        if(handle == INVALID_HANDLE_VALUE) {
            int e = (int)GetLastError();
            throw XInterfaceError(formatString("INVALID HANDLE %d for %s\n", e, name.c_str()));
        }
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
    if(handle) CloseHandle(handle);
    handle = nullptr;
}

int
CyFXUSBDevice::initialize();
void
CyFXUSBDevice::finalize();

int
CyUSB3Device::controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, const uint8_t *wbuf, int len) {
    uint8_t buf[sizeof(SingleTransfer) + len];
    auto tr = reiterpret_cast<SingleTransfer *>(buf);
    *tr = SingleTransfer{}; //0 fill.
    std::copy(wbuf, wbuf + len, &buf[sizeof(SingleTransfer)]);
    tr.bmRequest = type;
    tr.request = request;
    tr.value = value;
    tr.index = index;
    tr.length = len;
    tr.timeOut = 1; //sec?
    tr.endpointAddress = 0;
    tr.bufferOffset = sizeof(SingleTransfer);
    tr.bufferLength = length;
    return ioctl(IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER, &buf, sizeof(buf), &buf, sizeof(buf));
}

int
CyUSB3Device::controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                               uint16_t index, uint8_t *rdbuf, int len) {
    uint8_t buf[sizeof(SingleTransfer) + len];
    auto tr = reiterpret_cast<SingleTransfer *>(buf);
    *tr = SingleTransfer{}; //0 fill.
    tr.bmRequest = 0x80u | type;
    tr.request = request;
    tr.value = value;
    tr.index = index;
    tr.length = len;
    tr.timeOut = 1; //sec?
    tr.endpointAddress = 0;
    tr.bufferOffset = sizeof(SingleTransfer);
    tr.bufferLength = length;
    int ret = ioctl(IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER, &buf, sizeof(buf), &buf, sizeof(buf));
    if((ret < sizeof(SingleTransfer)) || (ret > sizeof(SingleTransfer) + len))
        throw XInterfaceError(i18n("Size mismatch during control transfer."));
    std::copy(buf + sizeof(SingleTransfer), buf + ret, rdbuf);
    return ret;
}


XString
CyUSB3Device::getString(int descid) {
    uint8_t buf[2];
    //Reads common descriptor.
    controlRead(CtrlReq::GET_DESC,
        CtrlReqType::STD, USB_STRING_DESCRIPTOR_TYPE * 0x100u + descid,
        LANGID, buf, sizeof(buf));
    int len = buf[0];
    {
        //Reads string descriptor.
        uint8_t buf[len] = {};
        int ret = controlRead(CtrlReq::GET_DESC,
            CtrlReqType::STD, USB_STRING_DESCRIPTOR_TYPE * 0x100u + descid,
            LANGID, buf, len);
        if(ret <= 2)
            throw XInterfaceError(i18n("Size mismatch during control transfer."));
        uint8_t desc_type = buf[1];
        if(desc_type != USB_STRING_DESCRIPTOR_TYPE)
            throw XInterfaceError(i18n("Size mismatch during control transfer."));
        return (char*)&buf[2];
    }
}

CyFXUSBDevice::AsyncIO
CyUSB3Device::asyncBulkWrite(int pipe, const uint8_t *buf, int len) {
    unique_ptr<std::vector<uint8_t>> ioctlbuf(new std::vector<uint8_t>(sizeof(SingleTransfer) + len));
    auto tr = reiterpret_cast<SingleTransfer *>(&ioctlbuf->at(0));
    *tr = SingleTransfer{}; //0 fill.
    std::copy(buf, buf + len, &ioctlbuf->at(sizeof(SingleTransfer)));
    tr.timeOut = 1; //sec?
    tr.endpointAddress = pipe;
    tr.bufferOffset = sizeof(SingleTransfer);
    tr.bufferLength = len;
    auto ret = async_ioctl(IOCTL_ADAPT_SEND_NON_EP0_TRANSFER,
        &ioctlbuf->at(0), ioctlbuf->size(), &ioctlbuf->at(0), ioctlbuf->size());
    ret.ioctlbuf = std::move(ioctlbuf);
    return std::move(ret);
}

CyFXUSBDevice::AsyncIO
CyUSB3Device::asyncBulkRead(int pipe, uint8_t* buf, int len) {
    unique_ptr<std::vector<uint8_t>> ioctlbuf(new std::vector<uint8_t>(sizeof(SingleTransfer) + len));
    auto tr = reiterpret_cast<SingleTransfer *>(&ioctlbuf->at(0));
    *tr = SingleTransfer{}; //0 fill.
    tr.timeOut = 1; //sec?
    tr.endpointAddress = pipe;
    tr.bufferOffset = sizeof(SingleTransfer);
    tr.bufferLength = len;
    auto ret = async_ioctl(IOCTL_ADAPT_SEND_NON_EP0_TRANSFER,
        &ioctlbuf->at(0), ioctlbuf->size(), &ioctlbuf->at(0), ioctlbuf->size());
    ret.ioctlbuf = std::move(ioctlbuf);
    ret.ioctlbuf_rdpos = &ioctlbuf->at(tr.bufferOffset);
    ret.rdbuf = buf;
    return std::move(ret);
}


