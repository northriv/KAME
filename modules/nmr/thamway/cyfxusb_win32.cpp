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

constexpr uint32_t IOCTL_EZUSB_GET_STRING_DESCRIPTOR = 0x222044;
//constexpr uint32_t IOCTL_EZUSB_ANCHOR_DOWNLOAD = 0x22201c;
constexpr uint32_t IOCTL_EZUSB_VENDOR_REQUEST = 0x222014;
constexpr uint32_t IOCTL_EZUSB_BULK_WRITE = 0x222051;
constexpr uint32_t IOCTL_EZUSB_BULK_READ = 0x22204e;

constexpr uint32_t IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER = 0x220020;
constexpr uint32_t IOCTL_ADAPT_SEND_NON_EP0_TRANSFER = 0x220024;
constexpr uint32_t IOCTL_ADAPT_SEND_NON_EP0_DIRECT = 0x22004b;

struct CyFXUSBDevice::AsyncIO::Transfer {
    OVERLAPPED overlap;
    HANDLE handle;
    unique_ptr<std::vector<uint8_t>> ioctlbuf;
    void *ioctlbuf_rdpos = nullptr;
    uint8_t *rdbuf = nullptr;
};

CyFXUSBDevice::AsyncIO::AsyncIO(unique_ptr<Transfer>&& t) : m_transfer(std::forward(t)) {
}
bool
CyFXUSBDevice::AsyncIO::hasFinished() {
    return HasOverlappedIoCompleted( &ptr()->overlap);
}
int64_t
CyFXUSBDevice::AsyncIO::waitFor() {
    if(m_count_imm) return m_count_imm;
    DWORD num;
    GetOverlappedResult(ptr()->handle, &ptr()->overlap, &num, true);
    finalize(num);
    if(rdbuf) {
        std::copy(ptr()->ioctlbuf_rdpos, &ptr()->ioctlbuf->at(0) + num, rdbuf);
    }
    return num;
}
CyFXUSBDevice::AsyncIO
CyFXWin32USBDevice::async_ioctl(uint64_t code, const void *in, ssize_t size_in, void *out, ssize_t size_out) {
    DWORD nbyte;
    CyFXUSBDevice::AsyncIO::Transfer tr;
    tr.handle = handle;
    AsyncIO async(tr);
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
CyUSB3Device::controlWrite(uint8_t request, uint16_t value,
                               uint16_t index, const uint8_t *wbuf, int len) {
    uint8_t buf[sizeof(SingleTransfer) + len];
    auto tr = reiterpret_cast<SingleTransfer *>(buf);
    *tr = SingleTransfer{}; //0 fill.
    std::copy(wbuf, wbuf + len, &buf[sizeof(SingleTransfer)]);
    tr.recipient = 0; //Device
    tr.type = 2; //Vendor Request
    tr.direction = 0; //OUT, 0x40
    tr.request = request;
    tr.value = value;
    tr.index = index;
    tr.length = len;
    tr.timeOut = 1; //sec?
    tr.endpointAddress = 0x00;
    tr.bufferOffset = sizeof(SingleTransfer);
    tr.bufferLength = length;
    return ioctl(IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER, &buf, sizeof(buf), &buf, sizeof(buf));
}

int
CyUSB3Device::controlRead(uint8_t request, uint16_t value,
                               uint16_t index, uint8_t *rdbuf, int len) {
    uint8_t buf[sizeof(SingleTransfer) + len];
    auto tr = reiterpret_cast<SingleTransfer *>(buf);
    *tr = SingleTransfer{}; //0 fill.
    tr.recipient = 0; //Device
    tr.type = 2; //Vendor Request
    tr.direction = 1; //IN, 0xc0
    tr.request = request;
    tr.value = value;
    tr.index = index;
    tr.length = len;
    tr.timeOut = 1; //sec?
    tr.endpointAddress = 0x00;
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
    // Get the header to find-out the number of languages, size of lang ID list
    int len = sizeof(USB_COMMON_DESCRIPTOR);
    uint8_t buf[sizeof(SingleTransfer) + len];
    auto tr = reiterpret_cast<SingleTransfer *>(buf);
    *tr = SingleTransfer{}; //0 fill.
    tr.recipient = 1; //HOST
    tr.type = 0; //
    tr.direction = 0; //OUT
    tr.request = USB_REQUEST_GET_DESCRIPTOR;
    tr.value = USB_STRING_DESCRIPTOR_TYPE * 0x100u + descid;
    tr.index = LANGID;
    tr.length = len;
    tr.timeOut = 1; //sec?
    tr.bufferOffset = sizeof(SingleTransfer);
    tr.bufferLength = length;
    int ret = ioctl(IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER, &buf, sizeof(buf), &buf, sizeof(buf));
    if((ret < sizeof(SingleTransfer)) || (ret > sizeof(SingleTransfer) + len))
        throw XInterfaceError(i18n("Size mismatch during control transfer."));

    USB_COMMON_DESCRIPTOR common_desc = {};
    std::copy(buf + sizeof(SingleTransfer), buf + ret, common_desc);

    {
        // Get the entire descriptor
        int len = common_desc.bLength;
        uint8_t buf[sizeof(SingleTransfer) + len];
        auto tr = reiterpret_cast<SingleTransfer *>(buf);
        *tr = SingleTransfer{}; //0 fill.
        tr.recipient = 1; //HOST
        tr.type = 0; //
        tr.direction = 0; //OUT
        tr.request = USB_REQUEST_GET_DESCRIPTOR;
        tr.value = USB_STRING_DESCRIPTOR_TYPE * 0x100u + descid;
        tr.index = LANGID;
        tr.length = len;
        tr.timeOut = 1; //sec?
        tr.bufferOffset = sizeof(SingleTransfer);
        tr.bufferLength = length;
        int ret = ioctl(IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER, &buf, sizeof(buf), &buf, sizeof(buf));
        if((ret < sizeof(SingleTransfer)) || (ret > sizeof(SingleTransfer) + len) ||
                (len <= 2))
            throw XInterfaceError(i18n("Size mismatch during control transfer."));
        uint8_t sig = buf[sizeof(SingleTransfer) + 1];
        if(sig != 3)
            throw XInterfaceError(i18n("Size mismatch during control transfer."));
        buf.back() = '\0';
        return (char*)&buf[sizeof(SingleTransfer) + 2];
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

unsigned int
CyUSB3Device::vendorID();
unsigned int
CyUSB3Device::productID();

