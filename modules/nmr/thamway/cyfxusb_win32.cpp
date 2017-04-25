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

namespace CyUSB {


}
typedef struct _ISO_ADV_PARAMS{
#define DIR_HOST_TO_DEVICE 0
#define DIR_DEVICE_TO_HOST 1

#define DEVICE_SPEED_UNKNOWN        0x00000000
#define DEVICE_SPEED_LOW_FULL       0x00000001
#define DEVICE_SPEED_HIGH           0x00000002
#define DEVICE_SPEED_SUPER			0x00000004



#define USB_ISO_ID                  0x4945
#define USB_ISO_CMD_ASAP            0x8000
#define USB_ISO_CMD_CURRENT_FRAME   0x8001
#define USB_ISO_CMD_SET_FRAME       0x8002

typedef struct _ISO_ADV_PARAMS {

    USHORT isoId;
    USHORT isoCmd;

    ULONG ulParam1;
    ULONG ulParam2;

} ISO_ADV_PARAMS, *PISO_ADV_PARAMS;

typedef struct _ISO_PACKET_INFO {
    ULONG Status;
    ULONG Length;
} ISO_PACKET_INFO, *PISO_PACKET_INFO;



#endif // #ifndef DRIVER

typedef struct _SET_TRANSFER_SIZE_INFO {
    UCHAR EndpointAddress;
    ULONG TransferSize;
} SET_TRANSFER_SIZE_INFO, *PSET_TRANSFER_SIZE_INFO;


//
// Macro to extract function out of the device io control code
//
#ifdef WIN_98_DDK
#define DEVICE_TYPE_FROM_CTL_CODE(ctrlCode)     (((ULONG)(ctrlCode & 0xffff0000)) >> 16)
#endif
#define FUNCTION_FROM_CTL_CODE(ctrlCode)     (((ULONG)(ctrlCode & 0x00003FFC)) >> 2)
#define ACCESS_FROM_CTL_CODE(ctrlCode)       (((ULONG)(ctrlCode & 0x000C0000)) >> 14)
//#define METHOD_FROM_CTL_CODE(ctrlCode)       (((ULONG)(ctrlCode & 0x00000003)))


#define IOCTL_ADAPT_INDEX 0x0000

// Get the driver version
#define IOCTL_ADAPT_GET_DRIVER_VERSION         CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Get the current USBDI version
#define IOCTL_ADAPT_GET_USBDI_VERSION         CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+1, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Get the current device alt interface settings from driver
#define IOCTL_ADAPT_GET_ALT_INTERFACE_SETTING CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+2, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Set the device interface and alt interface setting
#define IOCTL_ADAPT_SELECT_INTERFACE          CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+3, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Get device address from driver
#define IOCTL_ADAPT_GET_ADDRESS               CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+4, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Get number of endpoints for current interface and alt interface setting from driver
#define IOCTL_ADAPT_GET_NUMBER_ENDPOINTS      CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+5, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Get the current device power state
#define IOCTL_ADAPT_GET_DEVICE_POWER_STATE    CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+6,   METHOD_BUFFERED, FILE_ANY_ACCESS)

// Set the device power state
#define IOCTL_ADAPT_SET_DEVICE_POWER_STATE    CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+7,   METHOD_BUFFERED, FILE_ANY_ACCESS)

// Send a raw packet to endpoint 0
#define IOCTL_ADAPT_SEND_EP0_CONTROL_TRANSFER CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+8, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Send/receive data to/from nonep0
#define IOCTL_ADAPT_SEND_NON_EP0_TRANSFER     CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+9, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Simulate a disconnect/reconnect
#define IOCTL_ADAPT_CYCLE_PORT                CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+10, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Reset the pipe
#define IOCTL_ADAPT_RESET_PIPE                CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+11, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Reset the device
#define IOCTL_ADAPT_RESET_PARENT_PORT         CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+12, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Get the current transfer size of an endpoint (in number of bytes)
#define IOCTL_ADAPT_GET_TRANSFER_SIZE         CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+13, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Set the transfer size of an endpoint (in number of bytes)
#define IOCTL_ADAPT_SET_TRANSFER_SIZE         CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+14, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Return the name of the device
#define IOCTL_ADAPT_GET_DEVICE_NAME           CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+15, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Return the "Friendly Name" of the device
#define IOCTL_ADAPT_GET_FRIENDLY_NAME         CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+16, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Abort all outstanding transfers on the pipe
#define IOCTL_ADAPT_ABORT_PIPE                CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+17, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Send/receive data to/from nonep0 w/ direct buffer acccess (no buffering)
#define IOCTL_ADAPT_SEND_NON_EP0_DIRECT       CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+18, METHOD_NEITHER, FILE_ANY_ACCESS)

// Return device speed
#define IOCTL_ADAPT_GET_DEVICE_SPEED          CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+19, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Get the current USB frame number
#define IOCTL_ADAPT_GET_CURRENT_FRAME         CTL_CODE(FILE_DEVICE_UNKNOWN, IOCTL_ADAPT_INDEX+20, METHOD_BUFFERED, FILE_ANY_ACCESS)

#define NUMBER_OF_ADAPT_IOCTLS 21 // Last IOCTL_ADAPT_INDEX + 1

};

using CyFXUSBDevice::AsyncIO::Transfer = OVERLAPPED;

template <class tHANDLE>
CyFXUSBDevice::AsyncIO::AsyncIO(tHANDLE h) : m_transfer(new OVERLAPPED{}), m_count_imm(-1), handle(h) {
}
bool
CyFXUSBDevice::AsyncIO::hasFinished() {
    return HasOverlappedIoCompleted(ptr());
}
int64_t
CyFXUSBDevice::AsyncIO::waitFor() {
    if(m_count_imm) return m_count_imm;
    DWORD num;
    GetOverlappedResult((HANDLE)handle, ptr(), &num, true);
    finalize(num);
    return num;
}
CyFXUSBDevice::ASyncIO
CyFXWin32USBDevice::async_ioctl(uint64_t code, const void *in, ssize_t size_in, void *out, ssize_t size_out) {
    DWORD nbyte;
    AsyncIO async(handle);
    if( !DeviceIoControl(handle, code, in, size_in, out, size_out, &nbyte, async.ptr())) {
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


int
CyFXUSBDevice::vendorReq() {


    bmRequest.Recipient = 0; // Device
    bmRequest.Type = 2; // Vendor
    bmRequest.Direction = 1; // IN command (from Device to Host)
    //bmreq = 0xc0; //vendor request in.



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
void
CyFXUSBDevice::halt() {
    vendorRequestIn(0xA0, 0xe600, 0x00, 0x01, 1);
}

void
CyFXUSBDevice::run() {
    vendorRequestIn(0xA0, 0xe600, 0x00, 0x01, 0);
}

XString
CyFXUSBDevice::getString(const USBDevice &dev, int descid);
void
CyFXUSBDevice::download(const USBDevice &dev, uint8_t* image, int len);
int
CyFXUSBDevice::bulkWrite(int pipe, uint8_t *buf, int len);
int
CyFXUSBDevice::bulkRead(int pipe, const uint8_t* buf, int len);

unsigned int
CyFXUSBDevice::vendorID();
unsigned int
CyFXUSBDevice::productID();


CyFXUSBDevice::AsyncIO
CyFXUSBDevice::asyncBulkWrite(int pipe, uint8_t *buf, int len);
CyFXUSBDevice::AsyncIO
CyFXUSBDevice::asyncBulkRead(int pipe, const uint8_t *buf, int len);

