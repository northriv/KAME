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

typedef struct _WORD_SPLIT {
    UCHAR lowByte;
    UCHAR hiByte;
} WORD_SPLIT, *PWORD_SPLIT;



typedef struct _SETUP_PACKET {

    union {
        BM_REQ_TYPE bmReqType;
        UCHAR bmRequest;
    };

    UCHAR bRequest;

    union {
        WORD_SPLIT wVal;
        USHORT wValue;
    };

    union {
        WORD_SPLIT wIndx;
        USHORT wIndex;
    };

    union {
        WORD_SPLIT wLen;
        USHORT wLength;
    };

    ULONG ulTimeOut;

} SETUP_PACKET, *PSETUP_PACKET;

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


typedef struct _SINGLE_TRANSFER {
    union {
        SETUP_PACKET SetupPacket;
        ISO_ADV_PARAMS IsoParams;
    };

    UCHAR reserved;

    UCHAR ucEndpointAddress;
    ULONG NtStatus;
    ULONG UsbdStatus;
    ULONG IsoPacketOffset;
    ULONG IsoPacketLength;
    ULONG BufferOffset;
    ULONG BufferLength;
} SINGLE_TRANSFER, *PSINGLE_TRANSFER;

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

}

int
CyFXUSBDevice::ioControl() {

}
int
CyFXUSBDevice::vendorReq() {
    typedef struct _BM_REQ_TYPE {
        UCHAR   Recipient:2;
        UCHAR   Reserved:3;
        UCHAR   Type:2;
        UCHAR   Direction:1;
    } BM_REQ_TYPE, *PBM_REQ_TYPE;

        typedef struct _VENDOR_REQUEST_IN {
            BYTE    bRequest;
            WORD    wValue;
            WORD    wIndex;
            WORD    wLength;
            BYTE    direction;
            BYTE    bData;
        } VENDOR_REQUEST_IN;

    bmRequest.Recipient = 0; // Device
    bmRequest.Type = 2; // Vendor
    bmRequest.Direction = 1; // IN command (from Device to Host)
    //bmreq = 0xc0; //vendor request in.

    typedef struct _SINGLE_TRANSFER { union {
    SETUP_PACKET SetupPacket;
    ISO_ADV_PARAMS IsoParams; };
    UCHAR Reserved;
    UCHAR ucEndpointAddress;
    ULONG NtStatus;
    ULONG UsbdStatus;
    uint32_t IsoPacketOffset;
    ULONG IsoPacketLength;
    ULONG BufferOffset;
    ULONG BufferLength;
    } SINGLE_TRANSFER, *PSINGLE_TRANSFER;

}

CyFXUSBDevice::~CyFXUSBDevice() {
    close();
}
CyFXUSBDevice::List
CyFXUSBDevice::enumerateDevices() {
    CyFXUSBDevice::List list;
    for(int idx = 0; idx < 8; ++idx) {
        XString name = formatString("\\\\.\\Ezusb-%d",n);
        fprintf(stderr, "cusb: opening device: %s\n", name.c_str());
        HANDLE h = CreateFileA(name.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            0,
            OPEN_EXISTING,
            0,
            0); //FILE_FLAG_OVERLAPPED
        if(h == INVALID_HANDLE_VALUE) {
            int e = (int)GetLastError();
            if(e != ERROR_FILE_NOT_FOUND)
                throw XInterfaceError(formatString("cusb: INVALID HANDLE %d for %s\n", e, name));
            if(idx == 0) break;
            s_ezusbActivated = true;
            return std::move(list);
        }
        list.emprace_back({h, d});
    }
    s_ezusbActivated = false;

}
void
CyFXUSBDevice::open();
void
CyFXUSBDevice::close() {
    CloseHandle(handle);
    handle = nullptr;
}

int
CyFXUSBDevice::initialize();
void
CyFXUSBDevice::finalize();
void
CyFXUSBDevice::halt(const USBDevice &dev) {
    unsigned long nbyte;
    VENDOR_REQUEST_IN vreq;
    vreq.bRequest = 0xA0;
    vreq.wValue = 0xe600;
    vreq.wIndex = 0x00;
    vreq.wLength = 0x01;
    vreq.bData = 1;
    vreq.direction = 0x00;
    if(devioctrl(dev.handle,
                            IOCTL_Ezusb_VENDOR_REQUEST,
                            &vreq,
                            sizeof(VENDOR_REQUEST_IN),
                            &nbyte)) {
        throw XInterfaecError("i8051 halt err.\n");
    }
}

void
CyFXUSBDevice::run(const USBDevice &dev);
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

CyFXUSBDevice::AsyncIO::AsyncIO(void *h);
CyFXUSBDevice::AsyncIO::~AsyncIO();
void
CyFXUSBDevice::AsyncIO::waitFor();
void
CyFXUSBDevice::AsyncIO::abort();
CyFXUSBDevice::AsyncIO
CyFXUSBDevice::asyncBulkWrite(int pipe, uint8_t *buf, int len);
CyFXUSBDevice::AsyncIO
CyFXUSBDevice::asyncBulkRead(int pipe, const uint8_t *buf, int len);

