#include "libusb2cusb.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include "fx2fw.h"

#define FX2_DEF_VID 0x4b4
#define FX2_DEF_PID 0x8613 //cypress default FX2.
#define THAMWAY_VID 0x547
#define THAMWAY_PID 0x1002

#define USB_TIMEOUT 3000

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

int cusblib_initialize() {
    int ret = libusb_init(NULL);
    if(ret) {
        fprintf(stderr, "Error during initialization of libusb: %s\n", libusb_error_name(ret));
        return -1;
    }

    return USBList().size;
}
void cusblib_finalize() {
    libusb_exit(NULL);
}
int usb_open(int n, usb_handle *h) {
    USBList devlist;
    libusb_device *pdev = devlist[n];
    if( !pdev) return -1;

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
        fprintf(stderr, "Error opening dev. #%d in libusb: %s\n", n, libusb_error_name(ret));
       return -1;
    }

    unsigned char manu[256] = {}, prod[256] = {}, serial[256] = {};
    libusb_get_string_descriptor_ascii( *h, desc.iManufacturer, manu, 255);
    libusb_get_string_descriptor_ascii( *h, desc.iProduct, prod, 255);
    libusb_get_string_descriptor_ascii( *h, desc.iSerialNumber, serial, 255);
    fprintf(stderr, "USB %d: PID=%d,VID=%d,BUS#%d,ADDR=%d;%s;%s;%s.\n",
        n, desc.idProduct, desc.idVendor, bus_num, addr, manu, prod, serial);

//    ret = libusb_set_auto_detach_kernel_driver( *h, 1);
//    if(ret) {
//        fprintf(stderr, "USB %d: Warning auto detach is not supported: %s\n", n, libusb_error_name(ret));
//    }
//    ret = libusb_set_configuration( *h, 1);
    ret = libusb_claim_interface( *h, 0);
    if(ret) {
        fprintf(stderr, "USB %d: Error claiming interface: %s\n", n, libusb_error_name(ret));
        return -1;
    }
    ret = libusb_set_interface_alt_setting( *h, 0 , 0 );
    if(ret) {
        fprintf(stderr, "USB %d: Error ALT setting for interface: %s\n", n, libusb_error_name(ret));
        return -1;
    }
    return 0;
}

int usb_close(usb_handle *h) {
    libusb_release_interface( *h,0);
    libusb_close( *h);
    return 0;
}
//Writes the CPUCS register of i8051.
int usb_halt(usb_handle *h) {
    unsigned char byte = 1; //halt
    int ret = libusb_control_transfer( *h,
        LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE, //0x40
        0xA0, //internal
        0xE600, 0x00, &byte, 0x01, USB_TIMEOUT);
    if(ret != 1 ) {
        fprintf(stderr, "Error: FX2 i8051 could not halt: %s\n", libusb_error_name(ret));
        return -1;
    }
    return 0;
}
//Writes the CPUCS register of i8051.
int usb_run(usb_handle *h) {
    unsigned char byte = 0; //run
    int ret = libusb_control_transfer( *h,
       LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE, //0x40
       0xA0, //internal
       0xE600, 0x00, &byte, 0x01, USB_TIMEOUT);
    if(ret != 1 ) {
        fprintf(stderr, "Error: FX2 i8051 could not start: %s\n", libusb_error_name(ret));
        return -1;
    }
    return 0;
}
int usb_dwnload(usb_handle *h, uint8_t *image, int len) {
    int addr = 0;
    //A0 anchor download.
    int ret = libusb_control_transfer( *h,
        LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE, //0x40
        0xA0, //internal
        addr, 0x00, image, len, USB_TIMEOUT);
    if(ret != len) {
        fprintf(stderr, "Error: FX2 write to RAM failed: %s\n", libusb_error_name(ret));
        return -1;
    }
    return 0;
}
int usb_bulk_write(usb_handle *h, int pipe, uint8_t* buf, int len) {
    int transferred;
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

    int ret = libusb_bulk_transfer( *h, LIBUSB_ENDPOINT_OUT | ep, buf, len, &transferred, USB_TIMEOUT);
    if(ret) {
        fprintf(stderr, "Error during USB Bulk writing: %s\n", libusb_error_name(ret));
        return -1;
    }
    return 0;
}
int usb_bulk_read(usb_handle *h, int pipe, uint8_t *buf, int len) {
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
        buf += l; //transferred?
        len -= l; //transferred?
        cnt += transferred;
    }
    return cnt;
}
int usb_get_string(usb_handle *h, int idx, char *s){
    int ret = libusb_get_string_descriptor_ascii( *h, idx, (unsigned char*)s, 127);
    if(ret < 0) {
        fprintf(stderr, "Error during USB get string desc.: %s\n", libusb_error_name(ret));
        return -1;
    }
    s[ret] = '\0';
    return 0;
}

int cusb_init(int n, usb_handle *h, uint8_t* fw, signed char *str1, signed char *str2) {
    char s1[128] = {}, s2[128] = {};
    if(usb_open(n, h)) return -1;
    usb_get_string(h, 1, s1); //may fail.
    fprintf(stderr, "USB: Device: %s\n", s1);
    if( !usb_get_string(h, 2, s2)) {
        fprintf(stderr, "USB: Ver: %s\n", s2);
        if(s2[0] != str2[0]) {
            fprintf(stderr, "USB: Not Thamway's device\n");
            return -1;
        }
    }
    unsigned int version = atoi(s2);
    if(strcmp((const char *)str1,(const char *)s1)|| (version < atoi((char*)str2)) ){
        if(usb_halt(h)) return(-1);
        fprintf(stderr, "USB: Downloading the firmware to the device. This process takes a few seconds....\n");
        if(usb_dwnload(h,fw,CUSB_DWLSIZE)) return(-1);
        if(usb_run(h)) return(-1);
        usb_close(h);
        sleep(1); //for thamway
        if(usb_open(n, h)) {
            fprintf(stderr, "Try again.\n");
            sleep(2); //for thamway
            if(usb_open(n, h)) {
                return -1;
            }
        }
        fprintf(stderr, "USB: successfully downloaded\n");
    }
    return(0);
}
