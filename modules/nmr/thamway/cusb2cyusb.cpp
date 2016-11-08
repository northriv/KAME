#include "cusb2cyusb.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>

#define USB_TIMEOUT 3000

int cusblib_initialize() {
    return cyusb_open();
}
void cusblib_finalize() {
    cyusb_close();
}

int usb_close(usb_handle *h) {
}
int usb_halt(usb_handle *h) {
    unsigned char byte = 1;
    int ret = cyusb_control_transfer( *h, 0x40, 0xA0, 0xE600, 0x00, &byte, 0x01, USB_TIMEOUT);
    if(ret != 1 ) {
        fprintf(stderr, "Error: FX2 i8051 could not halt.\n");
        return -1;
    }
    return 0;
}
int usb_run(usb_handle *h) {
    unsigned char byte = 0;
    int ret = cyusb_control_transfer( *h, 0x40, 0xA0, 0xE600, 0x00, &byte, 0x01, USB_TIMEOUT);
    if(ret != 1 ) {
        fprintf(stderr, "Error: FX2 i8051 could not start.\n");
        return -1;
    }
    return 0;
}
int usb_dwnload(usb_handle *h, uint8_t *image, int len) {
    int addr = 0;
    //A0 anchor download.
    int ret = cyusb_control_transfer( *h, 0x40, 0xA0, addr, 0x00, image, len, USB_TIMEOUT);
    if(ret != len) {
        fprintf(stderr, "Error: FX2 write to RAM failed..\n");
        return -1;
    }
    return 0;
}
int usb_bulk_write(usb_handle *h, int pipe, uint8_t* buf, int len) {
    int transferred;
    int ret = cyusb_bulk_transfer( *h, pipe, buf, len, &transferred, USB_TIMEOUT);
    if(ret) {
        cyusb_error(ret);
        return -1;
    }
    return 0;
}
int usb_bulk_read(usb_handle *h, int pipe, uint8_t *buf, int len) {
    int cnt = 0;
    for(int i = 0; len > 0;){
        int l = std::min(len, 0x8000);
        int transferred;
        int ret = cyusb_bulk_transfer( *h, pipe, buf, l, &transferred, USB_TIMEOUT);
        if(ret) {
            cyusb_error(ret);
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
    if(ret < 0)
        return -1;
    s[ret] = '\0';
    return 0;
}

int cusb_init(int n, usb_handle *h, uint8_t* fw, signed char *str1, signed char *str2) {
    char s1[128] = {}, s2[128] = {};
    *h = cyusb_gethandle(n);
    fprintf(stderr, "CyUSB: successfully opened\n");
    usb_get_string(h, 1, s1);
    fprintf(stderr, "CyUSB: Device: %s\n", s1);
    if( !usb_get_string(h, 2, s2)) {
        fprintf(stderr, "CyUSB: Ver: %s\n", s2);
        if(s2[0] != str2[0]) {
            fprintf(stderr, "CyUSB: Not Thamway's device\n");
            return -1;
        }
    }
    unsigned int version = atoi(s2);
    if(strcmp((const char *)str1,(const char *)s1)|| (version < atoi((char*)str2)) ){
        if(usb_halt(h)) return(-1);
        fprintf(stderr, "cusb: Downloading the firmware to the device. This process takes a few seconds....\n");
        if(usb_dwnload(h,fw,CUSB_DWLSIZE)) return(-1);
        if(usb_run(h)) return(-1);
        usleep(1500000); //for thamway
        fprintf(stderr, "CyUSB: successfully downloaded\n");
    }
    return(0);
}
