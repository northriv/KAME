#ifndef CSUB2LIBUSB_H
#define CSUB2LIBUSB_H

#include "cyusb.h"

#define	CUSB_DWLSIZE 0x2000

using usb_handle = cyusb_handle*;

int cusblib_initialize();
void cusblib_finalize();

int usb_close(usb_handle *h);
int usb_halt(usb_handle *h);
int usb_run(usb_handle *h);
int usb_dwnload(usb_handle *h, uint8_t* image, int len);
int usb_bulk_write(usb_handle *h, int pipe, uint8_t *buf, int len);
int usb_bulk_read(usb_handle *h, int pipe, uint8_t* buf, int len);
int cusb_init(int n, usb_handle *h, uint8_t *fw, signed char *str1, signed char *str2);

#endif // CSUB2LIBUSB_H
