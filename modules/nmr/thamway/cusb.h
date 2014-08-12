//2002.1.14 Ver1.0 by optimize
#include "c:\cypress\usb\drivers\ezusbdrv\ezusbsys.h"

typedef unsigned char u8;
typedef signed char s8;
typedef unsigned short int u16;
typedef signed short int s16;
typedef unsigned int u32;
typedef signed int s32;

#define CUSB_DEBUG   0
#define	CUSB_DWLSIZE 0x2000 //for thamway

s32 usb_open(s32 n,HANDLE *h);
s32 usb_close(HANDLE *h);
s32 usb_halt(HANDLE *h);
s32 usb_run(HANDLE *h);
s32 usb_dwnload(HANDLE *h,u8 *image,s32 len);
s32 usb_resetpipe(HANDLE *h,ULONG p);
s32 usb_bulk_write(HANDLE *h,s32 pipe,u8 *buf,s32 len);
s32 usb_bulk_read(HANDLE *h,s32 pipe,u8 *buf,s32 len);
s32 cusb_init(s32 n,HANDLE *h,u8 *fw,s8 *str1,s8 *str2);
