/*** FX2FW Ver1.00 by OPTIMIZE ***/
#include <windows.h>
#include <stdio.h>
#include <process.h>
#include <winioctl.h>

#include "cusb.h"

s32 usb_open(s32 n,HANDLE *h,HANDLE *hmutex){
    HANDLE th;
    s32 i;
    char name[256],muname[32];
    if(n==-1){
        for(i=0;i<8;i++){
//            sprintf(muname,"Ezusb-%d",i);
            if((th=OpenMutexA(MUTEX_ALL_ACCESS,TRUE,muname))!=NULL){
                CloseHandle(th);
                continue;
            }
            sprintf(name,"\\\\.\\Ezusb-%d",i);
            *h = CreateFileA(name,
                GENERIC_READ | GENERIC_WRITE, //according to thamway
                FILE_SHARE_READ | FILE_SHARE_WRITE, //according to thamway
                0,
                OPEN_EXISTING,
                0,
                0);
            if(*h == INVALID_HANDLE_VALUE) {
                int e = (int)GetLastError();
                if(e != 2)
                    fprintf(stderr, "cusb: INVALID HANDLE %d for %s\n", e, name);
                continue;
            }
            else{
                *hmutex=CreateMutexA(NULL,FALSE,muname);
                break;
            }
        }
        if(i==8) return(-1);
    }
    else{
        sprintf(name,"\\\\.\\Ezusb-%d",n);
        fprintf(stderr, "cusb: opening device: %s\n", name);
        *h = CreateFileA(name,
            GENERIC_READ | GENERIC_WRITE, //according to thamway
            FILE_SHARE_READ | FILE_SHARE_WRITE, //according to thamway
            0,
            OPEN_EXISTING,
            0,
            0);
        if(*h == INVALID_HANDLE_VALUE) {
            int e = (int)GetLastError();
            if(e != ERROR_FILE_NOT_FOUND)
                fprintf(stderr, "cusb: INVALID HANDLE %d for %s\n", e, name);
            return(-1);
        }
    }
    return(0);
}

s32 usb_close(HANDLE *h,HANDLE *hmutex){
   CloseHandle(*h);
   if( *hmutex){
       CloseHandle( *hmutex);
   }
   return(0);
}

s32 usb_halt(HANDLE *h){
    unsigned long nbyte;
    BOOLEAN ret = FALSE;
    VENDOR_REQUEST_IN vreq;

    vreq.bRequest = 0xA0;
    vreq.wValue = 0xe600;
    vreq.wIndex = 0x00;
    vreq.wLength = 0x01;
    vreq.bData = 1;
    vreq.direction = 0x00;
    ret = DeviceIoControl (*h,
                            IOCTL_Ezusb_VENDOR_REQUEST,
                            &vreq,
                            sizeof(VENDOR_REQUEST_IN),
                            NULL,
                            0,
                            &nbyte,
                            NULL);
    if(ret==FALSE){
        printf("i8051 halt err.\n");
        return(-1);
    }
    return(0);
}

s32 usb_run(HANDLE *h){
    unsigned long nbyte;
    BOOLEAN ret = FALSE;
    VENDOR_REQUEST_IN vreq;

    vreq.bRequest = 0xA0;
    vreq.wValue = 0xe600;
    vreq.wIndex = 0x00;
    vreq.wLength = 0x01;
    vreq.bData = 0;
    vreq.direction = 0x00;
    ret = DeviceIoControl (*h,
                            IOCTL_Ezusb_VENDOR_REQUEST,
                            &vreq,
                            sizeof(VENDOR_REQUEST_IN),
                            NULL,
                            0,
                            &nbyte,
                            NULL);
    if(ret==FALSE){
        printf("i8051 run err.\n");
        return(-1);
    }
    return(0);
}

s32 usb_dwnload(HANDLE *h,u8 *image,s32 len){
    unsigned long nbyte;
    BOOLEAN ret = FALSE;

    ret = DeviceIoControl (*h,
                            IOCTL_Ezusb_ANCHOR_DOWNLOAD,
                            image,
                            len,
                            NULL,
                            0,
                            &nbyte,
                            NULL);
    if(ret==FALSE){
        printf("usb dwnload err.\n");
        return(-1);
    }
    return(0);
}

s32 usb_resetpipe(HANDLE *h,ULONG p){
    unsigned long nbyte;
    BOOLEAN ret = FALSE;

    ret = DeviceIoControl (*h,
                            IOCTL_Ezusb_RESETPIPE,
                            &p,
                            sizeof(ULONG),
                            NULL,
                            0,
                            &nbyte,
                            NULL);
    if(ret==FALSE){
        return(-1);
    }
    return(0);
}

s32 usb_get_string(HANDLE *h,s32 idx,s8 *s){
    unsigned long nbyte;
    s32 i;
    u8  pvbuf[2+128];
    GET_STRING_DESCRIPTOR_IN sin;
    BOOLEAN ret = FALSE;

    sin.Index=idx;
    sin.LanguageId=27;
    ret = DeviceIoControl (*h,
                            IOCTL_Ezusb_GET_STRING_DESCRIPTOR,
                            &sin,
                            sizeof(GET_STRING_DESCRIPTOR_IN),
                            pvbuf,
                            sizeof (pvbuf),
                            &nbyte,
                            NULL);
    if(ret==FALSE){
        fprintf(stderr, "cusb: ioctl failed %d\n", (int)GetLastError());
        return(-1);
    }
    for(i=0;i<pvbuf[0]/2-1;i++){
        *(s++)=(s8)pvbuf[2+i*2];
    }
    *s=0;
    return(0);
}

s32 usb_bulk_write(HANDLE *h,s32 pipe,u8 *buf,s32 len){
    unsigned long nbyte;
    s32 i,l;
    BOOLEAN ret = FALSE;
    BULK_TRANSFER_CONTROL bulk_control;

    bulk_control.pipeNum = pipe;

//    fprintf(stderr, "cusb: bulkwrite %d len=%d\n", (int)pipe, (int)len);
    for(i=0;len>0;){
        if(len>0x8000){
            l=0x8000;
        }
        else{
            l=len;
        }
        ret = DeviceIoControl (*h,
                            IOCTL_EZUSB_BULK_WRITE,
                            &bulk_control,
                            sizeof(BULK_TRANSFER_CONTROL),
                            buf+i,
                            l,
                            &nbyte,
                            NULL);
        if(ret==FALSE){
            fprintf(stderr, "cusb: ioctl failed %d\n", (int)GetLastError());
            return(-1);
        }
        i+=l;
        len-=l;
    }
    return(0);
}

s32 usb_bulk_read(HANDLE *h,s32 pipe,u8 *buf,s32 len){
    unsigned long nbyte;
    s32 i,l,cnt;
    BOOLEAN ret = FALSE;
    BULK_TRANSFER_CONTROL bulk_control;

    bulk_control.pipeNum = pipe;
    for(i=cnt=0;len>0;){
        if(len>0x8000){
            l=0x8000;
        }
        else{
            l=len;
        }
        ret = DeviceIoControl (*h,
                            IOCTL_EZUSB_BULK_READ,
                            &bulk_control,
                            sizeof(BULK_TRANSFER_CONTROL),
                            buf+i,
                            l,
                            &nbyte,
                            NULL);
        if(ret==FALSE){
            fprintf(stderr, "cusb: ioctl failed %d\n", (int)GetLastError());
            return(-1);
        }
        i+=l;
        len-=l;
        cnt+=nbyte;
    }
    return(cnt);
}

s32 cusb_init(s32 n,HANDLE *h,HANDLE *hmutex,u8 *fw,s8 *str1,s8 *str2){
    s8 s1[128],s2[128];
    s1[0] = '\0'; s2[0] = '\0';
    if(usb_open(n,h,hmutex)) return(-1);
    fprintf(stderr, "Ez-USB: cusb: successfully opened\n");
    if(usb_get_string(h,1,s1)) {}// return(-1);
    fprintf(stderr, "cusb: Device: %s\n", s1);
    if(usb_get_string(h,2,s2)) {} //return(-1);
    {
        fprintf(stderr, "cusb: Ver: %s\n", s2);
        if(s2[0] != str2[0]) {
            fprintf(stderr, "cusb: Not Thamway's device\n");
            return -1;
        }
    }
    unsigned int version = atoi(s2);
    if(strcmp((const char *)str1,(const char *)s1)|| (version < atoi(str2)) ){
        if(usb_halt(h)) return(-1);
        fprintf(stderr, "cusb: Downloading the firmware to the device. This process takes a few seconds....\n");
        if(usb_dwnload(h,fw,CUSB_DWLSIZE)) return(-1);
        if(usb_run(h)) return(-1);
        usb_close(h,hmutex);
        for(;;){
            Sleep(2500); //for thamway
            if(usb_open(n,h,hmutex) == 0) {
                Sleep(20);
                break;
            }
        }
        fprintf(stderr, "Ez-USB: cusb: successfully downloaded\n");
    }
    return(0);
}
