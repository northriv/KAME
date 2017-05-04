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
#include <QFile>
#include <QDir>
#include <QApplication>
#include <QStandardPaths>

#define FW_DWLSIZE 0x2000
#define GPIFWAVE_SIZE 172

#ifdef KAME_THAMWAY_USB_DIR
    #define KAME_THAMWAY_USB_DIR ""
#endif

template <class USBDevice>
XMutex XCyFXUSBInterface<USBDevice>::s_mutex;
template <class USBDevice>
int XCyFXUSBInterface<USBDevice>::s_refcnt = 0;
template <class USBDevice>
typename USBDevice::List XCyFXUSBInterface<USBDevice>::s_devices;

template <class USBDevice>
XCyFXUSBInterface<USBDevice>::XCyFXUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
    : XCustomCharInterface(name, runtime, driver) {
    XScopedLock<XMutex> slock(s_mutex);
    try {
        if( !(s_refcnt++))
            openAllEZUSBdevices();

        iterate_commit([=](Transaction &tr){
            for(auto &&x : s_devices) {
                XString name = examineDeviceAfterFWLoad(x);
                x->label = name;
                if(name.length()) {
                    tr[ *device()].add(name);
                }
            }
        });
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}

template <class USBDevice>
XCyFXUSBInterface<USBDevice>::~XCyFXUSBInterface() {
    if(isOpened()) close();

    XScopedLock<XMutex> slock(s_mutex);
    s_refcnt--;
    if( !s_refcnt)
        closeAllEZUSBdevices();
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::openAllEZUSBdevices() {

    QDir dir(QApplication::applicationDirPath());
    auto load_firm = [&dir](char *data, int expected_size, const char *filename){
        QString path =
    #ifdef WITH_KDE
            KStandardDirs::locate("appdata", filename);
    #else
            QStandardPaths::locate(QStandardPaths::DataLocation, filename);
        if(path.isEmpty()) {
            //for macosx/win
            QDir dir(QApplication::applicationDirPath());
    #if defined __MACOSX__ || defined __APPLE__
            //For macosx application bundle.
            dir.cdUp();
    #endif
            QString path2 = KAME_THAMWAY_USB_DIR;
            path2 += filename;
            dir.filePath(path2);
            if(dir.exists())
                path = dir.absoluteFilePath(path2);
        }
    #endif
        dir.filePath(path);
        if( !dir.exists())
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF/firmware file ") +
                filename + i18n_noncontext(" not found."), __FILE__, __LINE__);
        QFile file(dir.absoluteFilePath(path));
        if( !file.open(QIODevice::ReadOnly))
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF/firmware file ") +
                filename + i18n_noncontext(" not found."), __FILE__, __LINE__);
        int size = file.read(data, expected_size);
        if(size != expected_size)
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF/firmware file ") +
                filename + i18n_noncontext(" is not proper."), __FILE__, __LINE__);
    };

    s_devices = USBDevice::enumerateDevices(true);

    {
        //loads firmware onto RAM, if needed.
        bool is_written = false;
        for(auto &&x : s_devices) {
            if( !x) continue;
            try {
                x->open();
                switch(examineDeviceBeforeFWLoad(x)) {
                case DEVICE_STATUS::UNSUPPORTED:
                    x.reset();
                case DEVICE_STATUS::READY:
                    x->close();
                    continue;
                case DEVICE_STATUS::FW_NOT_LOADED:
                    break;
                }
                x->halt();
                char fw[FW_DWLSIZE];
                load_firm(fw, sizeof(fw), firmware(x).c_str());
                fprintf(stderr, "USB: Downloading the firmware to the device. This process takes a few seconds....\n");
                x->downloadFX2((uint8_t*)fw, sizeof(fw));
                x->close();
                is_written = true;
            }
            catch (XInterface::XInterfaceError &e) {
                x->close();
                e.print();
                x.reset();
                continue;
            }
        }
        if(is_written) {
            msecsleep(2000); //waits before enumeration of devices.
            s_devices = USBDevice::enumerateDevices(false); //enumerates devices again.
        }
    }

    for(auto &&x : s_devices) {
        if( !x) continue;
        try {
            x->open();

            char gpif[GPIFWAVE_SIZE];
            load_firm(gpif, sizeof(gpif), gpifWave(x).c_str());
            setWave(x, (uint8_t*)gpif);
        }
        catch (XInterface::XInterfaceError &e) {
            x->close();
            e.print();
            x.reset();
            continue;
        }
    }
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::closeAllEZUSBdevices() {
    shared_ptr<CyFXUSBDevice> lastdev;
    for(auto &&x : s_devices) {
        if( !x) continue;
        x->close();
        lastdev = x;
    }
    lastdev->finalize();
    s_devices.clear();
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::open() throw (XInterfaceError &) {
    Snapshot shot( *this);
    try {
        for(auto &&x : s_devices) {
            if( !x) continue;
            if(shot[ *device()].to_str() == x->label)
                m_usbDevice = dynamic_pointer_cast<USBDevice>(x);
        }
        if( !m_usbDevice)
            throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    }
    catch (XInterface::XInterfaceError &e) {
        m_usbDevice.reset();
        throw e;
    }
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::close() throw (XInterfaceError &) {
    m_usbDevice.reset();
}
