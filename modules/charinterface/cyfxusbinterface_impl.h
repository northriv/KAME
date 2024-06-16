/***************************************************************************
        Copyright (C) 2002-2017 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

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

template <class USBDevice>
XMutex XCyFXUSBInterface<USBDevice>::s_mutex;
template <class USBDevice>
int XCyFXUSBInterface<USBDevice>::s_refcnt = 0;
template <class USBDevice>
typename USBDevice::List XCyFXUSBInterface<USBDevice>::s_devices;

template <class USBDevice>
XCyFXUSBInterface<USBDevice>::XCyFXUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
    : XCustomCharInterface(name, runtime, driver) {

}

template <class USBDevice>
XCyFXUSBInterface<USBDevice>::~XCyFXUSBInterface() {

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
        #if QT_VERSION >= QT_VERSION_CHECK(5,4,0)
            QStandardPaths::locate(QStandardPaths::AppDataLocation, filename);
        #else
            QStandardPaths::locate(QStandardPaths::DataLocation, filename);
        #endif
        if(path.isEmpty()) {
            //for macosx/win
            QDir dir(QApplication::applicationDirPath());
            path = dir.absoluteFilePath(filename);
    #if defined __MACOSX__ || defined __APPLE__
            //For macosx application bundle.
            if( !dir.exists(filename)) {
                //dir is wrong: ".../Contents/MacOS" or  ".../Contents"
                auto relpath = QString("Resources/") + filename;
                path = dir.absoluteFilePath(relpath);
                if( !dir.exists(relpath)) {
                    dir.cdUp();
                    path = dir.absoluteFilePath(relpath);
                }
            }
    #endif
        }
    #endif
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

    gMessagePrint("USB FX: Initializing/opening all the devices");
    s_devices = USBDevice::enumerateDevices();

    bool is_written = false;
    {
        //loads firmware onto RAM, if needed.
        for(auto &&x : s_devices) {
            if( !x) continue;
            try {
                switch(examineDeviceBeforeFWLoad(x)) {
                case DEVICE_STATUS::UNSUPPORTED:
                    x->close();
                    x.reset();
                    continue;
                case DEVICE_STATUS::READY:
                    x->close();
                    continue;
                case DEVICE_STATUS::FW_NOT_LOADED:
                    //firmware is yet to be loaded.
                    x->halt();
                    char fw[FW_DWLSIZE];
                    load_firm(fw, sizeof(fw), firmware(x).c_str());
                    gMessagePrint("USB FX: Downloading the firmware to the device.\nThis process takes a few seconds....");
                    x->downloadFX2((uint8_t*)fw, sizeof(fw));
                    x->run();
                    x->close();
                    is_written = true;
                    break;
                }
            }
            catch (XInterface::XInterfaceError &e) {
                x->close();
                e.print();
                x.reset();
                continue;
            }
        }
    }
    if(is_written) {
        int org_count = s_devices.size();
        for(int retry: {0,1}) {
            msecsleep(2000); //waits for enumeration of reboot devices.
            s_devices = USBDevice::enumerateDevices(); //enumerates devices again.
            if(s_devices.size() >= org_count)
                break;
        }
    }

    for(auto &&x : s_devices) {
        if( !x) continue;
        try {
            if(is_written) {
                switch(examineDeviceBeforeFWLoad(x)) {
                case DEVICE_STATUS::UNSUPPORTED:
                    x.reset();
                    continue;
                case DEVICE_STATUS::READY:
                    if( !gpifWave(x).empty()) {
                        x->open();
                        char gpif[GPIFWAVE_SIZE];
                        load_firm(gpif, sizeof(gpif), gpifWave(x).c_str());
                        setWave(x, (uint8_t*)gpif);
                        x->close();
                    }
                    break;
                case DEVICE_STATUS::FW_NOT_LOADED:
                    gErrPrint("USB FX: firmware download was failed.");
                    x.reset();
                    continue;
                }
            }
        }
        catch (XInterface::XInterfaceError &e) {
            x->close();
            e.print();
            x.reset();
            continue;
        }
    }
    gMessagePrint("USB FX: initialization done.");
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::closeAllEZUSBdevices() {
    s_devices.clear();
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::initialize() {
    control()->setUIEnabled(false);
    device()->setUIEnabled(false);
    m_threadInit.reset(new XThread{shared_from_this(), [this](const atomic<bool>&) {
        XScopedLock<XMutex> slock(s_mutex);
        try {
            if( !(s_refcnt++)) {
                openAllEZUSBdevices();
            }

            for(auto &&x : s_devices) {
                if( !x) continue;
                XString name = examineDeviceAfterFWLoad(x);
                if(name.length()) {
                    auto shot = iterate_commit([=](Transaction &tr){
                        tr[ *device()].add(name);
                    });
                    m_candidates.emplace(name, x);
                }
            }
        }
        catch (XInterface::XInterfaceError &e) {
            e.print();
        }
        control()->setUIEnabled(true);
        device()->setUIEnabled(true);
    }});
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::finalize() {
    this->m_threadInit.reset();
    XScopedLock<XMutex> slock(s_mutex);
    m_usbDevice.reset();
    m_candidates.clear();
    s_refcnt--;
    if( !s_refcnt)
        closeAllEZUSBdevices();
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::open() {
    Snapshot shot( *this);
    auto it = m_candidates.find(shot[ *device()].to_str());
    if(it != m_candidates.end()) {
        m_usbDevice = it->second;
        usb()->openForSharing();
    }
    else {
        throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    }
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::close() {
    if(usb())
        usb()->unref();
    m_usbDevice.reset();
}
