/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
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

    iterate_commit([=](Transaction &tr){
        m_lsnOnItemRefreshRequested = tr[ *device()].onItemRefreshRequested().connectWeakly(
            shared_from_this(), &XCyFXUSBInterface<USBDevice>::onItemRefreshRequested);
    });
}

template <class USBDevice>
XCyFXUSBInterface<USBDevice>::~XCyFXUSBInterface() {

}


template <class USBDevice>
typename USBDevice::List
XCyFXUSBInterface<USBDevice>::pickupNewDev(const typename USBDevice::List &enumerated) {
    auto found = enumerated;
    for(auto it = found.begin(); it != found.end();) {
        for(auto &&y: s_devices) {
            if( *it && y && ( **it == *y)) {//the same device in the static list.
                it->reset();
                break;
            }
        }
        if( !*it)
            it = found.erase(it);
        else
            it++;
    }
    return found;
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

    auto enumerated_devices = USBDevice::enumerateDevices();
    auto found_devices = pickupNewDev(enumerated_devices);
    if( !s_devices.size() && found_devices.size())
        gMessagePrint("USB FX: Initializing/opening all the devices");

    bool is_written = false;
    {
        //loads firmware onto RAM, if needed.
        for(auto &&x : found_devices) {
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
        int org_count = enumerated_devices.size();
        for(int retry: {0,1}) {
            msecsleep(2000); //waits for enumeration of reboot devices.
            enumerated_devices = USBDevice::enumerateDevices(); //enumerates devices again.
            //New USB devices after possible firmware loading.
            found_devices = pickupNewDev(enumerated_devices);
            if(enumerated_devices.size() >= org_count)
                break;
        }

        for(auto &&x : found_devices) {
            if( !x) continue;
            try {
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
            catch (XInterface::XInterfaceError &e) {
                x->close();
                e.print();
                x.reset();
                continue;
            }
        }
    }
    //Adds found devices to the static list.
    s_devices.insert(s_devices.end(), found_devices.begin(), found_devices.end());

    if(is_written)
        gMessagePrint("USB FX: FW initialization done.");
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::closeAllEZUSBdevices() {
    s_devices.clear();
}

//todo XComboBox::onAboutToOpen.
template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::initialize(bool instatiation) {
    m_threadInit.reset(new XThread{shared_from_this(), [this, instatiation](const atomic<bool>&) {
        XScopedLock<XMutex> slock(s_mutex);
        try {
            if(instatiation)
                s_refcnt++;
            openAllEZUSBdevices(); //finds new devices.

            for(auto &&x : s_devices) {
                if( !x) continue;
                //this if statement is always true when instatiation == true.
                if(std::find_if(m_candidates.begin(), m_candidates.end(), [&x](auto &y){return y.second == x;} )
                    == m_candidates.end()) {
                    //yet to be added in the combo box.
                    XString name = examineDeviceAfterFWLoad(x);
                    if(name.length()) {
                        if(m_candidates.find(name) != m_candidates.end())
                            name += formatString(":%u", x->serialNo()); //duplicated name
                        auto shot = iterate_commit([=](Transaction &tr){
                            tr[ *device()].add(name);
                        });
                        m_candidates.emplace(name, x);
                    }
                }
            }
        }
        catch (XInterface::XInterfaceError &e) {
            e.print();
        }
    }});
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::finalize() {    
    this->m_threadInit.reset(); //waiting for thread termination.
    close();
    XScopedLock<XMutex> slock(s_mutex);
    m_candidates.clear();
    s_refcnt--;
    if( !s_refcnt)
        closeAllEZUSBdevices();
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::open() {
    std::string dev_name = Snapshot( *device())[ *device()].to_str();
    for(int retry: {0, 1}) {
        auto it = m_candidates.find(dev_name);
        if(it != m_candidates.end()) {
            auto dev = it->second; //must hold shared_ptr for mutex.
            XScopedLock<XRecursiveMutex> lock(dev->mutex);
            try {
                dev->openForSharing();
            }
            catch (XInterface::XInterfaceError &e) {
                dev->unref();
            //assuming the device has been disconnected.
                { XScopedLock<XMutex> slock(s_mutex);
                    for(auto &&x : s_devices) {
                        if(x == it->second)
                            x.reset();
                    }
                    m_candidates.erase(it);
                    //refreshes the combobox.
                    device()->iterate_commit([=](Transaction &tr){
                        tr[ *device()].clear();
                        for(auto &&x: m_candidates)
                            tr[ *device()].add(x.first);
                    });
                } //unlocks before launching initialization thread.
                if(retry == 0) {
                    initialize(false); //enumerate devices and retries with the same device name for reconnection.
                    m_threadInit.reset(); //waiting for thread termination.
                    trans( *device()).str(dev_name);
                    continue;
                }
                throw e;
            }
            m_usbDevice = dev;
            return; //succeeded.
        }
    }
    throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
}

template <class USBDevice>
void
XCyFXUSBInterface<USBDevice>::close() {
    auto usb__ = usb();
    if(usb__) {
        bool already_locked = usb__->mutex.isLockedByCurrentThread();
        XScopedLock<XRecursiveMutex> lock(usb__->mutex);
        if(usb()) //checks again after acquirring lock.
            usb()->unref();
        m_usbDevice.reset();
        //Usually caller for close() does not need scoped lock, but....
        if(already_locked)
            usb__->mutex.unlock(); //the caller cannot unlock because usb() was lost.
    }
}
