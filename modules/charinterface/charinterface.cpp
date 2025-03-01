/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "charinterface.h"
//---------------------------------------------------------------------------
#include "measure.h"
#include "xnodeconnector.h"
#include <string>
#include <stdarg.h>
#include "driver.h"
#include "gpib.h"
#include "serial.h"
#include "tcp.h"
#include "dummyport.h"

//---------------------------------------------------------------------------
#define SNPRINT_BUF_SIZE 128

XThreadLocal<std::vector<char> > XCustomCharInterface::s_tlBuffer;

std::deque<weak_ptr<XPort>> XPort::s_openedPorts; //should be guarded by s_mutex.
XMutex XPort::s_mutex;

XCustomCharInterface::XCustomCharInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
    XInterface(name, runtime, driver) {

}
void
XCustomCharInterface::setEOS(const char *str) {
    m_eos = str;
}

int
XCustomCharInterface::scanf(const char *fmt, ...) const {
    if( !buffer().size())
        throw XConvError(__FILE__, __LINE__);
    bool addednull = false;
    if(buffer().back() != '\0') {
        buffer_receive().push_back('\0');
        addednull = true;
    }

    int ret;
    va_list ap;

    va_start(ap, fmt);

    ret = vsscanf( &buffer()[0], fmt, ap);

    va_end(ap);

    if(addednull)
        buffer_receive().pop_back();
    return ret;
}
double
XCustomCharInterface::toDouble() const {
    double x;
    int ret = this->scanf("%lf", &x);
    if(ret != 1)
        throw XConvError(__FILE__, __LINE__);
    return x;
}
int
XCustomCharInterface::toInt() const {
    int x;
    int ret = this->scanf("%d", &x);
    if(ret != 1)
        throw XConvError(__FILE__, __LINE__);
    return x;
}
unsigned int
XCustomCharInterface::toUInt() const {
    unsigned int x;
    int ret = this->scanf("%u", &x);
    if(ret != 1)
        throw XConvError(__FILE__, __LINE__);
    return x;
}

XString
XCustomCharInterface::toStr() const {
    return XString( &buffer()[0]);
}
XString
XCustomCharInterface::toStrSimplified() const {
    return QString( &buffer()[0]).simplified();
}
void
XCustomCharInterface::query(const XString &str) {
    query(str.c_str());
}

void
XCustomCharInterface::sendf(const char *fmt, ...) {
    va_list ap;
    int buf_size = SNPRINT_BUF_SIZE;
    std::vector<char> buf;
    for(;;) {
        buf.resize(buf_size);
        int ret;

        va_start(ap, fmt);

        ret = vsnprintf(&buf[0], buf_size, fmt, ap);

        va_end(ap);

        if(ret < 0) throw XConvError(__FILE__, __LINE__);
        if(ret < buf_size) break;

        buf_size *= 2;
    }

    this->send(&buf[0]);
}

void
XCustomCharInterface::query(const char *str) {
    XScopedLock<XInterface> lock(*this);
    send(str);
    receive();
}
void
XCustomCharInterface::queryf(const char *fmt, ...) {
    va_list ap;
    int buf_size = SNPRINT_BUF_SIZE;
    std::vector<char> buf;
    for(;;) {
        buf.resize(buf_size);
        int ret;

        va_start(ap, fmt);

        ret = vsnprintf(&buf[0], buf_size, fmt, ap);

        va_end(ap);

        if(ret < 0) throw XConvError(__FILE__, __LINE__);
        if(ret < buf_size) break;

        buf_size *= 2;
    }

    this->query(&buf[0]);
}


XCharInterface::XCharInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) : 
    XCustomCharInterface(name, runtime, driver),
    m_serialEOS("\n"),
    m_bGPIBUseSerialPollOnWrite(true),
    m_bGPIBUseSerialPollOnRead(true),
    m_gpibWaitBeforeWrite(1),
    m_gpibWaitBeforeRead(2),
    m_gpibWaitBeforeSPoll(1),
    m_gpibMAVbit(0x10),
    m_serialBaudRate(9600),
    m_serialStopBits(2),
    m_serialParity(PARITY_NONE),
    m_serial7Bits(false),
    m_serialFlushBeforeWrite(true),
    m_serialHasEchoBack(false),
    m_script_send(create<XStringNode>("Send", true)),
    m_script_query(create<XStringNode>("Query", true)) {

	iterate_commit([=](Transaction &tr){
	#ifdef USE_GPIB
        if(typeid(XGPIBPort) == typeid(XPrologixGPIBPort))
            tr[ *device()].add("GPIB");
        else
            tr[ *device()].add({"GPIB", "PrologixGPIBUSB"});
    #endif
    #ifdef USE_SERIAL
        tr[ *device()].add("SERIAL");
    #endif
    #ifdef USE_TCP
        tr[ *device()].add("TCP/IP");
    #endif
        tr[ *device()].add("DUMMY");
  
        m_lsnOnSendRequested = tr[ *m_script_send].onValueChanged().connectWeakly(
			shared_from_this(), &XCharInterface::onSendRequested);
		m_lsnOnQueryRequested = tr[ *m_script_query].onValueChanged().connectWeakly(
			shared_from_this(), &XCharInterface::onQueryRequested);
    });
}

void XCharInterface::openPort(shared_ptr<XPort> port) {
    m_xport = port->open(this); //for XAddressedPort, returned port may be one already exists.
}
void
XCharInterface::open() {
	m_xport.reset();
    Snapshot shot( *this);
	{
		shared_ptr<XPort> port;
	#ifdef USE_GPIB
		if(shot[ *device()].to_str() == "GPIB") {
			port.reset(new XGPIBPort(this));
            port->setEOS(eos().c_str());
        }
        else if(shot[ *device()].to_str() == "PrologixGPIBUSB") {
            port.reset(new XPrologixGPIBPort(this));
            port->setEOS(eos().c_str());
        }
    #endif
    #ifdef USE_SERIAL
        if(shot[ *device()].to_str() == "SERIAL") {
            port.reset(new XSerialPort(this));
            const char *seos = eos().length() ? eos().c_str() : serialEOS().c_str();
            port->setEOS(seos);
        }
    #endif
    #ifdef USE_TCP
        if(shot[ *device()].to_str() == "TCP/IP") {
			port.reset(new XTCPPort(this));
            port->setEOS(eos().c_str());
        }
    #endif
        if(shot[ *device()].to_str() == "DUMMY") {
			port.reset(new XDummyPort(this));
            port->setEOS(eos().c_str());
        }
          
		if( !port) {
			throw XOpenInterfaceError(__FILE__, __LINE__);
		}
        openPort(port);
	}
}
void
XCharInterface::close() {
	m_xport.reset();
}

void
XCharInterface::send(const XString &str) {
    this->send(str.c_str());
}

void
XCharInterface::send(const char *str) {
    XScopedLock<XCharInterface> lock(*this);
    try {
        auto port = m_xport;
        if( !port)
            throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
        dbgPrint(driver()->getLabel() + " Sending:\"" + dumpCString(str) + "\"");
        XScopedLock<XPort> plock( *port);
        port->sendTo(this, str);
    }
    catch (XCommError &e) {
        e.print(driver()->getLabel() + i18n(" SendError, because "));
        throw e;
    }
}
void
XCharInterface::write(const char *sendbuf, int size) {
    Snapshot shot( *this);
    try {
        auto port = m_xport;
        if( !port)
            throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
        dbgPrint(driver()->getLabel() + formatString(" Sending %d bytes", size));
        XScopedLock<XPort> plock( *port);
        port->writeTo(this, sendbuf, size);
    }
    catch (XCommError &e) {
        e.print(driver()->getLabel() + i18n(" SendError, because "));
        throw e;
    }
}
void
XCharInterface::receive() {
    XScopedLock<XCharInterface> lock(*this);
    try {
        auto port = m_xport;
        if( !port)
            throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
        dbgPrint(driver()->getLabel() + " Receiving...");
        XScopedLock<XPort> plock( *port);
        port->receiveFrom(this);
        assert(buffer().size());
        dbgPrint(driver()->getLabel() + " Received;\"" +
                 dumpCString((const char*)&buffer()[0]) + "\"");
    }
    catch (XCommError &e) {
        e.print(driver()->getLabel() + i18n(" ReceiveError, because "));
        throw e;
    }
}
void
XCharInterface::receive(unsigned int length) {
    XScopedLock<XCharInterface> lock(*this);
    Snapshot shot( *this);
    try {
        auto port = m_xport;
        if( !port)
            throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
        dbgPrint(driver()->getLabel() + QString(" Receiving %1 bytes...").arg(length));
        XScopedLock<XPort> plock( *port);
        port->receiveFrom(this, length);
        dbgPrint(driver()->getLabel() + QString(" %1 bytes Received.").arg(buffer().size()));
    }
    catch (XCommError &e) {
        e.print(driver()->getLabel() + i18n(" ReceiveError, because "));
        throw e;
    }
}
void
XCharInterface::onSendRequested(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XCharInterface> lock( *this);
	if(!isOpened())
		throw XInterfaceError(i18n("Port is not opened."), __FILE__, __LINE__);
    this->send(shot[ *m_script_send].to_str());
}
void
XCharInterface::onQueryRequested(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XCharInterface> lock( *this);
    if(!isOpened())
		throw XInterfaceError(i18n("Port is not opened."), __FILE__, __LINE__);
    this->query(shot[ *m_script_query].to_str());
	iterate_commit([=](Transaction &tr){
		tr[ *m_script_query] = XString(&buffer()[0]);
		tr.unmark(m_lsnOnQueryRequested);
    });
}
void
XCharInterface::lock() {
    XInterface::lock();
    if(isOpened())
        openedPort()->lock();
}
void
XCharInterface::unlock() {
    if(isOpened())
        openedPort()->unlock();
    XInterface::unlock();
}
bool
XCharInterface::isLocked() const {
    return XInterface::isLocked();
}
