/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
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
#include "dummyport.h"

//---------------------------------------------------------------------------
#define SNPRINT_BUF_SIZE 128

XThreadLocal<std::vector<char> > XPort::s_tlBuffer;


XCharInterface::XCharInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) : 
    XInterface(name, runtime, driver), 
    m_bGPIBUseSerialPollOnWrite(true),
    m_bGPIBUseSerialPollOnRead(true),
    m_gpibWaitBeforeWrite(1),
    m_gpibWaitBeforeRead(2),
    m_gpibWaitBeforeSPoll(1),
    m_gpibMAVbit(0x10),
    m_serialBaudRate(9600),
    m_serialStopBits(2),
    m_script_send(create<XStringNode>("Send", true)),
    m_script_query(create<XStringNode>("Query", true))
{
#ifdef USE_GPIB
	device()->add("GPIB");
#endif
	device()->add("SERIAL");
	device()->add("DUMMY");
  
	m_lsnOnSendRequested = m_script_send->onValueChanged().connectWeak(
		shared_from_this(), &XCharInterface::onSendRequested);
	m_lsnOnQueryRequested = m_script_query->onValueChanged().connectWeak(
		shared_from_this(), &XCharInterface::onQueryRequested);
}
void
XCharInterface::setEOS(const char *str) {
    m_eos = str;
}
         
void
XCharInterface::open() throw (XInterfaceError &)
{        
	m_xport.reset();
    
	{
		shared_ptr<XPort> port;
	#ifdef USE_GPIB
		if(device()->to_str() == "GPIB") {
			port.reset(new XGPIBPort(this));
		}
	#endif
		if(device()->to_str() == "SERIAL") {
			port.reset(new XSerialPort(this));
		}
		if(device()->to_str() == "DUMMY") {
			port.reset(new XDummyPort(this));
		}
          
		if(!port) {
			throw XOpenInterfaceError(__FILE__, __LINE__);
		}
          
		port->open();
		m_xport.swap(port);
	}
}
void
XCharInterface::close() throw (XInterfaceError &)
{
	m_xport.reset();
}
int
XCharInterface::scanf(const char *fmt, ...) const {
	int ret;
	va_list ap;

	va_start(ap, fmt);

	ret = vsscanf(&buffer()[0], fmt, ap);

	va_end(ap);
	return ret;    
}
double
XCharInterface::toDouble() const throw (XConvError &) {
    double x;
    int ret = sscanf(&buffer()[0], "%lf", &x);
    if(ret != 1) throw XConvError(__FILE__, __LINE__);
    return x;
}
int
XCharInterface::toInt() const throw (XConvError &) {
    int x;
    int ret = sscanf(&buffer()[0], "%d", &x);
    if(ret != 1) throw XConvError(__FILE__, __LINE__);
    return x;
}
unsigned int
XCharInterface::toUInt() const throw (XConvError &) {
    unsigned int x;
    int ret = sscanf(&buffer()[0], "%u", &x);
    if(ret != 1) throw XConvError(__FILE__, __LINE__);
    return x;
}

const std::vector<char> &
XCharInterface::buffer() const {return m_xport->buffer();}

void
XCharInterface::send(const std::string &str) throw (XCommError &)
{
    this->send(str.c_str());
}
void
XCharInterface::send(const char *str) throw (XCommError &)
{
	XScopedLock<XCharInterface> lock(*this);
	try {
		dbgPrint(driver()->getLabel() + " Sending:\"" + dumpCString(str) + "\"");
		m_xport->send(str);
	}
	catch (XCommError &e) {
		e.print(driver()->getLabel() + KAME::i18n(" SendError, because "));
		throw e;
	}
}
void
XCharInterface::sendf(const char *fmt, ...) throw (XInterfaceError &)
{
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
XCharInterface::write(const char *sendbuf, int size) throw (XCommError &)
{
	XScopedLock<XCharInterface> lock(*this);
	try {
		dbgPrint(driver()->getLabel() + QString().sprintf(" Sending %d bytes", size));
		m_xport->write(sendbuf, size);
	}
	catch (XCommError &e) {
		e.print(driver()->getLabel() + KAME::i18n(" SendError, because "));
		throw e;
	}
}
void
XCharInterface::receive() throw (XCommError &)
{
	XScopedLock<XCharInterface> lock(*this);
	try {
		dbgPrint(driver()->getLabel() + " Receiving...");
		m_xport->receive();
		ASSERT(buffer().size());
		dbgPrint(driver()->getLabel() + " Received;\"" + 
				 dumpCString((const char*)&buffer()[0]) + "\"");
	}
	catch (XCommError &e) {
        e.print(driver()->getLabel() + KAME::i18n(" ReceiveError, because "));
        throw e;
	}
}
void
XCharInterface::receive(unsigned int length) throw (XCommError &)
{
	XScopedLock<XCharInterface> lock(*this);
	try {
		dbgPrint(driver()->getLabel() + QString(" Receiving %1 bytes...").arg(length));
		m_xport->receive(length);
		dbgPrint(driver()->getLabel() + QString("%1 bytes Received.").arg(buffer().size())); 
	}
	catch (XCommError &e) {
		e.print(driver()->getLabel() + KAME::i18n(" ReceiveError, because "));
		throw e;
	}
}
void
XCharInterface::query(const std::string &str) throw (XCommError &)
{
    query(str.c_str());
}
void
XCharInterface::query(const char *str) throw (XCommError &)
{
	XScopedLock<XCharInterface> lock(*this);
	send(str);
	receive();
}
void
XCharInterface::queryf(const char *fmt, ...) throw (XInterfaceError &)
{
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
void
XCharInterface::onSendRequested(const shared_ptr<XValueNodeBase> &)
{
	shared_ptr<XPort> port = m_xport;
    if(!port)
		throw XInterfaceError(KAME::i18n("Port is not opened."), __FILE__, __LINE__);
    port->send(m_script_send->to_str().c_str());
}
void
XCharInterface::onQueryRequested(const shared_ptr<XValueNodeBase> &)
{
	shared_ptr<XPort> port = m_xport;    
    if(!port)
		throw XInterfaceError(KAME::i18n("Port is not opened."), __FILE__, __LINE__);
    XScopedLock<XCharInterface> lock(*this);
    port->send(m_script_query->to_str().c_str());
    port->receive();
    m_lsnOnQueryRequested->mask();
    m_script_query->value(std::string(&port->buffer()[0]));
    m_lsnOnQueryRequested->unmask();
}

XPort::XPort(XCharInterface *interface)
	: m_pInterface(interface)
{
}
XPort::~XPort()
{
}
