/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef GPIB_H_
#define GPIB_H_

#include "charinterface.h"

#define USE_GPIB

#include "serial.h"
//Prologix GPIB USB
class XPrologixGPIBPort : public XAddressedPort<XSerialPortWithInitialSetting> {
public:
    XPrologixGPIBPort(XCharInterface *interface) : XAddressedPort<XSerialPortWithInitialSetting>(interface) {}
    virtual ~XPrologixGPIBPort() {}

    virtual shared_ptr<XPort> open(const XCharInterface *pInterface) override;

    virtual void sendTo(XCharInterface *intf, const char *str) override;
    virtual void writeTo(XCharInterface *intf, const char *sendbuf, int size) override;
    virtual void receiveFrom(XCharInterface *intf) override;
    virtual void receiveFrom(XCharInterface *intf, unsigned int length) override;
private:
    void setupAddrEOSAndSend(XCharInterface *intf, std::string cmd);
    unsigned int m_lastAddr = 256u;
    void gpib_spoll_before_read(XCharInterface *intf);
};

#ifdef HAVE_LINUX_GPIB
#define GPIB_NI
#endif

#ifdef HAVE_NI4882
#define GPIB_NI
#endif

#ifdef GPIB_NI

class XNIGPIBPort : public XPort
{
public:
	XNIGPIBPort(XCharInterface *interface);
	virtual ~XNIGPIBPort();
 
    virtual shared_ptr<XPort> open(const XCharInterface *pInterface) override;
    virtual void send(const char *str) override;
    virtual void write(const char *sendbuf, int size) override;
    virtual void receive() override;
    virtual void receive(unsigned int length) override;

private:
	int m_ud;
    void gpib_close();
    void gpib_open();
    void gpib_reset();
    void gpib_spoll_before_write();
    void gpib_spoll_before_read();
	XString gpibStatus(const XString &msg);
    unsigned int gpib_receive(unsigned int est_length, unsigned int max_length);
	static int s_cntOpened;
	static XMutex s_lock;

    bool m_bGPIBUseSerialPollOnWrite;
    bool m_bGPIBUseSerialPollOnRead;
    int m_gpibWaitBeforeWrite;
    int m_gpibWaitBeforeRead;
    int m_gpibWaitBeforeSPoll;
    unsigned char m_gpibMAVbit; //! don't check if zero

    unsigned int m_address;
};

typedef XNIGPIBPort XGPIBPort;

#else
    using XGPIBPort = XPrologixGPIBPort;
#endif /*GPIB_NI*/

#endif /*GPIB_H_*/
