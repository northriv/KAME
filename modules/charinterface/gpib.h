/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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

#ifdef HAVE_LINUX_GPIB
#define GPIB_NI
#endif

#ifdef HAVE_NI4882
#define GPIB_NI
#endif

#if defined GPIB_WIN32_IF_4304 || defined GPIB_NI
#define USE_GPIB
#endif

#ifdef GPIB_NI

class XNIGPIBPort : public XPort
{
public:
	XNIGPIBPort(XCharInterface *interface);
	virtual ~XNIGPIBPort();
 
    virtual void open(const XCharInterface *pInterface);
    virtual void send(const char *str);
    virtual void write(const char *sendbuf, int size);
    virtual void receive();
    virtual void receive(unsigned int length);

private:
	int m_ud;
    void gpib_close();
    void gpib_open();
    void gpib_reset();
    void gpib_spoll_before_write();
    void gpib_spoll_before_read();
	XString gpibStatus(const XString &msg);
	unsigned int gpib_receive(unsigned int est_length, unsigned int max_length)
		throw (XInterface::XCommError &);
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

#endif /*GPIB_NI*/

#endif /*GPIB_H_*/
