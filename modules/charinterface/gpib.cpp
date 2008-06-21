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
#include "gpib.h"

#include <errno.h>

#include <string.h>

#include "support.h"

#ifdef GPIB_WIN32_NI
#include "Decl-32.h"
#endif

#ifdef GPIB_LINUX_NI
#define __inline__  __inline
#include <gpib/ib.h>
#endif

#define MIN_BUF_SIZE 1024

#ifdef GPIB_NI

int
XNIGPIBPort::s_cntOpened = 0;
XMutex
XNIGPIBPort::s_lock;

QString
XNIGPIBPort::gpibStatus(const QString &msg)
{
	QString sta, err, cntl;
	if(ThreadIbsta() & DCAS) sta += "DCAS ";
	if(ThreadIbsta() & DTAS) sta += "DTAS ";
	if(ThreadIbsta() & LACS) sta += "LACS ";
	if(ThreadIbsta() & TACS) sta += "TACS ";
	if(ThreadIbsta() & ATN) sta += "ATN ";
	if(ThreadIbsta() & CIC) sta += "CIC ";
	if(ThreadIbsta() & REM) sta += "REM ";
	if(ThreadIbsta() & LOK) sta += "LOK ";
	if(ThreadIbsta() & CMPL) sta += "CMPL ";
	if(ThreadIbsta() & EVENT) sta += "EVENT ";
	if(ThreadIbsta() & SPOLL) sta += "SPOLL ";
	if(ThreadIbsta() & RQS) sta += "RQSE ";
	if(ThreadIbsta() & SRQI) sta += "SRQI ";
	if(ThreadIbsta() & END) sta += "END ";
	if(ThreadIbsta() & TIMO) sta += "TIMO ";
	if(ThreadIbsta() & ERR) sta += "ERR ";
	switch(ThreadIberr()) {
	case EDVR: err = "EDVR"; break;
	case ECIC: err = "ECIC"; break;
	case ENOL: err = "ENOL"; break;
	case EADR: err = "EADR"; break;
	case EARG: err = "EARG"; break;
	case ESAC: err = "ESAC"; break;
	case EABO: err = "EABO"; break;
	case ENEB: err = "ENEB"; break;
	case EDMA: err = "EDMA"; break;
	case EOIP: err = "EOIP"; break;
	case ECAP: err = "ECAP"; break;
	case EFSO: err = "EFSO"; break;
	case EBUS: err = "EBUS"; break;
	case ESTB: err = "ESTB"; break;
	case ESRQ: err = "ESRQ"; break;
	case ETAB: err = "ETAB"; break;
	default: err = formatString("%u",ThreadIberr()); break;
	}
	if((ThreadIberr() == EDVR) || (ThreadIberr() == EFSO)) {
        char buf[256];
	#ifdef __linux__
        char *s = strerror_r(ThreadIbcntl(), buf, sizeof(buf));
        cntl = formatString("%d",(int)ThreadIbcntl()) + " " + s;
	#else        
        if(strerror_r(ThreadIbcntl(), buf, sizeof(buf))) {
            cntl = formatString("%d",(int)ThreadIbcntl());
        }
        else {
            cntl = formatString("%d",(int)ThreadIbcntl()) + " " + buf;
        }
	#endif
        errno = 0;
	}
	else {
        cntl = formatString("%d",(int)ThreadIbcntl());
	}
	return QString("GPIB %1: addr %2, sta %3, err %4, cntl %5")
		.arg(msg)
		.arg((int)*m_pInterface->address())
		.arg(sta)
		.arg(err)
		.arg(cntl);
}

XNIGPIBPort::XNIGPIBPort(XCharInterface *interface)
	: XPort(interface), m_ud(-1)
{

}
XNIGPIBPort::~XNIGPIBPort()
{
    try {
        gpib_close();
    }
    catch(...) {
    }
} 
void
XNIGPIBPort::open() throw (XInterface::XCommError &)
{
	int port = QString(m_pInterface->port()->to_str()).toInt();
	{
		XScopedLock<XMutex> lock(s_lock);
		if(s_cntOpened == 0) {
			dbgPrint(KAME::i18n("GPIB: Sending IFC"));
			SendIFC (port);
			msecsleep(100);
		}
		s_cntOpened++;
	}
  
	Addr4882_t addrtbl[2];
	int eos = 0;
	if(m_pInterface->eos().length()) {
		eos = 0x1400 + m_pInterface->eos()[m_pInterface->eos().length() - 1];
	}
	m_ud = ibdev(port, 
				 *m_pInterface->address(), 0, T3s, 1, eos);
	if(m_ud < 0) {
		throw XInterface::XCommError(
			gpibStatus(KAME::i18n("opening gpib device faild")), __FILE__, __LINE__);
	}
	ibclr(m_ud);
	ibeos(m_ud, eos);
	addrtbl[0] = *m_pInterface->address();
	addrtbl[1] = NOADDR;
	EnableRemote(port, addrtbl);
}
void
XNIGPIBPort::gpib_close() throw (XInterface::XCommError &)
{
	if(m_ud >= 0) ibonl(m_ud, 0);
	m_ud=-1;
	{
		XScopedLock<XMutex> lock(s_lock);
		s_cntOpened--;
	}
}
void
XNIGPIBPort::gpib_reset() throw (XInterface::XCommError &)
{
    gpib_close();
    msecsleep(100);
    open();
}

void
XNIGPIBPort::send(const char *str) throw (XInterface::XCommError &)
{
	ASSERT(m_pInterface->isOpened());
  
	std::string buf(str);
	buf += m_pInterface->eos();
	ASSERT(buf.length() == strlen(str) + m_pInterface->eos().length());
	this->write(buf.c_str(), buf.length());
}
void
XNIGPIBPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &)
{
	ASSERT(m_pInterface->isOpened());
  
	gpib_spoll_before_write();
  
	for(int i = 0; ; i++)
	{
		msecsleep(m_pInterface->gpibWaitBeforeWrite());
		int ret = ibwrt(m_ud, sendbuf, size);
		if(ret & ERR)
		{
			switch(ThreadIberr()) {
			case EDVR:
			case EFSO:
				if(i < 2) {
					dbgPrint("EDVR/EFSO, try to continue");
					msecsleep(10 * i + 10);
					continue;
				}
				throw XInterface::XCommError(
					gpibStatus(KAME::i18n("too many EDVR/EFSO")), __FILE__, __LINE__);
			}
			gErrPrint(gpibStatus(KAME::i18n("ibwrt err")));
			gpib_reset();
			if(i < 2) {
				gErrPrint(KAME::i18n("try to continue"));
				continue;
			}
			throw XInterface::XCommError(gpibStatus(""), __FILE__, __LINE__);
		}
		if((ret & END) && (ret & CMPL))
		{
			break;
		}
		sendbuf += ThreadIbcntl();
		size -= ThreadIbcntl();        
		if(ret & CMPL) {
			dbgPrint("ibwrt terminated without END");
			continue;
		}
		gErrPrint(gpibStatus(KAME::i18n("ibwrt terminated without CMPL")));
	}
}
void
XNIGPIBPort::receive() throw (XInterface::XCommError &) {
    unsigned int len = gpib_receive(MIN_BUF_SIZE, 1000000uL);
    buffer().resize(len + 1);
    buffer()[len] = '\0';
}
void
XNIGPIBPort::receive(unsigned int length) throw (XInterface::XCommError &)
{
    unsigned int len = gpib_receive(length, length);
    buffer().resize(len);
}

unsigned int
XNIGPIBPort::gpib_receive(unsigned int est_length, unsigned int max_length)
	throw (XInterface::XCommError &) {
	ASSERT(m_pInterface->isOpened());

	gpib_spoll_before_read();
	int len = 0;
	for(int i = 0; ; i++)
	{
		unsigned int buf_size = std::min(max_length, len + est_length);
		if(buffer().size() < buf_size)
			buffer().resize(buf_size);
		msecsleep(m_pInterface->gpibWaitBeforeRead());
		int ret = ibrd(m_ud, &buffer()[len], buf_size - len);
		if(ret & ERR)
		{
			switch(ThreadIberr()) {
			case EDVR:
			case EFSO:
				if(i < 2) {
					dbgPrint("EDVR/EFSO, try to continue");
					msecsleep(10 * i + 10);
					continue;
				}
				throw XInterface::XCommError(
					gpibStatus(KAME::i18n("too many EDVR/EFSO")), __FILE__, __LINE__);
			}
			gErrPrint(gpibStatus(KAME::i18n("ibrd err")));
//              gpib_reset();
			if(i < 2) {
				gErrPrint(KAME::i18n("try to continue"));
				continue;
			}
			throw XInterface::XCommError(gpibStatus(""), __FILE__, __LINE__);
		}
		if(ThreadIbcntl() > buf_size - len)
			throw XInterface::XCommError(gpibStatus(KAME::i18n("libgpib error.")), __FILE__, __LINE__);
		len += ThreadIbcntl();
		if((ret & END) && (ret & CMPL))
		{
			break;
		}
		if(ret & CMPL) {
			if(len == max_length)
				break;
			dbgPrint("ibrd terminated without END");
			continue;
			break;
		}
		gErrPrint(gpibStatus(KAME::i18n("ibrd terminated without CMPL")));
	}
	return len;
}
void
XNIGPIBPort::gpib_spoll_before_read() throw (XInterface::XCommError &)
{
	if(m_pInterface->gpibUseSerialPollOnRead())
	{
		for(int i = 0; ; i++)
		{
			if(i > 30)
			{
				throw XInterface::XCommError(
					gpibStatus(KAME::i18n("too many spoll timeouts")), __FILE__, __LINE__);
			}
			msecsleep(m_pInterface->gpibWaitBeforeSPoll());
			unsigned char spr;
			int ret = ibrsp(m_ud,(char*)&spr);
			if(ret & ERR)
			{
				switch(ThreadIberr()) {
				case EDVR:
				case EFSO:
					dbgPrint("EDVR/EFSO, try to continue");
					msecsleep(10 * i + 10);
					continue;
				}
				gErrPrint(gpibStatus(KAME::i18n("ibrsp err")));
				gpib_reset();
				throw XInterface::XCommError(gpibStatus(KAME::i18n("ibrsp failed")), __FILE__, __LINE__);
			}
			if((spr & m_pInterface->gpibMAVbit()) == 0)
			{
				//MAV isn't detected
				msecsleep(10 * i + 10);
				continue;
			}
            
			break;
		}
	}
}
void 
XNIGPIBPort::gpib_spoll_before_write() throw (XInterface::XCommError &)
{
	if(m_pInterface->gpibUseSerialPollOnWrite())
	{
		for(int i = 0; ; i++)
		{
			if(i > 10)
			{
				throw XInterface::XCommError(
					gpibStatus(KAME::i18n("too many spoll timeouts")), __FILE__, __LINE__);
			}
			msecsleep(m_pInterface->gpibWaitBeforeSPoll());
			unsigned char spr;
			int ret = ibrsp(m_ud,(char*)&spr);
			if(ret & ERR)
			{
				switch(ThreadIberr()) {
				case EDVR:
				case EFSO:
					dbgPrint("EDVR/EFSO, try to continue");
					msecsleep(10 * i + 10);
					continue;
				}
				gErrPrint(gpibStatus(KAME::i18n("ibrsp err")));
				gpib_reset();
				throw XInterface::XCommError(gpibStatus(KAME::i18n("ibrsp failed")), __FILE__, __LINE__);
			}
			if((spr & m_pInterface->gpibMAVbit()))
			{
				//MAV detected
				if(i < 2) {
					msecsleep(5*i + 5);
					continue;
				}
				gErrPrint(gpibStatus(KAME::i18n("ibrd before ibwrt asserted")));
          
				// clear device's buffer
				gpib_receive(MIN_BUF_SIZE, 1000000uL);
				break;
			}

			break;
		}
	}
}


#endif /*GPIB_NI*/
