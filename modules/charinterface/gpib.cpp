/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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

#ifdef HAVE_LINUX_GPIB
#define __inline__  __inline
#include <gpib/ib.h>
#endif

#ifdef HAVE_NI488
    #if defined WINDOWS || defined __WIN32__
        #define DIRECT_ENTRY_NI488
        static int load_ni4882dll();
        static int free_ni4882dll();
    #endif // WINDOWS || __WIN32__
    inline int strerror_r(int err, char *buf, size_t len) {return strerror_s(buf,len,err); }
    #include <ni4882.h>
    extern "C" {
    static void (__stdcall *pEnableRemote)(int, const Addr4882_t*);
    #define EnableRemote (*pEnableRemote)
    static void (__stdcall *pSendIFC)(int);
    #define SendIFC (*pSendIFC)
    static unsigned long(__stdcall *pThreadIbsta)(void);
    #define ThreadIbsta() (pThreadIbsta())
    static unsigned long(__stdcall *pThreadIberr)(void);
    #define ThreadIberr() (pThreadIberr())
    static unsigned long(__stdcall *pThreadIbcnt)(void);
    #define ThreadIbcnt() (pThreadIbcnt())
    unsigned long (__stdcall *pibclr)(int);
    #define ibclr (*pibclr)
    unsigned long (__stdcall *pibconfig)(int, int, int );
    #define ibconfig (*pibconfig)
    static int(__stdcall *pibdev)(int, int, int, int, int, int);
    #define ibdev (*pibdev)
    static int(__stdcall *pibonl)(int, int);
    #define ibonl (*pibonl)
    unsigned long (__stdcall *pibrd)(int, void *, size_t);
    #define ibrd (*pibrd)
    unsigned long (__stdcall *pibrsp)(int, char *);
    #define ibrsp (*pibrsp)
    unsigned long (__stdcall *pibwrt)(int, const void *, size_t);
    #define ibwrt (*pibwrt)
    }
#endif //HAVE_NI488

#define MIN_BUF_SIZE 1024

#ifdef GPIB_NI

int
XNIGPIBPort::s_cntOpened = 0;
XMutex
XNIGPIBPort::s_lock;

XString
XNIGPIBPort::gpibStatus(const XString &msg) {
	XString sta, err, cntl;
	if(ThreadIbsta() & DCAS) sta += "DCAS ";
	if(ThreadIbsta() & DTAS) sta += "DTAS ";
	if(ThreadIbsta() & LACS) sta += "LACS ";
	if(ThreadIbsta() & TACS) sta += "TACS ";
	if(ThreadIbsta() & ATN) sta += "ATN ";
	if(ThreadIbsta() & CIC) sta += "CIC ";
	if(ThreadIbsta() & REM) sta += "REM ";
	if(ThreadIbsta() & LOK) sta += "LOK ";
	if(ThreadIbsta() & CMPL) sta += "CMPL ";
#ifdef HAVE_LINUX_GPIB
	if(ThreadIbsta() & EVENT) sta += "EVENT ";
	if(ThreadIbsta() & SPOLL) sta += "SPOLL ";
#endif //HAVE_LINUX_GPIB
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
    default: err = formatString("%u",(unsigned int)ThreadIberr()); break;
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
		.arg((int) ***m_pInterface->address())
		.arg(sta)
		.arg(err)
		.arg(cntl);
}

XNIGPIBPort::XNIGPIBPort(XCharInterface *interface)
	: XPort(interface), m_ud(-1) {

}
XNIGPIBPort::~XNIGPIBPort() {
    try {
        gpib_close();
    }
    catch(...) {
    }
} 
void
XNIGPIBPort::open() throw (XInterface::XCommError &) {
	Snapshot shot( *m_pInterface);
	int port = QString(shot[ *m_pInterface->port()].to_str()).toInt();
	{
		XScopedLock<XMutex> lock(s_lock);
		if(s_cntOpened == 0) {
#ifdef DIRECT_ENTRY_NI488
            if(load_ni4882dll())
                throw XInterface::XCommError(
                    gpibStatus(i18n("Loading NI4882.DLL failed.")), __FILE__, __LINE__);
#endif
            dbgPrint(i18n("GPIB: Sending IFC"));
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
				 shot[ *m_pInterface->address()], 0, T3s, 1, eos);
	if(m_ud < 0) {
		throw XInterface::XCommError(
			gpibStatus(i18n("opening gpib device faild")), __FILE__, __LINE__);
	}
	ibclr(m_ud);
	ibeos(m_ud, eos);
	addrtbl[0] = shot[ *m_pInterface->address()];
	addrtbl[1] = NOADDR;
	EnableRemote(port, addrtbl);
}
void
XNIGPIBPort::gpib_close() throw (XInterface::XCommError &) {
	if(m_ud >= 0) ibonl(m_ud, 0);
	m_ud=-1;
	{
		XScopedLock<XMutex> lock(s_lock);
		s_cntOpened--;
#ifdef DIRECT_ENTRY_NI488
        if(s_cntOpened == 0)
            free_ni4882dll();
#endif
	}
}
void
XNIGPIBPort::gpib_reset() throw (XInterface::XCommError &) {
    gpib_close();
    msecsleep(100);
    open();
}

void
XNIGPIBPort::send(const char *str) throw (XInterface::XCommError &) {
	assert(m_pInterface->isOpened());
  
	XString buf(str);
	buf += m_pInterface->eos();
	assert(buf.length() == strlen(str) + m_pInterface->eos().length());
	this->write(buf.c_str(), buf.length());
}
void
XNIGPIBPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &) {
	assert(m_pInterface->isOpened());
  
	gpib_spoll_before_write();
  
	for(int i = 0; ; i++) {
		msecsleep(m_pInterface->gpibWaitBeforeWrite());
		int ret = ibwrt(m_ud, const_cast<char*>(sendbuf), size);
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
					gpibStatus(i18n("too many EDVR/EFSO")), __FILE__, __LINE__);
			}
			gErrPrint(gpibStatus(i18n("ibwrt err")));
			gpib_reset();
			if(i < 2) {
				gErrPrint(i18n("try to continue"));
				continue;
			}
			throw XInterface::XCommError(gpibStatus(""), __FILE__, __LINE__);
		}
		size -= ThreadIbcntl();
		if((size == 0) && (ret & CMPL)) {
			//NI's ibwrt() terminates w/o END.
			break;
		}
		sendbuf += ThreadIbcntl();
		if(ret & CMPL) {
			dbgPrint("ibwrt interrupted.");
			continue;
		}
		gErrPrint(gpibStatus(i18n("ibwrt terminated without CMPL")));
	}
}
void
XNIGPIBPort::receive() throw (XInterface::XCommError &) {
    unsigned int len = gpib_receive(MIN_BUF_SIZE, 1000000uL);
    buffer().resize(len + 1);
    buffer()[len] = '\0';
}
void
XNIGPIBPort::receive(unsigned int length) throw (XInterface::XCommError &) {
    unsigned int len = gpib_receive(length, length);
    buffer().resize(len);
}

unsigned int
XNIGPIBPort::gpib_receive(unsigned int est_length, unsigned int max_length)
	throw (XInterface::XCommError &) {
	assert(m_pInterface->isOpened());

	gpib_spoll_before_read();
	int len = 0;
	for(int i = 0; ; i++) {
		unsigned int buf_size = std::min(max_length, len + est_length);
		if(buffer().size() < buf_size)
			buffer().resize(buf_size);
		msecsleep(m_pInterface->gpibWaitBeforeRead());
		int ret = ibrd(m_ud, &buffer()[len], buf_size - len);
		if(ret & ERR) {
			switch(ThreadIberr()) {
			case EDVR:
			case EFSO:
				if(i < 2) {
					dbgPrint("EDVR/EFSO, try to continue");
					msecsleep(10 * i + 10);
					continue;
				}
				throw XInterface::XCommError(
					gpibStatus(i18n("too many EDVR/EFSO")), __FILE__, __LINE__);
			}
			gErrPrint(gpibStatus(i18n("ibrd err")));
//              gpib_reset();
			if(i < 2) {
				gErrPrint(i18n("try to continue"));
				continue;
			}
			throw XInterface::XCommError(gpibStatus(""), __FILE__, __LINE__);
		}
		if(ThreadIbcntl() > buf_size - len)
			throw XInterface::XCommError(gpibStatus(i18n("libgpib error.")), __FILE__, __LINE__);
		len += ThreadIbcntl();
		if((ret & END) && (ret & CMPL)) {
			break;
		}
		if(ret & CMPL) {
			if(len == max_length)
				break;
			dbgPrint("ibrd terminated without END");
			continue;
		}
		gErrPrint(gpibStatus(i18n("ibrd terminated without CMPL")));
	}
	return len;
}
void
XNIGPIBPort::gpib_spoll_before_read() throw (XInterface::XCommError &) {
	if(m_pInterface->gpibUseSerialPollOnRead())
	{
		for(int i = 0; ; i++)
		{
			if(i > 30)
			{
				throw XInterface::XCommError(
					gpibStatus(i18n("too many spoll timeouts")), __FILE__, __LINE__);
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
				gErrPrint(gpibStatus(i18n("ibrsp err")));
				gpib_reset();
				throw XInterface::XCommError(gpibStatus(i18n("ibrsp failed")), __FILE__, __LINE__);
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
XNIGPIBPort::gpib_spoll_before_write() throw (XInterface::XCommError &) {
	if(m_pInterface->gpibUseSerialPollOnWrite()) {
		for(int i = 0; ; i++) {
			if(i > 10) {
				throw XInterface::XCommError(
					gpibStatus(i18n("too many spoll timeouts")), __FILE__, __LINE__);
			}
			msecsleep(m_pInterface->gpibWaitBeforeSPoll());
			unsigned char spr;
			int ret = ibrsp(m_ud,(char*)&spr);
			if(ret & ERR) {
				switch(ThreadIberr()) {
				case EDVR:
				case EFSO:
					dbgPrint("EDVR/EFSO, try to continue");
					msecsleep(10 * i + 10);
					continue;
				}
				gErrPrint(gpibStatus(i18n("ibrsp err")));
				gpib_reset();
				throw XInterface::XCommError(gpibStatus(i18n("ibrsp failed")), __FILE__, __LINE__);
			}
			if((spr & m_pInterface->gpibMAVbit())) {
				//MAV detected
				if(i < 2) {
					msecsleep(5*i + 5);
					continue;
				}
				gErrPrint(gpibStatus(i18n("ibrd before ibwrt asserted")));
          
				// clear device's buffer
				gpib_receive(MIN_BUF_SIZE, 1000000uL);
				break;
			}

			break;
		}
	}
}

#if defined DIRECT_ENTRY_NI488
#include <windows.h>

static HINSTANCE ni4882dll = NULL;

static int load_ni4882dll() {
    ni4882dll=LoadLibrary(L"NI4882.DLL");
    if(ni4882dll == NULL) {
        return -1;
    }

    pEnableRemote = (void (__stdcall *)
        (int, const Addr4882_t*))GetProcAddress(ni4882dll, "EnableRemote");
    pSendIFC = (void (__stdcall *)(int))GetProcAddress(ni4882dll, "SendIFC");
    pThreadIbsta = (unsigned long (__stdcall *)(void))GetProcAddress(ni4882dll, "ThreadIbsta");
    pThreadIberr = (unsigned long (__stdcall *)(void))GetProcAddress(ni4882dll, "ThreadIberr");
    pThreadIbcnt = (unsigned long (__stdcall *)(void))GetProcAddress(ni4882dll, "ThreadIbcnt");
    pibclr = (unsigned long (__stdcall *)(int)) GetProcAddress(ni4882dll, "ibclr");
    pibconfig = (unsigned long (__stdcall *)
        (int, int, int)) GetProcAddress(ni4882dll, "ibconfig");
    pibdev = (int (__stdcall *)
        (int, int, int, int, int, int)) GetProcAddress(ni4882dll, "ibdev");
    pibonl = (int (__stdcall *)(int, int)) GetProcAddress(ni4882dll, "ibonl");
    pibrd = (unsigned long (__stdcall *)
        (int, void *, size_t)) GetProcAddress(ni4882dll, "ibrd");
    pibrsp = (unsigned long (__stdcall *)
        (int, char*)) GetProcAddress(ni4882dll, "ibrsp");
    pibwrt = (unsigned long (__stdcall *)
        (int, const void *, size_t)) GetProcAddress(ni4882dll, "ibwrt");

    if((pEnableRemote == NULL) || (pSendIFC == NULL) ||
       (pThreadIbsta == NULL) || (pThreadIberr == NULL) || (pThreadIbcnt == NULL) ||
       (pibclr == NULL) || (pibconfig == NULL) || (pibdev == NULL) || (pibonl == NULL) ||
       (pibrd == NULL) || (pibrsp == NULL) || (pibwrt == NULL)) {
        free_ni4882dll();
        ni4882dll = NULL;
        return -1;
    }
    return 0;
}

static int free_ni4882dll() {
    if(ni4882dll != NULL)
        FreeLibrary(ni4882dll);
    return 0;
}
#endif // DIRECT_ENTRY_NI488


#endif /*GPIB_NI*/
