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
#include "serial.h"

#define TTY_WAIT 1 //ms
#define MIN_BUFFER_SIZE 256

#ifdef SERIAL_POSIX
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

XSerialPort::XSerialPort(XCharInterface *interface)
	: XPort(interface), m_scifd(-1) {

}
XSerialPort::~XSerialPort() {
    if(m_scifd >= 0) close(m_scifd);
}
#endif /*SERIAL_POSIX*/


#ifdef SERIAL_WIN32
#include <windows.h>
#undef PARITY_EVEN
#undef PARITY_ODD
#undef PARITY_NONE

XSerialPort::XSerialPort(XCharInterface *intf)
    : XPort(intf), m_handle(INVALID_HANDLE_VALUE) {
    C_ASSERT(sizeof(void *) == sizeof(HANDLE));
}
XSerialPort::~XSerialPort() {
    if(m_handle != INVALID_HANDLE_VALUE)
        CloseHandle(m_handle);
}
#endif /*SERIAL_WIN32*/

shared_ptr<XPort>
XSerialPort::open(const XCharInterface *pInterface) {
    Snapshot shot( *pInterface);

#ifdef SERIAL_POSIX
	struct termios ttyios;
    speed_t baudrate  = B9600;
    if((m_scifd = ::open(QString(shot[ *pInterface->port()].to_str()).toLocal8Bit().data(),
						 O_RDWR | O_NOCTTY | O_SYNC | O_NONBLOCK)) == -1) {
		throw XInterface::XCommError(i18n("tty open failed"), __FILE__, __LINE__);
	}
    
	tcsetpgrp(m_scifd, getpgrp());
      
	bzero( &ttyios, sizeof(ttyios));
//      tcgetattr(m_scifd, &ttyios);

    if( !m_forceInitialSetting) {
        switch(static_cast<int>(pInterface->serialBaudRate())) {
        case 2400: baudrate = B2400; break;
        case 4800: baudrate = B4800; break;
        case 9600: baudrate = B9600; break;
        case 19200: baudrate = B19200; break;
        case 38400: baudrate = B38400; break;
        case 57600: baudrate = B57600; break;
        case 115200: baudrate = B115200; break;
        case 230400: baudrate = B230400; break;
        default:
            throw XInterface::XCommError(i18n("Invalid Baudrate"), __FILE__, __LINE__);
        }
    }

	cfsetispeed( &ttyios, baudrate);
	cfsetospeed( &ttyios, baudrate);
	cfmakeraw( &ttyios);
	ttyios.c_cflag &= ~(PARENB | CSIZE);
    ttyios.c_cflag |= HUPCL | CLOCAL | CREAD;
    if( !m_forceInitialSetting) {
        if(pInterface->serialParity() == XCharInterface::PARITY_EVEN)
            ttyios.c_cflag |= PARENB;
        if(pInterface->serialParity() == XCharInterface::PARITY_ODD)
            ttyios.c_cflag |= PARENB | PARODD;
        if(pInterface->serial7Bits())
            ttyios.c_cflag |= CS7;
        else
            ttyios.c_cflag |= CS8;
        if(pInterface->serialStopBits() == 2)
            ttyios.c_cflag |= CSTOPB;
        if(pInterface->serialParity() == XCharInterface::PARITY_NONE)
            ttyios.c_iflag |= IGNPAR;
    }
    else {
        ttyios.c_cflag |= CS8;
        ttyios.c_iflag |= IGNPAR;
    }
	ttyios.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); //non-canonical mode
	ttyios.c_iflag |= IGNBRK;
	ttyios.c_cc[VMIN] = 0; //no min. size
	ttyios.c_cc[VTIME] = 30; //3sec time-out
	if(tcsetattr(m_scifd, TCSAFLUSH, &ttyios ) < 0)
		throw XInterface::XCommError(i18n("stty failed"), __FILE__, __LINE__);
	
    if(fcntl(m_scifd, F_SETFL, (~O_NONBLOCK) & fcntl(m_scifd, F_GETFL)) == - 1) {
		throw XInterface::XCommError(i18n("tty open failed"), __FILE__, __LINE__);
	}
#endif /*SERIAL_POSIX*/

#ifdef SERIAL_WIN32
    m_handle = CreateFileA( QString(shot[ *pInterface->port()].to_str()).toLocal8Bit().data(),
        GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (m_handle == INVALID_HANDLE_VALUE)
        throw XInterface::XCommError(i18n("tty open failed"), __FILE__, __LINE__);

    DCB dcb;
    GetCommState(m_handle, &dcb); //loading the original state
    if(m_forceInitialSetting) {
        dcb.BaudRate = 9600;
        dcb.ByteSize = 8;
        dcb.fParity = FALSE;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
    }
    else {
        dcb.BaudRate = static_cast<int>(pInterface->serialBaudRate());
        dcb.ByteSize = pInterface->serial7Bits() ? 7 : 8;
        switch((int)pInterface->serialParity()) {
        case XCharInterface::PARITY_EVEN:
            dcb.fParity = TRUE;
            dcb.Parity = EVENPARITY;
            break;
        case XCharInterface::PARITY_ODD:
            dcb.fParity = TRUE;
            dcb.Parity = ODDPARITY;
            break;
        default:
        case XCharInterface::PARITY_NONE:
            dcb.fParity = FALSE;
            dcb.Parity = NOPARITY;
            break;
        }
        dcb.StopBits = (pInterface->serialStopBits() == 2) ? TWOSTOPBITS : ONESTOPBIT;
    }
    if( !SetCommState(m_handle, &dcb))
        throw XInterface::XCommError(i18n("tty SetCommState failed"), __FILE__, __LINE__);

    COMMTIMEOUTS cto;
    GetCommTimeouts(m_handle, &cto); //loading the original timeout settings.
    cto.ReadIntervalTimeout = 0;
    cto.ReadTotalTimeoutMultiplier = 0;
    cto.ReadTotalTimeoutConstant = 3000;
    cto.WriteTotalTimeoutMultiplier = 0;
    cto.WriteTotalTimeoutConstant = 3000;
    if( !SetCommTimeouts(m_handle, &cto))
        throw XInterface::XCommError(i18n("tty SetCommTimeouts failed"), __FILE__, __LINE__);
#endif /*SERIAL_WIN32*/

    m_serialFlushBeforeWrite = pInterface->serialFlushBeforeWrite();
    m_serialHasEchoBack = pInterface->serialHasEchoBack();

    fprintf(stderr, "Serial port opened w/ baudrate=%d\n", (int)pInterface->serialBaudRate());

    return shared_from_this();
}
void
XSerialPort::send(const char *str) {
	XString buf(str);
    buf += eos();
    if(m_serialHasEchoBack) {
		this->write(str, strlen(str));	//every char should wait for echo back.
		this->write(buf.c_str() + strlen(str), buf.length() - strlen(str)); //EOS
		this->receive(); //wait for EOS.
	}
    else {
        this->write(buf.c_str(), buf.length());
	}
}
void
XSerialPort::write(const char *sendbuf, int size) {
    if(m_serialHasEchoBack && (size >= 2) && isprint(sendbuf[0])) {
		for(int cnt = 0; cnt < size; ++cnt) {
		//sends 1 char.
#ifdef SERIAL_POSIX
			write(sendbuf + cnt, 1);
#endif
#ifdef SERIAL_WIN32
            DWORD wcnt;
            WriteFile(m_handle, sendbuf + cnt, 1, &wcnt, NULL);
#endif
		//waits for echo back.
			for(;;) {
				receive(1);
				if(buffer()[0] == sendbuf[cnt])
					break;
				if(isspace(buffer()[0]))
					continue; //ignores spaces.
				throw XInterface::XCommError(
						formatString("inconsistent echo back %c against %c", buffer()[0], sendbuf[cnt]).c_str(),
						__FILE__, __LINE__);
			}
		}
		return;
	}

    if(m_serialFlushBeforeWrite) {
#ifdef SERIAL_POSIX
		for (;;) {
			int ret = tcflush(m_scifd, TCIFLUSH);
			if(ret < 0) {
				if(errno == EINTR) {
					dbgPrint("Serial, EINTR, try to continue.");
					continue;
				}
				throw XInterface::XCommError(i18n("tciflush error."), __FILE__, __LINE__);
			}
			break;
		}
#endif
#ifdef SERIAL_WIN32
        if( !PurgeComm(m_handle, PURGE_RXCLEAR)) {
                throw XInterface::XCommError(i18n("Serial PurgeComm error"), __FILE__, __LINE__);
        }
#endif
	}
      
	msecsleep(TTY_WAIT);

	int wlen = 0;
	do {
#ifdef SERIAL_POSIX
        int ret = ::write(m_scifd, sendbuf, size - wlen);
        if(ret < 0) {
            if(errno == EINTR) {
                dbgPrint("Serial, EINTR, try to continue.");
                continue;
            }
            else {
				throw XInterface::XCommError(i18n("write error"), __FILE__, __LINE__);
			}
        }
#endif
#ifdef SERIAL_WIN32
        DWORD ret;
        WriteFile(m_handle, sendbuf, size - wlen, &ret, NULL);
        if( !ret)
            throw XInterface::XCommError(i18n("write error"), __FILE__, __LINE__);
#endif
        wlen += ret;
        sendbuf += ret;
	} while (wlen < size);
}
void
XSerialPort::receive() {
//   for(;;) {
//        if(tcdrain(m_scifd) < 0) {
//            dbgPrint("tcdrain failed, continue.");
//            continue;
//        }
//        break;
//   }

	msecsleep(TTY_WAIT);
   
	buffer().resize(MIN_BUFFER_SIZE);
   
    const char *ceos = eos().c_str();
    unsigned int eos_len = strlen(ceos);
	unsigned int len = 0;
	for(;;) {
		if(buffer().size() <= len + 1) 
			buffer().resize(len + MIN_BUFFER_SIZE);
#ifdef SERIAL_POSIX
        int rlen = ::read(m_scifd, &buffer().at(len), 1);
		if(rlen < 0) {
			if(errno == EINTR) {
				dbgPrint("Serial, EINTR, try to continue.");
				continue;
			}
			else
				throw XInterface::XCommError(i18n("read error"), __FILE__, __LINE__);
		}
#endif
#ifdef SERIAL_WIN32
        DWORD rlen;
        ReadFile(m_handle, &buffer().at(len), 1, &rlen, NULL);
#endif
        if(rlen == 0) {
            buffer().at(len) = '\0';
            throw XInterface::XCommError(i18n("read time-out, buf=;") + &buffer().at(0), __FILE__, __LINE__);
        }
		len += rlen;
		if(len >= eos_len) {
            if( !strncmp(&buffer().at(len - eos_len), ceos, eos_len)) {
				break;
			}
		}
	}
    
	buffer().resize(len + 1);
	buffer().at(len) = '\0';
}
void
XSerialPort::receive(unsigned int length) {
//   for(;;) {
//        if(tcdrain(m_scifd) < 0) {
//            dbgPrint("tcdrain failed, continue.");
//            continue;
//        }
//        break;
//   }
   
	msecsleep(TTY_WAIT);
   
	buffer().resize(length);
	unsigned int len = 0;
   
	while(len < length) {
#ifdef SERIAL_POSIX
        int rlen = ::read(m_scifd, &buffer().at(len), 1);
		if(rlen < 0) {
			if(errno == EINTR) {
				dbgPrint("Serial, EINTR, try to continue.");
				continue;
			}
			else
				throw XInterface::XCommError(i18n("read error"), __FILE__, __LINE__);
		}
#endif
#ifdef SERIAL_WIN32
        DWORD rlen;
        ReadFile(m_handle, &buffer().at(len), 1, &rlen, NULL);
#endif
        if(rlen == 0)
            throw XInterface::XCommError(i18n("read time-out"), __FILE__, __LINE__);
		len += rlen;
	}
}    


