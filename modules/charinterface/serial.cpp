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
#include "serial.h"

#define TTY_WAIT 1 //ms
#define MIN_BUFFER_SIZE 256

#ifdef SERIAL_POSIX
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
 
XPosixSerialPort::XPosixSerialPort(XCharInterface *interface)
	: XPort(interface), m_scifd(-1) {

}
XPosixSerialPort::~XPosixSerialPort() {
    if(m_scifd >= 0) close(m_scifd);
}
void
XPosixSerialPort::open() throw (XInterface::XCommError &) {
	Snapshot shot( *m_pInterface);
	struct termios ttyios;
	speed_t baudrate;
	if((m_scifd = ::open(QString(shot[ *m_pInterface->port()].to_str()).toLatin1().data(),
						 O_RDWR | O_NOCTTY | O_SYNC | O_NONBLOCK)) == -1) {
		throw XInterface::XCommError(i18n("tty open failed"), __FILE__, __LINE__);
	}
    
	tcsetpgrp(m_scifd, getpgrp());
      
	bzero( &ttyios, sizeof(ttyios));
//      tcgetattr(m_scifd, &ttyios);

	switch(static_cast<int>(m_pInterface->serialBaudRate())) {
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

	cfsetispeed( &ttyios, baudrate);
	cfsetospeed( &ttyios, baudrate);
	cfmakeraw( &ttyios);
	ttyios.c_cflag &= ~(PARENB | CSIZE);
	if(m_pInterface->serialParity() == XCharInterface::PARITY_EVEN)
		ttyios.c_cflag |= PARENB;
	if(m_pInterface->serialParity() == XCharInterface::PARITY_ODD)
		ttyios.c_cflag |= PARENB | PARODD;
	if(m_pInterface->serial7Bits())
		ttyios.c_cflag |= CS7;
	else
		ttyios.c_cflag |= CS8;
	ttyios.c_cflag |= HUPCL | CLOCAL | CREAD;
	if(m_pInterface->serialStopBits() == 2)
		ttyios.c_cflag |= CSTOPB;
	ttyios.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); //non-canonical mode
	ttyios.c_iflag |= IGNBRK;
	if(m_pInterface->serialParity() == XCharInterface::PARITY_NONE)
		ttyios.c_iflag |= IGNPAR;
	ttyios.c_cc[VMIN] = 0; //no min. size
	ttyios.c_cc[VTIME] = 30; //3sec time-out
	if(tcsetattr(m_scifd, TCSAFLUSH, &ttyios ) < 0)
		throw XInterface::XCommError(i18n("stty failed"), __FILE__, __LINE__);
	
    if(fcntl(m_scifd, F_SETFL, (~O_NONBLOCK) & fcntl(m_scifd, F_GETFL)) == - 1) {
		throw XInterface::XCommError(i18n("tty open failed"), __FILE__, __LINE__);
	}
}
void
XPosixSerialPort::send(const char *str) throw (XInterface::XCommError &) {
	XString buf(str);
	if(m_pInterface->eos().length())
		buf += m_pInterface->eos();
	else
		buf += m_pInterface->serialEOS();
	if(m_pInterface->serialHasEchoBack()) {
		this->write(str, strlen(str));	//every char should wait for echo back.
		this->write(buf.c_str() + strlen(str), buf.length() - strlen(str)); //EOS
		this->receive(); //wait for EOS.
	}
	else {
		this->write(buf.c_str(), buf.length());
	}
}
void
XPosixSerialPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &) {
	assert(m_pInterface->isOpened());

	if(m_pInterface->serialHasEchoBack() && (size >= 2) && isprint(sendbuf[0])) {
		for(int cnt = 0; cnt < size; ++cnt) {
		//sends 1 char.
			write(sendbuf + cnt, 1);
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

	if(m_pInterface->serialFlushBeforeWrite()) {
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
	}
      
	msecsleep(TTY_WAIT);
   
	int wlen = 0;
	do {
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
        wlen += ret;
        sendbuf += ret;
	} while (wlen < size);
}
void
XPosixSerialPort::receive() throw (XInterface::XCommError &) {
	assert(m_pInterface->isOpened());
    
//   for(;;) {
//        if(tcdrain(m_scifd) < 0) {
//            dbgPrint("tcdrain failed, continue.");
//            continue;
//        }
//        break;
//   }

	msecsleep(TTY_WAIT);
   
	buffer().resize(MIN_BUFFER_SIZE);
   
	const char *eos = m_pInterface->eos().c_str();
	unsigned int eos_len = m_pInterface->eos().length();
	unsigned int len = 0;
	for(;;) {
		if(buffer().size() <= len + 1) 
			buffer().resize(len + MIN_BUFFER_SIZE);
		int rlen = ::read(m_scifd, &buffer().at(len), 1);
		if(rlen == 0) {
			buffer().at(len) = '\0';
			throw XInterface::XCommError(i18n("read time-out, buf=;") + &buffer().at(0), __FILE__, __LINE__);
		}
		if(rlen < 0) {
			if(errno == EINTR) {
				dbgPrint("Serial, EINTR, try to continue.");
				continue;
			}
			else
				throw XInterface::XCommError(i18n("read error"), __FILE__, __LINE__);
		}
		len += rlen;
		if(len >= eos_len) {
			if( !strncmp(&buffer().at(len - eos_len), eos, eos_len)) {
				break;
			}
		}
	}
    
	buffer().resize(len + 1);
	buffer().at(len) = '\0';
}
void
XPosixSerialPort::receive(unsigned int length) throw (XInterface::XCommError &) {
	assert(m_pInterface->isOpened());
   
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
		int rlen = ::read(m_scifd, &buffer().at(len), 1);
		if(rlen == 0)
			throw XInterface::XCommError(i18n("read time-out"), __FILE__, __LINE__);
		if(rlen < 0) {
			if(errno == EINTR) {
				dbgPrint("Serial, EINTR, try to continue.");
				continue;
			}
			else
				throw XInterface::XCommError(i18n("read error"), __FILE__, __LINE__);
		}
		len += rlen;
	}
}    


#endif /*SERIAL_POSIX*/


#ifdef SERIAL_QT

#include <QtSerialPort/QtSerialPort>

XQtSerialPort::XQtSerialPort(XCharInterface *interface)
    : XPort(interface), m_qport() {

}
XQtSerialPort::~XQtSerialPort() {
    if(m_qport)
        m_qport->close();
}
void
XQtSerialPort::open() throw (XInterface::XCommError &) {
    Snapshot shot( *m_pInterface);

    try {
        m_qport.reset(new QSerialPort(QString(shot[ *m_pInterface->port()].to_str()).toLatin1().data()));
        if( !m_qport->open(QIODevice::ReadWrite))
            throw m_qport->error();

        if( !m_qport->setBaudRate(static_cast<int>(m_pInterface->serialBaudRate())))
            throw m_qport->error();

        QSerialPort::Parity parity;
        switch((int)m_pInterface->serialParity()) {
        case XCharInterface::PARITY_EVEN:
            parity = QSerialPort::Parity::EvenParity; break;
        case XCharInterface::PARITY_ODD:
            parity = QSerialPort::Parity::OddParity; break;
        case XCharInterface::PARITY_NONE:
            parity = QSerialPort::Parity::NoParity; break;
        }
        if( !m_qport->setParity(parity))
            throw m_qport->error();

        if( !m_qport->setDataBits(m_pInterface->serial7Bits() ? QSerialPort::Data7 : QSerialPort::Data8))
            throw m_qport->error();

        if( !m_qport->setStopBits((m_pInterface->serialStopBits() == 2) ? QSerialPort::TwoStop : QSerialPort::OneStop))
            throw m_qport->error();


        if( !m_qport->setBreakEnabled(false))
            throw m_qport->error();

        if( !m_qport->flush())
            throw m_qport->error();
    }
    catch (QSerialPort::SerialPortError &e) {
        throw XInterface::XCommError(i18n("tty open failed") + ": " + m_qport->errorString(), __FILE__, __LINE__);
    }
}
void
XQtSerialPort::send(const char *str) throw (XInterface::XCommError &) {
    XString buf(str);
    if(m_pInterface->eos().length())
        buf += m_pInterface->eos();
    else
        buf += m_pInterface->serialEOS();
    if(m_pInterface->serialHasEchoBack()) {
        this->write(str, strlen(str));	//every char should wait for echo back.
        this->write(buf.c_str() + strlen(str), buf.length() - strlen(str)); //EOS
        this->receive(); //wait for EOS.
    }
    else {
        this->write(buf.c_str(), buf.length());
    }
}
void
XQtSerialPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &) {
    assert(m_pInterface->isOpened());

    if(m_pInterface->serialHasEchoBack() && (size >= 2) && isprint(sendbuf[0])) {
        for(int cnt = 0; cnt < size; ++cnt) {
        //sends 1 char.
            write(sendbuf + cnt, 1);
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

    if(m_pInterface->serialFlushBeforeWrite()) {
        throw XInterface::XCommError(i18n("Serial error") + ": " + m_qport->errorString(), __FILE__, __LINE__);
    }

    msecsleep(TTY_WAIT);

    int wlen = 0;
    do {
        int ret = m_qport->write(sendbuf, size - wlen);
        if(ret < 0)
            throw XInterface::XCommError(i18n("Serial error") + ": " + m_qport->errorString(), __FILE__, __LINE__);
        wlen += ret;
        sendbuf += ret;
    } while (wlen < size);
}
void
XQtSerialPort::receive() throw (XInterface::XCommError &) {
    assert(m_pInterface->isOpened());

    msecsleep(TTY_WAIT);

    buffer().resize(MIN_BUFFER_SIZE);

    const char *eos = m_pInterface->eos().c_str();
    unsigned int eos_len = m_pInterface->eos().length();
    unsigned int len = 0;
    for(;;) {
        if(buffer().size() <= len + 1)
            buffer().resize(len + MIN_BUFFER_SIZE);
        if( !m_qport->waitForReadyRead(3000)) //3sec to timeout.
            throw XInterface::XCommError(i18n("Serial error") + ": " + m_qport->errorString(), __FILE__, __LINE__);
        int rlen = m_qport->read( &buffer().at(len), 1);
        if(rlen < 0) {
            throw XInterface::XCommError(i18n("Serial error") + ": " + m_qport->errorString(), __FILE__, __LINE__);
        }
        len += rlen;
        if(len >= eos_len) {
            if( !strncmp(&buffer().at(len - eos_len), eos, eos_len)) {
                break;
            }
        }
    }

    buffer().resize(len + 1);
    buffer().at(len) = '\0';
}
void
XQtSerialPort::receive(unsigned int length) throw (XInterface::XCommError &) {
    assert(m_pInterface->isOpened());

    msecsleep(TTY_WAIT);

    buffer().resize(length);
    unsigned int len = 0;

    while(len < length) {
        if( !m_qport->waitForReadyRead(3000)) //3sec to timeout.
            throw XInterface::XCommError(i18n("Serial error") + ": " + m_qport->errorString(), __FILE__, __LINE__);
        int rlen = m_qport->read( &buffer().at(len), 1);
        if(rlen < 0) {
            throw XInterface::XCommError(i18n("Serial error") + ": " + m_qport->errorString(), __FILE__, __LINE__);
        }
        len += rlen;
    }
}

#endif /*SERIAL_QT*/
