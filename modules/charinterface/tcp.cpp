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
#include "tcp.h"

#ifdef TCP_SOCKET

#if defined WINDOWS || defined __WIN32__ || defined _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>

    static class WSockInit {
    public:
        WSockInit() {
            int ret = WSAStartup(MAKEWORD(2,0), &data);
            if(ret)
                fprintf(stderr, "WSAStartup() has failed %d\n", ret);
        }
        ~WSockInit() {
            int ret = WSACleanup();
            if(ret)
                fprintf(stderr, "WSACleanup() has failed %d\n", ret);
        }
    private:
        WSADATA data;
    } s_wsockinit;

#else
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <sys/select.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <errno.h>
    #include <unistd.h>
    #include <fcntl.h>
#endif
 
#define MIN_BUFFER_SIZE 256
#define MIN_RECV_SIZE 64

XTCPSocketPort::XTCPSocketPort(XCharInterface *intf)
    : XPort(intf), m_socket(-1) {

}
XTCPSocketPort::~XTCPSocketPort() {
    if(m_socket >= 0) {
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
        closesocket(m_socket);
#else
        close(m_socket);
#endif
    }
}
void
XTCPSocketPort::reopen_socket() {
    if(m_socket >= 0) {
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
        closesocket(m_socket);
#else
        close(m_socket);
#endif
    }
    open(0);
}

shared_ptr<XPort>
XTCPSocketPort::open(const XCharInterface *pInterface) {
    std::string ipaddr = portString();
	int colpos = ipaddr.find_first_of(':');
	if(colpos == std::string::npos)
        throw XInterface::XCommError(i18n("tcp socket creation failed"), __FILE__, __LINE__);
	unsigned int port;
	if(sscanf(ipaddr.substr(colpos + 1).c_str(), "%u", &port) != 1)
        throw XInterface::XCommError(i18n("tcp socket creation failed"), __FILE__, __LINE__);
	ipaddr = ipaddr.substr(0, colpos);

	m_socket = socket(AF_INET, SOCK_STREAM, 0);
	if(m_socket < 0) {
        throw XInterface::XCommError(i18n("tcp socket creation failed"), __FILE__, __LINE__);
	}

    timeval timeout = {};
    timeout.tv_sec  = m_timeout_sec;
    if(setsockopt(m_socket, SOL_SOCKET, SO_SNDTIMEO,  (char*)&timeout, sizeof(timeout)) ||
        setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO,  (char*)&timeout, sizeof(timeout))
        ){
        throw XInterface::XCommError(i18n("tcp socket setting options failed"), __FILE__, __LINE__);
    }

    int opt = 1;
    //disables NAGLE protocol
    if(setsockopt(m_socket, IPPROTO_TCP, TCP_NODELAY, (char*)&opt, sizeof(opt)))
        throw XInterface::XCommError(i18n("tcp socket setting options failed"), __FILE__, __LINE__);
//    opt = 0;
//    if(setsockopt(m_socket, SOL_SOCKET, SO_SNDBUF, (char*)&opt, sizeof(opt)))
//        throw XInterface::XCommError(i18n("tcp socket setting options failed"), __FILE__, __LINE__);

    sockaddr_in dstaddr = {}; //zero clear.
	dstaddr.sin_port = htons(port);
	dstaddr.sin_family = AF_INET;
	dstaddr.sin_addr.s_addr = inet_addr(ipaddr.c_str());

    //setting up non-blocking connect for shorter timeout.
//to non-blocking
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
    u_long iomode = 1;
    if(ioctlsocket(m_socket, FIONBIO, &iomode) != 0)
        throw XInterface::XCommError(i18n("tcp open failed"), __FILE__, __LINE__);
#else
    int flags = fcntl(m_socket, F_GETFL, 0);
    if(flags == -1) {
        throw XInterface::XCommError(i18n("tcp open failed"), __FILE__, __LINE__);
    }
    if(fcntl(m_socket, F_SETFL, flags | O_NONBLOCK) == -1) {
        throw XInterface::XCommError(i18n("tcp open failed"), __FILE__, __LINE__);
    }
#endif

    while(connect(m_socket, (struct sockaddr *) &dstaddr, sizeof(dstaddr)) != 0) {
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
        errno = WSAGetLastError();
        if(errno == WSAEWOULDBLOCK) {
#else
        if((errno == EINTR) || (errno == EAGAIN)) {
            dbgPrint("TCP/IP, EINTR/EAGAIN, trying to continue.");
            continue;
        }
        if(errno == EINPROGRESS) {
#endif
            errno = 0;
            fd_set fs;
            FD_ZERO(&fs);
            FD_SET(m_socket, &fs);
            int ret = ::select(m_socket + 1, NULL, &fs, NULL, &timeout); //awaiting.
            if(ret < 0)
                throw XInterface::XCommError(i18n("tcp open failed"), __FILE__, __LINE__);
            if(ret == 0)
                throw XInterface::XCommError(i18n("tcp time-out during connection."), __FILE__, __LINE__);
            break;
        }
        throw XInterface::XCommError(formatString_tr(I18N_NOOP("tcp open failed %u"), errno).c_str(), __FILE__, __LINE__);
    }

 //back to blocking I/O.
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
    iomode = 0;
    if(ioctlsocket(m_socket, FIONBIO, &iomode) != 0)
        throw XInterface::XCommError(i18n("tcp open failed"), __FILE__, __LINE__);
#else
    if(fcntl(m_socket, F_SETFL, flags) == -1) {
        throw XInterface::XCommError(i18n("tcp open failed"), __FILE__, __LINE__);
    }
#endif

    m_eosPosInBuffer = -1;

    return shared_from_this();
}
void
XTCPSocketPort::send(const char *str) {
    XString buf(str);
    buf += eos();
    this->write(buf.c_str(), buf.length());
}
void
XTCPSocketPort::write(const char *sendbuf, int size) {
    fd_set fs_org;
    FD_ZERO(&fs_org);
    FD_SET(m_socket , &fs_org);

    int wlen = 0;
	do {
        fd_set fs;
        memcpy( &fs, &fs_org, sizeof(fd_set));
        timeval timeout = {};
        timeout.tv_sec  = m_timeout_sec;
        int ret = ::select(m_socket + 1, NULL, &fs, NULL, &timeout); //awaiting.
        if(ret <= 0)
            throw XInterface::XCommError(i18n("tcp writing failed"), __FILE__, __LINE__);
        ret = ::send(m_socket, sendbuf, size - wlen, 0);
        if(ret <= 0) {
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
            errno = WSAGetLastError();
//            if((errno == WSAEINTR)) {
#endif
            if((errno == EINTR) || (errno == EAGAIN)) {
                dbgPrint("TCP/IP, EINTR/EAGAIN, trying to continue.");
                continue;
            }
            gErrPrint(i18n("write error, trying to reopen the socket"));
            reopen_socket();
            throw XInterface::XCommError(i18n("tcp writing failed"), __FILE__, __LINE__);
        }
        wlen += ret;
        sendbuf += ret;
	} while (wlen < size);
}
unsigned int
XTCPSocketPort::rearrangeBufferForNextReceive() {
    if(m_eosPosInBuffer < 0)
        return 0;
    unsigned int eos_len = eos().length();
    int len = (int)buffer().size() - (int)(m_eosPosInBuffer + eos_len);
    if(len > 0) {
        //last reading exceeded linetext + EOS + null char.
        buffer().at(m_eosPosInBuffer + eos_len) = m_byteAfterEOS; //replacing null char to the real byte data.
        std::copy(buffer().begin() + m_eosPosInBuffer + eos_len, buffer().end(), &buffer().at(0));
        m_eosPosInBuffer = -1; //no remaining data.
    }
    return len;
}

void
XTCPSocketPort::receive() {
    fd_set fs_org;
    FD_ZERO(&fs_org);
    FD_SET(m_socket , &fs_org);

    const char *ceos = eos().c_str();
    unsigned int eos_len = eos().length();
    unsigned int len = rearrangeBufferForNextReceive();

    for(;;) {
        if(len >= eos_len) {
            //finds EOS or (if EOS is not set) null char.
            auto it = std::search( &buffer().at(0), &buffer().at(len), ceos, ceos + std::max(eos_len, 1u));
            if(it != &buffer().at(len)) {
                m_eosPosInBuffer = it - &buffer().at(0);
                buffer().resize(std::max(m_eosPosInBuffer + eos_len + 1, len));
                m_byteAfterEOS = *(it + eos_len); //escapes the byte data at null char.
                *(it + eos_len) = '\0'; //termination for C-based func.
                if(len < m_eosPosInBuffer + eos_len + 1) {
                    m_eosPosInBuffer = -1; //no remaining data.
                }
                break;
            }
        }

        if(buffer().size() <= len + MIN_RECV_SIZE)
			buffer().resize(len + MIN_BUFFER_SIZE);
        char *bpos = &buffer().at(len);

        fd_set fs;
        memcpy( &fs, &fs_org, sizeof(fd_set));
        timeval timeout = {};
        timeout.tv_sec  = m_timeout_sec;
        int ret = ::select(m_socket + 1, &fs, NULL, NULL, &timeout); //awaiting for data.
        if(ret < 0)
            throw XInterface::XCommError(i18n("tcp reading failed"), __FILE__, __LINE__);
        if(ret == 0) {
            //timeout during select().
            buffer().at(len) = '\0';
            throw XInterface::XCommError(i18n("read time-out, buf=;") + &buffer().at(0), __FILE__, __LINE__);
        }
        ret = ::recv(m_socket, bpos , MIN_RECV_SIZE, 0);//MIN_RECV_SIZE bytes reading.
        //allows the case ret == 0.
        if(ret < 0) {
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
            errno = WSAGetLastError();
            if(errno == WSAEMSGSIZE) {
                len += MIN_RECV_SIZE;
                continue;
            }
            //            if((errno == WSAEINTR)) {
#endif
            if((errno == EINTR) || (errno == EAGAIN)) {
                dbgPrint("TCP/IP, EINTR/EAGAIN, trying to continue.");
                continue;
            }
            gErrPrint(formatString_tr(I18N_NOOP("read error %u, trying to reopen the socket"), errno).c_str());
            reopen_socket();
            throw XInterface::XCommError(i18n("tcp reading failed"), __FILE__, __LINE__);
        }
        len += ret;
	}
}
void
XTCPSocketPort::receive(unsigned int length) {
    fd_set fs_org;
    FD_ZERO(&fs_org);
    FD_SET(m_socket , &fs_org);

    unsigned int len = rearrangeBufferForNextReceive();

	while(len < length) {
        buffer().resize(length);

        fd_set fs;
        memcpy( &fs, &fs_org, sizeof(fd_set));
        timeval timeout = {};
        timeout.tv_sec  = m_timeout_sec;
        int ret = ::select(m_socket + 1, &fs, NULL, NULL, &timeout); //awaiting for data.
        if(ret < 0)
            throw XInterface::XCommError(i18n("tcp reading failed"), __FILE__, __LINE__);
        if(ret == 0) {
            //timeout during select().
            throw XInterface::XCommError(i18n("read time-out, buf=;") + &buffer().at(0), __FILE__, __LINE__);
        }
        ret = ::recv(m_socket, &buffer().at(len) , length - len, 0); //reads remaining bytes.
        //allows the case ret == 0.
        if(ret < 0) {
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
            errno = WSAGetLastError();
            if((errno == WSAEINTR)) {
#else
            if((errno == EINTR) || (errno == EAGAIN)) {
#endif
                dbgPrint("TCP/IP, EINTR/EAGAIN, trying to continue.");
                continue;
            }
            gErrPrint(i18n("read error, trying to reopen the socket"));
            reopen_socket();
            throw XInterface::XCommError(i18n("tcp reading failed"), __FILE__, __LINE__);
        }
        len += ret;
	}
    if(buffer().size() > length) {
        unsigned int eos_len = eos().length();
        m_eosPosInBuffer = length - eos_len;
        m_byteAfterEOS = buffer().at(length);
    }
}    

#endif //TCP_POSIX

