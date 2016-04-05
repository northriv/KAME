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
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <errno.h>
    #include <unistd.h>
#endif
 
#define MIN_BUFFER_SIZE 256

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
XTCPSocketPort::reopen_socket() throw (XInterface::XCommError &) {
    if(m_socket >= 0) {
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
        closesocket(m_socket);
#else
        close(m_socket);
#endif
    }
    open(0);
}

void
XTCPSocketPort::open(const XCharInterface *pInterface) throw (XInterface::XCommError &) {
	struct sockaddr_in dstaddr;

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

	struct timeval timeout;
    timeout.tv_sec  = 3; //3sec. timeout.
	timeout.tv_usec = 0;
	if(setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO,  (char*)&timeout, sizeof(timeout)) ||
		setsockopt(m_socket, SOL_SOCKET, SO_SNDTIMEO,  (char*)&timeout, sizeof(timeout))){
        throw XInterface::XCommError(i18n("tcp socket setting options failed"), __FILE__, __LINE__);
	}

    int opt = 1;
    if(setsockopt(m_socket, SOL_SOCKET, SO_KEEPALIVE, (char*)&opt, sizeof(opt)))
        throw XInterface::XCommError(i18n("tcp socket setting options failed"), __FILE__, __LINE__);

	memset( &dstaddr, 0, sizeof(dstaddr));
	dstaddr.sin_port = htons(port);
	dstaddr.sin_family = AF_INET;
	dstaddr.sin_addr.s_addr = inet_addr(ipaddr.c_str());

	if(connect(m_socket, (struct sockaddr *) &dstaddr, sizeof(dstaddr)) == -1) {
        throw XInterface::XCommError(i18n("tcp open failed"), __FILE__, __LINE__);
	}
}
void
XTCPSocketPort::send(const char *str) throw (XInterface::XCommError &) {
    XString buf(str);
    buf += eos();
    this->write(buf.c_str(), buf.length());
}
void
XTCPSocketPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &) {
	int wlen = 0;
	do {
        int ret = ::send(m_socket, sendbuf, size - wlen, 0);
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
void
XTCPSocketPort::receive() throw (XInterface::XCommError &) {
	buffer().resize(MIN_BUFFER_SIZE);
   
    const char *ceos = eos().c_str();
    unsigned int eos_len = eos().length();
	unsigned int len = 0;
	for(;;) {
		if(buffer().size() <= len + 1) 
			buffer().resize(len + MIN_BUFFER_SIZE);
        int rlen = ::recv(m_socket, &buffer().at(len), 1, 0);
		if(rlen == 0) {
			buffer().at(len) = '\0';
            throw XInterface::XCommError(i18n("read time-out, buf=;") + &buffer().at(0), __FILE__, __LINE__);
		}
		if(rlen < 0) {
#if defined WINDOWS || defined __WIN32__ || defined _WIN32
            errno = WSAGetLastError();
//            if((errno == WSAEINTR)) {
#endif
            if((errno == EINTR) || (errno == EAGAIN)) {
                dbgPrint("TCP/IP, EINTR/EAGAIN, trying to continue.");
                continue;
            }
            gErrPrint(formatPrint(i18n("read error %u, trying to reopen the socket"), errno);
            reopen_socket();
            throw XInterface::XCommError(i18n("tcp reading failed"), __FILE__, __LINE__);
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
XTCPSocketPort::receive(unsigned int length) throw (XInterface::XCommError &) {
	buffer().resize(length);
	unsigned int len = 0;
   
	while(len < length) {
        int rlen = ::recv(m_socket, &buffer().at(len), 1, 0);
		if(rlen == 0)
            throw XInterface::XCommError(i18n("read time-out"), __FILE__, __LINE__);
		if(rlen < 0) {
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
		len += rlen;
	}
}    

#endif //TCP_POSIX

