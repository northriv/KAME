/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <iostream>
#include <stdint.h>
#define TIOCSRS485 0x542F

uint16_t
crc16(const unsigned char *bytes, ssize_t count) {
	uint16_t z = 0xffffu;
	for(ssize_t i = 0; i < count; ++i) {
		uint16_t x = bytes[i];
		z ^= x;
		for(int shifts = 0; shifts < 8; ++shifts) {
			uint16_t lsb = z % 2;
			z = z >> 1;
			if(lsb)
				z ^= 0xa001u;
		}
	}
	return z;
}

int main() {
	struct termios ttyios;
	speed_t baudrate;
	int scifd;
	if((scifd = open("/dev/ttyUSB0",
						 O_RDWR | O_NOCTTY | O_NONBLOCK)) == -1) {
		abort();
	}
    
	tcsetpgrp(scifd, getpgrp());
      
	bzero( &ttyios, sizeof(ttyios));
//      tcgetattr(m_scifd, &ttyios);

	cfsetispeed( &ttyios, B9600);
	cfsetospeed( &ttyios, B9600);
	cfmakeraw( &ttyios);
	ttyios.c_cflag &= ~(PARENB | CSIZE);
		ttyios.c_cflag |= PARENB;
//		ttyios.c_cflag |= PARENB | PARODD;
//		ttyios.c_cflag |= CS7;
		ttyios.c_cflag |= CS8;
	ttyios.c_cflag |= HUPCL | CLOCAL | CREAD;
//		ttyios.c_cflag |= CSTOPB;
	ttyios.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); //non-canonical mode
	ttyios.c_iflag |= IGNBRK;
//		ttyios.c_iflag |= IGNPAR;
	ttyios.c_cc[VMIN] = 0; //no min. size
	ttyios.c_cc[VTIME] = 30; //1sec time-out
	if(tcsetattr(scifd, TCSAFLUSH, &ttyios ) < 0)
		abort();
	
//    if(fcntl(scifd, F_SETFL, (~O_NONBLOCK) & fcntl(scifd, F_GETFL)) == - 1) {
//		abort();
//	}

//    struct serial_rs485 rs485conf;
//	// Set RS485 mode:
//	rs485conf.flags |= SER_RS485_ENABLED;
//	if (ioctl(scifd, TIOCSRS485, &rs485conf) < 0) {
//		printf("ioctl error\n");
//		abort();
//	}

    unsigned char wbuf[] =
//    {0x03,0x03,0x03,0x80,0x00,0x02};
//    {0x03,0x08,0x00,0x00,0x34,0x56};
//   {0x03,0x10,0x01,0x80,0x00,0x01,0x02,0x0,0x1};
    {0x03,0x06,0x01,0x80,0x0,0x1};
    uint16_t crc = crc16(wbuf, 6);
	write(scifd, wbuf, sizeof(wbuf));
	write(scifd, &crc, 2);
	write(1, wbuf, sizeof(wbuf));
	write(1, &crc, 2);
	char rbuf[10];
	bzero(rbuf, 10);
//	usleep(300000);
	read(scifd, rbuf, 2);
	write(1, rbuf, 10);
}
