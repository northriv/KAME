/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
/*! \file
 * Linux-only helper: set an arbitrary (non-B<rate>) serial line speed.
 *
 * Some instruments KAME ships drivers for run at speeds that have no POSIX
 * `B<rate>` constant on any platform — e.g. the Yokogawa-style DC source in
 * modules/dcsource/userdcsource.cpp asks for 256000, which is a Windows-only
 * standard rate.  Linux can do it through the termios2 / `BOTHER` interface,
 * where c_ispeed / c_ospeed carry the literal bits per second.
 *
 * This lives in its own translation unit on purpose: <asm/termbits.h>, which
 * declares `struct termios2` and `BOTHER`, redefines `struct termios` and so
 * cannot be included in the same TU as glibc's <termios.h> (which serial.cpp
 * needs).  Keeping the two apart is the standard way around that, and it also
 * avoids hand-copying kernel UAPI constants into KAME.
 */
#ifdef __linux__

#include <asm/termbits.h>   /* struct termios2, BOTHER, CBAUD, TCGETS2/TCSETS2 */
#include <sys/ioctl.h>
#include <errno.h>

/*! Sets both input and output speed of \a fd to exactly \a rate bit/s.
 * Call AFTER tcsetattr(), which would otherwise overwrite the speed.
 * \return 0 on success, -1 with errno set otherwise. */
int kame_serial_set_custom_baud(int fd, unsigned int rate) {
    struct termios2 tio2;
    if(ioctl(fd, TCGETS2, &tio2) < 0)
        return -1;
    tio2.c_cflag &= ~CBAUD;
    tio2.c_cflag |= BOTHER;
    tio2.c_ispeed = rate;
    tio2.c_ospeed = rate;
    if(ioctl(fd, TCSETS2, &tio2) < 0)
        return -1;
    /* The driver reports back what it could actually program; a UART whose
     * divisor cannot express the request silently rounds.  Accept up to 2%,
     * which is the usual limit for reliable 8N1 framing. */
    if(ioctl(fd, TCGETS2, &tio2) == 0) {
        unsigned int got = tio2.c_ospeed;
        unsigned int diff = (got > rate) ? (got - rate) : (rate - got);
        if((unsigned long)diff * 50u > (unsigned long)rate) {
            errno = EINVAL;
            return -1;
        }
    }
    return 0;
}

#endif /* __linux__ */
