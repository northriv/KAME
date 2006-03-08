#include "serial.h"
#include <klocale.h>

#if defined WINDOWS || defined __WIN32__
 #include <windows.h>
#endif // WINDOWS || __WIN32__

#ifdef SERIAL_POSIX
 #include <termios.h>
 #include <unistd.h>
 #include <fcntl.h>
 #include <errno.h>
 
#define MIN_BUFFER_SIZE 256
#define TTY_WAIT 1 //ms

XPosixSerialPort::XPosixSerialPort(XInterface *interface)
 : XPort(interface), m_scifd(-1)
{

}
XPosixSerialPort::~XPosixSerialPort()
{
    if(m_scifd >= 0) close(m_scifd);
}
void
XPosixSerialPort::open() throw (XInterface::XCommError &)
{
  struct termios ttyios;
  speed_t baudrate;
      if((m_scifd = ::open(QString(m_pInterface->port()->to_str()).local8Bit(),
         O_RDWR | O_NOCTTY | O_SYNC)) == -1)
      {
           throw XInterface::XCommError(i18n("tty open failed"), __FILE__, __LINE__);
      }
      
      tcsetpgrp(m_scifd, getpgrp());
      
      bzero(&ttyios, sizeof(ttyios));
//      tcgetattr(m_scifd, &ttyios);

      switch(static_cast<int>(*m_pInterface->baudrate()))
        {
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

      cfsetispeed(&ttyios, baudrate);
      cfsetospeed(&ttyios, baudrate);
      cfmakeraw(&ttyios);
      ttyios.c_cflag &= ~(PARENB | CSIZE);
      ttyios.c_cflag |= HUPCL | CLOCAL | CSTOPB | CS8 | CREAD;
      ttyios.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); //non-canonical mode
      ttyios.c_iflag |= IGNBRK | IGNPAR;
      ttyios.c_cc[VMIN] = 0; //no min. size
      ttyios.c_cc[VTIME] = 30; //3sec time-out
      if(tcsetattr(m_scifd, TCSAFLUSH, &ttyios ) < 0)
            throw XInterface::XCommError(i18n("stty failed"), __FILE__, __LINE__);
}
void
XPosixSerialPort::send(const char *str) throw (XInterface::XCommError &)
{
    std::string buf(str);
    buf += m_pInterface->eos();
    this->write(buf.c_str(), buf.length());
}
void
XPosixSerialPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &)
{
   ASSERT(m_pInterface->isOpened());
      
      for (;;) {
          int ret = tcflush(m_scifd, TCIFLUSH);
          if(ret < 0) {
             dbgPrint(i18n("TCIFLUSH failed. errno=%1. continue.").arg(errno));
             continue;
          }
          break;
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
            else
            {
                throw XInterface::XCommError(i18n("write error"), __FILE__, __LINE__);
            }
        }
        wlen += ret;
        sendbuf += ret;
      } while (wlen < size);
}
void
XPosixSerialPort::receive() throw (XInterface::XCommError &)
{
   ASSERT(m_pInterface->isOpened());
    
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
   for(;;)
    {
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
          if(!strncmp(&buffer().at(len - eos_len), eos, eos_len))
          {
                break;
          }
      }
    }
    
   buffer().resize(len + 1);
   buffer().at(len) = '\0';
}
void
XPosixSerialPort::receive(unsigned int length) throw (XInterface::XCommError &)
{
   ASSERT(m_pInterface->isOpened());
   
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
   
   while(len < length)
    {
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
