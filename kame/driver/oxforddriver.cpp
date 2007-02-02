#include "oxforddriver.h"
#include <klocale.h>

XOxfordInterface::XOxfordInterface
    (const char *name, bool runtime, const shared_ptr<XDriver> &driver)
   : XCharInterface(name, runtime, driver) {
    setEOS("\r\n");
    setGPIBWaitBeforeSPoll(10);
}

void
XOxfordInterface::send(const char *str) throw (XInterface::XCommError &)
{
  ASSERT(strlen(str));
  if(str[0] == '$') {
      XCharInterface::send(str);
  }
  else {
      //Oxfords always send back echo
      query(str);
  }
}

void
XOxfordInterface::query(const char *str) throw (XInterface::XCommError &)
{
  lock();
  try {
      for(int i = 0; i < 30; i++)
        {
          XCharInterface::send(str);
          XCharInterface::receive();
          if(buffer().size() >= 1)
          if(buffer()[0] == str[0]) {
              unlock();
              return;
          }
          msecsleep(100);
        }
  }
  catch (XCommError &e) {
       unlock();
       throw e;
  }
  unlock();
  throw XCommError(KAME::i18n("Oxford Query Error, Initial doesn't match"), __FILE__, __LINE__);
}

void
XOxfordInterface::open() throw (XInterfaceError &)
{
  XCharInterface::open();
  //    XDriver::Send("@0");
  send("$Q2");
  //    msecsleep(100);
  //remote & unlocked
  send("C3");
}

void
XOxfordInterface::close()
{
    if(!isOpened()) return;
    try {
      send("C0"); //local
      XCharInterface::close();
    }
    catch (XCommError &e) {
      e.print(driver()->getLabel() + KAME::i18n(": close Oxford port failed, because"));
      return;
    }
}
void
XOxfordInterface::receive() throw (XCommError &) {
    XCharInterface::receive();
}
void
XOxfordInterface::receive(int length) throw (XCommError &) {
    XCharInterface::receive(length);
}


