#include "userdso.h"
#include "interface.h"
#include "xwavengraph.h"
#include <klocale.h>
//---------------------------------------------------------------------------
XTDS::XTDS(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
 XDSO(name, runtime, scalarentries, interfaces, thermometers, drivers) {
  const char* ch[] = {"CH1", "CH2", "CH3", "CH4", "MATH1", "MATH2", 0L};
  for(int i = 0; ch[i]; i++)
    {
        trace1()->add(ch[i]);
        trace2()->add(ch[i]);
    }

  interface()->setGPIBWaitBeforeWrite(20); //20msec
  interface()->setGPIBWaitBeforeSPoll(10); //10msec

  recordLength()->add("500");
  recordLength()->add("10000");
 }

void
XTDS::afterStart()
{
  interface()->send("HEADER ON");
  interface()->query("ACQ:STOPAFTER?");
  char buf[10];
  if(interface()->scanf(":ACQ%*s %9s", buf) != 1)
       throw XInterface::XConvError(__FILE__, __LINE__);
  singleSequence()->value(!strncmp(buf, "SEQ", 3));
  
  interface()->query("ACQ:MODE?");
  if(interface()->scanf(":ACQ%*s %9s", buf) != 1)
       throw XInterface::XConvError(__FILE__, __LINE__);
  if(!strncmp(buf, "AVE", 3))
    {
      interface()->query("ACQ:NUMAVG?");
      int x;
      if(interface()->scanf(":ACQ%*s %d", &x) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
      average()->value(x);
    }
  if(!strncmp(buf, "SAM", 3))
       average()->value(1);
  interface()->send("DATA:ENC RPB;WIDTH 2"); //MSB first RIB
}
void 
XTDS::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
  if(*average() == 1)
    {
      interface()->send("ACQ:MODE SAMPLE");

    }

  else
    {
      interface()->send("ACQ:MODE AVE;NUMAVG " + average()->to_str());
    }
}

void
XTDS::onSingleChanged(const shared_ptr<XValueNodeBase> &)
{
  if(*singleSequence())
    {
      interface()->send("ACQ:STOPAFTER SEQUENCE;STATE ON");
    }
  else
    {
      interface()->send("ACQ:STOPAFTER RUNSTOP;STATE ON");
    }
}
void
XTDS::onTrigPosChanged(const shared_ptr<XValueNodeBase> &)
{
    if(*trigPos() >= 0)
    	   interface()->sendf("HOR:DELAY:STATE OFF;TIME %.2g", (double)*trigPos());
    else
	   interface()->sendf("HOR:DELAY:STATE ON;TIME %.2g", -(*trigPos() - 50.0)/100.0* *timeWidth());    
}
void
XTDS::onTimeWidthChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->sendf("HOR:MAIN:SCALE %.1g", (double)*timeWidth()/10.0);
}
void
XTDS::onVFullScale1Changed(const shared_ptr<XValueNodeBase> &)
{
    std::string ch = trace1()->to_str();
	if(ch.empty()) return;
	interface()->sendf("%s:SCALE %.1g", ch.c_str(), (double)*vFullScale1()/10.0);
}
void
XTDS::onVFullScale2Changed(const shared_ptr<XValueNodeBase> &)
{
    std::string ch = trace2()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:SCALE %.1g", ch.c_str(), (double)*vFullScale2()/10.0);
}
void
XTDS::onVOffset1Changed(const shared_ptr<XValueNodeBase> &)
{
    std::string ch = trace1()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g", ch.c_str(), (double)*vOffset1());
}
void
XTDS::onVOffset2Changed(const shared_ptr<XValueNodeBase> &)
{
    std::string ch = trace2()->to_str();
    if(ch.empty()) return;
    interface()->sendf("%s:OFFSET %.8g", ch.c_str(), (double)*vOffset2());
}
void
XTDS::onRecordLengthChanged(const shared_ptr<XValueNodeBase> &)
{
	interface()->send("HOR:RECORD " + 
             recordLength()->to_str());
}
void
XTDS::onForceTriggerTouched(const shared_ptr<XNode> &)
{
	interface()->send("TRIG FORC");
}

void
XTDS::startSequence()
{                      
  interface()->send("ACQ:STATE ON");
}


int
XTDS::acqCount(bool *seq_busy)
{
      interface()->query("ACQ:NUMACQ?;:BUSY?");
      int n;
      int busy;
      if(interface()->scanf(":ACQ%*s %d;:BUSY %d", &n, &busy) != 2)
            throw XInterface::XConvError(__FILE__, __LINE__);
      *seq_busy = busy;
      return n;
}

double
XTDS::getTimeInterval()
{
  interface()->query("WFMP?");
  char *cp = strstr(&interface()->buffer()[0], "XIN");
  if(!cp) throw XInterface::XConvError(__FILE__, __LINE__);
  double x;
  int ret = sscanf(cp, "%*s %lf", &x);
  if(ret != 1) throw XInterface::XConvError(__FILE__, __LINE__);
  return x;
}

void
XTDS::getWave(std::deque<std::string> &channels)
{
  interface()->lock();
  try {
      int pos = 1;
      int width = 20000;
      for(std::deque<std::string>::iterator it = channels.begin();
            it != channels.end(); it++)
        {
          int rsize = (2 * width + 1024);
          interface()->sendf("DATA:SOURCE %s;START %u;STOP %u;:WAVF?",
    		     (const char *)it->c_str(),
                  pos, pos + width);
          interface()->receive(rsize);
          rawData().insert(rawData().end(), 
                interface()->buffer().begin(), interface()->buffer().end());
        }
  }
  catch (XKameError& e) {
      interface()->unlock();
      throw e;
  }
  interface()->unlock();
}
void
XTDS::convertRaw() throw (XRecordError&) {
  double xin = 0;
  double yin[256], yoff[256];
  int width = 0;
  double xoff = 0;
  int triggerpos;

  int size = rawData().size();
  char *buf = &rawData()[0];
  
  int ch_cnt = 0;
  //scan # of channels etc.
  char *cp = buf;
  for(;;)
    {
      if(cp >= &buf[size]) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
      if(*cp == ':') cp++;
      if(!strncasecmp(cp, "XIN", 3))
        	sscanf(cp, "%*s %lf", &xin);
      if(!strncasecmp(cp, "PT_O", 4))
        	sscanf(cp, "%*s %d", &triggerpos);
      if(!strncasecmp(cp, "XZE", 3))
        	sscanf(cp, "%*s %lf", &xoff);
      if(!strncasecmp(cp, "YMU", 3))
        	sscanf(cp, "%*s %lf", &yin[ch_cnt - 1]);
      if(!strncasecmp(cp, "YOF", 3))
        	sscanf(cp, "%*s %lf", &yoff[ch_cnt - 1]);
      if(!strncasecmp(cp, "NR_P", 4))
        {
          ch_cnt++;
	      sscanf(cp, "%*s %d", &width);
        }
      if(!strncasecmp(cp, "CURV", 4))
        {
	   for(;;)
	    {
	      cp = index(cp, '#');
	      if(!cp) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	      int x;
	      if(sscanf(cp, "#%1d", &x) != 1) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	      char fmt[9];
          if(snprintf(fmt, sizeof(fmt), "#%%*1d%%%ud", x) < 0)
             throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	      int yyy;
	      if(sscanf(cp, fmt, &yyy) != 1) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	      if(yyy == 0) throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	      cp += 2 + x;
           
	      cp += yyy;
	      if(*cp != ',') break;
	    }
        }
      char *ncp = index(cp, ';');
      if(!ncp)
	cp = index(cp, ':');
      else
	cp = ncp;
      if(!cp) break;
      cp++;
    }
  if((width <= 0) || (width > size/2)) throw XBufferUnderflowRecordError(__FILE__, __LINE__);

  if(triggerpos != 0)
      xoff = -triggerpos * xin;

  setRecordDim(ch_cnt, xoff, xin, width);
  
  cp = buf;
  for(int j = 0; j < ch_cnt; j++)
    {
      double *wave = waveRecorded(j);
      cp = index(cp, '#');
      if(!cp) break;
      int x;
      if(sscanf(cp, "#%1d", &x) != 1) break;
      char fmt[9];
      if(snprintf(fmt, sizeof(fmt), "#%%*1d%%%ud", x) < 0)
         throw XBufferUnderflowRecordError(__FILE__, __LINE__);
      int yyy;
      if(sscanf(cp, fmt, &yyy) != 1) break;
      if(yyy == 0) break;
      cp += 2 + x;
                

      int i = 0;
      for(; i < std::min(width, yyy/2); i++)
        	{
        	  double val = *((unsigned char *)cp) * 0x100;
        	  val += *((unsigned char *)cp + 1);
        	  *(wave++) += yin[j] * (val - yoff[j] - 0.5);
        	  cp += 2;
        	}
      for(; i < width; i++) {
      	  *(wave++) = 0.0;
      }
    }  
}
