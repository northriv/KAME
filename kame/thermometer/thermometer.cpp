//---------------------------------------------------------------------------
#include <math.h>
#include "thermometer.h"
//---------------------------------------------------------------------------
XThermometerList::XThermometerList(const char *name, bool runtime)
 : XCustomTypeListNode<XThermometer>(name, runtime) {
}

#define LIST XThermometerList
DECLARE_TYPE_HOLDER

REGISTER_TYPE(RawThermometer)
REGISTER_TYPE(LakeShore)
REGISTER_TYPE(CryoConcept)
REGISTER_TYPE(ScientificInstruments)
/*
XThermometerList::TypeHolder::Creator<XRawThermometer>
        g_thermometer_type_raw(XThermometerList::s_types, "Raw");
XThermometerList::TypeHolder::Creator<XLakeShore>
        g_thermometer_type_lakeshore(XThermometerList::s_types, "LakeShore");
XThermometerList::TypeHolder::Creator<XCryoConcept>
        g_thermometer_type_cryoconcept(XThermometerList::s_types, "CryoConcept");
XThermometerList::TypeHolder::Creator<XScientificInstruments>
        g_thermometer_type_si(XThermometerList::s_types, "ScientificInstruments");
*/
XThermometer::XThermometer(const char *name, bool runtime) : 
    XNode(name, runtime),
    m_tempMin(create<XDoubleNode>("TMin", false)),
    m_tempMax(create<XDoubleNode>("TMax", false))
  {
    tempMin()->value(0.0);
    tempMax()->value(1e4);
  }

XLakeShore::XLakeShore(const char *name, bool runtime) :
 XThermometer(name, runtime),
 m_resMin(create<XDoubleNode>("RMin", false)),
 m_resMax(create<XDoubleNode>("RMax", false)),
 m_zu(create<XDoubleListNode>("Zu", false)),
 m_zl(create<XDoubleListNode>("Zl", false)),
 m_ai(create<XDouble2DNode>("Ai", false))
 {
}

double
XLakeShore::getRawValue(double temp) const
{
  //using Newton's method
  double x, y, dypdx, val;
  if(temp < *tempMin()) return *resMax();
  if(temp > *tempMax()) return *resMin();
  //    x = (log10(RMin) + log10(RMax)) / 2;
  val = *resMin();
  for(double dy = 0.0001;;dy *= 2)
    {
      if(dy > 1.0) return *resMin();
      double t = (double)KAME::rand() / (RAND_MAX - 1);
      x = (log10(*resMax()) * t + log10(*resMin()) * (1 - t));
      for(int i = 0; i < 100; i++)
	{
	  y = getTemp(pow(10, x)) - temp;
	  if(fabs(y) < dy)
	    {
	      val = pow(10, x);
	      return val;
	    }
	  dypdx = (y - (getTemp(pow(10, x - 0.00001)) - temp)) / 0.00001;
	  if(dypdx != 0) x -= y / dypdx;
	  if((x > log10(*resMax())) || (x < log10(*resMin())) || (dypdx == 0))
	    {
	      double t = (double)KAME::rand() / (RAND_MAX - 1);
	      x = (log10(*resMax()) * t + log10(*resMin()) * (1 - t));
	    }
	}
    }
  return val;
}

double
XLakeShore::getTemp(double res) const
{
  double temp = 0, z, u = 0;
  if(res > *resMax()) return *tempMin();
  if(res < *resMin()) return *tempMax();
  z = log10(res);
  zu()->childLock();
  zl()->childLock();
  unsigned int n;
  for(n = 0; n < zu()->count(); n++)
    {
      u = (z - *(*zu())[n] + z - *(*zl())[n]) / (*(*zu())[n] - *(*zl())[n]);
      if((u >= -1) && (u <= 1)) break;
    }
  zl()->childUnlock();
  if(n >= zu()->count()) { zu()->childUnlock(); return 0; }
  zu()->childUnlock();
  ai()->childLock();
  (*ai())[n]->childLock();
  for(unsigned int i = 0; i < (*ai())[n]->count(); i++)
    {
      temp += *(*(*ai())[n])[i] * cos(i * acos(u));
    }
  (*ai())[n]->childUnlock();
  ai()->childUnlock();
  return temp;
}

XCryoConcept::XCryoConcept(const char *name, bool runtime) :
 XThermometer(name, runtime),
 m_resMin(create<XDoubleNode>("RMin", false)),
 m_resMax(create<XDoubleNode>("RMax", false)),
 m_ai(create<XDoubleListNode>("Ai", false)),
 m_a(create<XDoubleNode>("A", false)),
 m_t0(create<XDoubleNode>("T0", false)),
 m_r0(create<XDoubleNode>("R0", false)),
 m_rCrossover(create<XDoubleNode>("RCrossover", false))
 {
  }
  
double
XCryoConcept::getRawValue(double temp) const
{
  //using Newton's method
  double x, y, dypdx, val;
  if(temp < *tempMin()) return *resMax();
  if(temp > *tempMax()) return *resMin();
  //    x = (log10(RMin) + log10(RMax)) / 2;
  val = *resMin();
  for(double dy = 0.0001;;dy *= 2)
    {
      if(dy > 1.0) return *resMin();
      double t = (double)KAME::rand() / (RAND_MAX - 1);
      x = (log10(*resMax()) * t + log10(*resMin()) * (1 - t));
      for(int i = 0; i < 100; i++)
	{
	  y = getTemp(pow(10, x)) - temp;
	  if(fabs(y) < dy)
	    {
	      val = pow(10, x);
	      return val;
	    }
	  dypdx = (y - (getTemp(pow(10, x - 0.00001)) - temp)) / 0.00001;
	  if(dypdx != 0) x -= y / dypdx;
	  if((x > log10(*resMax())) || (x < log10(*resMin())) || (dypdx == 0))
	    {
	      double t = (double)KAME::rand() / (RAND_MAX - 1);
	      x = (log10(*resMax()) * t + log10(*resMin()) * (1 - t));
	    }
	}
    }
  return val;
}
double
XCryoConcept::getTemp(double res) const
{
  if(res > *resMax()) return *tempMin();
  if(res < *resMin()) return *tempMax();
  if(res < *rCrossover())
    { 
      double y = 0, r = 1.0;
      double x = log10(res);
      ai()->childLock();
      for(unsigned int i = 0; i < ai()->count(); i++)
	{
	  y += *(*ai())[i] * r;
	  r *= x;
	}
      ai()->childUnlock();
      return pow(10.0, y);
    }
  else
    {
      return *t0() / pow(log10(res / *r0()), *a());
    }
}

XScientificInstruments::XScientificInstruments(const char *name, bool runtime) : 
  XThermometer(name, runtime),
 m_resMin(create<XDoubleNode>("RMin", false)),
 m_resMax(create<XDoubleNode>("RMax", false)),
 m_abcde(create<XDoubleListNode>("ABCDE", false)),
 m_abc(create<XDoubleListNode>("ABC", false)),
 m_rCrossover(create<XDoubleNode>("RCrossover", false))
 {
  }

double
XScientificInstruments
::getRawValue(double temp) const
{
  //using Newton's method
  double x, y, dypdx, val;
  if(temp < *tempMin()) return *resMax();
  if(temp > *tempMax()) return *resMin();
  //    x = (log10(RMin) + log10(RMax)) / 2;
  val = *resMin();
  for(double dy = 0.0001;;dy *= 2)
    {
      if(dy > 1.0) return *resMin();
      double t = (double)KAME::rand() / (RAND_MAX - 1);
      x = (log10(*resMax()) * t + log10(*resMin()) * (1 - t));
      for(int i = 0; i < 100; i++)
	{
	  y = getTemp(pow(10, x)) - temp;
	  if(fabs(y) < dy)
	    {
	      val = pow(10, x);
	      return val;
	    }
	  dypdx = (y - (getTemp(pow(10, x - 0.00001)) - temp)) / 0.00001;
	  if(dypdx != 0) x -= y / dypdx;
	  if((x > log10(*resMax())) || (x < log10(*resMin())) || (dypdx == 0))
	    {
	      double t = (double)KAME::rand() / (RAND_MAX - 1);
	      x = (log10(*resMax()) * t + log10(*resMin()) * (1 - t));
	    }
	}
    }
  return val;
}

double
XScientificInstruments
::getTemp(double res) const
{
  if(res > *resMax()) return *tempMin();
  if(res < *resMin()) return *tempMax();
  double y = 0.0;
  double lx = log(res);
  if(res > *rCrossover())
    { 
	      abcde()->childLock();
	      if(abcde()->count() >= 5) {
	      		y = (*(*abcde())[0] + *(*abcde())[2]*lx + *(*abcde())[4]*lx*lx)
	      			/ (1.0 + *(*abcde())[1]*lx + *(*abcde())[3]*lx*lx);
	      }
	      abcde()->childUnlock();
	      return y;
    }
  else
    {
	      abc()->childLock();
	      if(abc()->count() >= 3) {
	      		y = 1.0/(*(*abc())[0] + *(*abc())[1]*res*lx + *(*abc())[2]*res*res);
	      }
	      abc()->childUnlock();
	      return y;
    }
}
