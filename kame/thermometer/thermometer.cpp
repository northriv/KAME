//---------------------------------------------------------------------------
#include <math.h>
#include <klocale.h>

#include "thermometer.h"
#include "cspline.h"
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
REGISTER_TYPE(ApproxThermometer)

XThermometer::XThermometer(const char *name, bool runtime) : 
    XNode(name, runtime),
    m_tempMin(create<XDoubleNode>("TMin", false)),
    m_tempMax(create<XDoubleNode>("TMax", false))
  {
    tempMin()->value(1e-3);
    tempMax()->value(1e3);
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
  unsigned int n;
  { XScopedReadLock<XRecursiveRWLock> lock(zu()->childMutex());
      { XScopedReadLock<XRecursiveRWLock> lock(zl()->childMutex());
      for(n = 0; n < zu()->count(); n++)
        {
          u = (z - *(*zu())[n] + z - *(*zl())[n]) / (*(*zu())[n] - *(*zl())[n]);
          if((u >= -1) && (u <= 1)) break;
        }
      if(n >= zu()->count())
        return 0;
      }
  }
  { XScopedReadLock<XRecursiveRWLock> lock(ai()->childMutex());
      { XScopedReadLock<XRecursiveRWLock> lock((*ai())[n]->childMutex());
          for(unsigned int i = 0; i < (*ai())[n]->count(); i++)
            {
              temp += *(*(*ai())[n])[i] * cos(i * acos(u));
            }
      }
  }
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
      { XScopedReadLock<XRecursiveRWLock> lock(ai()->childMutex());
          for(unsigned int i = 0; i < ai()->count(); i++)
        	{
        	  y += *(*ai())[i] * r;
        	  r *= x;
        	}
      }
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
          { XScopedReadLock<XRecursiveRWLock> lock(abcde()->childMutex());
        	      if(abcde()->count() >= 5) {
        	      		y = (*(*abcde())[0] + *(*abcde())[2]*lx + *(*abcde())[4]*lx*lx)
        	      			/ (1.0 + *(*abcde())[1]*lx + *(*abcde())[3]*lx*lx);
        	      }
          }
	      return y;
    }
  else
    {
          { XScopedReadLock<XRecursiveRWLock> lock(abc()->childMutex());
        	      if(abc()->count() >= 3) {
        	      		y = 1.0/(*(*abc())[0] + *(*abc())[1]*res*lx + *(*abc())[2]*res*res);
        	      }
          }
	      return y;
    }
}

XApproxThermometer::XApproxThermometer(const char *name, bool runtime) :
 XThermometer(name, runtime),
 m_resList(create<XDoubleListNode>("ResList", false)),
 m_tempList(create<XDoubleListNode>("TempList", false))
{
}

double
XApproxThermometer
::getTemp(double res) const
{
    atomic_shared_ptr<CSplineApprox> approx(m_approx);
    if(!approx) {
        std::map<double, double> pts;
        { XScopedReadLock<XRecursiveRWLock> lock(m_resList->childMutex());
            { XScopedReadLock<XRecursiveRWLock> lock(m_tempList->childMutex());
                for(unsigned int i = 0; i < std::min(m_resList->count(), m_tempList->count()); i++) {
                    pts.insert(std::pair<double, double>(log(*(*m_resList)[i]), log(*(*m_tempList)[i])));
                }
            }
        }
        if(pts.size() < 4)
            throw XKameError(
                i18n("XApproxThermometer, Too small number of points"), __FILE__, __LINE__);
        approx.reset(new CSplineApprox(pts));
        m_approx = approx;
    }
    return exp(approx->approx(log(res)));
}
double
XApproxThermometer
::getRawValue(double temp) const
{
    atomic_shared_ptr<CSplineApprox> approx(m_approx_inv);
    if(!approx) {
        std::map<double, double> pts;
        { XScopedReadLock<XRecursiveRWLock> lock(m_resList->childMutex());
            { XScopedReadLock<XRecursiveRWLock> lock(m_tempList->childMutex());
                for(unsigned int i = 0; i < std::min(m_resList->count(), m_tempList->count()); i++) {
                    pts.insert(std::pair<double, double>(log(*(*m_tempList)[i]), log(*(*m_resList)[i])));
                }
            }
        }
        if(pts.size() < 4)
            throw XKameError(
                i18n("XApproxThermometer, Too small number of points"), __FILE__, __LINE__);
        approx.reset(new CSplineApprox(pts));
        m_approx_inv = approx;
    }
    return exp(approx->approx(log(temp)));
}

