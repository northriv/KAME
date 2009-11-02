/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
//---------------------------------------------------------------------------
#include <math.h>

#include "thermometer.h"
#include "cspline.h"
#include "rand.h"

//---------------------------------------------------------------------------
XThermometerList::XThermometerList(const char *name, bool runtime)
 : XCustomTypeListNode<XThermometer>(name, runtime) {
}

DECLARE_TYPE_HOLDER(XThermometerList)

REGISTER_TYPE(XThermometerList, LakeShore, "LakeShore");
REGISTER_TYPE(XThermometerList, ScientificInstruments, "Scientific Instruments");
REGISTER_TYPE(XThermometerList, ApproxThermometer, "Cubic-spline");

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
      double t = randMT19937();
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
	      double t = randMT19937();
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
  XNode::NodeList::reader zu_list(zu()->children());
  if(!zu_list) return 0;
  XNode::NodeList::reader zl_list(zl()->children());
  if(!zl_list) return 0;
  for(n = 0; n < zu_list->size(); n++)
    {
        double zu = *dynamic_pointer_cast<XDoubleNode>(zu_list->at(n));
        double zl = *dynamic_pointer_cast<XDoubleNode>(zl_list->at(n));
      u = (z - zu + z - zl) / (zu - zl);
      if((u >= -1) && (u <= 1)) break;
    }
  if(n >= zu_list->size())
    return 0;
  XNode::NodeList::reader ai_list(ai()->children());
  if(!ai_list) return 0;
  XNode::NodeList::reader ai_n_list(ai_list->at(n)->children());
  if(!ai_n_list) return 0;
  for(unsigned int i = 0; i < ai_n_list->size(); i++)
    {
        double ai_n_i = *dynamic_pointer_cast<XDoubleNode>(ai_n_list->at(i));
        temp += ai_n_i * cos(i * acos(u));
    }
  return temp;
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
      double t = randMT19937();
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
	      double t = randMT19937();
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
      XNode::NodeList::reader abcde_list(abcde()->children());
      if(!abcde_list) return 0;
      if(abcde_list->size() >= 5) {
        double a = *dynamic_pointer_cast<XDoubleNode>(abcde_list->at(0));
        double b = *dynamic_pointer_cast<XDoubleNode>(abcde_list->at(1));
        double c = *dynamic_pointer_cast<XDoubleNode>(abcde_list->at(2));
        double d = *dynamic_pointer_cast<XDoubleNode>(abcde_list->at(3));
        double e = *dynamic_pointer_cast<XDoubleNode>(abcde_list->at(4));
  		y = (a + c*lx + e*lx*lx)
  			/ (1.0 + b*lx + d*lx*lx);
      }
      return y;
    }
  else
    {
        XNode::NodeList::reader abc_list(abc()->children());
        if(!abc_list) return 0;
        if(abc_list->size() >= 3) {
            double a = *dynamic_pointer_cast<XDoubleNode>(abc_list->at(0));
            double b = *dynamic_pointer_cast<XDoubleNode>(abc_list->at(1));
            double c = *dynamic_pointer_cast<XDoubleNode>(abc_list->at(2));
      		y = 1.0/(a + b*res*lx + c*res*res);
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
      XNode::NodeList::reader res_list(m_resList->children());
      if(!res_list) return 0;
      XNode::NodeList::reader temp_list(m_tempList->children());
      if(!temp_list) return 0;
        for(unsigned int i = 0; i < std::min(res_list->size(), temp_list->size()); i++) {
            double res = *dynamic_pointer_cast<XDoubleNode>(res_list->at(i));
            double temp = *dynamic_pointer_cast<XDoubleNode>(temp_list->at(i));
            pts.insert(std::pair<double, double>(log(res), log(temp)));
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
      XNode::NodeList::reader res_list(m_resList->children());
      if(!res_list) return 0;
      XNode::NodeList::reader temp_list(m_tempList->children());
      if(!temp_list) return 0;
        for(unsigned int i = 0; i < std::min(res_list->size(), temp_list->size()); i++) {
            double res = *dynamic_pointer_cast<XDoubleNode>(res_list->at(i));
            double temp = *dynamic_pointer_cast<XDoubleNode>(temp_list->at(i));
            pts.insert(std::pair<double, double>(log(temp), log(res)));
        }
        if(pts.size() < 4)
            throw XKameError(
                i18n("XApproxThermometer, Too small number of points"), __FILE__, __LINE__);
        approx.reset(new CSplineApprox(pts));
        m_approx_inv = approx;
    }
    return exp(approx->approx(log(temp)));
}

