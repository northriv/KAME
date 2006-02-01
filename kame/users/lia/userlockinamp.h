#ifndef userlockinampH
#define userlockinampH

#include "lockinamp.h"
//---------------------------------------------------------------------------
//Stanford Research SR830 Lock-in Amplifier
class XSR830 : public XLIA
{
 XNODE_OBJECT
 protected:
  XSR830(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 protected:
  virtual void get(double *cos, double *sin);
  virtual void changeOutput(double volt);
  virtual void changeFreq(double freq);
  virtual void changeSensitivity(int);
  virtual void changeTimeConst(int);
  virtual void afterStart();
  virtual void beforeStop();

  int m_cCount;
};

//ANDEEN HAGERLING 2500A 1kHz Ultra-Precision Capcitance Bridge
class XAH2500A : public XLIA
{
 XNODE_OBJECT
 protected:
  XAH2500A(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 protected:
  virtual void get(double *cos, double *sin);
  virtual void changeOutput(double volt);
  virtual void changeFreq(double freq);
  virtual void changeSensitivity(int);
  virtual void changeTimeConst(int);
  virtual void afterStart();
  virtual void beforeStop();
};

#endif
