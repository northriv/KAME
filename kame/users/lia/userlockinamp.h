#ifndef userlockinampH
#define userlockinampH

#include "lockinamp.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------
//Stanford Research SR830 Lock-in Amplifier
class XSR830 : public XCharDeviceDriver<XLIA>
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

  //! Be called just after opening interface. Call start() inside this routine appropriately.
  virtual void open() throw (XInterface::XInterfaceError &);
  //! Be called for closing interfaces.
  virtual void afterStop();

  int m_cCount;
};

//ANDEEN HAGERLING 2500A 1kHz Ultra-Precision Capcitance Bridge
class XAH2500A : public XCharDeviceDriver<XLIA>
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
  //! Be called just after opening interface. Call start() inside this routine appropriately.
  virtual void open() throw (XInterface::XInterfaceError &);
  //! Be called for closing interfaces.
  virtual void afterStop();
};

#endif
