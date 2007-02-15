#ifndef PULSERDRIVERNIDAQ_H_
#define PULSERDRIVERNIDAQ_H_

#include "pulserdrivernidaqmx.h"

#ifdef HAVE_NI_DAQMX

class XNIDAQSSeriesPulser : public XNIDAQmxPulser
{
 XNODE_OBJECT
 protected:
  XNIDAQSSeriesPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
   : XNIDAQmxPulser(name, runtime, scalarentries, interfaces, thermometers, drivers) {}
 public:
  virtual ~XNIDAQSSeriesPulser() {}

 protected:
	virtual void open() throw (XInterface::XInterfaceError &);
    //! time resolution [ms]
    virtual double resolution() const;
    virtual double resolutionQAM() const;
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return true;}
};

class XNIDAQMSeriesPulser : public XNIDAQmxPulser
{
 XNODE_OBJECT
 protected:
  XNIDAQMSeriesPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
    : XNIDAQmxPulser(name, runtime, scalarentries, interfaces, thermometers, drivers) {}
 public:
  virtual ~XNIDAQMSeriesPulser() {}

 protected:
	virtual void open() throw (XInterface::XInterfaceError &);
    //! time resolution [ms]
    virtual double resolution() const;
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return false;}
};

class XNIDAQMSeriesWithSSeriesPulser : public XNIDAQmxPulser
{
 XNODE_OBJECT
 protected:
  XNIDAQMSeriesWithSSeriesPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XNIDAQMSeriesWithSSeriesPulser() {}

 protected:
	virtual void open() throw (XInterface::XInterfaceError &);
    //! time resolution [ms]
    virtual double resolution() const;
    virtual double resolutionQAM() const;
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return true;}
    
	virtual const shared_ptr<XNIDAQmxInterface> &intfAO() const {return m_ao_interface;} 
	virtual const shared_ptr<XNIDAQmxInterface> &intfCtr() const {return m_ctr_interface;} 
 private:
 
	const shared_ptr<XNIDAQmxInterface> m_ao_interface;
	shared_ptr<XNIDAQmxInterface> m_ctr_interface;
	shared_ptr<XListener> m_lsnOnOpenAO, m_lsnOnCloseAO;
	void onOpenAO(const shared_ptr<XInterface> &);
	void onCloseAO(const shared_ptr<XInterface> &);
};

#endif //HAVE_NI_DAQMX

#endif /*PULSERDRIVERNIDAQ_H_*/
