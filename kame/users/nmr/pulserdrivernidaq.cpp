#include "pulserdrivernidaq.h"

#ifdef HAVE_NI_DAQMX

#include "interface.h"
#include <klocale.h>

double XNIDAQSSeriesPulser::resolution() const {
     return (0.5/(1e3));
}
double XNIDAQSSeriesPulser::resolutionQAM() const {
const unsigned int OVERSAMP_AO = 1;
     return resolution() / OVERSAMP_AO;
}
double XNIDAQMSeriesPulser::resolution() const {
     return (1.0/(1e3));
}
double XNIDAQMSeriesWithSSeriesPulser::resolution() const {
     return (10.0/(1e3));
}
double XNIDAQMSeriesWithSSeriesPulser::resolutionQAM() const {
const unsigned int OVERSAMP_AO = 1;
     return resolution() / OVERSAMP_AO;
}

XNIDAQMSeriesWithSSeriesPulser::XNIDAQMSeriesWithSSeriesPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XNIDAQmxPulser(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_ao_interface(XNode::create<XNIDAQmxInterface>("SubInterface", false,
            dynamic_pointer_cast<XDriver>(this->shared_from_this())))
{
    interfaces->insert(m_ao_interface);
    m_lsnOnOpenAO = m_ao_interface->onOpen().connectWeak(false,
    	 this->shared_from_this(), &XNIDAQMSeriesWithSSeriesPulser::onOpenAO);
    m_lsnOnCloseAO = m_ao_interface->onClose().connectWeak(false, 
    	this->shared_from_this(), &XNIDAQMSeriesWithSSeriesPulser::onCloseAO);

    m_ctr_interface = intfDO();
}
void
XNIDAQSSeriesPulser::open() throw (XInterface::XInterfaceError &)
{
 	openAODO();
	this->start();	
}
void
XNIDAQMSeriesPulser::open() throw (XInterface::XInterfaceError &)
{
 	openDO();
	this->start();	
}
void
XNIDAQMSeriesWithSSeriesPulser::open() throw (XInterface::XInterfaceError &)
{
   m_ctr_interface = intfDO();
 	openDO();
	this->start();	
}
void
XNIDAQMSeriesWithSSeriesPulser::onOpenAO(const shared_ptr<XInterface> &)
{
	try {
	    m_ctr_interface = intfAO();
		openAODO();
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
	    close();
	}
}
void
XNIDAQMSeriesWithSSeriesPulser::onCloseAO(const shared_ptr<XInterface> &)
{
	stop();
}

#endif //HAVE_NI_DAQMX
