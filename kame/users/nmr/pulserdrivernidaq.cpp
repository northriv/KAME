#include "pulserdrivernidaq.h"

#ifdef HAVE_NI_DAQMX

#include "interface.h"
#include <klocale.h>

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
	if(std::string(interface()->productInfo()->series) != "S")
		throw XInterface::XInterfaceError(KAME::i18n("Product-type mismatch."), __FILE__, __LINE__);
 	openAODO();
	this->start();	
}
void
XNIDAQMSeriesPulser::open() throw (XInterface::XInterfaceError &)
{
	if(std::string(interface()->productInfo()->series) != "M")
		throw XInterface::XInterfaceError(KAME::i18n("Product-type mismatch."), __FILE__, __LINE__);
 	openDO();
	this->start();	
}
void
XNIDAQMSeriesWithSSeriesPulser::open() throw (XInterface::XInterfaceError &)
{
	if(std::string(interface()->productInfo()->series) != "M")
		throw XInterface::XInterfaceError(KAME::i18n("Product-type mismatch."), __FILE__, __LINE__);
 	m_ctr_interface = intfDO();
 	openDO();
	this->start();	
}
void
XNIDAQMSeriesWithSSeriesPulser::onOpenAO(const shared_ptr<XInterface> &)
{
	try {
		if(std::string(intfAO()->productInfo()->series) != "S")
			throw XInterface::XInterfaceError(KAME::i18n("Product-type mismatch."), __FILE__, __LINE__);
	    m_ctr_interface = intfAO();
		openAODO();
		//DMA is slower than interrupts!
		CHECK_DAQMX_RET(DAQmxSetAODataXferMech(m_taskAO, 
	    	formatString("%s/ao0:1", intfAO()->devName()).c_str(),
			DAQmx_Val_Interrupts));
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
