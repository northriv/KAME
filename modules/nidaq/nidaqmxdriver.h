/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef NIDAQMXDRIVER_H_
#define NIDAQMXDRIVER_H_

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include "interface.h"
#include "driver.h"
#include "atomic_queue.h"

#ifdef HAVE_NI_DAQMX
	#include <NIDAQmx.h>
	#define CHECK_DAQMX_ERROR(ret) XNIDAQmxInterface::checkDAQmxError(ret, __FILE__, __LINE__)
	
	/*#define CHECK_DAQMX_RET(ret, msg) {dbgPrint(# ret);\
	  if(CHECK_DAQMX_ERROR(ret, msg) > 0) {gWarnPrint(QString(msg) + " " + XNIDAQmxInterface::getNIDAQmxErrMessage()); } }
	*/
	#define CHECK_DAQMX_RET(ret) {int _code = ret; \
		if(CHECK_DAQMX_ERROR(_code) > 0) {gWarnPrint(XNIDAQmxInterface::getNIDAQmxErrMessage(_code)); } }
#else
	#define CHECK_DAQMX_ERROR(ret) (0)
	#define CHECK_DAQMX_RET(ret)
	typedef unsigned int TaskHandle;
	typedef int64_t int64;
	typedef uint64_t uInt64;
	typedef int32_t int32;
	typedef uint32_t uInt32;
	typedef int16_t int16;
	typedef uint16_t uInt16;
	typedef int8_t int8;
	typedef uint8_t uInt8;
	typedef double float64;
#endif //HAVE_NI_DAQMX

class XNIDAQmxTask;

class XNIDAQmxInterface : public XInterface
{
	XNODE_OBJECT
protected:
	XNIDAQmxInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
public:
	virtual ~XNIDAQmxInterface() {}
 
	static QString getNIDAQmxErrMessage();
	static QString getNIDAQmxErrMessage(int status);
	static int checkDAQmxError(int ret, const char*file, int line);

	virtual bool isOpened() const {return m_devname.length();}
  
	//! e.g. "Dev1".
	const char*devName() const {return m_devname.c_str();}
	//! Split camma-separated strings.
	static void parseList(const char *list, std::deque<std::string> &buf);
  
	//! Each task must call this to tell the reference frequency.
	void synchronizeClock(TaskHandle task);
    
	class XNIDAQmxRoute {
	public:
	  	XNIDAQmxRoute(const char*src, const char*dst, int *ret = NULL);
	  	~XNIDAQmxRoute();
	private:
	  	std::string m_src, m_dst;
	};
	
	//! e.g. "PCI-6111".
	const char* productType() const {
		const ProductInfo* p = m_productInfo;
		return p ? p->type : 0L;
	}
	//! e.g. "S", "M".
	const char* productSeries() const {
		const ProductInfo* p = m_productInfo;
		return p ? p->series : 0L;
	}
	//! e.g. "PCI", "PXI". Never "PCIe" or "PXIe".
	const char* busArchType() const;

	enum FLAGS {
		FLAG_BUGGY_DMA_AO = 0x10u, FLAG_BUGGY_DMA_AI = 0x20u, 
		FLAG_BUGGY_DMA_DI = 0x40u, FLAG_BUGGY_DMA_DO = 0x80u,
		FLAG_BUGGY_XFER_COND_AO = 0x100u, FLAG_BUGGY_XFER_COND_AI = 0x200u,
		FLAG_BUGGY_XFER_COND_DI = 0x400u, FLAG_BUGGY_XFER_COND_DO = 0x800u};
	//! e.g. FLAG_BUGGY_DMA_AO.
	int productFlags() const {return m_productInfo->flags;}
	//! \return 0 if hw timed transfer is not supported.
	double maxAIRate(unsigned int /*num_scans*/) const {return m_productInfo->ai_max_rate;}
	double maxAORate(unsigned int /*num_scans*/) const {return m_productInfo->ao_max_rate;}
	double maxDIRate(unsigned int /*num_scans*/) const {return m_productInfo->di_max_rate;}
	double maxDORate(unsigned int /*num_scans*/) const {return m_productInfo->do_max_rate;}
	  
	class SoftwareTrigger : public enable_shared_from_this<SoftwareTrigger> {
	protected:
		SoftwareTrigger(const char *label, unsigned int bits);
	public:
		static shared_ptr<SoftwareTrigger> create(const char *label, unsigned int bits);
		static void unregister(const shared_ptr<SoftwareTrigger> &);
		const char *label() const {return m_label.c_str();}
		void setArmTerm(const char *arm_term) {m_armTerm = arm_term;}
		const char *armTerm() const {return m_armTerm.c_str();}
			
		void start(float64 freq);
		float64 freq() const {return m_freq;} //!< [Hz].
		unsigned int bits() const {return m_bits;}
		void stop();
		void forceStamp(uint64_t now, float64 freq);
		void stamp(uint64_t cnt);
		template <typename T>
		void changeValue(T oldval, T val, uint64_t time) {
			if(((m_risingEdgeMask & val) & (m_risingEdgeMask & ~oldval))
			   || ((m_fallingEdgeMask & ~val) & (m_fallingEdgeMask & oldval))) {
				if(time < m_endOfBlank) return;
				stamp(time);
			}
		}
		void connect(uint32_t rising_edge_mask, 
					 uint32_t falling_edge_mask) throw (XInterface::XInterfaceError &);
		void disconnect();
		//! \arg blankterm in seconds.
		void setBlankTerm(float64 blankterm) {
			m_blankTerm = lrint(blankterm * freq());
			memoryBarrier();
		}
		//! for restarting connected task.
		XTalker<shared_ptr<SoftwareTrigger> > &onStart() {return m_onStart;}
		//! for changeing list.
		static XTalker<shared_ptr<SoftwareTrigger> > &onChange() {return s_onChange;}
			
		void clear(uint64_t now, float64 freq);
		uint64_t tryPopFront(uint64_t threshold, float64 freq);
			
		typedef std::deque<shared_ptr<XNIDAQmxInterface::SoftwareTrigger> > SoftwareTriggerList;
		typedef SoftwareTriggerList::iterator SoftwareTriggerList_it;
		static const atomic_shared_ptr<SoftwareTriggerList> &virtualTrigList() {
			return s_virtualTrigList;
		}
	private:
		void _clear();
		const std::string m_label;
		std::string m_armTerm;
		unsigned int m_bits;
		uint32_t m_risingEdgeMask, m_fallingEdgeMask;
		uint64_t m_blankTerm, m_endOfBlank;
		float64 m_freq; //!< [Hz].
		enum {QUEUE_SIZE = 8192};
		typedef atomic_queue_reserved<uint64_t, QUEUE_SIZE> FastQueue;
		FastQueue m_fastQueue;
		typedef std::deque<uint64_t> SlowQueue;
		SlowQueue m_slowQueue;
		atomic<unsigned int> m_slowQueueSize;
		XMutex m_mutex;
		XTalker<shared_ptr<SoftwareTrigger> > m_onStart;
		static XTalker<shared_ptr<SoftwareTrigger> > s_onChange;
		static atomic_shared_ptr<SoftwareTriggerList> s_virtualTrigList;
	};
protected:
	virtual void open() throw (XInterfaceError &);
	//! This can be called even if has already closed.
	virtual void close() throw (XInterfaceError &);
private:
	struct ProductInfo {
	  	const char *type;
	  	const char *series;
	  	int flags;
	  	unsigned long ai_max_rate; //!< [kHz]
	  	unsigned long ao_max_rate; //!< [kHz]
	  	unsigned long di_max_rate; //!< [kHz]
	  	unsigned long do_max_rate; //!< [kHz]
	};
	friend class SoftwareTrigger;
	std::string m_devname;
	const ProductInfo* m_productInfo;
	static const ProductInfo sc_productInfoList[];
};

template<class tDriver>
class XNIDAQmxDriver : public tDriver
{
	XNODE_OBJECT
protected:
	XNIDAQmxDriver(const char *name, bool runtime, 
				   const shared_ptr<XScalarEntryList> &scalarentries,
				   const shared_ptr<XInterfaceList> &interfaces,
				   const shared_ptr<XThermometerList> &thermometers,
				   const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XNIDAQmxDriver() {}
protected:
	const shared_ptr<XNIDAQmxInterface> &interface() const {return m_interface;}
	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open() throw (XInterface::XInterfaceError &) {this->start();}
	//! Be called during stopping driver. Call interface()->stop() inside this routine.
	virtual void close() throw (XInterface::XInterfaceError &) {interface()->stop();}
	void onOpen(const shared_ptr<XInterface> &);
	void onClose(const shared_ptr<XInterface> &);
	//! This should not cause an exception.
	virtual void afterStop();
private:
	shared_ptr<XListener> m_lsnOnOpen, m_lsnOnClose;
  
	const shared_ptr<XNIDAQmxInterface> m_interface;
};
template<class tDriver>
XNIDAQmxDriver<tDriver>::XNIDAQmxDriver(const char *name, bool runtime, 
										const shared_ptr<XScalarEntryList> &scalarentries,
										const shared_ptr<XInterfaceList> &interfaces,
										const shared_ptr<XThermometerList> &thermometers,
										const shared_ptr<XDriverList> &drivers) :
    tDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_interface(XNode::create<XNIDAQmxInterface>("Interface", false,
												 dynamic_pointer_cast<XDriver>(this->shared_from_this())))
{
    interfaces->insert(m_interface);
    m_lsnOnOpen = interface()->onOpen().connectWeak(
		this->shared_from_this(), &XNIDAQmxDriver<tDriver>::onOpen);
    m_lsnOnClose = interface()->onClose().connectWeak( 
    	this->shared_from_this(), &XNIDAQmxDriver<tDriver>::onClose);
}
template<class tDriver>
void
XNIDAQmxDriver<tDriver>::onOpen(const shared_ptr<XInterface> &)
{
	try {
		open();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Opening interface failed, because "));
		close();
	}
}
template<class tDriver>
void
XNIDAQmxDriver<tDriver>::onClose(const shared_ptr<XInterface> &)
{
	try {
		this->stop();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Stopping driver failed, because "));
	}
}
template<class tDriver>
void
XNIDAQmxDriver<tDriver>::afterStop() {
	try {
		this->close();
	}
	catch (XInterface::XInterfaceError &e) {
		e.print();
	}
}

#endif /*NIDAQMXDRIVER_H_*/
