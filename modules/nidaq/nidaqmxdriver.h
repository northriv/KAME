/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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

#include "interface.h"
#include "driver.h"
#include "atomic_queue.h"

#ifdef HAVE_NI_DAQMX
	#include <NIDAQmx.h>
	#define CHECK_DAQMX_ERROR(ret) XNIDAQmxInterface::checkDAQmxError(ret, __FILE__, __LINE__)
	
	/*#define CHECK_DAQMX_RET(ret, msg) {dbgPrint(# ret);\
	  if(CHECK_DAQMX_ERROR(ret, msg) > 0) {gWarnPrint(QString(msg) + " " + XNIDAQmxInterface::getNIDAQmxErrMessage()); } }
	*/
	#define CHECK_DAQMX_RET(ret) {int code__ = ret; \
		if(CHECK_DAQMX_ERROR(code__) > 0) {gWarnPrint(XNIDAQmxInterface::getNIDAQmxErrMessage(code__)); } }
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

class XNIDAQmxInterface : public XInterface {
public:
	XNIDAQmxInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
	virtual ~XNIDAQmxInterface() {}
 
	static XString getNIDAQmxErrMessage();
	static XString getNIDAQmxErrMessage(int status);
	static int checkDAQmxError(int ret, const char*file, int line);

	virtual bool isOpened() const {return m_devname.length();}
  
	//! e.g. "Dev1".
	const char*devName() const {return m_devname.c_str();}
	//! Split camma-separated strings.
	static void parseList(const char *list, std::deque<XString> &buf);
  
	void synchronizeClock(TaskHandle task);
    
	class XNIDAQmxRoute {
	public:
	  	XNIDAQmxRoute(const char*src, const char*dst, int *ret = NULL);
	  	~XNIDAQmxRoute();
	private:
	  	XString m_src, m_dst;
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

	//! Stores and reads time stamps between synchronized devices.
	class SoftwareTrigger : public enable_shared_from_this<SoftwareTrigger> {
	protected:
		SoftwareTrigger(const char *label, unsigned int bits);
	public:
		static shared_ptr<SoftwareTrigger> create(const char *label, unsigned int bits);
		static void unregister(const shared_ptr<SoftwareTrigger> &);
		const char *label() const {return m_label.c_str();}
		void setArmTerm(const char *arm_term) {m_armTerm = arm_term;}
		const char *armTerm() const {return m_armTerm.c_str();}
		bool isPersistentCoherentMode() const {return m_isPersistentCoherent;}
		void setPersistentCoherentMode(bool x) {m_isPersistentCoherent = x;}
		void start(float64 freq);
		float64 freq() const {return m_freq;} //!< [Hz].
		unsigned int bits() const {return m_bits;}
		void stop();
		void forceStamp(uint64_t now, float64 freq);
		void stamp(uint64_t cnt);
		//! \param time unit in 1/freq().
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
		//! \param blankterm in seconds, not to stamp so frequently.
		void setBlankTerm(float64 blankterm) {
			m_blankTerm = llrint(blankterm * freq());
			memoryBarrier();
		}
        using STRGTalker = Transactional::Talker<shared_ptr<SoftwareTrigger>>;
		//! for restarting connected task.
        STRGTalker &onStart() {return m_onStart;}
        //! for changing list.
        static STRGTalker &onChange() {return s_onChange;}
		//! clears all time stamps.
		void clear();
		//! clears past time stamps.
		void clear(uint64_t now, float64 freq);
		//! \return if not, zero will be returned.
		//! \param freq frequency of reader.
		//! \param threshold upper bound to be pop, unit in 1/\a freq (2nd param.).
		uint64_t tryPopFront(uint64_t threshold, float64 freq);
			
		typedef std::deque<shared_ptr<XNIDAQmxInterface::SoftwareTrigger> > SoftwareTriggerList;
		typedef SoftwareTriggerList::iterator SoftwareTriggerList_it;
		static const atomic_shared_ptr<SoftwareTriggerList> &virtualTrigList() {
			return s_virtualTrigList;
		}
	private:
		void clear_();
		const XString m_label;
		XString m_armTerm;
		unsigned int m_bits;
		uint32_t m_risingEdgeMask, m_fallingEdgeMask;
		uint64_t m_blankTerm;
		uint64_t m_endOfBlank; //!< next stamp must not be less than this.
		float64 m_freq; //!< [Hz].
		enum {QUEUE_SIZE = 8192};
		typedef atomic_queue_reserved<uint64_t, QUEUE_SIZE> FastQueue;
		FastQueue m_fastQueue; //!< recorded stamps.
		typedef std::deque<uint64_t> SlowQueue;
		SlowQueue m_slowQueue; //!< recorded stamps, when \a m_fastQueue is full.
		atomic<unsigned int> m_slowQueueSize;
		XMutex m_mutex; //!< for \a m_slowQueue.
        STRGTalker m_onStart;
        static STRGTalker s_onChange;
		static atomic_shared_ptr<SoftwareTriggerList> s_virtualTrigList;
		bool m_isPersistentCoherent;
	};
protected:
	virtual void open() throw (XInterfaceError &);
	//! This can be called even if has already closed.
	virtual void close() throw (XInterfaceError &);
private:
	//! \return true if an external source is detected and rounted.
	bool routeExternalClockSource(const char *dev, const char *inp_term);

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
	XString m_devname;
	const ProductInfo* m_productInfo;
	static const ProductInfo sc_productInfoList[];
};

template<class tDriver>
class XNIDAQmxDriver : public tDriver {
public:
	XNIDAQmxDriver(const char *name, bool runtime, 
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XNIDAQmxDriver() {}
protected:
	const shared_ptr<XNIDAQmxInterface> &interface() const {return m_interface;}
	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open() throw (XKameError &) {this->start();}
	//! Be called during stopping driver. Call interface()->stop() inside this routine.
	virtual void close() throw (XKameError &) {interface()->stop();}
	void onOpen(const Snapshot &shot, XInterface *);
	void onClose(const Snapshot &shot, XInterface *);
	//! This should not cause an exception.
	virtual void closeInterface();
private:
	shared_ptr<XListener> m_lsnOnOpen, m_lsnOnClose;
  
	const shared_ptr<XNIDAQmxInterface> m_interface;
};
template<class tDriver>
XNIDAQmxDriver<tDriver>::XNIDAQmxDriver(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    tDriver(name, runtime, ref(tr_meas), meas),
	m_interface(XNode::create<XNIDAQmxInterface>("Interface", false,
												 dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_interface);
    this->iterate_commit([=](Transaction &tr){
	    m_lsnOnOpen = tr[ *interface()].onOpen().connectWeakly(
			this->shared_from_this(), &XNIDAQmxDriver<tDriver>::onOpen);
	    m_lsnOnClose = tr[ *interface()].onClose().connectWeakly(
	    	this->shared_from_this(), &XNIDAQmxDriver<tDriver>::onClose);
    });
}
template<class tDriver>
void
XNIDAQmxDriver<tDriver>::onOpen(const Snapshot &shot, XInterface *) {
	try {
		open();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + i18n(": Opening driver failed, because "));
		onClose(shot, NULL);
	}
}
template<class tDriver>
void
XNIDAQmxDriver<tDriver>::onClose(const Snapshot &shot, XInterface *) {
	try {
		this->stop();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + i18n(": Stopping driver failed, because "));
		closeInterface();
	}
}
template<class tDriver>
void
XNIDAQmxDriver<tDriver>::closeInterface() {
	try {
		this->close();
	}
	catch (XInterface::XInterfaceError &e) {
		e.print();
	}
}

#endif /*NIDAQMXDRIVER_H_*/
