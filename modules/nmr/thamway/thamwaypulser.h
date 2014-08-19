/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "pulserdriver.h"
#include "chardevicedriver.h"
#include "charinterface.h"
#include <vector>

class XThamwayUSBPulser;
class XThamwayCharPulser;

//! pulser driver
class XThamwayPulser : public XPulser {
public:
	XThamwayPulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XThamwayPulser() {}

	struct Payload : public XPulser::Payload {
    private:
		friend class XThamwayPulser;
        friend class XThamwayUSBPulser;
        friend class XThamwayCharPulser;
        struct Pulse {
			uint32_t term_n_cmd;
			uint32_t data;
		};
		std::deque<Pulse> m_patterns;
	};

	//! time resolution [ms]
    virtual double resolution() const;
protected:
    virtual void open() throw (XKameError &) = 0;
    //! Sends patterns to pulser or turns off.
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) = 0;

    virtual void getStatus(bool *running = 0L, bool *extclk_det = 0L) = 0;

    //! Converts RelPatList to native patterns
    virtual void createNativePatterns(Transaction &tr);
    virtual double resolutionQAM() const {return 0.0;}
    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const;
    //! existense of AO ports.
    virtual bool hasQAMPorts() const {return false;}
private:
	//! Add 1 pulse pattern
	//! \param term a period of the pattern to appear
	//! \param pattern a pattern for digital, to appear
	int pulseAdd(Transaction &tr, uint64_t term, uint16_t pattern);
};

#if defined USE_EZUSB
    #include "ezusbthamway.h"
    class XThamwayPGCUSBInterface : public XWinCUSBInterface {
    public:
        XThamwayPGCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
            : XWinCUSBInterface(name, runtime, driver, 0, "PG32") {}
        virtual ~XThamwayPGCUSBInterface() {}
    };

    class XThamwayUSBPulser : public XCharDeviceDriver<XThamwayPulser, XThamwayPGCUSBInterface> {
    public:
        XThamwayUSBPulser(const char *name, bool runtime,
            Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
            XCharDeviceDriver<XThamwayPulser, XThamwayPGCUSBInterface>(name, runtime, ref(tr_meas), meas) {}
        virtual ~XThamwayUSBPulser() {}
    protected:
        virtual void open() throw (XKameError &);
        //! Sends patterns to pulser or turns off.
        virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern);

        virtual void getStatus(bool *running = 0L, bool *extclk_det = 0L);
    };
#endif
class XThamwayCharPulser : public XCharDeviceDriver<XThamwayPulser>  {
public:
    XThamwayCharPulser(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
        XCharDeviceDriver<XThamwayPulser>(name, runtime, ref(tr_meas), meas) {}
    virtual ~XThamwayCharPulser() {}
protected:
    virtual void open() throw (XKameError &);
    //! Sends patterns to pulser or turns off.
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern);

    virtual void getStatus(bool *running = 0L, bool *extclk_det = 0L);
};
