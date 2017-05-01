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
    virtual ~XThamwayPulser() = default;

	struct Payload : public XPulser::Payload {
    private:
		friend class XThamwayPulser;
        friend class XThamwayUSBPulser;
        friend class XThamwayCharPulser;
        struct Pulse {
			uint32_t term_n_cmd;
			uint32_t data;
		};
        std::vector<Pulse> m_patterns;
    };

	//! time resolution [ms]
    virtual double resolution() const override;
protected:
    virtual void getStatus(bool *running = 0L, bool *extclk_det = 0L) = 0;

    //! Converts RelPatList to native patterns
    virtual void createNativePatterns(Transaction &tr) override;
    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const override;
private:
	//! Add 1 pulse pattern
	//! \param term a period of the pattern to appear
	//! \param pattern a pattern for digital, to appear
	int pulseAdd(Transaction &tr, uint64_t term, uint16_t pattern);
};

#if defined USE_THAMWAY_USB
#include "thamwayusbinterface.h"
    class XThamwayPGCUSBInterface : public XThamwayFX2USBInterface {
    public:
        XThamwayPGCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
            : XThamwayFX2USBInterface(name, runtime, driver, 0, "PG32") {}
        virtual ~XThamwayPGCUSBInterface() = default;
    };
    #define ADDR_OFFSET_PGQAM 0x60
    class XThamwayPGQAMCUSBInterface : public XThamwayFX2USBInterface {
    public:
        XThamwayPGQAMCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
            : XThamwayFX2USBInterface(name, runtime, driver, ADDR_OFFSET_PGQAM, "PG027QAM") {}
        virtual ~XThamwayPGQAMCUSBInterface() = default;
    };

    class XThamwayUSBPulser : public XCharDeviceDriver<XThamwayPulser, XThamwayPGCUSBInterface> {
    public:
        XThamwayUSBPulser(const char *name, bool runtime,
            Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
            XCharDeviceDriver<XThamwayPulser, XThamwayPGCUSBInterface>(name, runtime, ref(tr_meas), meas) {}
        virtual ~XThamwayUSBPulser() = default;
    protected:
        virtual void open() throw (XKameError &) override;
        //! Sends patterns to pulser or turns off.
        virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) override;

        virtual void getStatus(bool *running = 0L, bool *extclk_det = 0L) override;

        enum : uint32_t {QAM_PERIOD = 10}; //40ns * 10

        virtual double resolutionQAM() const override {return 0.0;}
        //! existense of AO ports.
        virtual bool hasQAMPorts() const override {return !!interfaceQAM();}

        virtual shared_ptr<XThamwayPGQAMCUSBInterface> interfaceQAM() const {return nullptr;}
    private:
    };

    class XThamwayUSBPulserWithQAM : public XThamwayUSBPulser {
    public:
        XThamwayUSBPulserWithQAM(const char *name, bool runtime,
            Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
        virtual ~XThamwayUSBPulserWithQAM() = default;
    protected:
        virtual void open() throw (XKameError &) override;
        virtual void close() throw (XKameError &) override;

        virtual double resolutionQAM() const override {return resolution() * QAM_PERIOD;}

        virtual shared_ptr<XThamwayPGQAMCUSBInterface> interfaceQAM() const override {return m_interfaceQAM;}
    private:
        const shared_ptr<XThamwayPGQAMCUSBInterface> m_interfaceQAM;
    };
#endif
class XThamwayCharPulser : public XCharDeviceDriver<XThamwayPulser>  {
public:
    XThamwayCharPulser(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
        XCharDeviceDriver<XThamwayPulser>(name, runtime, ref(tr_meas), meas) {}
    virtual ~XThamwayCharPulser() = default;
protected:
    virtual void open() throw (XKameError &) override;
    //! Sends patterns to pulser or turns off.
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) override;

    virtual void getStatus(bool *running = 0L, bool *extclk_det = 0L) override;

    virtual double resolutionQAM() const override {return 0.0;}
    //! existense of AO ports.
    virtual bool hasQAMPorts() const override {return false;}
};
