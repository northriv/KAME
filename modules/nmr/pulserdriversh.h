/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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
#include <vector>

//! My pulser driver
class XSHPulser : public XCharDeviceDriver<XPulser>
{
	XNODE_OBJECT
protected:
	XSHPulser(const char *name, bool runtime,
			  const shared_ptr<XScalarEntryList> &scalarentries,
			  const shared_ptr<XInterfaceList> &interfaces,
			  const shared_ptr<XThermometerList> &thermometers,
			  const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XSHPulser() {}

protected:
    //! send patterns to pulser or turn-off
    virtual void changeOutput(bool output, unsigned int blankpattern);
    //! convert RelPatList to native patterns
    virtual void createNativePatterns();
    //! time resolution [ms]
    virtual double resolution() const;
    virtual double resolutionQAM() const {return resolution();}
    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const {return resolution();}
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return true;}

private:
	int setAUX2DA(double volt, int addr);
	int insertPreamble(unsigned short startpattern);
	int finishPulse();
	//! Add 1 pulse pattern
	//! \param term a period to next pattern
	//! \param pattern a pattern for digital, to appear
	int pulseAdd(uint64_t term, uint32_t pattern, bool firsttime, bool dryrun);
	uint32_t m_lastPattern;
	uint64_t m_dmaTerm;  
  
	std::vector<unsigned char> m_zippedPatterns;
  
	int m_waveformPos[PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX];
  
};
