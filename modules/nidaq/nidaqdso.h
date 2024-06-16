/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef nidaqdsoH
#define nidaqdsoH

#include "dsorealtimeacq.h"
#include "nidaqmxdriver.h"

//! Software DSO w/ NI DAQmx
class XNIDAQmxDSO : public XRealTimeAcqDSO<XNIDAQmxDriver<XDSO>> {
public:
	XNIDAQmxDSO(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XNIDAQmxDSO() = default;
protected:
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;
    //! Be called during stopping driver. Call interface()->stop() inside this routine.
    virtual void close() override;

    //! Changes the instrument state so that it can wait for a trigger (arm).
    virtual void startAcquision() override;
    //! Prepares the instrument state just before startAcquision().
    virtual void commitAcquision() override;
    //! From a triggerable state to a commited state.
    virtual void stopAcquision() override;
    //! From any state to unconfigured state.
    virtual void clearAcquision() override;
    //! \return # of configured channels.
    virtual unsigned int getNumOfChannels() override;
    //! \return Additional informations of channels to be stored.
    virtual XString getChannelInfoStrings() override;
    //! \return Trigger candidates
    virtual std::deque<XString> hardwareTriggerNames() override;
    //! Prepares instrumental setups for timing.
    virtual double setupTimeBase() override;
    //! Prepares instrumental setups for channels.
    virtual void setupChannels() override;
    //! Prepares instrumental setups for trigger.
    virtual void setupHardwareTrigger() override;
    //! Clears trigger settings.
    virtual void disableHardwareTriggers() override;
    //! \return # of samples per channel acquired from the arm.
    virtual uint64_t getTotalSampsAcquired() override;
    //! \return # of new samples per channel stored in the driver's ring buffer from the current read position.
    virtual uint32_t getNumSampsToBeRead() override;
    //! Sets the position for the next reading operated by a readAcqBuffer() function.
    //! \arg pos position from the hardware arm.
    //! \return true if the operation is sucessful
    virtual bool setReadPositionAbsolute(uint64_t pos) override;
    //! Sets the position for the next reading operated by a readAcqBuffer() function.
    virtual void setReadPositionFirstPoint() override;
    //! Copies data from driver's ring buffer from the current read position.
    //! The position for the next reading will be advanced by the return value.
    //! \arg buf to which 16bitxChannels stream is stored, packed by channels first.
    //! \return # of samples per channel read.
    virtual uint32_t readAcqBuffer(uint32_t size, tRawAI *buf) override;

    virtual bool isDRFCoherentSGSupported() const override {return true;}
private:
    TaskHandle m_task;

	static int32 onTaskDone_(TaskHandle task, int32 status, void*);
	void onTaskDone(TaskHandle task, int32 status);
};

#endif
