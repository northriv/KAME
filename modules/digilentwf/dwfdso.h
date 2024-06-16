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
#ifndef dwfdsoH
#define dwfdsoH

#include "dso.h"
//---------------------------------------------------------------------------

//! Digilent WaveForms
class XDigilentWFInterface : public XInterface {
public:
    XDigilentWFInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual bool isOpened() const override;

    int hdwf() const noexcept {return m_hdwf;}
    void throwWFError(XString &&msg, const char *file, int line);
protected:
    virtual void open() override;
    //! This can be called even if has already closed.
    virtual void close() override;
private:
    int m_hdwf;
};

template<class tDriver>
class XDigilentWFDriver : public tDriver {
public:
    XDigilentWFDriver(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
    const shared_ptr<XDigilentWFInterface> &interface() const {return m_interface;}
    int hdwf() const noexcept {return interface()->hdwf();}
    void throwWFError(XString &&msg, const char *file, int line) {
        return interface()->throwWFError(std::move(msg), file, line);
    }

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() {this->start();}
    //! Be called during stopping driver. Call interface()->stop() inside this routine.
    virtual void close() {interface()->stop();}

    void onOpen(const Snapshot &shot, XInterface *);
    void onClose(const Snapshot &shot, XInterface *);
    //! This should not cause an exception.
    virtual void closeInterface() override;
private:
    shared_ptr<Listener> m_lsnOnOpen, m_lsnOnClose;

    const shared_ptr<XDigilentWFInterface> m_interface;
};

//! Digilent WaveForms Analog input as DSO
class XDigilentWFDSO : public XDigilentWFDriver<XDSO> {
public:
    XDigilentWFDSO(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! Converts the raw to a display-able style.
    virtual void convertRaw(RawDataReader &reader, Transaction &tr) override;
protected:
    virtual void onTrace1Changed(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onTrace2Changed(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onTrace3Changed(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onTrace4Changed(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onSingleChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigPosChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVOffset1Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVOffset2Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVOffset3Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVOffset4Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) override;

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;
    //! Be called during stopping driver. Call interface()->stop() inside this routine.
    virtual void close() override;

    virtual double getTimeInterval() override;
    //! clear count or start sequence measurement
    virtual void startSequence() override;
    virtual int acqCount(bool *seq_busy) override;

    //! Loads waveforms and settings from the instrument.
    virtual void getWave(shared_ptr<RawData> &writer, std::deque<XString> &channels) override;
private:
    void disableChannels();
    void createChannels(const Snapshot &);
    void setupTiming(const Snapshot &);
    void setupTrigger(const Snapshot &);

    unique_ptr<XThread> m_threadReadAI;
    void *executeReadAI(const atomic<bool> &);
    void acquire(const atomic<bool> &);

    void clearAcquision();

    unsigned int m_preTriggerPos;

    struct DSORawRecord {
        unsigned int numCh() const {return channels.size();}
        unsigned int recordLength() const {return record.size() / numCh();}
        unsigned int accumCount;
        int acqCount;
        double interval;
        std::vector<int> channels;
        std::vector<double> record;
    };
    atomic_shared_ptr<DSORawRecord> m_accum;
    //! for moving av.
    std::deque<std::vector<double> > m_record_av;
};


template<class tDriver>
XDigilentWFDriver<tDriver>::XDigilentWFDriver(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    tDriver(name, runtime, tr_meas, meas),
    m_interface(XNode::create<XDigilentWFInterface>("Interface", false,
                                                 dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_interface);
    this->iterate_commit([=](Transaction &tr){
        m_lsnOnOpen = tr[ *interface()].onOpen().connectWeakly(
            this->shared_from_this(), &XDigilentWFDriver<tDriver>::onOpen);
        m_lsnOnClose = tr[ *interface()].onClose().connectWeakly(
            this->shared_from_this(), &XDigilentWFDriver<tDriver>::onClose);
    });
}
template<class tDriver>
void
XDigilentWFDriver<tDriver>::onOpen(const Snapshot &shot, XInterface *) {
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
XDigilentWFDriver<tDriver>::onClose(const Snapshot &shot, XInterface *) {
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
XDigilentWFDriver<tDriver>::closeInterface() {
    try {
        this->close();
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}
#endif
