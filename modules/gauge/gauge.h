/***************************************************************************
        Copyright (C) 2002-2020 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef GAUGE_H
#define GAUGE_H

#include "primarydriverwiththread.h"
#include "xnodeconnector.h"
#include "pfeifferprotocol.h"

class XScalarEntry;

class XGauge : public XPrimaryDriverWithThread {
public:
    XGauge(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    //! usually nothing to do
    virtual ~XGauge() = default;
    //! show all forms belonging to driver
    virtual void showForms() override {}

    struct Payload : public XPrimaryDriver::Payload {
        unsigned int channelNum() const {return m_pressures.size();}
        double pressure(unsigned int ch) const {return m_pressures[ch];}
    private:
        friend class XGauge;
        std::vector<double> m_pressures;
    };
protected:
    //! This function will be called when raw data are written.
    //! Implement this function to convert the raw data to the record (Payload).
    //! \sa analyze()
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    //! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
    //! This might be called even if the record is invalid (time() == false).
    virtual void visualize(const Snapshot &shot) override;

    //! register channel names in your constructor
    //! \param channel_names array of pointers to channel name. ends with null pointer.
    void createChannels(Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
                        const char **channel_names);

    //! reads pressure sensor value from the instrument
    virtual double getPressure(unsigned int idx) = 0; //[Pa]
private:
    std::deque<shared_ptr<XScalarEntry> > m_entries;

    virtual void *execute(const atomic<bool> &) override;
};
#endif
