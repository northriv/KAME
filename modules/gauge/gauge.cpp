/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "gauge.h"
#include "interface.h"
#include "analyzer.h"


XGauge::XGauge(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas) {
}

void
XGauge::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    for(unsigned int ch = 0; ch < tr[ *this].m_pressures.size(); ch++) {
        tr[ *this].m_pressures[ch] = reader.pop<double>();
        m_entries[ch]->value(tr, tr[ *this].m_pressures[ch]);
    }
}
void
XGauge::visualize(const Snapshot &shot) {
}

void
XGauge::createChannels(Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
    const char **channel_names) {
  shared_ptr<XScalarEntryList> entries(meas->scalarEntries());

  for(int i = 0; channel_names[i]; i++) {
        shared_ptr<XScalarEntry> entry(create<XScalarEntry>(
            channel_names[i], false,
           dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4g"));
         m_entries.push_back(entry);
         entries->insert(tr_meas, entry);
    }
  iterate_commit([=](Transaction &tr){
    tr[ *this].m_pressures.resize(m_entries.size());
  });
}

void *
XGauge::execute(const atomic<bool> &terminated) {
    while( !terminated) {
        msecsleep(100);

        auto writer = std::make_shared<RawData>();
        // try/catch exception of communication errors
        try {
            unsigned int num = m_entries.size();
            for(unsigned int ch = 0; ch < num; ch++)
                writer->push((double)getPressure(ch));
        }
        catch (XKameError &e) {
            e.print(getLabel());
            continue;
        }

        finishWritingRaw(writer, XTime::now(), XTime::now());
    }
    return NULL;
}
