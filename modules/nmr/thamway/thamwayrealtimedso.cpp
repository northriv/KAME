/***************************************************************************
        Copyright (C) 2002-2017 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/

#include "thamwayrealtimedso.h"
#include "dsorealtimeacq_impl.h"

constexpr double SMPL_PER_SEC = 5e6; //5MSmps/s

REGISTER_TYPE(XDriverList, ThamwayPROT3DSO, "Thamway PROT3 digital streaming DSO");

#define NUM_MAX_CH 2

XThamwayPROT3DSO::XThamwayPROT3DSO(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XRealTimeAcqDSO<XCharDeviceDriver<XDSO, XThamwayFX3USBInterface>>(name, runtime, tr_meas, meas) {

    std::vector<shared_ptr<XNode>> unnecessary_ui{
        trace3(), trace4(),
        vFullScale1(), vFullScale2(), vFullScale3(), vFullScale4(),
        vOffset1(), vOffset2(), vOffset3(), vOffset4(),
        trigLevel(),
        timeWidth()
    };
    iterate_commit([=](Transaction &tr){
        tr[ *recordLength()] = lrint(SMPL_PER_SEC * 0.01);
        tr[ *fetchMode()] = "Averaging";
        for(auto &&x: {trace1(), trace2()}) {
            tr[ *x].add({"CH1", "CH2"});
        }
        tr[ *trace1()] = "CH2";
        tr[ *trace2()] = "CH1";
        tr[ *average()] = 1;
        for(auto &&x: unnecessary_ui)
            tr[ *x].disable();
    });
}

void
XThamwayPROT3DSO::startAcquision() {
    if(m_acqThreads.empty())
        commitAcquision();
}
void
XThamwayPROT3DSO::commitAcquision() {
    {
        XScopedLock<XMutex> lock(m_acqMutex);
        //allocates buffers.
        m_chunks.resize(NumChunks);
        m_totalSmps = 0;
        m_wrChunkEnd = 0;
        m_wrChunkBegin = 0;
        m_currRdChunk = 0;
        m_currRdPos = 0;
        for(auto &&x: m_chunks) {
            x.data.resize(ChunkSize);
            x.data.shrink_to_fit();
            x.data.clear();
        }
        if(isMemLockAvailable()) {
            mlock(this, sizeof(XThamwayPROT3DSO));
            for(auto &&x: m_chunks) {
                mlock(&x.data[0], x.data.capacity() * sizeof(tRawAI));
            }
        }
    }

    m_acqThreads.resize(NumThreads);
    for(auto &&x: m_acqThreads) {
        x.reset(new  XThread<XThamwayPROT3DSO>(shared_from_this(), &XThamwayPROT3DSO::execute));
        x->resume();
    }

    //waits until async IOs have been submitted.
    for(;;) {
        msecsleep(10);
        {
            XScopedLock<XMutex> lock(m_acqMutex);
            if(m_wrChunkEnd >= NumThreads)
                break;
        }
    }
}
void
XThamwayPROT3DSO::stopAcquision() {
    XScopedLock<XMutex> lock(m_acqMutex);
    for(auto &&x: m_acqThreads) {
        x->terminate();
    }
    for(auto &&x: m_acqThreads) {
        x->waitFor();
    }
    m_acqThreads.clear();
    m_chunks.clear();
}

void
XThamwayPROT3DSO::clearAcquision() {
}

unsigned int
XThamwayPROT3DSO::getNumOfChannels() {
    return NUM_MAX_CH;
}

XString
XThamwayPROT3DSO::getChannelInfoStrings() {
    return {};
}

std::deque<XString>
XThamwayPROT3DSO::hardwareTriggerNames() {
    return {};
}

double
XThamwayPROT3DSO::setupTimeBase() {
    return 1.0 / SMPL_PER_SEC;
}

void
XThamwayPROT3DSO::setupChannels() {
    int ch_num = 0;
    for(auto &&trace: {trace1(), trace2(), trace3(), trace4()}) {
        for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
            m_coeffAI[ch_num][i] = 0.0;
        m_coeffAI[ch_num][1] = 1.0 / 32768.0; //+-1V F.S.
        ch_num++;
    }
}

void
XThamwayPROT3DSO::setupHardwareTrigger() {
}

void
XThamwayPROT3DSO::disableHardwareTriggers() {
}

uint64_t
XThamwayPROT3DSO::getTotalSampsAcquired() {
    return m_totalSmps;
}

uint32_t
XThamwayPROT3DSO::getNumSampsToBeRead() {
    XScopedLock<XMutex> lock(m_acqMutex);
    uint64_t rdpos_abs = m_chunks[m_currRdChunk].posAbs + m_currRdPos;
    return m_totalSmps - rdpos_abs;
}

bool
XThamwayPROT3DSO::setReadPositionAbsolute(uint64_t pos) {
    XScopedLock<XMutex> lock(m_acqMutex);
    //searching for corresponding chunk.
    for(m_currRdChunk = m_wrChunkEnd; m_currRdChunk != m_wrChunkBegin;) {
        uint64_t pos_abs = m_chunks[m_currRdChunk].posAbs;
        uint64_t pos_abs_end = pos_abs + m_chunks[m_currRdChunk].data.size();
        if((pos >= pos_abs) && (pos < pos_abs_end)) {
            m_currRdPos = pos - pos_abs;
            return true;
        }
        m_currRdChunk++; if(m_currRdChunk == m_chunks.size()) m_currRdChunk = 0;
    }
    return false;
}
void
XThamwayPROT3DSO::setReadPositionFirstPoint() {
    XScopedLock<XMutex> lock(m_acqMutex);
    m_currRdChunk = 0;
    m_currRdPos = 0;
}

uint32_t
XThamwayPROT3DSO::readAcqBuffer(uint32_t size, tRawAI *buf) {
    Snapshot shot( *this);
    bool swap_traces = (shot[ *trace1()].to_str() == "CH2");
    uint32_t samps_read = 0;
    while(size) {
        auto &chunk = m_chunks[m_currRdChunk];
        if(m_currRdChunk == m_wrChunkBegin) {
            break;
        }
        uint32_t len = std::min((uint32_t)chunk.data.size() - m_currRdPos, size);
        if(swap_traces) {
            tRawAI *rdpos = &chunk.data[0] + m_currRdPos;
            tRawAI *rdpos_end = rdpos + len;
            for(;rdpos < rdpos_end;) {
                tRawAI ch2 = *rdpos++;
                tRawAI ch1 = *rdpos++;
                *buf++ = ch1;
                *buf++ = ch2;
            }
        }
        else {
            std::copy(chunk.data.begin() + m_currRdPos, chunk.data.begin() + m_currRdPos + len, buf);
            buf += len;
        }
        samps_read += len;
        size -= len;
        m_currRdChunk++; if(m_currRdChunk == m_chunks.size()) m_currRdChunk = 0;
        m_currRdPos = 0;
    }
    return samps_read;
}


void
XThamwayPROT3DSO::open() throw (XKameError &) {
    XRealTimeAcqDSO<XCharDeviceDriver<XDSO, XThamwayFX3USBInterface>>::open();
}
void
XThamwayPROT3DSO::close() throw (XKameError &) {
    XScopedLock<XInterface> lock( *interface());
    stopAcquision();

    iterate_commit([=](Transaction &tr){
        for(auto &&x: {trace1(), trace2(), trace3(), trace4()})
            tr[ *x].clear();
    });
    XRealTimeAcqDSO<XCharDeviceDriver<XDSO, XThamwayFX3USBInterface>>::close();
}

void*
XThamwayPROT3DSO::execute(const atomic<bool> &terminated) {
    Transactional::setCurrentPriorityMode(Priority::HIGHEST);

    enum class Collision {BufferUnderflow, IOStall};
    while( !terminated) {
        ssize_t wridx; //index of a chunk for async. IO.
        auto fn = [&]() -> CyFXUSBDevice::AsyncIO {
            XScopedLock<XMutex> lock(m_acqMutex);
            ssize_t next_idx = m_wrChunkEnd;
            wridx = next_idx;
            auto &chunk = m_chunks[next_idx++];
            if(chunk.ioInProgress) {
                throw Collision::IOStall;
            }
            if(next_idx == m_chunks.size()) {
                    next_idx = 0;
            }
            if(next_idx == m_currRdChunk) {
                throw Collision::BufferUnderflow;
            }
            m_wrChunkEnd = next_idx;
            chunk.data.resize(ChunkSize);
            chunk.ioInProgress = true;
            return interface()->asyncReceive( (char*)&chunk.data[0], chunk.data.size() * sizeof(tRawAI));
        };
        try {
            auto async = fn(); //issues async. IO sequentially.
            while( !async.hasFinished() && !terminated)
                msecsleep(20);
            if(terminated) break;
            auto count = async.waitFor() / sizeof(tRawAI);
            auto &chunk = m_chunks[wridx];
            {
                XScopedLock<XMutex> lock(m_acqMutex);
                chunk.ioInProgress = false;
                if(count != chunk.data.size()) {
                //Pulse generation has stopped.
                    fprintf(stderr, "Pulse generation has stopped.\n");
                    throw Collision::IOStall;
                }
                chunk.data.resize(count);
                if(wridx == m_wrChunkBegin) {
                    while( !m_chunks[wridx].ioInProgress && (wridx != m_wrChunkEnd)) {
                        m_chunks[wridx].posAbs = m_totalSmps;
                        m_totalSmps += m_chunks[wridx].data.size();
                        wridx++; if(wridx == m_chunks.size()) wridx = 0;
                        m_wrChunkBegin = wridx;
                    }
                }
            }
        }
        catch (XInterface::XInterfaceError &e) {
            e.print();
            m_chunks[wridx].data.clear();
            m_chunks[wridx].ioInProgress = false;
            msecsleep(100);
            continue;
        }
        catch (Collision &c) {
            switch (c) {
            case Collision::IOStall:
                msecsleep(20);
                continue;
            case Collision::BufferUnderflow:
                gErrPrint(i18n("Buffer Underflow."));
                msecsleep(20);
                continue;
            }
        }
    }
    return nullptr;
}

