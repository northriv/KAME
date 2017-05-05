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

constexpr double SMPL_PER_SEC = 5000000.0; //5MSmps/s

REGISTER_TYPE(XDriverList, ThamwayPROT3DSO, "Thamway PROT3 digital streaming DSO");

#define NUM_MAX_CH 2

XThamwayPROT3DSO::XThamwayPROT3DSO(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XRealTimeAcqDSO<XCharDeviceDriver<XDSO, XThamwayFX3USBInterface>>(name, runtime, tr_meas, meas) {

    iterate_commit([=](Transaction &tr){
        for(auto &&x: {vFullScale1(), vFullScale2(), vFullScale3(), vFullScale4()}) {
            x->disable();
        }
    });
    vOffset1()->disable();
    vOffset2()->disable();
    vOffset3()->disable();
    vOffset4()->disable();
}

void
XThamwayPROT3DSO::startAcquision() {
    XScopedLock<XMutex> lock(m_acqMutex);
    m_totalSmps = 0;
    m_currRdChunk = m_wrChunkBegin;
    m_currRdPos = 0;
    for(auto &&x : m_chunks) {
        x.data.clear();
        x.posAbs = 0;
    }
}
void
XThamwayPROT3DSO::commitAcquision() {
}
void
XThamwayPROT3DSO::stopAcquision() {
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
}

void
XThamwayPROT3DSO::setupHardwareTrigger() {
}

void
XThamwayPROT3DSO::disableHardwareTriggers() {
}

uint64_t
XThamwayPROT3DSO::getTotalSampsAcquired() {
    XScopedLock<XMutex> lock(m_acqMutex);
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
    uint32_t samps_read = 0;
    while(size) {
        auto &chunk = m_chunks[m_currRdChunk];
        if(m_currRdChunk == m_wrChunkBegin) {
            break;
        }
        uint32_t len = std::min((uint32_t)chunk.data.size() - m_currRdPos, size);
        std::copy(chunk.data.begin() + m_currRdPos, chunk.data.begin() + m_currRdPos + len, buf);
        samps_read += len;
        buf += len;
        size -= len;
        m_currRdChunk++; if(m_currRdChunk == m_chunks.size()) m_currRdChunk = 0;
        m_currRdPos = 0;
    }
    return samps_read;
}


void
XThamwayPROT3DSO::open() throw (XKameError &) {
    XRealTimeAcqDSO<XCharDeviceDriver<XDSO, XThamwayFX3USBInterface>>::open();
    //allocates buffers.
    m_chunks.resize(NumChunks);
    m_totalSmps = 0;
    m_wrChunkEnd = 0;
    m_wrChunkBegin = 0;
    m_currRdChunk = 0;
    m_currRdPos = 0;

    m_acqThreads.resize(NumThreads, {shared_from_this(), &XThamwayPROT3DSO::execute});
    for(auto &&x: m_acqThreads) {
        x.resume();
    }

}
void
XThamwayPROT3DSO::close() throw (XKameError &) {
    XScopedLock<XInterface> lock( *interface());
    for(auto &&x: m_acqThreads) {
        x.terminate();
        x.waitFor();
    }
    m_acqThreads.clear();
    m_chunks.clear();

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
        ssize_t wridx;
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
            return interface()->asyncReceive( (char*)&chunk.data[0], chunk.data.size());
        };
        try {
            auto async = fn(); //issues async. IO sequentially.
            auto count = async.waitFor();
            auto &chunk = m_chunks[wridx];
            {
                XScopedLock<XMutex> lock(m_acqMutex);
                chunk.ioInProgress = false;
                if(count != chunk.data.size()) {
                //Pulse generation has stopped.
                    fprintf(stderr, "Pulse generation has stopped.\n");
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

