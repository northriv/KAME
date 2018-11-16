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
#include <cstring>

constexpr double SMPL_PER_SEC = 5e6; //5MSmps

REGISTER_TYPE(XDriverList, ThamwayPROT3DSO, "Thamway PROT3 digital streaming DSO");

constexpr unsigned int NUM_MAX_CH = 2;

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
    if(m_acqThreads.size())
        return;

    fprintf(stderr, "start acq.\n");
    if(m_swapTraces)
        fprintf(stderr, " with traces swapped.\n");

    commitAcquision();
    for(int i = 0; i < NumThreads; ++i)
        m_acqThreads.emplace_back(new XThread(shared_from_this(), &XThamwayPROT3DSO::executeAsyncRead));

    //reads out remaining data
    char buf[ChunkSize];
    int64_t cnt_remain = 0;
    for(int rdsize = ChunkSize / 16;; rdsize *= 2) {
        if(rdsize > sizeof(buf))
            break;
//            throw XInterface::XInterfaceError(i18n("Starting acquision has failed during buffer cleanup."), __FILE__, __LINE__);
        auto async = interface()->asyncReceive(buf, rdsize);
        if( !async->hasFinished()) {
            msecsleep(rdsize / NUM_MAX_CH / sizeof(tRawAI) * 1000 / SMPL_PER_SEC + 1);
//            if( !async->hasFinished())
//                break; //not remaining?
        }
        int64_t retsize = async->waitFor();
        cnt_remain += retsize;
//        if(retsize < rdsize)
//            break; //end of data.
    }
    fprintf(stderr, "Remaining data was %lld bytes\n", (long long)cnt_remain);

    //waits until async IOs have been submitted.
    for(unsigned int retry = 0;; ++retry) {
        msecsleep(10);
        for(auto &&x: m_acqThreads) {
            if((retry > 20) || x->isTerminated()) {
                fprintf(stderr, "Starting failed.\n");
                stopAcquision();
                throw XInterface::XInterfaceError(i18n("Starting acquision has failed."), __FILE__, __LINE__);
            }
        }
        {
            XScopedLock<XMutex> lock(m_acqMutex);
            if((m_wrChunkEnd - m_wrChunkBegin + m_chunks.size()) % m_chunks.size() >= NumThreads)
                break;
        }
    }
}
void
XThamwayPROT3DSO::commitAcquision() {
    fprintf(stderr, "commit acq.: ");
    stopAcquision();
}
void
XThamwayPROT3DSO::stopAcquision() {
    clearAcquision();
    {
        XScopedLock<XMutex> lock(m_acqMutex);
        fprintf(stderr, "stop acq., total=%llu\n", (long long unsigned int)getTotalSampsAcquired());
        //allocates buffers.
        m_chunks.resize(NumChunks);
        m_totalSmpsPerCh = 0;
        m_wrChunkEnd = 0;
        m_wrChunkBegin = 0;
        m_currRdChunk = 0;
        m_currRdPos = 0;
        for(auto &&x: m_chunks) {
            x.data.reserve(ChunkSize);
            x.data.clear();
            x.ioInProgress = false;
            x.posAbsPerCh = 0;
        }
        if(isMemLockAvailable()) {
            mlock(this, sizeof(XThamwayPROT3DSO));
            for(auto &&x: m_chunks) {
                mlock(&x.data[0], x.data.capacity() * sizeof(tRawAI));
            }
        }
    }
}

void
XThamwayPROT3DSO::clearAcquision() {
    fprintf(stderr, "clear ac");
    for(auto &&x: m_acqThreads) {
        x->terminate();
    }
    for(auto &&x: m_acqThreads) {
        x->join();
    }
    m_acqThreads.clear();
    fprintf(stderr, "q.\n");
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
    for(int ch_num: {0,1,2,3}) {
        for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
            m_coeffAI[ch_num][i] = 0.0;
        m_coeffAI[ch_num][1] = 2.5 / 32768.0; //+-5V F.S.
    }
    Snapshot shot( *this);
    m_swapTraces = (shot[ *trace1()].to_str() == "CH2");
}

void
XThamwayPROT3DSO::setupHardwareTrigger() {
}

void
XThamwayPROT3DSO::disableHardwareTriggers() {
}

uint64_t
XThamwayPROT3DSO::getTotalSampsAcquired() {
    return m_totalSmpsPerCh;
}

uint32_t
XThamwayPROT3DSO::getNumSampsToBeRead() {
    uint64_t rdpos_abs_per_ch = m_chunks[m_currRdChunk].posAbsPerCh + m_currRdPos / getNumOfChannels();
    assert(m_totalSmpsPerCh >= rdpos_abs_per_ch);
    auto x = m_totalSmpsPerCh - rdpos_abs_per_ch;
    if(x > 0) x--; //-1, not to proceed m_currRdChunk.
    return x;
}

bool
XThamwayPROT3DSO::setReadPositionAbsolute(uint64_t pos) {
    XScopedLock<XMutex> lock(m_acqMutex);
    //searching for corresponding chunk.
    for(m_currRdChunk = m_wrChunkEnd; m_currRdChunk != m_wrChunkBegin;) {
        uint64_t pos_abs_per_ch = m_chunks[m_currRdChunk].posAbsPerCh;
        uint64_t pos_abs_per_ch_end = pos_abs_per_ch +
            m_chunks[m_currRdChunk].data.size() / getNumOfChannels();
        if((pos >= pos_abs_per_ch) && (pos < pos_abs_per_ch_end)) {
            m_currRdPos = (pos - pos_abs_per_ch) * getNumOfChannels();
//            fprintf(stderr, "Set readpos at %u, chunk %u, rpos %u.\n", (unsigned int)pos, (unsigned int)m_currRdChunk, (unsigned int)m_currRdPos);
            return true;
        }
        m_currRdChunk++;
        if(m_currRdChunk == m_chunks.size()) m_currRdChunk = 0;
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
    size *= getNumOfChannels();

    auto memcpy_wordswap = [](tRawAI *dst, const tRawAI *src, size_t byte_size) {
        size_t len = byte_size / sizeof(tRawAI);
        const tRawAI *src_end = src + len;
        auto *src_end_pre = (const tRawAI*)(((uintptr_t)src + 15) / 16 * 16);
        while(src < src_end_pre) {
            tRawAI ch1, ch2;
            ch2 = *src++; ch1 = *src++; *dst++ = ch1; *dst++ = ch2;
        }
        if(((uintptr_t)dst % 8 == 0) && (sizeof(tRawAI) == 2)) {
            //unrolls loop.
            auto src_end64 = reinterpret_cast<uint64_t*>((uintptr_t)(src + len)/ 16 * 16);
            auto *src64 = reinterpret_cast<const uint64_t*>(src);
            auto *dst64 = reinterpret_cast<uint64_t*>(dst);
            while(src64 < src_end64) {
//                    ch2 = *rdpos++; ch1 = *rdpos++; *buf++ = ch1; *buf++ = ch2;
//                    ch2 = *rdpos++; ch1 = *rdpos++; *buf++ = ch1; *buf++ = ch2;
//                    ch2 = *rdpos++; ch1 = *rdpos++; *buf++ = ch1; *buf++ = ch2;
//                    ch2 = *rdpos++; ch1 = *rdpos++; *buf++ = ch1; *buf++ = ch2;
                //equiv to above.
                uint64_t f = 0xffff0000ffffuLL;
                auto llw_swapped =
                    ((*src64 & f) * 0x10000u) + ((*src64 / 0x10000u) & f);
                *dst64++ = llw_swapped;
                src64++;
                llw_swapped =
                    ((*src64 & f) * 0x10000u) + ((*src64 / 0x10000u) & f);
                *dst64++ = llw_swapped;
                src64++;
            }
            src = (const tRawAI*)src64;
            dst = (tRawAI*)dst64;
        }
        while(src < src_end) {
            tRawAI ch1, ch2;
            ch2 = *src++; ch1 = *src++; *dst++ = ch1; *dst++ = ch2;
        }
    };

    for(auto &chunk = m_chunks[m_currRdChunk]; size;) {
        readBarrier();
        if(chunk.ioInProgress) {
            fprintf(stderr, "Unexpected collision\n");
            break; //collision.
        }
        if(m_currRdChunk == m_wrChunkBegin) {
            fprintf(stderr, "Unexpected collision\n");
            break; //nothing to read.
        }
        if(chunk.data.size() < m_currRdPos) {
            fprintf(stderr, "??? %d < %d\n", (int)chunk.data.size(), m_currRdPos);
            break;
        }

        ssize_t len = std::min((uint32_t)chunk.data.size() - m_currRdPos, size);
        if(m_swapTraces) {
            //copies data with word swapping.
            //test results i7 2.5GHz, OSX10.12, 3.7GB/s
            memcpy_wordswap(buf, &chunk.data[m_currRdPos], len * sizeof(tRawAI));
        }
        else {
            //Simple copy without swap.
            //test results i7 2.5GHz, OSX10.12, 4.5GB/s
            std::memcpy(buf, &chunk.data[m_currRdPos], len * sizeof(tRawAI));
        }
        buf += len;
        samps_read += len;
        size -= len;
        m_currRdPos += len;
        readBarrier();
        if(chunk.ioInProgress) {
            fprintf(stderr, "Ring buffer has been overwritten\n");
            break; //collision.
        }
        if(m_currRdPos == chunk.data.size()) {
            XScopedLock<XMutex> lock(m_acqMutex);
            m_currRdChunk++;
            if(m_currRdChunk == m_chunks.size()) m_currRdChunk = 0;
            chunk = m_chunks[m_currRdChunk];
            m_currRdPos = 0;
            if(m_currRdChunk == m_wrChunkBegin) {
                fprintf(stderr, "Unexpected collision\n");
                break; //nothing to read.
            }
        }
    }
    return samps_read / getNumOfChannels();
}


void*
XThamwayPROT3DSO::executeAsyncRead(const atomic<bool> &terminated) {
    Transactional::setCurrentPriorityMode(Priority::HIGHEST);

    enum class Collision {IOStall, Stopped};
    while( !terminated) {
        ssize_t wridx; //index of a chunk for async. IO.
        auto issue_async_read = [&]() {
            //Lambda fn to issue async IO and reserves a chunk atomically.
            XScopedLock<XMutex> lock(m_acqMutex);
            ssize_t next_idx = m_wrChunkEnd;
            wridx = next_idx;
            auto &chunk = m_chunks[next_idx++];
            if(chunk.ioInProgress)
                throw Collision::IOStall;
            if(next_idx == m_chunks.size())
                next_idx = 0;
            m_wrChunkEnd = next_idx;
            chunk.data.resize(ChunkSize, 0x4f4f);
            chunk.ioInProgress = true;
            writeBarrier();
            return interface()->asyncReceive( (char*)&chunk.data[0],
                    chunk.data.size() * sizeof(tRawAI));
        };
        try {
            auto async = issue_async_read();
//            dbgPrint(formatString("asyncRead for %u initiated", (unsigned int)wridx));
//            fprintf(stderr, "asyncRead for %u initiated\n", (unsigned int)wridx);
            while( !async->hasFinished() && !terminated)
                msecsleep(20);
            if(terminated) {
                break;
            }
            auto count = async->waitFor() / sizeof(tRawAI);
//          dbgPrint(formatString("read for %u count=%u", (unsigned int)wridx, (unsigned int)count));
//          fprintf(stderr, "read for %u count=%u", (unsigned int)wridx, (unsigned int)count);
            auto &chunk = m_chunks[wridx];
            {
                XScopedLock<XMutex> lock(m_acqMutex);
                chunk.ioInProgress = false;
                chunk.data.resize(count);
                if(wridx == m_wrChunkBegin) {
                    //rearranges indices to indicate ready for read.
                    while( !m_chunks[wridx].ioInProgress && (wridx != m_wrChunkEnd)) {
                        m_chunks[wridx].posAbsPerCh = m_totalSmpsPerCh;
                        writeBarrier();
                        m_totalSmpsPerCh += m_chunks[wridx].data.size() / getNumOfChannels();
                        wridx++;
                        if(wridx == m_chunks.size()) wridx = 0;
                        m_wrChunkBegin = wridx;
                    }
//                  dbgPrint(formatString("wrBegin=%u, total=%f sec", (unsigned int)wridx, (double)m_totalSmpsPerCh / 5e6));
//                  fprintf(stderr, "wrBegin=%u, total=%f sec\n", (unsigned int)wridx, (double)m_totalSmpsPerCh / 5e6);
                }
                if(count == 0) {
                //Pulse generation has stopped.
                    throw Collision::Stopped;
                }
            }
        }
        catch (XInterface::XInterfaceError &e) {
            e.print();
            XScopedLock<XMutex> lock(m_acqMutex);
            for(auto &&x: m_acqThreads) {
                x->terminate();
            }
            break;
        }
        catch (Collision &c) {
            switch (c) {
            case Collision::IOStall:
                fprintf(stderr, "IO stall.\n");
                dbgPrint("IO stall.");
                msecsleep(20);
                continue;
            case Collision::Stopped:
                dbgPrint("Pulse generation has stopped.");
                fprintf(stderr, "Pulse generation has stopped.\n");
                continue;
            }
        }
    }
    fprintf(stderr, "Thread fin.\n");
    return nullptr;
}

