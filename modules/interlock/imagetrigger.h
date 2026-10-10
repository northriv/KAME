/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef imagetriggerH
#define imagetriggerH
//---------------------------------------------------------------------------
#include "scalarinterlock.h"
#include <deque>

class XDigitalCamera;
class X2DImage;
class QImage;
class Ui_FrmImageTrigger;
typedef QForm<QMainWindow, Ui_FrmImageTrigger> FrmImageTrigger;

//! Saves a camera's frames around events, unattended: when a watched entry
//! crosses its threshold (a person detector's score, a motion or Correlation
//! tool...), the frames just before and after go to numbered PNG files, and
//! after a hold-off it watches again by itself.  The recorder counterpart of
//! XScalarInterlock, whose conditions it shares -- but where the interlock
//! latches and treats a silent source as a fault, this one re-arms on its
//! own and never fires on silence or NaN, which only show in the status.
class XImageTrigger : public XPrimaryDriverWithThread {
public:
    XImageTrigger(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XImageTrigger() = default;

    static constexpr unsigned int NumConditions = 2;
    using tCamera = XItemNode<XDriverList, XDigitalCamera>;
    enum class State {Disarmed = 0, Watching = 1, Capturing = 2, HoldOff = 3};

    const shared_ptr<tCamera> &camera() const {return m_camera;}
    const shared_ptr<XInterlockCondition> &condition(unsigned int i) const {return m_conditions.at(i);}
    const shared_ptr<XUIntNode> &consecutive() const {return m_consecutive;}
    const shared_ptr<XDoubleNode> &holdOff() const {return m_holdOff;}  //!< [s]
    const shared_ptr<XUIntNode> &framesBefore() const {return m_framesBefore;}
    const shared_ptr<XUIntNode> &framesAfter() const {return m_framesAfter;}
    const shared_ptr<XDoubleNode> &interval() const {return m_interval;} //!< [s] between saved frames
    //! Template of the series, e.g. ".../cam.png" -> cam_0001_<time>.png ...; empty saves nothing.
    const shared_ptr<XStringNode> &fileName() const {return m_fileName;}
    //! 0 keeps every file; otherwise the oldest of the series are deleted beyond it.
    const shared_ptr<XUIntNode> &maxFiles() const {return m_maxFiles;}
    const shared_ptr<XStringNode> &status() const {return m_status;}
    //! Watching while on; saved, and created last, as in XScalarInterlock.
    const shared_ptr<XBoolNode> &armed() const {return m_armed;}

    struct Payload : public XPrimaryDriver::Payload {
        State state() const {return m_state;}
        unsigned int events() const {return m_events;}
        State m_state = State::Disarmed;
        unsigned int m_events = 0;
    };
protected:
    //! No interface: the first arming starts the thread (see XScalarInterlock).
    virtual void closeInterface() override {}
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    virtual void visualize(const Snapshot &) override {}
private:
    //! A frame kept for saving: the camera's counts are shared, not copied.
    struct Frame {
        local_shared_ptr<const std::vector<uint32_t>> counts;
        unsigned int width = 0, height = 0, stride = 0, firstPixel = 0;
        XTime time;
        uint64_t seq = 0;
    };

    virtual void *execute(const atomic<bool> &terminated) override;
    void onArmedChanged(const Snapshot &shot, XValueNodeBase *);
    void onCameraChanged(const Snapshot &, XValueNodeBase *);
    void onSettingChanged(const Snapshot &, XValueNodeBase *);
    //! On the camera's committing thread: keeps one frame per Interval, and only that.
    void onCameraRecord(const Snapshot &shot, XDriver *driver);
    //! The frame as an image -- counts above 8 bits kept only when
    //! \a lossless (PNG), scaled down otherwise; null on a size mismatch.
    static QImage frameImage(const Frame &frame, bool lossless);
    //! \return an error message, empty on success.
    XString save(const QImage &img, const XTime &time, const XString &templ);
    //! Shows \a img, the frame last saved (or that would have been), with its event and time.
    void showLastShot(const QImage &img, const XTime &time, unsigned int event);
    void enforceMaxFiles(const XString &templ, unsigned int max_files);
    void writeRecord(State state, unsigned int events);

    const shared_ptr<tCamera> m_camera;
    std::vector<shared_ptr<XInterlockCondition>> m_conditions;
    const shared_ptr<XUIntNode> m_consecutive;
    const shared_ptr<XDoubleNode> m_holdOff;
    const shared_ptr<XUIntNode> m_framesBefore, m_framesAfter;
    const shared_ptr<XDoubleNode> m_interval;
    const shared_ptr<XStringNode> m_fileName;
    const shared_ptr<XUIntNode> m_maxFiles;
    const shared_ptr<XStringNode> m_status;
    shared_ptr<XBoolNode> m_armed;
    const shared_ptr<XScalarEntry> m_entryEvents;

    shared_ptr<Listener> m_lsnOnArmed, m_lsnOnCamera, m_lsnOnSetting, m_lsnOnCameraRecord;
    std::atomic<bool> m_running{false};

    //! Guards the ring and the two below; held briefly, never across a transaction.
    XMutex m_ringMutex;
    std::deque<Frame> m_ring;
    uint64_t m_lastSeq = 0;
    XTime m_lastKept;
    std::atomic<double> m_intervalCache{1.0};
    std::atomic<unsigned int> m_ringCapacity{10};

    unsigned int m_fileSeq = 0; //!< thread only; 0: look the series up again.
    XString m_fileTemplate;     //!< thread only; the template m_fileSeq belongs to.

    const qshared_ptr<FrmImageTrigger> m_form;
    std::deque<xqcon_ptr> m_conUIs;
    shared_ptr<X2DImage> m_lastShot; //!< needs m_form, hence made in the constructor's body.
};

#endif
