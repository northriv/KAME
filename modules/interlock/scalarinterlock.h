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

#ifndef scalarinterlockH
#define scalarinterlockH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"
#include "xitemnode.h"
#include "analyzer.h"
#include <atomic>
#include <limits>

class QMainWindow;
class Ui_FrmScalarInterlock;
typedef QForm<QMainWindow, Ui_FrmScalarInterlock> FrmScalarInterlock;

//! One watched scalar entry.  It faults when the value crosses Threshold
//! (for Consecutive samples in a row), is NaN, or stops arriving -- the last
//! two so that a dead source holds the interlock tripped instead of letting
//! it pass.  Samples are taken from the entry's own Value node, which talks
//! on every assignment (the same value included), whichever driver or math
//! tool writes it.
class XInterlockCondition : public XNode {
public:
    XInterlockCondition(const char *name, bool runtime, Transaction &tr_meas,
        const shared_ptr<XScalarEntryList> &entries);
    virtual ~XInterlockCondition() = default;

    using tEntry = XItemNode<XScalarEntryList, XScalarEntry>;
    enum class Mode {Off = 0, TripBelow = 1, TripAbove = 2};

    const shared_ptr<tEntry> &entry() const {return m_entry;}
    const shared_ptr<XComboNode> &mode() const {return m_mode;}
    const shared_ptr<XDoubleNode> &threshold() const {return m_threshold;}

    bool isEnabled() const {return m_modeCache.load() != (int)Mode::Off;}
    //! Starts the "no update" clock afresh: a source gets one timeout to speak
    //! after the interlock is armed or the entry is chosen.
    void restartWatchdog(int64_t now_ns);
    //! Why this condition holds the interlock tripped, or empty when it does not.
    //! \a now_ns from steadyNS(); \a timeout [s].
    XString fault(int64_t now_ns, double timeout, unsigned int consecutive) const;
    //! The last value seen, for the status line.
    double lastValue() const {return m_value.load();}

    static int64_t steadyNS();
private:
    void onSelectionChanged(const Snapshot &, XValueNodeBase *);
    void onSettingChanged(const Snapshot &, XValueNodeBase *);
    void onValueChanged(const Snapshot &shot, XValueNodeBase *node);

    const shared_ptr<tEntry> m_entry;
    const shared_ptr<XComboNode> m_mode;
    const shared_ptr<XDoubleNode> m_threshold;
    const weak_ptr<XScalarEntryList> m_entries;

    shared_ptr<Listener> m_lsnOnSelection, m_lsnOnSetting, m_lsnOnValue;

    //! Written by the listeners (on whatever thread commits), read by the
    //! interlock's thread.  Plain atomics, not the STM: the reader polls.
    std::atomic<int> m_modeCache{(int)Mode::Off};
    std::atomic<double> m_thresholdCache{0.0};
    std::atomic<const XValueNodeBase *> m_valueNode{nullptr}; //!< the selected entry's, to ignore a stale listener.
    std::atomic<double> m_value{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<unsigned int> m_badStreak{0};
    std::atomic<int64_t> m_lastUpdateNS{0}, m_watchdogFromNS{0};
};

//! One thing done on a trip: an operation on a chosen driver -- stop a motor,
//! switch off a laser, RF or an output, close a valve, set a relay channel.
//! Operations are found by the driver's node names, not its C++ type, so a
//! driver from a module with no core library (funcsynth, arbfunc) works too;
//! the Operation combo offers what the chosen driver actually has.
class XInterlockAction : public XNode {
public:
    XInterlockAction(const char *name, bool runtime, Transaction &tr_meas,
        const shared_ptr<XDriverList> &drivers);
    virtual ~XInterlockAction() = default;

    using tDriver = XItemNode<XDriverList, XDriver>;
    const shared_ptr<tDriver> &driver() const {return m_driver;}
    const shared_ptr<XComboNode> &operation() const {return m_operation;}

    //! Why this row cannot act, or empty when it can or is unused (no
    //! Operation).  A row that cannot act holds the interlock tripped: a
    //! broken action must show on arming, not when it is needed.
    XString problem() const;
    bool isUsed() const;
    //! On the trip ( fresh) does the operation; afterwards only re-asserts
    //! it -- a motor still moving, an output switched back on -- at most every
    //! 0.3 s.  Interlock thread only; outside any transaction, since the
    //! driver's listener talks to its hardware from here.
    void perform(bool fresh, int64_t now_ns);
private:
    struct Operation;
    static const std::vector<Operation> &operations();
    //! The nodes of \a drv that \a op acts on; empty when it has none.
    static std::vector<shared_ptr<XNode>> targets(const shared_ptr<XDriver> &drv, const Operation &op);
    //! The selected operation and the nodes it acts on, or nullptr.
    const Operation *resolve(std::vector<shared_ptr<XNode>> &nodes, shared_ptr<XDriver> &drv) const;
    void onDriverChanged(const Snapshot &, XValueNodeBase *);

    const shared_ptr<tDriver> m_driver;
    const shared_ptr<XComboNode> m_operation;
    shared_ptr<Listener> m_lsnOnDriver;
    int64_t m_lastNS = 0; //!< interlock thread only.
};

//! Acts when watched scalar entries leave their range -- for instance a
//! Correlation math tool aimed at a pattern a stage hides when it goes too
//! far: stops motors, switches off lasers, RF and outputs, closes valves,
//! sets relays.  Trips latch: the actions stay in force, re-asserted while
//! something undoes them, until Reset is pressed with every condition
//! healthy.  A source that goes silent or turns NaN trips it too.  Not a last
//! line of defence: KAME itself can fail, so hardware limits stay in place.
class XScalarInterlock : public XPrimaryDriverWithThread {
public:
    XScalarInterlock(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XScalarInterlock() = default;

    static constexpr unsigned int NumConditions = 4;
    static constexpr unsigned int NumActions = 6;
    enum class State {Disarmed = 0, Armed = 1, Tripped = 2};

    //! Monitoring runs while this is on; saved, so a loaded setup is armed
    //! again (and trips until its sources speak).  Created last, so a .kam
    //! restores it after the conditions it arms.
    const shared_ptr<XBoolNode> &armed() const {return m_armed;}
    const shared_ptr<XBoolNode> &tripped() const {return m_tripped;}
    //! Clears a trip, refused while any condition still faults.
    const shared_ptr<XTouchableNode> &reset() const {return m_reset;}
    const shared_ptr<XStringNode> &status() const {return m_status;}
    const shared_ptr<XUIntNode> &consecutive() const {return m_consecutive;}
    const shared_ptr<XDoubleNode> &watchdogTimeout() const {return m_watchdogTimeout;} //!< [s]
    const shared_ptr<XInterlockCondition> &condition(unsigned int i) const {return m_conditions.at(i);}
    const shared_ptr<XInterlockAction> &action(unsigned int i) const {return m_actions.at(i);}

    struct Payload : public XPrimaryDriver::Payload {
        State state() const {return m_state;}
        unsigned int tripCount() const {return m_tripCount;}
        State m_state = State::Disarmed;
        unsigned int m_tripCount = 0;
    };
protected:
    //! No interface: the first arming starts the thread, which then runs until
    //! stop() (driver released, KAME quitting) and idles while disarmed --
    //! starting and stopping it with Armed would let a quick off/on leave the
    //! box ticked with nothing watching.
    virtual void closeInterface() override {}
    virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) override;
    virtual void visualize(const Snapshot &) override {}
private:
    virtual void *execute(const atomic<bool> &terminated) override;
    void onArmedChanged(const Snapshot &shot, XValueNodeBase *);
    void onResetTouched(const Snapshot &, XTouchableNode *);
    void writeRecord(State state, unsigned int trip_count);

    const shared_ptr<XBoolNode> m_tripped;
    const shared_ptr<XTouchableNode> m_reset;
    const shared_ptr<XStringNode> m_status;
    const shared_ptr<XUIntNode> m_consecutive;
    const shared_ptr<XDoubleNode> m_watchdogTimeout;
    std::vector<shared_ptr<XInterlockCondition>> m_conditions;
    std::vector<shared_ptr<XInterlockAction>> m_actions;
    shared_ptr<XBoolNode> m_armed;
    const shared_ptr<XScalarEntry> m_entryState;

    shared_ptr<Listener> m_lsnOnArmed, m_lsnOnReset;
    std::atomic<bool> m_resetRequested{false};
    std::atomic<bool> m_running{false}; //!< thread started and not yet returned.

    const qshared_ptr<FrmScalarInterlock> m_form;
    std::deque<xqcon_ptr> m_conUIs;
};

#endif
