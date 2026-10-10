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
#include "scalarinterlock.h"
#include "ui_scalarinterlockform.h"
#include <QStatusBar>
#include <chrono>
#include <cmath>

REGISTER_TYPE(XDriverList, ScalarInterlock, "Scalar Interlock");

int64_t
XInterlockCondition::steadyNS() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

XInterlockCondition::XInterlockCondition(const char *name, bool runtime, Transaction &tr_meas,
    const shared_ptr<XScalarEntryList> &entries) :
    XNode(name, runtime),
    m_entry(create<tEntry>("Entry", false, ref(tr_meas), entries)),
    m_mode(create<XComboNode>("Mode", false)),
    m_threshold(create<XDoubleNode>("Threshold", false)),
    m_entries(entries) {
    iterate_commit([=](Transaction &tr){
        tr[ *m_mode].add({"Off", "Trip if below", "Trip if above"});
        tr[ *m_mode] = (int)Mode::Off;
        //No flags: it only reconnects, and FLAG_AVOID_DUP is legal only with
        //FLAG_MAIN_THREAD_CALL (asserted in Listener's constructor).
        m_lsnOnSelection = tr[ *m_entry].onValueChanged().connectWeakly(
            shared_from_this(), &XInterlockCondition::onSelectionChanged);
        m_lsnOnSetting = tr[ *m_mode].onValueChanged().connectWeakly(
            shared_from_this(), &XInterlockCondition::onSettingChanged);
        tr[ *m_threshold].onValueChanged().connect(m_lsnOnSetting);
    });
}

void
XInterlockCondition::restartWatchdog(int64_t now_ns) {
    m_watchdogFromNS = now_ns;
}

void
XInterlockCondition::onSettingChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    m_modeCache = (int)shot[ *m_mode];
    m_thresholdCache = (double)shot[ *m_threshold];
    m_badStreak = 0; //counted against the old threshold.
}

void
XInterlockCondition::onSelectionChanged(const Snapshot &, XValueNodeBase *) {
    shared_ptr<XScalarEntry> entry = Snapshot( *this)[ *m_entry];
    m_lsnOnValue.reset();
    m_valueNode = entry ? entry->value().get() : nullptr;
    m_badStreak = 0;
    m_value = std::numeric_limits<double>::quiet_NaN();
    m_lastUpdateNS = 0;
    restartWatchdog(steadyNS());
    auto entries = m_entries.lock();
    if( !entry || !entries)
        return;
    //Through the entry list, as XValGraph does: the entry is linked under it
    //as well as under the driver or tool that owns it.
    entries->iterate_commit([=](Transaction &tr){
        if( !tr.isUpperOf( *entry))
            return; //not listed (any more): no listener, so the watchdog trips.
        m_lsnOnValue = tr[ *entry->value()].onValueChanged().connectWeakly(
            shared_from_this(), &XInterlockCondition::onValueChanged);
    });
}

void
XInterlockCondition::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    //Runs on the thread that committed the value -- a camera's acquisition
    //thread for a math tool -- so it only records; the interlock thread acts.
    if(node != m_valueNode.load())
        return; //a listener outliving a reselection.
    const double v = shot[ *static_cast<XDoubleNode *>(node)];
    const int mode = m_modeCache;
    const double th = m_thresholdCache;
    const bool bad = std::isnan(v) ||
        ((mode == (int)Mode::TripBelow) && (v < th)) ||
        ((mode == (int)Mode::TripAbove) && (v > th));
    if(bad)
        ++m_badStreak;
    else
        m_badStreak = 0;
    m_value = v;
    m_lastUpdateNS = steadyNS();
}

XString
XInterlockCondition::fault(int64_t now_ns, double timeout, unsigned int consecutive) const {
    const int mode = m_modeCache;
    if(mode == (int)Mode::Off)
        return {};
    if( !m_valueNode.load())
        return i18n("no entry chosen");
    const int64_t last = m_lastUpdateNS;
    if(now_ns - std::max(last, m_watchdogFromNS.load()) > (int64_t)(timeout * 1e9))
        return last ? i18n("no update") : i18n("no data");
    if(m_badStreak >= std::max(1u, consecutive)) {
        const double v = m_value;
        if(std::isnan(v))
            return i18n("value is NaN");
        return formatString("%.4g %s %.4g", v,
            (mode == (int)Mode::TripBelow) ? "<" : ">", (double)m_thresholdCache);
    }
    return {};
}

struct XInterlockAction::Operation {
    XString label; //!< shown, and saved in a .kam -- so never translated.
    //! Empty: \a node is the driver's own child.  Otherwise it is looked up
    //! in every child whose name starts with this, e.g. each "Laser<n>"
    //! channel of a laser controller -- whose "Tec<n>" channels have an
    //! "Enabled" too, which must be left alone.
    XString channelPrefix;
    XString node;
    enum class Kind {Touch, SetFalse, SetTrue} kind;
    bool reassertWhileNotReady; //!< Touch again while the driver's "Ready" is false.
    bool inEveryChild = false; //!< \a node in each of the driver's children, e.g. each graph's Dump.
};

//! Only operations whose result is the safe state -- and Dump, which
//! records what the driver showed at the trip (each graph with a FileName
//! writes; images go to numbered files).  Left out on purpose: a magnet
//! supply (its own SafeCond entries ramp it down slowly; cutting it is the
//! danger), a turbo pump (stopping it vents the vacuum), heater ranges.
const std::vector<XInterlockAction::Operation> &
XInterlockAction::operations() {
    static const std::vector<Operation> ops = [] {
        std::vector<Operation> v = {
            {"Stop motor", "", "StopMotor", Operation::Kind::Touch, true},
            {"Laser off", "Laser", "Enabled", Operation::Kind::SetFalse, false},
            {"RF off", "", "RFON", Operation::Kind::SetFalse, false},
            {"Output off", "", "Output", Operation::Kind::SetFalse, false},
            {"Close valve", "", "CloseValve", Operation::Kind::Touch, false},
            {"Dump", "", "Dump", Operation::Kind::Touch, false, true},
        };
        for(unsigned int ch = 1; ch <= 8; ++ch) {
            const XString chname = "Channel" + std::to_string(ch);
            v.push_back({chname + " off", "", chname, Operation::Kind::SetFalse, false});
            v.push_back({chname + " on", "", chname, Operation::Kind::SetTrue, false});
        }
        return v;
    }();
    return ops;
}

std::vector<shared_ptr<XNode>>
XInterlockAction::targets(const shared_ptr<XDriver> &drv, const Operation &op) {
    std::vector<shared_ptr<XNode>> nodes;
    if( !drv)
        return nodes;
    auto add = [&](const shared_ptr<XNode> &parent) {
        auto node = parent->getChild(op.node);
        if(op.kind == Operation::Kind::Touch)
            node = dynamic_pointer_cast<XTouchableNode>(node);
        else
            node = dynamic_pointer_cast<XBoolNode>(node);
        if(node)
            nodes.push_back(node);
    };
    if( !op.inEveryChild && op.channelPrefix.empty()) {
        add(drv);
        return nodes;
    }
    Snapshot shot( *drv);
    if(shot.size())
        for(auto &&child: *shot.list())
            if(op.inEveryChild ||
                (child->getName().compare(0, op.channelPrefix.size(), op.channelPrefix) == 0))
                add(child);
    return nodes;
}

XInterlockAction::XInterlockAction(const char *name, bool runtime, Transaction &tr_meas,
    const shared_ptr<XDriverList> &drivers) :
    XNode(name, runtime),
    m_driver(create<tDriver>("Driver", false, ref(tr_meas), drivers)),
    m_operation(create<XComboNode>("Operation", false)) {
    iterate_commit([=](Transaction &tr){
        m_lsnOnDriver = tr[ *m_driver].onValueChanged().connectWeakly(
            shared_from_this(), &XInterlockAction::onDriverChanged);
    });
}

void
XInterlockAction::onDriverChanged(const Snapshot &, XValueNodeBase *) {
    shared_ptr<XDriver> drv = Snapshot( *this)[ *m_driver];
    std::vector<XString> labels;
    for(auto &op: operations())
        if( !targets(drv, op).empty())
            labels.push_back(op.label);
    //clear() keeps the chosen label, and add() restores it when offered:
    //a .kam may set Operation before its driver exists.
    iterate_commit([=](Transaction &tr){
        tr[ *m_operation].clear();
        for(auto &label: labels)
            tr[ *m_operation].add(label);
    });
}

const XInterlockAction::Operation *
XInterlockAction::resolve(std::vector<shared_ptr<XNode>> &nodes, shared_ptr<XDriver> &drv) const {
    Snapshot shot( *this);
    drv = shot[ *m_driver];
    const XString label = shot[ *m_operation].to_str();
    for(auto &op: operations()) {
        if(op.label != label)
            continue;
        nodes = targets(drv, op);
        return nodes.empty() ? nullptr : &op;
    }
    return nullptr;
}

bool
XInterlockAction::isUsed() const {
    return !Snapshot( *this)[ *m_operation].to_str().empty();
}

XString
XInterlockAction::problem() const {
    const XString label = Snapshot( *this)[ *m_operation].to_str();
    if(label.empty())
        return {};
    std::vector<shared_ptr<XNode>> nodes;
    shared_ptr<XDriver> drv;
    if(resolve(nodes, drv))
        return {};
    if( !drv)
        return label + ": " + XString(i18n("no driver chosen"));
    return label + ": " + XString(i18n("not available on")) + " " + drv->getLabel();
}

void
XInterlockAction::perform(bool fresh, int64_t now_ns) {
    constexpr int64_t REASSERT_NS = 300'000'000;
    if( !fresh && (now_ns - m_lastNS < REASSERT_NS))
        return;
    std::vector<shared_ptr<XNode>> nodes;
    shared_ptr<XDriver> drv;
    const Operation *op = resolve(nodes, drv);
    if( !op)
        return;
    for(auto &node: nodes) {
        switch(op->kind) {
        case Operation::Kind::Touch: {
            bool again = fresh;
            if( !fresh && op->reassertWhileNotReady) {
                auto ready = dynamic_pointer_cast<XBoolNode>(drv->getChild("Ready"));
                again = ready && !(bool)Snapshot( *ready)[ *ready];
            }
            if(again) {
                trans( *static_pointer_cast<XTouchableNode>(node)).touch();
                m_lastNS = now_ns;
            }
            break;
        }
        case Operation::Kind::SetFalse:
        case Operation::Kind::SetTrue: {
            auto b = static_pointer_cast<XBoolNode>(node);
            const bool want = (op->kind == Operation::Kind::SetTrue);
            if(fresh || ((bool)Snapshot( *b)[ *b] != want)) {
                trans( *b) = want;
                m_lastNS = now_ns;
            }
            break;
        }
        }
    }
}

XScalarInterlock::XScalarInterlock(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_tripped(create<XBoolNode>("Tripped", true)),
    m_reset(create<XTouchableNode>("Reset", true)),
    m_status(create<XStringNode>("Status", true)),
    m_consecutive(create<XUIntNode>("Consecutive", false)),
    m_watchdogTimeout(create<XDoubleNode>("WatchdogTimeout", false)),
    m_entryState(create<XScalarEntry>("State", false,
        dynamic_pointer_cast<XDriver>(shared_from_this()), "%.0f")),
    m_form(new FrmScalarInterlock) {

    for(unsigned int i = 0; i < NumConditions; ++i)
        m_conditions.push_back(create<XInterlockCondition>(
            formatString("Condition%u", i + 1).c_str(), false, ref(tr_meas), meas->scalarEntries()));
    for(unsigned int i = 0; i < NumActions; ++i)
        m_actions.push_back(create<XInterlockAction>(
            formatString("Action%u", i + 1).c_str(), false, ref(tr_meas), meas->drivers()));
    m_armed = create<XBoolNode>("Armed", false);

    meas->scalarEntries()->insert(tr_meas, m_entryState);

    m_form->statusBar()->hide();
    m_form->setWindowTitle(i18n("Scalar Interlock - ") + getLabel());

    QComboBox *cmb_entries[NumConditions] = {m_form->m_cmbEntry1, m_form->m_cmbEntry2,
        m_form->m_cmbEntry3, m_form->m_cmbEntry4};
    QComboBox *cmb_modes[NumConditions] = {m_form->m_cmbMode1, m_form->m_cmbMode2,
        m_form->m_cmbMode3, m_form->m_cmbMode4};
    QLineEdit *ed_thresholds[NumConditions] = {m_form->m_edThreshold1, m_form->m_edThreshold2,
        m_form->m_edThreshold3, m_form->m_edThreshold4};
    QComboBox *cmb_drivers[NumActions] = {m_form->m_cmbActionDriver1, m_form->m_cmbActionDriver2,
        m_form->m_cmbActionDriver3, m_form->m_cmbActionDriver4,
        m_form->m_cmbActionDriver5, m_form->m_cmbActionDriver6};
    QComboBox *cmb_operations[NumActions] = {m_form->m_cmbActionOp1, m_form->m_cmbActionOp2,
        m_form->m_cmbActionOp3, m_form->m_cmbActionOp4,
        m_form->m_cmbActionOp5, m_form->m_cmbActionOp6};
    m_conUIs = {
        xqcon_create<XQToggleButtonConnector>(m_armed, m_form->m_ckbArmed),
        xqcon_create<XQLedConnector>(m_tripped, m_form->m_ledTripped),
        xqcon_create<XQButtonConnector>(m_reset, m_form->m_btnReset),
        xqcon_create<XQLabelConnector>(m_status, m_form->m_lblStatus),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_consecutive, m_form->m_spbConsecutive),
        xqcon_create<XQLineEditConnector>(m_watchdogTimeout, m_form->m_edWatchdog),
    };
    for(unsigned int i = 0; i < NumConditions; ++i) {
        auto &c = m_conditions[i];
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(c->entry(), cmb_entries[i], ref(tr_meas)));
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(c->mode(), cmb_modes[i], Snapshot( *c->mode())));
        m_conUIs.push_back(xqcon_create<XQLineEditConnector>(c->threshold(), ed_thresholds[i]));
    }
    for(unsigned int i = 0; i < NumActions; ++i) {
        auto &a = m_actions[i];
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(a->driver(), cmb_drivers[i], ref(tr_meas)));
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(a->operation(), cmb_operations[i], Snapshot( *a->operation())));
    }

    iterate_commit([=](Transaction &tr){
        tr[ *m_consecutive] = 3; //~0.1 s at 30 fps: one bad frame is not a trip.
        tr[ *m_watchdogTimeout] = 0.5;
        tr[ *m_status] = i18n("Disarmed");
        tr[ *m_tripped].setUIEnabled(false); //an indicator, not a control.
        m_lsnOnArmed = tr[ *m_armed].onValueChanged().connectWeakly(
            shared_from_this(), &XScalarInterlock::onArmedChanged);
        m_lsnOnReset = tr[ *m_reset].onTouch().connectWeakly(
            shared_from_this(), &XScalarInterlock::onResetTouched);
    });
}

void
XScalarInterlock::onArmedChanged(const Snapshot &shot, XValueNodeBase *) {
    if(shot[ *m_armed] && !m_running.exchange(true))
        start();
}
void
XScalarInterlock::onResetTouched(const Snapshot &, XTouchableNode *) {
    m_resetRequested = true; //the thread decides, against the conditions as they stand.
}

void
XScalarInterlock::writeRecord(State state, unsigned int trip_count) {
    auto writer = std::make_shared<RawData>();
    writer->push((uint32_t)state);
    writer->push((uint32_t)trip_count);
    const XTime now = XTime::now();
    finishWritingRaw(writer, now, now);
}

void
XScalarInterlock::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    const uint32_t state = reader.pop<uint32_t>();
    const uint32_t trips = reader.pop<uint32_t>();
    if(state > (uint32_t)State::Tripped)
        throw XRecordError(i18n("Unknown interlock state."), __FILE__, __LINE__);
    tr[ *this].m_state = (State)state;
    tr[ *this].m_tripCount = trips;
    m_entryState->value(tr, state);
}

void *
XScalarInterlock::execute(const atomic<bool> &terminated) {
    constexpr int64_t TICK_MS = 50;
    constexpr int64_t HEARTBEAT_NS = 1'000'000'000; //State is recorded at least this often.

    bool was_armed = false, latched = false;
    unsigned int trips = Snapshot( *this)[ *this].tripCount();
    XString cause, shown;
    State last_state = State::Disarmed;
    int64_t last_record_ns = 0;

    auto show = [&](const XString &status) {
        if(status == shown)
            return;
        iterate_commit([&](Transaction &tr){
            tr[ *m_status] = status;
            tr[ *m_tripped] = latched;
        });
        shown = status;
    };

    while( !terminated) {
        const int64_t now = XInterlockCondition::steadyNS();
        Snapshot shot( *this);
        State state;
        if( !shot[ *m_armed]) {
            was_armed = false;
            latched = false;
            state = State::Disarmed;
            show(i18n("Disarmed"));
        }
        else {
            if( !was_armed) {
                //Each source gets one timeout to speak, counted from now.
                for(auto &c: m_conditions)
                    c->restartWatchdog(now);
                m_resetRequested = false;
                was_armed = true;
            }
            const double timeout = std::max(0.05, (double)shot[ *m_watchdogTimeout]);
            const unsigned int consecutive = shot[ *m_consecutive];
            XString fault;
            unsigned int watched = 0;
            for(unsigned int i = 0; i < NumConditions; ++i) {
                auto &c = m_conditions[i];
                if( !c->isEnabled())
                    continue;
                ++watched;
                XString f = c->fault(now, timeout, consecutive);
                if( !f.empty() && fault.empty())
                    fault = formatString("Condition%u: ", i + 1) + f;
            }
            unsigned int acting = 0;
            for(unsigned int i = 0; i < NumActions; ++i) {
                auto &a = m_actions[i];
                if( !a->isUsed())
                    continue;
                ++acting;
                XString p = a->problem();
                if( !p.empty() && fault.empty())
                    fault = formatString("Action%u: ", i + 1) + p;
            }
            bool fresh_trip = false;
            if( !latched && !fault.empty()) {
                latched = true;
                fresh_trip = true;
                ++trips;
                cause = fault;
            }
            if(m_resetRequested.exchange(false) && latched && fault.empty())
                latched = false; //refused, silently, while a fault persists: the status says so.
            if(latched) {
                //Outside any transaction: each action runs the target driver's
                //listener here, and that talks to the hardware.
                for(auto &a: m_actions)
                    if(a->isUsed())
                        a->perform(fresh_trip, now);
            }
            state = latched ? State::Tripped : State::Armed;
            //formatString() translates its format itself.
            if(latched) {
                const XString now_text = fault.empty() ?
                    XString(i18n("clear now, press Reset")) : XString(i18n("still faulting"));
                show(formatString("TRIPPED (#%u) %s -- %s", trips, cause.c_str(), now_text.c_str()));
            }
            else if(watched)
                show(formatString("Armed: %u condition(s), %u action(s)", watched, acting));
            else
                show(i18n("Armed, but no condition is on"));
        }
        if((state != last_state) || (now - last_record_ns > HEARTBEAT_NS)) {
            writeRecord(state, trips);
            last_state = state;
            last_record_ns = now;
        }
        msecsleep(TICK_MS);
    }
    //Stopped from outside -- Measurement > Stop, or the driver being released:
    //untick Armed rather than leave it ticked with nothing watching.  Ticking
    //it again starts a new thread.
    latched = false;
    iterate_commit([&](Transaction &tr){
        tr[ *m_armed] = false;
        tr[ *m_tripped] = false;
        tr[ *m_status] = i18n("Disarmed (measurement stopped)");
    });
    writeRecord(State::Disarmed, trips);
    m_running = false;
    return nullptr;
}
