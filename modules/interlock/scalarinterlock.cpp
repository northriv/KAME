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
#include "motor.h"
#include <QStatusBar>
#include <chrono>
#include <cmath>

REGISTER_TYPE(XDriverList, ScalarInterlock, "Scalar Interlock (stops motors)");

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
    for(unsigned int i = 0; i < NumMotors; ++i)
        m_motors.push_back(create<tMotor>(
            formatString("Motor%u", i + 1).c_str(), false, ref(tr_meas), meas->drivers()));
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
    QComboBox *cmb_motors[NumMotors] = {m_form->m_cmbMotor1, m_form->m_cmbMotor2,
        m_form->m_cmbMotor3, m_form->m_cmbMotor4};
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
    for(unsigned int i = 0; i < NumMotors; ++i)
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(m_motors[i], cmb_motors[i], ref(tr_meas)));

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
    constexpr int64_t RESTOP_NS = 300'000'000; //a motor still reporting moving is stopped again this often.
    constexpr int64_t HEARTBEAT_NS = 1'000'000'000; //State is recorded at least this often.

    bool was_armed = false, latched = false;
    unsigned int trips = Snapshot( *this)[ *this].tripCount();
    XString cause, shown;
    State last_state = State::Disarmed;
    int64_t last_record_ns = 0;
    std::vector<int64_t> last_stop_ns(NumMotors, 0);

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
                //Outside any transaction: the touch runs the motor driver's
                //listener here, and that talks to the hardware.
                for(unsigned int j = 0; j < NumMotors; ++j) {
                    shared_ptr<XMotorDriver> motor = shot[ *m_motors[j]];
                    if( !motor)
                        continue;
                    const bool moving = !(bool)Snapshot( *motor)[ *motor->ready()];
                    if(fresh_trip || (moving && (now - last_stop_ns[j] > RESTOP_NS))) {
                        trans( *motor->stopMotor()).touch();
                        last_stop_ns[j] = now;
                    }
                }
            }
            state = latched ? State::Tripped : State::Armed;
            //formatString() translates its format itself.
            if(latched) {
                const XString now_text = fault.empty() ?
                    XString(i18n("clear now, press Reset")) : XString(i18n("still faulting"));
                show(formatString("TRIPPED (#%u) %s -- %s", trips, cause.c_str(), now_text.c_str()));
            }
            else if(watched)
                show(formatString("Armed, watching %u", watched));
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
