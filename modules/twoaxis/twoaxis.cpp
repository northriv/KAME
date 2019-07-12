#include "twoaxis.h"
#include "ui_twoaxisform.h"
#include "interface.h"
#include "analyzer.h"

REGISTER_TYPE(XDriverList, TwoAxis, "Controller of Two-Axis Rotator");

XTwoAxis::XTwoAxis(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XSecondaryDriver(name, runtime, ref(tr_meas), meas),
      m_rot1(create<XItemNode < XDriverList, XMotorDriver> >(
          "ROT1", false, ref(tr_meas), meas->drivers(), false)),
      m_rot2(create<XItemNode < XDriverList, XMotorDriver> >(
          "ROT2", false, ref(tr_meas), meas->drivers(), false)),
      m_theta(create<XScalarEntry>("Theta", false,
                                   dynamic_pointer_cast<XDriver>(shared_from_this()))),
      m_phi(create<XScalarEntry>("Phi", false,
                                   dynamic_pointer_cast<XDriver>(shared_from_this()))),
      m_record_step(create<XScalarEntry>("RecordStep", false,
                                   dynamic_pointer_cast<XDriver>(shared_from_this()))),
      m_target_theta(create<XDoubleNode>("TargetTheta", true)),
      m_target_phi(create<XDoubleNode>("TargetPhi", true)),
      m_offset_theta(create<XDoubleNode>("OffsetTheta", true)),
      m_offset_phi(create<XDoubleNode>("OffsetPhi", true)),
      m_max_theta(create<XDoubleNode>("MaxTheta", true)),
      m_min_theta(create<XDoubleNode>("MinTheta", true)),
      m_max_phi(create<XDoubleNode>("MaxPhi", true)),
      m_min_phi(create<XDoubleNode>("MinPhi", true)),
      m_step(create<XIntNode>("Step", true)),
      m_wait(create<XDoubleNode>("Wait", true)),
      m_rot1_per_theta(create<XDoubleNode>("ROT1PerTheta", true)),
      m_rot1_per_phi(create<XDoubleNode>("ROT1PerPhi", true)),
      m_rot2_per_theta(create<XDoubleNode>("ROT2PerTheta", true)),
      m_rot2_per_phi(create<XDoubleNode>("ROT2PerPhi", true)),
      m_abort(create<XTouchableNode>("Abort", true)),
      m_ready(create<XBoolNode>("Ready", true)),
      m_running(create<XBoolNode>("Running", true)),
      m_form(new FrmTwoaxis),
      m_statusPrinter(XStatusPrinter::create(m_form.get())){

    m_form->setWindowTitle(i18n("Controller of Two-Axis Rotator") + getLabel() );

    meas->scalarEntries()->insert(tr_meas, theta());
    meas->scalarEntries()->insert(tr_meas, phi());
    meas->scalarEntries()->insert(tr_meas, record_step());

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(rot1(), m_form->m_cmbROT1, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(rot2(), m_form->m_cmbROT2, ref(tr_meas)),
        xqcon_create<XQLCDNumberConnector>(m_theta->value(), m_form->m_lcdTheta),
        xqcon_create<XQLCDNumberConnector>(m_phi->value(), m_form->m_lcdPhi),
        xqcon_create<XQLineEditConnector>(target_theta(), m_form->m_edTargetTheta),
        xqcon_create<XQLineEditConnector>(target_phi(), m_form->m_edTargetPhi),
        xqcon_create<XQLineEditConnector>(offset_theta(), m_form->m_edOffsetTheta),
        xqcon_create<XQLineEditConnector>(offset_phi(), m_form->m_edOffsetPhi),
        xqcon_create<XQLineEditConnector>(max_theta(), m_form->m_edMaxTheta),
        xqcon_create<XQLineEditConnector>(min_theta(), m_form->m_edMinTheta),
        xqcon_create<XQLineEditConnector>(max_phi(), m_form->m_edMaxPhi),
        xqcon_create<XQLineEditConnector>(min_phi(), m_form->m_edMinPhi),
        xqcon_create<XQLineEditConnector>(step(), m_form->m_edStep),
        xqcon_create<XQLineEditConnector>(wait(), m_form->m_edWait),
        xqcon_create<XQLineEditConnector>(rot1_per_theta(), m_form->m_edROT1PerTheta),
        xqcon_create<XQLineEditConnector>(rot1_per_phi(), m_form->m_edROT1PerPhi),
        xqcon_create<XQLineEditConnector>(rot2_per_theta(), m_form->m_edROT2PerTheta),
        xqcon_create<XQLineEditConnector>(rot2_per_phi(), m_form->m_edROT2PerPhi),
        xqcon_create<XQButtonConnector>(m_abort, m_form->m_btnAbort),
        xqcon_create<XQLedConnector>(m_ready, m_form->m_ledReady),
        xqcon_create<XQLedConnector>(m_running, m_form->m_ledRunning)
    };

    connect(rot1());
    connect(rot2());

    iterate_commit([=](Transaction &tr){
        tr[ *m_ready] = false;
        tr[ *m_running] = false;
        tr[ *m_offset_theta] =0.0;
        tr[ *m_offset_phi] = 0.0;
        tr[ *m_max_theta] = 180.0;
        tr[ *m_min_theta] = -180.0;
        tr[ *m_max_phi] = 180.0;
        tr[ *m_min_phi] = -180.0;
        tr[ *m_wait] = 1.0;
        tr[ *m_step] = 100;
        tr[ *m_rot1_per_theta] = 80.0;
        tr[ *m_rot2_per_theta] = -80.0;
        tr[ *m_rot1_per_phi] = 40.0;
        tr[ *m_rot2_per_phi] = 40.0;
        tr[ *m_step] = 100;
        tr[ *abort()].setUIEnabled(false);
        m_lsnOnTargetThetaChanged = tr[ *m_target_theta].onValueChanged().connectWeakly(
            shared_from_this(), &XTwoAxis::onTargetThetaChanged);
        m_lsnOnTargetPhiChanged = tr[ *m_target_phi].onValueChanged().connectWeakly(
            shared_from_this(), &XTwoAxis::onTargetPhiChanged);
        m_lsnOnAbortTouched = tr[ *m_abort].onTouch().connectWeakly(
            shared_from_this(), &XTwoAxis::onAbortTouched);
    });
}

XTwoAxis::~XTwoAxis() {
}

void
XTwoAxis::showForms() {
    m_form->showNormal();
    m_form->raise();
}

bool
XTwoAxis::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) const {
    const shared_ptr<XMotorDriver> rot1__ = shot_this[ *rot1()];
    const shared_ptr<XMotorDriver> rot2__ = shot_this[ *rot2()];
    if(!rot1__ || !rot2__ || rot1__ == rot2__)
        return false;
    if(emitter != rot1__.get() && emitter != rot2__.get())
        return false;
    return true;
}

void XTwoAxis::onTargetThetaChanged(const Snapshot &shot, XValueNodeBase *node) {
    Snapshot shot_this( *this);
    if (shot_this[ *target_theta()] > shot_this[ *max_theta()]
            || shot_this[ *target_theta()] < shot_this[ *min_theta()]) {
        m_statusPrinter->printWarning(i18n("Invalid target theta."));

    } else if(abs(shot[ *rot1_per_theta()] * shot[ *rot2_per_phi()] - shot[*rot2_per_theta()] * shot[ *rot1_per_phi()]) < 1){
       m_statusPrinter->printWarning(i18n("Maybe invalid setting for rot per theta/phi."));

    } else {
        initSweep(shot_this);
        iterate_commit([=](Transaction &tr){
            tr[ *this].isTheta = true;
            tr[ *this].startAngle = shot_this[ *m_theta->value()];
            double delta_theta = (shot_this[*m_theta->value()] - shot_this[ *target_theta()]) / shot_this[ *step()];
            double det = shot[ *rot1_per_theta()] * shot[ *rot2_per_phi()] - shot[*rot2_per_theta()] * shot[ *rot1_per_phi()];
            tr[ *this].deltaROT[0] = shot_this[ *rot2_per_phi()] * delta_theta / det;
            tr[ *this].deltaROT[1] = - shot_this[ *rot1_per_phi()] * delta_theta / det;
        });
    }
}

void XTwoAxis::onTargetPhiChanged(const Snapshot &shot, XValueNodeBase *node) {
    Snapshot shot_this( *this);
    if (shot_this[ *target_phi()] > shot_this[ *max_phi()]
            || shot_this[ *target_phi()] < shot_this[ *min_phi()]) {
        m_statusPrinter->printWarning(i18n("Invalid target phi."));

    } else if(abs(shot[ *rot1_per_theta()] * shot[ *rot2_per_phi()] - shot[*rot2_per_theta()] * shot[ *rot1_per_phi()]) < 1){
        m_statusPrinter->printWarning(i18n("Maybe invalid setting for rot per theta/phi."));

    } else {
        initSweep(shot_this);
        iterate_commit([=](Transaction &tr){
            tr[ *this].isTheta = false;
            tr[ *this].startAngle = shot_this[ *m_phi->value()];
            double delta_phi = (shot_this[*m_phi->value()] - shot_this[ *target_phi()]) / shot_this[ *step()];
            double det = shot[ *rot1_per_theta()] * shot[ *rot2_per_phi()] - shot[*rot2_per_theta()] * shot[ *rot1_per_phi()];
            tr[ *this].deltaROT[0] = - shot_this[ *rot2_per_theta()] * delta_phi / det;
            tr[ *this].deltaROT[1] = shot_this[ *rot1_per_theta()] * delta_phi / det;
        });
    }
}

void XTwoAxis::initSweep(const Snapshot &shot) {
    shared_ptr<XMotorDriver> rot1__ = shot[ *rot1()];
    shared_ptr<XMotorDriver> rot2__ = shot[ *rot2()];
    const shared_ptr<XMotorDriver> rots[] = {rot1__, rot2__};

    for(auto &&rot: rots) {
        if(rot) {
            rot->iterate_commit([=](Transaction &tr){
                tr[ *rot->active()] = true; // Activate motor.
            });
            rot->iterate_commit([=](Transaction &tr){
                tr[ *rot->stopMotor()].touch();
            });
        }
    }
    iterate_commit([=](Transaction &tr){

        tr[ *m_running] = true;
        record_step()->value(tr, 0);

        tr[ *target_theta()].setUIEnabled(false);
        tr[ *target_phi()].setUIEnabled(false);
        tr[ *offset_theta()].setUIEnabled(false);
        tr[ *offset_phi()].setUIEnabled(false);
        tr[ *rot1_per_theta()].setUIEnabled(false);
        tr[ *rot2_per_theta()].setUIEnabled(false);
        tr[ *rot1_per_phi()].setUIEnabled(false);
        tr[ *rot2_per_phi()].setUIEnabled(false);
        tr[ *step()].setUIEnabled(false);

        tr[ *rot1()].setUIEnabled(false);
        tr[ *rot2()].setUIEnabled(false);
        tr[ *abort()].setUIEnabled(true);

        tr[ *this].currentStep = 0;
        tr[ *this].isWaitStable = false;
        tr[ *this].isMoving = false;
        tr[ *this].startROT[0] = shot[ *rot1__->position()->value()];
        tr[ *this].startROT[1] = shot[ *rot2__->position()->value()];
        tr[ *this].timeROTChanged = XTime::now();
        tr[ *this].timeRecordChanged = XTime::now();
    });

}
void
XTwoAxis::endSweep(Transaction &tr){
    tr[ *m_running] = false;
    record_step()->value(tr, -1);

    tr[ *target_theta()].setUIEnabled(true);
    tr[ *target_phi()].setUIEnabled(true);
    tr[ *offset_theta()].setUIEnabled(true);
    tr[ *offset_phi()].setUIEnabled(true);
    tr[ *rot1_per_theta()].setUIEnabled(true);
    tr[ *rot2_per_theta()].setUIEnabled(true);
    tr[ *rot1_per_phi()].setUIEnabled(true);
    tr[ *rot2_per_phi()].setUIEnabled(true);
    tr[ *step()].setUIEnabled(true);

    tr[ *rot1()].setUIEnabled(true);
    tr[ *rot2()].setUIEnabled(true);

    tr[ *this].currentStep = 0;
    tr[ *this].isWaitStable = false;
    tr[ *this].isMoving = false;
    tr[ *this].timeROTChanged = {};
    tr[ *this].timeRecordChanged = {};
}

void
XTwoAxis::onAbortTouched(const Snapshot &shot, XTouchableNode *) {
    iterate_commit_while([=](Transaction &tr)->bool{
        if( !tr[ *m_running])
            return false;
        endSweep(tr);
        return true;
    });

}

void
XTwoAxis::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) throw (XRecordError&) {
    Snapshot &shot_this(tr);
    shared_ptr<XMotorDriver> rot1__ = shot_this[ *rot1()];
    shared_ptr<XMotorDriver> rot2__ = shot_this[ *rot2()];
    double pos1, pos2;
    bool ready1, ready2;

    if(!rot1__ || !rot2__) {
        throw XSkippedRecordError(__FILE__, __LINE__);
    }

    if(emitter == rot1__.get()){
        pos1 = shot_emitter[ *rot1__->position()->value()];
        ready1 = shot_emitter[ *rot1__->ready()];
        pos2 = shot_others[ *rot2__->position()->value()];
        ready2 = shot_others[ *rot2__->ready()];
    } else {
        pos1 = shot_others[ *rot1__->position()->value()];
        ready1 = shot_others[ *rot1__->ready()];
        pos2 = shot_emitter[ *rot2__->position()->value()];
        ready2 = shot_emitter[ *rot2__->ready()];
    }
    double var = shot_this[ *offset_theta()] + pos1 / shot_this[ *rot1_per_theta()] +  pos2 / shot_this[ *rot2_per_theta()];
    theta()->value(tr, var);
    var = shot_this[ *offset_phi()] + pos1 / shot_this[ *rot1_per_phi()] + pos2 / shot_this[ *rot2_per_phi()];
    phi()->value(tr, var);
    tr[ *m_ready] = ready1 && ready2;


    if(shot_this[ *running()]) {
        if(shot_this[ *ready()] && !tr[ *this].isWaitMove){
            if(!tr[ *this].isWaitStable){
                tr[ *this].timeROTChanged = XTime::now();
            } else if(XTime::now() - tr[ *this].timeROTChanged > shot_this[ *wait()]){
                record_step()->value(tr, tr[ *this].currentStep);
                tr[ *this].timeRecordChanged = XTime::now();
            } else if(XTime::now() - tr[ *this].timeRecordChanged > shot_this[ *wait()]){
                tr[ *this].isWaitStable = false;
                tr[ *this].isWaitMove = true;
                tr[ *this].currentStep += 1;
                if(tr[ *this].currentStep > shot_this[ *step()]) {
                    endSweep(tr);
                } else {
                    rot1__->iterate_commit([=](Transaction &tr){
                        tr[ *rot1__->target()] = tr[ *this].startROT[0] + tr[ *this].deltaROT[0] * tr[ *this].currentStep;
                    });
                    rot2__->iterate_commit([=](Transaction &tr){
                        tr[ *rot2__->target()] = tr[ *this].startROT[1] + tr[ *this].deltaROT[1] * tr[ *this].currentStep;
                    });
                }
            }
        } else if(!shot_this[ *ready()]){
            tr[ *this].isWaitMove = false;
        }
    }
}

void
XTwoAxis::visualize(const Snapshot &shot_this) {
}

